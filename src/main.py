"""
main.py
-------
Kitchen Hazard System — main loop.

Controls
--------
  Q / ESC  : quit
  E        : manually register exit (point camera at door and press E)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time

from camera         import RealSenseCamera
from mapper         import OccupancyMapper, GRID_RES, GRID_W, GRID_H, HAZARD, EXIT, FREE, OCCUPIED
from detector       import HazardDetector, draw_detections
from planner        import astar, simplify_path
from navigator      import path_to_instructions
from audio_feedback import AudioFeedback

# ── Tuning ────────────────────────────────────────────────────────────────────
YOLO_MODEL           = "yolov8n.pt"
ROUTE_REPLAN_SEC     = 3.0       # replan escape route every N seconds
NEAR_THRESHOLD_M     = 0.8       # immediate obstacle warning distance (metres)
FLOOR_OBS_THRESHOLD  = 0.5       # depth threshold for small floor objects (metres)
FLOOR_OBS_MIN_PX     = 300       # min pixel area to count as floor obstacle
EXIT_LOCK_FRAMES     = 20        # reduced from 30 — easier to lock exit via depth gap

# ── Grid visualisation colours (BGR) ─────────────────────────────────────────
CELL_COLOURS = {
    0: (50,  50,  50 ),   # unknown
    1: (20,  200, 20 ),   # free
    2: (60,  60,  180),   # occupied
    3: (0,   0,   255),   # hazard
    4: (0,   255, 255),   # exit
}
GRID_VIS_SCALE = 2


def render_grid(grid, path=None, user_cell=None, scale=GRID_VIS_SCALE):
    vis_h = GRID_H * scale
    vis_w = GRID_W * scale
    img   = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)

    for val, colour in CELL_COLOURS.items():
        rows, cols = np.where(grid == val)
        for r, c in zip(rows, cols):
            img[r * scale:(r + 1) * scale,
                c * scale:(c + 1) * scale] = colour

    if path:
        for r, c in path:
            cv2.rectangle(
                img,
                (c * scale, r * scale),
                ((c + 1) * scale, (r + 1) * scale),
                (255, 180, 0), -1
            )

    if user_cell:
        r, c = user_cell
        cv2.circle(img,
                   (c * scale + scale // 2, r * scale + scale // 2),
                   scale + 1, (255, 255, 255), -1)
    return img


def run():
    camera   = RealSenseCamera()
    mapper   = OccupancyMapper()
    detector = HazardDetector(YOLO_MODEL)
    audio    = AudioFeedback(cooldown_sec=2.5, stable_sec=0.5)

    camera.start()

    last_replan_time = 0.0
    current_path     = []
    instructions     = []
    exit_locked      = False
    exit_confirm_cnt = 0
    prev_hazard      = False
    prev_time        = time.time()

    print("[Main] System running.")
    print("[Main] Controls: Q/ESC = quit | E = register exit manually")

    try:
        while True:
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now

            depth_units = camera.get_depth_units(depth_frame)
            h, w = color_image.shape[:2]
            depth_m = depth_image.astype(np.float32) * depth_units

            # ── 1. Update occupancy map ──────────────────────────────────────
            mapper.reset_hazards()
            mapper.update(depth_image, depth_units, camera)

            # ── 2. YOLO + fire + door detection ─────────────────────────────
            det_result = detector.detect(color_image, depth_frame, camera)

            # ── 3. Register hazards in map ───────────────────────────────────
            hazard_labels = []
            for hz in det_result.hazards:
                wx, _, wz = hz.world_xyz
                mapper.mark_hazard(wx, wz)
                hazard_labels.append(hz.label)

            # ── 4. Small floor obstacle detection (depth mask) ───────────────
            # Catches small objects YOLO misses (phone case, fan, etc.)
            floor_mask = (depth_m > 0.1) & (depth_m < FLOOR_OBS_THRESHOLD)
            # Only look at bottom third of frame (floor level)
            floor_roi = floor_mask.copy()
            floor_roi[:2 * h // 3, :] = False

            if floor_roi.sum() > FLOOR_OBS_MIN_PX:
                # Find closest floor obstacle
                floor_depths = np.where(floor_roi, depth_m, 999.0)
                min_idx = np.unravel_index(floor_depths.argmin(), floor_depths.shape)
                floor_dist = depth_m[min_idx]
                fx, fy = min_idx[1], min_idx[0]
                direction = "LEFT" if fx < w/3 else "RIGHT" if fx > 2*w/3 else "CENTER"

                # Draw on frame
                cv2.circle(color_image, (fx, fy), 10, (0, 165, 255), -1)
                cv2.putText(color_image, f"Floor obj {floor_dist:.2f}m",
                            (fx - 30, fy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # Add to hazard labels for audio
                hazard_labels.append("floor obstacle")

            # ── 5. Register exit (auto depth gap or manual) ──────────────────
            if not exit_locked and det_result.doors:
                best_door = min(det_result.doors, key=lambda d: d.depth_m)
                wx, _, wz = best_door.world_xyz
                mapper.mark_exit(wx, wz)
                exit_confirm_cnt += 1
                if exit_confirm_cnt >= EXIT_LOCK_FRAMES:
                    exit_locked = True
                    print("[Main] Exit position locked automatically.")
                    audio.speak("Exit located. Escape route ready.", force=True)
            elif not exit_locked:
                exit_confirm_cnt = max(0, exit_confirm_cnt - 1)

            # ── 6. Locate user ───────────────────────────────────────────────
            cam_x, cam_z = mapper.camera_world_pos()
            if det_result.persons:
                nearest = min(det_result.persons, key=lambda p: p.depth_m)
                ux, _, uz = nearest.world_xyz
                user_cell = mapper.world_to_cell(ux, uz)
            else:
                user_cell = mapper.world_to_cell(cam_x, cam_z)

            # ── 7. Path planning ─────────────────────────────────────────────
            if (now - last_replan_time) > ROUTE_REPLAN_SEC and mapper.exit_cell:
                raw_path = astar(mapper.grid, user_cell, mapper.exit_cell)
                if raw_path:
                    current_path = simplify_path(raw_path, tolerance=3)
                    instructions = path_to_instructions(
                        current_path,
                        hazard_labels=hazard_labels,
                        exit_found=exit_locked,
                    )
                    audio.speak_route(instructions)
                else:
                    current_path = []
                    audio.speak("No clear path to exit. Stay low.", force=True)
                last_replan_time = now

            # ── 8. Immediate obstacle audio warning ──────────────────────────
            valid_near = (depth_m > 0.1) & (depth_m < NEAR_THRESHOLD_M)
            hazard_now = valid_near.any()

            if hazard_now:
                masked   = np.where(valid_near, depth_m, 999.0)
                min_idx  = np.unravel_index(masked.argmin(), masked.shape)
                min_dist = depth_m[min_idx]
                cx_obs   = min_idx[1]
                direction = "LEFT" if cx_obs < w/3 else "RIGHT" if cx_obs > 2*w/3 else "CENTER"
                audio.speak_hazard(min_dist, direction, hazard_now=True)
            else:
                if prev_hazard:
                    audio.speak_hazard(None, "CENTER", hazard_now=False, force_clear=True)

            prev_hazard = hazard_now

            # ── 9. Overlays ──────────────────────────────────────────────────
            draw_detections(color_image, det_result)

            cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            exit_txt = (
                "EXIT LOCKED" if exit_locked
                else f"Searching exit ({exit_confirm_cnt}/{EXIT_LOCK_FRAMES}) | Press E to set manually"
            )
            cv2.putText(color_image, exit_txt, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if instructions:
                for i, line in enumerate(instructions[:4]):
                    cv2.putText(color_image, line, (10, 90 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

            # ── 10. Display ──────────────────────────────────────────────────
            depth_vis    = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_vis    = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            grid_img     = render_grid(mapper.grid, path=current_path, user_cell=user_cell)
            grid_resized = cv2.resize(grid_img, (h, h))

            combined = np.hstack([color_image, depth_vis, grid_resized])
            cv2.imshow("Kitchen Hazard System  |  Q=quit  E=set exit", combined)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("e"):
                # Manually register exit 1.5m ahead of camera
                cam_x, cam_z = mapper.camera_world_pos()
                mapper.mark_exit(cam_x, cam_z + 1.5)
                exit_locked      = True
                exit_confirm_cnt = EXIT_LOCK_FRAMES
                audio.speak("Exit registered manually. Escape route ready.", force=True)
                print("[Main] Exit manually registered.")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        print("[Main] Shutting down...")
        audio.cancel_route()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
