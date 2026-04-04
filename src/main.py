"""
main.py  (optimised)
--------------------
Optimisations applied:
  1. Vectorised grid rendering — numpy LUT instead of Python loops
  2. Depth mask computed once and reused
  3. FPS averaged over 30 frames for stable readout
  4. Display resize done once per frame

Controls
--------
  Q / ESC  : quit
  E        : manually register exit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time
from collections import deque

from camera         import RealSenseCamera
from mapper         import OccupancyMapper, GRID_RES, GRID_W, GRID_H, HAZARD, EXIT, FREE, OCCUPIED
from detector       import HazardDetector, draw_detections
from planner        import astar, simplify_path
from navigator      import path_to_instructions
from audio_feedback import AudioFeedback

# ── Model paths ───────────────────────────────────────────────────────────────
GENERAL_MODEL = "yolov8n.pt"
DOOR_MODEL    = "doors.pt"
FIRE_MODEL    = "best.pt"

# ── Tuning ────────────────────────────────────────────────────────────────────
ROUTE_REPLAN_SEC    = 3.0
NEAR_THRESHOLD_M    = 0.8
FLOOR_OBS_THRESHOLD = 0.5
FLOOR_OBS_MIN_PX    = 300
EXIT_LOCK_FRAMES    = 20
FPS_WINDOW          = 30    # average FPS over this many frames

# ── Grid colour lookup table (BGR) ────────────────────────────────────────────
# Shape (5, 3) — index by cell value to get BGR colour
GRID_LUT = np.array([
    [50,  50,  50 ],   # 0 unknown
    [20,  200, 20 ],   # 1 free
    [60,  60,  180],   # 2 occupied
    [0,   0,   255],   # 3 hazard
    [0,   255, 255],   # 4 exit
], dtype=np.uint8)

GRID_VIS_SCALE = 2


def render_grid(grid, path=None, user_cell=None, scale=GRID_VIS_SCALE):
    """Vectorised grid render — numpy LUT, no Python loops."""
    img_small = GRID_LUT[grid]   # (H, W, 3) directly
    img = cv2.resize(
        img_small,
        (GRID_W * scale, GRID_H * scale),
        interpolation=cv2.INTER_NEAREST
    )

    if path:
        for r, c in path:
            cv2.rectangle(img,
                          (c * scale, r * scale),
                          ((c + 1) * scale, (r + 1) * scale),
                          (255, 180, 0), -1)
    if user_cell:
        r, c = user_cell
        cv2.circle(img,
                   (c * scale + scale // 2, r * scale + scale // 2),
                   scale + 1, (255, 255, 255), -1)
    return img


def run():
    camera   = RealSenseCamera()
    mapper   = OccupancyMapper()
    detector = HazardDetector(
        general_model_path=GENERAL_MODEL,
        door_model_path=DOOR_MODEL,
        fire_model_path=FIRE_MODEL,
    )
    audio = AudioFeedback(cooldown_sec=2.5, stable_sec=0.5)

    camera.start()

    last_replan_time = 0.0
    current_path     = []
    instructions     = []
    exit_locked      = False
    exit_confirm_cnt = 0
    prev_hazard      = False

    # Stable FPS counter
    frame_times = deque(maxlen=FPS_WINDOW)
    prev_time   = time.time()

    print("[Main] System running.")
    print("[Main] Q/ESC = quit | E = register exit manually")

    try:
        while True:
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            now = time.time()
            frame_times.append(now - prev_time)
            prev_time = now
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            depth_units = camera.get_depth_units(depth_frame)
            h, w = color_image.shape[:2]

            # Compute depth_m once — reused for obstacle + floor detection
            depth_m = depth_image.astype(np.float32) * depth_units

            # ── 1. Update occupancy map ──────────────────────────────────────
            mapper.reset_hazards()
            mapper.update(depth_image, depth_units, camera)

            # ── 2. Detection ─────────────────────────────────────────────────
            det_result = detector.detect(color_image, depth_frame, camera)

            # ── 3. Register hazards in map ───────────────────────────────────
            hazard_labels = []
            for hz in det_result.hazards:
                wx, _, wz = hz.world_xyz
                mapper.mark_hazard(wx, wz)
                hazard_labels.append(hz.label)

            # ── 4. Floor obstacle detection (vectorised) ─────────────────────
            floor_mask = (depth_m > 0.1) & (depth_m < FLOOR_OBS_THRESHOLD)
            floor_mask[:2 * h // 3, :] = False   # bottom third only

            if floor_mask.sum() > FLOOR_OBS_MIN_PX:
                floor_depths = np.where(floor_mask, depth_m, 999.0)
                min_idx      = np.unravel_index(floor_depths.argmin(), floor_depths.shape)
                floor_dist   = depth_m[min_idx]
                fx, fy       = min_idx[1], min_idx[0]
                cv2.circle(color_image, (fx, fy), 10, (0, 165, 255), -1)
                cv2.putText(color_image, f"Floor {floor_dist:.2f}m",
                            (fx - 25, fy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                hazard_labels.append("floor obstacle")

            # ── 5. Register exit ─────────────────────────────────────────────
            if not exit_locked and det_result.doors:
                best_door = max(det_result.doors, key=lambda d: d.confidence)
                wx, _, wz = best_door.world_xyz
                mapper.mark_exit(wx, wz)
                exit_confirm_cnt += 1
                if exit_confirm_cnt >= EXIT_LOCK_FRAMES:
                    exit_locked = True
                    print("[Main] Exit locked.")
                    audio.speak("Exit located. Escape route ready.", force=True)
            elif not exit_locked:
                exit_confirm_cnt = max(0, exit_confirm_cnt - 1)

            # ── 6. Locate user ───────────────────────────────────────────────
            cam_x, cam_z = mapper.camera_world_pos()
            if det_result.persons:
                nearest   = min(det_result.persons, key=lambda p: p.depth_m)
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

            # ── 8. Immediate obstacle warning ────────────────────────────────
            valid_near = (depth_m > 0.1) & (depth_m < NEAR_THRESHOLD_M)
            hazard_now = bool(valid_near.any())

            if hazard_now:
                masked    = np.where(valid_near, depth_m, 999.0)
                min_idx   = np.unravel_index(masked.argmin(), masked.shape)
                min_dist  = float(depth_m[min_idx])
                cx_obs    = int(min_idx[1])
                direction = "LEFT" if cx_obs < w/3 else "RIGHT" if cx_obs > 2*w/3 else "CENTER"
                audio.speak_hazard(min_dist, direction, hazard_now=True)
            else:
                if prev_hazard:
                    audio.speak_hazard(None, "CENTER", hazard_now=False, force_clear=True)
            prev_hazard = hazard_now

            # ── 9. Draw detections + overlays ────────────────────────────────
            draw_detections(color_image, det_result)

            cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            exit_txt = (
                "EXIT LOCKED" if exit_locked
                else f"Searching ({exit_confirm_cnt}/{EXIT_LOCK_FRAMES}) | E=set manually"
            )
            cv2.putText(color_image, exit_txt, (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if det_result.doors:
                best = max(det_result.doors, key=lambda d: d.confidence)
                method = "model" if best.confidence > 0.50 else "depth"
                cv2.putText(color_image,
                            f"Door: {method} {best.confidence:.0%} {best.direction}",
                            (10, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            if instructions:
                for i, line in enumerate(instructions[:4]):
                    cv2.putText(color_image, line, (10, 105 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

            # ── 10. Compose display ──────────────────────────────────────────
            depth_vis    = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            grid_img     = render_grid(mapper.grid, path=current_path, user_cell=user_cell)
            grid_resized = cv2.resize(grid_img, (h, h), interpolation=cv2.INTER_NEAREST)

            combined = np.hstack([color_image, depth_vis, grid_resized])
            cv2.imshow("Kitchen Hazard System  |  Q=quit  E=set exit", combined)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("e"):
                cam_x, cam_z = mapper.camera_world_pos()
                mapper.mark_exit(cam_x, cam_z + 1.5)
                exit_locked      = True
                exit_confirm_cnt = EXIT_LOCK_FRAMES
                audio.speak("Exit registered. Escape route ready.", force=True)
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