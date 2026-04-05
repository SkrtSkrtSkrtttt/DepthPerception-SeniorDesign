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
from mapper         import OccupancyMapper, GRID_RES, GRID_W, GRID_H
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
FPS_WINDOW          = 30

# ── Grid colour LUT (BGR) ─────────────────────────────────────────────────────
GRID_LUT = np.array([
    [50,  50,  50 ],
    [20,  200, 20 ],
    [60,  60,  180],
    [0,   0,   255],
    [0,   255, 255],
], dtype=np.uint8)

GRID_VIS_SCALE = 2

# ── HUD colours (BGR) ─────────────────────────────────────────────────────────
HUD_BG        = (30,  30,  30 )
COL_WHITE     = (255, 255, 255)
COL_GREEN     = (60,  220, 60 )
COL_ORANGE    = (0,   165, 255)
COL_RED       = (60,  60,  240)
COL_YELLOW    = (0,   220, 220)
COL_GRAY      = (160, 160, 160)
COL_CYAN      = (255, 220, 0  )


def render_grid(grid, path=None, user_cell=None, scale=GRID_VIS_SCALE):
    img_small = GRID_LUT[grid]
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
                   scale + 2, (255, 255, 255), -1)
    return img


def draw_hud(frame, fps, exit_locked, exit_confirm_cnt, det_result, hazard_now, min_dist=None):
    """
    Draw a semi-transparent HUD panel in the top-left corner.
    All status info lives here — nothing floats over detections.
    """
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 280
    panel_h = 170
    margin  = 10

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (margin, margin),
                  (margin + panel_w, margin + panel_h),
                  HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Panel border
    cv2.rectangle(frame,
                  (margin, margin),
                  (margin + panel_w, margin + panel_h),
                  (80, 80, 80), 1)

    x0 = margin + 10
    y  = margin + 22
    dy = 24

    # ── FPS ──────────────────────────────────────────────────────────────────
    fps_col = COL_GREEN if fps >= 15 else COL_ORANGE if fps >= 8 else COL_RED
    cv2.putText(frame, f"FPS  {fps:.0f}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_col, 1, cv2.LINE_AA)
    y += dy

    # ── Exit status ───────────────────────────────────────────────────────────
    if exit_locked:
        cv2.putText(frame, "EXIT  LOCKED", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GREEN, 1, cv2.LINE_AA)
    else:
        pct = int((exit_confirm_cnt / EXIT_LOCK_FRAMES) * 100)
        cv2.putText(frame, f"EXIT  searching {pct}%  [E]=set",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_ORANGE, 1, cv2.LINE_AA)
    y += dy

    # ── Obstacle warning ──────────────────────────────────────────────────────
    if hazard_now and min_dist is not None:
        cv2.putText(frame, f"OBSTACLE  {min_dist:.2f}m",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_RED, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "OBSTACLE  clear",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GREEN, 1, cv2.LINE_AA)
    y += dy

    # ── Fire / smoke ──────────────────────────────────────────────────────────
    fire_dets  = [d for d in det_result.hazards if d.label == "fire"]
    smoke_dets = [d for d in det_result.hazards if d.label == "smoke"]

    if fire_dets:
        best_fire = max(fire_dets, key=lambda d: d.confidence)
        cv2.putText(frame,
                    f"FIRE  {best_fire.confidence:.0%}  {best_fire.direction}  {best_fire.depth_m:.1f}m",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_RED, 2, cv2.LINE_AA)
    elif smoke_dets:
        best_smoke = max(smoke_dets, key=lambda d: d.confidence)
        cv2.putText(frame,
                    f"SMOKE  {best_smoke.confidence:.0%}  {best_smoke.direction}",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_ORANGE, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "FIRE    none detected",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GRAY, 1, cv2.LINE_AA)
    y += dy

    # ── Door ─────────────────────────────────────────────────────────────────
    if det_result.doors:
        best_door = max(det_result.doors, key=lambda d: d.confidence)
        method = "model" if best_door.confidence > 0.50 else "depth"
        cv2.putText(frame,
                    f"DOOR  {best_door.confidence:.0%}  {best_door.direction}  [{method}]",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "DOOR    not visible",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GRAY, 1, cv2.LINE_AA)
    y += dy

    # ── Person ────────────────────────────────────────────────────────────────
    if det_result.persons:
        nearest = min(det_result.persons, key=lambda p: p.depth_m)
        cv2.putText(frame,
                    f"PERSON  {nearest.depth_m:.1f}m  {nearest.direction}",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_CYAN, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "PERSON  not detected",
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GRAY, 1, cv2.LINE_AA)


def draw_instruction_bar(frame, instructions):
    """
    Draw escape route instructions as a bar at the bottom of the frame.
    Semi-transparent dark background, yellow text.
    """
    if not instructions:
        return

    h, w = frame.shape[:2]
    bar_h   = 28 * min(len(instructions[:3]), 3) + 16
    bar_y   = h - bar_h - 5

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    for i, line in enumerate(instructions[:3]):
        cv2.putText(frame, f"{i+1}. {line}",
                    (10, bar_y + 22 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_YELLOW, 1, cv2.LINE_AA)


def draw_panel_labels(combined, cam_w, depth_w, grid_w, h):
    """Draw small labels at the top of each panel."""
    label_y = 18
    font    = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(combined, "Camera", (8, label_y),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(combined, "Depth", (cam_w + 8, label_y),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(combined, "Grid map", (cam_w + depth_w + 8, label_y),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


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
    min_dist_display = None

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
            depth_m = depth_image.astype(np.float32) * depth_units

            # ── 1. Update map ────────────────────────────────────────────────
            mapper.reset_hazards()
            mapper.update(depth_image, depth_units, camera)

            # ── 2. Detection ─────────────────────────────────────────────────
            det_result = detector.detect(color_image, depth_frame, camera)

            # ── 3. Register hazards ──────────────────────────────────────────
            hazard_labels = []
            for hz in det_result.hazards:
                wx, _, wz = hz.world_xyz
                mapper.mark_hazard(wx, wz)
                hazard_labels.append(hz.label)

            # ── 4. Floor obstacle detection ──────────────────────────────────
            floor_mask = (depth_m > 0.1) & (depth_m < FLOOR_OBS_THRESHOLD)
            floor_mask[:2 * h // 3, :] = False
            if floor_mask.sum() > FLOOR_OBS_MIN_PX:
                floor_depths = np.where(floor_mask, depth_m, 999.0)
                min_idx      = np.unravel_index(floor_depths.argmin(), floor_depths.shape)
                floor_dist   = depth_m[min_idx]
                fx, fy       = min_idx[1], min_idx[0]
                cv2.circle(color_image, (fx, fy), 8, COL_ORANGE, -1)
                cv2.putText(color_image, f"{floor_dist:.2f}m",
                            (fx + 10, fy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_ORANGE, 1, cv2.LINE_AA)
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
                    instructions = ["No clear path found — stay low."]
                    audio.speak("No clear path to exit. Stay low.", force=True)
                last_replan_time = now

            # ── 8. Obstacle warning ──────────────────────────────────────────
            valid_near = (depth_m > 0.1) & (depth_m < NEAR_THRESHOLD_M)
            hazard_now = bool(valid_near.any())

            if hazard_now:
                masked           = np.where(valid_near, depth_m, 999.0)
                min_idx          = np.unravel_index(masked.argmin(), masked.shape)
                min_dist_display = float(depth_m[min_idx])
                cx_obs           = int(min_idx[1])
                direction        = "LEFT" if cx_obs < w/3 else "RIGHT" if cx_obs > 2*w/3 else "CENTER"
                audio.speak_hazard(min_dist_display, direction, hazard_now=True)
            else:
                min_dist_display = None
                if prev_hazard:
                    audio.speak_hazard(None, "CENTER", hazard_now=False, force_clear=True)
            prev_hazard = hazard_now

            # ── 9. Draw detections ───────────────────────────────────────────
            draw_detections(color_image, det_result)

            # ── 10. Draw HUD + instruction bar ───────────────────────────────
            draw_hud(color_image, fps, exit_locked, exit_confirm_cnt,
                     det_result, hazard_now, min_dist_display)
            draw_instruction_bar(color_image, instructions)

            # ── 11. Compose display ──────────────────────────────────────────
            depth_vis    = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            grid_img     = render_grid(mapper.grid, path=current_path, user_cell=user_cell)
            grid_resized = cv2.resize(grid_img, (h, h), interpolation=cv2.INTER_NEAREST)

            combined = np.hstack([color_image, depth_vis, grid_resized])

            # Panel labels at very top
            draw_panel_labels(combined, w, w, h, h)

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