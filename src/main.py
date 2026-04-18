"""
main.py  (thermal integrated, audio disabled)
---------------------------------------------
Based on the latest priority-audio main.py, but:

- all audio usage is disabled/commented out for now
- thermal_receiver.py is integrated
- thermal status is shown in the HUD
- "high heat" is added to hazard_labels when threshold is exceeded

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

from camera           import RealSenseCamera
from mapper           import OccupancyMapper, GRID_RES, GRID_W, GRID_H
from detector         import HazardDetector, draw_detections
from planner          import astar, simplify_path
from navigator        import path_to_instructions
# from audio_feedback import AudioFeedback
from thermal_receiver import ThermalReceiver

# ── Model paths ───────────────────────────────────────────────────────────────
GENERAL_MODEL = "yolov8n.pt"
DOOR_MODEL    = "doors.pt"
FIRE_MODEL    = "best.pt"

# ── Tuning ────────────────────────────────────────────────────────────────────
ROUTE_REPLAN_SEC         = 8.0
NEAR_THRESHOLD_M         = 0.8
FLOOR_OBS_THRESHOLD      = 0.5
FLOOR_OBS_MIN_PX         = 300
EXIT_LOCK_FRAMES         = 20
FPS_WINDOW               = 30

THERMAL_ALERT_THRESHOLD_F = 95.0

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
HUD_BG     = (30,  30,  30)
COL_WHITE  = (255, 255, 255)
COL_GREEN  = (60,  220, 60)
COL_ORANGE = (0,   165, 255)
COL_RED    = (60,  60,  240)
COL_YELLOW = (0,   220, 220)
COL_GRAY   = (160, 160, 160)
COL_CYAN   = (255, 220, 0)


def render_grid(grid, path=None, user_cell=None, scale=GRID_VIS_SCALE):
    img_small = GRID_LUT[grid]
    img = cv2.resize(
        img_small,
        (GRID_W * scale, GRID_H * scale),
        interpolation=cv2.INTER_NEAREST
    )
    if path:
        for r, c in path:
            cv2.rectangle(
                img,
                (c * scale, r * scale),
                ((c + 1) * scale, (r + 1) * scale),
                (255, 180, 0),
                -1
            )
    if user_cell:
        r, c = user_cell
        cv2.circle(
            img,
            (c * scale + scale // 2, r * scale + scale // 2),
            scale + 2,
            (255, 255, 255),
            -1
        )
    return img


def draw_hud(frame, fps, exit_locked, exit_confirm_cnt, det_result,
             hazard_now, thermal_state, min_dist=None):
    h, w = frame.shape[:2]
    panel_w, panel_h, margin = 330, 205, 10

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (margin + panel_w, margin + panel_h),
        HUD_BG,
        -1
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(
        frame,
        (margin, margin),
        (margin + panel_w, margin + panel_h),
        (80, 80, 80),
        1
    )

    x0 = margin + 10
    y  = margin + 22
    dy = 24

    # FPS
    fps_col = COL_GREEN if fps >= 15 else COL_ORANGE if fps >= 8 else COL_RED
    cv2.putText(
        frame, f"FPS  {fps:.0f}", (x0, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_col, 1, cv2.LINE_AA
    )
    y += dy

    # Exit
    if exit_locked:
        cv2.putText(
            frame, "EXIT  LOCKED", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GREEN, 1, cv2.LINE_AA
        )
    else:
        pct = int((exit_confirm_cnt / EXIT_LOCK_FRAMES) * 100)
        cv2.putText(
            frame, f"EXIT  searching {pct}%  [E]=set", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_ORANGE, 1, cv2.LINE_AA
        )
    y += dy

    # Obstacle
    if hazard_now and min_dist is not None:
        cv2.putText(
            frame, f"OBSTACLE  {min_dist:.2f}m", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_RED, 2, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame, "OBSTACLE  clear", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GREEN, 1, cv2.LINE_AA
        )
    y += dy

    # Thermal
    if not thermal_state.connected:
        cv2.putText(
            frame, "THERMAL  disconnected", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GRAY, 1, cv2.LINE_AA
        )
    elif thermal_state.max_temp_f is None:
        cv2.putText(
            frame, "THERMAL  connected [raw only]", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_ORANGE, 1, cv2.LINE_AA
        )
    elif thermal_state.above_user_threshold:
        cv2.putText(
            frame, f"THERMAL  ALERT  {thermal_state.max_temp_f:.1f}F", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_RED, 2, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame, f"THERMAL  {thermal_state.max_temp_f:.1f}F", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GREEN, 1, cv2.LINE_AA
        )
    y += dy

    # Fire / smoke
    fire_dets  = [d for d in det_result.hazards if d.label == "fire"]
    smoke_dets = [d for d in det_result.hazards if d.label == "smoke"]
    if fire_dets:
        best = max(fire_dets, key=lambda d: d.confidence)
        cv2.putText(
            frame,
            f"FIRE  {best.confidence:.0%}  {best.direction}  {best.depth_m:.1f}m",
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_RED, 2, cv2.LINE_AA
        )
    elif smoke_dets:
        best = max(smoke_dets, key=lambda d: d.confidence)
        cv2.putText(
            frame,
            f"SMOKE  {best.confidence:.0%}  {best.direction}",
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_ORANGE, 1, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame, "FIRE    none detected", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GRAY, 1, cv2.LINE_AA
        )
    y += dy

    # Door
    if det_result.doors:
        best = max(det_result.doors, key=lambda d: d.confidence)
        meth = "model" if best.confidence > 0.50 else "depth"
        cv2.putText(
            frame,
            f"DOOR  {best.confidence:.0%}  {best.direction}  [{meth}]",
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GREEN, 1, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame, "DOOR    not visible", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GRAY, 1, cv2.LINE_AA
        )
    y += dy

    # Person
    if det_result.persons:
        nearest = min(det_result.persons, key=lambda p: p.depth_m)
        cv2.putText(
            frame,
            f"PERSON  {nearest.depth_m:.1f}m  {nearest.direction}",
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_CYAN, 1, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame, "PERSON  not detected", (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GRAY, 1, cv2.LINE_AA
        )


def draw_instruction_bar(frame, instructions):
    if not instructions:
        return
    h, w = frame.shape[:2]
    lines = instructions[:3]
    bar_h = 28 * len(lines) + 16
    bar_y = h - bar_h - 5

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, f"{i+1}. {line}",
            (10, bar_y + 22 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_YELLOW, 1, cv2.LINE_AA
        )


def draw_panel_labels(combined, cam_w, depth_w, h):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Camera",   (8, 18), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(combined, "Depth",    (cam_w + 8, 18), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(combined, "Grid map", (cam_w * 2 + 8, 18), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def run():
    camera   = RealSenseCamera()
    mapper   = OccupancyMapper()
    detector = HazardDetector(
        general_model_path=GENERAL_MODEL,
        door_model_path=DOOR_MODEL,
        fire_model_path=FIRE_MODEL,
    )

    # audio = AudioFeedback()
    audio = None

    thermal = ThermalReceiver(
        host="0.0.0.0",
        port=5001,
        alert_threshold_f=THERMAL_ALERT_THRESHOLD_F,
        keep_latest_frame=True,
        verbose=False,
    )
    thermal.start()

    camera.start()

    last_replan_time = 0.0
    last_instructions = []
    current_path = []
    instructions = []
    exit_locked = False
    exit_confirm_cnt = 0
    prev_hazard = False
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

            thermal_state = thermal.get_latest_state()
            thermal_alert_now = (
                thermal_state.connected and thermal_state.above_user_threshold
            )

            # ── 1. Update map ────────────────────────────────────────────────
            mapper.reset_hazards()
            mapper.update(depth_image, depth_units, camera)

            # ── 2. Detection ─────────────────────────────────────────────────
            det_result = detector.detect(color_image, depth_frame, camera)

            # ── 3. Register hazards ──────────────────────────────────────────
            hazard_labels = []
            fire_detected = False
            smoke_detected = False

            for hz in det_result.hazards:
                wx, _, wz = hz.world_xyz
                mapper.mark_hazard(wx, wz)
                hazard_labels.append(hz.label)

                if hz.label == "fire" and not fire_detected:
                    fire_detected = True
                    # if audio is not None:
                    #     audio.alert_fire(direction=hz.direction)

                elif hz.label == "smoke" and not smoke_detected:
                    smoke_detected = True
                    # if audio is not None:
                    #     audio.alert_smoke(direction=hz.direction)

            if thermal_alert_now:
                hazard_labels.append("high heat")

            # ── 4. Floor obstacle detection ──────────────────────────────────
            floor_mask = (depth_m > 0.1) & (depth_m < FLOOR_OBS_THRESHOLD)
            floor_mask[:2 * h // 3, :] = False
            if floor_mask.sum() > FLOOR_OBS_MIN_PX:
                floor_depths = np.where(floor_mask, depth_m, 999.0)
                min_idx      = np.unravel_index(floor_depths.argmin(), floor_depths.shape)
                floor_dist   = depth_m[min_idx]
                fx, fy       = min_idx[1], min_idx[0]
                cv2.circle(color_image, (fx, fy), 8, COL_ORANGE, -1)
                cv2.putText(
                    color_image, f"{floor_dist:.2f}m",
                    (fx + 10, fy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_ORANGE, 1, cv2.LINE_AA
                )
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
                    # if audio is not None:
                    #     audio.speak_status("Exit located. Escape route ready.")
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
                    if instructions != last_instructions:
                        # if audio is not None:
                        #     audio.speak_route(instructions)
                        last_instructions = instructions
                else:
                    current_path = []
                    instructions = ["No clear path found — stay low."]
                    # if audio is not None:
                    #     audio.speak_status("No clear path to exit. Stay low.")
                last_replan_time = now

            # ── 7b. Stop repeating if user reached exit ──────────────────────
            if exit_locked and mapper.exit_cell:
                if user_cell == mapper.exit_cell:
                    # if audio is not None:
                    #     audio.cancel_route()
                    #     audio.speak_status("You have reached the exit. Get out now.")
                    exit_locked = False

            # ── 8. Obstacle warning ──────────────────────────────────────────
            valid_near = (depth_m > 0.1) & (depth_m < NEAR_THRESHOLD_M)
            hazard_now = bool(valid_near.any())

            if hazard_now:
                masked           = np.where(valid_near, depth_m, 999.0)
                min_idx          = np.unravel_index(masked.argmin(), masked.shape)
                min_dist_display = float(depth_m[min_idx])
                cx_obs           = int(min_idx[1])
                direction        = "LEFT" if cx_obs < w/3 else "RIGHT" if cx_obs > 2*w/3 else "CENTER"
                # if audio is not None:
                #     audio.speak_hazard(min_dist_display, direction, hazard_now=True)
            else:
                min_dist_display = None
                if prev_hazard:
                    # if audio is not None:
                    #     audio.speak_hazard(None, "CENTER", hazard_now=False, force_clear=True)
                    pass
            prev_hazard = hazard_now

            # ── 9. Thermal warning placeholder ───────────────────────────────
            if thermal_alert_now and thermal_state.max_temp_f is not None:
                pass
                # if audio is not None:
                #     audio.speak(
                #         f"High heat detected. Maximum temperature "
                #         f"{thermal_state.max_temp_f:.1f} degrees Fahrenheit."
                #     )

            # ── 10. Draw detections + HUD ────────────────────────────────────
            draw_detections(color_image, det_result)
            draw_hud(
                color_image,
                fps,
                exit_locked,
                exit_confirm_cnt,
                det_result,
                hazard_now,
                thermal_state,
                min_dist_display
            )
            draw_instruction_bar(color_image, instructions)

            # ── 11. Compose display ──────────────────────────────────────────
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            grid_img     = render_grid(mapper.grid, path=current_path, user_cell=user_cell)
            grid_resized = cv2.resize(grid_img, (h, h), interpolation=cv2.INTER_NEAREST)

            combined = np.hstack([color_image, depth_vis, grid_resized])
            draw_panel_labels(combined, w, w, h)

            cv2.imshow("Kitchen Hazard System  |  Q=quit  E=set exit", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("e"):
                cam_x, cam_z = mapper.camera_world_pos()
                mapper.mark_exit(cam_x, cam_z + 1.5)
                exit_locked      = True
                exit_confirm_cnt = EXIT_LOCK_FRAMES
                # if audio is not None:
                #     audio.speak_status("Exit registered. Escape route ready.")
                print("[Main] Exit manually registered.")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        print("[Main] Shutting down...")
        # if audio is not None:
        #     audio.cancel_route()
        thermal.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()