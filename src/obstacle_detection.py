import cv2
import numpy as np
import pyrealsense2 as rs
import time
from audio_feedback import AudioFeedback
from hazard_stub import detect_hazards_rgb  # (still unused; placeholder for future YOLO)

NEAR_THRESHOLD_M = 0.8  # meters

def run_obstacle_and_motion_demo():
    """Run the RealSense obstacle + motion detection demo with audio + person detection (HOG)."""

    # ----- RealSense setup -----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)

    print("[INFO] Starting RealSense pipeline...")
    pipeline.start(config)

    prev_depth = None
    prev_time = time.time()

    # ----- Audio feedback -----
    audio = AudioFeedback(cooldown_sec=2.0, stable_sec=0.6)
    prev_hazard_state = False
    prev_person_state = False

    # ----- Person detection (OpenCV HOG + SVM) -----
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # This method is not the most effective but current test: tune detection sensitivity
    HOG_WINSTRIDE = (8, 8)
    HOG_PADDING = (8, 8)
    HOG_SCALE = 1.05
    PERSON_WEIGHT_THRESHOLD = 0.4  # raise to reduce false positives

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            h, w, _ = color_image.shape

            # Depth visualization (heatmap)
            depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # ----- Near-obstacle mask ("virtual cane") -----
            depth_m = depth_image * depth_frame.get_units()  # depth in meters
            near_mask = np.zeros_like(depth_image, dtype=np.uint8)

            valid = (depth_m > 0) & (depth_m < NEAR_THRESHOLD_M)
            near_mask[valid] = 255

            # Clean up noise
            kernel = np.ones((7, 7), np.uint8)
            near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_OPEN, kernel)
            near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                near_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            min_depth_seen = None
            closest_direction = "CENTER"

            # ----- Obstacle detection -----
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue

                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cx, cy = x + w_box // 2, y + h_box // 2

                d = depth_frame.get_distance(cx, cy)  # meters

                # Direction for this obstacle
                if cx < w / 3:
                    dir_for_this = "LEFT"
                elif cx > 2 * w / 3:
                    dir_for_this = "RIGHT"
                else:
                    dir_for_this = "CENTER"

                # Keep closest obstacle and its direction
                if d > 0 and ((min_depth_seen is None) or (d < min_depth_seen)):
                    min_depth_seen = d
                    closest_direction = dir_for_this

                # Draw bounding box
                cv2.rectangle(
                    color_image,
                    (x, y),
                    (x + w_box, y + h_box),
                    (0, 0, 255),
                    2,
                )

            # ----- Motion detection (depth differencing) -----
            motion_mask_vis = np.zeros_like(color_image)

            if prev_depth is not None:
                diff = cv2.absdiff(depth_image, prev_depth)
                _, motion_mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                motion_mask = motion_mask.astype(np.uint8)
                motion_mask = cv2.medianBlur(motion_mask, 5)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

                motion_mask_vis = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

            prev_depth = depth_image.copy()

            # ----- Person detection (HOG) -----
            # For speed, run on a smaller frame and scale results back up
            scale_down = 0.5
            small = cv2.resize(color_image, (int(w * scale_down), int(h * scale_down)))
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            rects, weights = hog.detectMultiScale(
                gray_small,
                winStride=HOG_WINSTRIDE,
                padding=HOG_PADDING,
                scale=HOG_SCALE,
            )

            # Track nearest detected person (distance via depth at bbox center)
            person_found = False
            nearest_person_dist = None
            nearest_person_dir = "CENTER"

            for (rx, ry, rw, rh), wt in zip(rects, weights):
                if wt < PERSON_WEIGHT_THRESHOLD:
                    continue

                # Scale bbox back to original coordinates
                x = int(rx / scale_down)
                y = int(ry / scale_down)
                w_box = int(rw / scale_down)
                h_box = int(rh / scale_down)

                cx = x + w_box // 2
                cy = y + h_box // 2

                # Clamp sample point to frame bounds
                cx = max(0, min(cx, w - 1))
                cy = max(0, min(cy, h - 1))

                d = depth_frame.get_distance(cx, cy)

                # Direction for this person
                if cx < w / 3:
                    dir_for_this = "LEFT"
                elif cx > 2 * w / 3:
                    dir_for_this = "RIGHT"
                else:
                    dir_for_this = "CENTER"

                # Keep nearest person
                if d > 0:
                    person_found = True
                    if (nearest_person_dist is None) or (d < nearest_person_dist):
                        nearest_person_dist = d
                        nearest_person_dir = dir_for_this

                # Draw person box (blue) + label
                cv2.rectangle(color_image, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                label = f"Person ({wt:.2f})"
                if d > 0:
                    label += f" {d:.2f}m {dir_for_this}"
                cv2.putText(
                    color_image,
                    label,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2,
                )

            # ----- Overlays: FPS, nearest obstacle, direction -----
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(
                color_image,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Person status overlay
            if person_found and (nearest_person_dist is not None):
                cv2.putText(
                    color_image,
                    f"Person: {nearest_person_dist:.2f} m ({nearest_person_dir})",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
                # Place obstacle info lower so it doesn't overlap the person line
                obstacle_y0 = 85
            else:
                obstacle_y0 = 55

            # Obstacle status overlay
            if min_depth_seen is not None:
                cv2.putText(
                    color_image,
                    f"Closest obstacle: {min_depth_seen:.2f} m",
                    (10, obstacle_y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    color_image,
                    "NEAR HAZARD",
                    (10, obstacle_y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    color_image,
                    f"Direction: {closest_direction}",
                    (10, obstacle_y0 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    color_image,
                    "No close obstacles",
                    (10, obstacle_y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # ----- Audio feedback (TTS) -----
            # Priority: announce near hazard obstacles; otherwise announce person detection.
            hazard_now = (min_depth_seen is not None)

            if hazard_now:
                audio.speak_hazard(
                    dist_m=min_depth_seen,
                    direction=closest_direction,
                    hazard_now=True,
                    force_clear=(prev_hazard_state and not hazard_now),
                )
            else:
                # If hazard cleared, say "Clear" once
                if prev_hazard_state and not hazard_now:
                    audio.speak_hazard(
                        dist_m=None,
                        direction="CENTER",
                        hazard_now=False,
                        force_clear=True,
                    )

                # Person callouts (rate limited via cooldown/stable logic)
                if person_found and (nearest_person_dist is not None):
                    # Debounce person announcements by using same speak_hazard method with custom wording
                    msg = f"Person detected. {nearest_person_dist:.2f} meters. {nearest_person_dir}."
                    audio.speak(msg, force=(not prev_person_state))
                else:
                    # Person just disappeared
                    if prev_person_state:
                        audio.speak("Person not detected.", force=True)

            prev_hazard_state = hazard_now
            prev_person_state = person_found

            # ----- Combine views -----
            near_mask_vis = cv2.cvtColor(near_mask, cv2.COLOR_GRAY2BGR)

            top_row = np.hstack((color_image, depth_vis))
            bottom_row = np.hstack((near_mask_vis, motion_mask_vis))
            combined = np.vstack((top_row, bottom_row))

            cv2.imshow(
                "Depth Perception – Color | Depth | Near-Obstacle | Motion", combined
            )

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    except Exception as e:
        print("[ERROR]", e)

    finally:
        print("[INFO] Stopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_obstacle_and_motion_demo()
