import cv2
import numpy as np
import pyrealsense2 as rs
import time
from hazard_stub import detect_hazards_rgb

NEAR_THRESHOLD_M = 0.8  # meters


def run_obstacle_and_motion_demo():
    """Run the RealSense obstacle + motion detection demo."""

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
            direction_text = "CENTER"

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue

                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cx, cy = x + w_box // 2, y + h_box // 2

                d = depth_frame.get_distance(cx, cy)  # meters
                if d > 0:
                    if (min_depth_seen is None) or (d < min_depth_seen):
                        min_depth_seen = d

                # Decide left/center/right in image coordinates
                if cx < w / 3:
                    direction_text = "LEFT"
                elif cx > 2 * w / 3:
                    direction_text = "RIGHT"
                else:
                    direction_text = "CENTER"

                # Draw bounding box
                cv2.rectangle(color_image, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

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

            if min_depth_seen is not None:
                cv2.putText(
                    color_image,
                    f"Closest obstacle: {min_depth_seen:.2f} m",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    color_image,
                    "NEAR HAZARD",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    color_image,
                    f"Direction: {direction_text}",
                    (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    color_image,
                    "No close obstacles",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

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
    # Allow running this module directly
    run_obstacle_and_motion_demo()
