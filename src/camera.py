"""
camera.py
---------
Handles RealSense D435 pipeline setup, frame capture, and depth/color alignment.
"""

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.intrinsics = None
        self._running = False

    def start(self):
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        # Cache intrinsics for 3D projection
        depth_profile = profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        self._running = True
        print("[Camera] Pipeline started.")

    def get_frames(self):
        """
        Returns (color_image, depth_image, depth_frame) or (None, None, None) on failure.
        color_image  : np.ndarray (H, W, 3) BGR
        depth_image  : np.ndarray (H, W) uint16, raw depth units
        depth_frame  : rs.depth_frame  (for get_distance() calls)
        """
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image, depth_frame

    def deproject_pixel(self, px, py, depth_m):
        """
        Convert a 2-D pixel + depth (metres) → 3-D point [X, Y, Z] in camera space.
        """
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [px, py], depth_m)

    def get_depth_units(self, depth_frame):
        return depth_frame.get_units()

    def stop(self):
        if self._running:
            self.pipeline.stop()
            self._running = False
            print("[Camera] Pipeline stopped.")
