"""
mapper.py
---------
Builds and maintains a 2-D top-down occupancy grid of the kitchen.

Since the camera has NO angle encoder, we use Open3D's ICP (Iterative Closest
Point) to estimate how much the camera rotated between frames, then project
each depth frame into a shared world coordinate grid.

Grid conventions
----------------
- Origin (0, 0) = camera start position
- X axis  = camera's initial right direction
- Z axis  = camera's initial forward direction
- Grid cell size = GRID_RES metres
- Cell values:
    0   = unknown
    1   = free
    2   = occupied / obstacle
    3   = hazard zone (inflated around detected hazard)
    4   = exit / door
"""

import numpy as np
import open3d as o3d

# ── Grid parameters ──────────────────────────────────────────────────────────
GRID_RES    = 0.05          # metres per cell (5 cm)
GRID_W      = 400           # cells → 20 m wide
GRID_H      = 400           # cells → 20 m deep
ORIGIN_X    = GRID_W // 2  # camera starts at grid centre
ORIGIN_Z    = GRID_H // 2

# ── ICP parameters ───────────────────────────────────────────────────────────
VOXEL_SIZE          = 0.05   # downsample before ICP (metres)
ICP_MAX_ITER        = 30
ICP_DISTANCE_THRESH = 0.1    # metres

# ── Occupancy values ─────────────────────────────────────────────────────────
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2
HAZARD   = 3
EXIT     = 4

HAZARD_INFLATE_M = 0.6   # safety bubble around hazards (metres)
FLOOR_HEIGHT_MIN = -0.1  # points below this (camera frame Y) = floor
FLOOR_HEIGHT_MAX =  0.3  # points above this = potential obstacle at floor level


class OccupancyMapper:
    def __init__(self):
        self.grid = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.uint8)
        self.camera_pose = np.eye(4)          # 4×4 transform: camera → world
        self.prev_pcd    = None               # previous downsampled point cloud
        self.exit_cell   = None               # (row, col) of detected exit
        self.hazard_cells = []                # list of (row, col) hazard centres

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, depth_image, depth_units, camera):
        """
        Call once per frame.
        1. Build point cloud from depth image.
        2. ICP against previous cloud → estimate camera motion.
        3. Project floor-level points into occupancy grid.
        """
        pcd = self._depth_to_pcd(depth_image, depth_units, camera)
        if pcd is None or len(pcd.points) < 100:
            return

        pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=VOXEL_SIZE * 2, max_nn=30
            )
        )

        # ── ICP pose update ──────────────────────────────────────────────────
        if self.prev_pcd is not None:
            delta = self._icp(self.prev_pcd, pcd_down)
            self.camera_pose = self.camera_pose @ delta   # accumulate

        self.prev_pcd = pcd_down

        # ── Project into grid ────────────────────────────────────────────────
        self._project_to_grid(pcd_down)

    def mark_hazard(self, world_x, world_z):
        """Mark a hazard at world coordinates (x, z) and inflate a safety bubble."""
        row, col = self._world_to_cell(world_x, world_z)
        if not self._in_bounds(row, col):
            return
        self.hazard_cells.append((row, col))

        inflate_cells = int(HAZARD_INFLATE_M / GRID_RES)
        for dr in range(-inflate_cells, inflate_cells + 1):
            for dc in range(-inflate_cells, inflate_cells + 1):
                r, c = row + dr, col + dc
                if self._in_bounds(r, c):
                    self.grid[r, c] = HAZARD

    def mark_exit(self, world_x, world_z):
        """Register the exit/door location in world coordinates."""
        row, col = self._world_to_cell(world_x, world_z)
        if self._in_bounds(row, col):
            self.exit_cell = (row, col)
            self.grid[row, col] = EXIT
            print(f"[Mapper] Exit registered at world ({world_x:.2f}, {world_z:.2f}) → cell ({row},{col})")

    def get_user_cell(self, world_x, world_z):
        return self._world_to_cell(world_x, world_z)

    def world_to_cell(self, world_x, world_z):
        return self._world_to_cell(world_x, world_z)

    def cell_to_world(self, row, col):
        x = (col - ORIGIN_X) * GRID_RES
        z = (row - ORIGIN_Z) * GRID_RES
        return x, z

    def camera_world_pos(self):
        """Return (x, z) of current camera position in world space."""
        t = self.camera_pose[:3, 3]
        return t[0], t[2]

    def reset_hazards(self):
        """Clear all hazard markings (call at start of each frame)."""
        self.hazard_cells = []
        self.grid[self.grid == HAZARD] = FREE

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _depth_to_pcd(self, depth_image, depth_units, camera):
        """Convert depth image to Open3D point cloud using camera intrinsics."""
        intr = camera.intrinsics
        h, w = depth_image.shape

        # Pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_image.astype(np.float32) * depth_units

        # Mask invalid
        valid = (z > 0.2) & (z < 6.0)
        z = z[valid]
        u = u[valid]
        v = v[valid]

        x = (u - intr.ppx) * z / intr.fx
        y = (v - intr.ppy) * z / intr.fy

        pts = np.stack([x, y, z], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def _icp(self, source, target):
        """Return 4×4 transform that maps source → target (camera delta)."""
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            ICP_DISTANCE_THRESH,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=ICP_MAX_ITER
            ),
        )
        return result.transformation

    def _project_to_grid(self, pcd_down):
        """
        Transform point cloud into world frame, keep only floor-level points,
        and mark corresponding grid cells as OCCUPIED or FREE.
        """
        pts = np.asarray(pcd_down.points)
        if len(pts) == 0:
            return

        # Transform to world frame
        pts_h   = np.hstack([pts, np.ones((len(pts), 1))])
        world   = (self.camera_pose @ pts_h.T).T
        wx, wy, wz = world[:, 0], world[:, 1], world[:, 2]

        # Keep points that are near floor level (camera-Y ≈ height above floor)
        floor_mask = (wy > FLOOR_HEIGHT_MIN) & (wy < FLOOR_HEIGHT_MAX)
        wx = wx[floor_mask]
        wz = wz[floor_mask]

        rows = (wz / GRID_RES + ORIGIN_Z).astype(int)
        cols = (wx / GRID_RES + ORIGIN_X).astype(int)

        valid = (
            (rows >= 0) & (rows < GRID_H) &
            (cols >= 0) & (cols < GRID_W)
        )
        rows, cols = rows[valid], cols[valid]

        # Everything seen at floor level = occupied (obstacle footprint)
        for r, c in zip(rows, cols):
            if self.grid[r, c] not in (HAZARD, EXIT):
                self.grid[r, c] = OCCUPIED

        # Mark camera's current cell as free
        cam_x, cam_z = self.camera_world_pos()
        cr, cc = self._world_to_cell(cam_x, cam_z)
        if self._in_bounds(cr, cc):
            self.grid[cr, cc] = FREE

    def _world_to_cell(self, world_x, world_z):
        col = int(world_x / GRID_RES + ORIGIN_X)
        row = int(world_z / GRID_RES + ORIGIN_Z)
        return row, col

    def _in_bounds(self, row, col):
        return 0 <= row < GRID_H and 0 <= col < GRID_W
