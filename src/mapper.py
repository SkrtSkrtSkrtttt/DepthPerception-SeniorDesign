"""
mapper.py  (optimised)
----------------------
Optimisations applied:
  1. ICP only runs every ICP_INTERVAL frames — camera doesn't teleport
  2. Grid projection uses vectorised numpy — no Python loops over points
  3. Camera cell marked free using array index directly
"""

import numpy as np
import open3d as o3d

# ── Grid parameters ───────────────────────────────────────────────────────────
GRID_RES = 0.05
GRID_W   = 400
GRID_H   = 400
ORIGIN_X = GRID_W // 2
ORIGIN_Z = GRID_H // 2

# ── ICP parameters ────────────────────────────────────────────────────────────
VOXEL_SIZE          = 0.05
ICP_MAX_ITER        = 30
ICP_DISTANCE_THRESH = 0.1

# ── ICP skip interval ─────────────────────────────────────────────────────────
ICP_INTERVAL = 3    # run ICP every 3rd frame only

# ── Occupancy values ──────────────────────────────────────────────────────────
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2
HAZARD   = 3
EXIT     = 4

HAZARD_INFLATE_M = 0.6
FLOOR_HEIGHT_MIN = -0.1
FLOOR_HEIGHT_MAX =  0.3


class OccupancyMapper:
    def __init__(self):
        self.grid        = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.uint8)
        self.camera_pose = np.eye(4)
        self.prev_pcd    = None
        self.exit_cell   = None
        self.hazard_cells = []
        self._frame_count = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, depth_image, depth_units, camera):
        self._frame_count += 1
        run_icp = (self._frame_count % ICP_INTERVAL == 0)

        pcd = self._depth_to_pcd(depth_image, depth_units, camera)
        if pcd is None or len(pcd.points) < 100:
            return

        pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=VOXEL_SIZE * 2, max_nn=30
            )
        )

        # ICP — only on designated frames
        if run_icp and self.prev_pcd is not None:
            delta = self._icp(self.prev_pcd, pcd_down)
            self.camera_pose = self.camera_pose @ delta

        self.prev_pcd = pcd_down
        self._project_to_grid(pcd_down)

    def mark_hazard(self, world_x, world_z):
        row, col = self._world_to_cell(world_x, world_z)
        if not self._in_bounds(row, col):
            return
        self.hazard_cells.append((row, col))

        inflate = int(HAZARD_INFLATE_M / GRID_RES)
        rows = np.clip(np.arange(row - inflate, row + inflate + 1), 0, GRID_H - 1)
        cols = np.clip(np.arange(col - inflate, col + inflate + 1), 0, GRID_W - 1)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        self.grid[rr, cc] = HAZARD

    def mark_exit(self, world_x, world_z):
        row, col = self._world_to_cell(world_x, world_z)
        if self._in_bounds(row, col):
            self.exit_cell = (row, col)
            self.grid[row, col] = EXIT
            print(f"[Mapper] Exit at world ({world_x:.2f}, {world_z:.2f}) → cell ({row},{col})")

    def get_user_cell(self, world_x, world_z):
        return self._world_to_cell(world_x, world_z)

    def world_to_cell(self, world_x, world_z):
        return self._world_to_cell(world_x, world_z)

    def cell_to_world(self, row, col):
        return (col - ORIGIN_X) * GRID_RES, (row - ORIGIN_Z) * GRID_RES

    def camera_world_pos(self):
        t = self.camera_pose[:3, 3]
        return t[0], t[2]

    def reset_hazards(self):
        self.hazard_cells = []
        self.grid[self.grid == HAZARD] = FREE

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _depth_to_pcd(self, depth_image, depth_units, camera):
        intr = camera.intrinsics
        h, w = depth_image.shape

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z    = depth_image.astype(np.float32) * depth_units

        valid = (z > 0.2) & (z < 6.0)
        z = z[valid]; u = u[valid]; v = v[valid]

        x = (u - intr.ppx) * z / intr.fx
        y = (v - intr.ppy) * z / intr.fy

        pts = np.stack([x, y, z], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def _icp(self, source, target):
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            ICP_DISTANCE_THRESH,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=ICP_MAX_ITER
            ),
        )
        return result.transformation

    def _project_to_grid(self, pcd_down):
        """Vectorised grid projection — no Python loops over points."""
        pts = np.asarray(pcd_down.points)
        if len(pts) == 0:
            return

        # Transform to world frame
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        world = (self.camera_pose @ pts_h.T).T
        wx, wy, wz = world[:, 0], world[:, 1], world[:, 2]

        # Keep floor-level points only
        floor_mask = (wy > FLOOR_HEIGHT_MIN) & (wy < FLOOR_HEIGHT_MAX)
        wx = wx[floor_mask]
        wz = wz[floor_mask]

        # Convert to grid indices (vectorised)
        rows = (wz / GRID_RES + ORIGIN_Z).astype(int)
        cols = (wx / GRID_RES + ORIGIN_X).astype(int)

        valid = (rows >= 0) & (rows < GRID_H) & (cols >= 0) & (cols < GRID_W)
        rows, cols = rows[valid], cols[valid]

        # Mark occupied — avoid overwriting hazard/exit cells
        mask = np.isin(self.grid[rows, cols], [HAZARD, EXIT], invert=True)
        self.grid[rows[mask], cols[mask]] = OCCUPIED

        # Mark camera position as free
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