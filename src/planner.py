"""
planner.py
----------
A* path planning on the 2-D occupancy grid produced by mapper.py.

Finds the shortest safe path from the user's current cell to the
registered exit cell, avoiding OCCUPIED and HAZARD cells.

Returns a list of (row, col) waypoints from start → exit.
"""

import heapq
import numpy as np
from mapper import UNKNOWN, FREE, OCCUPIED, HAZARD, EXIT, GRID_H, GRID_W

# Cells we are allowed to traverse
TRAVERSABLE = {FREE, UNKNOWN, EXIT}

# Movement: 8-directional
MOVES = [
    (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),   # cardinal
    (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),   # diagonal
]
MOVE_COSTS = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]


def heuristic(a, b):
    """Octile distance heuristic for 8-directional movement."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (1.414 - 1) * min(dr, dc)


def astar(grid, start: tuple, goal: tuple):
    """
    A* search on the occupancy grid.

    Parameters
    ----------
    grid  : np.ndarray (H, W) uint8 — from OccupancyMapper.grid
    start : (row, col)
    goal  : (row, col)

    Returns
    -------
    List of (row, col) from start to goal (inclusive), or [] if no path found.
    """
    if start == goal:
        return [start]

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))

    came_from = {start: None}
    g_score   = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            return _reconstruct(came_from, current)

        for (dr, dc), cost in zip(MOVES, MOVE_COSTS):
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                continue
            if grid[nr, nc] not in TRAVERSABLE:
                continue

            tentative_g = g_score[current] + cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f, neighbor))

    return []   # no path found


def simplify_path(path, tolerance=2):
    """
    Reduce path to key waypoints using Ramer-Douglas-Peucker-style thinning.
    Keeps every `tolerance`-th point plus the final point.
    """
    if len(path) <= 2:
        return path
    simplified = path[::tolerance]
    if simplified[-1] != path[-1]:
        simplified.append(path[-1])
    return simplified


def _reconstruct(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path
