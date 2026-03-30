"""
navigator.py
------------
Converts a list of grid-cell waypoints into natural language directions
suitable for audio playback.

Strategy
--------
1. Walk the waypoint list and compute bearing changes between consecutive steps.
2. Group steps into segments of the same direction.
3. Convert each segment to plain English: "Move forward 3 steps", "Turn left", etc.
4. Prepend hazard warnings when relevant.
"""

import math
from typing import List, Tuple, Optional
from mapper import GRID_RES


# ── Cardinal / intercardinal direction labels ──────────────────────────────
# Bearing in degrees (0 = North/forward, clockwise positive)
DIR_LABELS = [
    (  22.5, "forward"),
    (  67.5, "forward-right"),
    ( 112.5, "right"),
    ( 157.5, "back-right"),
    ( 202.5, "back"),
    ( 247.5, "back-left"),
    ( 292.5, "left"),
    ( 337.5, "forward-left"),
    ( 360.0, "forward"),
]

STEP_SIZE_M = GRID_RES    # 1 grid cell = GRID_RES metres


def bearing(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Return bearing in degrees [0, 360) from cell a to cell b. 0 = row-decreasing (forward)."""
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    # atan2: angle from positive-x axis; we want 0 = north (row-decreasing)
    angle = math.degrees(math.atan2(dc, -dr)) % 360
    return angle


def bearing_to_label(deg: float) -> str:
    for threshold, label in DIR_LABELS:
        if deg < threshold:
            return label
    return "forward"


def relative_turn(prev_bearing: Optional[float], curr_bearing: float) -> str:
    """Describe turn needed to go from prev_bearing to curr_bearing."""
    if prev_bearing is None:
        return bearing_to_label(curr_bearing)

    diff = (curr_bearing - prev_bearing + 360) % 360

    if diff < 22.5 or diff > 337.5:
        return ""                   # no significant turn
    elif diff < 112.5:
        return "turn right"
    elif diff < 247.5:
        return "turn around"
    else:
        return "turn left"


def path_to_instructions(
    path: List[Tuple[int, int]],
    hazard_labels: List[str] = None,
    exit_found: bool = True,
) -> List[str]:
    """
    Convert waypoint path → list of spoken instruction strings.

    Parameters
    ----------
    path          : list of (row, col) from A*
    hazard_labels : list of hazard names currently active (e.g. ["fire", "knife"])
    exit_found    : whether an exit was located

    Returns
    -------
    List of instruction strings (say them in order).
    """
    instructions = []

    # ── Safety preamble ───────────────────────────────────────────────────────
    if hazard_labels:
        hazard_str = " and ".join(set(hazard_labels))
        instructions.append(f"Warning: {hazard_str} detected. Follow the escape route.")

    if not exit_found:
        instructions.append("Exit not yet located. Moving to safe area.")

    if not path or len(path) < 2:
        instructions.append("Stay where you are. No clear path found.")
        return instructions

    # ── Build segments ────────────────────────────────────────────────────────
    segments = []   # list of (direction_label, step_count)
    prev_b   = None
    seg_dir  = None
    seg_count = 0

    for i in range(1, len(path)):
        b     = bearing(path[i - 1], path[i])
        label = bearing_to_label(b)

        if label == seg_dir:
            seg_count += 1
        else:
            if seg_dir is not None:
                segments.append((seg_dir, seg_count, prev_b, b))
            seg_dir   = label
            seg_count = 1

        prev_b = b

    if seg_dir:
        segments.append((seg_dir, seg_count, prev_b, None))

    # ── Convert segments to sentences ─────────────────────────────────────────
    last_bearing = None
    for idx, (label, count, entry_b, _) in enumerate(segments):
        turn = relative_turn(last_bearing, entry_b) if last_bearing is not None else ""
        dist_m = round(count * STEP_SIZE_M, 1)

        parts = []
        if turn:
            parts.append(turn)

        if label in ("forward", "back"):
            parts.append(f"move {label} {dist_m} metres")
        else:
            parts.append(f"move {label} {dist_m} metres")

        instructions.append(". ".join(p.capitalize() for p in parts) + ".")
        last_bearing = entry_b

    # ── Final callout ─────────────────────────────────────────────────────────
    instructions.append("You have reached the exit. Get out now.")
    return instructions


def format_for_display(instructions: List[str]) -> str:
    """Join instructions into a single multi-line string for on-screen display."""
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(instructions))
