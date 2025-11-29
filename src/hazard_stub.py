"""
hazard_stub.py

Placeholder for smoke/fire hazard detection using a future YOLO model.
Right now this just returns an empty list so the rest of the pipeline
can call it without breaking.
"""

from typing import List, Tuple
import numpy as np

# Detection format: (x, y, w, h, label, score)
Detection = Tuple[int, int, int, int, str, float]


def detect_hazards_rgb(frame_bgr: np.ndarray) -> List[Detection]:
    """
    Detect visual hazards (e.g., smoke, fire) in an RGB frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Current RGB frame in BGR format (OpenCV default).

    Returns
    -------
    List[Detection]
        Empty for now; later will contain bounding boxes and labels
        for hazards detected by a YOLO model.
    """
    # TODO: integrate YOLO model here in a later phase.
    return []
