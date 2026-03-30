"""
detector.py
-----------
YOLOv8-based detection for:
  - Kitchen hazards  (fire, knife, scissors, bottle, and more)
  - People           (for user localisation)
  - Doors            (depth-gap heuristic — no COCO door class)

Returns structured DetectionResult objects with 3-D world positions
derived from the aligned depth frame.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO

# ── COCO class indices ────────────────────────────────────────────────────────
PERSON_CLASS = 0

# All classes we treat as hazards / obstacles worth flagging
HAZARD_CLASSES = {
    43: "knife",
    76: "scissors",
    39: "bottle",
    41: "cup",
    45: "bowl",
    63: "laptop",
    67: "cell phone",
    72: "tv",
    65: "remote",
    56: "chair",
    57: "couch",
    58: "potted plant",
    60: "dining table",
    73: "book",
    77: "teddy bear",
}

# ── Confidence thresholds ─────────────────────────────────────────────────────
YOLO_CONF   = 0.25   # lowered to catch more objects
PERSON_CONF = 0.40

# ── Fire / flame detection thresholds (HSV colour heuristic) ─────────────────
FIRE_HUE_LOW  = np.array([0,  100, 200], dtype=np.uint8)
FIRE_HUE_HIGH = np.array([35, 255, 255], dtype=np.uint8)
FIRE_MIN_AREA = 500   # pixels²

# ── Door detection (depth gap) ────────────────────────────────────────────────
DOOR_GAP_THRESHOLD = 0.70   # fraction of middle rows that must be far/open
DOOR_MIN_WIDTH_PX  = 60     # minimum column width to count as a doorway
DOOR_FAR_DIST_M    = 2.5    # beyond this = "open space / doorway"


@dataclass
class Detection:
    label:      str
    confidence: float
    bbox:       Tuple[int, int, int, int]    # x1, y1, x2, y2 (pixels)
    depth_m:    float                         # distance to centre, metres
    world_xyz:  Tuple[float, float, float]   # X, Y, Z in camera space
    direction:  str                           # LEFT / CENTER / RIGHT


@dataclass
class DetectionResult:
    hazards: List[Detection] = field(default_factory=list)
    persons: List[Detection] = field(default_factory=list)
    doors:   List[Detection] = field(default_factory=list)


class HazardDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        print(f"[Detector] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.names = self.model.names
        print(f"[Detector] Loaded {len(self.names)} classes.")
        print("[Detector] Ready.")

    def detect(self, color_image, depth_frame, camera) -> DetectionResult:
        """
        Run full detection pipeline on one frame pair.

        Parameters
        ----------
        color_image : np.ndarray  (H, W, 3) BGR
        depth_frame : rs.depth_frame
        camera      : RealSenseCamera

        Returns
        -------
        DetectionResult
        """
        result = DetectionResult()
        h, w = color_image.shape[:2]

        # ── 1. YOLOv8 inference ──────────────────────────────────────────────
        yolo_results = self.model(color_image, conf=YOLO_CONF, verbose=False)[0]

        for box in yolo_results.boxes:
            cid   = int(box.cls[0])
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = self.names.get(cid, str(cid))

            det = self._make_detection(
                label, conf, x1, y1, x2, y2, depth_frame, camera, w
            )
            if det is None:
                continue

            if cid == PERSON_CLASS and conf >= PERSON_CONF:
                result.persons.append(det)
            elif cid in HAZARD_CLASSES:
                result.hazards.append(det)

        # ── 2. Fire heuristic ────────────────────────────────────────────────
        result.hazards.extend(
            self._detect_fire(color_image, depth_frame, camera, w)
        )

        # ── 3. Door via depth gap ────────────────────────────────────────────
        door_det = self._detect_door_depth(color_image, depth_frame, camera, w, h)
        if door_det:
            result.doors.append(door_det)

        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_detection(
        self, label, conf, x1, y1, x2, y2, depth_frame, camera, frame_w
    ) -> Optional[Detection]:
        cx = max(0, min((x1 + x2) // 2, frame_w - 1))
        cy = (y1 + y2) // 2
        d  = depth_frame.get_distance(cx, cy)

        if d <= 0 or d > 8.0:
            return None

        xyz = camera.deproject_pixel(cx, cy, d)
        direction = (
            "LEFT"   if cx < frame_w / 3 else
            "RIGHT"  if cx > 2 * frame_w / 3 else
            "CENTER"
        )
        return Detection(
            label=label, confidence=conf,
            bbox=(x1, y1, x2, y2),
            depth_m=d, world_xyz=tuple(xyz),
            direction=direction,
        )

    def _detect_fire(self, color_image, depth_frame, camera, frame_w) -> List[Detection]:
        hsv  = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, FIRE_HUE_LOW, FIRE_HUE_HIGH)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for cnt in contours:
            if cv2.contourArea(cnt) < FIRE_MIN_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            det = self._make_detection(
                "fire", 0.90, x, y, x + bw, y + bh,
                depth_frame, camera, frame_w
            )
            if det:
                dets.append(det)
        return dets

    def _detect_door_depth(
        self, color_image, depth_frame, camera, frame_w, frame_h
    ) -> Optional[Detection]:
        """
        Detect an open doorway as a wide vertical band of far/zero depth.
        Purely geometric — no COCO class needed.
        """
        depth_units = depth_frame.get_units()
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_units

        mid_top    = frame_h // 4
        mid_bottom = 3 * frame_h // 4
        mid_band   = depth_image[mid_top:mid_bottom, :]

        far_mask   = (mid_band == 0) | (mid_band > DOOR_FAR_DIST_M)
        col_scores = far_mask.mean(axis=0)
        open_cols  = np.where(col_scores > DOOR_GAP_THRESHOLD)[0]

        if len(open_cols) < DOOR_MIN_WIDTH_PX:
            return None

        cx = int(open_cols.mean())
        cy = frame_h // 2
        d  = depth_frame.get_distance(cx, cy)
        if d <= 0:
            d = DOOR_FAR_DIST_M

        xyz = camera.deproject_pixel(cx, cy, d)
        direction = (
            "LEFT"   if cx < frame_w / 3 else
            "RIGHT"  if cx > 2 * frame_w / 3 else
            "CENTER"
        )

        # Draw exit indicator on frame
        cv2.arrowedLine(
            color_image,
            (cx, frame_h // 2 + 40),
            (cx, frame_h // 2 - 40),
            (0, 255, 0), 3
        )
        cv2.putText(
            color_image, "EXIT?",
            (cx - 20, frame_h // 2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        return Detection(
            label="door", confidence=0.75,
            bbox=(open_cols[0], mid_top, open_cols[-1], mid_bottom),
            depth_m=d, world_xyz=tuple(xyz),
            direction=direction,
        )


def draw_detections(image, result: DetectionResult):
    """Draw bounding boxes on image in-place."""
    colour_map = {
        "hazard": (0,   0,   255),
        "person": (255, 0,   0  ),
        "door":   (0,   255, 0  ),
    }

    def _draw(det, cat):
        x1, y1, x2, y2 = det.bbox
        col = colour_map[cat]
        cv2.rectangle(image, (x1, y1), (x2, y2), col, 2)
        txt = f"{det.label} {det.depth_m:.2f}m {det.direction}"
        cv2.putText(
            image, txt, (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2
        )

    for d in result.hazards:
        _draw(d, "hazard")
    for d in result.persons:
        _draw(d, "person")
    for d in result.doors:
        _draw(d, "door")

    return image