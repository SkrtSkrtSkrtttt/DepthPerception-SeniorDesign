"""
detector.py  (optimised + fire model)
--------------------------------------
Three models running together:
  1. yolov8n.pt  — general COCO: hazards + people
  2. doors.pt    — dedicated door detector (every 10 frames)
  3. best.pt     — fire + smoke detector (every 5 frames)

Colour heuristic for fire has been removed entirely.
Requires: pip install dill  (needed to load best.pt)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO

# ── COCO class indices ────────────────────────────────────────────────────────
PERSON_CLASS = 0

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
YOLO_CONF   = 0.25
PERSON_CONF = 0.40
DOOR_CONF   = 0.40
FIRE_CONF   = 0.35   # slightly lower — better to catch a real fire early

# ── Inference resolution ──────────────────────────────────────────────────────
INFER_W = 320
INFER_H = 240

# ── Model run intervals ───────────────────────────────────────────────────────
DOOR_CHECK_INTERVAL = 10   # doors don't move
FIRE_CHECK_INTERVAL = 5    # fire moves fast — check more often than doors

# ── Depth gap fallback ────────────────────────────────────────────────────────
DOOR_GAP_THRESHOLD = 0.70
DOOR_MIN_WIDTH_PX  = 60
DOOR_FAR_DIST_M    = 2.5


@dataclass
class Detection:
    label:      str
    confidence: float
    bbox:       Tuple[int, int, int, int]
    depth_m:    float
    world_xyz:  Tuple[float, float, float]
    direction:  str


@dataclass
class DetectionResult:
    hazards: List[Detection] = field(default_factory=list)
    persons: List[Detection] = field(default_factory=list)
    doors:   List[Detection] = field(default_factory=list)


class HazardDetector:
    def __init__(
        self,
        general_model_path: str = "yolov8n.pt",
        door_model_path:    str = "doors.pt",
        fire_model_path:    str = "best.pt",
    ):
        print(f"[Detector] Loading general model: {general_model_path}")
        self.general_model = YOLO(general_model_path)
        self.general_names = self.general_model.names

        print(f"[Detector] Loading door model: {door_model_path}")
        self.door_model = YOLO(door_model_path)

        print(f"[Detector] Loading fire/smoke model: {fire_model_path}")
        self.fire_model = YOLO(fire_model_path)
        print(f"[Detector] Fire model classes: {self.fire_model.names}")

        self._frame_count     = 0
        self._last_door_dets: List[Detection] = []
        self._last_fire_dets: List[Detection] = []

        print("[Detector] All models loaded. Colour fire heuristic disabled.")

    def detect(self, color_image, depth_frame, camera) -> DetectionResult:
        self._frame_count += 1
        result = DetectionResult()
        h, w = color_image.shape[:2]

        # Downscale once — shared across all models
        small   = cv2.resize(color_image, (INFER_W, INFER_H))
        scale_x = w / INFER_W
        scale_y = h / INFER_H

        # ── 1. General YOLO — hazards + people ──────────────────────────────
        general_results = self.general_model(
            small, conf=YOLO_CONF, verbose=False
        )[0]

        for box in general_results.boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
            x1, y1 = int(sx1 * scale_x), int(sy1 * scale_y)
            x2, y2 = int(sx2 * scale_x), int(sy2 * scale_y)
            label  = self.general_names.get(cid, str(cid))

            det = self._make_detection(
                label, conf, x1, y1, x2, y2, depth_frame, camera, w
            )
            if det is None:
                continue

            if cid == PERSON_CLASS and conf >= PERSON_CONF:
                result.persons.append(det)
            elif cid in HAZARD_CLASSES:
                result.hazards.append(det)

        # ── 2. Fire + smoke model ────────────────────────────────────────────
        if self._frame_count % FIRE_CHECK_INTERVAL == 0:
            fire_results = self.fire_model(
                small, conf=FIRE_CONF, verbose=False
            )[0]

            new_fire_dets = []
            for box in fire_results.boxes:
                conf  = float(box.conf[0])
                cid   = int(box.cls[0])
                label = self.fire_model.names.get(cid, "fire")
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                x1, y1 = int(sx1 * scale_x), int(sy1 * scale_y)
                x2, y2 = int(sx2 * scale_x), int(sy2 * scale_y)

                det = self._make_detection(
                    label, conf, x1, y1, x2, y2, depth_frame, camera, w
                )
                if det:
                    new_fire_dets.append(det)

            self._last_fire_dets = new_fire_dets

        # Use cached fire detections on skipped frames
        for det in self._last_fire_dets:
            result.hazards.append(det)

            # Draw fire/smoke boxes (orange for fire, grey for smoke)
            x1, y1, x2, y2 = det.bbox
            col = (0, 80, 255) if det.label == "fire" else (160, 160, 160)
            cv2.rectangle(color_image, (x1, y1), (x2, y2), col, 2)
            cv2.rectangle(color_image, (x1, y1 - 20), (x1 + 130, y1), col, -1)
            cv2.putText(
                color_image,
                f"{det.label.upper()} {det.confidence:.0%}",
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # ── 3. Door model — every DOOR_CHECK_INTERVAL frames ─────────────────
        if self._frame_count % DOOR_CHECK_INTERVAL == 0:
            door_results = self.door_model(
                small, conf=DOOR_CONF, verbose=False
            )[0]

            new_door_dets = []
            for box in door_results.boxes:
                conf = float(box.conf[0])
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                x1, y1 = int(sx1 * scale_x), int(sy1 * scale_y)
                x2, y2 = int(sx2 * scale_x), int(sy2 * scale_y)

                det = self._make_detection(
                    "door", conf, x1, y1, x2, y2, depth_frame, camera, w
                )
                if det:
                    new_door_dets.append(det)

            self._last_door_dets = new_door_dets

        door_found_by_model = len(self._last_door_dets) > 0
        result.doors.extend(self._last_door_dets)

        # Draw door boxes
        for det in self._last_door_dets:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(color_image, (x1, y1 - 20), (x1 + 110, y1), (0, 255, 0), -1)
            cv2.putText(color_image, f"DOOR {det.confidence:.0%}",
                        (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.arrowedLine(
                color_image,
                ((x1 + x2) // 2, (y1 + y2) // 2 + 30),
                ((x1 + x2) // 2, (y1 + y2) // 2 - 30),
                (0, 255, 0), 3
            )

        # ── 4. Depth gap fallback — only if door model found nothing ─────────
        if not door_found_by_model:
            depth_det = self._detect_door_depth(
                color_image, depth_frame, camera, w, h
            )
            if depth_det:
                result.doors.append(depth_det)

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

    def _detect_door_depth(
        self, color_image, depth_frame, camera, frame_w, frame_h
    ) -> Optional[Detection]:
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

        cv2.arrowedLine(color_image,
                        (cx, frame_h // 2 + 40),
                        (cx, frame_h // 2 - 40),
                        (0, 200, 0), 2)
        cv2.putText(color_image, "EXIT? (depth)",
                    (cx - 40, frame_h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)

        return Detection(
            label="door", confidence=0.50,
            bbox=(open_cols[0], mid_top, open_cols[-1], mid_bottom),
            depth_m=d, world_xyz=tuple(xyz),
            direction=direction,
        )


def draw_detections(image, result: DetectionResult):
    """Draw hazard and person boxes. Fire/door boxes drawn inline above."""
    def _draw(det, col):
        # Skip fire/smoke — already drawn with custom styling inline
        if det.label in ("fire", "smoke"):
            return
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), col, 2)
        txt = f"{det.label} {det.depth_m:.2f}m {det.direction}"
        cv2.putText(image, txt, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    for d in result.hazards:
        _draw(d, (0, 0, 255))
    for d in result.persons:
        _draw(d, (255, 0, 0))

    return image