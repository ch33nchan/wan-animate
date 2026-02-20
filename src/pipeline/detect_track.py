from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config.schema import DetectorConfig, TrackerConfig


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)


@dataclass
class Detection:
    bbox: BBox
    score: float


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    smoothed_score: float
    hits: int = 1
    missed: int = 0
    frame_boxes: Dict[int, BBox] = field(default_factory=dict)
    total_area: float = 0.0


@dataclass
class TrackingResult:
    tracks: Dict[int, TrackState]
    frame_to_track_ids: Dict[int, List[int]]


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


class PersonDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._retried_on_cpu = False

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.cfg.model)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._load()
        device = self.cfg.device
        if device == "cuda":
            device = "cuda:0"
        try:
            results = self._model.predict(
                source=frame,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                classes=[self.cfg.person_class_id],
                device=device,
                verbose=False,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and "cuda" in msg and not self._retried_on_cpu:
                self._retried_on_cpu = True
                self.cfg.device = "cpu"
                self._model = None
                self._load()
                results = self._model.predict(
                    source=frame,
                    conf=self.cfg.conf,
                    iou=self.cfg.iou,
                    classes=[self.cfg.person_class_id],
                    device="cpu",
                    verbose=False,
                )
            else:
                raise
        if not results:
            return []

        out: List[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return out

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            out.append(Detection(bbox=BBox(x1, y1, x2, y2), score=float(conf[i])))

        return out


class TrackManager:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1

    @property
    def tracks(self) -> Dict[int, TrackState]:
        return self._tracks

    def _smooth_bbox(self, old: BBox, new: BBox) -> BBox:
        a = self.cfg.smoothing_alpha
        return BBox(
            x1=a * old.x1 + (1.0 - a) * new.x1,
            y1=a * old.y1 + (1.0 - a) * new.y1,
            x2=a * old.x2 + (1.0 - a) * new.x2,
            y2=a * old.y2 + (1.0 - a) * new.y2,
        )

    def update(self, detections: List[Detection], frame_index: int) -> List[int]:
        for t in self._tracks.values():
            t.missed += 1

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()

        candidates: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for track_id, track in self._tracks.items():
                iou = bbox_iou(det.bbox, track.bbox)
                if iou >= self.cfg.iou_threshold:
                    candidates.append((iou, track_id, det_idx))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, track_id, det_idx in candidates:
            if track_id in assigned_tracks or det_idx in assigned_detections:
                continue
            track = self._tracks[track_id]
            det = detections[det_idx]
            track.bbox = self._smooth_bbox(track.bbox, det.bbox)
            track.smoothed_score = (
                self.cfg.smoothing_alpha * track.smoothed_score
                + (1.0 - self.cfg.smoothing_alpha) * det.score
            )
            track.hits += 1
            track.missed = 0
            track.frame_boxes[frame_index] = track.bbox
            track.total_area += track.bbox.area
            assigned_tracks.add(track_id)
            assigned_detections.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=det.bbox,
                smoothed_score=det.score,
                hits=1,
                missed=0,
                frame_boxes={frame_index: det.bbox},
                total_area=det.bbox.area,
            )

        to_remove = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed > self.cfg.max_missed
        ]
        for track_id in to_remove:
            del self._tracks[track_id]

        visible_track_ids: List[int] = []
        for track_id, track in self._tracks.items():
            if frame_index in track.frame_boxes and track.hits >= self.cfg.min_hits:
                visible_track_ids.append(track_id)

        return sorted(visible_track_ids)


def run_detection_tracking(
    frames: List[np.ndarray],
    detector_cfg: DetectorConfig,
    tracker_cfg: TrackerConfig,
) -> TrackingResult:
    detector = PersonDetector(detector_cfg)
    tracker = TrackManager(tracker_cfg)

    frame_to_track_ids: Dict[int, List[int]] = {}
    for idx, frame in enumerate(frames):
        detections = detector.detect(frame)
        track_ids = tracker.update(detections, idx)
        frame_to_track_ids[idx] = track_ids

    return TrackingResult(tracks=tracker.tracks, frame_to_track_ids=frame_to_track_ids)


def rank_tracks_for_fallback(
    tracking_result: TrackingResult,
    frame_width: int,
    frame_height: int,
) -> List[int]:
    cx = frame_width * 0.5
    cy = frame_height * 0.5

    scored: List[Tuple[float, int]] = []
    for track_id, track in tracking_result.tracks.items():
        if track.hits <= 0:
            continue
        avg_area = track.total_area / max(1, track.hits)
        tcx, tcy = track.bbox.center()
        center_dist = np.hypot(tcx - cx, tcy - cy)
        center_penalty = center_dist / max(1.0, np.hypot(cx, cy))
        score = (track.smoothed_score * 2.0) + (avg_area / max(1.0, frame_width * frame_height)) - center_penalty
        scored.append((score, track_id))

    scored.sort(reverse=True)
    return [track_id for _, track_id in scored]
