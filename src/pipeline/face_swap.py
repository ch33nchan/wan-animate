from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config.schema import SwapConfig
from src.pipeline.detect_track import BBox


@dataclass
class SwapResult:
    frame: np.ndarray
    swapped: bool
    confidence: float


class InsightFaceSwapper:
    def __init__(self, cfg: SwapConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self.face_app = None
        self.swapper = None
        self.source_face = None

    def load(self, ref_image_path: str) -> None:
        try:
            import insightface
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise RuntimeError("insightface is required for face swapping") from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device.lower() == "cpu":
            providers = ["CPUExecutionProvider"]

        app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1, det_size=(self.cfg.face_det_size, self.cfg.face_det_size))

        model_path = self.cfg.inswapper_model_path or "inswapper_128.onnx"
        swapper = get_model(model_path, providers=providers)

        ref_image = cv2.imread(str(Path(ref_image_path)))
        if ref_image is None:
            raise FileNotFoundError(f"Could not read reference image: {ref_image_path}")

        ref_faces = app.get(ref_image)
        if not ref_faces:
            raise RuntimeError("No face found in reference image")

        source_face = max(ref_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        self.face_app = app
        self.swapper = swapper
        self.source_face = source_face

    def _select_target_face(self, roi_faces: list, roi_w: int, roi_h: int):
        cx = roi_w * 0.5
        cy = roi_h * 0.5

        def score(face: object) -> float:
            bbox = face.bbox
            fx = 0.5 * (bbox[0] + bbox[2])
            fy = 0.5 * (bbox[1] + bbox[3])
            area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            dist = np.hypot(fx - cx, fy - cy)
            return float(area - dist * 100.0)

        return max(roi_faces, key=score)

    def swap_on_bbox(self, frame: np.ndarray, target_bbox: BBox) -> SwapResult:
        if self.face_app is None or self.swapper is None or self.source_face is None:
            raise RuntimeError("Swapper not loaded")

        h, w = frame.shape[:2]
        x1 = max(0, int(target_bbox.x1))
        y1 = max(0, int(target_bbox.y1))
        x2 = min(w, int(target_bbox.x2))
        y2 = min(h, int(target_bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return SwapResult(frame=frame, swapped=False, confidence=0.0)

        roi = frame[y1:y2, x1:x2].copy()
        roi_faces = self.face_app.get(roi)
        if not roi_faces:
            return SwapResult(frame=frame, swapped=False, confidence=0.0)

        target_face = self._select_target_face(roi_faces, roi.shape[1], roi.shape[0])
        swapped_roi = self.swapper.get(roi, target_face, self.source_face, paste_back=True)

        out = frame.copy()
        out[y1:y2, x1:x2] = swapped_roi

        face_bbox = target_face.bbox
        face_area = max(1.0, (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1]))
        conf = float(min(1.0, face_area / max(1.0, roi.shape[0] * roi.shape[1])))
        return SwapResult(frame=out, swapped=True, confidence=conf)
