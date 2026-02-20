from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config.schema import FallbackConfig, Stage2Config
from src.pipeline.detect_track import BBox, TrackingResult


@dataclass
class Stage2Event:
    frame_index: int
    reason: str
    confidence: float


class FullCharacterReplacer:
    def __init__(
        self,
        stage2_cfg: Stage2Config,
        fallback_cfg: FallbackConfig,
        temporal_alpha: float,
        device: str,
    ) -> None:
        self.stage2_cfg = stage2_cfg
        self.fallback_cfg = fallback_cfg
        self.temporal_alpha = float(max(0.0, min(1.0, temporal_alpha)))
        self.device = device

        self._seg_model = None
        self._seg_retried_cpu = False

    def _load_seg_model(self) -> None:
        if self._seg_model is not None:
            return
        from ultralytics import YOLO

        self._seg_model = YOLO(self.stage2_cfg.seg_model)

    def _read_reference(self, ref_image_path: str) -> np.ndarray:
        image = cv2.imread(str(Path(ref_image_path)))
        if image is None:
            raise FileNotFoundError(f"Could not read reference image: {ref_image_path}")
        return image

    def _bbox_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (w - 1, h - 1), 255, thickness=-1)
        return mask

    def _run_segmentation(self, roi: np.ndarray) -> Optional[np.ndarray]:
        self._load_seg_model()

        device = "cuda:0" if self.device == "cuda" else "cpu"
        try:
            results = self._seg_model.predict(
                source=roi,
                conf=self.stage2_cfg.seg_conf,
                classes=[0],
                device=device,
                verbose=False,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and "cuda" in msg and not self._seg_retried_cpu:
                self._seg_retried_cpu = True
                self.device = "cpu"
                self._seg_model = None
                self._load_seg_model()
                results = self._seg_model.predict(
                    source=roi,
                    conf=self.stage2_cfg.seg_conf,
                    classes=[0],
                    device="cpu",
                    verbose=False,
                )
            else:
                return None

        if not results:
            return None

        res = results[0]
        if res.masks is None or res.boxes is None:
            return None

        masks_data = res.masks.data
        boxes_data = res.boxes.xyxy

        if masks_data is None or boxes_data is None:
            return None

        masks = masks_data.cpu().numpy()
        boxes = boxes_data.cpu().numpy()
        if len(masks) == 0:
            return None

        roi_h, roi_w = roi.shape[:2]
        cx = roi_w * 0.5
        cy = roi_h * 0.5

        best_idx = -1
        best_score = -1e18
        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes[i]
            area = max(1.0, (x2 - x1) * (y2 - y1))
            bx = 0.5 * (x1 + x2)
            by = 0.5 * (y1 + y2)
            dist = np.hypot(bx - cx, by - cy)
            score = area - 50.0 * dist
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            return None

        mask = (masks[best_idx] > 0.5).astype(np.uint8) * 255
        if mask.shape[0] != roi_h or mask.shape[1] != roi_w:
            mask = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        return mask

    def _resize_reference(self, ref: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        ref_h, ref_w = ref.shape[:2]
        if ref_h <= 0 or ref_w <= 0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)

        scale = max(out_w / ref_w, out_h / ref_h)
        nw = max(1, int(round(ref_w * scale)))
        nh = max(1, int(round(ref_h * scale)))
        resized = cv2.resize(ref, (nw, nh), interpolation=cv2.INTER_CUBIC)

        x0 = max(0, (nw - out_w) // 2)
        y0 = max(0, (nh - out_h) // 2)
        crop = resized[y0 : y0 + out_h, x0 : x0 + out_w]
        if crop.shape[0] != out_h or crop.shape[1] != out_w:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        return crop

    def _replace_roi(self, roi: np.ndarray, ref_resized: np.ndarray, mask: np.ndarray) -> np.ndarray:
        blur = self.stage2_cfg.mask_blur
        if blur % 2 == 0:
            blur += 1
        alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
        alpha *= self.stage2_cfg.overlay_alpha
        alpha = alpha[..., None]

        out = (roi.astype(np.float32) * (1.0 - alpha) + ref_resized.astype(np.float32) * alpha).astype(
            np.uint8
        )
        return out

    def _frame_confidence(self, mask: np.ndarray) -> float:
        active = int((mask > 0).sum())
        if active <= 0:
            return 0.0
        return min(1.0, active / float(max(1, self.stage2_cfg.min_mask_pixels)))

    def apply(
        self,
        stage1_frames: List[np.ndarray],
        tracking_result: TrackingResult,
        target_track_id: int,
        ref_image_path: str,
    ) -> Tuple[List[np.ndarray], List[Stage2Event], int]:
        output: List[np.ndarray] = []
        events: List[Stage2Event] = []
        replaced_frames = 0

        target_track = tracking_result.tracks.get(target_track_id)
        if target_track is None:
            for idx, frame in enumerate(stage1_frames):
                output.append(frame)
                events.append(Stage2Event(frame_index=idx, reason="missing_target_track", confidence=0.0))
            return output, events, 0

        ref_image = self._read_reference(ref_image_path)
        prev_frame: Optional[np.ndarray] = None

        for idx, frame in enumerate(stage1_frames):
            bbox = target_track.frame_boxes.get(idx)
            if bbox is None:
                output.append(frame)
                events.append(Stage2Event(frame_index=idx, reason="track_not_visible", confidence=0.0))
                continue

            x1 = max(0, int(bbox.x1))
            y1 = max(0, int(bbox.y1))
            x2 = min(frame.shape[1], int(bbox.x2))
            y2 = min(frame.shape[0], int(bbox.y2))
            if x2 <= x1 or y2 <= y1:
                output.append(frame)
                events.append(Stage2Event(frame_index=idx, reason="invalid_bbox", confidence=0.0))
                continue

            roi = frame[y1:y2, x1:x2].copy()
            mask = self._run_segmentation(roi)
            if mask is None and self.stage2_cfg.use_bbox_mask_fallback:
                mask = self._bbox_mask(roi.shape[0], roi.shape[1])

            if mask is None:
                output.append(frame)
                events.append(Stage2Event(frame_index=idx, reason="mask_unavailable", confidence=0.0))
                continue

            confidence = self._frame_confidence(mask)
            if confidence < self.fallback_cfg.stage2_confidence_threshold:
                output.append(frame)
                events.append(
                    Stage2Event(
                        frame_index=idx,
                        reason="stage2_confidence_below_threshold",
                        confidence=confidence,
                    )
                )
                continue

            ref_resized = self._resize_reference(ref_image, roi.shape[1], roi.shape[0])
            replaced_roi = self._replace_roi(roi, ref_resized, mask)

            out_frame = frame.copy()
            out_frame[y1:y2, x1:x2] = replaced_roi

            if prev_frame is not None and self.temporal_alpha > 0.0:
                out_frame = cv2.addWeighted(prev_frame, self.temporal_alpha, out_frame, 1.0 - self.temporal_alpha, 0.0)

            prev_frame = out_frame
            replaced_frames += 1
            output.append(out_frame)

        return output, events, replaced_frames
