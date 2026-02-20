from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.config.schema import FallbackConfig, GeminiConfig, Stage2Config
from src.pipeline.detect_track import TrackingResult


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
        gemini_cfg: GeminiConfig,
        temporal_alpha: float,
        device: str,
    ) -> None:
        self.stage2_cfg = stage2_cfg
        self.fallback_cfg = fallback_cfg
        self.gemini_cfg = gemini_cfg
        self.temporal_alpha = float(max(0.0, min(1.0, temporal_alpha)))
        self.device = device

        self._seg_model = None
        self._seg_kind = "uninitialized"
        self._seg_retried_cpu = False

    @property
    def seg_backend(self) -> str:
        return self._seg_kind

    def _load_seg_model(self) -> None:
        if self._seg_model is not None:
            return

        sam3_path = Path(self.stage2_cfg.sam3_model_path)

        # Try SAM backend first (SAM3-compatible path from workflow), fallback to YOLO segmentation.
        try:
            from ultralytics import SAM

            sam_model = str(sam3_path) if sam3_path.exists() else self.stage2_cfg.seg_model_fallback
            self._seg_model = SAM(sam_model)
            self._seg_kind = "sam"
            return
        except Exception:
            pass

        from ultralytics import YOLO

        self._seg_model = YOLO(self.stage2_cfg.seg_model_fallback)
        self._seg_kind = "yolo_seg"

    def _image_to_b64(self, image: np.ndarray) -> str:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _extract_json(self, text: str) -> Optional[dict]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _gemini_points(
        self,
        frame: np.ndarray,
        target_description: str,
        api_key_env_var: str,
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        h, w = frame.shape[:2]
        fallback_points = [
            (w // 2, h // 2),
            (w // 2, int(h * 0.3)),
            (int(w * 0.35), int(h * 0.65)),
            (int(w * 0.65), int(h * 0.65)),
            (int(w * 0.1), int(h * 0.1)),
            (int(w * 0.9), int(h * 0.1)),
            (int(w * 0.1), int(h * 0.9)),
            (int(w * 0.9), int(h * 0.9)),
        ]
        fallback_labels = [1, 1, 1, 1, 0, 0, 0, 0]

        if not self.gemini_cfg.enabled:
            return fallback_points, fallback_labels

        api_key = os.getenv(api_key_env_var)
        if not api_key:
            return fallback_points, fallback_labels

        prompt = (
            "Select SAM prompt points for full-body segmentation of this target person. "
            f"Target: {target_description}. "
            "Return strict JSON only: "
            '{"positive":[{"x":int,"y":int}],"negative":[{"x":int,"y":int}]}. '
            "Provide exactly 4 positive body points and 4 negative background points."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": self._image_to_b64(frame),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        url = self.gemini_cfg.api_url_template.format(model=self.gemini_cfg.model_name, api_key=api_key)
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.gemini_cfg.timeout_sec) as resp:
                body = json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return fallback_points, fallback_labels

        text = ""
        for cand in body.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("text"):
                    text = part["text"]
                    break
            if text:
                break

        parsed = self._extract_json(text)
        if not parsed:
            return fallback_points, fallback_labels

        pos = parsed.get("positive", [])
        neg = parsed.get("negative", [])
        points: List[Tuple[int, int]] = []
        labels: List[int] = []

        for p in pos[:4]:
            if isinstance(p, dict) and "x" in p and "y" in p:
                x = int(max(0, min(w - 1, round(float(p["x"])))) )
                y = int(max(0, min(h - 1, round(float(p["y"])))) )
                points.append((x, y))
                labels.append(1)

        for p in neg[:4]:
            if isinstance(p, dict) and "x" in p and "y" in p:
                x = int(max(0, min(w - 1, round(float(p["x"])))) )
                y = int(max(0, min(h - 1, round(float(p["y"])))) )
                points.append((x, y))
                labels.append(0)

        if len(points) < 4:
            return fallback_points, fallback_labels

        return points, labels

    def _foreground_mask_ref(self, ref: np.ndarray) -> np.ndarray:
        # Remove white studio-like background from reference to avoid white cutout artifacts.
        hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
        white_bg = cv2.inRange(hsv, (0, 0, 170), (180, 80, 255))
        fg = cv2.bitwise_not(white_bg)
        fg = cv2.medianBlur(fg, 5)
        kernel = np.ones((5, 5), dtype=np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        return fg

    def _bbox_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (w - 1, h - 1), 255, thickness=-1)
        return mask

    def _predict_mask_sam(self, roi: np.ndarray, points: List[Tuple[int, int]], labels: List[int]) -> Optional[np.ndarray]:
        device = "cuda:0" if self.device == "cuda" else "cpu"

        try:
            results = self._seg_model.predict(
                source=roi,
                points=points,
                labels=labels,
                device=device,
                verbose=False,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and "cuda" in msg and not self._seg_retried_cpu:
                self._seg_retried_cpu = True
                if self.stage2_cfg.require_gpu:
                    return None
                self.device = "cpu"
                results = self._seg_model.predict(
                    source=roi,
                    points=points,
                    labels=labels,
                    device="cpu",
                    verbose=False,
                )
            else:
                return None

        if not results:
            return None
        res = results[0]
        if res.masks is None:
            return None
        masks = res.masks.data.cpu().numpy()
        if len(masks) == 0:
            return None
        mask = (masks[0] > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != roi.shape[:2]:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _predict_mask_yolo(self, roi: np.ndarray) -> Optional[np.ndarray]:
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
                if self.stage2_cfg.require_gpu:
                    return None
                self.device = "cpu"
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

        masks = res.masks.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
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
        if mask.shape[:2] != roi.shape[:2]:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _run_segmentation(
        self,
        roi: np.ndarray,
        sam_points: List[Tuple[int, int]],
        sam_labels: List[int],
    ) -> Optional[np.ndarray]:
        self._load_seg_model()

        if self._seg_kind == "sam":
            mask = self._predict_mask_sam(roi, sam_points, sam_labels)
            if mask is not None:
                return mask

        return self._predict_mask_yolo(roi)

    def _resize_reference(self, ref: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        ref_h, ref_w = ref.shape[:2]
        scale = max(out_w / max(1, ref_w), out_h / max(1, ref_h))
        nw = max(1, int(round(ref_w * scale)))
        nh = max(1, int(round(ref_h * scale)))
        resized = cv2.resize(ref, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x0 = max(0, (nw - out_w) // 2)
        y0 = max(0, (nh - out_h) // 2)
        crop = resized[y0 : y0 + out_h, x0 : x0 + out_w]
        if crop.shape[:2] != (out_h, out_w):
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        return crop

    def _replace_roi(
        self,
        roi: np.ndarray,
        ref_resized: np.ndarray,
        ref_fg_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        blur = self.stage2_cfg.mask_blur
        if blur % 2 == 0:
            blur += 1

        ref_mask_resized = cv2.resize(ref_fg_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined = cv2.bitwise_and(target_mask, ref_mask_resized)

        alpha = cv2.GaussianBlur(combined, (blur, blur), 0).astype(np.float32) / 255.0
        alpha *= self.stage2_cfg.overlay_alpha
        alpha = alpha[..., None]

        out = (roi.astype(np.float32) * (1.0 - alpha) + ref_resized.astype(np.float32) * alpha).astype(np.uint8)
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
        target_description: str,
        api_key_env_var: str,
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

        ref_image = cv2.imread(str(Path(ref_image_path)))
        if ref_image is None:
            raise FileNotFoundError(f"Could not read reference image: {ref_image_path}")
        ref_fg_mask = self._foreground_mask_ref(ref_image)

        first_frame = stage1_frames[0]
        points, labels = self._gemini_points(first_frame, target_description, api_key_env_var)

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

            rel_points = []
            for px, py in points:
                rel_points.append((max(0, min(roi.shape[1] - 1, px - x1)), max(0, min(roi.shape[0] - 1, py - y1))))

            mask = self._run_segmentation(roi, rel_points, labels)
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
            replaced_roi = self._replace_roi(roi, ref_resized, ref_fg_mask, mask)

            out_frame = frame.copy()
            out_frame[y1:y2, x1:x2] = replaced_roi

            if prev_frame is not None and self.temporal_alpha > 0.0:
                out_frame = cv2.addWeighted(prev_frame, self.temporal_alpha, out_frame, 1.0 - self.temporal_alpha, 0.0)

            prev_frame = out_frame
            replaced_frames += 1
            output.append(out_frame)

        return output, events, replaced_frames
