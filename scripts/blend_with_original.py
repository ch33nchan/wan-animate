#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class StreamInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def get_stream_info(path: str) -> StreamInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return StreamInfo(fps=fps, width=width, height=height, frame_count=frame_count)


class SequentialFrameFetcher:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.idx = -1
        self.frame: Optional[np.ndarray] = None
        self.eof = False

    def get(self, target_idx: int) -> Optional[np.ndarray]:
        if self.eof:
            return self.frame
        while self.idx < target_idx:
            ok, frame = self.cap.read()
            if not ok:
                self.eof = True
                return self.frame
            self.idx += 1
            self.frame = frame
        return self.frame

    def close(self) -> None:
        self.cap.release()


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def color_match_in_mask(src_bgr: np.ndarray, dst_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = dst_bgr.astype(np.float32).copy()
    valid = mask > 0.01
    if valid.sum() < 64:
        return dst_bgr

    src = src_bgr.astype(np.float32)
    dst = dst_bgr.astype(np.float32)

    for c in range(3):
        s_vals = src[..., c][valid]
        d_vals = dst[..., c][valid]
        s_mean, s_std = float(s_vals.mean()), float(s_vals.std() + 1e-6)
        d_mean, d_std = float(d_vals.mean()), float(d_vals.std() + 1e-6)
        out[..., c] = (out[..., c] - d_mean) * (s_std / d_std) + s_mean

    return np.clip(out, 0, 255).astype(np.uint8)


def keep_largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    best_label = 0
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_area = area
            best_label = label

    if best_label == 0:
        return np.zeros_like(mask)

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(1.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    area_b = max(1.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    return inter / (area_a + area_b - inter + 1e-6)


def _select_target_bbox(
    bboxes: np.ndarray,
    confs: np.ndarray,
    frame_w: int,
    frame_h: int,
    prev_bbox: Optional[np.ndarray],
) -> np.ndarray:
    center = np.array([frame_w * 0.5, frame_h * 0.5], dtype=np.float32)
    diag = float(np.hypot(frame_w, frame_h)) + 1e-6

    best_idx = 0
    best_score = -1e9
    for i in range(bboxes.shape[0]):
        box = bboxes[i]
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        dist_score = 1.0 - (float(np.hypot(cx - center[0], cy - center[1])) / diag)
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        area_score = min(1.0, area / float(frame_w * frame_h))
        conf_score = float(confs[i])

        score = 0.45 * dist_score + 0.35 * conf_score + 0.20 * area_score
        if prev_bbox is not None:
            score += 0.70 * _bbox_iou(box, prev_bbox)

        if score > best_score:
            best_score = score
            best_idx = i

    return bboxes[best_idx]


class SAM2MaskProvider:
    def __init__(
        self,
        detector_model: str,
        sam2_model: str,
        seg_model: str,
        device: str,
        det_conf: float,
        img_size: int,
    ):
        try:
            from ultralytics import SAM, YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is required for --mask_mode sam2") from exc

        self.detector = YOLO(detector_model)
        self.segmenter = SAM(sam2_model)
        self.seg_model = YOLO(seg_model) if seg_model else None
        self.device = device
        self.det_conf = det_conf
        self.img_size = img_size
        self.prev_bbox: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cudaerrormemoryallocation" in msg or "cuda error" in msg

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _predict_with_fallback(self, model, **kwargs):
        try:
            return model.predict(**kwargs, device=self.device, verbose=False)
        except Exception as exc:
            if self.device != "cpu" and self._is_cuda_oom(exc):
                self._empty_cuda_cache()
                self.device = "cpu"
                return model.predict(**kwargs, device="cpu", verbose=False)
            raise

    def get_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        det = self._predict_with_fallback(
            self.detector,
            source=frame_bgr,
            classes=[0],
            conf=self.det_conf,
            imgsz=self.img_size,
        )[0]

        if det.boxes is None or len(det.boxes) == 0:
            return self.prev_mask.copy() if self.prev_mask is not None else np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

        boxes = det.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = det.boxes.conf.detach().cpu().numpy().astype(np.float32)
        target = _select_target_bbox(boxes, confs, frame_bgr.shape[1], frame_bgr.shape[0], self.prev_bbox)

        x1 = int(clamp(int(round(target[0])), 0, frame_bgr.shape[1] - 1))
        y1 = int(clamp(int(round(target[1])), 0, frame_bgr.shape[0] - 1))
        x2 = int(clamp(int(round(target[2])), x1 + 1, frame_bgr.shape[1]))
        y2 = int(clamp(int(round(target[3])), y1 + 1, frame_bgr.shape[0]))
        bbox = [x1, y1, x2, y2]

        seg = self._predict_with_fallback(
            self.segmenter,
            source=frame_bgr,
            bboxes=[bbox],
        )[0]

        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        if seg.masks is not None and seg.masks.data is not None and len(seg.masks.data) > 0:
            m = seg.masks.data[0].detach().cpu().numpy()
            mask = (m > 0.5).astype(np.uint8) * 255

        # SAM occasionally returns near-rectangular box masks; fallback to YOLO-seg person mask.
        if self.seg_model is not None and self._is_boxy(mask):
            seg_fallback = self._predict_with_fallback(
                self.seg_model,
                source=frame_bgr,
                classes=[0],
                conf=max(0.1, self.det_conf * 0.8),
                imgsz=self.img_size,
            )[0]
            if seg_fallback.masks is not None and seg_fallback.masks.data is not None and len(seg_fallback.masks.data) > 0:
                boxes_f = (
                    seg_fallback.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                    if seg_fallback.boxes is not None and len(seg_fallback.boxes) > 0
                    else np.zeros((0, 4), dtype=np.float32)
                )
                if boxes_f.shape[0] > 0:
                    ious = np.array([_bbox_iou(b, np.array(bbox, dtype=np.float32)) for b in boxes_f], dtype=np.float32)
                    best_i = int(np.argmax(ious))
                    if ious[best_i] > 0.1:
                        m2 = seg_fallback.masks.data[best_i].detach().cpu().numpy()
                        mask2 = (m2 > 0.5).astype(np.uint8) * 255
                        if int(mask2.sum()) > 0:
                            mask = mask2

        if mask.sum() == 0 and self.prev_mask is not None:
            mask = self.prev_mask.copy()

        self.prev_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.prev_mask = mask.copy()
        return mask

    @staticmethod
    def _is_boxy(mask: np.ndarray) -> bool:
        ys, xs = np.where(mask > 0)
        if xs.size < 64:
            return False
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        box_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
        mask_area = int((mask > 0).sum())
        fill = mask_area / float(box_area)
        return fill > 0.84


def run() -> None:
    parser = argparse.ArgumentParser(description="Blend generated target back into original video")
    parser.add_argument("--original", required=True, help="Original source video")
    parser.add_argument("--generated", required=True, help="Generated WAN output video")
    parser.add_argument("--mask", default="", help="Mask video path when --mask_mode video")
    parser.add_argument("--mask_mode", choices=["video", "sam2"], default="video")
    parser.add_argument("--out", required=True, help="Final composited output video")

    parser.add_argument("--detector_model", default="yolov8n.pt")
    parser.add_argument("--sam2_model", default="sam2.1_b.pt")
    parser.add_argument("--seg_model", default="yolov8n-seg.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--det_conf", type=float, default=0.20)
    parser.add_argument("--img_size", type=int, default=960)

    parser.add_argument("--edge_blur", type=int, default=21)
    parser.add_argument("--mask_dilate", type=int, default=5)
    parser.add_argument("--mask_erode", type=int, default=1)
    parser.add_argument("--mask_threshold", type=int, default=96)
    parser.add_argument("--min_component_area_ratio", type=float, default=0.01)
    parser.add_argument("--max_row_coverage", type=float, default=0.65)
    parser.add_argument("--alpha_scale", type=float, default=0.95)
    parser.add_argument("--temporal_alpha", type=float, default=0.80)
    parser.add_argument("--protect_bottom_ratio", type=float, default=0.16)
    parser.add_argument("--diff_threshold", type=float, default=16.0)
    parser.add_argument("--keep_audio", action="store_true")
    args = parser.parse_args()

    if args.mask_mode == "video" and not args.mask:
        raise RuntimeError("--mask is required when --mask_mode video")

    orig_info = get_stream_info(args.original)
    gen_info = get_stream_info(args.generated)
    mask_info = get_stream_info(args.mask) if args.mask_mode == "video" else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp_out = args.out if not args.keep_audio else args.out + ".video_only.mp4"

    writer = cv2.VideoWriter(
        tmp_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        orig_info.fps,
        (orig_info.width, orig_info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {tmp_out}")

    orig_cap = cv2.VideoCapture(args.original)
    if not orig_cap.isOpened():
        raise RuntimeError(f"Cannot open original video: {args.original}")

    gen_fetcher = SequentialFrameFetcher(args.generated)
    mask_fetcher = SequentialFrameFetcher(args.mask) if args.mask_mode == "video" else None
    sam_provider = (
        SAM2MaskProvider(
            detector_model=args.detector_model,
            sam2_model=args.sam2_model,
            seg_model=args.seg_model,
            device=args.device,
            det_conf=args.det_conf,
            img_size=args.img_size,
        )
        if args.mask_mode == "sam2"
        else None
    )

    k_blur = args.edge_blur if args.edge_blur % 2 == 1 else args.edge_blur + 1
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.mask_dilate, args.mask_dilate))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.mask_erode, args.mask_erode))

    prev_alpha: Optional[np.ndarray] = None
    total = orig_info.frame_count if orig_info.frame_count > 0 else None
    pbar = tqdm(total=total, desc="Blending", unit="frame")

    frame_idx = 0
    while True:
        ok, orig = orig_cap.read()
        if not ok:
            break

        t = frame_idx / max(orig_info.fps, 1e-6)
        gen_idx = clamp(int(round(t * gen_info.fps)), 0, max(gen_info.frame_count - 1, 0))
        gen = gen_fetcher.get(gen_idx)

        if gen is None:
            writer.write(orig)
            frame_idx += 1
            pbar.update(1)
            continue

        if gen.shape[:2] != orig.shape[:2]:
            gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC)

        if args.mask_mode == "video":
            mask_idx = clamp(int(round(t * float(mask_info.fps))), 0, max(mask_info.frame_count - 1, 0))
            msk = mask_fetcher.get(mask_idx)
            if msk is None:
                writer.write(orig)
                frame_idx += 1
                pbar.update(1)
                continue
            if msk.shape[:2] != orig.shape[:2]:
                msk = cv2.resize(msk, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
            msk_gray = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY) if msk.ndim == 3 else msk
            _, m = cv2.threshold(msk_gray, args.mask_threshold, 255, cv2.THRESH_BINARY)
        else:
            m = sam_provider.get_mask(orig)

        min_area = int(orig.shape[0] * orig.shape[1] * args.min_component_area_ratio)
        m = keep_largest_component(m, min_area=min_area)

        max_cov = max(0.05, min(1.0, args.max_row_coverage))
        row_coverage = (m > 0).mean(axis=1)
        if max_cov < 0.999:
            high_rows = row_coverage > max_cov
            if np.any(high_rows):
                m[high_rows, :] = (m[high_rows, :] * 0.35).astype(np.uint8)

        if args.mask_erode > 1:
            m = cv2.erode(m, kernel_e, iterations=1)
        if args.mask_dilate > 1:
            m = cv2.dilate(m, kernel_d, iterations=1)

        m = cv2.GaussianBlur(m, (k_blur, k_blur), 0)
        alpha = (m.astype(np.float32) / 255.0) * args.alpha_scale

        # Constrain blend to truly changed regions to avoid rectangular/halo artifacts.
        diff = np.mean(np.abs(gen.astype(np.float32) - orig.astype(np.float32)), axis=2)
        diff_mask = np.where(diff > args.diff_threshold, 255, 0).astype(np.uint8)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        diff_mask = cv2.GaussianBlur(diff_mask, (9, 9), 0)
        alpha *= (diff_mask.astype(np.float32) / 255.0)

        protect_h = int(orig.shape[0] * args.protect_bottom_ratio)
        if protect_h > 0:
            alpha[orig.shape[0] - protect_h :, :] = 0.0

        if prev_alpha is None:
            prev_alpha = alpha
        else:
            alpha = args.temporal_alpha * prev_alpha + (1.0 - args.temporal_alpha) * alpha
            prev_alpha = alpha

        gen_matched = color_match_in_mask(orig, gen, alpha)
        a3 = np.repeat(alpha[:, :, None], 3, axis=2)
        out = (a3 * gen_matched.astype(np.float32) + (1.0 - a3) * orig.astype(np.float32)).astype(np.uint8)
        writer.write(out)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    writer.release()
    orig_cap.release()
    gen_fetcher.close()
    if mask_fetcher is not None:
        mask_fetcher.close()

    if args.keep_audio:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_out,
                "-i",
                args.original,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                args.out,
            ],
            check=True,
        )
        os.remove(tmp_out)


if __name__ == "__main__":
    run()
