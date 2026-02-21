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


def run() -> None:
    parser = argparse.ArgumentParser(description="Blend generated target back into original video using mask")
    parser.add_argument("--original", required=True, help="Original source video")
    parser.add_argument("--generated", required=True, help="Generated WAN output video")
    parser.add_argument("--mask", required=True, help="Mask video from preprocess (src_mask.mp4)")
    parser.add_argument("--out", required=True, help="Final composited output video")
    parser.add_argument("--edge_blur", type=int, default=21)
    parser.add_argument("--mask_dilate", type=int, default=5)
    parser.add_argument("--mask_erode", type=int, default=1)
    parser.add_argument("--mask_threshold", type=int, default=96)
    parser.add_argument("--min_component_area_ratio", type=float, default=0.01)
    parser.add_argument("--max_row_coverage", type=float, default=0.65)
    parser.add_argument("--alpha_scale", type=float, default=0.95)
    parser.add_argument("--temporal_alpha", type=float, default=0.80)
    parser.add_argument("--protect_bottom_ratio", type=float, default=0.16)
    parser.add_argument("--keep_audio", action="store_true")
    args = parser.parse_args()

    orig_info = get_stream_info(args.original)
    gen_info = get_stream_info(args.generated)
    mask_info = get_stream_info(args.mask)

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
    mask_fetcher = SequentialFrameFetcher(args.mask)

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
        mask_idx = clamp(int(round(t * mask_info.fps)), 0, max(mask_info.frame_count - 1, 0))

        gen = gen_fetcher.get(gen_idx)
        msk = mask_fetcher.get(mask_idx)

        if gen is None or msk is None:
            writer.write(orig)
            frame_idx += 1
            pbar.update(1)
            continue

        if gen.shape[:2] != orig.shape[:2]:
            gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC)
        if msk.shape[:2] != orig.shape[:2]:
            msk = cv2.resize(msk, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

        msk_gray = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY) if msk.ndim == 3 else msk
        _, m = cv2.threshold(msk_gray, args.mask_threshold, 255, cv2.THRESH_BINARY)
        min_area = int(orig.shape[0] * orig.shape[1] * args.min_component_area_ratio)
        m = keep_largest_component(m, min_area=min_area)

        max_cov = max(0.05, min(1.0, args.max_row_coverage))
        row_coverage = (m > 0).mean(axis=1)
        m[row_coverage > max_cov, :] = 0

        if args.mask_erode > 1:
            m = cv2.erode(m, kernel_e, iterations=1)
        if args.mask_dilate > 1:
            m = cv2.dilate(m, kernel_d, iterations=1)

        m = cv2.GaussianBlur(m, (k_blur, k_blur), 0)
        alpha = (m.astype(np.float32) / 255.0) * args.alpha_scale

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
