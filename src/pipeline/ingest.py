from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    total_frames: int
    duration_sec: float


@dataclass
class DecodedVideo:
    info: VideoInfo
    frames: List[np.ndarray]


def decode_video(video_path: str, max_frames: Optional[int] = None) -> DecodedVideo:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: List[np.ndarray] = []
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")

    if total_frames <= 0:
        total_frames = len(frames)

    info = VideoInfo(
        fps=fps,
        width=width if width > 0 else int(frames[0].shape[1]),
        height=height if height > 0 else int(frames[0].shape[0]),
        total_frames=total_frames,
        duration_sec=float(total_frames) / fps,
    )
    return DecodedVideo(info=info, frames=frames)


def sample_keyframe_indices(
    total_frames: int,
    fps: float,
    keyframes: int,
    first_seconds: int,
) -> List[int]:
    if total_frames <= 0:
        return []

    max_frame = min(total_frames - 1, int(first_seconds * fps) - 1)
    if max_frame < 0:
        max_frame = total_frames - 1

    if keyframes <= 1:
        return [max_frame // 2]

    indices: List[int] = []
    for i in range(keyframes):
        t = i / max(1, keyframes - 1)
        idx = int(round(t * max_frame))
        indices.append(max(0, min(total_frames - 1, idx)))

    return sorted(set(indices))
