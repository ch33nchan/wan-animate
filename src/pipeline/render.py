from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.config.schema import RenderConfig


def write_silent_video(frames: List[np.ndarray], fps: float, output_path: str) -> None:
    if not frames:
        raise ValueError("No frames to render")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    for frame in frames:
        writer.write(frame)
    writer.release()


def extract_audio(input_video: str, audio_out: str, cfg: RenderConfig) -> bool:
    cmd = [
        cfg.ffmpeg_bin,
        "-y",
        "-i",
        input_video,
        "-vn",
        "-acodec",
        "copy",
        audio_out,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0


def mux_audio(silent_video: str, audio_path: str, output_video: str, cfg: RenderConfig) -> None:
    cmd = [
        cfg.ffmpeg_bin,
        "-y",
        "-i",
        silent_video,
        "-i",
        audio_path,
        "-c:v",
        cfg.video_codec,
        "-preset",
        cfg.preset,
        "-crf",
        str(cfg.crf),
        "-pix_fmt",
        cfg.pix_fmt,
        "-c:a",
        cfg.audio_codec,
        "-shortest",
        output_video,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {proc.stderr[-2000:]}")


def finalize_output(
    input_video: str,
    silent_video: str,
    output_video: str,
    cfg: RenderConfig,
) -> None:
    tmp_audio = str(Path(output_video).with_suffix(".audio.tmp"))
    has_audio = extract_audio(input_video=input_video, audio_out=tmp_audio, cfg=cfg)

    if has_audio:
        mux_audio(silent_video=silent_video, audio_path=tmp_audio, output_video=output_video, cfg=cfg)
        Path(tmp_audio).unlink(missing_ok=True)
    else:
        shutil.move(silent_video, output_video)
