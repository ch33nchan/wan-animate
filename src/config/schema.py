from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class DetectorConfig(BaseModel):
    model: str = "yolov8n.pt"
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.5, ge=0.0, le=1.0)
    person_class_id: int = 0
    device: str = "cuda"


class TrackerConfig(BaseModel):
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_missed: int = Field(default=20, ge=1)
    smoothing_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    min_hits: int = Field(default=2, ge=1)


class GeminiConfig(BaseModel):
    enabled: bool = True
    model_name: str = "gemini-2.5-flash"
    timeout_sec: int = Field(default=20, ge=1)
    keyframes: int = Field(default=3, ge=1)
    first_seconds: int = Field(default=8, ge=1)
    max_candidates: int = Field(default=6, ge=1)
    api_url_template: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    )


class SwapConfig(BaseModel):
    face_det_size: int = Field(default=640, ge=256)
    inswapper_model_path: Optional[str] = None
    enhancer: Literal["none", "gfpgan", "codeformer"] = "none"


class BlendConfig(BaseModel):
    temporal_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    feather_kernel: int = Field(default=11, ge=1)
    use_seamless_clone: bool = True


class RenderConfig(BaseModel):
    ffmpeg_bin: str = "ffmpeg"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = "medium"
    pix_fmt: str = "yuv420p"


class FallbackConfig(BaseModel):
    stage2_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    gemini: GeminiConfig = GeminiConfig()
    swap: SwapConfig = SwapConfig()
    blend: BlendConfig = BlendConfig()
    render: RenderConfig = RenderConfig()
    fallback: FallbackConfig = FallbackConfig()


def load_config(config_path: Optional[str]) -> PipelineConfig:
    if not config_path:
        return PipelineConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping.")

    try:
        return PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid config: {exc}") from exc
