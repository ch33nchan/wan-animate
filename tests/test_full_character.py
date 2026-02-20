import numpy as np

from src.config.schema import FallbackConfig, GeminiConfig, Stage2Config
from src.pipeline.full_character import FullCharacterReplacer
from src.pipeline.detect_track import TrackingResult


def test_stage2_returns_original_when_target_missing() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    tracking = TrackingResult(tracks={}, frame_to_track_ids={0: [], 1: [], 2: []})

    replacer = FullCharacterReplacer(
        stage2_cfg=Stage2Config(),
        fallback_cfg=FallbackConfig(),
        gemini_cfg=GeminiConfig(enabled=False),
        temporal_alpha=0.7,
        device="cpu",
    )

    out_frames, events, replaced = replacer.apply(
        stage1_frames=frames,
        tracking_result=tracking,
        target_track_id=1,
        ref_image_path="/tmp/nonexistent.png",
        target_description="main character",
        api_key_env_var="GEMINI_API_KEY",
    )

    assert replaced == 0
    assert len(out_frames) == len(frames)
    assert len(events) == len(frames)
    assert all(event.reason == "missing_target_track" for event in events)
