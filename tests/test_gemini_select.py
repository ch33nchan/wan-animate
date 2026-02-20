from src.config.schema import GeminiConfig
from src.pipeline.detect_track import BBox, TrackState, TrackingResult
from src.pipeline.gemini_select import _extract_json, choose_target_track


def test_extract_json_from_wrapped_text() -> None:
    text = "result:\n{\"track_id\": 3, \"reason\": \"closest\"}\nthanks"
    parsed = _extract_json(text)
    assert parsed is not None
    assert parsed["track_id"] == 3


def test_choose_target_track_fallback_without_key() -> None:
    frames = [__import__("numpy").zeros((100, 100, 3), dtype=__import__("numpy").uint8)]
    track = TrackState(
        track_id=1,
        bbox=BBox(10, 10, 60, 90),
        smoothed_score=0.9,
        hits=3,
        missed=0,
        frame_boxes={0: BBox(10, 10, 60, 90)},
        total_area=4000,
    )
    tracking = TrackingResult(tracks={1: track}, frame_to_track_ids={0: [1]})

    result = choose_target_track(
        frames=frames,
        tracking_result=tracking,
        frame_width=100,
        frame_height=100,
        keyframe_indices=[0],
        target_description="main character",
        cfg=GeminiConfig(enabled=True),
        api_key_env_var="__MISSING_KEY__",
    )

    assert result.selected_track_id == 1
    assert result.used_gemini is False
