from pathlib import Path

import pytest

from src.config.schema import PipelineConfig, load_config


def test_default_config_loads() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.detector.model == "yolov8n.pt"
    assert cfg.gemini.keyframes == 3


def test_yaml_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
tracker:
  iou_threshold: 0.42
gemini:
  enabled: false
""".strip()
    )

    cfg = load_config(str(config_path))
    assert cfg.tracker.iou_threshold == pytest.approx(0.42)
    assert cfg.gemini.enabled is False
