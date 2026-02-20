from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.config.schema import FallbackConfig
from src.pipeline.detect_track import TrackingResult


@dataclass
class Stage2Event:
    frame_index: int
    reason: str
    confidence: float


class FullCharacterReplacer:
    def __init__(self, cfg: FallbackConfig) -> None:
        self.cfg = cfg

    def _estimate_confidence(
        self,
        frame: np.ndarray,
        has_track: bool,
    ) -> float:
        if not has_track:
            return 0.0
        sharpness = float(np.var(frame)) / 5000.0
        return max(0.0, min(1.0, sharpness))

    def apply(
        self,
        stage1_frames: List[np.ndarray],
        tracking_result: TrackingResult,
        target_track_id: int,
    ) -> Tuple[List[np.ndarray], List[Stage2Event]]:
        output: List[np.ndarray] = []
        events: List[Stage2Event] = []

        track = tracking_result.tracks.get(target_track_id)
        for idx, frame in enumerate(stage1_frames):
            has_track = bool(track and idx in track.frame_boxes)
            confidence = self._estimate_confidence(frame, has_track)

            if confidence < self.cfg.stage2_confidence_threshold:
                output.append(frame)
                events.append(
                    Stage2Event(
                        frame_index=idx,
                        reason="stage2_confidence_below_threshold",
                        confidence=confidence,
                    )
                )
                continue

            # Stage 2 hook: currently returns stage1 frame to guarantee continuity.
            output.append(frame)

        return output, events
