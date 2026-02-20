from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.config.schema import GeminiConfig
from src.pipeline.detect_track import TrackingResult, rank_tracks_for_fallback


@dataclass
class GeminiSelectionResult:
    selected_track_id: int
    used_gemini: bool
    reason: str
    candidate_track_ids: List[int]


def _extract_json(text: str) -> Optional[Dict[str, object]]:
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


def _render_candidate_overlay(
    frame: np.ndarray,
    frame_idx: int,
    track_ids: List[int],
    tracking_result: TrackingResult,
) -> np.ndarray:
    canvas = frame.copy()
    for track_id in track_ids:
        track = tracking_result.tracks.get(track_id)
        if track is None:
            continue
        bbox = track.frame_boxes.get(frame_idx)
        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in (bbox.x1, bbox.y1, bbox.x2, bbox.y2)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"ID {track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _img_to_b64(image: np.ndarray) -> str:
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _call_gemini(
    cfg: GeminiConfig,
    api_key: str,
    target_description: str,
    candidates: List[int],
    image_b64: str,
) -> int:
    prompt = (
        "You are selecting one person track ID from a video frame. "
        f"Target description: '{target_description}'. "
        f"Candidate IDs: {candidates}. "
        "Return strict JSON only with schema: "
        '{"track_id": <int>, "reason": "<short>"}. '
        "Choose exactly one from the candidate IDs."
    )

    url = cfg.api_url_template.format(model=cfg.model_name, api_key=api_key)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
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

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Gemini HTTPError: {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Gemini URLError") from exc

    candidates_resp = body.get("candidates", [])
    text_parts: List[str] = []
    for cand in candidates_resp:
        content = cand.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                text_parts.append(text)

    if not text_parts:
        raise RuntimeError("Gemini returned no text")

    parsed = _extract_json("\n".join(text_parts))
    if not parsed or "track_id" not in parsed:
        raise RuntimeError("Gemini response missing track_id")

    selected = int(parsed["track_id"])
    if selected not in candidates:
        raise RuntimeError("Gemini selected non-candidate track_id")
    return selected


def choose_target_track(
    frames: List[np.ndarray],
    tracking_result: TrackingResult,
    frame_width: int,
    frame_height: int,
    keyframe_indices: List[int],
    target_description: str,
    cfg: GeminiConfig,
    api_key_env_var: str,
) -> GeminiSelectionResult:
    ranked = rank_tracks_for_fallback(tracking_result, frame_width, frame_height)
    if not ranked:
        raise RuntimeError("No candidate tracks found")

    candidates = ranked[: cfg.max_candidates]

    if not cfg.enabled:
        return GeminiSelectionResult(
            selected_track_id=candidates[0],
            used_gemini=False,
            reason="gemini_disabled",
            candidate_track_ids=candidates,
        )

    api_key = os.getenv(api_key_env_var)
    if not api_key:
        return GeminiSelectionResult(
            selected_track_id=candidates[0],
            used_gemini=False,
            reason="missing_api_key",
            candidate_track_ids=candidates,
        )

    for idx in keyframe_indices:
        if idx < 0 or idx >= len(frames):
            continue
        overlay = _render_candidate_overlay(frames[idx], idx, candidates, tracking_result)
        image_b64 = _img_to_b64(overlay)
        try:
            selected = _call_gemini(cfg, api_key, target_description, candidates, image_b64)
            return GeminiSelectionResult(
                selected_track_id=selected,
                used_gemini=True,
                reason="gemini_success",
                candidate_track_ids=candidates,
            )
        except Exception:
            continue

    return GeminiSelectionResult(
        selected_track_id=candidates[0],
        used_gemini=False,
        reason="gemini_failed_fallback",
        candidate_track_ids=candidates,
    )
