import json
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


class GeminiTrackSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_description": ("STRING", {"multiline": True}),
                "candidates_json": ("STRING", {"multiline": True}),
                "model_name": ("STRING", {"default": "gemini-2.5-pro"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("selected_track_id", "raw_response")
    FUNCTION = "run"
    CATEGORY = "Wan/Gemini"

    def run(self, target_description: str, candidates_json: str, model_name: str, temperature: float):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        try:
            candidates: List[Dict[str, Any]] = json.loads(candidates_json)
        except Exception as exc:
            raise RuntimeError(f"Invalid candidates_json: {exc}") from exc

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        prompt = (
            "You are selecting the best target track for character replacement.\n"
            "Return ONLY JSON object with keys: track_id (int), confidence (0-1), reason (short).\n\n"
            f"Target description:\n{target_description}\n\n"
            f"Candidate tracks JSON:\n{json.dumps(candidates, ensure_ascii=False)}\n"
        )

        resp = model.generate_content(
            prompt,
            generation_config={"temperature": float(temperature)},
        )
        text = (resp.text or "").strip()
        parsed = _extract_json(text)

        track_id = int(parsed["track_id"])
        return (track_id, text)


NODE_CLASS_MAPPINGS = {
    "GeminiTrackSelector": GeminiTrackSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiTrackSelector": "Gemini Track Selector (2.5 Pro)",
}
