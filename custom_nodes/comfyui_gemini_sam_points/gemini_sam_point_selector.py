import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class GeminiSAMPointSelector:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple[Any, ...]]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOX",),
                "model_name": ("STRING", {"default": "gemini-3-pro", "multiline": False}),
                "api_key_env_var": ("STRING", {"default": "GEMINI_API_KEY", "multiline": False}),
                "target_description": (
                    "STRING",
                    {"default": "main character", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_coords", "negative_coords")
    FUNCTION = "select_points"
    CATEGORY = "WanAnimate/Gemini"

    def select_points(
        self,
        image: Any,
        bboxes: Any,
        model_name: str,
        api_key_env_var: str,
        target_description: str,
    ) -> Tuple[str, str]:
        frame = self._extract_first_frame(image)
        width, height = frame.size
        bbox = self._extract_primary_bbox(bboxes, width, height)
        fallback_positive, fallback_negative = self._fallback_points(bbox, width, height)

        api_key = os.getenv(api_key_env_var.strip())
        if not api_key:
            return json.dumps(fallback_positive), json.dumps(fallback_negative)

        try:
            image_b64 = self._image_to_b64(frame)
            positive, negative = self._call_gemini(
                api_key=api_key,
                model_name=model_name.strip(),
                image_b64=image_b64,
                width=width,
                height=height,
                bbox=bbox,
                target_description=target_description.strip(),
            )
        except Exception:
            positive, negative = fallback_positive, fallback_negative

        positive = self._sanitize_points(positive, width, height)
        negative = self._sanitize_points(negative, width, height)

        if not positive:
            positive = fallback_positive
        if not negative:
            negative = fallback_negative

        return json.dumps(positive), json.dumps(negative)

    def _extract_first_frame(self, image: Any) -> Image.Image:
        try:
            tensor = image[0].cpu().numpy()
        except Exception as exc:
            raise ValueError("IMAGE input is not a valid ComfyUI tensor batch.") from exc
        tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(tensor)

    def _extract_primary_bbox(self, bboxes: Any, width: int, height: int) -> Dict[str, int]:
        default_box = {
            "x1": int(width * 0.2),
            "y1": int(height * 0.1),
            "x2": int(width * 0.8),
            "y2": int(height * 0.95),
        }
        if bboxes is None:
            return default_box

        candidates = self._flatten_bbox_candidates(bboxes)
        best = None
        best_area = -1

        for item in candidates:
            parsed = self._parse_bbox(item)
            if not parsed:
                continue
            x1, y1, x2, y2 = parsed
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        if best is None:
            return default_box
        return best

    def _flatten_bbox_candidates(self, bboxes: Any) -> List[Any]:
        if isinstance(bboxes, list):
            out: List[Any] = []
            for item in bboxes:
                if isinstance(item, list):
                    out.extend(item)
                else:
                    out.append(item)
            return out
        return [bboxes]

    def _parse_bbox(self, item: Any) -> Optional[Tuple[int, int, int, int]]:
        if isinstance(item, dict):
            if all(k in item for k in ("x1", "y1", "x2", "y2")):
                return (
                    int(item["x1"]),
                    int(item["y1"]),
                    int(item["x2"]),
                    int(item["y2"]),
                )
            if all(k in item for k in ("startX", "startY", "endX", "endY")):
                return (
                    int(item["startX"]),
                    int(item["startY"]),
                    int(item["endX"]),
                    int(item["endY"]),
                )
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            return (int(item[0]), int(item[1]), int(item[2]), int(item[3]))
        return None

    def _fallback_points(
        self, bbox: Dict[str, int], width: int, height: int
    ) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        positive = [
            {"x": cx, "y": cy},
            {"x": cx, "y": int(y1 + 0.3 * (y2 - y1))},
            {"x": int(x1 + 0.35 * (x2 - x1)), "y": int(y1 + 0.6 * (y2 - y1))},
            {"x": int(x1 + 0.65 * (x2 - x1)), "y": int(y1 + 0.6 * (y2 - y1))},
        ]
        margin = 20
        negative = [
            {"x": max(0, x1 - margin), "y": max(0, y1 - margin)},
            {"x": min(width - 1, x2 + margin), "y": max(0, y1 - margin)},
            {"x": max(0, x1 - margin), "y": min(height - 1, y2 + margin)},
            {"x": min(width - 1, x2 + margin), "y": min(height - 1, y2 + margin)},
        ]
        return positive, negative

    def _image_to_b64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_gemini(
        self,
        api_key: str,
        model_name: str,
        image_b64: str,
        width: int,
        height: int,
        bbox: Dict[str, int],
        target_description: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        prompt = (
            "You are selecting segmentation prompts for SAM. "
            f"Target object: {target_description}. "
            f"Image size: width={width}, height={height}. "
            "Use bbox as a hint only, not absolute truth: "
            f"x1={bbox['x1']}, y1={bbox['y1']}, x2={bbox['x2']}, y2={bbox['y2']}. "
            "Return strict JSON only, no markdown, with this schema: "
            '{"positive":[{"x":int,"y":int}],"negative":[{"x":int,"y":int}]}. '
            "Choose 4 positive points on the target and 4 negative points on nearby background."
        )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            f"?key={api_key}"
        )
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Gemini API HTTPError: {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Gemini API URLError") from exc

        parsed = json.loads(body)
        text = self._extract_text_response(parsed)
        point_json = self._extract_json(text)
        data = json.loads(point_json)
        return data.get("positive", []), data.get("negative", [])

    def _extract_text_response(self, response: Dict[str, Any]) -> str:
        candidates = response.get("candidates", [])
        for cand in candidates:
            content = cand.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                text = part.get("text")
                if text:
                    return text
        raise ValueError("Gemini response did not include text content.")

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON object found in Gemini response text.")

    def _sanitize_points(
        self, points: List[Dict[str, Any]], width: int, height: int
    ) -> List[Dict[str, int]]:
        clean: List[Dict[str, int]] = []
        for p in points:
            if not isinstance(p, dict):
                continue
            if "x" not in p or "y" not in p:
                continue
            x = int(max(0, min(width - 1, round(float(p["x"])))))
            y = int(max(0, min(height - 1, round(float(p["y"])))))
            clean.append({"x": x, "y": y})
        return clean


NODE_CLASS_MAPPINGS = {
    "GeminiSAMPointSelector": GeminiSAMPointSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiSAMPointSelector": "Gemini SAM Point Selector",
}
