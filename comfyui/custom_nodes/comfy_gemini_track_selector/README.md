# Gemini Track Selector (ComfyUI Node)

Returns `selected_track_id` from a JSON list of candidate tracks using Gemini.

## Inputs
- `target_description` (STRING)
- `candidates_json` (STRING): JSON array of objects containing at least `track_id`
- `model_name` (STRING): default `gemini-2.5-pro`
- `temperature` (FLOAT)

## Outputs
- `selected_track_id` (INT)
- `raw_response` (STRING)

## Required env var
- `GEMINI_API_KEY`

## Example `candidates_json`
```json
[
  {"track_id": 1, "summary": "man in blue suit at center"},
  {"track_id": 2, "summary": "woman in white shirt at left"}
]
```
