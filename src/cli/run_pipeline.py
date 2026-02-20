from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.config.schema import PipelineConfig, load_config
from src.pipeline.blend import temporal_smooth_frame
from src.pipeline.detect_track import TrackingResult, run_detection_tracking
from src.pipeline.face_swap import InsightFaceSwapper
from src.pipeline.full_character import FullCharacterReplacer
from src.pipeline.gemini_select import GeminiSelectionResult, choose_target_track
from src.pipeline.ingest import decode_video, sample_keyframe_indices
from src.pipeline.render import finalize_output, write_silent_video


def _gpu_metadata() -> Dict[str, object]:
    data: Dict[str, object] = {}
    try:
        import onnxruntime as ort

        data["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        data["onnxruntime_providers"] = []

    data["cuda_visible_devices"] = os.getenv("CUDA_VISIBLE_DEVICES", "")
    return data


def _swap_stage1(
    frames: List[np.ndarray],
    tracking: TrackingResult,
    target_track_id: int,
    swapper: InsightFaceSwapper,
    temporal_alpha: float,
) -> tuple[List[np.ndarray], int]:
    out_frames: List[np.ndarray] = []
    prev_swapped = None
    swaps = 0

    target_track = tracking.tracks.get(target_track_id)
    if target_track is None:
        return frames, 0

    for idx, frame in enumerate(tqdm(frames, desc="stage1_face_swap", unit="frame")):
        bbox = target_track.frame_boxes.get(idx)
        if bbox is None:
            out_frames.append(frame)
            continue

        swap_result = swapper.swap_on_bbox(frame, bbox)
        frame_out = swap_result.frame
        if swap_result.swapped:
            swaps += 1
            if prev_swapped is not None:
                frame_out = temporal_smooth_frame(prev_swapped, frame_out, temporal_alpha)
            prev_swapped = frame_out

        out_frames.append(frame_out)

    return out_frames, swaps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H100 Python-only character replacement pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--ref", required=True, help="Reference character image path")
    parser.add_argument("--out", required=True, help="Output video path")
    parser.add_argument("--target", default="main character", help="Target description")
    parser.add_argument("--config", default=None, help="Optional YAML config")
    parser.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mode", default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report-json", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)

    config: PipelineConfig = load_config(args.config)
    config.detector.device = args.device

    t0 = time.perf_counter()

    decoded = decode_video(args.video, max_frames=args.max_frames)
    frames = decoded.frames

    t1 = time.perf_counter()
    tracking = run_detection_tracking(frames, config.detector, config.tracker)
    t2 = time.perf_counter()

    keyframes = sample_keyframe_indices(
        total_frames=len(frames),
        fps=decoded.info.fps,
        keyframes=config.gemini.keyframes,
        first_seconds=config.gemini.first_seconds,
    )

    selection: GeminiSelectionResult = choose_target_track(
        frames=frames,
        tracking_result=tracking,
        frame_width=decoded.info.width,
        frame_height=decoded.info.height,
        keyframe_indices=keyframes,
        target_description=args.target,
        cfg=config.gemini,
        api_key_env_var=args.gemini_api_key_env,
    )

    t3 = time.perf_counter()

    swapper = InsightFaceSwapper(config.swap, device=args.device)
    swapper.load(args.ref)
    stage1_frames, swapped_count = _swap_stage1(
        frames=frames,
        tracking=tracking,
        target_track_id=selection.selected_track_id,
        swapper=swapper,
        temporal_alpha=config.blend.temporal_alpha,
    )
    t4 = time.perf_counter()

    stage2_events = []
    output_frames = stage1_frames
    if args.mode == "stage2":
        stage2 = FullCharacterReplacer(config.fallback)
        output_frames, stage2_events = stage2.apply(stage1_frames, tracking, selection.selected_track_id)

    t5 = time.perf_counter()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="char_replace_") as tmp_dir:
        silent_path = str(Path(tmp_dir) / "silent.mp4")
        write_silent_video(output_frames, decoded.info.fps, silent_path)
        finalize_output(args.video, silent_path, str(out_path), config.render)

    t6 = time.perf_counter()

    report_path = Path(args.report_json) if args.report_json else out_path.with_suffix(".report.json")
    report = {
        "input": {
            "video": str(Path(args.video).resolve()),
            "ref": str(Path(args.ref).resolve()),
            "target": args.target,
            "mode": args.mode,
            "max_frames": args.max_frames,
            "seed": seed,
        },
        "video": {
            "fps": decoded.info.fps,
            "width": decoded.info.width,
            "height": decoded.info.height,
            "decoded_frames": len(frames),
            "source_total_frames": decoded.info.total_frames,
            "duration_sec": decoded.info.duration_sec,
        },
        "selection": {
            "selected_track_id": selection.selected_track_id,
            "used_gemini": selection.used_gemini,
            "reason": selection.reason,
            "candidates": selection.candidate_track_ids,
            "keyframes": keyframes,
        },
        "stage1": {
            "swapped_frames": swapped_count,
            "track_hits": tracking.tracks.get(selection.selected_track_id).hits
            if selection.selected_track_id in tracking.tracks
            else 0,
        },
        "stage2": {
            "enabled": args.mode == "stage2",
            "fallback_events": [event.__dict__ for event in stage2_events],
        },
        "timing_sec": {
            "decode": t1 - t0,
            "detect_track": t2 - t1,
            "gemini_select": t3 - t2,
            "face_swap": t4 - t3,
            "stage2": t5 - t4,
            "render": t6 - t5,
            "total": t6 - t0,
        },
        "runtime": _gpu_metadata(),
        "output_video": str(out_path.resolve()),
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved output video: {out_path.resolve()}")
    print(f"Saved run report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
