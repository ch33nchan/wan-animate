#!/usr/bin/env python3
"""
Headless ComfyUI runner for Wan2.2Animate.gemini3pro workflow.

Fetches live node schemas from /object_info to correctly map widget values
to input names, resolves SetNode/GetNode virtual links, patches inputs,
converts to API format, submits, polls, copies output.

Usage (on charizard with ComfyUI already running):
    python3 run_headless.py \
        --video /mnt/data1/srini/ComfyUI/input/13.mp4 \
        --ref /mnt/data1/srini/ComfyUI/input/char_ref.png \
        --prompt "wearing grey top and blue jeans" \
        --target "main guy wearing suit" \
        --out /mnt/data1/srini/outputs/result.mp4
"""

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# UI-only node types the backend never sees
_VIRTUAL_TYPES = {"SetNode", "GetNode", "Note", "MarkdownNote", "PrimitiveNode", "Reroute"}


# ---------------------------------------------------------------------------
# Fetch live node schemas from ComfyUI
# ---------------------------------------------------------------------------

def fetch_object_info(host: str, port: int) -> Dict[str, Any]:
    url = f"http://{host}:{port}/object_info"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Could not fetch /object_info: {exc}") from exc


def build_widget_order(object_info: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    For each node class, return the ordered list of widget input names
    (primitive types: INT, FLOAT, STRING, BOOLEAN, COMBO/list).
    This matches the order ComfyUI assigns widget_values positionally.
    """
    table: Dict[str, List[str]] = {}
    for class_type, info in object_info.items():
        widget_names: List[str] = []
        for section in ("required", "optional"):
            for name, spec in info.get("input", {}).get(section, {}).items():
                if not isinstance(spec, list) or not spec:
                    continue
                t = spec[0]
                if isinstance(t, list):  # COMBO dropdown
                    widget_names.append(name)
                elif isinstance(t, str) and t.upper() in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                    widget_names.append(name)
        table[class_type] = widget_names
    return table


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def nodes_by_id(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(n["id"]): n for n in graph["nodes"]}


def links_by_id(graph: Dict[str, Any]) -> Dict[int, List]:
    return {lnk[0]: lnk for lnk in graph.get("links", [])}


# ---------------------------------------------------------------------------
# Patch widget values (auto-detect node IDs by type)
# ---------------------------------------------------------------------------

def find_nodes(nmap: Dict[str, Dict[str, Any]], ntype: str) -> List[str]:
    return [nid for nid, node in nmap.items() if node.get("type") == ntype]


def patch_nodes(
    nmap: Dict[str, Dict[str, Any]],
    video_filename: str,
    ref_filename: str,
    prompt: str,
    target_description: str,
    output_prefix: str,
    seed: int,
) -> None:
    video_nodes = find_nodes(nmap, "VHS_LoadVideo")
    image_nodes = find_nodes(nmap, "LoadImage")
    text_nodes  = find_nodes(nmap, "WanVideoTextEncodeCached")
    sam_nodes   = find_nodes(nmap, "GeminiSAMPointSelector")
    output_nodes = find_nodes(nmap, "VHS_VideoCombine")
    sampler_nodes = find_nodes(nmap, "WanVideoSampler")
    lora_nodes = find_nodes(nmap, "WanVideoLoraSelectMulti")
    detect_nodes = find_nodes(nmap, "OnnxDetectionModelLoader")

    if not video_nodes:
        raise RuntimeError("VHS_LoadVideo node not found")
    if not image_nodes:
        raise RuntimeError("LoadImage node not found")
    if not text_nodes:
        raise RuntimeError("WanVideoTextEncodeCached node not found")
    if not output_nodes:
        raise RuntimeError("VHS_VideoCombine node not found")
    if not sampler_nodes:
        raise RuntimeError("WanVideoSampler node not found")

    vhs_wv = nmap[video_nodes[0]]["widgets_values"]
    if isinstance(vhs_wv, dict):
        vhs_wv["video"] = video_filename
    elif isinstance(vhs_wv, list) and vhs_wv:
        vhs_wv[0] = video_filename

    for nid in image_nodes:
        wv = nmap[nid].get("widgets_values", [])
        if isinstance(wv, list) and wv:
            wv[0] = ref_filename
        elif isinstance(wv, dict):
            wv["image"] = ref_filename

    txt_wv = nmap[text_nodes[0]].get("widgets_values", [])
    if isinstance(txt_wv, list) and len(txt_wv) >= 3:
        txt_wv[2] = prompt

    for nid in sam_nodes:
        wv = nmap[nid].get("widgets_values", [])
        if isinstance(wv, list):
            while len(wv) < 3:
                wv.append("")
            wv[2] = target_description

    out_wv = nmap[output_nodes[0]].get("widgets_values", [])
    if isinstance(out_wv, dict):
        out_wv["filename_prefix"] = output_prefix
    elif isinstance(out_wv, list) and len(out_wv) >= 3:
        out_wv[2] = output_prefix

    samp_wv = nmap[sampler_nodes[0]].get("widgets_values", [])
    if isinstance(samp_wv, list) and len(samp_wv) >= 4:
        samp_wv[3] = seed

    if lora_nodes:
        lora_wv = nmap[lora_nodes[0]].get("widgets_values", [])
        if isinstance(lora_wv, list):
            for i in range(5):
                idx = i * 2
                if idx < len(lora_wv):
                    lora_wv[idx] = "none"

    # Detection models: force known good filenames
    for nid in detect_nodes:
        wv = nmap[nid].get("widgets_values", [])
        if isinstance(wv, list):
            if len(wv) >= 1:
                wv[0] = "vitpose-l-wholebody.onnx"
            if len(wv) >= 2:
                wv[1] = "yolov10m.onnx"
            if len(wv) >= 3:
                wv[2] = "CUDAExecutionProvider"


# ---------------------------------------------------------------------------
# SetNode / GetNode resolution
# ---------------------------------------------------------------------------

def build_named_store(
    nmap: Dict[str, Dict[str, Any]],
    lmap: Dict[int, List],
) -> Dict[str, Tuple[str, int]]:
    store: Dict[str, Tuple[str, int]] = {}
    for nid, node in nmap.items():
        if node.get("type") != "SetNode":
            continue
        wv = node.get("widgets_values", [])
        name = wv[0] if isinstance(wv, list) else next(iter(wv.values()))
        for inp in node.get("inputs", []):
            lid = inp.get("link")
            if lid is None:
                continue
            lnk = lmap.get(lid)
            if lnk:
                store[name] = (str(lnk[1]), lnk[2])
    return store


def make_link_resolver(
    nmap: Dict[str, Dict[str, Any]],
    lmap: Dict[int, List],
    store: Dict[str, Tuple[str, int]],
):
    get_resolved: Dict[str, Tuple[str, int]] = {}
    for nid, node in nmap.items():
        if node.get("type") != "GetNode":
            continue
        wv = node.get("widgets_values", [])
        name = wv[0] if isinstance(wv, list) else next(iter(wv.values()))
        if name in store:
            get_resolved[nid] = store[name]

    def resolve(link_id: int, _visited: Optional[set] = None) -> Optional[Tuple[str, int]]:
        if _visited is None:
            _visited = set()
        if link_id in _visited:
            return None
        _visited.add(link_id)
        lnk = lmap.get(link_id)
        if lnk is None:
            return None
        src_id, src_slot = str(lnk[1]), lnk[2]
        if src_id in get_resolved:
            return get_resolved[src_id]
        src_node = nmap.get(src_id, {})
        if src_node.get("type") == "SetNode":
            for inp in src_node.get("inputs", []):
                lid2 = inp.get("link")
                if lid2 is not None:
                    return resolve(lid2, _visited)
            return None
        return (src_id, src_slot)

    return resolve


# ---------------------------------------------------------------------------
# Core conversion: UI graph -> ComfyUI API prompt dict
# ---------------------------------------------------------------------------

def graph_to_api(
    graph: Dict[str, Any],
    nmap: Dict[str, Dict[str, Any]],
    lmap: Dict[int, List],
    widget_order: Dict[str, List[str]],
) -> Dict[str, Any]:
    store = build_named_store(nmap, lmap)
    resolve_link = make_link_resolver(nmap, lmap, store)

    api: Dict[str, Any] = {}

    for nid, node in nmap.items():
        node_type = node.get("type", "")
        # Skip virtual nodes AND bypassed (muted) nodes (mode=4)
        if node_type in _VIRTUAL_TYPES or node.get("mode") == 4:
            continue

        inputs: Dict[str, Any] = {}

        wv = node.get("widgets_values", [])

        if isinstance(wv, dict):
            # VHS nodes store as dict; strip UI-only keys
            for k, v in wv.items():
                if k == "videopreview":
                    continue
                inputs[k] = v
        else:
            # Positional widget values.
            # Some widget inputs are "promoted" to link inputs in the UI graph:
            # they appear in node["inputs"] with a "widget" key AND have a link.
            # When promoted, their value slot in widgets_values is still present
            # but the link takes precedence — we still need to consume the slot
            # to keep positional alignment, but we don't emit it as a widget value
            # (the link assignment below will set it).
            #
            # Promoted-but-UNLINKED widgets (link==null) must still be emitted
            # as widget values from the slot.

            # Build set of widget names that are promoted AND currently linked
            promoted_and_linked: set = set()
            for inp in node.get("inputs", []):
                if "widget" in inp and inp.get("link") is not None:
                    promoted_and_linked.add(inp["widget"]["name"])

            schema_names = widget_order.get(node_type, [])
            val_idx = 0
            for name in schema_names:
                if val_idx >= len(wv):
                    break
                val = wv[val_idx]
                val_idx += 1
                if name in promoted_and_linked:
                    # Slot consumed for alignment; link will override below
                    continue
                inputs[name] = val

        # Connected inputs (links override / add)
        for inp in node.get("inputs", []):
            lid = inp.get("link")
            if lid is None:
                continue
            resolved = resolve_link(lid)
            if resolved is None:
                continue
            src_id, src_slot = resolved
            inp_name = inp.get("widget", {}).get("name") or inp["name"]
            inputs[inp_name] = [src_id, src_slot]

        api[nid] = {
            "class_type": node_type,
            "inputs": inputs,
            "_meta": {"title": node.get("title", node_type)},
        }

    return api


# ---------------------------------------------------------------------------
# Bypass nodes that trigger ComfyUI validator bugs
# ---------------------------------------------------------------------------

def bypass_problematic_nodes(api: Dict[str, Any]) -> Dict[str, Any]:
    """
    GetImageSizeAndCount node 180 causes ComfyUI's recursive validator to
    exceed Python's default recursion limit because it fans out to 8+ nodes.
    Fix: remove node 180 from the API and rewire all consumers directly.

    Node 180 slot mapping (from workflow):
      slot 0 = image passthrough  -> rewire to ["63", 0]  (VHS_LoadVideo)
      slot 1 = width              -> rewire to ["150", 0] (INTConstant Width)
      slot 2 = height             -> rewire to ["151", 0] (INTConstant Height)
      slot 3 = count              -> not used by any downstream node
    """
    BYPASS_NODE = "180"
    SLOT_MAP = {
        0: ["63", 0],    # image -> VHS_LoadVideo output
        1: ["150", 0],   # width -> INTConstant Width
        2: ["151", 0],   # height -> INTConstant Height
    }

    if BYPASS_NODE not in api:
        return api

    # Rewire all consumers
    for nid, entry in api.items():
        if nid == BYPASS_NODE:
            continue
        for k, v in list(entry.get("inputs", {}).items()):
            if isinstance(v, list) and len(v) == 2 and str(v[0]) == BYPASS_NODE:
                slot = v[1]
                replacement = SLOT_MAP.get(slot)
                if replacement:
                    entry["inputs"][k] = replacement

    # Remove the node itself
    if BYPASS_NODE in api:
        del api[BYPASS_NODE]

    # --- Prune broken links ---
    # Because we removed bypassed/muted nodes in graph_to_api(), some inputs
    # might still point to them. Delete those inputs to avoid validation errors.
    valid_ids = set(api.keys())
    for nid, entry in api.items():
        inputs = entry.get("inputs", {})
        keys_to_delete = []
        for k, v in inputs.items():
            # Link format: ["node_id", slot_index]
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                target_id = v[0]
                if target_id not in valid_ids:
                    # Target node was bypassed/removed -> delete this input connection
                    keys_to_delete.append(k)
        
        for k in keys_to_delete:
            del inputs[k]

    return api


# ---------------------------------------------------------------------------
# Schema-version fixups
#
# Some nodes in the workflow were saved with an older schema that had extra
# or differently-ordered widget fields. After the generic widget mapping runs,
# we detect and correct known mismatches by inspecting the actual values.
# ---------------------------------------------------------------------------

def fixup_schema_mismatches(
    api: Dict[str, Any],
    nmap: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    for nid, entry in api.items():
        cls = entry.get("class_type", "")
        inputs = entry.get("inputs", {})

        if cls == "WanVideoSampler":
            # Old schema: steps, cfg, min_cfg, seed, seed_mode, force_offload,
            #             scheduler, riflex_freq_index, riflex_apply_in_full,
            #             denoise_strength, denoise_start_step, denoise_end_step, use_teacache
            # New schema: steps, cfg, shift, seed, force_offload, scheduler,
            #             riflex_freq_index, denoise_strength, batched_cfg,
            #             rope_function, start_step, end_step, add_noise_to_samples
            #
            # Detect mismatch: force_offload should be bool, scheduler should be str
            if not isinstance(inputs.get("force_offload"), bool) or not isinstance(
                inputs.get("scheduler"), str
            ):
                # Re-map from the raw widget_values of the node
                wv = nmap[nid].get("widgets_values", [])
                # Old positional layout (14 values):
                # [steps, cfg, min_cfg, seed, seed_mode, force_offload, scheduler,
                #  riflex_freq_index, riflex_apply_in_full, denoise_strength,
                #  denoise_start_step, denoise_end_step, use_teacache, <extra>]
                if len(wv) >= 13:
                    inputs["steps"]             = wv[0]
                    inputs["cfg"]               = wv[1]
                    # wv[2] = min_cfg (dropped)
                    inputs["seed"]              = wv[3]
                    # wv[4] = seed_mode (dropped)
                    inputs["force_offload"]     = wv[5]
                    inputs["scheduler"]         = wv[6]
                    inputs["riflex_freq_index"] = wv[7]
                    # wv[8] = riflex_apply_in_full (dropped)
                    raw_denoise = wv[9]
                    inputs["denoise_strength"]  = raw_denoise if isinstance(raw_denoise, (int, float)) else 1.0
                    # wv[10] = denoise_start_step -> start_step
                    raw_start = wv[10]
                    inputs["start_step"]        = raw_start if isinstance(raw_start, int) else 0
                    # wv[11] = denoise_end_step -> end_step
                    raw_end = wv[11]
                    inputs["end_step"]          = raw_end if isinstance(raw_end, int) else -1
                    # wv[12] = use_teacache (dropped / handled by cache_args)
                    # New fields not in old schema — use safe defaults
                    inputs["batched_cfg"]         = False
                    inputs["rope_function"]        = "comfy"
                    inputs["add_noise_to_samples"] = False
                    inputs["shift"]                = float(wv[2]) if isinstance(wv[2], (int, float)) else 5.0

        elif cls == "WanVideoAnimateEmbeds":
            # Old schema had: width, height, num_frames, use_zero_padding,
            #                 frame_window_size, noise_aug_strength, latent_strength,
            #                 clip_embed_strength, enable_vae_encode
            # New schema:     force_offload, frame_window_size, colormatch,
            #                 pose_strength, face_strength, tiled_vae
            # width/height/num_frames are now link-only (no widget slot).
            # The generic code already handles this correctly via promoted_and_linked,
            # but if force_offload ended up as an int/str, fix it.
            if not isinstance(inputs.get("force_offload"), bool):
                inputs["force_offload"] = False
            inputs.setdefault("frame_window_size", 77)
            inputs.setdefault("colormatch", "disabled")
            inputs.setdefault("pose_strength", 1.0)
            inputs.setdefault("face_strength", 1.0)
            inputs.setdefault("tiled_vae", False)

        elif cls == "WanVideoLoraSelectMulti":
            # lora_0/lora_1 must be in the installed LoRA list.
            # If they're not found, ComfyUI rejects with "value_not_in_list".
            # We can't fix missing files, but we can ensure 'none' entries are correct.
            for i in range(5):
                k = f"lora_{i}"
                if inputs.get(k) == "none":
                    inputs[k] = "none"

        elif cls == "OnnxDetectionModelLoader":
            # Old widget order in workflow: [vitpose_model, yolo_model, onnx_device]
            # New schema order:             [vitpose_model, yolo_model, onnx_device]
            # These match — no fix needed. But if swapped, detect and fix.
            vp = inputs.get("vitpose_model", "")
            ym = inputs.get("yolo_model", "")
            if "yolo" in vp.lower() and "vitpose" in ym.lower():
                inputs["vitpose_model"], inputs["yolo_model"] = ym, vp

    return api


# ---------------------------------------------------------------------------
# ComfyUI HTTP helpers
# ---------------------------------------------------------------------------

def copy_inputs(video_src: Path, ref_src: Path, comfyui_input: Path) -> Tuple[str, str]:
    comfyui_input.mkdir(parents=True, exist_ok=True)
    for src in (video_src, ref_src):
        dst = comfyui_input / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
    return video_src.name, ref_src.name


def submit_prompt(host: str, port: int, api_prompt: Dict[str, Any]) -> str:
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": api_prompt, "client_id": client_id}).encode()
    req = urllib.request.Request(
        url=f"http://{host}:{port}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        raise RuntimeError(f"ComfyUI /prompt {exc.code}: {detail}") from exc
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id in response: {body}")
    return prompt_id


def poll_until_done(
    host: str,
    port: int,
    prompt_id: str,
    poll_interval: float = 5.0,
    timeout: float = 3600.0,
) -> Dict[str, Any]:
    url = f"http://{host}:{port}/history/{prompt_id}"
    deadline = time.time() + timeout
    print(f"Polling {prompt_id} ...", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                history = json.loads(resp.read())
        except Exception:
            time.sleep(poll_interval)
            continue
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(f"Workflow error: {status.get('messages', [])}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out after {timeout}s")


def find_output_video(
    entry: Dict[str, Any],
    comfyui_output: Path,
    output_prefix: str,
) -> Optional[Path]:
    for node_out in entry.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []):
                fname = item.get("filename", "")
                subfolder = item.get("subfolder", "")
                ftype = item.get("type", "output")
                base = comfyui_output if ftype == "output" else comfyui_output.parent / "temp"
                candidate = base / subfolder / fname if subfolder else base / fname
                if candidate.exists():
                    return candidate
    hits = sorted(
        list(comfyui_output.glob(f"{output_prefix}*.mp4"))
        + list((comfyui_output.parent / "temp").glob(f"{output_prefix}*.mp4")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Headless ComfyUI Wan2.2Animate runner")
    parser.add_argument("--video", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--prompt", default="wearing grey top and blue jeans")
    parser.add_argument("--target", default="main guy wearing suit")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--workflow",
        default=str(
            Path(__file__).parent.parent / "workflows" / "Wan2.2Animate.gemini3pro.json"
        ),
    )
    parser.add_argument("--comfyui-input",  default="/mnt/data1/srini/ComfyUI/input")
    parser.add_argument("--comfyui-output", default="/mnt/data1/srini/ComfyUI/output")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=None, help="Sampler seed (random if omitted)")
    parser.add_argument(
        "--dump-api",
        metavar="FILE",
        help="Write resolved API JSON to FILE and exit (for debugging)",
    )
    args = parser.parse_args()

    import os
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big") % (2**32)

    video_src = Path(args.video).resolve()
    ref_src   = Path(args.ref).resolve()
    out_path  = Path(args.out).resolve()
    comfyui_input  = Path(args.comfyui_input)
    comfyui_output = Path(args.comfyui_output)
    workflow_path  = Path(args.workflow)

    for p, label in ((video_src, "video"), (ref_src, "ref"), (workflow_path, "workflow")):
        if not p.exists():
            sys.exit(f"{label} not found: {p}")

    print(f"Fetching node schemas from {args.host}:{args.port} ...", flush=True)
    object_info = fetch_object_info(args.host, args.port)
    widget_order = build_widget_order(object_info)

    print(f"Copying inputs -> {comfyui_input}", flush=True)
    video_name, ref_name = copy_inputs(video_src, ref_src, comfyui_input)

    print("Loading workflow ...", flush=True)
    graph = load_graph(workflow_path)
    nmap  = nodes_by_id(graph)
    lmap  = links_by_id(graph)

    print(f"Patching (seed={seed}) ...", flush=True)
    patch_nodes(
        nmap,
        video_filename=video_name,
        ref_filename=ref_name,
        prompt=args.prompt,
        target_description=args.target,
        output_prefix=out_path.stem,
        seed=seed,
    )

    print("Resolving graph -> API format ...", flush=True)
    api_prompt = graph_to_api(graph, nmap, lmap, widget_order)
    api_prompt = fixup_schema_mismatches(api_prompt, nmap)
    api_prompt = bypass_problematic_nodes(api_prompt)

    if args.dump_api:
        Path(args.dump_api).write_text(json.dumps(api_prompt, indent=2))
        print(f"API JSON written to {args.dump_api}. Exiting.")
        return

    print(f"Submitting to {args.host}:{args.port} ...", flush=True)
    prompt_id = submit_prompt(args.host, args.port, api_prompt)
    print(f"prompt_id={prompt_id}", flush=True)

    entry = poll_until_done(args.host, args.port, prompt_id, args.poll_interval, args.timeout)
    print("Done. Finding output ...", flush=True)

    result = find_output_video(entry, comfyui_output, out_path.stem)
    if result is None:
        sys.exit(f"Output not found in {comfyui_output}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result, out_path)
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
