#!/usr/bin/env python3
"""Run task instruction -> Qwen objects -> SAM3 dimmed-video pipeline.

Qwen sees the instruction and, by default, only frame zero. The shared prompt
contract is in ``vllm_space/sam3_prompt.py``. SAM3 receives separate noun phrases.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Locate the configuration; all project paths are then resolved from base.json.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "base.json").is_file():
        sys.path.insert(0, str(_candidate))
        break
from submit_paths import base_path, path as configured_path, interpreter, runtime_env, writable

PROJECT_ROOT = base_path()
DATASET_ROOT = PROJECT_ROOT / "RoboTwinData" / "dataset"
SAM3_ROOT = PROJECT_ROOT / "sam3"
DEFAULT_CHECKPOINT = configured_path("sam_checkpoint")
DEFAULT_QWEN_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_SAM3_URL = ",".join(f"http://127.0.0.1:{port}" for port in range(8090, 8097))
# One source of truth for the CLI and the standalone Qwen client.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from vllm_space.sam3_prompt import MODEL_NAME, PROMPT_TEMPLATE, PROMPT_VERSION


def first_frame_data_url(video_path: Path) -> str:
    """Decode exactly one frame; never send a video or later frames to Qwen."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"could not decode frame zero: {video_path}")
    finally:
        capture.release()
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"could not encode frame zero: {video_path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def validate_output_path(video_path: Path, output_path: Path) -> None:
    """Keep all dataset inputs read-only, even with --allow-existing-output."""
    output = output_path.resolve()
    if output == video_path.resolve():
        raise ValueError("output must differ from the input video")
    if output.is_relative_to(DATASET_ROOT.resolve()):
        raise ValueError("outputs must be outside RoboTwinData/dataset; use auto_mask/outputs")


def parse_instruction_index(index: int) -> Tuple[str, int]:
    """Map the public 0..199 index to ``seen``/``unseen`` and local index."""

    if not 0 <= index < 200:
        raise ValueError("instruction-index must be in [0, 199]")
    return ("seen", index) if index < 100 else ("unseen", index - 100)


def split_object_phrases(content: str) -> List[str]:
    """Parse Qwen's comma-separated, explanation-free response."""

    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    if not cleaned or cleaned.upper() == "NONE":
        return []
    # The template requests commas. Accept a full-width comma as a harmless
    # robustness measure for a server/tokenizer that emits Chinese punctuation.
    phrases = [part.strip() for part in re.split(r"[,，]", cleaned)]
    phrases = list(dict.fromkeys(part for part in phrases if part and part.upper() != "NONE"))
    if len(phrases) > 16 or any("\n" in part or len(part.split()) > 12 for part in phrases):
        raise ValueError(f"Qwen did not return a short comma-separated prompt list: {cleaned!r}")
    return phrases


def load_instruction(
    task: str,
    split: str,
    episode: int,
    instruction_index: int,
    *,
    dataset_root: Path = DATASET_ROOT,
) -> Tuple[Path, str, str, int]:
    """Load one episode's instruction selected by its public 0..199 index."""

    if episode < 0:
        raise ValueError("episode must be non-negative")
    if split not in {"aloha-agilex_clean_50", "aloha-agilex_randomized_500"}:
        raise ValueError(f"unsupported split: {split}")
    group, local_index = parse_instruction_index(instruction_index)
    task_root = (dataset_root / task / split).resolve()
    if not task_root.is_dir():
        raise FileNotFoundError(f"dataset split does not exist: {task_root}")
    instruction_path = task_root / "instructions" / f"episode{episode}.json"
    video_path = task_root / "video" / f"episode{episode}.mp4"
    if not instruction_path.is_file():
        raise FileNotFoundError(f"instruction file does not exist: {instruction_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"video file does not exist: {video_path}")
    with instruction_path.open("r", encoding="utf-8") as handle:
        instructions = json.load(handle)
    try:
        task_description = instructions[group][local_index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"{instruction_path} does not contain {group}[{local_index}]"
        ) from exc
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError(f"empty task description at {instruction_path}:{group}[{local_index}]")
    return video_path, task_description, group, local_index


def extract_operation_objects(
    task_description: str,
    *,
    api_url: str = DEFAULT_QWEN_URL,
    first_frame_url: Optional[str] = None,
    timeout: float = 120.0,
) -> Tuple[str, List[str]]:
    """Call the existing vLLM OpenAI-compatible endpoint."""

    prompt = PROMPT_TEMPLATE.format(task_description=task_description)
    content: Any = prompt
    if first_frame_url is not None:
        if not first_frame_url.startswith("data:image/jpeg;base64,"):
            raise ValueError("first_frame_url must be a JPEG data URL")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": first_frame_url}},
        ]
    api_url = api_url.rstrip("/")
    if not api_url.endswith("/chat/completions"):
        api_url += "/chat/completions" if api_url.endswith("/v1") else "/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach vLLM at {api_url}: {exc.reason}") from exc
    try:
        raw_content = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected vLLM response: {result!r}") from exc
    if result["choices"][0].get("finish_reason") == "length":
        raise RuntimeError("Qwen prompt list was truncated; refusing partial segmentation")
    return raw_content, split_object_phrases(raw_content)


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"service HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach service at {url}: {exc.reason}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected service response from {url}: {result!r}")
    if result.get("ok") is False:
        raise RuntimeError(result.get("error", f"service failed: {result!r}"))
    return result


def select_worker_url(urls: str, worker_index: int) -> str:
    """Select one URL from a comma-separated worker pool."""

    candidates = [item.strip().rstrip("/") for item in urls.split(",") if item.strip()]
    if not candidates:
        raise ValueError("at least one worker URL is required")
    if worker_index < 0:
        raise ValueError("worker-index must be non-negative")
    return candidates[worker_index % len(candidates)]


def dim_with_service(
    *,
    service_url: str,
    video_path: Path,
    output_path: Path,
    prompts: Sequence[str],
    outside_scale: float,
    score_threshold_detection: float,
    new_det_thresh: float = 0.5,
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "video": str(video_path),
        "output": str(output_path),
        "text_prompts": list(prompts),
        "outside_scale": outside_scale,
        "score_threshold_detection": score_threshold_detection,
        "new_det_thresh": new_det_thresh,
    }
    return _post_json(service_url.rstrip("/") + "/dim", payload, timeout)


def dim_with_cli(
    *,
    video_path: Path,
    output_path: Path,
    prompts: Sequence[str],
    checkpoint: Path,
    outside_scale: float,
    score_threshold_detection: float,
    new_det_thresh: float = 0.5,
    gpu: Optional[str],
    timeout: float,
) -> None:
    """Run the same SAM3 CLI as sam3/test_sam3.sh for one video."""

    command = [
        sys.executable,
        str(SAM3_ROOT / "tools" / "sam3_video_dim.py"),
        "--video",
        str(video_path),
        "--output",
        str(output_path),
        "--checkpoint",
        str(checkpoint),
        "--outside-scale",
        str(outside_scale),
        "--score-threshold-detection",
        str(score_threshold_detection),
        "--new-det-thresh",
        str(new_det_thresh),
    ]
    for prompt in prompts:
        command.extend(["--text-prompt", prompt])
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(command, cwd=str(SAM3_ROOT), env=env, check=True, timeout=timeout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract target objects with Qwen and dim a video outside SAM3 masks."
    )
    parser.add_argument("--task", required=True, help="RoboTwinData task directory name")
    parser.add_argument(
        "--split",
        choices=("aloha-agilex_clean_50", "aloha-agilex_randomized_500"),
        default="aloha-agilex_randomized_500",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--instruction-index",
        type=int,
        required=True,
        help="0..99 selects seen[i], 100..199 selects unseen[i-100]",
    )
    parser.add_argument("--video", type=Path, help="Optional explicit video path")
    parser.add_argument("--output", type=Path, help="Defaults to auto_mask/outputs/<task>/<split>/episodeN_dim.mp4")
    parser.add_argument(
        "--qwen-url",
        default=DEFAULT_QWEN_URL,
        help="Qwen URL or comma-separated worker URLs",
    )
    parser.add_argument("--sam3-mode", choices=("service", "cli"), default="service")
    parser.add_argument(
        "--sam3-url",
        default=DEFAULT_SAM3_URL,
        help="SAM3 URL or comma-separated worker URLs",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=None,
        help="worker selected from URL pools; defaults to episode modulo pool size",
    )
    parser.add_argument("--sam3-gpu", default=None, help="CUDA_VISIBLE_DEVICES for cli mode")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--outside-scale", type=float, default=0.2)
    parser.add_argument("--score-threshold-detection", type=float, default=0.3)
    parser.add_argument("--new-det-thresh", type=float, default=0.5)
    parser.add_argument("--qwen-text-only", action="store_true", help="Disable the frame-zero image input")
    parser.add_argument("--qwen-timeout", type=float, default=120.0)
    parser.add_argument("--sam3-timeout", type=float, default=7200.0)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow overwriting the selected _dim.mp4 output",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.outside_scale <= 1.0:
        raise SystemExit("--outside-scale must be between 0 and 1")
    if not 0.0 <= args.score_threshold_detection <= 1.0:
        raise SystemExit("--score-threshold-detection must be between 0 and 1")

    if not 0.0 <= args.new_det_thresh <= 1.0:
        raise SystemExit("--new-det-thresh must be between 0 and 1")

    selected_video, task_description, group, local_index = load_instruction(
        args.task, args.split, args.episode, args.instruction_index
    )
    video_path = args.video.resolve() if args.video else selected_video.resolve()
    if not video_path.is_file():
        raise SystemExit(f"video file does not exist: {video_path}")
    output_path = (
        args.output.resolve()
        if args.output
        else PROJECT_ROOT / "auto_mask" / "outputs" / args.task / args.split / (video_path.stem + "_dim.mp4")
    )
    validate_output_path(video_path, output_path)
    if output_path.exists() and not args.allow_existing_output:
        raise SystemExit(
            f"output already exists: {output_path}; pass --allow-existing-output to overwrite"
        )

    print(f"instruction_index={args.instruction_index} ({group}[{local_index}])")
    print(f"task_description={task_description}")
    worker_index = args.worker_index if args.worker_index is not None else args.episode
    qwen_url = select_worker_url(args.qwen_url, worker_index)
    sam3_url = select_worker_url(args.sam3_url, worker_index)
    print(f"worker_index={worker_index} qwen_url={qwen_url} sam3_url={sam3_url}")
    raw_objects, prompts = extract_operation_objects(
        task_description, api_url=qwen_url, timeout=args.qwen_timeout,
        first_frame_url=None if args.qwen_text_only else first_frame_data_url(video_path)
    )
    print(f"qwen_raw={raw_objects}")
    if not prompts:
        raise SystemExit("Qwen returned NONE; no SAM3 prompt can be generated")
    print("sam3_prompts=" + json.dumps(prompts, ensure_ascii=False))

    if args.sam3_mode == "service":
        result = dim_with_service(
            service_url=sam3_url,
            video_path=video_path,
            output_path=output_path,
            prompts=prompts,
            outside_scale=args.outside_scale,
            score_threshold_detection=args.score_threshold_detection,
            new_det_thresh=args.new_det_thresh,
            timeout=args.sam3_timeout,
        )
        print(f"sam3_service={result}")
    else:
        dim_with_cli(
            video_path=video_path,
            output_path=output_path,
            prompts=prompts,
            checkpoint=args.checkpoint,
            outside_scale=args.outside_scale,
            score_threshold_detection=args.score_threshold_detection,
            new_det_thresh=args.new_det_thresh,
            gpu=args.sam3_gpu,
            timeout=args.sam3_timeout,
        )
    metadata = {
        "video": str(video_path), "output": str(output_path),
        "task": args.task, "split": args.split, "episode": args.episode,
        "instruction_index": args.instruction_index, "instruction": task_description,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "qwen_first_frame": not args.qwen_text_only,
        "qwen_raw": raw_objects, "sam3_prompts": prompts,
        "score_threshold_detection": args.score_threshold_detection,
        "new_det_thresh": args.new_det_thresh, "outside_scale": args.outside_scale,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

