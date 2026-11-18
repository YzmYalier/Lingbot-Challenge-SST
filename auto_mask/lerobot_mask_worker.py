#!/usr/bin/env python3
"""SAM3 phrase-mask worker for ``pipeline_lerobot.py`` (runs in the ``sam3`` env).

One worker owns one GPU. It reads stage-1 job files ``<queue>/jobs/<ep>.json``
(written by the orchestrator after Qwen), loads the episode's cam_high frames
once (a PNG folder the orchestrator decoded from the v30 video), and runs one
SAM3 propagation per DISTINCT phrase. Each phrase's per-frame binary mask is
packed into ``<work>/masks/<ep>.npz`` (key = phrase index, value =
``np.packbits(mask(T,H,W))``). Instructions are unioned later from these
phrase masks, so 100 instructions sharing 5 phrases cost 5 SAM3 passes.

Jobs with ``"format": "objects"`` (the Qwen-judged pipeline) keep SAM3's object ids
instead: key = phrase index, value = a (T,H,W) label volume (``k + 1`` = k-th object
of ``obj_ids[i]`` in the sidecar json, which also holds per-frame object areas), saved
compressed under ``<work>/<masks_subdir>/``.

One SAM3 session is reused across an episode's phrases via ``reset_session``:
measured bit-identical to fresh sessions and ~20% faster (no frame reload).
Files are claimed with an atomic rename so any number of workers can share a
queue and restart safely. Each orchestrator (one per node in multi-node runs)
owns its own ``<queue>``; ``masks/`` and ``failed/`` are shared under ``<work>``.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Locate the configuration; all project paths are then resolved from base.json.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "base.json").is_file():
        sys.path.insert(0, str(_candidate))
        break
from submit_paths import base_path, path as configured_path, interpreter, runtime_env, writable

PROJECT_ROOT = base_path()
sys.path.insert(0, str(PROJECT_ROOT / "sam3"))
from tools.sam3_video_masker import SAM3VideoMasker  # noqa: E402

DEFAULT_CHECKPOINT = configured_path("sam_checkpoint")


def propagate(predictor, request: dict, phrase: str, bf16_non_robot: bool, bf16_all: bool):
    """Use BF16 autocast according to the selected per-phrase policy."""
    use_bf16 = bf16_all or (bf16_non_robot and phrase.strip().lower() != "a robot arm")
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    with context:
        yield from predictor.handle_stream_request(request)



def phrase_masks(masker: SAM3VideoMasker, frames_dir: Path, phrases, n_frames: int, hw, bf16_non_robot: bool = False, bf16_all: bool = False):
    """Return {phrase: bool (T,H,W)}; one session, reset between phrases."""
    predictor = masker.predictor
    session = predictor.handle_request(
        {"type": "start_session", "resource_path": str(frames_dir), "offload_video_to_cpu": True}
    )
    sid = session["session_id"]
    out = {}
    try:
        for i, phrase in enumerate(phrases):
            if i:
                predictor.handle_request({"type": "reset_session", "session_id": sid})
            predictor.handle_request(
                {"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": phrase}
            )
            masks = np.zeros((n_frames, *hw), dtype=bool)
            seen = 0
            for result in propagate(predictor,
                {
                    "type": "propagate_in_video",
                    "session_id": sid,
                    "propagation_direction": "forward",
                    "evict_cached_frame_outputs": True,
                }, phrase, bf16_non_robot, bf16_all
            ):
                idx = int(result["frame_index"])
                if not 0 <= idx < n_frames:
                    raise RuntimeError(f"SAM3 returned frame index {idx} for {n_frames} frames")
                m = SAM3VideoMasker._merge_masks(
                    result.get("outputs", {}).get("out_binary_masks"), hw[0], hw[1], masker._import_cv2()
                )
                masks[idx] |= m
                seen += 1
            if seen != n_frames:
                raise RuntimeError(f"phrase {phrase!r}: SAM3 yielded {seen}/{n_frames} frames")
            out[phrase] = masks
    finally:
        predictor.handle_request({"type": "close_session", "session_id": sid})
    return out


def phrase_objects(masker: SAM3VideoMasker, frames_dir: Path, phrases, n_frames: int, hw, bf16_non_robot: bool = False, bf16_all: bool = False):
    """Return {phrase: (labels (T,H,W) uint8/uint16, obj_ids [int], overlap_px)}.

    ``labels[t] == k + 1`` where object ``obj_ids[k]`` is output at frame ``t`` (0 = none).
    Offline SAM3 already makes a phrase's objects disjoint (per-pixel non-overlap); on a
    leftover tie the later id wins and the pixels are counted in ``overlap_px``.
    Same session / propagation as :func:`phrase_masks`; the union of a phrase's labels is
    exactly its :func:`phrase_masks` mask.
    """
    predictor = masker.predictor
    cv2 = masker._import_cv2()
    session = predictor.handle_request(
        {"type": "start_session", "resource_path": str(frames_dir), "offload_video_to_cpu": True}
    )
    sid = session["session_id"]
    out = {}
    try:
        for i, phrase in enumerate(phrases):
            if i:
                predictor.handle_request({"type": "reset_session", "session_id": sid})
            predictor.handle_request(
                {"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": phrase}
            )
            labels = np.zeros((n_frames, *hw), dtype=np.uint16)
            pos: dict = {}
            overlap = seen = 0
            for result in propagate(predictor,
                {
                    "type": "propagate_in_video",
                    "session_id": sid,
                    "propagation_direction": "forward",
                    "evict_cached_frame_outputs": True,
                }, phrase, bf16_non_robot, bf16_all
            ):
                idx = int(result["frame_index"])
                if not 0 <= idx < n_frames:
                    raise RuntimeError(f"SAM3 returned frame index {idx} for {n_frames} frames")
                outputs = result.get("outputs", {}) or {}
                ids = outputs.get("out_obj_ids")
                masks = outputs.get("out_binary_masks")
                if ids is not None and masks is not None and len(ids):
                    masks = np.asarray(masks)
                    if len(masks) != len(ids):
                        raise RuntimeError(f"{len(ids)} obj ids vs {len(masks)} masks")
                    frame = labels[idx]
                    for obj_id, m in zip(np.asarray(ids).tolist(), masks):
                        m = SAM3VideoMasker._merge_masks(m, hw[0], hw[1], cv2)
                        if not m.any():
                            continue
                        k = pos.setdefault(int(obj_id), len(pos))
                        overlap += int((frame[m] != 0).sum())
                        frame[m] = k + 1
                seen += 1
            if seen != n_frames:
                raise RuntimeError(f"phrase {phrase!r}: SAM3 yielded {seen}/{n_frames} frames")
            if len(pos) > 65534:
                raise RuntimeError(f"phrase {phrase!r}: {len(pos)} objects")
            if len(pos) < 255:
                labels = labels.astype(np.uint8)
            out[phrase] = (labels, sorted(pos, key=pos.get), overlap)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": sid})
    return out


def label_areas(labels: np.ndarray, n_obj: int) -> list:
    """[t][k] pixel count of object k at frame t."""
    T = labels.shape[0]
    flat = labels.reshape(T, -1).astype(np.int64) + (np.arange(T, dtype=np.int64) * (n_obj + 1))[:, None]
    counts = np.bincount(flat.ravel(), minlength=T * (n_obj + 1)).reshape(T, n_obj + 1)
    return counts[:, 1:].tolist()


def claim(jobs_dir: Path, me: str):
    for path in sorted(jobs_dir.glob("*.json")):
        target = path.with_suffix(f".claimed.{me}")
        try:
            os.rename(path, target)
        except OSError:
            continue  # another worker took it
        return target
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--queue", type=Path, help="job queue dir (default: <work>/queue/local)")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--score-threshold-detection", type=float, default=0.3)
    ap.add_argument("--new-det-thresh", type=float, default=0.5)
    ap.add_argument("--bf16-all", action="store_true",
                    help="Use BF16 autocast for video propagation of every phrase")
    ap.add_argument("--bf16-non-robot", action="store_true",
                    help="Use BF16 autocast for non-robot propagation; robot stays on the current path")
    ap.add_argument("--exit-when-idle", type=float, default=0.0,
                    help="exit after this many idle seconds once <queue>/QUEUE_CLOSED exists (0 = wait forever)")
    args = ap.parse_args()

    queue = args.queue or args.work / "queue" / "local"
    jobs, masks_dir, failed = queue / "jobs", args.work / "masks", args.work / "failed"
    for d in (jobs, masks_dir, failed):
        d.mkdir(parents=True, exist_ok=True)
    me = f"{socket.gethostname()}_{os.getpid()}_gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"
    masker = SAM3VideoMasker(
        checkpoint_path=args.checkpoint,
        score_threshold_detection=args.score_threshold_detection,
        new_det_thresh=args.new_det_thresh,
    ).load()
    print(f"[{me}] ready", flush=True)
    idle_since = None
    while True:
        job_path = claim(jobs, me)
        if job_path is None:
            if (queue / "QUEUE_CLOSED").exists():
                idle_since = idle_since or time.time()
                if args.exit_when_idle and time.time() - idle_since > args.exit_when_idle:
                    break
            time.sleep(2)
            continue
        idle_since = None
        job = json.loads(job_path.read_text())
        ep = job["episode_index"]
        # mask_name lets one episode accumulate several mask files (the orchestrator only
        # queues phrases that no earlier file covers).
        fmt = job.get("format", "merged")
        dest_dir = args.work / job["masks_subdir"] if job.get("masks_subdir") else masks_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{job.get('mask_name', f'{ep:06d}')}.npz"
        t0 = time.time()
        try:
            frames_dir = Path(job["frames_dir"])
            hw = tuple(job["hw"])
            phrases = job["phrases"]
            tmp = dest.with_suffix(".tmp.npz")
            if fmt == "objects":
                # key "<i>": labels of phrase i (see phrase_objects); ids/areas in the sidecar
                res = phrase_objects(masker, frames_dir, phrases, job["n_frames"], hw, args.bf16_non_robot, args.bf16_all)
                np.savez_compressed(tmp, **{str(i): res[p][0] for i, p in enumerate(phrases)})
                extra = {"obj_ids": [res[p][1] for p in phrases],
                         "obj_areas": [label_areas(res[p][0], len(res[p][1])) for p in phrases],
                         "overlap_px": [res[p][2] for p in phrases],
                         "area": [int((res[p][0] != 0).sum()) for p in phrases]}
            elif fmt == "merged":
                res = phrase_masks(masker, frames_dir, phrases, job["n_frames"], hw, args.bf16_non_robot, args.bf16_all)
                np.savez(tmp, **{str(i): np.packbits(res[p], axis=None) for i, p in enumerate(phrases)})
                extra = {"area": [int(res[p].sum()) for p in phrases]}
            else:
                raise ValueError(f"unknown mask format {fmt!r}")
            os.replace(tmp, dest)
            meta = dict(job, seconds=round(time.time() - t0, 2), worker=me,
                        propagation_precision=["bf16" if args.bf16_all or (args.bf16_non_robot and p.strip().lower() != "a robot arm")
                                               else "baseline" for p in phrases], **extra)
            side = dest.with_suffix(".json.tmp")  # the orchestrator polls for the .json
            side.write_text(json.dumps(meta, ensure_ascii=False))
            os.replace(side, dest.with_suffix(".json"))
            job_path.unlink()
            print(f"[{me}] DONE ep={ep} frames={job['n_frames']} phrases={len(phrases)} "
                  f"{time.time() - t0:.1f}s", flush=True)
        except Exception as exc:  # keep the worker alive; record and move on
            (failed / f"{ep:06d}.txt").write_text(traceback.format_exc())
            os.replace(job_path, failed / f"{ep:06d}.json")
            print(f"[{me}] FAIL ep={ep}: {type(exc).__name__}: {exc}", flush=True)
            try:
                masker.unload()
            except Exception:
                pass
            masker.load()
    masker.unload()
    print(f"[{me}] exit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
