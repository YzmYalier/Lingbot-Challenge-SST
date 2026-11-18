#!/usr/bin/env python3
"""Build SAM3-dimmed cam_high copies of RoboTwin_lerobot_v30 (run in the ``lingbotvla`` env).

Modes
-----
single -> ``RoboTwin_lerobot_v30_sam_single``: each episode's own v30 instruction drives one
          dimmed ``observation.images.cam_high``. Index, data and other cameras are unchanged.
multi  -> ``RoboTwin_lerobot_v30_sam_multi``: each episode keeps its 100 *seen* instructions
          (episodes ``tasks`` column, in order) and ``observation.images.cam_high_<i>`` is the
          video dimmed for instruction ``i`` (1-based). The plain cam_high is replaced.

Per episode (every stage is resumable from ``--work``):
  1. Qwen (vLLM, temperature 0) sees the instruction + v30 cam_high frame 0 -> noun phrases.
  2. The v30 cam_high segment is decoded to ``<work>/frames/<ep>/<i>.png`` and queued;
     ``lerobot_mask_worker.py`` (sam3 env, one per GPU) runs SAM3 once per DISTINCT phrase and
     keeps SAM3's object ids (per-phrase label volumes in ``<work>/masks_obj``).
  2b. Judge (``--judge``, default on): every object id is shown to the same Qwen
     (``sam3/tools/sam3_mask_judge.py``: original frame + mask image, yes/no logprobs), in
     causal order -- on the first frame it is output, and again when a rejected id grows to
     ``--judge-regrow`` x its last-verdict area. A frame shows an id iff its latest verdict up
     to that frame is "yes" (same rule as the eval stream server's /step_judged). Each
     (phrase, id) is judged once per episode with the first instruction that produced the
     phrase; verdicts are cached in ``<work>/<mode>/judged/judge/<ep>.json``.
  3. Compose: per instruction, union the kept objects of its phrases (the final mask), save it
     losslessly (FFV1 gray mkv), render the style (default: outside x 0.2), compute
     lerobot-style stats and AV1-encode a per-episode segment with the same frame count.
  4. Assemble: concatenate segments in the source file layout (so from/to_timestamp stay
     valid), rewrite meta (episodes, stats, info, tasks) and copy the final masks to
     ``<out>/sam_masks/`` (+ ``index.parquet``, ``judge_log.parquet``). lerobot only reads the
     keys in ``meta/info.json``, so the extra directory does not change loading.
     ``composite_lerobot.py`` re-renders a dataset in another style from these masks.

Subset runs (``--episodes`` / ``--per-task``; smoke tests) write a self-consistent,
re-indexed dataset with every camera re-encoded into chunk-000/file-000.

Multi-node (see root ``process_data.sh --stage masks --shard R/N``): every node runs stages 1-3 with ``--shard R/N``
(episodes ``ep % N == R``) against the same shared ``--work``; each shard has its own SAM3
queue ``<work>/queue/rank<R>of<N>``. One node then runs ``--stage assemble`` unsharded.
"""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures as cf
import itertools
import json
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Locate the configuration; all project paths are then resolved from base.json.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "base.json").is_file():
        sys.path.insert(0, str(_candidate))
        break
from submit_paths import base_path, path as configured_path, interpreter, runtime_env, writable

PROJECT_ROOT = base_path()
AUTO_MASK = PROJECT_ROOT / "auto_mask"
sys.path.insert(0, str(AUTO_MASK))
sys.path.append(str(PROJECT_ROOT / "sam3"))  # tools.sam3_mask_judge (numpy-only at import)
from mask_style import MaskStyle, add_style_args, style_from_args  # noqa: E402
from pipeline import DEFAULT_QWEN_URL, PROMPT_VERSION, extract_operation_objects  # noqa: E402
from vllm_space.sam3_prompt import MODEL_NAME  # noqa: E402

import lerobot.datasets.compute_stats as cs  # noqa: E402
from lerobot.datasets.video_utils import concatenate_video_files  # noqa: E402
from tools.sam3_mask_judge import QwenMaskJudge, judge_track, keep_timeline  # noqa: E402

DEFAULT_SRC = configured_path("raw")
RAW_ROOT = configured_path("instructions")
MAPPING = configured_path("mapping")
SAM3_PYTHON = interpreter("sam3")
WORKER = AUTO_MASK / "lerobot_mask_worker.py"
CAM = "observation.images.cam_high"
N_MULTI = 100
# Bump when the meaning of cached masks / judge files / segments changes.
FORMAT = "objects-judge-v1"
MASKS_SUBDIR = "masks_obj"
MASK_CODEC = {"vcodec": "ffv1", "pix_fmt": "gray", "container": "mkv", "values": "0 / 255"}
# lerobot encode_video_frames defaults (libsvtav1, yuv420p, g=2, crf=30, preset 12).
ENCODE_OPTIONS = {"g": "2", "crf": "30", "preset": "12"}
TS_EPS = 1e-4
SAM3_ARGS = {"score_threshold_detection": 0.3, "new_det_thresh": 0.5}


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def write_json(path: Path, obj, indent=None) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))
    os.replace(tmp, path)


def norm_phrase(phrase: str) -> str:
    return " ".join(phrase.split()).lower()


def quiet_stderr(log_path: str) -> None:
    """Pool initializer: SVT-AV1 prints a banner to fd 2 for every encoder it opens."""
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 2)
    av.logging.set_level(av.logging.ERROR)


# ----------------------------------------------------------------------------- source meta
class Source:
    """Read-only view of the v30 dataset meta plus the raw instruction files."""

    def __init__(self, root: Path):
        self.root = root
        self.info = json.loads((root / "meta" / "info.json").read_text())
        self.fps = int(self.info["fps"])
        files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        self.ep_table = pa.concat_tables([pq.read_table(f) for f in files]).combine_chunks()
        cols = [c for c in self.ep_table.column_names if not c.startswith("stats/")]
        self.eps = self.ep_table.select(cols).to_pandas().set_index("episode_index", drop=False)
        if not (self.eps.index.values == np.arange(len(self.eps))).all():
            raise ValueError("expected episode_index == row order 0..N-1")
        self.mapping = json.loads(MAPPING.read_text())["mapping"]
        h, w = self.info["features"][CAM]["shape"][1:]
        self.hw = (int(h), int(w))

    def n_video_frames(self, ep: int, key: str = CAM) -> int:
        r = self.eps.loc[ep]
        return int(round((r[f"videos/{key}/to_timestamp"] - r[f"videos/{key}/from_timestamp"]) * self.fps))

    def video_path(self, ep: int, key: str = CAM) -> Path:
        r = self.eps.loc[ep]
        c, f = int(r[f"videos/{key}/chunk_index"]), int(r[f"videos/{key}/file_index"])
        return self.root / self.info["video_path"].format(video_key=key, chunk_index=c, file_index=f)

    def task_name(self, ep: int) -> str:
        return self.mapping[str(ep)][0]

    def seen_instructions(self, ep: int) -> list[str]:
        task, rel = self.mapping[str(ep)]
        base, name = rel.split("/data/")
        path = RAW_ROOT / task / base / "instruction" / name.replace(".hdf5", ".json")
        seen = json.loads(path.read_text())["seen"]
        if len(seen) != N_MULTI:
            raise ValueError(f"{path}: expected {N_MULTI} seen instructions, got {len(seen)}")
        return seen

    def instructions(self, ep: int, mode: str) -> list[str]:
        if mode == "single":
            return [str(self.eps.loc[ep, "tasks"][0])]
        return self.seen_instructions(ep)


def decode_segment(path: Path, from_ts: float, to_ts: float, n_expected: int, first_only=False):
    """RGB frames of one episode inside a concatenated v30 file (episodes start on keyframes)."""
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(round(from_ts / float(stream.time_base))), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * frame.time_base)
            if t < from_ts - TS_EPS:
                continue
            if t >= to_ts - TS_EPS:
                break
            out.append(frame.to_ndarray(format="rgb24"))
            if first_only:
                return out
    if len(out) != n_expected:
        raise RuntimeError(f"{path} [{from_ts},{to_ts}): decoded {len(out)} frames, expected {n_expected}")
    return out


def source_frames(src: Source, ep: int, key: str = CAM, first_only=False):
    r = src.eps.loc[ep]
    return decode_segment(src.video_path(ep, key), float(r[f"videos/{key}/from_timestamp"]),
                          float(r[f"videos/{key}/to_timestamp"]), src.n_video_frames(ep, key), first_only)


def jpeg_data_url(rgb: np.ndarray) -> str:
    # Same encoding as pipeline.first_frame_data_url (BGR frame, JPEG q95).
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("could not JPEG-encode frame 0")
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


# ----------------------------------------------------------------------------- video + stats
def encode_video(frames, path: Path, fps: int) -> None:
    tmp = path.with_name(path.stem + ".tmp.mp4")
    h, w = frames[0].shape[:2]
    with av.open(str(tmp), "w") as out:
        stream = out.add_stream("libsvtav1", fps, options=ENCODE_OPTIONS)
        stream.pix_fmt, stream.width, stream.height = "yuv420p", w, h
        for a in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    os.replace(tmp, path)


def image_stats(frames: np.ndarray) -> dict:
    """lerobot compute_episode_stats for one video episode, (T,H,W,3) uint8 RGB.

    Same sampling/downsampling as lerobot; ``count`` follows the v30 convention
    (video frames x downsampled pixels), which reproduces the source stats.
    """
    sampled = np.stack([cs.auto_downsample_height_width(frames[i].transpose(2, 0, 1))
                        for i in cs.sample_indices(len(frames))])
    s = cs.get_feature_stats(sampled, axis=(0, 2, 3), keepdims=True, quantile_list=cs.DEFAULT_QUANTILES)
    out = {k: np.squeeze(v / 255.0, axis=0).tolist() for k, v in s.items() if k != "count"}
    out["count"] = [int(len(frames) * sampled.shape[2] * sampled.shape[3])]
    return out


def encode_mask_video(keep: np.ndarray, path: Path, fps: int) -> None:
    """(T,H,W) bool -> lossless FFV1 gray mkv (0 / 255), frame-aligned with the camera video."""
    tmp = path.with_name(path.stem + ".tmp.mkv")
    T, h, w = keep.shape
    with av.open(str(tmp), "w") as out:
        stream = out.add_stream("ffv1", fps)
        stream.pix_fmt, stream.width, stream.height = "gray", w, h
        for t in range(T):
            img = keep[t].view(np.uint8) * np.uint8(255)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="gray")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    os.replace(tmp, path)


def decode_mask_video(path: Path) -> np.ndarray:
    """Inverse of :func:`encode_mask_video` -> (T,H,W) bool."""
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="gray") > 127 for f in c.decode(video=0)])


# ----------------------------------------------------------------------------- stage 1: Qwen
_QWEN_RR = itertools.count()


def qwen_one(instruction: str, url: str, api_urls: list[str], retries: int = 4):
    """-> (raw, phrases, error). A malformed/truncated answer is recorded (the instruction
    then gets an undimmed video); service/network errors are retried, then raised so the
    episode fails and is retried on the next run instead of being cached."""
    for attempt in range(retries):
        api_url = api_urls[next(_QWEN_RR) % len(api_urls)]  # round-robin over vLLM instances
        try:
            raw, phrases = extract_operation_objects(instruction, api_url=api_url, first_frame_url=url)
            return raw, phrases, None
        except ValueError as exc:  # split_object_phrases rejected the answer
            return "", [], f"{type(exc).__name__}: {exc}"
        except RuntimeError as exc:
            if "truncated" in str(exc):
                return "", [], f"RuntimeError: {exc}"
            if attempt + 1 == retries:
                raise
        except OSError:
            if attempt + 1 == retries:
                raise
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def qwen_episode(src: Source, ep: int, mode: str, api_urls: list[str], req_pool, path: Path) -> int:
    instructions = src.instructions(ep, mode)
    url = jpeg_data_url(source_frames(src, ep, first_only=True)[0])
    uniq = list(dict.fromkeys(instructions))
    futs = {s: req_pool.submit(qwen_one, s, url, api_urls) for s in uniq}
    res = {s: f.result() for s, f in futs.items()}
    phrases = [[norm_phrase(p) for p in res[s][1]] for s in instructions]
    phrases = [list(dict.fromkeys(p for p in ps if p)) for ps in phrases]
    write_json(path, {
        "episode_index": ep, "mode": mode, "prompt_version": PROMPT_VERSION,
        "task": src.task_name(ep), "instructions": instructions, "phrases": phrases,
        "raw": [res[s][0] for s in instructions], "errors": [res[s][2] for s in instructions],
    })
    return ep


def load_qwen(path: Path, src: Source, ep: int, mode: str):
    if not path.exists():
        return None
    q = json.loads(path.read_text())
    if q.get("prompt_version") != PROMPT_VERSION or q.get("instructions") != src.instructions(ep, mode):
        return None
    return q


def distinct_phrases(q: dict) -> list[str]:
    return list(dict.fromkeys(p for ps in q["phrases"] for p in ps))


# ----------------------------------------------------------------------------- stage 2b: judge
def load_phrase_objects(entry):
    """mask_index entry -> (labels (T,H,W), obj_ids [int], areas (T, n_obj) pixel counts)."""
    npz, key, side = entry
    meta = json.loads(Path(side).read_text())
    k = int(key)
    with np.load(npz) as z:
        labels = z[key]
    ids = [int(i) for i in meta["obj_ids"][k]]
    areas = np.asarray(meta["obj_areas"][k], dtype=np.int64).reshape(labels.shape[0], len(ids))
    return labels, ids, areas


def phrase_instructions(q: dict) -> dict:
    """phrase -> the first instruction (slot order) that produced it: the judge's context."""
    out = {}
    for ins, ps in zip(q["instructions"], q["phrases"]):
        for p in ps:
            out.setdefault(p, ins)
    return out


def judge_signature(judge: QwenMaskJudge, args) -> dict:
    return dict(judge.config(), regrow=float(args.judge_regrow), regrow_min_area=float(args.judge_regrow_min_area))


def judge_entry_ok(entry, sig: dict, instruction: str, mask_ref) -> bool:
    return (entry is not None and entry.get("config") == sig and entry.get("instruction") == instruction
            and entry.get("mask") == mask_ref)


def mask_ref(entry) -> list:
    return [Path(entry[0]).name, entry[1]]


def judge_todo(path: Path, q: dict, masks_of: dict, sig: dict) -> list:
    """Phrases whose cached verdicts are missing or stale."""
    done = json.loads(path.read_text())["phrases"] if path.exists() else {}
    instr = phrase_instructions(q)
    return [p for p in distinct_phrases(q)
            if not judge_entry_ok(done.get(p), sig, instr[p], mask_ref(masks_of[p]))]


_JUDGE_RR = itertools.count()


def judge_objects(frames: np.ndarray, items: list, judges: list, sig: dict, chain_pool) -> tuple:
    """Judge every object of ``items`` = [(phrase, instruction, mask_index entry)] on ``frames``.

    Objects are independent chains of requests (judge_track) and run concurrently on
    ``chain_pool``. A Qwen outage raises (strict); parse errors keep the object.
    -> ({phrase: {"obj_ids", "verdicts": {obj_id: [verdict, ...]}}}, number of requests)
    """
    T, H, W = frames.shape[:3]
    url_cache = {}

    def frame_url(t):
        if t not in url_cache:
            url_cache[t] = judges[0].frame_image(frames[t])
        return url_cache[t]

    futs, out = {}, {}
    for phrase, instruction, entry in items:
        labels, ids, areas = load_phrase_objects(entry)
        if labels.shape != (T, H, W):
            raise RuntimeError(f"{phrase!r}: labels {labels.shape} vs frames {(T, H, W)}")
        for k, obj_id in enumerate(ids):
            def ask(t, labels=labels, k=k, phrase=phrase, instruction=instruction):
                judge = judges[next(_JUDGE_RR) % len(judges)]
                return judge.ask(frame_url(t), frames[t], labels[t] == k + 1, instruction, phrase, strict=True)
            fut = chain_pool.submit(judge_track, (areas[:, k] / float(H * W)).tolist(), ask,
                                    sig["regrow"], sig["regrow_min_area"])
            futs[fut] = (phrase, obj_id)
        out[phrase] = {"obj_ids": ids, "verdicts": {}}
    n_calls = 0
    try:
        for fut in cf.as_completed(futs):
            phrase, obj_id = futs[fut]
            out[phrase]["verdicts"][str(obj_id)] = fut.result()
            n_calls += len(out[phrase]["verdicts"][str(obj_id)])
    except BaseException:
        for f in futs:
            f.cancel()
        raise
    return out, n_calls


def judge_episode(src: Source, ep: int, q: dict, masks_of: dict, judges: list, sig: dict,
                  chain_pool, path: Path) -> dict:
    """Judge the not-yet-judged phrases of an episode and cache the verdicts in ``path``."""
    t0 = time.time()
    old = json.loads(path.read_text())["phrases"] if path.exists() else {}
    instr = phrase_instructions(q)
    todo = judge_todo(path, q, masks_of, sig)
    result = {p: old[p] for p in distinct_phrases(q) if p not in todo}
    n_calls = 0
    if todo:
        frames = np.stack(source_frames(src, ep))
        res, n_calls = judge_objects(frames, [(p, instr[p], masks_of[p]) for p in todo], judges, sig, chain_pool)
        for p in todo:
            result[p] = {"instruction": instr[p], "mask": mask_ref(masks_of[p]), "config": sig, **res[p]}
    write_json(path, {"episode_index": ep, "phrases": result})
    n_obj = sum(len(e["obj_ids"]) for e in result.values())
    n_rej = sum(1 for e in result.values() for vs in e["verdicts"].values() if vs and not vs[-1]["keep"])
    return {"ep": ep, "objects": n_obj, "rejected_final": n_rej, "calls": n_calls, "seconds": time.time() - t0}


# ----------------------------------------------------------------------------- stage 3: compose
def kept_mask(labels: np.ndarray, ids: list, verdicts) -> np.ndarray:
    """(T,H,W) bool: pixels of objects shown at each frame. ``verdicts`` None = keep all."""
    if verdicts is None or not ids:
        return labels != 0
    T = labels.shape[0]
    table = np.zeros((T, len(ids) + 1), dtype=bool)
    for k, obj_id in enumerate(ids):
        table[:, k + 1] = keep_timeline(verdicts.get(str(obj_id), []), T)
    return table[np.arange(T)[:, None, None], labels]


def compose_episode(src_root: str, ep: int, qwen_path: str, masks_of: dict, judge_path, seg_dir: str,
                    done_path: str, style: dict, signature: dict) -> dict:
    """Union kept objects per instruction, save the mask, render, stats, encode (CPU worker)."""
    t0 = time.time()
    src = _cached_source(src_root)
    style = MaskStyle.from_dict(style)
    q = json.loads(Path(qwen_path).read_text())
    judged = json.loads(Path(judge_path).read_text())["phrases"] if judge_path else None
    frames = np.stack(source_frames(src, ep))
    T, H, W = frames.shape[:3]
    masks, raw = {}, {}
    for p in distinct_phrases(q):
        labels, ids, _ = load_phrase_objects(masks_of[p])
        if labels.shape != (T, H, W):
            raise RuntimeError(f"ep {ep} {p!r}: labels {labels.shape} vs frames {(T, H, W)}")
        masks[p] = kept_mask(labels, ids, None if judged is None else judged[p]["verdicts"])
        raw[p] = labels != 0
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    groups, slot_group, key_to_group = [], [], {}
    for ps in q["phrases"]:
        key = tuple(sorted(ps))
        if key not in key_to_group:
            g = len(groups)
            if ps:
                keep = np.zeros((T, H, W), dtype=bool)
                keep_raw = np.zeros((T, H, W), dtype=bool)
                for p in ps:
                    keep |= masks[p]
                    keep_raw |= raw[p]
            else:  # Qwen returned NONE / failed: nothing to highlight, leave undimmed
                keep = keep_raw = np.ones((T, H, W), dtype=bool)
            out = style.render(frames, keep) if ps else frames
            seg, mfile = seg_dir / f"g{g:03d}.mp4", seg_dir / f"g{g:03d}_mask.mkv"
            encode_video(out, seg, src.fps)
            encode_mask_video(keep, mfile, src.fps)
            groups.append({"phrases": list(key), "file": str(seg), "mask_file": str(mfile),
                           "keep_frac": round(float(keep.mean()), 5),
                           "keep_frac_unjudged": round(float(keep_raw.mean()), 5),
                           "stats": image_stats(out)})
            key_to_group[key] = g
        slot_group.append(key_to_group[key])
    write_json(Path(done_path), {"episode_index": ep, "n_frames": T, "groups": groups, "signature": signature,
                                 "slot_group": slot_group, "seconds": round(time.time() - t0, 2)})
    return {"ep": ep, "groups": len(groups), "seconds": time.time() - t0}


_SOURCE_CACHE: dict = {}


def _cached_source(root: str) -> Source:
    if root not in _SOURCE_CACHE:
        _SOURCE_CACHE[root] = Source(Path(root))
    return _SOURCE_CACHE[root]


def reencode_episode(src_root: str, ep: int, key: str, out: str) -> int:
    """Subset runs only: copy another camera's episode by decode + re-encode."""
    src = _cached_source(src_root)
    frames = source_frames(src, ep, key)
    encode_video(frames, Path(out), src.fps)
    return len(frames)


def concat_segments(segs: list[str], out: str, n_expected: int) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    concatenate_video_files([Path(s) for s in segs], out)
    with av.open(str(out)) as c:
        n = sum(1 for p in c.demux(video=0) if p.pts is not None)
    if n != n_expected:
        raise RuntimeError(f"{out}: {n} frames after concat, expected {n_expected}")
    return n


# ----------------------------------------------------------------------------- orchestration
class Work:
    """``<work>/{frames,masks_obj,failed}`` are shared by both modes (SAM3 masks are keyed by
    episode + phrase, so whichever mode runs second reuses them). Qwen answers live under
    ``<work>/<mode>/qwen``; judge verdicts, segments (+ final mask videos) and done markers under
    ``<work>/<mode>/<variant>/`` (variant = judged / nojudge), so the two never overwrite each
    other. Every per-episode file is only touched by the shard owning that episode, so nodes
    share these dirs safely. The SAM3 queue (``<queue>/jobs``, ``QUEUE_CLOSED``) belongs to one
    orchestrator; logs are per host."""

    def __init__(self, root: Path, mode: str, queue_tag: str = "local", variant: str = "judged"):
        self.root = root
        self.queue = root / "queue" / queue_tag
        self.frames, self.jobs = root / "frames", self.queue / "jobs"
        self.masks, self.failed = root / MASKS_SUBDIR, root / "failed"
        self.logs = root / "logs" / socket.gethostname()
        self.qwen = root / mode / "qwen"
        v = root / mode / variant
        self.judge, self.segs, self.done = v / "judge", v / "segments", v / "done"
        for d in (self.qwen, self.frames, self.jobs, self.masks, self.failed, self.judge, self.segs,
                  self.done, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def q(self, ep):
        return self.qwen / f"{ep:06d}.json"

    def judge_json(self, ep):
        return self.judge / f"{ep:06d}.json"

    def done_json(self, ep):
        return self.done / f"{ep:06d}.json"


def mask_index(work: Work, ep: int, n_frames: int) -> dict:
    """phrase -> [npz path, key, sidecar json] over every object-mask file SAM3 wrote for this episode."""
    idx = {}
    for side in sorted(work.masks.glob(f"{ep:06d}_*.json")):
        meta = json.loads(side.read_text())
        npz = side.with_suffix(".npz")
        if meta.get("n_frames") != n_frames or meta.get("format") != "objects" or not npz.exists():
            continue
        for k, p in enumerate(meta["phrases"]):
            idx.setdefault(p, [str(npz), str(k), str(side)])
    return idx


def done_ok(work: Work, ep: int, q: dict, signature: dict) -> bool:
    p = work.done_json(ep)
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    keys = [tuple(sorted(ps)) for ps in q["phrases"]]
    return (d.get("signature") == signature and len(d["slot_group"]) == len(keys)
            and all(tuple(d["groups"][g]["phrases"]) == k for g, k in zip(d["slot_group"], keys))
            and all(Path(g["file"]).exists() and Path(g["mask_file"]).exists() for g in d["groups"]))


def write_frames_and_job(src: Source, work: Work, ep: int, phrases: list[str], mask_name: str) -> None:
    fdir = work.frames / f"{ep:06d}"
    frames = source_frames(src, ep)
    if fdir.exists():
        shutil.rmtree(fdir)
    tmp = fdir.with_name(fdir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for i, a in enumerate(frames):  # SAM3 image-folder loader wants "<int>.png"
        cv2.imwrite(str(tmp / f"{i}.png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    os.replace(tmp, fdir)
    job = {"episode_index": ep, "frames_dir": str(fdir), "hw": list(src.hw), "phrases": phrases,
           "n_frames": len(frames), "mask_name": mask_name, "format": "objects",
           "masks_subdir": MASKS_SUBDIR}
    write_json(work.jobs / f"{ep:06d}.json", job)


def launch_sam3_workers(work: Work, gpus: list[str], fresh: bool = True) -> list[subprocess.Popen]:
    if fresh:  # drop this queue's leftovers; run_masks re-queues whatever masks are still missing
        for p in work.jobs.iterdir():
            p.unlink()
        (work.queue / "QUEUE_CLOSED").unlink(missing_ok=True)
    tmpdir = AUTO_MASK / "tmp" / socket.gethostname()
    tmpdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for g in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS="4", TMPDIR=str(tmpdir))
        logf = open(work.logs / f"sam3_gpu{g}.log", "a")
        procs.append(subprocess.Popen(
            [str(SAM3_PYTHON), str(WORKER), "--work", str(work.root), "--queue", str(work.queue),
             "--exit-when-idle", "30",
             "--score-threshold-detection", str(SAM3_ARGS["score_threshold_detection"]),
             "--new-det-thresh", str(SAM3_ARGS["new_det_thresh"])],
            env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT)))
    log(f"launched {len(procs)} SAM3 workers on GPUs {','.join(gpus)} (logs in {work.logs})")
    return procs


def run_masks(args, src: Source, work: Work, episodes: list[int]) -> list[int]:
    """Stages 1-3 for ``episodes``; returns the episodes that failed."""
    mode = args.mode
    procs = launch_sam3_workers(work, args.sam3_gpus) if args.sam3_gpus and args.stage != "qwen" else []
    req_pool = cf.ThreadPoolExecutor(args.qwen_concurrency)    # HTTP requests
    qwen_pool = cf.ThreadPoolExecutor(args.qwen_episodes)      # episodes in Qwen (FIFO)
    ep_pool = cf.ThreadPoolExecutor(args.prep_threads)         # PNG frame dumps for SAM3
    judge_pool = cf.ThreadPoolExecutor(args.judge_episodes)    # episodes in the judge stage
    chain_pool = cf.ThreadPoolExecutor(args.judge_concurrency)  # per-object judge request chains
    cpu_pool = cf.ProcessPoolExecutor(args.compose_workers, mp_context=mp.get_context("spawn"),
                                      initializer=quiet_stderr, initargs=(str(work.logs / "encode.log"),))
    state = {}            # ep -> "qwen" | "sam" | "judge" | "compose" | "done" | "failed"
    qwen_futs, prep_futs, judge_futs, comp_futs = {}, {}, {}, {}
    failed = {}
    t_start = time.time()
    n_sam_phr = 0
    judge_tot = collections.Counter()

    pending_mask = {}     # ep -> mask file name the SAM3 workers will write
    to_prep = collections.deque()  # (ep, phrases, mask_name) waiting for a SAM3 queue slot

    def start_compose(ep, masks_of):
        state[ep] = "compose"
        jpath = str(work.judge_json(ep)) if args.judge else None
        comp_futs[cpu_pool.submit(compose_episode, str(src.root), ep, str(work.q(ep)), masks_of, jpath,
                                  str(work.segs / f"{ep:06d}"), str(work.done_json(ep)),
                                  args.style_obj.to_dict(), args.signature)] = ep

    def after_masks(ep, masks_of):
        q = load_qwen(work.q(ep), src, ep, mode)
        if args.judge and judge_todo(work.judge_json(ep), q, masks_of, args.judge_sig):
            state[ep] = "judge"
            judge_futs[judge_pool.submit(judge_episode, src, ep, q, masks_of, args.judges, args.judge_sig,
                                         chain_pool, work.judge_json(ep))] = (ep, masks_of)
        else:
            start_compose(ep, masks_of)

    def after_qwen(ep):
        nonlocal n_sam_phr
        q = load_qwen(work.q(ep), src, ep, mode)
        if q is None:
            raise RuntimeError(f"ep {ep}: qwen result missing/stale after stage 1")
        if done_ok(work, ep, q, args.signature) or args.stage == "qwen":
            state[ep] = "done"
            return
        idx = mask_index(work, ep, src.n_video_frames(ep))
        missing = [p for p in distinct_phrases(q) if p not in idx]
        if not missing:
            after_masks(ep, idx)
            return
        state[ep] = "sam"
        for stale in (work.failed / f"{ep:06d}.json", work.failed / f"{ep:06d}.txt"):
            stale.unlink(missing_ok=True)
        n_sam_phr += len(missing)
        name = f"{ep:06d}_{len(list(work.masks.glob(f'{ep:06d}_*.json'))):02d}"
        pending_mask[ep] = name
        to_prep.append((ep, missing, name))

    for ep in episodes:
        if load_qwen(work.q(ep), src, ep, mode) is not None:
            after_qwen(ep)
        else:
            state[ep] = "qwen"
            qwen_futs[qwen_pool.submit(qwen_episode, src, ep, mode, args.qwen_urls, req_pool, work.q(ep))] = ep
    log(f"{len(episodes)} episodes: " + str(dict(collections.Counter(state.values()))))
    closed = False
    released = False
    last_report = 0.0
    try:
        while True:
            for fut in [f for f in qwen_futs if f.done()]:
                ep = qwen_futs.pop(fut)
                try:
                    fut.result()
                    after_qwen(ep)
                except Exception as exc:
                    failed[ep] = f"qwen: {type(exc).__name__}: {exc}"
                    state[ep] = "failed"
            for fut in [f for f in prep_futs if f.done()]:
                ep = prep_futs.pop(fut)
                try:
                    fut.result()
                except Exception as exc:
                    failed[ep] = f"frames: {type(exc).__name__}: {exc}"
                    state[ep] = "failed"
            in_flight = sum(1 for e, st in state.items() if st == "sam") - len(to_prep)
            while to_prep and in_flight < args.max_queued:  # caps PNG frames on disk
                ep, missing, name = to_prep.popleft()
                prep_futs[ep_pool.submit(write_frames_and_job, src, work, ep, missing, name)] = ep
                in_flight += 1
            for ep in [e for e, s in state.items() if s == "sam"]:
                if (work.failed / f"{ep:06d}.json").exists():
                    failed[ep] = "sam3: " + (work.failed / f"{ep:06d}.txt").read_text()[-600:]
                    state[ep] = "failed"
                    continue
                if (work.masks / f"{pending_mask[ep]}.json").exists():
                    shutil.rmtree(work.frames / f"{ep:06d}", ignore_errors=True)
                    try:
                        after_masks(ep, mask_index(work, ep, src.n_video_frames(ep)))
                    except Exception as exc:
                        failed[ep] = f"masks: {type(exc).__name__}: {exc}"
                        state[ep] = "failed"
            for fut in [f for f in judge_futs if f.done()]:
                ep, masks_of = judge_futs.pop(fut)
                try:
                    r = fut.result()
                    judge_tot.update({k: r[k] for k in ("objects", "rejected_final", "calls")})
                    start_compose(ep, masks_of)
                except Exception as exc:
                    failed[ep] = f"judge: {type(exc).__name__}: {exc}"
                    state[ep] = "failed"
            for fut in [f for f in comp_futs if f.done()]:
                ep = comp_futs.pop(fut)
                try:
                    fut.result()
                    state[ep] = "done"
                except Exception as exc:
                    failed[ep] = f"compose: {type(exc).__name__}: {exc}"
                    state[ep] = "failed"
            if not released and not qwen_futs and state and args.stage != "qwen":
                released = True
                if args.release_qwen_gpus:  # Qwen is done for this run: give its GPU(s) to SAM3
                    log("all Qwen answers cached; stopping Qwen and adding SAM3 on GPU(s) "
                        + ",".join(args.release_qwen_gpus))
                    subprocess.run(["bash", str(AUTO_MASK / "stop_services.sh")], check=False)
                    time.sleep(20)
                    procs += launch_sam3_workers(work, args.release_qwen_gpus, fresh=False)
            if not closed and not qwen_futs and not prep_futs and not to_prep:
                (work.queue / "QUEUE_CLOSED").touch()
                closed = True
            counts = collections.Counter(state.values())
            if time.time() - last_report > args.report_every or counts["done"] + counts["failed"] == len(state):
                last_report = time.time()
                el = time.time() - t_start
                log(f"[{el / 60:.1f} min] " + " ".join(f"{k}={counts[k]}" for k in
                    ("qwen", "sam", "judge", "compose", "done", "failed")) + f" | queued SAM3 phrase-passes={n_sam_phr}"
                    + (f" | judged objects={judge_tot['objects']} rejected={judge_tot['rejected_final']} "
                       f"calls={judge_tot['calls']}" if args.judge else ""))
            if counts["done"] + counts["failed"] == len(state):
                break
            if procs and "sam" in counts and all(p.poll() is not None for p in procs):
                raise RuntimeError("all SAM3 workers exited while episodes are still waiting for masks")
            time.sleep(1.0)
    finally:
        (work.queue / "QUEUE_CLOSED").touch()
        for pool in (qwen_pool, req_pool, ep_pool, judge_pool, chain_pool, cpu_pool):
            pool.shutdown(cancel_futures=True)
        for p in procs:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.terminate()
    for ep, msg in sorted(failed.items()):
        log(f"FAILED ep={ep}: {msg.strip().splitlines()[-1] if msg.strip() else msg}")
    write_json(work.queue / "failed_episodes.json", {str(k): v for k, v in sorted(failed.items())}, indent=1)
    return sorted(failed)


# ----------------------------------------------------------------------------- stage 4: assemble
def parse_episodes(spec: str, n: int) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [e for e in out if not 0 <= e < n]
    if bad:
        raise SystemExit(f"episode indices out of range: {bad[:5]}")
    return sorted(set(out))


def parse_shard(spec: str) -> tuple[int, int]:
    parts = spec.split("/")
    if len(parts) != 2 or not all(part.isdecimal() for part in parts):
        raise SystemExit(f"bad --shard {spec!r}; expected R/N")
    r, n = map(int, parts)
    if not 0 <= r < n:
        raise SystemExit(f"bad --shard {spec!r}; expected 0 <= R < N")
    return r, n


def select_episodes(args, src: Source) -> list[int]:
    n = len(src.eps)
    eps = parse_episodes(args.episodes, n) if args.episodes else list(range(n))
    if args.per_task:
        seen = collections.Counter()
        keep = []
        for e in eps:
            t = src.task_name(e)
            if seen[t] < args.per_task:
                seen[t] += 1
                keep.append(e)
        eps = keep
    if args.shard:
        eps = [e for e in eps if e % args.shard[1] == args.shard[0]]
    return eps


def cam_keys(mode: str) -> list[str]:
    return [CAM] if mode == "single" else [f"{CAM}_{i}" for i in range(1, N_MULTI + 1)]


def image_stats_type():
    nested = pa.list_(pa.list_(pa.list_(pa.float64())))
    return {k: (pa.list_(pa.int64()) if k == "count" else nested)
            for k in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")}


def aggregate_from_table(table: pa.Table) -> dict:
    """meta/stats.json from per-episode stats columns (lerobot aggregate_feature_stats)."""
    feats = collections.defaultdict(dict)
    for name in table.column_names:
        if name.startswith("stats/"):
            feat, stat = name[len("stats/"):].rsplit("/", 1)
            feats[feat][stat] = name
    out = {}
    for feat, cols in feats.items():
        per_ep = [dict() for _ in range(table.num_rows)]
        for stat, col in cols.items():
            for i, v in enumerate(table.column(col).to_pylist()):
                per_ep[i][stat] = np.asarray(v, dtype=np.int64 if stat == "count" else np.float64)
        agg = cs.aggregate_feature_stats(per_ep)
        out[feat] = {k: np.asarray(v).tolist() for k, v in agg.items()}
    return out


def video_layout(src: Source, episodes: list[int], key: str, subset: bool):
    """-> ({(chunk, file): [ep, ...]}, {ep: (chunk, file, from_ts, to_ts)}) for a source video key.

    Full runs keep the source layout (so cam outputs reuse from/to_timestamp verbatim);
    subset runs pack every episode into chunk-000/file-000.
    """
    files, where = collections.defaultdict(list), {}
    if subset:
        t = 0
        for ep in episodes:
            n = src.n_video_frames(ep, key)
            files[(0, 0)].append(ep)
            where[ep] = (0, 0, t / src.fps, (t + n) / src.fps)
            t += n
        return dict(files), where
    p = f"videos/{key}/"
    for ep in sorted(episodes, key=lambda e: (src.eps.loc[e, p + "chunk_index"], src.eps.loc[e, p + "file_index"],
                                              src.eps.loc[e, p + "from_timestamp"])):
        r = src.eps.loc[ep]
        cf_ = (int(r[p + "chunk_index"]), int(r[p + "file_index"]))
        files[cf_].append(ep)
        where[ep] = (*cf_, float(r[p + "from_timestamp"]), float(r[p + "to_timestamp"]))
    return dict(files), where


def assemble(args, src: Source, work: Work, episodes: list[int]) -> None:
    mode, out_final = args.mode, args.out
    subset = len(episodes) != len(src.eps)
    done = {}
    missing = []
    for ep in episodes:
        q = load_qwen(work.q(ep), src, ep, mode)
        if q is None or not done_ok(work, ep, q, args.signature):
            missing.append(ep)
        else:
            done[ep] = json.loads(work.done_json(ep).read_text())
    if missing:
        raise SystemExit(f"{len(missing)} episodes are not finished (e.g. {missing[:10]}); re-run --stage all")
    if out_final.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_final} exists; pass --overwrite to replace it")
    out = out_final.with_name(out_final.name + ".partial")
    shutil.rmtree(out, ignore_errors=True)
    (out / "meta").mkdir(parents=True)
    new_id = {ep: i for i, ep in enumerate(episodes)} if subset else {ep: ep for ep in episodes}
    keys = cam_keys(mode)
    other_vids = [k for k, v in src.info["features"].items() if v["dtype"] == "video" and k != CAM]
    vpath = src.info["video_path"]
    pool = cf.ProcessPoolExecutor(args.compose_workers, mp_context=mp.get_context("spawn"),
                                  initializer=quiet_stderr, initargs=(str(work.logs / "encode.log"),))
    t0 = time.time()
    try:
        # ---- cam outputs: concat per-episode segments in the cam_high layout
        files, where = video_layout(src, episodes, CAM, subset)
        futs = []
        for slot, key in enumerate(keys):
            for (c, f), eps in files.items():
                segs = [done[ep]["groups"][done[ep]["slot_group"][slot]]["file"] for ep in eps]
                n = sum(done[ep]["n_frames"] for ep in eps)
                dst = out / vpath.format(video_key=key, chunk_index=c, file_index=f)
                futs.append(pool.submit(concat_segments, segs, str(dst), n))
        # ---- other cameras: copied (full) or re-encoded per episode (subset)
        other_where = {}
        for key in other_vids:
            ofiles, other_where[key] = video_layout(src, episodes, key, subset)
            if not subset:
                for (c, f) in ofiles:
                    rel = vpath.format(video_key=key, chunk_index=c, file_index=f)
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src.root / rel, out / rel)
                continue
            seg_dir = work.root / "subset_other" / key
            seg_dir.mkdir(parents=True, exist_ok=True)
            segs = [str(seg_dir / f"{ep:06d}.mp4") for ep in episodes]
            for ep, s in zip(episodes, segs):
                futs.append(pool.submit(reencode_episode, str(src.root), ep, key, s))
        for fut in cf.as_completed(futs):
            fut.result()
        if subset:  # concat after all re-encodes finished
            cfuts = []
            for key in other_vids:
                segs = [str(work.root / "subset_other" / key / f"{ep:06d}.mp4") for ep in episodes]
                n = sum(src.n_video_frames(ep, key) for ep in episodes)
                cfuts.append(pool.submit(concat_segments, segs,
                                         str(out / vpath.format(video_key=key, chunk_index=0, file_index=0)), n))
            for fut in cf.as_completed(cfuts):
                fut.result()
    finally:
        pool.shutdown(cancel_futures=True)
    log(f"videos written in {(time.time() - t0) / 60:.1f} min")
    write_sam_masks(out, done, new_id)

    # ---- data parquet
    if subset:
        tables = []
        for (c, f) in sorted({(int(src.eps.loc[e, "data/chunk_index"]), int(src.eps.loc[e, "data/file_index"]))
                              for e in episodes}):
            path = src.root / src.info["data_path"].format(chunk_index=c, file_index=f)
            t = pq.read_table(path)
            tables.append(t.filter(pc.is_in(t.column("episode_index"), pa.array(episodes, pa.int64()))))
        data = pa.concat_tables(tables).combine_chunks()
        order = np.lexsort((data.column("frame_index").to_numpy(),
                            [new_id[e] for e in data.column("episode_index").to_pylist()]))
        data = data.take(pa.array(order))
        meta_schema = pq.read_schema(path).metadata
        data = data.set_column(data.schema.get_field_index("episode_index"), "episode_index",
                               pa.array([new_id[e] for e in data.column("episode_index").to_pylist()], pa.int64()))
        data = data.set_column(data.schema.get_field_index("index"), "index",
                               pa.array(np.arange(data.num_rows, dtype=np.int64)))
        data = data.replace_schema_metadata(meta_schema)
        dpath = out / src.info["data_path"].format(chunk_index=0, file_index=0)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data, dpath)
    else:
        for path in sorted((src.root / "data").glob("chunk-*/file-*.parquet")):
            dst = out / path.relative_to(src.root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
    write_meta(args, src, work, episodes, done, new_id, where, other_where, out, subset)
    if out_final.exists():
        shutil.rmtree(out_final)
    os.replace(out, out_final)
    log(f"wrote {out_final} ({len(episodes)} episodes, {len(keys)} cam_high key(s))")


def link_or_copy(src_path: str, dst: Path) -> None:
    try:
        os.link(src_path, dst)  # same filesystem: no extra space (work files are never rewritten in place)
    except OSError:
        shutil.copy2(src_path, dst)


def write_sam_masks(out: Path, done: dict, new_id: dict) -> None:
    """Final masks: ``sam_masks/episode_<new id>/g<NNN>.mkv`` (one per phrase group, FFV1 gray).

    ``sam_masks/index.parquet`` (written in write_meta) maps (episode, video_key) -> file.
    """
    jobs = []
    for ep, d in done.items():
        edir = out / "sam_masks" / f"episode_{new_id[ep]:06d}"
        edir.mkdir(parents=True, exist_ok=True)
        for g, grp in enumerate(d["groups"]):
            jobs.append((grp["mask_file"], edir / f"g{g:03d}.mkv"))
    with cf.ThreadPoolExecutor(32) as pool:
        list(pool.map(lambda j: link_or_copy(*j), jobs))
    log(f"sam_masks: {len(jobs)} mask videos")


def write_meta(args, src, work, episodes, done, new_id, where, other_where, out: Path, subset: bool) -> None:
    mode = args.mode
    keys = cam_keys(mode)
    rows = pa.array([int(e) for e in episodes], pa.int64())
    et = src.ep_table.take(rows)
    lengths = et.column("length").to_numpy()

    def setcol(t, name, arr):
        i = t.schema.get_field_index(name)
        return t.set_column(i, name, arr) if i >= 0 else t.append_column(name, arr)

    if subset:
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        zeros = pa.array(np.zeros(len(episodes), np.int64))
        et = setcol(et, "episode_index", pa.array([new_id[e] for e in episodes], pa.int64()))
        et = setcol(et, "dataset_from_index", pa.array(starts))
        et = setcol(et, "dataset_to_index", pa.array(starts + lengths))
        for name in ("data/chunk_index", "data/file_index", "meta/episodes/chunk_index", "meta/episodes/file_index"):
            et = setcol(et, name, zeros)
        for key, w in other_where.items():
            p = f"videos/{key}/"
            et = setcol(et, p + "chunk_index", zeros)
            et = setcol(et, p + "file_index", zeros)
            et = setcol(et, p + "from_timestamp", pa.array([w[e][2] for e in episodes], pa.float64()))
            et = setcol(et, p + "to_timestamp", pa.array([w[e][3] for e in episodes], pa.float64()))
        # data stats of a subset are recomputed below from the kept episodes
    # ---- cam columns: drop the source cam_high, add the new key(s)
    drop = [n for n in et.column_names if n.startswith((f"videos/{CAM}/", f"stats/{CAM}/"))]
    et = et.drop_columns(drop)
    stypes = image_stats_type()
    for slot, key in enumerate(keys):
        p = f"videos/{key}/"
        et = et.append_column(p + "chunk_index", pa.array([where[e][0] for e in episodes], pa.int64()))
        et = et.append_column(p + "file_index", pa.array([where[e][1] for e in episodes], pa.int64()))
        et = et.append_column(p + "from_timestamp", pa.array([where[e][2] for e in episodes], pa.float64()))
        et = et.append_column(p + "to_timestamp", pa.array([where[e][3] for e in episodes], pa.float64()))
        st = [done[e]["groups"][done[e]["slot_group"][slot]]["stats"] for e in episodes]
        for stat, typ in stypes.items():
            et = et.append_column(f"stats/{key}/{stat}", pa.array([s[stat] for s in st], typ))
    # ---- tasks: multi stores the 100 seen instructions per episode (slot i <-> cam_high_{i+1})
    tasks_df = pd.read_parquet(src.root / "meta" / "tasks.parquet")
    if mode == "multi":
        instr = {e: json.loads(work.q(e).read_text())["instructions"] for e in episodes}
        et = setcol(et, "tasks", pa.array([instr[e] for e in episodes], pa.list_(pa.string())))
        new = [t for e in episodes for t in instr[e] if t not in tasks_df.index]
        new = list(dict.fromkeys(new))
        if new:  # source task_index values stay valid; new instructions are appended
            add = pd.DataFrame({"task_index": np.arange(len(tasks_df), len(tasks_df) + len(new))}, index=new)
            tasks_df = pd.concat([tasks_df, add])
    if subset:  # keep only tasks referenced by the kept episodes; data task_index is remapped
        used = list(dict.fromkeys(t for ts in et.column("tasks").to_pylist() for t in ts))
        data_path = out / src.info["data_path"].format(chunk_index=0, file_index=0)
        data = pq.read_table(data_path)
        old_names = tasks_df.sort_values("task_index").index.tolist()
        data_tasks = [old_names[i] for i in data.column("task_index").to_pylist()]
        used = list(dict.fromkeys(data_tasks + used))
        tasks_df = pd.DataFrame({"task_index": np.arange(len(used), dtype=np.int64)}, index=used)
        remap = {t: i for i, t in enumerate(used)}
        data = data.set_column(data.schema.get_field_index("task_index"), "task_index",
                               pa.array([remap[t] for t in data_tasks], pa.int64()))
        pq.write_table(data, data_path)
    tasks_df.index.name = None
    tasks_df.to_parquet(out / "meta" / "tasks.parquet")
    # ---- episodes parquet (same chunk/file split as the source meta)
    mc, mf = et.column("meta/episodes/chunk_index").to_numpy(), et.column("meta/episodes/file_index").to_numpy()
    for c, f in sorted(set(zip(mc.tolist(), mf.tolist()))):
        sel = np.nonzero((mc == c) & (mf == f))[0]
        dst = out / "meta" / "episodes" / f"chunk-{c:03d}" / f"file-{f:03d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(et.take(pa.array(sel)), dst)
    # ---- stats.json
    if subset:
        stats = aggregate_from_table(et)
    else:
        stats = json.loads((src.root / "meta" / "stats.json").read_text())
        stats.pop(CAM, None)
        stats.update(aggregate_from_table(et.select([n for n in et.column_names
                                                     if any(n.startswith(f"stats/{k}/") for k in keys)])))
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    # ---- info.json
    info = json.loads(json.dumps(src.info))
    features = {}
    for k, v in info["features"].items():  # new key(s) take cam_high's place in the order
        if k == CAM:
            rendered_feature = json.loads(json.dumps(v))
            # The repaired source may be H.264, while encode_video writes AV1.
            rendered_feature.setdefault('info', {}).update({'video.codec': 'av1', 'video.pix_fmt': 'yuv420p'})
            features.update({key: json.loads(json.dumps(rendered_feature)) for key in keys})
        else:
            features[k] = v
    info["features"] = features
    info["total_episodes"] = len(episodes)
    info["total_frames"] = int(lengths.sum())
    info["total_tasks"] = int(len(tasks_df))
    info["splits"] = {"train": f"0:{len(episodes)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    # ---- provenance: Qwen phrases per (episode, slot)
    prov = []
    for e in episodes:
        q = json.loads(work.q(e).read_text())
        d = done[e]
        for slot, (ins, ps) in enumerate(zip(q["instructions"], q["phrases"])):
            prov.append({"episode_index": new_id[e], "source_episode_index": e, "video_key": keys[slot],
                         "instruction": ins, "phrases": ps, "qwen_raw": q["raw"][slot],
                         "qwen_error": q["errors"][slot],
                         "keep_frac": d["groups"][d["slot_group"][slot]]["keep_frac"],
                         "keep_frac_unjudged": d["groups"][d["slot_group"][slot]]["keep_frac_unjudged"],
                         "mask_file": f"sam_masks/episode_{new_id[e]:06d}/g{d['slot_group'][slot]:03d}.mkv"})
    pd.DataFrame(prov).to_parquet(out / "meta" / "sam_prompts.parquet")
    # ---- final masks index + judge log
    idx_rows = []
    for e in episodes:
        d = done[e]
        for slot, g in enumerate(d["slot_group"]):
            grp = d["groups"][g]
            idx_rows.append({"episode_index": new_id[e], "source_episode_index": e, "video_key": keys[slot],
                             "group": g, "file": f"sam_masks/episode_{new_id[e]:06d}/g{g:03d}.mkv",
                             "n_frames": d["n_frames"], "keep_frac": grp["keep_frac"],
                             "keep_frac_unjudged": grp["keep_frac_unjudged"], "phrases": grp["phrases"]})
    pd.DataFrame(idx_rows).to_parquet(out / "sam_masks" / "index.parquet")
    jrows = []
    if args.judge:
        for e in episodes:
            used = set(distinct_phrases(json.loads(work.q(e).read_text())))
            for phrase, ent in json.loads(work.judge_json(e).read_text())["phrases"].items():
                if phrase not in used:
                    continue
                for obj_id, vs in ent["verdicts"].items():
                    for v in vs:
                        jrows.append({"episode_index": new_id[e], "source_episode_index": e, "phrase": phrase,
                                      "instruction": ent["instruction"], "obj_id": int(obj_id),
                                      "frame_index": v["frame_index"], "area": v["area"], "p_yes": v["p_yes"],
                                      "keep": v["keep"], "error": v["error"]})
        pd.DataFrame(jrows, columns=["episode_index", "source_episode_index", "phrase", "instruction", "obj_id",
                                     "frame_index", "area", "p_yes", "keep", "error"]
                     ).to_parquet(out / "sam_masks" / "judge_log.parquet")
    style = args.style_obj
    (out / "meta" / "sam_info.json").write_text(json.dumps({
        "mode": mode, "source": str(src.root), "prompt_version": PROMPT_VERSION, "format": FORMAT,
        "style": style.to_dict(), "outside_scale": style.outside_scale,
        "judge": args.judge_sig if args.judge else None,
        "sam3": SAM3_ARGS, "encode": dict(ENCODE_OPTIONS, vcodec="libsvtav1", pix_fmt="yuv420p"),
        "masks": dict(MASK_CODEC, dir="sam_masks", index="sam_masks/index.parquet",
                      judge_log="sam_masks/judge_log.parquet" if args.judge else None,
                      note="final (judged, unioned) mask per video key; frame i <-> frame i of the "
                           "episode's video segment; same frame count"),
        "video_keys": keys, "subset": subset,
        "note": ("episodes.tasks[i] is the instruction of video key cam_high_{i+1}" if mode == "multi"
                 else "cam_high rendered with each episode's own instruction"),
    }, indent=2))


# ----------------------------------------------------------------------------- CLI
def qwen_alive(url: str) -> bool:
    base = url.split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", choices=["single", "multi"], required=True)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, help="output dataset (default: <src>_sam_<mode>)")
    ap.add_argument("--work", type=Path, default=AUTO_MASK / "outputs" / "lerobot_sam_work_judged",
                    help="resumable cache (Qwen answers, SAM3 masks, verdicts, segments); shared by both modes")
    ap.add_argument("--stage", choices=["all", "qwen", "masks", "assemble"], default="all",
                    help="qwen = only cache Qwen phrases; masks = up to per-episode segments")
    ap.add_argument("--episodes", default="", help="subset, e.g. '0-9,120' (default: all)")
    ap.add_argument("--per-task", type=int, default=0, help="keep the first N episodes of each task")
    ap.add_argument("--shard", default="", help="'R/N': only episodes with ep %% N == R (multi-node; "
                    "stages qwen/masks only, assemble runs unsharded on one node)")
    ap.add_argument("--qwen-url", default=DEFAULT_QWEN_URL,
                    help="comma-separated vLLM endpoints; requests are spread round-robin")
    ap.add_argument("--sam3-gpus", default="1,2,3,4,5,6,7", help="'' = do not launch workers")
    ap.add_argument("--release-qwen-gpus", default="",
                    help="once every Qwen answer is cached, run stop_services.sh and start SAM3 workers "
                         "on these GPUs (e.g. '0', the Qwen GPU)")
    ap.add_argument("--qwen-concurrency", type=int, default=96)
    ap.add_argument("--qwen-episodes", type=int, default=0,
                    help="episodes in flight in Qwen (default: single = --qwen-concurrency, multi = 8)")
    ap.add_argument("--prep-threads", type=int, default=8)
    ap.add_argument("--max-queued", type=int, default=32, help="max episodes with PNG frames waiting for SAM3")
    ap.add_argument("--compose-workers", type=int, default=48)
    ap.add_argument("--report-every", type=float, default=60.0)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    g = ap.add_argument_group("Qwen mask judge (sam3/tools/sam3_mask_judge.py)")
    g.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True,
                   help="judge every SAM3 object with Qwen (--no-judge: keep all, the old behaviour)")
    g.add_argument("--judge-url", default="", help="comma-separated vLLM endpoints (default: --qwen-url)")
    g.add_argument("--judge-model", default=MODEL_NAME, help="served model name")
    g.add_argument("--judge-threshold", type=float, default=0.5, help="keep an object iff P(yes) >= this")
    g.add_argument("--judge-outside-scale", type=float, default=0.2, help="outside-mask brightness in image 2")
    g.add_argument("--judge-regrow", type=float, default=2.0,
                   help="re-judge a rejected id once its area is this x its last-verdict area (0 = off)")
    g.add_argument("--judge-regrow-min-area", type=float, default=0.002,
                   help="floor (fraction of frame) for the reference area of --judge-regrow")
    g.add_argument("--judge-timeout", type=float, default=120.0, help="seconds per Qwen request")
    g.add_argument("--judge-concurrency", type=int, default=64, help="object request chains in flight")
    g.add_argument("--judge-episodes", type=int, default=8, help="episodes in the judge stage at once")
    sg = ap.add_argument_group("rendering of the cam_high output (see mask_style.py, composite_lerobot.py)")
    add_style_args(sg)
    args = ap.parse_args()
    args.style_obj = style_from_args(args)
    args.sam3_gpus = [g for g in args.sam3_gpus.split(",") if g.strip()]
    args.qwen_urls = [u.strip() for u in args.qwen_url.split(",") if u.strip()]
    args.release_qwen_gpus = [g for g in args.release_qwen_gpus.split(",") if g.strip()]
    args.qwen_episodes = args.qwen_episodes or (args.qwen_concurrency if args.mode == "single" else 8)
    if args.judge and args.release_qwen_gpus:
        raise SystemExit("--release-qwen-gpus stops Qwen, but the judge needs it until the end; use --no-judge")
    judge_urls = [u.strip() for u in (args.judge_url or args.qwen_url).split(",") if u.strip()]
    args.judges = [QwenMaskJudge(u, model=args.judge_model, threshold=args.judge_threshold,
                                 outside_scale=args.judge_outside_scale, timeout=args.judge_timeout,
                                 retries=4) for u in judge_urls] if args.judge else []
    args.judge_sig = judge_signature(args.judges[0], args) if args.judge else None
    args.signature = {"format": FORMAT, "judge": args.judge_sig, "style": args.style_obj.to_dict()}
    if args.out is None:
        default_style = MaskStyle()
        tag = "" if args.style_obj == default_style else "_" + args.style_obj.tag()
        variant = "judged" if args.judge else "nojudge"
        args.out = args.src.with_name(f"{args.src.name}_sam_{variant}_{args.mode}{tag}")
    if args.shard:
        r, n = parse_shard(args.shard)
        if args.stage in ("all", "assemble"):
            raise SystemExit("--shard needs --stage qwen or masks; assemble once without --shard")
        args.shard = (r, n)
    for p in (args.out, args.work):
        rp = p.resolve()
        if rp.is_relative_to(RAW_ROOT.resolve()) or rp == args.src.resolve():
            raise SystemExit(f"refusing to write to {p}")

    src = Source(args.src)
    work = Work(args.work, args.mode, f"rank{args.shard[0]}of{args.shard[1]}" if args.shard else "local",
                "judged" if args.judge else "nojudge")
    episodes = select_episodes(args, src)
    log(f"mode={args.mode} episodes={len(episodes)} src={args.src} out={args.out} work={args.work}")
    if args.stage in ("all", "qwen", "masks"):
        need_qwen = any(load_qwen(work.q(ep), src, ep, args.mode) is None for ep in episodes)
        urls = list(args.qwen_urls) if need_qwen else []
        if args.judge and args.stage != "qwen":
            urls += [j.url.split("/v1/")[0] + "/v1/chat/completions" for j in args.judges]
        dead = [u for u in dict.fromkeys(urls) if not qwen_alive(u)]
        if dead:
            raise SystemExit(f"Qwen is not reachable at {dead}; start it with vllm_space/start_qwen.sh or use process_data.sh")
        failed = run_masks(args, src, work, episodes)
        if failed:
            log(f"{len(failed)} episodes failed (see {work.queue / 'failed_episodes.json'}); "
                "re-run the same command to retry them")
            return 1
    if args.stage in ("all", "assemble"):
        assemble(args, src, work, episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
