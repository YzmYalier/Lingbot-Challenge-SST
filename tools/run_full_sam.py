#!/usr/bin/env python3
"""Full-data adapter for memory.md section 5; never consumes a previous SAM dataset."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, path, runtime_env, writable
from services import qwen_url
import rebuild_robotwin_sam as R
from sam_rebuild_progress import Progress, observe_assembly


def lock_file(stack, file, mode):
    handle = stack.enter_context(file.open("a"))
    fcntl.flock(handle, mode)


def bind_work(args, signature):
    """A work directory cannot silently mix inputs, prompt code or model versions."""
    root = base_path()
    def key(p):
        return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
    files = [root / "base.json", path("mapping"), path("sam_checkpoint"),
             path("qwen_model") / "config.json", path("qwen_model") / "model.safetensors.index.json"]
    # Weight contents are covered by the submission inventory; stat catches replacement.
    weights = {key(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
               for p in [path("sam_checkpoint"), *path("qwen_model").glob("*.safetensors")]}
    files = [p for p in files if p != path("sam_checkpoint") and p.name != "base.json"]
    files += [*args.raw.rglob("repair_manifest.json"), *args.raw.glob("meta/**/*.json"),
              *args.raw.glob("meta/**/*.parquet"), *args.raw.glob("data/*/*.parquet"),
              *path("instructions").rglob("*.json")]
    for folder in ("auto_mask", "tools", "sam3/tools"):
        files.extend((root / folder).glob("*.py"))
    files.append(root / "vllm_space/sam3_prompt.py")
    hashes = {key(p): R.file_hash(p) for p in sorted(set(files))}
    expected = {"version": 1, "base_path": str(root), "raw": str(args.raw),
                "output": str(args.out), "signature": signature, "sha256": hashes, "weights": weights}
    stamp = args.work / "full_pipeline_manifest.json"
    with (args.work / "manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if stamp.exists():
            if json.loads(stamp.read_text()) != expected:
                raise ValueError("Work inputs/code/model paths changed. Use a new paths.work; do not reuse old masks.")
        else:
            if any(p.name not in ("manifest.lock", "pipeline.lock", "locks") for p in args.work.iterdir()):
                raise ValueError("Refusing a work directory with unrecognized pre-existing caches")
            R.write_json(stamp, expected)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["masks", "assemble"], required=True)
    ap.add_argument("--shard", default="")
    ap.add_argument("--qwen-url", default=qwen_url())
    ap.add_argument("--sam-gpus", default=config()["processing"]["sam_gpus"])
    opt = ap.parse_args()
    if opt.stage == "assemble" and opt.shard:
        ap.error("Assemble must cover all episodes without --shard")
    args = SimpleNamespace(raw=path("raw"), out=writable(path("sam_output")),
                           work=writable(path("work") / "sam-full-work"),
                           pipeline=base_path() / "auto_mask/pipeline_lerobot.py",
                           qwen_url=opt.qwen_url, sam3_gpus=opt.sam_gpus,
                           compose_workers=int(config()["processing"]["compose_workers"]), report_every=30)
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}")
    if not json.loads((args.raw / "repair_report.json").read_text())["complete"]:
        raise ValueError("Raw repair is incomplete")
    env = runtime_env("full-sam")
    os.environ.update(env)
    tempfile.tempdir = env["TMPDIR"]
    p = R.import_pipeline(args.pipeline)
    if (p.FORMAT != "objects-judge-v1" or p.PROMPT_VERSION != "robotwin-sam3-v3.5"
            or p.SAM3_ARGS != {"score_threshold_detection": 0.3, "new_det_thresh": 0.5}):
        raise ValueError("Pipeline semantics differ from the documented dataset")
    p.AUTO_MASK = writable(path("runtime") / socket.gethostname() / "full-sam")
    p.concat_segments = R.copy_episode_segment
    p.ENCODE_OPTIONS = {**p.ENCODE_OPTIONS, "svtav1-params": "lp=2"}
    shard = p.parse_shard(opt.shard) if opt.shard else None
    tag = f"rank{shard[0]}of{shard[1]}" if shard else "local"
    judge = p.QwenMaskJudge(args.qwen_url, model="Qwen/Qwen3.8-27B", threshold=0.5, outside_scale=0.2)
    sig = p.judge_signature(judge, SimpleNamespace(judge_regrow=2.0, judge_regrow_min_area=0.002))
    signature = {"format": p.FORMAT, "judge": sig, "style": p.MaskStyle("outline", 0.1, 2).to_dict()}
    args.work.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        mode = fcntl.LOCK_EX if opt.stage == "assemble" or not shard else fcntl.LOCK_SH
        lock_file(stack, args.work / "pipeline.lock", mode | fcntl.LOCK_NB)
        bind_work(args, signature)
        (args.work / "locks").mkdir(exist_ok=True)
        lock_file(stack, args.work / "locks" / (tag + ".lock"), fcntl.LOCK_EX | fcntl.LOCK_NB)
        src = p.Source(args.raw)
        if len(src.eps) != 2500 or int(src.eps["length"].sum()) != 548893:
            raise ValueError("Only the complete 2500-episode framefixed dataset is accepted")
        work = p.Work(args.work, "multi", tag)
        argv = R.pipeline_argv(args, {"judge": sig}, opt.stage)
        if opt.shard:
            argv += ["--shard", opt.shard]
        if opt.stage == "masks":
            # NONE is valid: the no-phrase episode still needs an empty judge document.
            for ep in range(len(src.eps)):
                if shard and ep % shard[1] != shard[0]:
                    continue
                if not work.judge_json(ep).exists():
                    R.write_json(work.judge_json(ep), {"episode_index": ep, "phrases": {}})
            R.call_pipeline(p, argv, encoder_threads=2)
        else:
            with Progress(args.work / "full-assembly-progress", 30) as progress:
                progress.stage("validate_masks", len(src.eps))
                R.check_complete(p, src, work, signature, progress)
                for ep in range(len(src.eps)):
                    q = json.loads(work.q(ep).read_text())
                    if any(q["errors"]):
                        raise ValueError(f"Episode {ep}: cached Qwen extraction error")
                    doc = json.loads(work.judge_json(ep).read_text())
                    for entry in doc["phrases"].values():
                        for verdicts in entry["verdicts"].values():
                            if any(v["error"] is not None or v["p_yes"] is None for v in verdicts):
                                raise ValueError(f"Episode {ep}: cached Qwen judge error")
                others = sum(v["dtype"] == "video" and k != p.CAM for k, v in src.info["features"].items())
                total = len(src.eps) * (len(p.cam_keys("multi")) + others)
                with observe_assembly(p, progress, total):
                    R.call_pipeline(p, argv, encoder_threads=2)


if __name__ == "__main__":
    main()
