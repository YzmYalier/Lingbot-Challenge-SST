#!/usr/bin/env python3
"""Rebuild framefixed raw and full judged SAM multi from the packaged inputs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, interpreter, path, runtime_env, writable
from services import Service, check_qwen, qwen_command, qwen_url
from prepare_data_inputs import check_prepared_inputs


def file_hash(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(file, obj):
    file = writable(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_name(file.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temp.replace(file)


def check_inputs():
    for name, version in (("zip_v21", "v2.1"), ("zip_v30", "v3.0")):
        with zipfile.ZipFile(path(name)) as archive:
            infos = [x for x in archive.namelist() if x.endswith("meta/info.json")]
            if len(infos) != 1:
                raise ValueError(f"{name}: expected one dataset")
            info = json.loads(archive.read(infos[0]))
            if (info["codebase_version"], info["total_episodes"], info["total_frames"], info["fps"]) != (version, 2500, 548893, 15):
                raise ValueError(f"{name}: unexpected dataset metadata")
    check_prepared_inputs(require_report=False)
    print("Inputs OK: 2500 episodes, 548893 frames, 250000 seen instruction slots.", flush=True)


def extract_reference():
    target = writable(path("work") / "reference_v30")
    archive_path = path("zip_v30")
    fingerprint = {"sha256": file_hash(archive_path), "size": archive_path.stat().st_size}
    if target.exists():
        stamp = json.loads((target / "extraction_manifest.json").read_text())
        if stamp["archive"] != fingerprint or any(file_hash(target / k) != v for k, v in stamp["files"].items()):
            raise ValueError("Extracted v30 reference differs from its ZIP; use a new work path")
        return target
    temp = writable(target.with_name(target.name + ".partial"))
    temp.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        infos = [x for x in archive.namelist() if x.endswith("meta/info.json")]
        if len(infos) != 1:
            raise ValueError("Expected one v30 dataset in ZIP")
        prefix = infos[0][:-len("meta/info.json")]
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith(prefix):
                continue
            relative = entry.filename[len(prefix):]
            if not relative.startswith(("data/", "meta/")):
                continue
            if ".." in PurePosixPath(relative).parts:
                raise ValueError(f"Unsafe ZIP member: {entry.filename}")
            out = writable(temp / relative)
            if not out.is_relative_to(temp):
                raise ValueError(f"ZIP member escapes destination: {entry.filename}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, out.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            hashes[relative] = file_hash(out)
    write_json(temp / "extraction_manifest.json", {"archive": fingerprint, "files": hashes})
    temp.rename(target)
    return target


def check_raw():
    raw = path("raw")
    report = json.loads((raw / "repair_report.json").read_text())
    info = json.loads((raw / "meta/info.json").read_text())
    if not report.get("complete") or report["videos_checked"] != 7500:
        raise ValueError("Raw repair is incomplete")
    if (info["total_episodes"], info["total_frames"]) != (2500, 548893):
        raise ValueError("Raw output is not the full dataset")
    if len(list((raw / "videos").rglob("*.mp4"))) != 7500:
        raise ValueError("Raw video files are missing")


def run(command, name, env):
    print(shlex.join(map(str, command)), flush=True)
    log_path = writable(path("work") / "logs" / (name + ".log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(list(map(str, command)), cwd=base_path(), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True, bufsize=1)
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
        finally:
            # Only this invocation's process group, including its SAM workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def check_environment(role, env):
    exe = interpreter(role)
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise FileNotFoundError(f"Configure interpreters.{role} in base.json or install environments/{role}: {exe}")
    imports = {
        "data": "import av,cv2,numpy,pandas,pyarrow,torch; import lerobot.datasets.compute_stats; av.codec.Codec('libsvtav1','w'); av.codec.Codec('ffv1','w')",
        "sam3": "import torch,sam3,cv2,numpy,einops,timm,psutil; from sam3.model_builder import build_sam3_video_predictor",
        "vllm": "import vllm; from vllm.entrypoints.cli.main import main",
    }
    subprocess.run([str(exe), "-B", "-c", imports[role]], cwd=base_path(), env=env, check=True)
    print(f"Environment OK: {role}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["all", "inputs", "raw", "masks", "assemble", "validate"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="Print paths and execution plan; do not write or start models")
    ap.add_argument("--check", action="store_true", help="Check inputs, models and selected environments; do not convert")
    ap.add_argument("--workers", type=int, default=config()["processing"]["workers"])
    ap.add_argument("--sam-gpus", default=config()["processing"]["sam_gpus"])
    ap.add_argument("--qwen-url", help="Existing Qwen endpoint; omit to start/stop a private local Qwen service")
    ap.add_argument("--shard", default="", help="R/N; only with --stage masks. Assemble once after all shards finish")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be positive")
    if args.stage == "inputs":
        if args.shard:
            ap.error("inputs is a single-writer CPU stage; do not pass --shard")
        command = [sys.executable, "-B", str(base_path() / "tools/prepare_data_inputs.py"),
                   "--workers", str(min(args.workers, 32))]
        command += ["--dry-run"] if args.dry_run else ["--check"] if args.check else []
        subprocess.run(command, cwd=base_path(), check=True)
        return
    if args.shard and args.stage != "masks":
        ap.error("--shard is only supported for masks")
    if args.shard:
        try:
            rank, count = map(int, args.shard.split("/"))
            assert 0 <= rank < count
        except (ValueError, AssertionError):
            ap.error("--shard must be R/N with 0 <= R < N")
    gpu_stage = args.stage in ("all", "masks")
    gpu_ids = args.sam_gpus.split(",")
    if gpu_stage and (not all(x.isdigit() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids)):
        ap.error("--sam-gpus must contain distinct numeric GPU IDs")
    # Validate output locations before even creating caches or extracting files.
    outputs = [writable(path(name)) for name in ("raw", "sam_output", "work", "runtime")]
    protected = [path(name) for name in ("zip_v21", "zip_v30", "mapping", "instructions", "sam_checkpoint", "qwen_model")]
    protected += [base_path() / name for name in ("auto_mask", "sam3", "vllm_space", "tools", "RoboTwin", "environments")]
    for i, dest in enumerate(outputs):
        for other in outputs[i+1:] + protected:
            if dest.is_relative_to(other) or other.is_relative_to(dest):
                raise ValueError(f"Output/cache overlaps another input or output: {dest}, {other}")
    print(json.dumps({"base_path": str(base_path()), "stage": args.stage,
                      "raw": str(path("raw")), "sam_output": str(path("sam_output")),
                      "work": str(path("work")), "sam_gpus": args.sam_gpus}, indent=2), flush=True)
    if args.dry_run:
        print("inputs: recover missing mapping/instructions from pinned public archives (CPU, exact numeric matching)")
        print("raw: extract v30 data/meta -> match full state/action -> copy and decode v21 videos")
        print("masks: all 2500 episodes -> Qwen v3.5 -> SAM3 -> object judge -> outline/0.1/2")
        print("assemble: validate all masks and Qwen errors -> publish 100 high + 2 wrist cameras")
        if gpu_stage and not args.qwen_url:
            print(shlex.join(qwen_command()))
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if not args.check and args.stage in ("all", "raw") and (
            not path("mapping").is_file() or not path("instructions").is_dir()):
        subprocess.run([sys.executable, "-B", str(base_path() / "tools/prepare_data_inputs.py"),
                        "--workers", str(min(args.workers, 32))], cwd=base_path(), check=True)
    check_inputs()
    env = runtime_env("process-data")
    check_environment("data", env)
    if gpu_stage:
        for file in (path("sam_checkpoint"), path("qwen_model") / "config.json"):
            if not file.is_file():
                raise FileNotFoundError(file)
        check_environment("sam3", env)
        if not args.qwen_url:
            check_environment("vllm", env)
            index = json.loads((path("qwen_model") / "model.safetensors.index.json").read_text())
            for name in set(index["weight_map"].values()):
                if not (path("qwen_model") / name).is_file():
                    raise FileNotFoundError(name)
    if args.check:
        return
    work = writable(path("work"))
    work.mkdir(parents=True, exist_ok=True)
    lock_name = "masks-" + args.shard.replace("/", "of") if args.shard else "process"
    with (work / (lock_name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data_py = str(interpreter("data"))
        if args.stage in ("all", "raw"):
            if path("raw").exists():
                check_raw()
                print("Complete raw already exists; preserving it.", flush=True)
            else:
                reference = extract_reference()
                run([data_py, "-B", base_path() / "tools/rebuild_robotwin_v30_from_v21.py",
                     "--source-v21", path("zip_v21"), "--reference-v30", reference,
                     "--out", path("raw"), "--workers", str(args.workers)], "raw", env)
                check_raw()
        if args.stage == "raw":
            return
        check_raw()
        if args.stage == "all" and path("sam_output").exists():
            print("SAM output already exists; validating without overwriting it.", flush=True)
        elif args.stage in ("all", "masks"):
            service = None
            try:
                url = args.qwen_url or qwen_url()
                if args.qwen_url:
                    check_qwen(url)
                else:
                    service = Service("qwen", qwen_command(), config()["qwen"]).start()
                command = [data_py, "-B", base_path() / "tools/run_full_sam.py", "--stage", "masks",
                           "--qwen-url", url, "--sam-gpus", args.sam_gpus]
                if args.shard:
                    command += ["--shard", args.shard]
                run(command, "sam-masks" + args.shard.replace("/", "of"), env)
            finally:
                if service:
                    service.stop()
        if args.stage in ("all", "assemble") and not path("sam_output").exists():
            run([data_py, "-B", base_path() / "tools/run_full_sam.py", "--stage", "assemble"], "sam-assemble", env)
        if args.stage in ("all", "assemble", "validate"):
            run([data_py, "-B", base_path() / "tools/validate_outputs.py"], "validation", env)


if __name__ == "__main__":
    main()
