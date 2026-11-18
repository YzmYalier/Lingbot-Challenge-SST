#!/usr/bin/env python3
"""Recover all RoboTwin mappings/instructions from pinned public source archives.

No models or images are decoded. Episode identity is the exact SHA256 of the
complete float32 state AND action trajectories, using the same representation as
rebuild_robotwin_v30_from_v21.signature. Never infer identity from task text or
archive order. Production publication always requires all 2500 original IDs.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, interpreter, path, runtime_env, writable
from download_public_weights import load_manifest
from tools.remote_zip import RangeReader

EPISODES, FRAMES, SEEN = 2500, 548893, 100
PARTS = ("left_arm_joint_states", "left_ee_joint_states",
         "right_arm_joint_states", "right_ee_joint_states")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(target, data):
    target = writable(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)


def checked_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or not p.parts:
        raise ValueError(f"Unsafe member path: {value}")
    return p


def instruction_relative(task, relative):
    if not re.fullmatch(r"[a-z0-9_]+", task):
        raise ValueError(f"Invalid task: {task}")
    p = checked_relative(relative)
    if len(p.parts) != 3 or p.parts[1] != "data" or p.suffix != ".hdf5":
        raise ValueError(f"Unexpected raw episode path: {relative}")
    return Path(task) / p.parts[0] / "instruction" / (p.stem + ".json")


def validate_instruction(data):
    doc = json.loads(data)
    seen = doc.get("seen")
    if not isinstance(seen, list) or len(seen) != SEEN:
        raise ValueError("Each episode must retain exactly 100 original seen instructions")
    if any(not isinstance(s, str) or not s.strip() or "{" in s or "}" in s for s in seen):
        raise ValueError("Instructions contain empty text or uninstantiated placeholders")
    # Do not deduplicate, sort, paraphrase or pad the list. Repetition is valid.
    return doc


def trajectory_signature(state, action):
    import numpy as np
    h = hashlib.sha256()
    arrays = [np.asarray(x, dtype="<f4") for x in (state, action)]
    if arrays[0].shape != arrays[1].shape or arrays[0].ndim != 2 or arrays[0].shape[1] != 14:
        raise ValueError("Expected matching (frames, 14) state/action arrays")
    if not len(arrays[0]):
        raise ValueError("Empty trajectory")
    for a in arrays:
        if not np.isfinite(a).all():
            raise ValueError("Non-finite numeric trajectory")
        h.update(np.asarray(a.shape, dtype="<i8").tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def hdf5_signature(data):
    import h5py
    import numpy as np
    arrays = []
    with h5py.File(io.BytesIO(data), "r") as f:
        for group in ("state", "action"):
            parts = []
            for field in PARTS:
                a = np.asarray(f[f"{group}/{field}"][()], dtype="<f4")
                if a.ndim == 1:
                    a = a[:, None]
                if a.ndim != 2:
                    raise ValueError(f"Unexpected shape for {group}/{field}: {a.shape}")
                parts.append(a)
            arrays.append(np.concatenate(parts, axis=1))
    return trajectory_signature(*arrays), len(arrays[0])


def read_cache(file, identity):
    checksum = file.with_suffix(file.suffix + ".sha256")
    if file.is_file() and checksum.is_file():
        data = file.read_bytes()
        if sha256(data) == checksum.read_text().strip():
            doc = json.loads(data)
            if doc.get("identity") == identity:
                return doc["records"]
    return None


def save_cache(file, identity, records):
    atomic_json(file, {"identity": identity, "records": records})
    file.with_suffix(file.suffix + ".sha256").write_text(sha256(file.read_bytes()) + "\n")


@contextmanager
def source_archive(entry, local, work):
    if local.is_file():
        if local.stat().st_size != entry["size"]:
            raise ValueError(f"Wrong local archive size: {local}")
        h = hashlib.sha256()
        with local.open("rb") as f:
            for data in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(data)
        if h.hexdigest() != entry["sha256"]:
            raise ValueError(f"Wrong local archive SHA256: {local}")
        with zipfile.ZipFile(local) as z:
            yield z
    else:
        with RangeReader(entry["url"], entry["size"], writable(work / "http")) as reader:
            with zipfile.ZipFile(reader) as z:
                # One request per compressed HDF5 instead of a separate request
                # for its local ZIP header. Never prefetch standalone videos.
                for member in z.infolist():
                    if member.filename.endswith(".hdf5"):
                        start = member.header_offset
                        aligned = (start // 65536) * 65536
                        if not (reader.cache / f"{aligned}-65536.bin").exists():
                            reader.member_prefetch[start] = min(entry["size"] - start,
                                                               member.compress_size + 16384)
                yield z


def reference_signatures(entry, work):
    import numpy as np
    import pyarrow.parquet as pq
    identity = {"schema": 1, "archive_sha256": entry["sha256"], "kind": "v30"}
    cache = work / "reference_signatures.json"
    cached = read_cache(cache, identity)
    if cached is not None:
        return cached
    grouped = defaultdict(list)
    with source_archive(entry, path("zip_v30"), work) as z:
        infos = [n for n in z.namelist() if n.endswith("meta/info.json")]
        if len(infos) != 1:
            raise ValueError("Expected one LeRobot dataset")
        info = json.loads(z.read(infos[0]))
        if (info["codebase_version"], info["total_episodes"], info["total_frames"], info["fps"]) != ("v3.0", EPISODES, FRAMES, 15):
            raise ValueError("Wrong reference dataset version or size")
        prefix = infos[0][:-len("meta/info.json")]
        names = sorted(n for n in z.namelist() if n.startswith(prefix + "data/") and n.endswith(".parquet"))
        for i, name in enumerate(names):
            table = pq.ParquetFile(io.BytesIO(z.read(name))).read(
                columns=["episode_index", "frame_index", "observation.state", "action"], use_threads=False)
            ep = table.column("episode_index").to_numpy()
            frames = table.column("frame_index").to_numpy()
            state = np.asarray(table.column("observation.state").to_pylist(), dtype="<f4")
            action = np.asarray(table.column("action").to_pylist(), dtype="<f4")
            starts = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1, len(ep)]
            for start, stop in zip(starts[:-1], starts[1:]):
                grouped[int(ep[start])].append((frames[start:stop], state[start:stop], action[start:stop]))
            print(f"Reference numeric shard {i+1}/{len(names)} read", flush=True)
    records, signatures = [], set()
    for ep, chunks in sorted(grouped.items()):
        frames, state, action = [np.concatenate([c[i] for c in chunks]) for i in range(3)]
        if not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f"Non-contiguous or reordered reference frames: {ep}")
        signature = trajectory_signature(state, action)
        if signature in signatures:
            raise ValueError(f"Ambiguous duplicate reference trajectory: {ep}")
        signatures.add(signature)
        records.append({"episode_index": ep, "length": len(frames), "signature": signature})
    validate_reference(records)
    save_cache(cache, identity, records)
    return records


def validate_reference(records, *, episodes=EPISODES, frames=FRAMES):
    if {r["episode_index"] for r in records} != set(range(episodes)) or len(records) != episodes:
        raise ValueError("Reference must contain every original episode ID exactly once")
    if sum(r["length"] for r in records) != frames:
        raise ValueError("Reference frame count mismatch")
    if len({r["signature"] for r in records}) != episodes:
        raise ValueError("Ambiguous reference numeric signatures")


def task_records(entry, work, targets=None, cancelled=None):
    task = entry["task"]
    identity = {"schema": 1, "archive_sha256": entry["sha256"], "task": task}
    cache = work / "tasks" / (task + ".json")
    records = read_cache(cache, identity)
    if records is not None:
        print(f"{task}: using verified numeric/instruction cache", flush=True)
        return records
    records = []
    with source_archive(entry, path("input_archives") / task / "demo_clean.zip", work) as z:
        names = sorted(n for n in z.namelist() if "/data/" in n and n.endswith(".hdf5"))
        if len(names) != 50:
            raise ValueError(f"{task}: expected 50 clean episodes, got {len(names)}")
        for i, name in enumerate(names):
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Input preparation cancelled after another task failed")
            member = checked_relative(name)
            if len(member.parts) != 4 or member.parts[0] != task:
                raise ValueError(f"Wrong task archive member: {name}")
            relative = str(PurePosixPath(*member.parts[1:]))
            instr_name = instruction_relative(task, relative).as_posix()
            instruction = z.read(instr_name)
            validate_instruction(instruction)
            raw = z.read(name)  # ZipFile checks member CRC before returning.
            signature, length = hdf5_signature(raw)
            if targets is not None and (signature not in targets or targets[signature]["length"] != length):
                raise ValueError(f"No exact reference match for {task}/{relative}; refusing guessed episode mapping")
            records.append({"task": task, "relative": relative, "length": length,
                            "signature": signature, "raw_sha256": sha256(raw),
                            "instruction_sha256": sha256(instruction),
                            "instruction_base64": base64.b64encode(instruction).decode("ascii")})
            if (i + 1) % 10 == 0:
                print(f"{task}: {i+1}/50 numeric trajectories and original instruction files", flush=True)
    save_cache(cache, identity, records)
    return records


def match_records(reference, records, *, episodes=EPISODES, frames=FRAMES):
    validate_reference(reference, episodes=episodes, frames=frames)
    targets = {r["signature"]: r for r in reference}
    matched, sources = {}, set()
    for row in records:
        key = (row["task"], row["relative"])
        if key in sources:
            raise ValueError(f"Duplicate raw source: {key}")
        sources.add(key)
        ref = targets.get(row["signature"])
        if ref is None or row["length"] != ref["length"]:
            raise ValueError(f"Raw trajectory has no exact state/action match: {key}")
        ep = ref["episode_index"]
        if ep in matched:
            raise ValueError(f"More than one raw trajectory matches episode {ep}")
        data = base64.b64decode(row["instruction_base64"], validate=True)
        if sha256(data) != row["instruction_sha256"]:
            raise ValueError(f"Instruction cache checksum mismatch: {key}")
        validate_instruction(data)
        instruction_relative(*key)
        matched[ep] = row
    if set(matched) != set(range(episodes)):
        raise ValueError(f"Refusing partial publication: matched {len(matched)}/{episodes} episodes")
    return matched


def publish(matched, source_manifest, reference_entry):
    # No production override of these full-dataset requirements.
    if set(matched) != set(range(EPISODES)) or sum(r["length"] for r in matched.values()) != FRAMES:
        raise ValueError("Only the complete original dataset may be published")
    mapping = {str(ep): [r["task"], r["relative"]] for ep, r in sorted(matched.items())}
    mapping_path = writable(path("mapping"))
    instructions = writable(path("instructions"))
    planned = {}
    for row in matched.values():
        rel = instruction_relative(row["task"], row["relative"])
        planned[rel.as_posix()] = base64.b64decode(row["instruction_base64"], validate=True)
    # Preflight every existing file before any publication; never overwrite a
    # different dataset. A previous interrupted publication can be completed.
    if mapping_path.exists() and json.loads(mapping_path.read_text())["mapping"] != mapping:
        raise ValueError("Existing mapping differs; refusing to overwrite it")
    existing = {p.relative_to(instructions).as_posix() for p in instructions.rglob("*") if p.is_file()}
    if existing - set(planned):
        raise ValueError("Existing instruction directory contains unrelated files")
    for rel, data in planned.items():
        dest = writable(instructions / rel)
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError(f"Existing instruction differs: {rel}")
    for rel, data in planned.items():
        dest = writable(instructions / rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temporary = dest.with_name(dest.name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(dest)
    if not mapping_path.exists():
        atomic_json(mapping_path, {"mapping": mapping})
    report = {"schema_version": 1, "complete": True, "episodes": EPISODES, "frames": FRAMES,
              "seen_slots": EPISODES * SEEN, "revision": source_manifest["revision"],
              "reference_sha256": reference_entry["sha256"],
              "matching": "exact full chronological state/action float32 bytes and shapes",
              "instructions": "original JSON member bytes; no sorting, deduplication or regeneration",
              "archive_validation": "pinned revision and size; per-member ZIP CRC; full archive SHA256 only when a local ZIP is used",
              "historical_submission_identity": "Not verified against the unavailable historical data_inputs files",
              "mapping_sha256": sha256(mapping_path.read_bytes()),
              "instruction_sha256": {rel: sha256(data) for rel, data in sorted(planned.items())},
              "episode_sources": [{"episode_index": ep, **{k: r[k] for k in
                  ("task", "relative", "length", "signature", "raw_sha256", "instruction_sha256")}}
                  for ep, r in sorted(matched.items())]}
    atomic_json(path("input_preparation_report"), report)
    return report


def check_prepared_inputs(*, require_report=True):
    mapping_file, instructions = path("mapping"), path("instructions")
    if not mapping_file.is_file():
        raise FileNotFoundError("Run bash prepare_data_inputs.sh to recover all episode mappings and instructions")
    mapping = json.loads(mapping_file.read_text())["mapping"]
    if set(mapping) != {str(i) for i in range(EPISODES)} or len({tuple(v) for v in mapping.values()}) != EPISODES:
        raise ValueError("Expected a one-to-one mapping of all 2500 original episodes")
    hashes = {}
    for task, relative in mapping.values():
        rel = instruction_relative(task, relative)
        file = (instructions / rel).resolve()
        if not file.is_relative_to(instructions):
            raise ValueError("Instruction file escapes configured inputs")
        data = file.read_bytes()
        validate_instruction(data)
        hashes[rel.as_posix()] = sha256(data)
    report_file = path("input_preparation_report")
    if require_report or report_file.exists():
        report = json.loads(report_file.read_text())
        if not report.get("complete") or (report.get("episodes"), report.get("frames"), report.get("seen_slots")) != (EPISODES, FRAMES, EPISODES * SEEN):
            raise ValueError("Input preparation report is incomplete")
        if report["mapping_sha256"] != sha256(mapping_file.read_bytes()) or report["instruction_sha256"] != hashes:
            raise ValueError("Prepared inputs changed since numeric matching")
    return {"episodes": EPISODES, "seen_slots": EPISODES * SEEN, "instruction_files": len(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline verification of published inputs")
    mode.add_argument("--dry-run", action="store_true", help="Print paths; no network or writes")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")
    if args.check:
        print(json.dumps(check_prepared_inputs(), indent=2))
        return
    if args.dry_run:
        print(json.dumps({k: str(path(k)) for k in ("mapping", "instructions", "checkpoint_sam_info",
                          "input_sources_manifest", "input_work", "input_archives")}, indent=2))
        print("Pinned v30 numeric data + 50 clean archives -> exact matching -> 2500 instruction files; CPU only")
        return
    # The published inputs can be verified with stdlib alone, including on a
    # fresh source checkout without the optional recovery environment.
    if path("input_preparation_report").is_file():
        print(json.dumps(check_prepared_inputs(), indent=2))
        return
    exe = interpreter("inputs")
    if not exe.is_file():
        raise FileNotFoundError("Prepare the small CPU environment first: python3 tools/setup_envs.py inputs")
    if Path(sys.executable).absolute() != exe:
        env = runtime_env("prepare-inputs")
        os.execve(str(exe), [str(exe), "-B", str(Path(__file__).resolve()), *sys.argv[1:]], env)
    work = writable(path("input_work"))
    for output in (work, writable(path("mapping")), writable(path("instructions")), writable(path("input_preparation_report"))):
        for name in ("zip_v21", "zip_v30", "input_sources_manifest"):
            source = path(name)
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Input preparation output overlaps a protected source")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path("input_preparation_report").is_file():
            print(json.dumps(check_prepared_inputs(), indent=2))
            return
        sources = json.loads(path("input_sources_manifest").read_text())
        entries = sources["archives"]
        if len(entries) != 50 or len({e["task"] for e in entries}) != 50:
            raise ValueError("Expected exactly 50 unique task archives")
        for e in entries:
            if not re.fullmatch(r"[a-z0-9_]+", e["task"]):
                raise ValueError("Invalid source task name")
            expected = f"https://huggingface.co/datasets/{sources['repo']}/resolve/{sources['revision']}/dataset/{e['task']}/demo_clean.zip"
            if e["url"] != expected or not re.fullmatch(r"[a-f0-9]{64}", e["sha256"]):
                raise ValueError("Invalid pinned source archive")
        reference_entry = next(e for e in load_manifest()["files"] if e["destination"]["path_key"] == "zip_v30")
        if f"/resolve/{sources['revision']}/" not in reference_entry["url"]:
            raise ValueError("Raw and LeRobot revisions differ")
        reference = reference_signatures(reference_entry, work)
        validate_reference(reference)
        records = []
        cancelled = threading.Event()
        targets = {r["signature"]: r for r in reference}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(task_records, e, work, targets, cancelled): e["task"] for e in entries}
            try:
                for i, future in enumerate(as_completed(jobs), 1):
                    records.extend(future.result())
                    print(f"Completed source task {i}/50: {jobs[future]}", flush=True)
            except BaseException:
                cancelled.set()
                for future in jobs:
                    future.cancel()
                raise
        matched = match_records(reference, records)
        publish(matched, sources, reference_entry)
        print(json.dumps(check_prepared_inputs(), indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Input preparation interrupted; verified caches retained for resume.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Input preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
