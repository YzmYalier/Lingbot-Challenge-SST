#!/usr/bin/env python3
"""Audit full metadata and files, numeric/wrist hashes, and decoded sample timelines."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import path
from process_data import check_raw, file_hash, write_json
import rebuild_robotwin_sam as R


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_check(file, count, fps, shape, *, mask=False):
    decoded = 0
    with av.open(str(file)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for i, frame in enumerate(container.decode(stream)):
            require((frame.height, frame.width) == tuple(shape), f"Wrong dimensions: {file}")
            # Matroska FFV1 uses a 1 ms time base: 1/15 second rounds to 67 ms.
            # Permit at most half a container tick, not a fraction of a video frame.
            tolerance = max(1e-4, float(frame.time_base) / 2 + 1e-9) if mask else 1e-4
            require(frame.pts is not None and abs(float(frame.pts * frame.time_base) - i / fps) <= tolerance,
                    f"Wrong PTS at frame {i}: {file}")
            if mask:
                values = np.unique(frame.to_ndarray(format="gray"))
                require(set(values.tolist()) <= {0, 255}, f"Nonbinary mask: {file}")
            decoded += 1
    require(decoded == count, f"Wrong frame count {decoded} != {count}: {file}")


def main():
    from submit_paths import base_path
    check_raw()
    p = R.import_pipeline(base_path() / "auto_mask/pipeline_lerobot.py")
    raw = p.Source(path("raw"))
    output_root = path("sam_output")
    output_info = json.loads((output_root / "meta/info.json").read_text())
    # Source is a RAW-only reader: SAM multi has cam_high_1..100 and no cam_high.
    output_eps = pd.concat([pd.read_parquet(f) for f in sorted((output_root / "meta/episodes").glob("*/*.parquet"))],
                           ignore_index=True).set_index("episode_index", drop=False)
    out = SimpleNamespace(root=output_root, info=output_info, eps=output_eps,
                          fps=int(output_info["fps"]), hw=tuple(output_info["features"][p.cam_keys("multi")[0]]["shape"][1:]))
    out.video_path = lambda ep, key: p.Source.video_path(out, ep, key)
    info = json.loads((out.root / "meta/sam_info.json").read_text())
    require(Path(info["source"]).resolve() == raw.root, "SAM source is not the configured framefixed raw")
    require(info["mode"] == "multi" and not info["subset"] and info["format"] == "objects-judge-v1"
            and info["prompt_version"] == "robotwin-sam3-v3.5", "Wrong pipeline metadata")
    require(info["style"] == p.MaskStyle("outline", 0.1, 2).to_dict(), "Wrong render style")
    require(info["sam3"] == p.SAM3_ARGS, "Wrong SAM3 thresholds")
    require(bool(info["judge"]) and info["judge"]["threshold"] == 0.5
            and info["judge"]["outside_scale"] == 0.2, "Wrong judge settings")
    keys = p.cam_keys("multi")
    wrists = ["observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    require({k for k, v in out.info["features"].items() if v["dtype"] == "video"} == set(keys + wrists),
            "Expected 100 high and two wrist camera keys")
    require(len(out.eps) == 2500 and int(out.eps["length"].sum()) == 548893, "Incomplete dataset")
    for col in ("episode_index", "length", "dataset_from_index", "dataset_to_index"):
        require(np.array_equal(raw.eps[col], out.eps[col]), f"Changed {col}")
    for key in keys + wrists:
        start = out.eps[f"videos/{key}/from_timestamp"]
        end = out.eps[f"videos/{key}/to_timestamp"]
        require(np.allclose((end - start) * out.fps, out.eps["length"], atol=1e-4, rtol=0),
                f"Short/misaligned video segments: {key}")
        require(np.allclose(start, 0), f"Expected independent episode files: {key}")
    # Scan once: 255000 per-file stat calls are very slow on shared filesystems.
    existing_videos = {str(Path(directory, file).relative_to(out.root))
                       for directory, _, files in os.walk(out.root / "videos") for file in files}
    for key in keys + wrists:
        chunks = out.eps[f"videos/{key}/chunk_index"].to_numpy()
        files = out.eps[f"videos/{key}/file_index"].to_numpy()
        for chunk, file in set(zip(chunks, files)):
            relative = out.info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
            require(relative in existing_videos, f"Missing RGB file: {relative}")
    index = pd.read_parquet(out.root / "sam_masks/index.parquet")
    prompts = pd.read_parquet(out.root / "meta/sam_prompts.parquet")
    judge = pd.read_parquet(out.root / "sam_masks/judge_log.parquet")
    for name, table in (("index", index), ("prompts", prompts)):
        require(len(table) == 250000 and not table.duplicated(["episode_index", "video_key"]).any(),
                f"Missing/duplicate {name} slots")
        require(set(table.episode_index) == set(range(2500)) and set(table.video_key) == set(keys),
                f"Wrong {name} IDs/keys")
        require((table.groupby("episode_index").size() == 100).all(), f"Incomplete {name} instructions")
        require(np.array_equal(table.episode_index, table.source_episode_index), f"Reindexed {name}")
    require(prompts.qwen_error.isna().all(), "Qwen extraction errors exist")
    require(judge.error.isna().all() and judge.p_yes.notna().all(), "Qwen judge errors exist")
    require(judge.p_yes.between(0, 1).all(), "Invalid judge probability")
    indexed = index.set_index(["episode_index", "video_key"])
    prompted = prompts.set_index(["episode_index", "video_key"])
    aligned = prompted.reindex(indexed.index)
    require(np.array_equal(aligned.mask_file.to_numpy(), indexed["file"].to_numpy()), "Mask provenance mismatch")
    require(np.array_equal(index.n_frames.to_numpy(), raw.eps.length.reindex(index.episode_index).to_numpy()),
            "Mask frame-count metadata mismatch")
    slot_index = pd.MultiIndex.from_product([range(2500), keys], names=["episode_index", "video_key"])
    instructions = prompted.reindex(slot_index).instruction.to_numpy().reshape(2500, 100)
    for ep in range(2500):
        expected = raw.seen_instructions(ep)
        require(list(out.eps.loc[ep, "tasks"]) == expected, f"Instruction order changed: {ep}")
        require(instructions[ep].tolist() == expected, f"Prompt instruction order changed: {ep}")
        for key in wrists:
            require(file_hash(raw.video_path(ep, key)) == file_hash(out.video_path(ep, key)),
                    f"Changed wrist video: {ep}/{key}")
        if (ep + 1) % 100 == 0:
            print(f"Validated metadata/files: {ep+1}/2500 episodes", flush=True)
    for relative in index["file"].unique():
        file = (out.root / relative).resolve()
        require(file.is_relative_to(out.root) and file.is_file(), f"Missing/unsafe mask: {relative}")
    numeric = sorted(raw.root.glob("data/*/*.parquet"))
    require({x.relative_to(raw.root) for x in numeric} == {x.relative_to(out.root) for x in out.root.glob("data/*/*.parquet")},
            "Numeric parquet layout changed")
    for file in numeric:
        require(file_hash(file) == file_hash(out.root / file.relative_to(raw.root)), f"Changed numerical data: {file}")
    # Includes known historical short episodes and boundaries, plus spread-out samples.
    samples = sorted({0, 50, 93, 94, 2499, *np.linspace(0, 2499, 8, dtype=int).tolist()})
    sampled_keys = [keys[i] for i in (0, 32, 66, 99)]
    checked_masks = set()
    for ep in samples:
        n = int(raw.eps.loc[ep, "length"])
        for key in sampled_keys:
            decode_check(out.video_path(ep, key), n, out.fps, out.hw)
            mask = out.root / indexed.loc[(ep, key), "file"]
            if mask not in checked_masks:
                decode_check(mask, n, out.fps, out.hw, mask=True)
                checked_masks.add(mask)
    report = {"complete": True, "episodes": 2500, "frames": 548893, "top_slots": len(index),
              "mask_files": int(index["file"].nunique()), "wrist_hashes_checked": 5000,
              "numeric_files_checked": len(numeric), "judge_verdicts": len(judge),
              "qwen_errors": 0, "judge_errors": 0, "decoded_episode_samples": list(map(int, samples)),
              "decoded_rgb_files": len(samples) * len(sampled_keys), "decoded_mask_files": len(checked_masks),
              "note": "Full file/metadata/hash audit plus sampled PTS/binary-mask decoding. FFV1/MKV PTS permits half a container tick for rounding. Assembly also decodes every unique RGB segment and final mask before publication."}
    write_json(path("work") / "validation_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
