#!/usr/bin/env python3
"""Validate a dataset written by pipeline_lerobot.py (run in the ``lingbotvla`` env).

Checks, per sampled episode: every video key decodes (pyav, selecting by from/to_timestamp)
to exactly the source cam_high frame count; with saved masks (``sam_masks/``) the video equals
the style rendered from (source frames, saved mask) up to AV1 loss and the mask index covers
every (episode, key); without, every pixel is either ~unchanged or ~outside_scale x; stats
columns equal recomputed stats; and the lerobot loader (torchcodec, the training backend)
returns frames for first/last rows.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_lerobot as P  # noqa: E402
from mask_style import MaskStyle  # noqa: E402


def decode(root: Path, info, r, key):
    p = f"videos/{key}/"
    path = root / info["video_path"].format(video_key=key, chunk_index=int(r[p + "chunk_index"]),
                                            file_index=int(r[p + "file_index"]))
    return P.decode_segment(path, float(r[p + "from_timestamp"]), float(r[p + "to_timestamp"]),
                            int(round((r[p + "to_timestamp"] - r[p + "from_timestamp"]) * info["fps"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--src", type=Path, default=P.DEFAULT_SRC)
    ap.add_argument("--sample", type=int, default=6, help="episodes to check in depth")
    ap.add_argument("--keys", type=int, default=4, help="video keys per episode to check (multi)")
    args = ap.parse_args()
    root = args.dataset
    info = json.loads((root / "meta" / "info.json").read_text())
    sam = json.loads((root / "meta" / "sam_info.json").read_text())
    src = P.Source(args.src)
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").glob("*/*.parquet"))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    prov = pd.read_parquet(root / "meta" / "sam_prompts.parquet")
    keys = sam["video_keys"]
    vids = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    assert P.CAM not in vids or sam["mode"] == "single", "multi must not keep plain cam_high"
    assert all(k in vids for k in keys), "video keys missing from info.json"
    assert len(eps) == info["total_episodes"] and int(eps.length.sum()) == info["total_frames"]
    assert (eps.episode_index.values == np.arange(len(eps))).all()
    if sam["mode"] == "multi":
        assert all(len(t) == P.N_MULTI for t in eps.tasks), "multi: each episode needs 100 instructions"
    masks = None
    if "masks" in sam:
        masks = pd.read_parquet(root / sam["masks"]["index"])
        assert len(masks) == len(eps) * len(keys), "mask index must cover every (episode, key)"
        assert masks.groupby(["episode_index", "video_key"]).size().max() == 1
        missing = [f for f in masks.file.unique() if not (root / f).exists()]
        assert not missing, f"{len(missing)} mask files missing, e.g. {missing[:3]}"
        style = MaskStyle.from_dict(sam["style"])
        jl = root / (sam["masks"].get("judge_log") or "-")
        judge_txt = ""
        if jl.is_file():
            j = pd.read_parquet(jl)
            judge_txt = f", judge log {len(j)} verdicts ({int((~j.keep).sum())} no, {int(j.p_yes.isna().sum())} errors)"
        print(f"masks ok: {masks.file.nunique()} mask videos, style {style.tag()}, keep_frac "
              f"{masks.keep_frac.mean():.3f} (unjudged {masks.keep_frac_unjudged.mean():.3f}){judge_txt}")
    stats = json.loads((root / "meta" / "stats.json").read_text())
    assert all(k in stats for k in keys)
    print(f"meta ok: {len(eps)} episodes, {info['total_frames']} frames, {len(vids)} video keys, "
          f"{info['total_tasks']} tasks")

    random.seed(0)
    sample = sorted(random.sample(range(len(eps)), min(args.sample, len(eps))))
    worst = 0.0
    for i in sample:
        r = eps.iloc[i]
        se = int(prov[prov.episode_index == i].source_episode_index.iloc[0])
        orig = np.stack(P.source_frames(src, se)).astype(np.int16)
        ks = keys if len(keys) <= args.keys else [keys[0], keys[-1]] + random.sample(keys[1:-1], args.keys - 2)
        for key in ks:
            fr = np.stack(decode(root, info, r, key)).astype(np.int16)
            assert fr.shape == orig.shape, (i, key, fr.shape, orig.shape)
            if masks is not None:  # the video is the style rendered from the saved mask
                m = masks[(masks.episode_index == i) & (masks.video_key == key)].iloc[0]
                keep = P.decode_mask_video(root / m.file)
                assert keep.shape == orig.shape[:3], (i, key, keep.shape, orig.shape)
                expect = style.render(orig.astype(np.uint8), keep).astype(np.int16) if len(m.phrases) else orig
                kept = keep
                unexplained = float((np.abs(fr - expect).max(-1) >= 24).mean())
            else:  # pixels SAM3 kept are ~unchanged; dimmed ones are ~outside_scale x (up to AV1 loss)
                kept = np.abs(fr - orig).max(-1) < 24
                dimmed = np.abs(fr - orig * sam["outside_scale"]).max(-1) < 24
                unexplained = float((~kept & ~dimmed).mean())
            st = P.image_stats(fr.astype(np.uint8))
            def diff(s):
                stored = np.array([np.asarray(x[0], float) for x in r[f"stats/{key}/{s}"]])
                return float(np.abs(np.asarray(st[s], float).ravel() - stored.ravel()).max())
            # stored stats are of the pre-encode frames (v30 convention); AV1 loss moves the
            # decoded mean/std slightly and quantiles by a few 1/255 steps
            err, q50 = max(diff("mean"), diff("std")), diff("q50")
            row = prov[(prov.episode_index == i) & (prov.video_key == key)].iloc[0]
            worst = max(worst, unexplained)
            print(f"ep {i:4d} {key:32s} T={len(fr):3d} keep_frac={row.keep_frac:.3f} "
                  f"kept={kept.mean():.3f} unexplained={unexplained:.4f} stats_err(mean/std)={err:.4f} q50={q50:.3f} phrases={list(row.phrases)}")
            assert err < 0.01, "stored stats do not match the video"
    print(f"worst unexplained pixel fraction {worst:.4f}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/" + root.name, root=root, video_backend="torchcodec")
    # Inherited from v30: 625 episodes have 1-2 fewer cam_high frames than `length`. Their last
    # rows decode the next episode's first frame, or fail for the last episode in a file
    # (torchcodec "Invalid frame index"). The source dataset behaves the same; only rows with
    # a real frame are checked here.
    k0 = keys[0]
    nvid = np.round((eps[f"videos/{k0}/to_timestamp"] - eps[f"videos/{k0}/from_timestamp"]) * info["fps"]).astype(int)
    short = eps.episode_index[eps.length > nvid].tolist()
    print(f"episodes whose {k0} is shorter than length (inherited from source): {len(short)}")
    # a row is only decodable if EVERY camera has the frame (wrist cams are short sometimes too)
    nall = np.min([np.round((eps[f"videos/{k}/to_timestamp"] - eps[f"videos/{k}/from_timestamp"]) * info["fps"])
                   .astype(int) for k in vids], axis=0)
    full = eps[eps.length <= nall]
    rows = [0] + ([int(full.dataset_to_index.iloc[0]) - 1, int(full.dataset_to_index.iloc[-1]) - 1]
                  if len(full) else [])
    for idx in rows:
        item = ds[idx]
        shapes = {k: tuple(item[k].shape) for k in vids[:3] + vids[-1:]}
        print(f"loader idx {idx}: task={item['task'][:50]!r} {shapes}")
    print("OK")


if __name__ == "__main__":
    main()
