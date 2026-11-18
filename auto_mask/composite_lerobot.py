#!/usr/bin/env python3
"""Re-render the cam_high key(s) of a pipeline_lerobot.py dataset from its saved final masks.

Input: a dataset written by ``pipeline_lerobot.py`` (it has ``sam_masks/`` + ``meta/sam_info.json``)
and the original source dataset recorded in ``sam_info.json["source"]``. Every masked video key
is rendered again from (original v30 cam_high frames, final mask) with ``--style`` (see
mask_style.py), e.g. the outline style: mask unchanged, 2-px white ring outside the mask,
everything else x 0.1:

  python auto_mask/composite_lerobot.py RoboTwinData/lerobot_dataset/RoboTwin_lerobot_v30_sam_judged_single \\
      --style outline --outside-scale 0.1 --edge-width 2
  # -> RoboTwin_lerobot_v30_sam_judged_single_outline0.1_w2

Only the masked keys, their stats (episodes parquet + stats.json) and sam_info change;
everything else (data, other cameras, tasks, sam_masks) is hard-linked (copied across
filesystems). ``--preview DIR --episodes 0,5`` instead writes side-by-side mp4s
(original | mask | styled) for a quick look. Run in the ``lingbotvla`` env.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_lerobot as P  # noqa: E402
from mask_style import MaskStyle, add_style_args, style_from_args  # noqa: E402


def load(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    sam = json.loads((root / "meta" / "sam_info.json").read_text())
    if "masks" not in sam:
        raise SystemExit(f"{root} has no saved masks (built before sam_masks existed)")
    idx = pd.read_parquet(root / sam["masks"]["index"])
    return info, sam, idx


def render_segment(src_root: str, src_ep: int, mask_path: str, style: dict, out_path: str) -> dict:
    """(source frames, final mask) -> styled AV1 segment + its lerobot stats (CPU worker)."""
    src = P._cached_source(src_root)
    frames = np.stack(P.source_frames(src, src_ep))
    keep = P.decode_mask_video(Path(mask_path))
    if keep.shape != frames.shape[:3]:
        raise RuntimeError(f"{mask_path}: mask {keep.shape} vs frames {frames.shape}")
    out = MaskStyle.from_dict(style).render(frames, keep)
    P.encode_video(out, Path(out_path), src.fps)
    return P.image_stats(out)


def preview(args, root, info, sam, idx, style):
    """Side-by-side mp4 per (episode, group): original | mask | styled (libx264)."""
    src = P.Source(Path(sam["source"]))
    args.preview.mkdir(parents=True, exist_ok=True)
    eps = P.parse_episodes(args.episodes, int(info["total_episodes"])) if args.episodes else [0]
    for ep in eps:
        rows = idx[idx.episode_index == ep].drop_duplicates("group")
        for r in rows.itertuples():
            frames = np.stack(P.source_frames(src, int(r.source_episode_index)))
            keep = P.decode_mask_video(root / r.file)
            styled = style.render(frames, keep)
            m = np.repeat(keep[..., None].view(np.uint8) * np.uint8(255), 3, axis=-1)
            panel = np.concatenate([frames, m, styled], axis=2)
            path = args.preview / f"ep{ep:06d}_g{int(r.group):03d}_{style.tag()}.mp4"
            with av.open(str(path), "w") as c:
                s = c.add_stream("libx264", int(info["fps"]), options={"crf": "18"})
                s.pix_fmt, s.width, s.height = "yuv420p", panel.shape[2], panel.shape[1]
                for a in panel:
                    for pkt in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                        c.mux(pkt)
                for pkt in s.encode():
                    c.mux(pkt)
            P.log(f"preview {path} phrases={list(r.phrases)} keep_frac={r.keep_frac:.3f}")


def link_tree(src: Path, dst: Path, skip) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if path.is_dir() or skip(rel):
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        P.link_or_copy(str(path), dst / rel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("dataset", type=Path, help="dataset written by pipeline_lerobot.py")
    ap.add_argument("--out", type=Path, help="default: <dataset>_<style tag>")
    add_style_args(ap, default_style="outline", default_scale=0.1, default_width=2)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", type=Path, help="write side-by-side preview mp4s here instead")
    ap.add_argument("--episodes", default="", help="preview episodes, e.g. '0,5,100-102' (default 0)")
    args = ap.parse_args()
    style = style_from_args(args)
    root = args.dataset.resolve()
    info, sam, idx = load(root)
    if args.preview:
        preview(args, root, info, sam, idx, style)
        return 0
    out_final = (args.out or root.with_name(f"{root.name}_{style.tag()}")).resolve()
    if out_final == root:
        raise SystemExit("--out must differ from the input dataset")
    if out_final.exists() and not args.overwrite:
        raise SystemExit(f"{out_final} exists; pass --overwrite")
    out = out_final.with_name(out_final.name + ".partial")
    shutil.rmtree(out, ignore_errors=True)
    keys = sam["video_keys"]
    vpath = info["video_path"]
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").glob("*/*.parquet"))])
    eps = eps.sort_values("episode_index").set_index("episode_index", drop=False)
    t0 = time.time()

    # Everything except the masked videos, episodes meta, stats and sam_info is unchanged.
    key_dirs = {Path(vpath.format(video_key=k, chunk_index=0, file_index=0)).parts[1] for k in keys}
    def rewritten(rel: Path) -> bool:
        if rel.parts[0] == "videos":
            return len(rel.parts) > 1 and rel.parts[1] in key_dirs
        return rel.parts[0] == "meta" and (rel.parts[1:2] == ("episodes",)
                                           or rel.name in ("stats.json", "sam_info.json"))
    link_tree(root, out, rewritten)
    P.log(f"linked unchanged files in {time.time() - t0:.0f}s")

    tmp = Path(tempfile.mkdtemp(prefix="composite_", dir=out))
    seg_of, stats_of = {}, {}  # (ep, group) -> segment path / stats
    groups = idx.drop_duplicates(["episode_index", "group"])
    pool = cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                  initializer=P.quiet_stderr, initargs=(str(tmp / "encode.log"),))
    try:
        futs = {}
        for r in groups.itertuples():
            k = (int(r.episode_index), int(r.group))
            seg_of[k] = str(tmp / f"{k[0]:06d}_g{k[1]:03d}.mp4")
            futs[pool.submit(render_segment, sam["source"], int(r.source_episode_index), str(root / r.file),
                             style.to_dict(), seg_of[k])] = k
        done = 0
        for fut in cf.as_completed(futs):
            stats_of[futs[fut]] = fut.result()
            done += 1
            if done % 500 == 0 or done == len(futs):
                P.log(f"rendered {done}/{len(futs)} segments")
        group_of = {(int(r.episode_index), r.video_key): int(r.group) for r in idx.itertuples()}
        cfuts = []
        for key in keys:
            p = f"videos/{key}/"
            files = collections.defaultdict(list)
            order = eps.sort_values(p + "from_timestamp")
            for ep, c, f in zip(order.episode_index, order[p + "chunk_index"], order[p + "file_index"]):
                files[(int(c), int(f))].append(int(ep))
            for (c, f), file_eps in files.items():
                n = sum(int(round((eps.loc[ep, p + "to_timestamp"] - eps.loc[ep, p + "from_timestamp"])
                                  * info["fps"])) for ep in file_eps)
                cfuts.append(pool.submit(P.concat_segments, [seg_of[(ep, group_of[(ep, key)])] for ep in file_eps],
                                         str(out / vpath.format(video_key=key, chunk_index=c, file_index=f)), n))
        for fut in cf.as_completed(cfuts):
            fut.result()
    finally:
        pool.shutdown(cancel_futures=True)
    P.log(f"videos written in {(time.time() - t0) / 60:.1f} min")
    shutil.rmtree(tmp)

    # ---- episodes meta: replace the masked keys' stats columns, same file split as the input
    stypes = P.image_stats_type()
    for f in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        t = pq.read_table(f)
        ep_ids = t.column("episode_index").to_pylist()
        for key in keys:
            st = [stats_of[(e, group_of[(e, key)])] for e in ep_ids]
            for stat, typ in stypes.items():
                name = f"stats/{key}/{stat}"
                t = t.set_column(t.schema.get_field_index(name), name, pa.array([s[stat] for s in st], typ))
        dst = out / f.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst)
    et = pa.concat_tables([pq.read_table(f) for f in sorted((out / "meta" / "episodes").glob("*/*.parquet"))])
    stats = json.loads((root / "meta" / "stats.json").read_text())
    stats.update(P.aggregate_from_table(et.select([n for n in et.column_names
                                                   if any(n.startswith(f"stats/{k}/") for k in keys)])))
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    sam_out = dict(sam, style=style.to_dict(), outside_scale=style.outside_scale,
                   composited_from=str(root), composite_tool="auto_mask/composite_lerobot.py")
    (out / "meta" / "sam_info.json").write_text(json.dumps(sam_out, indent=2))
    if out_final.exists():
        shutil.rmtree(out_final)
    os.replace(out, out_final)
    P.log(f"wrote {out_final} ({len(eps)} episodes, {len(keys)} key(s), style {style.tag()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
