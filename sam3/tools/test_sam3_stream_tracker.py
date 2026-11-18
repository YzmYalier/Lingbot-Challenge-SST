#!/usr/bin/env python3
"""Compare streaming SAM3 masks against the offline video pipeline.

Run in the ``sam3`` env on one GPU:

    python tools/test_sam3_stream_tracker.py --video <mp4> --prompt robot --prompt "robot arm"

Offline is the reference used to build the training datasets
(``SAM3VideoMasker.iter_video_masks``, hotstart_delay=15, every frame). The
streaming tracker is run over the same video, optionally subsampled with
``--stride`` to imitate one frame per policy query, and per-frame IoU of the
union mask is reported on the shared frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sam3_stream_tracker import SAM3StreamTracker  # noqa: E402
from tools.sam3_video_masker import SAM3VideoMasker  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "ckpt" / "sam3.pt"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum()) / float(union)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--stride", type=int, default=1, help="feed every Nth frame to the stream")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--score-threshold-detection", type=float, default=0.3)
    ap.add_argument("--new-det-thresh", type=float, default=0.5)
    ap.add_argument("--memory-window", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    import cv2

    masker = SAM3VideoMasker(
        checkpoint_path=args.checkpoint,
        score_threshold_detection=args.score_threshold_detection,
        new_det_thresh=args.new_det_thresh,
    ).load()

    # ---- offline reference over the whole video (frames kept in RGB)
    offline: dict[int, np.ndarray] = {}
    frames: dict[int, np.ndarray] = {}
    t0 = time.time()
    for idx, frame_bgr, mask in masker.iter_video_masks(args.video, args.prompt):
        if args.max_frames and idx >= args.max_frames:
            break
        offline[idx] = mask
        if idx % args.stride == 0:
            frames[idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    offline_seconds = time.time() - t0
    indices = sorted(frames)
    print(f"offline: {len(offline)} frames in {offline_seconds:.1f}s; "
          f"streaming {len(indices)} frames (stride={args.stride})", flush=True)

    # ---- streaming over the subsampled frames, reusing the loaded model
    tracker = SAM3StreamTracker(masker=masker, memory_window=args.memory_window or None)
    t0 = time.time()
    session = tracker.start(frames[indices[0]], args.prompt)
    stream = {indices[0]: session.first_mask}
    for idx in indices[1:]:
        stream[idx] = session.step(frames[idx])
    stream_seconds = time.time() - t0
    per_frame = stream_seconds / max(1, len(indices))

    ious = [iou(stream[i], offline[i]) for i in indices]
    rows = [
        {
            "frame": int(i),
            "iou": round(float(v), 4),
            "stream_area": int(stream[i].sum()),
            "offline_area": int(offline[i].sum()),
        }
        for i, v in zip(indices, ious)
    ]
    summary = {
        "video": str(args.video),
        "prompts": args.prompt,
        "stride": args.stride,
        "frames_compared": len(indices),
        "iou_mean": round(float(np.mean(ious)), 4),
        "iou_median": round(float(np.median(ious)), 4),
        "iou_min": round(float(np.min(ious)), 4),
        "iou_p10": round(float(np.percentile(ious, 10)), 4),
        "stream_seconds": round(stream_seconds, 2),
        "stream_seconds_per_frame": round(per_frame, 3),
        "offline_seconds": round(offline_seconds, 2),
        "stream_mean_area": int(np.mean([stream[i].sum() for i in indices])),
        "offline_mean_area": int(np.mean([offline[i].sum() for i in indices])),
        "objects_last_frame": session.last_object_counts,
    }
    session.close()
    print(json.dumps(summary, indent=2))
    for row in rows[:5] + (["..."] if len(rows) > 10 else []) + rows[-5:]:
        print(row)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"summary": summary, "frames": rows}, indent=2))
        print(f"saved={args.json_out}")
    masker.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
