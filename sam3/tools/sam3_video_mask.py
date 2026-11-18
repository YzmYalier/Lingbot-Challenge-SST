"""CLI entry point for SAM3 video mask generation.

The reusable implementation lives in :mod:`sam3_video_masker`.  The class is
re-exported here to preserve the original import path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from sam3_video_masker import SAM3VideoMasker
else:
    from .sam3_video_masker import SAM3VideoMasker

__all__ = ["SAM3VideoMasker", "main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM3 text-prompt segmentation on a video and save a masked video."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--text-prompt",
        action="append",
        required=True,
        help=(
            "Text prompt; repeat this option for multiple classes, "
            '--text-prompt "person" --text-prompt "dog"'
        ),
    )
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument(
        "--version", choices=("sam3", "sam3.1"), default="sam3", help="SAM version"
    )
    parser.add_argument(
        "--mode", choices=("overlay", "mask"), default="overlay", help="Output style"
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity")
    parser.add_argument("--no-contours", action="store_true", help="Do not draw contours")
    parser.add_argument("--codec", default="mp4v", help="Intermediate OpenCV codec")
    parser.add_argument("--no-ffmpeg", action="store_true", help="Do not transcode to H.264")
    parser.add_argument("--no-async-loading", action="store_true")
    parser.add_argument("--keep-video-on-gpu", action="store_true")
    parser.add_argument("--output-prob-thresh", type=float, default=None)
    parser.add_argument("--score-threshold-detection", type=float, default=None)
    parser.add_argument("--new-det-thresh", type=float, default=None)
    parser.add_argument("--det-nms-thresh", type=float, default=None)
    parser.add_argument("--assoc-iou-thresh", type=float, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    masker = SAM3VideoMasker(
        checkpoint_path=args.checkpoint,
        version=args.version,
        async_loading_frames=not args.no_async_loading,
        score_threshold_detection=args.score_threshold_detection,
        new_det_thresh=args.new_det_thresh,
        det_nms_thresh=args.det_nms_thresh,
        assoc_iou_thresh=args.assoc_iou_thresh,
        default_output_prob_thresh=args.output_prob_thresh,
    )
    try:
        output = masker.write_video(
            video_path=args.video,
            text_prompt=args.text_prompt,
            output_path=args.output,
            mode=args.mode,
            alpha=args.alpha,
            draw_contours=not args.no_contours,
            codec=args.codec,
            use_ffmpeg=not args.no_ffmpeg,
            offload_video_to_cpu=not args.keep_video_on_gpu,
            output_prob_thresh=args.output_prob_thresh,
        )
    finally:
        masker.unload()
    print(f"Saved masked video to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
