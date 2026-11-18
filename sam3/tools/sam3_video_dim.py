"""Dim video pixels outside SAM3-detected masks.

The detected regions are preserved. Pixels outside the union of all requested
text-prompt masks are multiplied by ``outside_scale``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from sam3_video_masker import PathLike, SAM3VideoMasker
else:  # pragma: no cover
    from .sam3_video_masker import PathLike, SAM3VideoMasker


class SAM3VideoDimmer:
    """Use :class:`SAM3VideoMasker` and darken pixels outside target masks."""

    def __init__(self, masker: SAM3VideoMasker) -> None:
        if not isinstance(masker, SAM3VideoMasker):
            raise TypeError("masker must be a SAM3VideoMasker instance")
        self.masker = masker

    @staticmethod
    def _apply_dim(frame: np.ndarray, mask: np.ndarray, outside_scale: float) -> np.ndarray:
        if not 0.0 <= outside_scale <= 1.0:
            raise ValueError("outside_scale must be between 0 and 1")
        output = frame.copy()
        outside = ~np.asarray(mask, dtype=bool)
        # Use float32 to avoid uint8 multiplication truncation surprises.
        output[outside] = np.clip(
            output[outside].astype(np.float32) * outside_scale, 0, 255
        ).astype(np.uint8)
        return output

    def dim_video(
        self,
        video_path: PathLike,
        text_prompt: Union[str, Sequence[str]],
        output_path: PathLike,
        *,
        outside_scale: float = 0.3,
        codec: str = "mp4v",
        use_ffmpeg: bool = True,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
        output_prob_thresh: Optional[float] = None,
    ) -> Path:
        """Write a video preserving mask pixels and scaling the outside region.

        ``outside_scale=0.3`` means pixels outside detected masks are reduced to
        30% of their original value. ``outside_scale=0`` makes the outside
        black; ``outside_scale=1`` leaves the frame unchanged.
        """

        if not 0.0 <= outside_scale <= 1.0:
            raise ValueError("outside_scale must be between 0 and 1")

        return self.masker.write_video(
            video_path=video_path,
            text_prompt=text_prompt,
            output_path=output_path,
            codec=codec,
            use_ffmpeg=use_ffmpeg,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            output_prob_thresh=output_prob_thresh,
            frame_transform=lambda frame, mask: self._apply_dim(
                frame, mask, outside_scale
            ),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep SAM3 target regions bright and dim pixels outside their masks."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--text-prompt", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--outside-scale", type=float, default=0.3)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--no-ffmpeg", action="store_true")
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
    dimmer = SAM3VideoDimmer(masker)
    try:
        output = dimmer.dim_video(
            video_path=args.video,
            text_prompt=args.text_prompt,
            output_path=args.output,
            outside_scale=args.outside_scale,
            codec=args.codec,
            use_ffmpeg=not args.no_ffmpeg,
            offload_video_to_cpu=not args.keep_video_on_gpu,
            output_prob_thresh=args.output_prob_thresh,
        )
    finally:
        masker.unload()
    print(f"Saved dimmed video to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
