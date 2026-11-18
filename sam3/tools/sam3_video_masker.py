"""Reusable SAM3 video mask generation wrapper.

This module contains the model/session lifecycle and the reusable video-mask
iterator. Rendering a particular visual effect is intentionally separate from
mask generation so callers can overlay, darken, blur, or otherwise transform
the pixels outside the detected mask.
"""

from __future__ import annotations

import gc
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

# Locate the configuration; all project paths are then resolved from base.json.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "base.json").is_file():
        sys.path.insert(0, str(_candidate))
        break
from submit_paths import base_path, path as configured_path, interpreter, runtime_env, writable

_PROJECT_ROOT = base_path() / "sam3"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PathLike = Union[str, Path]
FrameTransform = Callable[[np.ndarray, np.ndarray], np.ndarray]


class SAM3VideoMasker:
    """Load SAM3 video inference and generate masks frame by frame.

    ``text_prompt`` accepted by :meth:`iter_video_masks` and :meth:`mask_video`
    can be one string or a sequence of strings. Different prompts are evaluated
    in independent SAM3 sessions and unioned frame by frame.

    Parameters such as ``score_threshold_detection`` and ``new_det_thresh`` are
    applied to the constructed video model before inference. They are optional;
    ``None`` keeps the official builder defaults.
    """

    def __init__(
        self,
        checkpoint_path: Optional[PathLike] = None,
        *,
        version: str = "sam3",
        async_loading_frames: bool = True,
        use_fa3: bool = True,
        use_rope_real: bool = True,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
        score_threshold_detection: Optional[float] = None,
        new_det_thresh: Optional[float] = None,
        det_nms_thresh: Optional[float] = None,
        assoc_iou_thresh: Optional[float] = None,
        default_output_prob_thresh: Optional[float] = None,
    ) -> None:
        if version not in {"sam3", "sam3.1"}:
            raise ValueError("version must be 'sam3' or 'sam3.1'")
        self.version = version
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.async_loading_frames = bool(async_loading_frames)
        self.use_fa3 = bool(use_fa3)
        self.use_rope_real = bool(use_rope_real)
        self.max_num_objects = int(max_num_objects)
        self.multiplex_count = int(multiplex_count)
        self.score_threshold_detection = self._optional_probability(
            score_threshold_detection, "score_threshold_detection"
        )
        self.new_det_thresh = self._optional_probability(new_det_thresh, "new_det_thresh")
        self.det_nms_thresh = self._optional_probability(det_nms_thresh, "det_nms_thresh")
        self.assoc_iou_thresh = self._optional_probability(
            assoc_iou_thresh, "assoc_iou_thresh"
        )
        self.default_output_prob_thresh = self._optional_probability(
            default_output_prob_thresh, "default_output_prob_thresh"
        )
        self.predictor: Any = None
        self._loaded_checkpoint_path: Optional[Path] = None

    @staticmethod
    def _optional_probability(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return value

    @property
    def is_loaded(self) -> bool:
        return self.predictor is not None

    def _resolve_checkpoint(self) -> Path:
        checkpoint_dir = _PROJECT_ROOT / "ckpt"
        filename = "sam3.1_multiplex.pt" if self.version == "sam3.1" else "sam3.pt"
        if self.checkpoint_path is None:
            candidate = configured_path("sam_checkpoint")
        elif self.checkpoint_path.is_dir():
            candidate = self.checkpoint_path / filename
        else:
            candidate = self.checkpoint_path
        if candidate.is_file():
            return candidate

        partial_files = sorted(
            path
            for path in candidate.parent.glob("*")
            if path.is_file()
            and path.name.endswith((".incomplete", ".part", ".tmp", ".crdownload"))
        )
        partial_hint = ""
        if partial_files:
            partial_hint = " Partial download files: " + ", ".join(
                str(path) for path in partial_files
            )
        raise FileNotFoundError(
            f"SAM3 checkpoint was not found at {candidate}. "
            f"Finish downloading the checkpoint before calling load().{partial_hint}"
        )

    def _apply_model_thresholds(self) -> None:
        """Apply optional detector thresholds after official model construction."""

        model = getattr(self.predictor, "model", None)
        if model is None:
            return
        for name in (
            "score_threshold_detection",
            "new_det_thresh",
            "det_nms_thresh",
            "assoc_iou_thresh",
        ):
            value = getattr(self, name)
            if value is not None and hasattr(model, name):
                setattr(model, name, value)
        if self.default_output_prob_thresh is not None:
            if hasattr(self.predictor, "default_output_prob_thresh"):
                self.predictor.default_output_prob_thresh = self.default_output_prob_thresh
            if hasattr(model, "default_output_prob_thresh"):
                model.default_output_prob_thresh = self.default_output_prob_thresh

    def set_thresholds(
        self,
        *,
        score_threshold_detection: Optional[float] = None,
        new_det_thresh: Optional[float] = None,
        det_nms_thresh: Optional[float] = None,
        assoc_iou_thresh: Optional[float] = None,
        default_output_prob_thresh: Optional[float] = None,
    ) -> "SAM3VideoMasker":
        """Set video thresholds before or after :meth:`load`."""

        values = {
            "score_threshold_detection": score_threshold_detection,
            "new_det_thresh": new_det_thresh,
            "det_nms_thresh": det_nms_thresh,
            "assoc_iou_thresh": assoc_iou_thresh,
            "default_output_prob_thresh": default_output_prob_thresh,
        }
        for name, value in values.items():
            if value is not None:
                setattr(self, name, self._optional_probability(value, name))
        if self.is_loaded:
            self._apply_model_thresholds()
        return self

    def load(self) -> "SAM3VideoMasker":
        """Construct the official SAM3 video predictor once."""

        if self.is_loaded:
            return self
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "SAM3 video predictor requires CUDA, but torch.cuda.is_available() "
                    "is False."
                )
            checkpoint = self._resolve_checkpoint()
            from sam3.model_builder import build_sam3_predictor

            builder_kwargs: Dict[str, Any] = {
                "checkpoint_path": str(checkpoint),
                "version": self.version,
                "async_loading_frames": self.async_loading_frames,
            }
            if self.version == "sam3.1":
                if self.default_output_prob_thresh is not None:
                    builder_kwargs["default_output_prob_thresh"] = (
                        self.default_output_prob_thresh
                    )
                builder_kwargs.update(
                    {
                        "use_fa3": self.use_fa3,
                        "use_rope_real": self.use_rope_real,
                        "max_num_objects": self.max_num_objects,
                        "multiplex_count": self.multiplex_count,
                    }
                )
            self.predictor = build_sam3_predictor(**builder_kwargs)
            self._loaded_checkpoint_path = checkpoint
            self._apply_model_thresholds()
            self._configure_inference_dtypes()
            return self
        except Exception:
            self.unload()
            raise

    def _configure_inference_dtypes(self) -> None:
        """Use the same dtype handling in the CLI and threaded HTTP service."""
        # The official tracker keeps a long-lived BF16 autocast context
        # on the thread that loads the model.
        # On this H200/PyTorch combination it leaks BF16 activations into
        # FP32 detector/decoder layers. Disable that persistent context and
        # install narrow Conv/Linear input hooks that align floating inputs
        # with each module's checkpoint dtype. This preserves the model and
        # all user-facing prompts/thresholds while avoiding mixed-dtype CUDA
        # kernels in both detector and tracker paths.
        import torch

        predictor_model = getattr(self.predictor, "model", None)
        tracker = getattr(predictor_model, "tracker", None)
        bf16_context = getattr(tracker, "bf16_context", None)
        if bf16_context is not None:
            bf16_context.__exit__(None, None, None)
            tracker.bf16_context = None
        self._dtype_hooks = []

        def align_inputs(module, inputs):
            weight = getattr(module, "weight", None)
            if weight is None:
                return inputs
            return tuple(
                item.to(dtype=weight.dtype)
                if isinstance(item, torch.Tensor) and torch.is_floating_point(item)
                else item
                for item in inputs
            )

        for module in self.predictor.model.modules():
            if isinstance(
                module,
                (
                    torch.nn.Conv1d,
                    torch.nn.Conv2d,
                    torch.nn.Conv3d,
                    torch.nn.Linear,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                self._dtype_hooks.append(module.register_forward_pre_hook(align_inputs))

    @staticmethod
    def _import_cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Video processing requires OpenCV. Install it with `pip install opencv-python`."
            ) from exc
        return cv2

    @staticmethod
    def _validate_prompt(text_prompt: str) -> str:
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")
        return text_prompt.strip()

    @classmethod
    def _normalize_prompts(
        cls, text_prompt: Union[str, Sequence[str]]
    ) -> List[str]:
        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        elif isinstance(text_prompt, Sequence):
            prompts = list(text_prompt)
        else:
            raise TypeError("text_prompt must be a string or a sequence of strings")
        if not prompts:
            raise ValueError("text_prompt must contain at least one prompt")
        return [cls._validate_prompt(prompt) for prompt in prompts]

    @staticmethod
    def _merge_masks(
        masks: Any, frame_height: int, frame_width: int, cv2: Any
    ) -> np.ndarray:
        if masks is None:
            merged = np.zeros((frame_height, frame_width), dtype=bool)
        else:
            merged = np.asarray(masks)
            if merged.size == 0:
                merged = np.zeros((frame_height, frame_width), dtype=bool)
            else:
                if merged.ndim == 2:
                    merged = merged[None, ...]
                if merged.ndim != 3:
                    raise ValueError(
                        "SAM3 output `out_binary_masks` must have shape [N, H, W], "
                        f"got {merged.shape}"
                    )
                merged = np.any(merged.astype(bool), axis=0)
        if merged.shape != (frame_height, frame_width):
            merged = cv2.resize(
                merged.astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return merged

    @staticmethod
    def _video_info(video_path: PathLike, cv2: Any) -> Tuple[float, int, int, int]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open input video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Could not determine video dimensions: {video_path}")
        return fps, width, height, frame_count

    def iter_video_masks(
        self,
        video_path: PathLike,
        text_prompt: Union[str, Sequence[str]],
        *,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
        output_prob_thresh: Optional[float] = None,
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(frame_index, frame_bgr, merged_mask)`` for every video frame."""

        prompts = self._normalize_prompts(text_prompt)
        input_path = Path(video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")
        if output_prob_thresh is not None:
            output_prob_thresh = self._optional_probability(
                output_prob_thresh, "output_prob_thresh"
            )
        cv2 = self._import_cv2()
        self.load()
        fps, width, height, frame_count = self._video_info(input_path, cv2)
        del fps

        with tempfile.TemporaryDirectory(prefix="sam3_masks-") as temp_dir:
            mask_store = None
            merged_masks: Dict[int, np.ndarray] = {}
            if frame_count > 0:
                mask_store = np.memmap(
                    Path(temp_dir) / "merged_masks.dat",
                    dtype=np.bool_,
                    mode="w+",
                    shape=(frame_count, height, width),
                )
                mask_store[:] = False

            for prompt in prompts:
                session_id: Optional[str] = None
                try:
                    request = {
                        "type": "start_session",
                        "resource_path": str(input_path),
                        "offload_video_to_cpu": offload_video_to_cpu,
                        "offload_state_to_cpu": offload_state_to_cpu,
                    }
                    session = self.predictor.handle_request(request)
                    session_id = session["session_id"]
                    add_prompt_request = {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": prompt,
                    }
                    if output_prob_thresh is not None:
                        add_prompt_request["output_prob_thresh"] = output_prob_thresh
                    self.predictor.handle_request(add_prompt_request)

                    propagation_request = {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "evict_cached_frame_outputs": True,
                    }
                    if output_prob_thresh is not None:
                        propagation_request["output_prob_thresh"] = output_prob_thresh
                    seen_frame_indices = set()
                    for result in self.predictor.handle_stream_request(
                        propagation_request
                    ):
                        frame_idx = int(result["frame_index"])
                        if frame_idx in seen_frame_indices:
                            continue
                        if frame_idx < 0 or (
                            frame_count > 0 and frame_idx >= frame_count
                        ):
                            raise RuntimeError(
                                f"SAM3 returned invalid frame index {frame_idx}"
                            )
                        seen_frame_indices.add(frame_idx)
                        outputs = result.get("outputs", {})
                        mask = self._merge_masks(
                            outputs.get("out_binary_masks"), height, width, cv2
                        )
                        if mask_store is not None:
                            mask_store[frame_idx] |= mask
                        elif frame_idx in merged_masks:
                            merged_masks[frame_idx] |= mask
                        else:
                            merged_masks[frame_idx] = mask
                finally:
                    if session_id is not None and self.predictor is not None:
                        try:
                            self.predictor.handle_request(
                                {"type": "close_session", "session_id": session_id}
                            )
                        except Exception:
                            pass

            if mask_store is not None:
                mask_store.flush()
            capture = cv2.VideoCapture(str(input_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"OpenCV could not reopen input video: {input_path}")
            try:
                frame_idx = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if mask_store is not None and frame_idx < frame_count:
                        mask = np.asarray(mask_store[frame_idx]).copy()
                    else:
                        mask = merged_masks.get(
                            frame_idx, np.zeros((height, width), dtype=bool)
                        )
                    yield frame_idx, frame, mask
                    frame_idx += 1
            finally:
                capture.release()

    # Short alias for callers that prefer a shorter name.
    iter_masks = iter_video_masks

    @staticmethod
    def render_mask_overlay(
        frame: np.ndarray,
        mask: np.ndarray,
        *,
        mode: str = "overlay",
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.45,
        draw_contours: bool = True,
    ) -> np.ndarray:
        """Render a mask overlay or a standalone colored mask."""

        if mode not in {"overlay", "mask"}:
            raise ValueError("mode must be 'overlay' or 'mask'")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        cv2 = SAM3VideoMasker._import_cv2()
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if mode == "mask":
            rendered = np.zeros_like(frame)
            rendered[mask] = np.asarray(color, dtype=np.uint8)
        else:
            rendered = frame.copy()
            if np.any(mask):
                color_layer = np.empty_like(rendered)
                color_layer[:] = np.asarray(color, dtype=np.uint8)
                rendered[mask] = cv2.addWeighted(
                    rendered[mask], 1.0 - alpha, color_layer[mask], alpha, 0.0
                )
        if draw_contours and np.any(mask):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(rendered, contours, -1, color, 2)
        return rendered

    @staticmethod
    def _finalize_video(
        intermediate_path: Path,
        output_path: Path,
        source_video_path: Path,
        *,
        use_ffmpeg: bool,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not use_ffmpeg or ffmpeg is None:
            if use_ffmpeg and ffmpeg is None:
                warnings.warn(
                    "ffmpeg was not found; keeping the OpenCV-generated video "
                    "without H.264 compatibility transcoding.",
                    RuntimeWarning,
                )
            shutil.copyfile(intermediate_path, output_path)
            return
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(intermediate_path),
            "-i",
            str(source_video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(
                "ffmpeg could not create a compatible H.264 output video. "
                "Set use_ffmpeg=False to keep the raw OpenCV output."
            ) from exc

    def write_video(
        self,
        video_path: PathLike,
        text_prompt: Union[str, Sequence[str]],
        output_path: PathLike,
        *,
        frame_transform: Optional[FrameTransform] = None,
        mode: str = "overlay",
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.45,
        draw_contours: bool = True,
        codec: str = "mp4v",
        use_ffmpeg: bool = True,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
        output_prob_thresh: Optional[float] = None,
    ) -> Path:
        """Generate masks and write a transformed video.

        ``frame_transform(frame, mask)`` can be supplied for custom effects.
        When omitted, the built-in overlay/mask renderer is used.
        """

        input_path = Path(video_path)
        output = Path(output_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")
        if output.exists() and input_path.resolve() == output.resolve():
            raise ValueError("output_path must differ from video_path")
        cv2 = self._import_cv2()
        fps, width, height, _ = self._video_info(input_path, cv2)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer: Any = None
        try:
            with tempfile.TemporaryDirectory(prefix="sam3_video-") as temp_dir:
                intermediate = Path(temp_dir) / "rendered_opencv.mp4"
                writer = cv2.VideoWriter(
                    str(intermediate), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"OpenCV could not open output video writer: {intermediate}"
                    )
                for _, frame, mask in self.iter_video_masks(
                    input_path,
                    text_prompt,
                    offload_video_to_cpu=offload_video_to_cpu,
                    offload_state_to_cpu=offload_state_to_cpu,
                    output_prob_thresh=output_prob_thresh,
                ):
                    if frame_transform is not None:
                        rendered = frame_transform(frame, mask)
                    else:
                        rendered = self.render_mask_overlay(
                            frame,
                            mask,
                            mode=mode,
                            color=color,
                            alpha=alpha,
                            draw_contours=draw_contours,
                        )
                    if not isinstance(rendered, np.ndarray) or rendered.shape != frame.shape:
                        raise ValueError(
                            "frame_transform must return a numpy array with the same "
                            "shape as the input frame"
                        )
                    writer.write(np.ascontiguousarray(rendered, dtype=np.uint8))
                writer.release()
                writer = None
                self._finalize_video(
                    intermediate, output, input_path, use_ffmpeg=use_ffmpeg
                )
        finally:
            if writer is not None:
                writer.release()
        return output

    def mask_video(self, *args: Any, **kwargs: Any) -> Path:
        """Backward-compatible alias for :meth:`write_video`."""

        return self.write_video(*args, **kwargs)

    def unload(self) -> None:
        for hook in getattr(self, "_dtype_hooks", []):
            hook.remove()
        self._dtype_hooks = []
        predictor = self.predictor
        self.predictor = None
        self._loaded_checkpoint_path = None
        if predictor is not None:
            try:
                predictor.shutdown()
            except Exception:
                pass
            del predictor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "SAM3VideoMasker":
        return self.load()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.unload()

    def __del__(self) -> None:
        try:
            self.unload()
        except Exception:
            pass
