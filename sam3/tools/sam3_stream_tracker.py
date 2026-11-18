"""Online (frame-by-frame) SAM3 text-prompt tracking for closed-loop evaluation.

The official SAM3 video API is offline: ``init_state`` decodes the whole video
up front and ``propagate_in_video`` runs over a fixed ``num_frames``. A policy
rollout only has the frames observed so far, so this module grows a SAM3
inference state by one frame per :meth:`SAM3StreamTracker.step` call and runs
the same per-frame detector + tracker update the offline propagation uses
(``Sam3VideoInference._run_single_frame_inference``).

Why growing ``num_frames`` is enough to make it causal: with one GPU the
detector buffers exactly the current frame and only prefetches frame ``t+1``
when ``t+1 < num_frames`` (``Sam3ImageOnVideoMultiGPU``), and the tracker's
memory selection only looks backwards. Keeping ``num_frames == t + 1`` while
frame ``t`` is processed therefore never touches a future frame.

Differences from the offline masks used for training (``SAM3VideoMasker``):

- Output delay. Offline, ``hotstart_delay=15`` holds every output back 15
  frames, so objects the hotstart heuristics remove later are hidden from
  their first frame. A closed loop cannot wait: the mask of frame ``t`` is
  returned immediately. The heuristics themselves still run unchanged; an
  object they remove disappears from that frame on.
- Frames are pre-processed like the image-folder / PIL-list loaders used to
  build the lerobot SAM datasets (PIL bilinear resize to 1008, /255, mean/std
  0.5, float16).
- Only the frames the caller feeds exist. With sparse feeding (one frame per
  policy query) the tracker sees a low-frame-rate video.

One prompt = one inference state, masks are unioned across prompts, the same
contract as ``SAM3VideoMasker.iter_video_masks``. Everything here only calls
existing SAM3 internals; no SAM3 source file is modified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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

from tools.sam3_video_masker import SAM3VideoMasker  # noqa: E402

PathLike = Union[str, Path]


def preprocess_frame(frame_rgb: np.ndarray, image_size: int, mean=0.5, std=0.5):
    """uint8 RGB (H, W, 3) -> normalized float16 CPU tensor (3, S, S).

    Mirrors ``io_utils._load_img_as_tensor`` + ``AsyncImageFrameLoader``:
    PIL bilinear resize, ``to_tensor`` (/255), then (x - mean) / std in fp16.
    """
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"expected uint8 (H, W, 3) RGB, got {frame_rgb.dtype} {frame_rgb.shape}")
    img = Image.fromarray(frame_rgb)
    img = TF.resize(img, size=(image_size, image_size))
    tensor = TF.to_tensor(img).to(dtype=torch.float16)
    tensor -= mean
    tensor /= std
    return tensor


class _FrameStore:
    """Growable, indexable frame list shared by every prompt's inference state.

    SAM3 indexes ``input_batch.img_batch[frame_idx]`` and calls ``len()`` on it;
    a non-tensor sequence is supported by ``Sam3Image._get_img_feats``. Only the
    current frame is ever read back (the tracker re-uses cached backbone
    features), so frames older than ``keep`` steps are released to bound memory.
    """

    def __init__(self, keep: int = 2) -> None:
        self._frames: List[Any] = []
        self.keep = max(1, int(keep))

    def append(self, tensor) -> int:
        self._frames.append(tensor)
        old = len(self._frames) - 1 - self.keep
        if old >= 0:
            self._frames[old] = None
        return len(self._frames) - 1

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index):
        frame = self._frames[index]
        if frame is None:
            raise IndexError(f"frame {index} was released (only the last {self.keep} are kept)")
        return frame


class StreamSession:
    """One rollout: a growing frame store plus one SAM3 inference state per prompt.

    Create with :meth:`SAM3StreamTracker.start`; feed later frames with
    :meth:`step`. Each call returns the union mask of all prompts for the frame
    just added, at the frame's own resolution.
    """

    def __init__(self, tracker: "SAM3StreamTracker", frame_rgb: np.ndarray,
                 prompts: Sequence[str]) -> None:
        self.tracker = tracker
        self.prompts = SAM3VideoMasker._normalize_prompts(list(prompts))
        self.height, self.width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
        self.frames = _FrameStore(keep=2)
        self.states: List[Dict[str, Any]] = []
        # Object ids removed by the hotstart heuristics so far, per prompt. The
        # offline loop accumulates the same set but applies it 15 frames late.
        self._removed: List[set] = []
        self.last_object_counts: List[int] = []
        self.closed = False
        self._first_mask = self._start(frame_rgb)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def first_mask(self) -> np.ndarray:
        return self._first_mask

    # ------------------------------------------------------------------ build
    def _new_state(self, first_image) -> Dict[str, Any]:
        """Mirror ``Sam3VideoInference.init_state`` for a 1-frame, growable video."""
        model = self.tracker.model
        state: Dict[str, Any] = {
            "image_size": model.image_size,
            "offload_state_to_cpu": False,
            "num_frames": 1,
            "orig_height": self.height,
            "orig_width": self.width,
            "constants": {},
        }
        model._construct_initial_input_batch(state, first_image.unsqueeze(0))
        # Swap the copied tensor for the shared, growable store (indexable +
        # len(), which is all the detector and tracker need from img_batch).
        state["input_batch"].img_batch = self.frames
        state["tracker_inference_states"] = []
        state["tracker_metadata"] = {}
        state["feature_cache"] = {}
        state["cached_frame_outputs"] = {}
        state["action_history"] = []
        # A 1-frame list would be classified as an image; this is a video.
        state["is_image_only"] = False
        return state

    def _grow(self, state: Dict[str, Any], frame_idx: int) -> None:
        """Append per-frame slots so ``frame_idx`` becomes the last frame."""
        import torch
        from sam3.model.data_misc import FindStage, convert_my_tensors
        from sam3.model.utils.misc import copy_data_to_device

        model = self.tracker.model
        assert state["num_frames"] == frame_idx, (state["num_frames"], frame_idx)
        text_id = int(state["input_batch"].find_inputs[0].text_ids.flatten()[0].item())
        stage = FindStage(
            img_ids=[frame_idx],
            text_ids=[text_id],
            input_boxes=[torch.zeros(258)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, 257)],
            input_points_mask=[torch.empty(0)],
            object_ids=[],
        )
        stage = copy_data_to_device(convert_my_tensors(stage), model.device, non_blocking=True)
        batch = state["input_batch"]
        batch.find_inputs.append(stage)
        batch.find_targets.append(None)
        batch.find_metadatas.append(None)
        for key in ("previous_stages_out", "per_frame_raw_point_input",
                    "per_frame_raw_box_input", "per_frame_visual_prompt",
                    "per_frame_geometric_prompt"):
            state[key].append(None)
        state["per_frame_cur_step"].append(0)
        state["num_frames"] = frame_idx + 1
        # Existing tracker states carry their own copy of the frame count.
        for tracker_state in state["tracker_inference_states"]:
            tracker_state["num_frames"] = frame_idx + 1

    # ---------------------------------------------------------------- running
    def _advance(self, i: int, frame_idx: int) -> np.ndarray:
        """One iteration of the offline propagation loop, emitted without delay."""
        model = self.tracker.model
        state = self.states[i]
        out = model._run_single_frame_inference(state, frame_idx, reverse=False)
        self._removed[i].update(out["removed_obj_ids"])
        suppressed = out["suppressed_obj_ids"]
        unconfirmed = out.get("unconfirmed_obj_ids", None)
        post = model._postprocess_output(state, out, self._removed[i], suppressed, unconfirmed)
        # The offline loop caches then evicts (evict_cached_frame_outputs=True).
        state["cached_frame_outputs"].pop(frame_idx, None)
        self.tracker._prune(state, frame_idx)
        masks = post["out_binary_masks"]
        self.last_object_counts[i] = int(len(masks))
        return SAM3VideoMasker._merge_masks(
            masks, self.height, self.width, SAM3VideoMasker._import_cv2()
        )

    def _start(self, frame_rgb: np.ndarray) -> np.ndarray:
        import torch

        model = self.tracker.model
        image = self.tracker.preprocess(frame_rgb)
        self.frames.append(image)
        union = np.zeros((self.height, self.width), dtype=bool)
        self.tracker._begin_frame((id(self), 0))
        try:
            with torch.inference_mode():
                for i, prompt in enumerate(self.prompts):
                    state = self._new_state(image)
                    self.states.append(state)
                    self._removed.append(set())
                    self.last_object_counts.append(0)
                    # Same as Sam3BasePredictor.add_prompt (bf16 autocast).
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model.add_prompt(state, frame_idx=0, text_str=prompt)
                    # propagate_in_video starts at the prompted frame and runs it
                    # again, so frame 0 is processed twice offline as well.
                    state["feature_cache"]["tracking_bounds"] = {
                        "max_frame_num_to_track": None,
                        "propagate_in_video_start_frame_idx": None,
                    }
                    union |= self._advance(i, 0)
        finally:
            self.tracker._end_frame()
        return union

    def step(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Add the next frame of the rollout and return its union mask (H, W) bool."""
        import torch

        if self.closed:
            raise RuntimeError("session is closed")
        if frame_rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame size changed: {frame_rgb.shape[:2]} vs {(self.height, self.width)}"
            )
        image = self.tracker.preprocess(frame_rgb)
        frame_idx = self.frames.append(image)
        union = np.zeros((self.height, self.width), dtype=bool)
        self.tracker._begin_frame((id(self), frame_idx))
        try:
            with torch.inference_mode():
                for i in range(len(self.states)):
                    self._grow(self.states[i], frame_idx)
                    union |= self._advance(i, frame_idx)
        finally:
            self.tracker._end_frame()
        return union

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.states.clear()
        self._removed.clear()
        self.frames = _FrameStore(keep=1)
        self.tracker._release_cuda_cache()


class SAM3StreamTracker:
    """Owns one loaded SAM3 video model; hands out :class:`StreamSession` objects.

    ``memory_window`` (frames) bounds per-session GPU memory on long rollouts by
    dropping non-conditioning tracker outputs older than the window. ``None``
    keeps everything, exactly like the offline propagation. The tracker attends
    to at most 7 memory frames and 16 object pointers, chosen among the most
    recent frames, so a window of a few dozen frames changes nothing in
    practice except when long stretches of frames have near-zero mask quality.
    """

    def __init__(
        self,
        checkpoint_path: Optional[PathLike] = None,
        *,
        score_threshold_detection: Optional[float] = 0.3,
        new_det_thresh: Optional[float] = 0.5,
        memory_window: Optional[int] = None,
        masker: Optional[SAM3VideoMasker] = None,
    ) -> None:
        self.masker = masker or SAM3VideoMasker(
            checkpoint_path=checkpoint_path,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
        )
        self.memory_window = None if memory_window in (None, 0) else int(memory_window)

    @property
    def model(self):
        if not self.masker.is_loaded:
            raise RuntimeError("call load() first")
        return self.masker.predictor.model

    @property
    def is_loaded(self) -> bool:
        return self.masker.is_loaded

    def load(self) -> "SAM3StreamTracker":
        self.masker.load()
        self._install_backbone_cache()
        return self

    # ------------------------------------------------------- backbone sharing
    # Every prompt has its own inference state, so the same frame would go
    # through the image backbone once per prompt. While a frame is processed
    # (``frame_key`` set), the first call computes the features and the other
    # prompts reuse them. Outside a frame (e.g. offline use of the same model)
    # the wrapper is a pass-through.
    _frame_key: Optional[Any] = None
    _cached: Optional[Any] = None
    share_backbone: bool = True

    def _install_backbone_cache(self) -> None:
        backbone = self.model.detector.backbone
        if getattr(backbone, "_stream_cache_installed", False):
            return
        original = backbone.forward_image
        tracker = self

        def forward_image(samples, *args, **kwargs):
            key = tracker._frame_key if tracker.share_backbone else None
            if key is not None and tracker._cached is not None and tracker._cached[0] == key:
                return tracker._cached[1]
            out = original(samples, *args, **kwargs)
            if key is not None:
                tracker._cached = (key, out)
            return out

        backbone.forward_image = forward_image
        backbone._stream_cache_installed = True

    def _begin_frame(self, key) -> None:
        self._frame_key = key
        self._cached = None

    def _end_frame(self) -> None:
        self._frame_key = None
        self._cached = None

    def unload(self) -> None:
        self.masker.unload()

    def preprocess(self, frame_rgb: np.ndarray):
        model = self.model
        mean = float(model.image_mean[0])
        std = float(model.image_std[0])
        if any(m != mean for m in model.image_mean) or any(s != std for s in model.image_std):
            raise NotImplementedError("per-channel mean/std differ; extend preprocess_frame")
        return preprocess_frame(frame_rgb, model.image_size, mean=mean, std=std)

    def start(self, frame_rgb: np.ndarray, prompts: Sequence[str]) -> StreamSession:
        """Start a rollout on its first frame. ``session.first_mask`` is frame 0's mask."""
        self.load()
        self._install_backbone_cache()  # also when the masker was loaded by someone else
        return StreamSession(self, np.ascontiguousarray(frame_rgb), prompts)

    def _prune(self, state: Dict[str, Any], frame_idx: int) -> None:
        if self.memory_window is None:
            return
        oldest = frame_idx - self.memory_window
        if oldest <= 0:
            return
        for tracker_state in state["tracker_inference_states"]:
            dicts = [tracker_state["output_dict"]]
            dicts += list(tracker_state["output_dict_per_obj"].values())
            for output_dict in dicts:
                non_cond = output_dict["non_cond_frame_outputs"]
                for t in [t for t in non_cond if t < oldest]:
                    del non_cond[t]

    @staticmethod
    def _release_cuda_cache() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
