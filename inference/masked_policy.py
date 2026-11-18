"""Drop-in wrapper for ``WebsocketClientPolicy`` that SAM3-masks cam_high.

Only ``observation.images.cam_high`` is replaced; wrist cameras, state and the
instruction pass through untouched, so the VLA server needs no change.

Frame feeding (``sam_frames``):
  vla -- SAM3 sees exactly the frames the VLA is queried with (one per chunk).
  all -- SAM3 also sees every executed step in between; the eval loop calls
         :meth:`MaskedPolicyClient.observe_step` after each ``take_action``.
         The VLA still gets the mask of its own query frame, but the tracker
         followed the scene at the full control rate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .cam_high_masker import CamHighMasker
from submit_paths import config

CAM_HIGH = "observation.images.cam_high"
SAM_FRAME_MODES = ("vla", "all")


class MaskedPolicyClient:
    def __init__(self, policy, masker: CamHighMasker, sam_frames: str = "vla") -> None:
        if sam_frames not in SAM_FRAME_MODES:
            raise ValueError(f"sam_frames must be one of {SAM_FRAME_MODES}, got {sam_frames!r}")
        self.policy = policy
        self.masker = masker
        self.sam_frames = sam_frames

    # Same surface as WebsocketClientPolicy -------------------------------------
    def get_server_metadata(self) -> Dict:
        return self.policy.get_server_metadata()

    def infer(self, obs: Dict) -> Dict:
        if obs.get("reset"):
            # A reset opens a new episode; its first observation starts SAM3.
            self.masker.end(None)
            return self.policy.infer(obs)
        frame = obs.get(CAM_HIGH)
        if frame is None:
            raise KeyError(f"MaskedPolicyClient needs {CAM_HIGH} in the observation")
        frame = np.asarray(frame)
        if not self.masker.active:
            dimmed = self.masker.begin(str(obs.get("task", "")), frame)
        else:
            dimmed = self.masker.observe(frame)
        obs = dict(obs)
        obs[CAM_HIGH] = dimmed
        return self.policy.infer(obs)

    def reset(self, robo_name: str) -> None:
        self.infer(dict(reset=True, robo_name=robo_name))

    # Eval-loop hooks ------------------------------------------------------------
    @property
    def wants_every_step(self) -> bool:
        return self.sam_frames == "all"

    def observe_step(self, head_rgb: np.ndarray) -> None:
        """``sam_frames=all``: feed an executed (non-VLA) frame to the tracker."""
        if self.sam_frames != "all" or not self.masker.active:
            return
        self.masker.observe(np.asarray(head_rgb))

    def end_episode(self, episode_index: int, success: Optional[bool]) -> None:
        """Close the SAM3 session and save the masked video under the eval video's index."""
        self.masker.end(episode_index, success)


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def build_policy_client(usr_args: Dict[str, Any], save_dir: Optional[Path] = None):
    """The eval client's policy object.

    ``sam_mask`` off (default): the plain ``WebsocketClientPolicy``, exactly
    what the original eval constructed. On: wrapped in :class:`MaskedPolicyClient`.
    """
    from script.deploy.websocket_client_policy import WebsocketClientPolicy

    policy = WebsocketClientPolicy(port=usr_args["port"])
    if not _flag(usr_args.get("sam_mask", False)):
        return policy

    s = config()["inference"]
    sam_url = str(usr_args.get("sam_url") or f'http://{s["host"]}:{int(usr_args.get("sam_port", s["port"]))}')
    # mask-Qwen = Qwen3.8-27B vLLM (prompt generation), NOT the VLA's Qwen3-VL-4B.
    q = config()["qwen"]
    qwen_url = str(usr_args.get("mask_qwen_url") or f'http://{q["host"]}:{q["port"]}')
    # The eval client eval()s every override value, so a bare `all` would arrive
    # as the builtin all(); map a callable back to its name.
    sam_frames = usr_args.get("sam_frames", "vla")
    sam_frames = sam_frames.__name__ if callable(sam_frames) else str(sam_frames)
    save_video = _flag(usr_args.get("sam_save_video", True))
    masker = CamHighMasker(
        qwen_url=qwen_url,
        sam_url=sam_url,
        outside_scale=float(usr_args.get("outside_scale", 0.1)),
        edge_width=int(usr_args.get("edge_width", 2)),
        judge=_flag(usr_args.get("sam_judge", True)),
        save_dir=save_dir,
        save_video=save_video,
        video_fps=int(usr_args.get("video_fps", 10)),
        log_prefix=f"[sam-mask {os.getpid()}]",
    )
    masker.sam.wait_ready(float(usr_args.get("sam_ready_timeout", 900)))
    print(f"\033[36m[sam-mask] cam_high masking ON: mask-qwen={qwen_url} sam3={sam_url} "
          f"sam_frames={sam_frames} outside_scale={masker.outside_scale} "
          f"edge_width={masker.edge_width} judge={masker.judge} "
          f"save_video={save_video}\033[0m", flush=True)
    return MaskedPolicyClient(policy, masker, sam_frames=sam_frames)
