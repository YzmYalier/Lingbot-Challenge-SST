#!/usr/bin/env python3
"""Smoke test for the sim-side masking wrapper, without a simulator or a VLA.

Needs a running mask-Qwen vLLM server and one ``sam3_stream_server.py``. Feeds
a recorded cam_high video through :class:`MaskedPolicyClient` (with a fake
policy that just records what it receives) and checks:

- cam_high is replaced by the dimmed frame, the other keys pass through;
- ``sam_frames=vla`` feeds one frame per ``infer``; ``all`` also feeds
  ``observe_step`` frames;
- a reset starts a new SAM3 session; ``end_episode`` writes the mp4 + json.

Run from the RoboTwin env:
    python test_cam_high_masker.py --video <mp4> --instruction "..." --sam-url http://127.0.0.1:9701
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import as package "sam_mask"

from sam_mask.cam_high_masker import CamHighMasker  # noqa: E402
from sam_mask.masked_policy import CAM_HIGH, MaskedPolicyClient  # noqa: E402


class FakePolicy:
    def __init__(self) -> None:
        self.received = []

    def get_server_metadata(self):
        return {"action_space": "joint"}

    def infer(self, obs):
        self.received.append(obs)
        return {"action": np.zeros((2, 14)), "server_timing": {}}


def read_frames(path: Path, n: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def run(mode: str, frames, args, out_dir: Path) -> dict:
    policy = FakePolicy()
    masker = CamHighMasker(qwen_url=args.qwen_url, sam_url=args.sam_url,
                           outside_scale=args.outside_scale, edge_width=args.edge_width,
                           judge=args.judge, save_dir=out_dir, save_video=True, video_fps=10)
    client = MaskedPolicyClient(policy, masker, sam_frames=mode)
    wrist = np.full_like(frames[0], 7)
    fed = 0
    for episode in range(2):
        client.infer({"reset": True, "robo_name": "robotwin"})
        for t, frame in enumerate(frames):
            if t % args.stride == 0:
                client.infer({CAM_HIGH: frame, "observation.images.cam_left_wrist": wrist,
                              "observation.state": np.zeros(14), "task": args.instruction})
                fed += 1
            elif client.wants_every_step:
                client.observe_step(frame)
                fed += 1
        client.end_episode(episode, success=(episode == 0))
    sent = [o for o in policy.received if not o.get("reset")]
    first = sent[0]
    meta = json.loads((out_dir / "episode0_success_cam_high_masked.json").read_text())
    checks = {
        "resets_forwarded": sum(1 for o in policy.received if o.get("reset")) == 2,
        "cam_high_replaced": not np.array_equal(first[CAM_HIGH], frames[0]),
        "cam_high_shape_dtype": first[CAM_HIGH].shape == frames[0].shape and first[CAM_HIGH].dtype == np.uint8,
        "wrist_untouched": first["observation.images.cam_left_wrist"] is wrist,
        "kept_pixels_identical": bool(np.any(first[CAM_HIGH] == frames[0])),
        "frames_per_episode": meta["frames"] == fed // 2,
        "mp4_written": (out_dir / "episode0_success_cam_high_masked.mp4").stat().st_size > 0,
        "episode1_files": (out_dir / "episode1_failure_cam_high_masked.json").exists(),
        "judge_recorded": (len(meta.get("judge_verdicts", [])) > 0) == args.judge,
        "edge_width_recorded": meta.get("edge_width", 0) == args.edge_width,
    }
    return {"mode": mode, "checks": checks, "prompts": meta["sam3_prompts"],
            "fallback": meta["fallback"], "frames": meta["frames"],
            "sam_seconds_mean": meta["sam_seconds_mean"],
            "mask_qwen_seconds": meta["mask_qwen_seconds"],
            "mask_area_first_last": [meta["mask_area_fraction"][0], meta["mask_area_fraction"][-1]],
            "judge_verdicts": [{k: v.get(k) for k in ("step", "prompt", "obj_id", "p_yes", "keep", "error")}
                               for v in meta.get("judge_verdicts", [])]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--qwen-url", default="http://127.0.0.1:8000")
    ap.add_argument("--sam-url", default="http://127.0.0.1:9701")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--outside-scale", type=float, default=0.2)
    ap.add_argument("--edge-width", type=int, default=0)
    ap.add_argument("--judge", action="store_true", help="needs a SAM3 server started with --judge-url")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    frames = read_frames(args.video, args.frames)
    ok = True
    for mode in ("vla", "all"):
        out_dir = (args.out / mode) if args.out else Path(tempfile.mkdtemp(prefix=f"sam_mask_{mode}_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run(mode, frames, args, out_dir)
        print(json.dumps(result, indent=1), flush=True)
        ok &= all(result["checks"].values())
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
