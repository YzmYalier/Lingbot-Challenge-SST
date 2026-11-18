#!/usr/bin/env python3
"""Persistent single-GPU SAM3 video dimming HTTP service.

The model is loaded once at startup and reused for each ``POST /dim`` request.
The service intentionally uses the existing ``SAM3VideoDimmer`` implementation
and does not alter its prompt or rendering parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Dict
from urllib.parse import urlparse


# Locate the configuration; all project paths are then resolved from base.json.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "base.json").is_file():
        sys.path.insert(0, str(_candidate))
        break
from submit_paths import base_path, path as configured_path, interpreter, runtime_env, writable

PROJECT_ROOT = base_path()
SAM3_ROOT = PROJECT_ROOT / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from tools.sam3_video_dim import SAM3VideoDimmer  # noqa: E402
from tools.sam3_video_masker import SAM3VideoMasker  # noqa: E402


class ServiceState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.masker = SAM3VideoMasker(
            checkpoint_path=args.checkpoint,
            version=args.version,
            async_loading_frames=not args.no_async_loading,
            score_threshold_detection=args.score_threshold_detection,
            new_det_thresh=args.new_det_thresh,
            det_nms_thresh=args.det_nms_thresh,
            assoc_iou_thresh=args.assoc_iou_thresh,
            default_output_prob_thresh=args.output_prob_thresh,
        )
        self.dimmer = SAM3VideoDimmer(self.masker)
        self.lock = Lock()
        self.masker.load()

    def close(self) -> None:
        self.masker.unload()

    def dim(self, request: Dict[str, Any]) -> Dict[str, Any]:
        video = Path(request["video"]).expanduser()
        output = Path(request["output"]).expanduser()
        prompts = request["text_prompts"]
        if isinstance(prompts, str):
            prompts = [prompts]
        if not video.is_file():
            raise FileNotFoundError(f"input video does not exist: {video}")
        if not isinstance(prompts, list) or not prompts or not all(
            isinstance(item, str) and item.strip() for item in prompts
        ):
            raise ValueError("text_prompts must be a non-empty list of strings")
        outside_scale = float(request.get("outside_scale", self.args.outside_scale))
        # Request thresholds belong inside the inference lock: another request
        # must not change an in-progress video's detector configuration.
        score_threshold = request.get("score_threshold_detection", self.args.score_threshold_detection)
        new_det_thresh = request.get("new_det_thresh", self.args.new_det_thresh)
        if output.resolve() == video.resolve():
            raise ValueError("output must differ from the input video")
        if output.resolve().is_relative_to((PROJECT_ROOT / "RoboTwinData" / "dataset").resolve()):
            raise ValueError("outputs must be outside RoboTwinData/dataset")
        with self.lock:
            self.masker.set_thresholds(
                score_threshold_detection=score_threshold,
                new_det_thresh=new_det_thresh,
            )
            saved = self.dimmer.dim_video(
                video_path=video,
                text_prompt=prompts,
                output_path=output,
                outside_scale=outside_scale,
                offload_video_to_cpu=not self.args.keep_video_on_gpu,
                offload_state_to_cpu=self.args.offload_state_to_cpu,
                output_prob_thresh=self.args.output_prob_thresh,
            )
        return {"ok": True, "output": str(saved), "prompts": prompts,
                "score_threshold_detection": score_threshold, "new_det_thresh": new_det_thresh}


class Handler(BaseHTTPRequestHandler):
    state: ServiceState

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/health":
            self._send(200, {"ok": True, "loaded": self.state.masker.is_loaded})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/dim":
            self._send(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.state.dim(request)
            self._send(200, result)
        except Exception as exc:  # keep service alive for the next request
            traceback.print_exc()
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        print("sam3_service: " + (format % args), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent SAM3 video dimming service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--outside-scale", type=float, default=0.2)
    parser.add_argument("--score-threshold-detection", type=float, default=0.3)
    parser.add_argument("--output-prob-thresh", type=float, default=None)
    parser.add_argument("--new-det-thresh", type=float, default=0.5)
    parser.add_argument("--det-nms-thresh", type=float, default=None)
    parser.add_argument("--assoc-iou-thresh", type=float, default=None)
    parser.add_argument("--no-async-loading", action="store_true")
    parser.add_argument("--keep-video-on-gpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.outside_scale <= 1.0:
        raise SystemExit("--outside-scale must be between 0 and 1")
    state = ServiceState(args)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"SAM3 service listening on http://{args.host}:{args.port}; "
        f"checkpoint={args.checkpoint}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
