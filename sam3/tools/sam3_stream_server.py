#!/usr/bin/env python3
"""HTTP service around :class:`SAM3StreamTracker` for closed-loop evaluation.

One process owns one GPU's SAM3 model and any number of rollout sessions; GPU
work is serialized by a lock (the same model cannot run two frames at once).
Stdlib only on the wire, so a client in any env (e.g. RoboTwin) can call it.

Endpoints (JSON bodies; images/masks are base64):
  GET  /health                       -> {"ok", "loaded", "sessions"}
  POST /start {client_id, prompts, frame, shape}  -> {"ok", "session_id", "mask", "shape", ...}
  POST /step  {session_id, frame, shape}          -> {"ok", "mask", "shape", ...}
  POST /end   {session_id}                         -> {"ok"}
  POST /start_judged {client_id, prompts, instruction, frame, shape}
                                     -> /start fields + {"rejected", "judged", "judge_seconds"}
  POST /step_judged  {session_id, frame, shape}   -> /step fields + {"rejected", "judged", "judge_seconds"}

The *_judged routes need ``--judge-url``: every new object id is checked
by Qwen (tools/sam3_mask_judge.py) and, if rejected, hidden from the mask. A
rejected id is judged again once its area reaches ``--judge-regrow`` x its
area at the last verdict (0 = never). ``rejected`` lists the hidden ids per prompt,
``judged`` the verdicts made on this frame.

``frame`` is the raw uint8 RGB buffer of an (H, W, 3) array, ``mask`` is
``np.packbits`` of the (H, W) bool union mask. ``client_id`` lets a restarted
client replace its own stale session instead of leaking GPU memory; sessions
idle for ``--idle-timeout`` seconds are closed as well.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import numpy as np

# Locate the configuration; all project paths are then resolved from base.json.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "base.json").is_file():
        sys.path.insert(0, str(_candidate))
        break
from submit_paths import base_path, path as configured_path, interpreter, runtime_env, writable

ROOT = base_path() / "sam3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sam3_stream_tracker import SAM3StreamTracker  # noqa: E402


def decode_frame(request: Dict[str, Any]) -> np.ndarray:
    h, w = (int(v) for v in request["shape"][:2])
    raw = base64.b64decode(request["frame"])
    if len(raw) != h * w * 3:
        raise ValueError(f"frame has {len(raw)} bytes, expected {h}x{w}x3")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)


def encode_mask(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(bool), axis=None).tobytes()).decode("ascii")


class ServiceState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tracker = SAM3StreamTracker(
            checkpoint_path=args.checkpoint,
            score_threshold_detection=args.score_threshold_detection,
            new_det_thresh=args.new_det_thresh,
            memory_window=args.memory_window,
        ).load()
        self.judge = None
        if args.judge_url:
            from tools.sam3_mask_judge import QwenMaskJudge

            self.judge = QwenMaskJudge(
                args.judge_url, model=args.judge_model, threshold=args.judge_threshold,
                outside_scale=args.judge_outside_scale, timeout=args.judge_timeout,
            )
        self.lock = threading.Lock()  # GPU work + session table
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.by_client: Dict[str, str] = {}

    # ------------------------------------------------------------- sessions
    def _close(self, session_id: str) -> None:
        entry = self.sessions.pop(session_id, None)
        if entry is None:
            return
        if self.by_client.get(entry["client_id"]) == session_id:
            self.by_client.pop(entry["client_id"], None)
        entry["session"].close()

    def _expire(self) -> None:
        if self.args.idle_timeout <= 0:
            return
        now = time.time()
        for sid in [s for s, e in self.sessions.items() if now - e["used"] > self.args.idle_timeout]:
            print(f"sam3_stream: closing idle session {sid}", flush=True)
            self._close(sid)

    def start(self, request: Dict[str, Any]) -> Dict[str, Any]:
        prompts = request["prompts"]
        if isinstance(prompts, str):
            prompts = [prompts]
        frame = decode_frame(request)
        client_id = str(request.get("client_id") or uuid.uuid4())
        with self.lock:
            self._expire()
            old = self.by_client.get(client_id)
            if old is not None:
                self._close(old)
            t0 = time.time()
            session = self.tracker.start(frame, prompts)
            seconds = time.time() - t0
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {"session": session, "client_id": client_id, "used": time.time()}
            self.by_client[client_id] = session_id
            mask = session.first_mask
            objects = list(session.last_object_counts)
        return {"ok": True, "session_id": session_id, "mask": encode_mask(mask),
                "shape": list(mask.shape), "seconds": round(seconds, 4),
                "objects": objects, "prompts": session.prompts}

    def step(self, request: Dict[str, Any]) -> Dict[str, Any]:
        frame = decode_frame(request)
        session_id = request["session_id"]
        with self.lock:
            entry = self.sessions.get(session_id)
            if entry is None:
                raise KeyError(f"unknown or expired session {session_id}")
            t0 = time.time()
            mask = entry["session"].step(frame)
            seconds = time.time() - t0
            entry["used"] = time.time()
            objects = list(entry["session"].last_object_counts)
            frame_index = entry["session"].num_frames - 1
        return {"ok": True, "mask": encode_mask(mask), "shape": list(mask.shape),
                "seconds": round(seconds, 4), "objects": objects, "frame_index": frame_index}

    # ------------------------------------------------ Qwen-judged sessions
    # /start_judged and /step_judged: same as /start and /step, but every new
    # object id is shown to the Qwen judge and dropped from the mask if Qwen
    # rejects it (re-judged when its area grows by --judge-regrow). The SAM3
    # work holds the GPU lock; the Qwen calls do not, so other sessions keep
    # running meanwhile.
    def _judge_frame(self, session_id: str, session, frame: np.ndarray) -> Dict[str, Any]:
        items = session.pending()
        t0 = time.time()
        verdicts = self.judge.judge(frame, session.instruction, items)
        judge_seconds = time.time() - t0
        with self.lock:
            if session_id not in self.sessions:
                raise KeyError(f"session {session_id} closed while judging")
            mask = session.apply(items, verdicts)
            objects = list(session.last_object_counts)
            rejected = session.rejected_ids()
            judged = list(session.verdict_log[-len(items):]) if items else []
        return {"mask": mask, "objects": objects, "rejected": rejected, "judged": judged,
                "judge_seconds": round(judge_seconds, 4)}

    def _require_judge(self) -> None:
        if self.judge is None:
            raise RuntimeError("server started without --judge-url; use /start and /step")

    def start_judged(self, request: Dict[str, Any]) -> Dict[str, Any]:
        from tools.sam3_mask_judge import start_judged_session

        self._require_judge()
        prompts = request["prompts"]
        if isinstance(prompts, str):
            prompts = [prompts]
        instruction = str(request["instruction"])
        frame = decode_frame(request)
        client_id = str(request.get("client_id") or uuid.uuid4())
        with self.lock:
            self._expire()
            old = self.by_client.get(client_id)
            if old is not None:
                self._close(old)
            t0 = time.time()
            session = start_judged_session(self.tracker, frame, prompts, instruction,
                                           regrow=self.args.judge_regrow,
                                           regrow_min_area=self.args.judge_regrow_min_area)
            seconds = time.time() - t0
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {"session": session, "client_id": client_id, "used": time.time()}
            self.by_client[client_id] = session_id
        out = self._judge_frame(session_id, session, frame)
        return {"ok": True, "session_id": session_id, "mask": encode_mask(out["mask"]),
                "shape": list(out["mask"].shape), "seconds": round(seconds, 4),
                "objects": out["objects"], "prompts": session.prompts, "rejected": out["rejected"],
                "judged": out["judged"], "judge_seconds": out["judge_seconds"]}

    def step_judged(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self._require_judge()
        frame = decode_frame(request)
        session_id = request["session_id"]
        with self.lock:
            entry = self.sessions.get(session_id)
            if entry is None:
                raise KeyError(f"unknown or expired session {session_id}")
            session = entry["session"]
            if not hasattr(session, "pending"):
                raise TypeError("session was started with /start; use /step")
            t0 = time.time()
            session.step(frame)
            seconds = time.time() - t0
            entry["used"] = time.time()
            frame_index = session.num_frames - 1
        out = self._judge_frame(session_id, session, frame)
        return {"ok": True, "mask": encode_mask(out["mask"]), "shape": list(out["mask"].shape),
                "seconds": round(seconds, 4), "objects": out["objects"], "frame_index": frame_index,
                "rejected": out["rejected"], "judged": out["judged"],
                "judge_seconds": out["judge_seconds"]}

    def end(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self._close(request["session_id"])
        return {"ok": True}

    def health(self) -> Dict[str, Any]:
        return {"ok": True, "loaded": self.tracker.is_loaded, "sessions": len(self.sessions)}

    def close(self) -> None:
        with self.lock:
            for sid in list(self.sessions):
                self._close(sid)
        self.tracker.unload()


class Handler(BaseHTTPRequestHandler):
    state: ServiceState
    routes = {"/start": "start", "/step": "step", "/end": "end",
              "/start_judged": "start_judged", "/step_judged": "step_judged"}

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/health":
            self._send(200, self.state.health())
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.routes.get(urlparse(self.path).path)
        if route is None:
            self._send(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send(200, getattr(self.state, route)(request))
        except Exception as exc:  # keep the service alive for other sessions
            traceback.print_exc()
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        if self.state.args.verbose:
            print("sam3_stream: " + (format % args), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming SAM3 mask service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--checkpoint", type=Path, default=configured_path("sam_checkpoint"))
    parser.add_argument("--score-threshold-detection", type=float, default=0.3)
    parser.add_argument("--new-det-thresh", type=float, default=0.5)
    parser.add_argument("--memory-window", type=int, default=0,
                        help="drop tracker memories older than N frames (0 = keep all)")
    parser.add_argument("--idle-timeout", type=float, default=3600.0,
                        help="close sessions unused for this many seconds (0 = never)")
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    parser.add_argument("--judge-url", default="",
                        help="Qwen vLLM server (e.g. http://127.0.0.1:8123) for /start_judged, /step_judged")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.8-27B", help="served model name")
    parser.add_argument("--judge-threshold", type=float, default=0.5, help="keep an object iff P(yes) >= this")
    parser.add_argument("--judge-outside-scale", type=float, default=0.2,
                        help="brightness of pixels outside the mask in the judge's image 2")
    parser.add_argument("--judge-regrow", type=float, default=2.0,
                        help="re-judge a rejected id once its area is this x its last-verdict area (0 = off)")
    parser.add_argument("--judge-regrow-min-area", type=float, default=0.002,
                        help="floor (fraction of frame) for the reference area of --judge-regrow")
    parser.add_argument("--judge-timeout", type=float, default=60.0, help="seconds per Qwen request")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    state = ServiceState(args)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"SAM3 stream service listening on http://{args.host}:{args.port}; "
          f"checkpoint={args.checkpoint}", flush=True)
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
