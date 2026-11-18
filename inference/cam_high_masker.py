"""Per-episode cam_high masking: mask-Qwen prompts once, SAM3 tracks every frame.

Sim side (RoboTwin env). Stdlib + numpy + cv2 only: Qwen goes through
``auto_mask.pipeline.extract_operation_objects`` (urllib), SAM3 through the
``sam3/tools/sam3_stream_server.py`` HTTP service.

Pixels outside the SAM3 union mask are scaled by ``outside_scale`` -- the same
``out = clip(frame * 0.2)`` rule ``auto_mask/pipeline_lerobot.py`` applied when
it built the ``*_sam_single`` / ``*_sam_multi`` training sets.

Optional, both off by default (= the behaviour above):
- ``judge``: every SAM3 object is judged by mask-Qwen on the SAM3 server
  (/start_judged, /step_judged; the server needs ``--judge-url``), as in the
  ``*_sam_judged_*`` training sets.
- ``edge_width`` > 0: ``auto_mask/mask_style.py`` outline rendering (white ring
  of that width just outside the mask, outside x ``outside_scale``), as in the
  ``composite_lerobot.py`` restyled sets (``*_outline<scale>_w<width>``).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Locate the configuration; all project paths are then resolved from base.json.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "base.json").is_file():
        sys.path.insert(0, str(_candidate))
        break
from submit_paths import base_path, path as configured_path, interpreter, runtime_env, writable

_PROJECT_ROOT = base_path()
_AUTO_MASK = _PROJECT_ROOT / "auto_mask"

FALLBACK_PROMPTS = ["robot", "robot arm"]


def _import_mask_qwen():
    """auto_mask.pipeline's Qwen client (stdlib-only; shares the v3.5 prompt template)."""
    if str(_AUTO_MASK) not in sys.path:
        sys.path.insert(0, str(_AUTO_MASK))
    import pipeline  # auto_mask/pipeline.py

    return pipeline


def jpeg_data_url(frame_rgb: np.ndarray) -> str:
    """Same encoding as ``auto_mask.pipeline.first_frame_data_url``: BGR frame, JPEG q95."""
    import cv2

    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("could not JPEG-encode the first frame")
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def norm_phrase(phrase: str) -> str:
    # Identical to pipeline_lerobot.norm_phrase so eval prompts match training.
    return " ".join(phrase.split()).lower()


def dim_outside(frame: np.ndarray, keep: np.ndarray, outside_scale: float) -> np.ndarray:
    out = frame.copy()
    outside = ~keep
    out[outside] = np.clip(out[outside].astype(np.float32) * outside_scale, 0, 255).astype(np.uint8)
    return out


def _import_mask_style():
    """auto_mask/mask_style.py: the renderer of the *_sam_judged_* / composite_lerobot datasets."""
    if str(_AUTO_MASK) not in sys.path:
        sys.path.insert(0, str(_AUTO_MASK))
    import mask_style  # auto_mask/mask_style.py

    return mask_style


class Sam3StreamClient:
    """Thin HTTP client for ``sam3_stream_server.py``."""

    def __init__(self, url: str, timeout: float = 600.0, client_id: Optional[str] = None) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = client_id or f"{os.uname().nodename}-{os.getpid()}-{uuid.uuid4().hex[:6]}"

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.url + route, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SAM3 stream HTTP {exc.code} on {route}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach SAM3 stream service at {self.url}: {exc.reason}") from exc
        if not result.get("ok"):
            raise RuntimeError(f"SAM3 stream {route} failed: {result.get('error', result)}")
        return result

    @staticmethod
    def _frame_payload(frame_rgb: np.ndarray) -> Dict[str, Any]:
        frame = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        return {"frame": base64.b64encode(frame.tobytes()).decode("ascii"), "shape": list(frame.shape)}

    @staticmethod
    def _mask(result: Dict[str, Any]) -> np.ndarray:
        h, w = result["shape"]
        bits = np.frombuffer(base64.b64decode(result["mask"]), dtype=np.uint8)
        return np.unpackbits(bits, count=h * w).reshape(h, w).astype(bool)

    def wait_ready(self, deadline_s: float = 900.0) -> None:
        end = time.time() + deadline_s
        while True:
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=10) as response:
                    if json.loads(response.read().decode("utf-8")).get("loaded"):
                        return
            except Exception:
                pass
            if time.time() > end:
                raise RuntimeError(f"SAM3 stream service at {self.url} not ready after {deadline_s}s")
            time.sleep(5)

    def start(self, frame_rgb: np.ndarray, prompts: List[str], instruction: Optional[str] = None):
        """``instruction`` given: a Qwen-judged session (/start_judged; server needs --judge-url)."""
        payload = {"client_id": self.client_id, "prompts": prompts, **self._frame_payload(frame_rgb)}
        if instruction is None:
            result = self._post("/start", payload)
        else:
            result = self._post("/start_judged", {**payload, "instruction": instruction})
        return result["session_id"], self._mask(result), result

    def step(self, session_id: str, frame_rgb: np.ndarray, judged: bool = False):
        result = self._post("/step_judged" if judged else "/step",
                            {"session_id": session_id, **self._frame_payload(frame_rgb)})
        return self._mask(result), result

    def end(self, session_id: str) -> None:
        try:
            self._post("/end", {"session_id": session_id})
        except Exception as exc:  # a stale session is reclaimed by client_id / idle timeout
            print(f"[sam-mask] warning: /end failed ({exc})", flush=True)


class CamHighMasker:
    """Episode lifecycle: ``begin`` (Qwen + SAM3 start) -> ``observe`` per frame -> ``end``.

    Every frame passed to :meth:`begin` / :meth:`observe` is fed to SAM3 once,
    in order, and the dimmed frame is returned. Which frames get fed (only the
    ones the VLA sees, or every executed step) is the caller's decision.
    """

    def __init__(
        self,
        *,
        qwen_url: str,
        sam_url: str,
        outside_scale: float = 0.2,
        edge_width: int = 0,
        judge: bool = False,
        qwen_timeout: float = 120.0,
        qwen_retries: int = 3,
        sam_timeout: float = 600.0,
        save_dir: Optional[Path] = None,
        save_video: bool = True,
        video_fps: int = 10,
        log_prefix: str = "[sam-mask]",
    ) -> None:
        if not 0.0 <= outside_scale <= 1.0:
            raise ValueError("outside_scale must be between 0 and 1")
        if int(edge_width) != edge_width or edge_width < 0:
            raise ValueError("edge_width must be an integer >= 0")
        self.qwen_url = qwen_url
        self.outside_scale = float(outside_scale)
        # edge_width 0: plain dim (the original eval); > 0: the outline style of
        # composite_lerobot.py (white ring of this width just outside the mask).
        self.edge_width = int(edge_width)
        self._style = (_import_mask_style().MaskStyle("outline", self.outside_scale, self.edge_width)
                       if self.edge_width > 0 else None)
        # judge: Qwen-judged SAM3 objects (/start_judged + /step_judged), as in *_sam_judged_*.
        self.judge = bool(judge)
        self.qwen_timeout = float(qwen_timeout)
        self.qwen_retries = max(1, int(qwen_retries))
        self.sam = Sam3StreamClient(sam_url, timeout=sam_timeout)
        self.save_dir = writable(save_dir) if save_dir else None
        self.save_video = bool(save_video) and self.save_dir is not None
        self.video_fps = int(video_fps)
        self.log_prefix = log_prefix
        self._pipeline = _import_mask_qwen()
        self._episode: Optional[Dict[str, Any]] = None

    def log(self, msg: str) -> None:
        print(f"{self.log_prefix} {msg}", flush=True)

    # ---------------------------------------------------------------- qwen
    def _qwen_prompts(self, instruction: str, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """-> {"prompts", "raw", "fallback", "error", "seconds"}; never raises."""
        t0 = time.time()
        url = jpeg_data_url(frame_rgb)
        raw, phrases, error = "", [], None
        for attempt in range(self.qwen_retries):
            try:
                raw, phrases = self._pipeline.extract_operation_objects(
                    instruction, api_url=self.qwen_url, first_frame_url=url, timeout=self.qwen_timeout)
                error = None
                break
            except ValueError as exc:  # split_object_phrases rejected the answer: no retry
                error = f"{type(exc).__name__}: {exc}"
                break
            except (RuntimeError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if "truncated" in str(exc):
                    break
                time.sleep(3 * (attempt + 1))
        prompts = list(dict.fromkeys(p for p in (norm_phrase(x) for x in phrases) if p))
        fallback = not prompts
        if fallback:
            prompts = list(FALLBACK_PROMPTS)
            if error is None:
                error = "Qwen returned NONE"
        return {"prompts": prompts, "raw": raw, "fallback": fallback, "error": error,
                "seconds": round(time.time() - t0, 3)}

    # ------------------------------------------------------------- episode
    @property
    def active(self) -> bool:
        return self._episode is not None

    def begin(self, instruction: str, frame_rgb: np.ndarray) -> np.ndarray:
        if self._episode is not None:
            self.end(None)
        qwen = self._qwen_prompts(instruction, frame_rgb)
        if qwen["fallback"]:
            self.log(f"mask-qwen fallback to {qwen['prompts']} ({qwen['error']})")
        t0 = time.time()
        session_id, mask, result = self.sam.start(frame_rgb, qwen["prompts"],
                                                  instruction if self.judge else None)
        sam_seconds = time.time() - t0
        self._episode = {
            "instruction": instruction, "qwen": qwen, "session_id": session_id,
            "frames": [], "area": [], "sam_seconds": [], "objects": [], "judged": [],
            "shape": list(frame_rgb.shape), "started": time.time(),
        }
        self.log(f"episode start prompts={qwen['prompts']} qwen={qwen['seconds']}s "
                 f"sam_start={sam_seconds:.2f}s")
        return self._record(frame_rgb, mask, sam_seconds, result)

    def observe(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self._episode is None:
            raise RuntimeError("observe() before begin()")
        t0 = time.time()
        mask, result = self.sam.step(self._episode["session_id"], frame_rgb, judged=self.judge)
        return self._record(frame_rgb, mask, time.time() - t0, result)

    def render(self, frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._style is None:
            return dim_outside(frame_rgb, mask, self.outside_scale)
        return self._style.render(frame_rgb[None], mask[None])[0]

    def _record(self, frame_rgb, mask, seconds, result) -> np.ndarray:
        dimmed = self.render(frame_rgb, mask)
        ep = self._episode
        ep["area"].append(round(float(mask.mean()), 5))
        ep["sam_seconds"].append(round(float(seconds), 4))
        ep["objects"].append(result.get("objects"))
        frame_index = len(ep["area"]) - 1
        for verdict in result.get("judged") or ():  # judge verdicts of this frame
            ep["judged"].append({"step": frame_index, **verdict})
        if self.save_video:
            ep["frames"].append(dimmed)
        return dimmed

    def end(self, episode_index: Optional[int], success: Optional[bool] = None) -> None:
        """Close the SAM3 session and write ``episode{N}_{tag}_cam_high_masked.{mp4,json}``."""
        ep, self._episode = self._episode, None
        if ep is None:
            return
        self.sam.end(ep["session_id"])
        if self.save_dir is None or episode_index is None:
            return
        tag = "unknown" if success is None else ("success" if success else "failure")
        stem = f"episode{episode_index}_{tag}_cam_high_masked"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        sam_s = ep["sam_seconds"]
        meta = {
            "episode": episode_index, "result": tag, "instruction": ep["instruction"],
            "prompt_version": getattr(self._pipeline, "PROMPT_VERSION", None),
            "mask_qwen_model": getattr(self._pipeline, "MODEL_NAME", None),
            "mask_qwen_raw": ep["qwen"]["raw"], "sam3_prompts": ep["qwen"]["prompts"],
            "fallback": ep["qwen"]["fallback"], "mask_qwen_error": ep["qwen"]["error"],
            "mask_qwen_seconds": ep["qwen"]["seconds"], "outside_scale": self.outside_scale,
            **({"edge_width": self.edge_width} if self.edge_width else {}),
            **({"judge": True, "judge_verdicts": ep["judged"]} if self.judge else {}),
            "frames": len(sam_s), "sam_seconds_total": round(sum(sam_s), 3),
            "sam_seconds_mean": round(sum(sam_s) / max(1, len(sam_s)), 4),
            "mask_area_fraction": ep["area"], "sam_objects_per_prompt": ep["objects"],
            "wall_seconds": round(time.time() - ep["started"], 2),
        }
        (self.save_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        if self.save_video and ep["frames"]:
            self._write_video(self.save_dir / f"{stem}.mp4", ep["frames"])

    def _write_video(self, path: Path, frames: List[np.ndarray]) -> None:
        # Encode from memory with communicate(): no long-lived pipe that can
        # stall the sim when the shared output mount is slow.
        h, w = frames[0].shape[:2]
        tmp = Path(runtime_env("inference-client")["TMPDIR"]) / f"sam_mask_{os.getpid()}_{uuid.uuid4().hex[:8]}.mp4"
        cmd = [str(configured_path("ffmpeg")), "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{w}x{h}", "-framerate", str(self.video_fps), "-i", "-",
               "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", str(tmp)]
        try:
            data = b"".join(np.ascontiguousarray(f, dtype=np.uint8).tobytes() for f in frames)
            subprocess.run(cmd, input=data, check=True, timeout=600)
            import shutil

            shutil.move(str(tmp), str(path))
        except Exception as exc:
            self.log(f"warning: could not write {path.name}: {exc}")
        finally:
            if tmp.exists():
                tmp.unlink()
