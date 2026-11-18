"""Qwen verification of SAM3 objects: keep an object only if Qwen says it is what its prompt meant.

Every object id a prompt produces is shown to a Qwen VLM once, on the first
frame it appears in (frame 0 for the initial objects, later frames for new
ids). A rejected id stays in the SAM3 tracker -- so the same detection keeps
matching it instead of coming back under a fresh id -- and is hidden from the
output masks. If a rejected id's mask later grows to ``regrow`` x (default 2x)
its area at the last verdict, it is judged again (a partial first view of the
object can be rejected wrongly); "yes" shows it again, "no" resets the area.

Question format (two images + text, one request per object):
  image 1  the frame, upscaled ``upscale`` x
  image 2  the same frame, pixels outside the object mask x ``outside_scale``,
           object outlined in yellow, upscaled the same way
  text     task instruction + prompt phrase + yes/no question (``QUESTION``)
The answer is read from the logprobs of the first generated token:
P(yes) = sum p(yes-like tokens) / (sum p(yes-like) + sum p(no-like)), and the
object is kept iff P(yes) >= ``threshold``.
"""

from __future__ import annotations

import base64
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tools.sam3_stream_tracker import SAM3StreamTracker, StreamSession
from tools.sam3_video_masker import SAM3VideoMasker

QUESTION = (
    "Image 1 is a frame from a robot workspace camera. Image 2 is the same frame where only one "
    "segmented region is kept visible (outlined in yellow) and everything else is darkened.\n"
    "The robot's task is: \"{instruction}\".\n"
    "For this task, the phrase \"{prompt}\" was chosen to refer to a specific object in the scene. "
    "Is the visible outlined region in image 2 that object, or a part of it? "
    "Note: the robot's own arms and grippers are never that object, unless the phrase itself names "
    "the robot, arm or gripper. Answer only yes or no."
)


def _jpeg_data_url(image_rgb: np.ndarray, cv2) -> str:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("could not JPEG-encode the judge image")
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def _chat_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")


class QwenMaskJudge:
    """Asks a vLLM-served Qwen VLM (OpenAI chat API) whether a mask is its prompt's object."""

    def __init__(
        self,
        url: str,
        *,
        model: str = "Qwen/Qwen3.8-27B",
        threshold: float = 0.5,
        outside_scale: float = 0.2,
        upscale: int = 2,
        timeout: float = 60.0,
        retries: int = 2,
        workers: int = 16,
    ) -> None:
        self.url = _chat_url(url)
        self.model = model
        self.threshold = float(threshold)
        self.outside_scale = float(outside_scale)
        self.upscale = max(1, int(upscale))
        self.timeout = float(timeout)
        self.retries = max(1, int(retries))
        self.workers = max(1, int(workers))
        self._cv2 = SAM3VideoMasker._import_cv2()

    # ---------------------------------------------------------------- images
    def _up(self, image: np.ndarray) -> np.ndarray:
        if self.upscale == 1:
            return image
        return self._cv2.resize(image, None, fx=self.upscale, fy=self.upscale,
                                interpolation=self._cv2.INTER_CUBIC)

    def frame_image(self, frame_rgb: np.ndarray) -> str:
        return _jpeg_data_url(self._up(frame_rgb), self._cv2)

    def mask_image(self, frame_rgb: np.ndarray, mask: np.ndarray) -> str:
        cv2 = self._cv2
        out = (frame_rgb.astype(np.float32) * self.outside_scale).astype(np.uint8)
        out[mask] = frame_rgb[mask]
        out = self._up(out)
        big = cv2.resize(mask.astype(np.uint8), out.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (255, 255, 0), 2)  # yellow in RGB
        return _jpeg_data_url(out, cv2)

    # ------------------------------------------------------------------- ask
    def _p_yes(self, images: Sequence[str], text: str) -> float:
        content: List[Dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in images]
        content.append({"type": "text", "text": text})
        payload = {
            "model": self.model, "temperature": 0.0, "max_tokens": 3,
            "logprobs": True, "top_logprobs": 10,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}],
        }
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        p_yes = p_no = 0.0
        for item in result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            token = item["token"].strip().lower()
            if token == "yes":
                p_yes += math.exp(item["logprob"])
            elif token == "no":
                p_no += math.exp(item["logprob"])
        if p_yes + p_no == 0.0:
            raise ValueError("neither yes nor no among the top logprobs")
        return p_yes / (p_yes + p_no)

    def _ask_one(self, frame_url: str, mask_url: str, text: str, strict: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        error = last = None
        for attempt in range(self.retries):
            try:
                p = self._p_yes([frame_url, mask_url], text)
                return {"p_yes": round(p, 5), "keep": p >= self.threshold, "error": None,
                        "seconds": round(time.time() - t0, 3)}
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
                error, last = f"{type(exc).__name__}: {exc}", exc
                if attempt + 1 < self.retries:
                    time.sleep(1.0 * (attempt + 1))
        # strict (offline dataset builds): an unreachable / rejecting server (HTTP errors and
        # timeouts are OSError) must fail the episode instead of silently keeping everything.
        if strict and isinstance(last, OSError):
            raise RuntimeError(f"Qwen judge request failed: {error}") from last
        # No answer is not a "no": keep the object, as without the judge.
        return {"p_yes": None, "keep": True, "error": error, "seconds": round(time.time() - t0, 3)}

    def ask(self, frame_url: str, frame_rgb: np.ndarray, mask: np.ndarray, instruction: str,
            prompt: str, strict: bool = False) -> Dict[str, Any]:
        """One verdict for one mask; ``frame_url`` is :meth:`frame_image` of ``frame_rgb``."""
        return self._ask_one(frame_url, self.mask_image(frame_rgb, mask),
                             QUESTION.format(instruction=instruction, prompt=prompt), strict=strict)

    def config(self) -> Dict[str, Any]:
        """Everything that changes a verdict (for caches)."""
        import hashlib

        return {"model": self.model, "threshold": self.threshold, "outside_scale": self.outside_scale,
                "upscale": self.upscale,
                "question_sha256": hashlib.sha256(QUESTION.encode("utf-8")).hexdigest()[:16]}

    def judge(self, frame_rgb: np.ndarray, instruction: str,
              items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """``items``: [{"prompt", "mask"(H, W bool)}] -> one verdict dict per item, in order."""
        if not items:
            return []
        frame_url = self.frame_image(frame_rgb)
        jobs = [(self.mask_image(frame_rgb, it["mask"]),
                 QUESTION.format(instruction=instruction, prompt=it["prompt"])) for it in items]
        with ThreadPoolExecutor(min(self.workers, len(jobs))) as pool:
            return list(pool.map(lambda job: self._ask_one(frame_url, *job), jobs))


class JudgedStreamSession(StreamSession):
    """:class:`StreamSession` whose objects are judged by :class:`QwenMaskJudge`.

    Per frame: ``step`` / construction run SAM3 as usual (rejected ids already
    hidden), :meth:`pending` lists the visible ids not judged yet plus the
    rejected ids that have grown enough to be judged again, the caller judges
    them (no GPU, no session lock needed), and :meth:`apply` records the
    verdicts and returns the frame's union mask without the rejected ids.

    Re-judging: a rejected id may first appear as a small piece of its object
    (e.g. only the hammer's tip) and grow into the whole object later. Its raw
    mask is therefore still read every frame, and once its area reaches
    ``regrow`` x its area at the last verdict it is judged again. A new "no"
    resets that reference area; a "yes" un-hides the id. ``regrow <= 0``
    disables re-judging (each id is judged exactly once).
    """

    def __init__(self, tracker: SAM3StreamTracker, frame_rgb: np.ndarray,
                 prompts: Sequence[str], instruction: str, regrow: float = 2.0,
                 regrow_min_area: float = 0.002) -> None:
        self.instruction = str(instruction)
        self.regrow = float(regrow)
        # Reference areas below this (fraction of the frame) count as this, so a
        # flickering speck does not trigger a re-judge by "doubling".
        self.regrow_min_area = float(regrow_min_area)
        self._rejected: List[set] = []
        self._verdicts: Dict[tuple, Dict[str, Any]] = {}  # (prompt_idx, obj_id) -> latest verdict
        self._frame_objects: List[List[tuple]] = []  # per prompt: [(obj_id, mask)] shown, last frame
        self._frame_rejected: List[List[tuple]] = []  # per prompt: [(obj_id, mask)] hidden, last frame
        self.verdict_log: List[Dict[str, Any]] = []
        super().__init__(tracker, frame_rgb, prompts)

    def _advance(self, i: int, frame_idx: int) -> np.ndarray:
        # Same as StreamSession._advance, plus the rejected ids are hidden and
        # the per-object masks are kept for pending() / apply().
        while len(self._rejected) <= i:
            self._rejected.append(set())
            self._frame_objects.append([])
            self._frame_rejected.append([])
        model = self.tracker.model
        state = self.states[i]
        out = model._run_single_frame_inference(state, frame_idx, reverse=False)
        self._removed[i].update(out["removed_obj_ids"])
        suppressed = out["suppressed_obj_ids"]
        unconfirmed = out.get("unconfirmed_obj_ids", None)
        hidden = self._removed[i] | self._rejected[i]
        post = model._postprocess_output(state, out, hidden, suppressed, unconfirmed)
        cv2 = SAM3VideoMasker._import_cv2()
        # Raw masks of the rejected ids still tracked, for the regrow check.
        # Read before the frame's outputs are evicted; the shown masks above are
        # unchanged (rejected ids stay out of the non-overlap step as before).
        rejected_objects = []
        if self.regrow > 0 and self._rejected[i]:
            skip = self._removed[i] | set(suppressed or ()) | set(unconfirmed or ())
            for obj_id, mask in out["obj_id_to_mask"].items():
                obj_id = int(obj_id)
                if obj_id in self._rejected[i] and obj_id not in skip:
                    mask = SAM3VideoMasker._merge_masks(mask.cpu().numpy(), self.height, self.width, cv2)
                    if mask.any():
                        rejected_objects.append((obj_id, mask))
        self._frame_rejected[i] = rejected_objects
        state["cached_frame_outputs"].pop(frame_idx, None)
        self.tracker._prune(state, frame_idx)
        objects = []
        for obj_id, mask in zip(post["out_obj_ids"], post["out_binary_masks"]):
            objects.append((int(obj_id), SAM3VideoMasker._merge_masks(mask, self.height, self.width, cv2)))
        self._frame_objects[i] = objects
        self.last_object_counts[i] = len(objects)
        return SAM3VideoMasker._merge_masks(post["out_binary_masks"], self.height, self.width, cv2)

    def pending(self) -> List[Dict[str, Any]]:
        """Objects of the last frame to judge: visible ids with no verdict yet,
        then rejected ids whose area reached ``regrow`` x their last-verdict area."""
        items = [{"prompt_idx": i, "prompt": self.prompts[i], "obj_id": obj_id, "mask": mask,
                  "rejudge": False}
                 for i, objects in enumerate(self._frame_objects)
                 for obj_id, mask in objects if (i, obj_id) not in self._verdicts]
        if self.regrow > 0:
            for i, objects in enumerate(self._frame_rejected):
                for obj_id, mask in objects:
                    ref = max(self._verdicts[(i, obj_id)]["area"], self.regrow_min_area)
                    if float(mask.mean()) >= self.regrow * ref:
                        items.append({"prompt_idx": i, "prompt": self.prompts[i], "obj_id": obj_id,
                                      "mask": mask, "rejudge": True})
        return items

    def apply(self, items: Sequence[Dict[str, Any]], verdicts: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Record verdicts for ``items`` (from :meth:`pending`) -> last frame's union mask."""
        frame_index = self.num_frames - 1
        for item, verdict in zip(items, verdicts):
            i, obj_id = item["prompt_idx"], item["obj_id"]
            entry = {"frame_index": frame_index, "prompt": item["prompt"], "obj_id": obj_id,
                     "area": round(float(item["mask"].mean()), 5),
                     "rejudge": bool(item.get("rejudge", False)), **verdict}
            self._verdicts[(i, obj_id)] = entry
            self.verdict_log.append(entry)
            if not verdict["keep"]:
                self._rejected[i].add(obj_id)
            else:
                self._rejected[i].discard(obj_id)
        union = np.zeros((self.height, self.width), dtype=bool)
        for i, objects in enumerate(self._frame_objects):
            # An id accepted on re-judge is only in _frame_rejected this frame
            # (SAM3 hid it); from the next frame on it is a normal shown object.
            revived = [(obj_id, mask) for obj_id, mask in self._frame_rejected[i]
                       if obj_id not in self._rejected[i]]
            kept = [mask for obj_id, mask in objects + revived if obj_id not in self._rejected[i]]
            self.last_object_counts[i] = len(kept)
            for mask in kept:
                union |= mask
        return union

    def rejected_ids(self) -> List[List[int]]:
        return [sorted(r) for r in self._rejected]


def start_judged_session(tracker: SAM3StreamTracker, frame_rgb: np.ndarray,
                         prompts: Sequence[str], instruction: str, regrow: float = 2.0,
                         regrow_min_area: float = 0.002) -> JudgedStreamSession:
    """Like :meth:`SAM3StreamTracker.start`, but returns a :class:`JudgedStreamSession`."""
    tracker.load()
    tracker._install_backbone_cache()
    return JudgedStreamSession(tracker, np.ascontiguousarray(frame_rgb), prompts, instruction,
                               regrow=regrow, regrow_min_area=regrow_min_area)


# ------------------------------------------------------------- offline (whole-video) replay
def judge_track(areas: Sequence[float], ask, regrow: float = 2.0,
                regrow_min_area: float = 0.002) -> List[Dict[str, Any]]:
    """Causal replay of :class:`JudgedStreamSession`'s rule for ONE object of an offline track.

    ``areas[t]`` is the object's shown area at frame ``t`` (fraction of the frame, 0 when
    SAM3 does not output it); ``ask(t)`` returns a judge verdict for its mask at frame ``t``.
    Same decisions as the stream session: judged on the first frame it is shown; once
    rejected, judged again on the first later frame whose area reaches ``regrow`` x the area
    of the last verdict (floored at ``regrow_min_area``); a "yes" ends the chain (accepted
    ids are never judged again). Only frames up to ``t`` decide frame ``t``.

    Difference to streaming, by construction of offline SAM3: a rejected object still takes
    part in SAM3's per-frame non-overlap step, so its area here is post-non-overlap.
    """
    verdicts: List[Dict[str, Any]] = []
    first = next((t for t, a in enumerate(areas) if a > 0), None)
    if first is None:
        return verdicts
    t = first
    while t is not None:
        v = dict(ask(t))
        v["frame_index"] = int(t)
        v["area"] = round(float(areas[t]), 5)  # stored like JudgedStreamSession.apply
        verdicts.append(v)
        if v["keep"] or regrow <= 0:
            break
        ref = max(v["area"], regrow_min_area)
        t = next((u for u in range(t + 1, len(areas)) if areas[u] > 0 and areas[u] >= regrow * ref), None)
    return verdicts


def keep_timeline(verdicts: Sequence[Dict[str, Any]], n_frames: int) -> np.ndarray:
    """(n_frames,) bool: frame ``t`` shows the object iff its latest verdict at or before ``t``
    is a keep. Before the first verdict the object is not shown by SAM3 anyway."""
    keep = np.zeros(n_frames, dtype=bool)
    for k, v in enumerate(verdicts):
        end = verdicts[k + 1]["frame_index"] if k + 1 < len(verdicts) else n_frames
        keep[v["frame_index"]:end] = bool(v["keep"])
    return keep
