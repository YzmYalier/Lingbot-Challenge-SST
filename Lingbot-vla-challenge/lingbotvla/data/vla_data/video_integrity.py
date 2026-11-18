"""Episode-local video bounds, independent of the decoder's file-level clamp."""

import warnings

import numpy as np


class VideoIntegrityError(ValueError):
    """A deterministic dataset defect; retrying a random sample cannot fix it."""


def validate_video_segments(meta, mode="error"):
    if mode not in {"error", "clamp"}:
        raise ValueError("video_integrity must be 'error' or 'clamp'")
    fps = float(meta.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise VideoIntegrityError(f"Invalid video fps: {fps}")
    fields = ["episode_index", "length"]
    for key in meta.video_keys:
        fields.extend(f"videos/{key}/{part}" for part in ("from_timestamp", "to_timestamp"))
    rows = meta.episodes.select_columns(fields)[:]
    lengths = np.asarray(rows["length"])
    if np.any(lengths <= 0):
        raise VideoIntegrityError("Episode lengths must be positive")
    problems = []
    for key in meta.video_keys:
        start = np.asarray(rows[f"videos/{key}/from_timestamp"], dtype=float) * fps
        stop = np.asarray(rows[f"videos/{key}/to_timestamp"], dtype=float) * fps
        if (not np.isfinite(start).all() or not np.isfinite(stop).all()
                or np.any(start < 0) or np.any(stop <= start)
                or np.any(np.abs(start - np.rint(start)) > 1e-3)
                or np.any(np.abs(stop - np.rint(stop)) > 1e-3)):
            raise VideoIntegrityError(f"{meta.root}: invalid frame boundaries for {key}")
        size = np.rint(stop).astype(np.int64) - np.rint(start).astype(np.int64)
        bad = np.flatnonzero(size != lengths)
        if len(bad):
            examples = [(int(rows["episode_index"][i]), int(lengths[i]), int(size[i])) for i in bad[:3]]
            problems.append(f"{key}: {len(bad)} mismatches, (episode,length,video_frames)={examples}")
    if problems:
        msg = (f"{meta.root}: video/trajectory mismatch. " + "; ".join(problems)
               + ". Rebuild from complete source videos. data.video_integrity=clamp only prevents "
               "cross-episode reads; it does not repair missing frames or temporal alignment.")
        if mode == "error":
            raise VideoIntegrityError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def episode_video_timestamps(episode, key, timestamps, fps, mode="error"):
    """Snap relative timestamps to frame indices and enforce the episode's bounds."""
    if mode not in {"error", "clamp"}:
        raise ValueError("video_integrity must be 'error' or 'clamp'")
    start = round(float(episode[f"videos/{key}/from_timestamp"]) * fps)
    stop = round(float(episode[f"videos/{key}/to_timestamp"]) * fps)
    if stop <= start:
        raise VideoIntegrityError(f"Empty video segment for {key}")
    result = []
    for ts in timestamps:
        if not np.isfinite(ts):
            raise VideoIntegrityError(f"Non-finite query timestamp for {key}: {ts}")
        index = start + round(float(ts) * fps)
        if mode == "error" and not start <= index < stop:
            raise VideoIntegrityError(f"{key}: frame {index} outside episode [{start},{stop})")
        result.append(max(start, min(stop - 1, index)) / fps)
    return result
