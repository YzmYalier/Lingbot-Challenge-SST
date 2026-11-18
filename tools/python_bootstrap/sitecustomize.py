"""Ignore stale editable checkouts in copied or externally supplied Conda envs.

Installed third-party dependencies and the standard library remain available.
Project modules are supplied exclusively through the launcher's PYTHONPATH.
"""
import os
from pathlib import Path
import sys

root = Path(os.environ["SUBMISSION_ROOT"]).resolve()
allowed = (root, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())


def permitted(entry):
    if not entry or entry.startswith("__editable__"):
        return False
    candidate = Path(entry).resolve()
    return any(candidate.is_relative_to(parent) for parent in allowed)


sys.path[:] = [entry for entry in sys.path if permitted(entry)]
sys.meta_path[:] = [finder for finder in sys.meta_path
                    if not getattr(finder, "__module__", "").startswith("__editable__")]
sys.path_hooks[:] = [hook for hook in sys.path_hooks
                     if not getattr(hook, "__module__", "").startswith("__editable__")]
