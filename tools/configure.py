#!/usr/bin/env python3
"""Relocate the bundle and configure interpreters without editing source code."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--relocate", action="store_true")
    roles = ("data", "sam3", "vllm", "vla", "robotwin")
    for role in roles:
        ap.add_argument(f"--{role}-python", help="Path relative to base_path, or an absolute interpreter path")
    args = ap.parse_args()
    file = Path(__file__).resolve().parents[1] / "base.json"
    cfg = json.loads(file.read_text())
    old_root = (file.parent / cfg["base_path"]).resolve()
    if args.relocate:
        cfg["base_path"] = "."
    for role in roles:
        value = getattr(args, role + "_python")
        if value:
            cfg["interpreters"][role] = value
    if args.relocate or any(getattr(args, role + "_python") for role in roles):
        file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    if args.relocate:
        # Published SAM metadata is consumed by the raw-image teacher. Rebase its
        # source after copying a completed dataset together with the submission.
        root = (file.parent / cfg["base_path"]).resolve()
        sam = root / cfg["paths"]["sam_output"]
        raw = root / cfg["paths"]["raw"]
        metadata = sam / "meta/sam_info.json"
        if metadata.is_file():
            if not sam.resolve().is_relative_to(root) or not raw.resolve().is_relative_to(root):
                raise ValueError("Published dataset must be inside base_path")
            info = json.loads(metadata.read_text())
            expected_old = old_root / cfg["paths"]["raw"]
            old_source = Path(info["source"])
            raw_relative = Path(cfg["paths"]["raw"])
            # With base_path='.', the old machine prefix is no longer stored in
            # base.json. Recognize the same published dataset by its suffix.
            moved_relative_source = (not raw_relative.is_absolute() and
                                     old_source.parts[-len(raw_relative.parts):] == raw_relative.parts)
            if old_source not in (expected_old, raw) and not moved_relative_source:
                raise ValueError("Existing SAM source does not match the configured raw dataset")
            if not json.loads((raw / "repair_report.json").read_text())["complete"]:
                raise ValueError("Relocated raw dataset is incomplete")
            info["source"] = str(raw)
            metadata.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
