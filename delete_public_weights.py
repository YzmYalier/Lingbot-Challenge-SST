#!/usr/bin/env python3
"""Delete files managed by download_public_weights.py and their .part files.

Default: delete all manifest entries, including models, configurations/tokenizers
and the two RoboTwin input dataset ZIPs (group: robotwin_data).
Use --dry-run to preview or --only GROUP (repeatable) to select model groups.
Directories, unlisted files, RoboTwin and Lingbot-vla-challenge are preserved.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from download_public_weights import DownloadError, destination, load_manifest, run_lock
from submit_paths import base_path, config, path, writable


def checked_target(entry):
    """Reject symlinked paths before allowing an unlink."""
    spec = entry["destination"]
    anchor = Path(config()["paths"][spec["path_key"]]).expanduser()
    if not anchor.is_absolute():
        anchor = base_path() / anchor
    if spec.get("parent", False):
        anchor = anchor.parent
    lexical = Path(os.path.abspath(anchor / spec["relative"]))
    resolved = destination(entry)
    if lexical != resolved:
        raise ValueError(f"Refusing symlinked deletion target: {lexical}")
    return resolved


def validate_file(target):
    if target.is_symlink() or target.resolve() != target:
        raise ValueError(f"Refusing symlinked deletion target: {target}")
    writable(target)
    if target.exists() and not target.is_file():
        raise ValueError(f"Deletion target is not a regular file: {target}")


def build_plan(entries):
    plan = []
    seen = set()
    for entry in entries:
        target = checked_target(entry)
        for candidate in (target, target.with_name(target.name + ".part")):
            validate_file(candidate)
            if candidate in seen:
                raise ValueError(f"Duplicate deletion target: {candidate}")
            seen.add(candidate)
            plan.append(candidate)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting files")
    parser.add_argument("--only", action="append", default=[], metavar="GROUP", help="Select a model/data group (repeatable); robotwin_data selects the two input ZIPs")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    groups = {source["id"] for source in manifest["sources"]}
    if set(args.only) - groups:
        parser.error("Unknown group; choose from: " + ", ".join(sorted(groups)))
    entries = [e for e in manifest["files"] if not args.only or e["group"] in args.only]
    state = writable(path("public_weights_state"))
    print(f"Root: {base_path()}; selected manifest files: {len(entries)}", flush=True)
    with run_lock(state):
        # Validate the entire plan before deleting the first file.
        plan = build_plan(entries)
        report = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "dry_run": args.dry_run, "selected_manifest_files": len(entries),
                  "files": [], "complete": False}
        try:
            for target in plan:
                validate_file(target)
                relative = str(target.relative_to(base_path()))
                if not target.exists():
                    report["files"].append({"path": relative, "status": "missing", "bytes": 0})
                    continue
                size = target.stat().st_size
                if not args.dry_run:
                    target.unlink()
                status = "would_delete" if args.dry_run else "deleted"
                report["files"].append({"path": relative, "status": status, "bytes": size})
                print(f"{status}: {relative}", flush=True)
            report["complete"] = True
        finally:
            report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            report["deleted_files"] = sum(f["status"] == "deleted" for f in report["files"])
            report["deleted_bytes"] = sum(f["bytes"] for f in report["files"] if f["status"] == "deleted")
            if not args.dry_run:
                temporary = state / "delete_last_run.json.tmp"
                temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                os.replace(temporary, state / "delete_last_run.json")
        count = sum(f["status"] in {"deleted", "would_delete"} for f in report["files"])
        print(f"{'Would delete' if args.dry_run else 'Deleted'} {count} files.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; completed deletions are recorded in delete_last_run.json.", file=sys.stderr)
        raise SystemExit(130)
    except (DownloadError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
