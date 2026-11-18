#!/usr/bin/env python3
"""Restore pinned public models and dataset ZIPs, checking MD5 and SHA-256.

Uses Python's standard library; run --help for offline verification and selection.
All paths resolve through the base.json next to this script, never the caller's cwd.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
from submit_paths import base_path, path, writable

CHUNK_SIZE = 8 * 1024 * 1024


class DownloadError(Exception):
    pass


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a Hugging Face token to a CDN or another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise DownloadError("Refusing redirect to a non-HTTPS URL")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            old = urllib.parse.urlsplit(req.full_url)
            new = urllib.parse.urlsplit(newurl)
            if (old.scheme, old.netloc) != (new.scheme, new.netloc):
                redirected.remove_header("Authorization")
        return redirected


def destination(entry):
    spec = entry["destination"]
    relative = PurePosixPath(spec["relative"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid relative path: {relative}")
    anchor = path(spec["path_key"])
    if spec.get("parent", False):
        anchor = anchor.parent
    result = writable(anchor / str(relative))
    # These directories are outside the requested download scope.
    for excluded in ("Lingbot-vla-challenge", "RoboTwin"):
        if excluded in result.relative_to(base_path()).parts:
            raise ValueError(f"{excluded} directory is excluded")
    return result


def load_manifest():
    manifest = json.loads(path("public_weights_manifest").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported public weight manifest version")
    seen = set()
    for entry in manifest["files"]:
        target = destination(entry)
        if target in seen:
            raise ValueError(f"Duplicate destination: {target}")
        seen.add(target)
        for name, length in (("md5", 32), ("sha256", 64)):
            if not re.fullmatch(r"[0-9a-f]{%d}" % length, entry[name]):
                raise ValueError(f"Invalid {name}: {target}")
        for candidate in [entry['url'], *entry.get('mirrors', {}).values()]:
            url = urllib.parse.urlsplit(candidate)
            if url.scheme != "https" or url.hostname not in {"huggingface.co", "modelscope.cn", "raw.githubusercontent.com"}:
                raise ValueError(f"Unsupported download origin: {candidate}")
        if entry["size"] < 0:
            raise ValueError(f"Invalid size: {target}")
    return manifest


def verify(target, entry):
    if not target.is_file():
        return {"ok": False, "reason": "missing"}
    before = target.stat()
    if before.st_size != entry["size"]:
        return {"ok": False, "reason": "size mismatch", "size": before.st_size}
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    processed = 0
    last_progress = time.monotonic()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            md5.update(chunk)
            sha256.update(chunk)
            processed += len(chunk)
            if time.monotonic() - last_progress >= 20:
                print(f"  checking {processed / 2**30:.2f}/{entry['size'] / 2**30:.2f} GiB", flush=True)
                last_progress = time.monotonic()
    after = target.stat()
    stable = (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    ok = stable and md5.hexdigest() == entry["md5"] and sha256.hexdigest() == entry["sha256"]
    return {"ok": ok, "reason": "verified" if ok else "hash mismatch or file changed",
            "size": processed, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def request_file(url, offset, timeout, opener):
    headers = {"User-Agent": "to-submit-public-weights/1", "Accept-Encoding": "identity"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and urllib.parse.urlsplit(url).hostname == "huggingface.co":
        headers["Authorization"] = "Bearer " + token
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)


def download(entry, target, *, retries, timeout, opener):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.is_symlink():
        raise DownloadError(f"Partial download must be a regular file: {partial}")
    partial = writable(partial)
    last_error = "download failed"
    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= entry["size"]:
                result = verify(partial, entry)
                if result["ok"]:
                    os.replace(partial, target)
                    return result
                partial.unlink()
                offset = 0
            try:
                response = request_file(entry["url"], offset, timeout, opener)
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code in (401, 403):
                    raise DownloadError(f"HTTP {code}: access denied for {entry['url']}") from None
                if code == 416 and offset:
                    partial.unlink()
                raise OSError(f"HTTP {code} for pinned source") from None
            with response:
                code = response.status
                if code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if (not match or int(match[1]) != offset or int(match[3]) != entry["size"]
                            or not offset <= int(match[2]) < entry["size"]):
                        raise OSError("Invalid Content-Range; refusing to append")
                elif code == 200:
                    # Some servers ignore Range: restart instead of appending twice.
                    offset = 0
                else:
                    raise OSError(f"Unexpected HTTP {code}")
                last_progress = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                        if offset + len(chunk) > entry["size"]:
                            raise OSError("Response exceeds pinned file size")
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_progress >= 15:
                            print(f"  downloaded {offset / 2**30:.2f}/{entry['size'] / 2**30:.2f} GiB", flush=True)
                            last_progress = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size != entry["size"]:
                raise OSError("Incomplete response; partial file retained for resume")
            result = verify(partial, entry)
            if not result["ok"]:
                partial.unlink()
                raise OSError("MD5/SHA-256 mismatch; discarded invalid download")
            os.replace(partial, target)
            return result
        except DownloadError:
            raise
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            # Avoid logging signed CDN URLs or request headers.
            last_error = str(exc) if "https://" not in str(exc) else type(exc).__name__
            if attempt < retries:
                print(f"  retry {attempt}/{retries}: {last_error}", flush=True)
                time.sleep(min(2 ** attempt, 10))
    raise DownloadError(f"{last_error}; retries exhausted ({retries})")


@contextlib.contextmanager
def run_lock(state):
    state.mkdir(parents=True, exist_ok=True)
    with (state / "download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError("Another weight download/check is running in this bundle") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def save_report(state, report):
    temporary = state / "last_run.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, state / "last_run.json")


def fetch_entry(index, entry, total, args):
    target = destination(entry)
    relative = str(target.relative_to(base_path()))
    print(f"[{index}/{total}] {relative}", flush=True)
    row = {"path": relative, "url": entry["url"], "expected_md5": entry["md5"]}
    try:
        result = verify(target, entry)
        if result["ok"]:
            row.update(status="verified", **result)
        elif args.check:
            row.update(status="failed", **result)
        else:
            print(f"  {relative}: {result['reason']}; downloading pinned source", flush=True)
            opener = urllib.request.build_opener(SafeRedirect())
            result = download(entry, target, retries=args.retries, timeout=args.timeout, opener=opener)
            row.update(status="downloaded", **result)
        if row["ok"]:
            print(f"  OK {relative} MD5 {row['md5']}", flush=True)
        else:
            print(f"  FAIL {relative}: {row['reason']}", flush=True)
    except (DownloadError, OSError) as exc:
        row.update(status="failed", ok=False, reason=str(exc))
        print(f"  FAIL {relative}: {exc}", file=sys.stderr, flush=True)
    return row


def main(argv=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify all selected files offline; never download")
    mode.add_argument("--list", "--dry-run", dest="list_only", action="store_true", help="List pinned URLs, relative paths and MD5 without reading weights")
    parser.add_argument("--only", action="append", default=[], metavar="GROUP", help="Select a model/data group (repeatable); default: all groups, including robotwin_data ZIPs")
    parser.add_argument("--file", action="append", default=[], metavar="RELATIVE_PATH", help="Select a destination relative to the bundle (repeatable)")
    parser.add_argument("--retries", type=int, default=4, help="Attempts per file (default: 4)")
    parser.add_argument("--timeout", type=float, default=60, help="Network I/O timeout in seconds (default: 60)")
    parser.add_argument("--workers", type=int, default=1, help="Independent files downloaded in parallel (1-16; default: 1)")
    parser.add_argument("--mirror", choices=['original', 'modelscope'], default='original',
                        help='Use a pinned byte-identical mirror where listed; hashes never change')
    args = parser.parse_args(argv)
    if args.retries < 1 or args.timeout <= 0:
        parser.error("--retries and --timeout must be positive")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    manifest = load_manifest()
    groups = {group["id"] for group in manifest["sources"]}
    unknown = set(args.only) - groups
    if unknown:
        parser.error(f"Unknown groups: {', '.join(sorted(unknown))}; choose from: {', '.join(sorted(groups))}")
    entries = [e for e in manifest["files"] if not args.only or e["group"] in args.only]
    if args.mirror != 'original':
        entries = [dict(e, url=e.get('mirrors', {}).get(args.mirror, e['url'])) for e in entries]
    if args.file:
        available = {str(destination(e).relative_to(base_path())) for e in entries}
        if set(args.file) - available:
            parser.error("Unknown --file path (or outside selected groups): " + ", ".join(sorted(set(args.file) - available)))
        entries = [e for e in entries if str(destination(e).relative_to(base_path())) in args.file]
    print(f"{len(entries)} files, {sum(e['size'] for e in entries) / 2**30:.2f} GiB; root: {base_path()}", flush=True)
    if args.list_only:
        for entry in entries:
            print(f"{entry['group']}  {destination(entry).relative_to(base_path())}\n  MD5 {entry['md5']}  bytes {entry['size']}\n  {entry['url']}")
        return 0
    state = writable(path("public_weights_state"))
    report = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "mode": "check" if args.check else "download", "files": [], "complete": False}
    failed = 0
    with run_lock(state):
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(fetch_entry, i, e, len(entries), args) for i, e in enumerate(entries, 1)]
                try:
                    for future in as_completed(futures):
                        row = future.result()
                        failed += int(not row['ok'])
                        report['files'].append(row)
                        save_report(state, report)  # Only this parent writes the shared report.
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            report["complete"] = True
        finally:
            report["failed"] = failed
            report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save_report(state, report)
    print(f"Finished: {len(entries) - failed} passed, {failed} failed. Report: {state / 'last_run.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; partial downloads are retained. Rerun to resume.", file=sys.stderr)
        raise SystemExit(130)
    except (DownloadError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
