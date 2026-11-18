#!/usr/bin/env python3
"""Optional aria2 range downloads, using the same pinned Qwen manifest and hashes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, path, runtime_env, writable
from download_public_weights import destination, load_manifest, run_lock, save_report, verify


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--connections', type=int, default=16)
    ap.add_argument('--install-tool', action='store_true', help='Create the configured private Conda aria2 environment if missing')
    args = ap.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.connections <= 16:
        ap.error('workers and connections must be in 1..16')
    binary = path('download_tools') / 'bin/aria2c'
    if not binary.exists() and args.install_tool:
        prefix = writable(path('download_tools'))
        if prefix.exists():
            raise ValueError(f'Refusing an unrecognized tool environment: {prefix}')
        env = runtime_env('download-tools')
        env['CONDA_REGISTER_ENVS'] = 'false'
        subprocess.run([str(path('conda')), 'create', '-y', '-p', str(prefix), '--override-channels',
                        '-c', 'conda-forge', '--no-default-packages', 'aria2'], env=env, check=True)
    if not binary.is_file():
        raise FileNotFoundError(f'Install aria2 in base.json paths.download_tools first: {binary}')
    state = writable(path('public_weights_state'))
    entries = [e for e in load_manifest()['files'] if e['group'] == 'qwen3_8']
    with run_lock(state):
        pending, lines, rows = [], [], []
        for entry in entries:
            target = destination(entry)
            checked = verify(target, entry)
            row = dict(path=str(target.relative_to(base_path())), **checked)
            if checked['ok']:
                row['status'] = 'verified'
                rows.append(row)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.part')
            if partial.is_symlink():
                raise ValueError('Partial file must not be a symlink')
            url = entry.get('mirrors', {}).get('modelscope', entry['url'])
            lines.extend([url, f' dir={target.parent}', f' out={partial.name}',
                          f" checksum=sha-256={entry['sha256']}"])
            pending.append((entry, target, partial, url))
        listing = state / 'aria2-input.txt'
        listing.write_text('\n'.join(lines) + '\n')
        report = dict(mode='aria2-download', complete=False, files=rows, failed=0,
                      started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save_report(state, report)
        if pending:
            command = [str(binary), '--input-file=' + str(listing), '--continue=true',
                       '--auto-file-renaming=false', '--allow-overwrite=true', '--file-allocation=none',
                       '--max-concurrent-downloads=' + str(args.workers), '--split=' + str(args.connections),
                       '--max-connection-per-server=' + str(args.connections), '--min-split-size=1M',
                       '--check-integrity=true', '--max-tries=6', '--retry-wait=3', '--timeout=60',
                       '--connect-timeout=30', '--summary-interval=30', '--enable-color=false',
                       '--console-log-level=warn', '--no-conf=true', '--no-netrc=true']
            with (state / 'aria2.log').open('a') as log:
                result = subprocess.run(command, cwd=base_path(), env=runtime_env('qwen-aria2'),
                                        stdout=log, stderr=subprocess.STDOUT)
            report['aria2_exit_code'] = result.returncode
        for entry, target, partial, url in pending:
            checked = verify(partial, entry)
            row = dict(path=str(target.relative_to(base_path())), url=url, **checked)
            if checked['ok']:
                os.replace(partial, target)
                partial.with_name(partial.name + '.aria2').unlink(missing_ok=True)
                row['status'] = 'downloaded'
            else:
                report['failed'] += 1
                row['status'] = 'failed'
            report['files'].append(row)
            save_report(state, report)
            print(json.dumps(row), flush=True)
        report['complete'] = not report['failed']
        report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_report(state, report)
        return int(bool(report['failed']))


if __name__ == '__main__':
    raise SystemExit(main())
