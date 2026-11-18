#!/usr/bin/env python3
"""Repair raw v3.0, then reuse verified TempSAM masks and rerun only damaged IDs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, interpreter, path, runtime_env, writable
from download_public_weights import load_manifest, verify
from process_data import check_raw, file_hash, run, write_json


def extract_original():
    """Keep the full old RGB videos for frame-by-frame alignment, not just metadata."""
    archive = path('zip_v30')
    entry = next(e for e in load_manifest()['files'] if e['destination']['path_key'] == 'zip_v30')
    result = verify(archive, entry)
    if not result['ok']:
        raise ValueError(f'v30 ZIP is not the pinned source: {result}')
    target = writable(path('raw_original'))
    stage = writable(target.with_name(target.name + '.partial'))
    marker_name = '.extraction_manifest.json'
    identity = {'archive_sha256': entry['sha256'], 'archive_size': entry['size']}
    if target.exists():
        stamp = json.loads((target / marker_name).read_text())
        if stamp.get('identity') != identity or not stamp.get('complete'):
            raise ValueError('Unrecognized original-v30 extraction; choose a new configured output')
        for relative, digest in stamp['files'].items():
            if file_hash(target / relative) != digest:
                raise ValueError(f'Original v30 extraction changed: {relative}')
        return target
    if stage.exists():
        stamp = json.loads((stage / marker_name).read_text())
        if stamp.get('identity') != identity:
            raise ValueError('Refusing an unrecognized original-v30 partial extraction')
    else:
        stage.mkdir(parents=True)
        stamp = {'identity': identity, 'complete': False, 'files': {}}
        write_json(stage / marker_name, stamp)
    with zipfile.ZipFile(archive) as z:
        infos = [n for n in z.namelist() if n.endswith('meta/info.json')]
        if len(infos) != 1:
            raise ValueError('Expected exactly one original v30 dataset')
        info = json.loads(z.read(infos[0]))
        if (info['codebase_version'], info['total_episodes'], info['total_frames'], info['fps']) != ('v3.0', 2500, 548893, 15):
            raise ValueError('Unexpected original v30 metadata')
        prefix = infos[0][:-len('meta/info.json')]
        members = [m for m in z.infolist() if not m.is_dir() and m.filename.startswith(prefix)]
        seen = set()
        for i, member in enumerate(members, 1):
            relative = member.filename[len(prefix):]
            rel = PurePosixPath(relative)
            if rel.is_absolute() or '..' in rel.parts or '\\' in relative or relative in seen:
                raise ValueError(f'Unsafe/duplicate archive member: {relative}')
            seen.add(relative)
            if rel.parts[0] not in ('data', 'meta', 'videos'):
                continue
            out = writable(stage / relative)
            if not out.is_relative_to(stage):
                raise ValueError('Archive path escapes staging directory')
            if out.is_file() and stamp['files'].get(relative) == file_hash(out):
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_name(out.name + '.tmp')
            digest = hashlib.sha256()
            with z.open(member) as source, temporary.open('wb') as dest:
                for chunk in iter(lambda: source.read(4 * 1024**2), b''):
                    digest.update(chunk)
                    dest.write(chunk)
            temporary.replace(out)
            stamp['files'][relative] = digest.hexdigest()
            write_json(stage / marker_name, stamp)
            if i % 25 == 0 or i == len(members):
                print(f'Extracted original v30: {i}/{len(members)} members', flush=True)
    stamp['complete'] = True
    write_json(stage / marker_name, stamp)
    stage.rename(target)
    return target


def validate_locations():
    outputs = [writable(path(k)) for k in ('raw_original', 'raw', 'sam_output', 'sam_reuse_work', 'sam_reuse_smoke')]
    protected = [path(k) for k in ('sam_reuse_source', 'zip_v21', 'zip_v30', 'instructions')]
    for i, dest in enumerate(outputs):
        for other in protected + outputs[:i]:
            if dest.is_relative_to(other) or other.is_relative_to(dest):
                raise ValueError(f'Overlapping output/source locations: {dest}, {other}')


def reuse_command(args, stage):
    work = path('sam_reuse_smoke') if args.smoke_episodes else path('sam_reuse_work')
    command = [interpreter('data'), '-B', base_path() / 'tools/rebuild_robotwin_sam.py',
               '--raw', path('raw'), '--old-sam', path('sam_reuse_source'),
               '--out', path('sam_output'), '--work', work,
               '--pipeline', base_path() / 'auto_mask/pipeline_lerobot.py', '--stage', stage,
               '--allow-source-relocation', '--rerender-style',
               '--compose-workers', str(args.compose_workers), '--encoder-threads', '2']
    if args.smoke_episodes:
        command += ['--smoke-episodes', args.smoke_episodes]
    if args.qwen_url:
        command += ['--qwen-url', args.qwen_url]
    if args.sam_gpus:
        command += ['--sam3-gpus', args.sam_gpus]
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['download', 'raw', 'plan', 'reuse', 'masks', 'assemble', 'validate', 'all'], default='all')
    ap.add_argument('--workers', type=int, default=config()['processing']['workers'])
    ap.add_argument('--compose-workers', type=int, default=config()['processing']['compose_workers'])
    ap.add_argument('--qwen-url', default='')
    ap.add_argument('--sam-gpus', default='')
    ap.add_argument('--smoke-episodes', default='', help='CPU reuse only; isolated work; never publishes a subset')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.workers < 1 or args.compose_workers < 1:
        ap.error('Worker counts must be positive')
    if args.smoke_episodes and args.stage != 'reuse':
        ap.error('--smoke-episodes is only allowed with --stage reuse')
    validate_locations()
    if args.dry_run:
        print(json.dumps({k: str(path(k)) for k in ('zip_v21', 'zip_v30', 'raw_original', 'raw',
                          'sam_reuse_source', 'sam_reuse_work', 'sam_output')}, indent=2))
        print('download -> raw repair -> alignment and CPU mask reuse -> damaged-ID SAM/judge -> full assembly -> validation')
        return
    executable = interpreter('data')
    if Path(sys.executable).absolute() != executable:
        env = runtime_env('incremental-repair')
        os.execve(str(executable), [str(executable), '-B', str(Path(__file__).resolve()), *sys.argv[1:]], env)
    env = runtime_env('incremental-repair')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    lock_dir = writable(path('sam_reuse_report').parent)
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ('download', 'all'):
            run([executable, '-B', base_path() / 'download_public_weights.py', '--only', 'robotwin_data'], 'incremental-download', env)
        if args.stage in ('raw', 'all'):
            if path('raw').exists():
                check_raw()
            else:
                original = extract_original()
                entry = next(e for e in load_manifest()['files'] if e['destination']['path_key'] == 'zip_v21')
                if not verify(path('zip_v21'), entry)['ok']:
                    raise ValueError('v21 ZIP is not the pinned complete source')
                run([executable, '-B', base_path() / 'tools/rebuild_robotwin_v30_from_v21.py',
                     '--source-v21', path('zip_v21'), '--reference-v30', original,
                     '--out', path('raw'), '--workers', str(args.workers)], 'incremental-raw', env)
                check_raw()
        stages = ['reuse', 'masks', 'assemble', 'validate'] if args.stage == 'all' else [args.stage]
        for stage in stages:
            if stage in ('plan', 'reuse', 'masks', 'assemble'):
                check_raw()
                if stage == 'masks':
                    plan_file = path('sam_reuse_work') / 'incremental_plan.json'
                    if not plan_file.is_file():
                        ap.error('Run --stage reuse before --stage masks')
                    plan = json.loads(plan_file.read_text())
                    if plan['recompute_episodes'] and (not args.qwen_url or not args.sam_gpus):
                        ap.error(f"{plan['recompute_count']} episodes require SAM/judge; provide --qwen-url and --sam-gpus. CPU work is retained.")
                run(reuse_command(args, stage), 'incremental-' + stage, env)
            elif stage == 'validate':
                run([executable, '-B', base_path() / 'tools/validate_outputs.py'], 'incremental-validate', env)


if __name__ == '__main__':
    main()
