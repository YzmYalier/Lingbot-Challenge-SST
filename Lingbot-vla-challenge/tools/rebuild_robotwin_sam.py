#!/usr/bin/env python3
"""Incrementally rebuild judged SAM derivatives of the repaired RoboTwin v3 dataset.

Normal cam_high episodes reuse final, judged masks after frame alignment checks.
Damaged episodes run the existing SAM3 + Qwen judge pipeline in a NEW workspace.
All rendered videos use repaired RGB. Existing datasets and pipeline code are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import fcntl
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd

# Also support importlib-based callers/tests outside this tools directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam_rebuild_progress import Progress, observe_assembly
from sam_rebuild_cpu import native_thread_limits, resolve_workers, reuse_results
from sam_rebuild_gpu import (add_arguments as gpu_arguments, compatible_single_manifest,
                             pipeline_runtime, scheduling)

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes dataset: {relative}')
    return path


@functools.lru_cache(maxsize=16384)
def check_segment(path, size, mtime_ns, expected, fps):
    """Decode each distinct encoded segment once per assembler worker."""
    count = 0
    with av.open(path) as c:
        stream = c.streams.video[0]
        stream.codec_context.thread_count = 1
        for i, frame in enumerate(c.decode(stream)):
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - i / fps) > 1e-4:
                raise ValueError(f'{path}: discontinuous timestamp at frame {i}')
            count += 1
    if count != expected:
        raise ValueError(f'{path}: {count} decoded frames, expected {expected}')


def copy_episode_segment(segments, out, n_expected):
    """Keep the repaired one-episode layout; avoid another lossy concatenation."""
    if len(segments) != 1:
        raise ValueError('Expected repaired one-episode-per-video layout; subset assembly is not supported')
    src, dest = Path(segments[0]), Path(out)
    stat = src.stat()
    with av.open(str(src)) as c:
        fps = float(c.streams.video[0].average_rate)
    check_segment(str(src), stat.st_size, stat.st_mtime_ns, n_expected, fps)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Only NEW workspace segments are linked. No links to original masks/data.
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)
    return n_expected


def import_pipeline(path):
    sys.path.insert(0, str(path.parent))
    p = importlib.import_module('pipeline_lerobot')
    if Path(p.__file__).resolve() != path:
        raise ValueError('A different pipeline_lerobot module is already imported')
    return p


def validate_pipeline(p, info):
    """Reject version/config drift instead of mixing different judge semantics."""
    judge = info.get('judge')
    if not judge or info.get('format') != 'objects-judge-v1':
        raise ValueError('Old SAM dataset must contain final objects-judge-v1 masks')
    instance = p.QwenMaskJudge('http://127.0.0.1:8000/v1/chat/completions',
                              model=judge['model'], threshold=judge['threshold'],
                              outside_scale=judge['outside_scale'])
    actual = p.judge_signature(instance, SimpleNamespace(judge_regrow=judge['regrow'],
                                                        judge_regrow_min_area=judge['regrow_min_area']))
    if (actual != judge or p.FORMAT != info['format'] or p.PROMPT_VERSION != info['prompt_version']
            or p.SAM3_ARGS != info['sam3']):
        raise ValueError('Pipeline prompt/judge/SAM settings differ from the old dataset')
    if info['style'] != p.MaskStyle('outline', 0.1, 2).to_dict():
        raise ValueError('Expected outline0.1_w2 source masks')
    return {'format': p.FORMAT, 'judge': actual, 'style': info['style']}


def make_plan(p, src, old, old_sam, info):
    if len(src.eps) != len(old.eps) or src.fps != old.fps:
        raise ValueError('Raw episode IDs/fps changed')
    for col in ('length', 'dataset_from_index', 'dataset_to_index'):
        if not np.array_equal(src.eps[col], old.eps[col]):
            raise ValueError(f'Raw episode mapping changed: {col}')
    if info.get('subset') or Path(info['source']).resolve() != old.root:
        raise ValueError('Old masks must belong to the original full v3 dataset')
    keys = p.cam_keys('multi')
    if info['mode'] != 'multi' or info['video_keys'] != keys:
        raise ValueError('Expected complete multi-instruction SAM dataset')
    index = pd.read_parquet(old_sam / 'sam_masks/index.parquet')
    if len(index) != len(src.eps) * len(keys) or index.duplicated(['episode_index', 'video_key']).any():
        raise ValueError('Incomplete/duplicate mask index')
    if set(index.episode_index) != set(src.eps.index) or not (index.episode_index == index.source_episode_index).all():
        raise ValueError('Old mask episode IDs do not match raw')
    reuse, redo, groups = [], [], {}
    for ep, rows in index.groupby('episode_index', sort=True):
        ep = int(ep)
        if set(rows.video_key) != set(keys):
            raise ValueError(f'ep {ep}: missing mask slots')
        rows = rows.set_index('video_key').loc[keys].reset_index()
        groups[ep] = rows
        n = int(src.eps.loc[ep, 'length'])
        if src.n_video_frames(ep) != n:
            raise ValueError(f'ep {ep}: repaired cam_high is still incomplete')
        if old.n_video_frames(ep) != n or not (rows.n_frames == n).all():
            redo.append(ep)
        else:
            reuse.append(ep)
    n_groups = int(index.file.nunique())
    plan = {'total_episodes': len(src.eps), 'total_frames': int(src.eps.length.sum()),
            'total_mask_groups': n_groups, 'reuse_episodes': reuse, 'recompute_episodes': redo,
            'initial_recompute_episodes': list(redo), 'alignment_failures': {}}
    update_counts(plan, src, groups)
    return plan, groups


def update_counts(plan, src, groups):
    redo = plan['recompute_episodes']
    plan.update(reuse_count=len(plan['reuse_episodes']), recompute_count=len(redo),
                recompute_frames=int(src.eps.loc[redo, 'length'].sum()),
                recompute_mask_groups=sum(int(groups[e].file.nunique()) for e in redo))
    plan['recompute_episode_percent'] = 100 * len(redo) / plan['total_episodes']
    plan['recompute_frame_percent'] = 100 * plan['recompute_frames'] / plan['total_frames']
    plan['recompute_group_percent'] = 100 * plan['recompute_mask_groups'] / plan['total_mask_groups']


def seed_prompts(p, src, work, old_sam, info, groups, episodes, progress=None):
    """Reuse language extraction only; damaged episodes never receive old judge caches."""
    prompts = pd.read_parquet(old_sam / 'meta/sam_prompts.parquet').set_index('episode_index')
    for ep in episodes:
        rows = prompts.loc[[ep]].set_index('video_key').loc[info['video_keys']]
        idx = groups[ep]
        if len(rows) != len(info['video_keys']) or not (rows.source_episode_index == ep).all():
            raise ValueError(f'ep {ep}: invalid prompt mapping')
        phrases = [list(ps) for ps in rows.phrases]
        if any(sorted(a) != sorted(b) for a, b in zip(phrases, idx.phrases)):
            raise ValueError(f'ep {ep}: prompts and final masks disagree')
        instructions = rows.instruction.tolist()
        if instructions != src.instructions(ep, 'multi'):
            raise ValueError(f'ep {ep}: instructions changed; cannot reuse prompt cache')
        q = {'episode_index': ep, 'mode': 'multi', 'prompt_version': info['prompt_version'],
             'task': src.task_name(ep), 'instructions': instructions, 'phrases': phrases,
             'raw': rows.qwen_raw.where(rows.qwen_raw.notna(), None).tolist(),
             'errors': rows.qwen_error.where(rows.qwen_error.notna(), None).tolist()}
        if work.q(ep).exists() and json.loads(work.q(ep).read_text()) != q:
            raise ValueError(f'ep {ep}: prompt cache differs from source')
        write_json(work.q(ep), q)
        if progress is not None:
            progress.advance(detail=f'episode={ep}')


def seed_judges(p, work, old_sam, signature, episodes, progress=None):
    logs = pd.read_parquet(old_sam / 'sam_masks/judge_log.parquet')
    logs = {int(ep): rows for ep, rows in logs.groupby('episode_index')}
    for ep in episodes:
        q = json.loads(work.q(ep).read_text())
        phrases = {phrase: {'instruction': ins, 'config': signature['judge'], 'verdicts': {}}
                   for phrase, ins in p.phrase_instructions(q).items()}
        rows = logs.get(ep)
        if rows is not None:
            for row in rows.to_dict('records'):
                phrase = row['phrase']
                if phrase not in phrases:
                    continue
                if row['source_episode_index'] != ep or row['instruction'] != phrases[phrase]['instruction']:
                    raise ValueError(f'ep {ep}: incompatible saved judge log')
                verdict = {k: row[k] for k in ('frame_index', 'area', 'p_yes', 'keep', 'error')}
                for k, value in verdict.items():
                    if pd.isna(value):
                        verdict[k] = None
                phrases[phrase]['verdicts'].setdefault(str(row['obj_id']), []).append(verdict)
        write_json(work.judge_json(ep), {'episode_index': ep, 'phrases': phrases,
                                        'reused_final_masks_from': str(old_sam)})
        if progress is not None:
            progress.advance(detail=f'episode={ep}')


class AlignmentError(ValueError):
    pass


def check_alignment(old, new, max_mae=6.0):
    """All frames, not just endpoints; tolerate re-encoding but reject clear shifts."""
    if old.shape != new.shape:
        raise AlignmentError(f'RGB shape differs: {old.shape} vs {new.shape}')
    errors, shifted = [], []
    for t in range(len(new)):
        # Full resolution checks avoid missing thin/moving objects.
        image = old[t].astype(np.int16)
        own = float(np.abs(image - new[t].astype(np.int16)).mean())
        errors.append(own)
        nearby = [float(np.abs(image - new[j].astype(np.int16)).mean())
                  for j in range(max(0, t - 2), min(len(new), t + 3)) if j != t]
        if own > max_mae or (nearby and own - min(nearby) > max(0.5, 0.2 * own)):
            shifted.append(t)
    if shifted:
        raise AlignmentError(f'RGB alignment uncertain at frames {shifted[:10]}; max MAE={max(errors):.3f}')
    return {'mean_mae': float(np.mean(errors)), 'max_mae': max(errors), 'frames_checked': len(new)}


def reusable_done(p, work, ep, signature, n):
    q = json.loads(work.q(ep).read_text())
    if not p.done_ok(work, ep, q, signature):
        return False
    d = json.loads(work.done_json(ep).read_text())
    hashes = d.get('reuse_file_sha256', {})
    return (d.get('n_frames') == n and bool(hashes)
            and all(Path(f).is_relative_to(work.root) and file_hash(f) == h for f, h in hashes.items()))


def reuse_episode(p, src, old, old_sam, work, ep, rows, signature, max_mae):
    start = time.monotonic()
    n = int(src.eps.loc[ep, 'length'])
    if reusable_done(p, work, ep, signature, n):
        return {'ep': ep, 'cached': True}
    frames = np.stack(p.source_frames(src, ep))
    try:
        alignment = check_alignment(np.stack(p.source_frames(old, ep)), frames, max_mae)
    except (AlignmentError, ValueError, av.FFmpegError) as exc:
        return {'ep': ep, 'recompute_reason': str(exc)}
    q = json.loads(work.q(ep).read_text())
    seg_dir = work.segs / f'{ep:06d}'
    seg_dir.mkdir(parents=True, exist_ok=True)
    style = p.MaskStyle.from_dict(signature['style'])
    groups, slot_group, by_phrase, hashes = [], [], {}, {}
    for slot, phrases in enumerate(q['phrases']):
        key = tuple(sorted(phrases))
        row = rows.iloc[slot]
        source = inside(old_sam, row['file'])
        if key not in by_phrase:
            g = len(groups)
            try:
                keep = p.decode_mask_video(source)
                if keep.shape != frames.shape[:3]:
                    raise AlignmentError(f'mask {source} shape {keep.shape} != {frames.shape[:3]}')
                if abs(float(keep.mean()) - float(row.keep_frac)) > 1e-5:
                    raise AlignmentError(f'mask {source}: keep fraction differs from index')
                if not phrases and not keep.all():
                    raise AlignmentError(f'mask {source}: NONE instruction must keep all pixels')
            except (ValueError, OSError, av.FFmpegError) as exc:
                return {'ep': ep, 'recompute_reason': str(exc)}
            rendered = style.render(frames, keep) if phrases else frames
            video, mask = seg_dir / f'g{g:03d}.mp4', seg_dir / f'g{g:03d}_mask.mkv'
            p.encode_video(rendered, video, src.fps)
            tmp = mask.with_suffix('.tmp.mkv')
            shutil.copyfile(source, tmp)  # never hardlink original masks into writable work
            os.replace(tmp, mask)
            hashes[str(video)], hashes[str(mask)] = file_hash(video), file_hash(mask)
            groups.append({'phrases': list(key), 'file': str(video), 'mask_file': str(mask),
                           'keep_frac': float(row.keep_frac), 'keep_frac_unjudged': float(row.keep_frac_unjudged),
                           'stats': p.image_stats(rendered), 'reused_mask_from': str(source)})
            by_phrase[key] = g
        g = by_phrase[key]
        if groups[g]['reused_mask_from'] != str(source):
            raise ValueError(f'ep {ep}: same phrases map to different old masks')
        slot_group.append(g)
    elapsed = time.monotonic() - start
    write_json(work.done_json(ep), {'episode_index': ep, 'n_frames': n, 'groups': groups,
               'signature': signature, 'slot_group': slot_group, 'seconds': elapsed,
               'alignment': alignment, 'reuse_file_sha256': hashes})
    return {'ep': ep, 'groups': len(groups), 'n_frames': n, 'seconds': elapsed, 'alignment': alignment}


def pipeline_argv(args, info, stage, episodes=None):
    j = info['judge']
    gpu = scheduling(args)
    endpoints = []
    for url in (args.qwen_url or 'http://127.0.0.1:8000').split(','):
        url = url.strip().rstrip('/')
        if not url.endswith('/chat/completions'):
            url += '/chat/completions' if url.endswith('/v1') else '/v1/chat/completions'
        endpoints.append(url)
    argv = [str(args.pipeline), '--src', str(args.raw), '--out', str(args.out), '--work', str(args.work),
            '--mode', 'multi', '--stage', stage, '--qwen-url', ','.join(endpoints),
            '--sam3-gpus', args.sam3_gpus or '', '--compose-workers',
            str(gpu['gpu_compose_workers'] if stage == 'masks' else args.compose_workers),
            '--report-every', str(getattr(args, 'report_every', 30)),
            '--prep-threads', str(gpu['prep_workers']), '--max-queued', str(gpu['max_queued']),
            '--qwen-concurrency', str(gpu['judge_concurrency']),
            '--judge-concurrency', str(gpu['judge_concurrency']),
            '--judge-episodes', str(gpu['judge_episodes']), '--judge',
            '--judge-model', j['model'], '--judge-threshold', str(j['threshold']),
            '--judge-outside-scale', str(j['outside_scale']), '--judge-regrow', str(j['regrow']),
            '--judge-regrow-min-area', str(j['regrow_min_area']),
            '--style', 'outline', '--outside-scale', '0.1', '--edge-width', '2']
    if episodes is not None:
        if stage != 'masks' or not episodes:
            raise ValueError('Only masks may select a nonempty subset; assembly must use ALL episode IDs')
        argv += ['--episodes', ','.join(map(str, episodes))]
    return argv


def call_pipeline(p, argv, encoder_threads=2):
    previous = sys.argv
    try:
        sys.argv = argv
        with pipeline_runtime(p, argv, encoder_threads):
            status = p.main()
        if status:
            raise RuntimeError(f'Pipeline failed with status {status}; resume the same command')
    finally:
        sys.argv = previous


def check_complete(p, src, work, signature, progress=None):
    """Prevent subset publication/reindexing or incomplete frame counts."""
    for ep in range(len(src.eps)):
        q = p.load_qwen(work.q(ep), src, ep, 'multi')
        if q is None or not p.done_ok(work, ep, q, signature):
            raise ValueError(f'ep {ep} is unfinished; run --stage all')
        done = json.loads(work.done_json(ep).read_text())
        expected = int(src.eps.loc[ep, 'length'])
        if done['n_frames'] != expected:
            raise ValueError(f'ep {ep}: stale frame count in done marker')
        for group in done['groups']:
            for key in ('file', 'mask_file'):
                if not Path(group[key]).resolve().is_relative_to(work.root):
                    raise ValueError(f'ep {ep}: segment escapes NEW work directory')
            shape = p.decode_mask_video(Path(group['mask_file'])).shape
            if shape != (expected, *src.hw):
                raise ValueError(f'ep {ep}: invalid final mask shape {shape}')
        if progress is not None:
            progress.advance(detail=f'episode={ep} groups={len(done["groups"])}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'old-sam', 'out', 'work', 'pipeline'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=['plan', 'reuse', 'masks', 'assemble', 'all'], default='all')
    parser.add_argument('--qwen-url', default='', help='Existing endpoint; required only for all/masks')
    parser.add_argument('--sam3-gpus', default='', help='Explicit free GPU IDs; required for all/masks')
    parser.add_argument('--compose-workers', default='auto', help='CPU processes, or auto from affinity/cgroup quota')
    parser.add_argument('--encoder-threads', type=int, default=2, help='SVT-AV1 threads per CPU reuse process')
    gpu_arguments(parser)
    parser.add_argument('--report-every', type=float, default=30,
                        help='Progress heartbeat interval in seconds (also passed to SAM/judge)')
    parser.add_argument('--alignment-max-mae', type=float, default=6.0)
    parser.add_argument('--smoke-episodes', default='', help='Only with --stage reuse; never publishes a dataset')
    parser.add_argument('--print-command', action='store_true', help='Read-only plan and commands')
    args = parser.parse_args()
    native_thread_limits()
    args.compose_workers = resolve_workers(args.compose_workers, args.encoder_threads)
    scheduling(args)  # Validate overrides before any work is written.
    if args.compose_workers < 1 or args.alignment_max_mae <= 0 or not 0 < args.report_every < float('inf'):
        parser.error('workers, alignment tolerance and report interval must be positive and finite')
    if args.smoke_episodes and args.stage != 'reuse':
        parser.error('--smoke-episodes requires --stage reuse')
    for name in ('raw', 'old_sam', 'out', 'work', 'pipeline'):
        setattr(args, name, getattr(args, name).resolve())
    print('[prepare] Validating source metadata, pipeline versions and workspace ownership...', flush=True)
    report = json.loads((args.raw / 'repair_report.json').read_text())
    if not report.get('complete'):
        raise ValueError('--raw must be a completed framefixed conversion')
    raw_manifest = json.loads((args.raw / 'repair_manifest.json').read_text())
    old_root = Path(raw_manifest['reference']).resolve()
    partial = args.out.with_name(args.out.name + '.partial')
    project = args.pipeline.parent.parent
    protected = [args.raw, old_root, args.old_sam, args.pipeline.parent,
                 project / 'sam3', project / 'vllm_space', project / 'RoboTwinData/dataset']
    destinations = [args.out, partial, args.work]
    for i, dest in enumerate(destinations):
        for other in protected + destinations[:i]:
            other = other.resolve()
            if dest.is_relative_to(other) or other.is_relative_to(dest):
                raise ValueError(f'Output/work paths must be separate from protected paths: {dest}, {other}')
    if args.out.exists():
        raise ValueError('Output already exists; never overwrite an existing dataset')
    p = import_pipeline(args.pipeline)
    info = json.loads((args.old_sam / 'meta/sam_info.json').read_text())
    signature = validate_pipeline(p, info)
    src, old = p.Source(args.raw), p.Source(old_root)
    plan, groups = make_plan(p, src, old, args.old_sam, info)
    inputs = [Path(importlib.import_module(obj.__module__).__file__) for obj in
              (p.Source, p.MaskStyle, p.QwenMaskJudge, p.extract_operation_objects)]
    inputs += [p.WORKER, project / 'vllm_space/sam3_prompt.py', project / 'sam3/tools/sam3_video_masker.py',
               p.MAPPING, Path(__file__).resolve(), Path(__file__).with_name('sam_rebuild_progress.py'),
               Path(__file__).with_name('sam_rebuild_cpu.py'),
               Path(__file__).with_name('sam_rebuild_gpu.py'),
               args.raw / 'repair_manifest.json', args.raw / 'repair_report.json']
    inputs += list((args.raw / 'meta').rglob('*.parquet')) + list((old_root / 'meta').rglob('*.parquet'))
    inputs += [args.old_sam / f for f in ('meta/sam_info.json', 'meta/sam_prompts.parquet',
                                        'sam_masks/index.parquet', 'sam_masks/judge_log.parquet')]
    stamp = {'version': 4, 'raw': str(args.raw), 'old_sam': str(args.old_sam), 'out': str(args.out),
             'signature': signature, 'alignment_max_mae': args.alignment_max_mae,
             'inputs_sha256': {str(f): file_hash(f) for f in sorted(set(inputs))}}
    marker = args.work / 'framefixed_sam_manifest.json'
    if args.work.exists() and (not marker.exists() or not compatible_single_manifest(json.loads(marker.read_text()), stamp)):
        raise ValueError('Refusing old/unrecognized/changed SAM workspace: choose a new --work')
    if partial.exists() and not marker.exists():
        raise ValueError('Refusing unowned partial output')
    plan_file = args.work / 'incremental_plan.json'
    if plan_file.exists():
        previous = json.loads(plan_file.read_text())
        plan['alignment_failures'] = previous['alignment_failures']
        promoted = {int(e) for e in plan['alignment_failures']}
        plan['recompute_episodes'] = sorted(set(plan['recompute_episodes']) | promoted)
        plan['reuse_episodes'] = [e for e in plan['reuse_episodes'] if e not in promoted]
        update_counts(plan, src, groups)
    summary = {k: v for k, v in plan.items() if not k.endswith('_episodes')}
    print(json.dumps(summary, indent=2), flush=True)
    if args.stage == 'plan' or args.print_command:
        print(json.dumps({'masks_arguments': pipeline_argv(args, info, 'masks', plan['recompute_episodes'])
                          if plan['recompute_episodes'] else [],
                          'assembly_arguments': pipeline_argv(args, info, 'assemble')}, indent=2))
        return
    if args.stage in ('all', 'masks') and plan['recompute_episodes'] and (not args.qwen_url or not args.sam3_gpus):
        parser.error('Provide --qwen-url and explicit --sam3-gpus for all/masks')
    args.work.mkdir(parents=True, exist_ok=True)
    # One orchestrator owns this isolated work tree (including partial output cleanup).
    with open(args.work / '.rebuild.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_owned(args, p, src, old, info, signature, plan, groups, marker, stamp, plan_file)


def run_owned(args, p, src, old, info, signature, plan, groups, marker, stamp, plan_file):
    with Progress(args.work, args.report_every) as progress:
        if marker.exists() and json.loads(marker.read_text()) != stamp:
            previous = json.loads(marker.read_text())
            write_json(marker.with_name(f'framefixed_sam_manifest.v{previous["version"]}.json'), previous)
            print(f'[resume] Upgrading audited v{previous["version"]} scheduling to v4; '
                  'completed CPU/SAM/judge episodes retained.', flush=True)
        write_json(marker, stamp)
        write_json(plan_file, plan)
        runtime = args.work / 'runtime'
        (runtime / 'tmp').mkdir(parents=True, exist_ok=True)
        os.environ['TMPDIR'] = str(runtime / 'tmp')
        os.environ['TRITON_CACHE_DIR'] = str(runtime / 'triton')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(runtime / 'torchinductor')
        tempfile.tempdir = str(runtime / 'tmp')
        p.AUTO_MASK = runtime  # SAM worker temporary files must not go into auto_mask/tmp
        p.concat_segments = copy_episode_segment
        work = p.Work(args.work, 'multi')
        selected = p.parse_episodes(args.smoke_episodes, len(src.eps)) if args.smoke_episodes else list(range(len(src.eps)))
        progress.stage('prepare_prompts', len(selected))
        seed_prompts(p, src, work, args.old_sam, info, groups, selected, progress)
        good = [ep for ep in selected if ep in set(plan['reuse_episodes'])]
        if args.stage in ('all', 'reuse'):
            progress.stage('reuse_judge_logs', len(good))
            seed_judges(p, work, args.old_sam, signature, good, progress)
            progress.stage('reuse_render', len(good))
            results, start = [], time.monotonic()
            with closing(reuse_results(args, p, src, old, work, good, groups, signature, runtime)) as completed:
                for result in completed:
                    results.append(result)
                    ep = result['ep']
                    if 'recompute_reason' in result:
                        plan['alignment_failures'][str(ep)] = result['recompute_reason']
                        plan['reuse_episodes'].remove(ep)
                        plan['recompute_episodes'] = sorted(set(plan['recompute_episodes']) | {ep})
                        work.judge_json(ep).unlink(missing_ok=True)
                        work.done_json(ep).unlink(missing_ok=True)
                        update_counts(plan, src, groups)
                        write_json(plan_file, plan)
                    progress.advance(cached=int(result.get('cached', False)),
                                     promoted=int('recompute_reason' in result),
                                     detail=f'episode={ep}; worker_pid={result["worker_pid"]}; fresh SAM total={plan["recompute_count"]}')
            write_json(args.work / ('smoke_report.json' if args.smoke_episodes else 'reuse_report.json'),
                       {'seconds': time.monotonic() - start, 'selected_episodes': selected,
                        'recompute_selected': [e for e in selected if e in plan['recompute_episodes']],
                        'results': results, 'published': False})
        if args.stage in ('all', 'masks') and plan['recompute_episodes']:
            if not args.qwen_url or not args.sam3_gpus:
                raise ValueError('Alignment checks found episodes requiring --qwen-url and --sam3-gpus')
            for ep in plan['recompute_episodes']:
                # Even an episode with no noun phrases needs an empty judge document
                # for the upstream composer. Never seed old verdicts for these IDs.
                if not work.judge_json(ep).exists():
                    write_json(work.judge_json(ep), {'episode_index': ep, 'phrases': {}})
            # This is the ONLY model-inference call, and it receives only damaged IDs.
            prompts = {ep: json.loads(work.q(ep).read_text()) for ep in plan['recompute_episodes']}
            finished = set()

            def count_finished():
                for ep in prompts.keys() - finished:
                    if p.done_ok(work, ep, prompts[ep], signature):
                        finished.add(ep)
                return len(finished)

            cached = count_finished()
            progress.stage('sam_judge_render', len(prompts), completed=cached, cached=cached,
                           refresh=count_finished)
            if cached < len(prompts):
                call_pipeline(p, pipeline_argv(args, info, 'masks', plan['recompute_episodes']), args.encoder_threads)
        if args.stage in ('all', 'assemble'):
            progress.stage('validate_masks', len(src.eps))
            check_complete(p, src, work, signature, progress)
            # Never pass --episodes to assembly: preserve all original IDs and numeric tables.
            other_cameras = sum(v['dtype'] == 'video' and k != p.CAM for k, v in src.info['features'].items())
            total_videos = len(src.eps) * (len(p.cam_keys('multi')) + other_cameras)
            with observe_assembly(p, progress, total_videos):
                call_pipeline(p, pipeline_argv(args, info, 'assemble'))
            progress.stage('publish_report', 1)
            write_json(args.out / 'incremental_sam_report.json',
                       dict(plan, old_sam=str(args.old_sam), raw=str(args.raw), complete=True,
                            signature=signature, provenance=stamp))
            progress.advance(detail=f'dataset={args.out}')


if __name__ == '__main__':
    main()
