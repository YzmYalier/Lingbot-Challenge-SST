#!/usr/bin/env python3
"""Shared-filesystem multi-node incremental SAM rebuild; original inputs stay read-only.

Every episode belongs to ep % NNODES. Each node renders its normal episodes on CPU,
starts local GPU0 Qwen during that work, then runs SAM+judge on its damaged episodes.
Only rank 0 validates/assembles the full dataset after all shards finish.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

import rebuild_robotwin_sam as R
from sam_rebuild_progress import Progress, observe_assembly
from sam_rebuild_cpu import native_thread_limits, resolve_workers, reuse_results
from sam_rebuild_gpu import AUDITED_REBUILD, add_arguments as gpu_arguments, scheduling


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def locked(path, *, shared=False, blocking=False):
    import fcntl
    # A read-only shared lock also works on an old single-node workspace.
    with Path(path).open('r' if shared else 'a') as handle:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(handle, mode | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f'Workspace is in use: {path}') from None
        yield


def shard_episodes(episodes, rank, nodes):
    return [ep for ep in episodes if ep % nodes == rank]


def sam_gpus(explicit=None, share_qwen_gpu=False):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible:
        allocated = [v.strip() for v in visible.split(',')]
    else:
        count = os.environ.get('HOST_GPU_NUM')
        if count is None:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True)
            allocated = result.split()
        else:
            allocated = list(map(str, range(int(count))))
    if '0' not in allocated or any(not g.isdigit() for g in allocated):
        raise ValueError('GPU0 must be allocated; CUDA_VISIBLE_DEVICES must use physical numeric GPU IDs')
    selected = [g.strip() for g in explicit.split(',')] if explicit is not None else [
        g for g in allocated if share_qwen_gpu or g != '0']
    if not selected or len(set(selected)) != len(selected) or any(g not in allocated for g in selected):
        raise ValueError('SAM needs distinct allocated GPU IDs; use --sam3-gpus 1,2,...')
    if '0' in selected and not share_qwen_gpu:
        raise ValueError('Add --share-qwen-gpu to run SAM alongside Qwen on GPU0')
    return ','.join(selected)


def prepare(args):
    if not read_json(args.raw / 'repair_report.json').get('complete'):
        raise ValueError('Raw conversion is incomplete')
    reference = Path(read_json(args.raw / 'repair_manifest.json')['reference']).resolve()
    project = args.pipeline.parent.parent
    protected = [args.raw, reference, args.old_sam, args.pipeline.parent, project / 'sam3',
                 project / 'vllm_space', project / 'miniconda3', project / 'RoboTwinData/dataset']
    protected += list(project.glob('lingbot-vla-v2*'))
    if args.reuse_work:
        protected.append(args.reuse_work)
    destinations = [args.work, args.out, args.out.with_name(args.out.name + '.partial')]
    for i, dest in enumerate(destinations):
        for other in protected + destinations[:i]:
            if dest.is_relative_to(other.resolve()) or other.resolve().is_relative_to(dest):
                raise ValueError(f'Output/work overlap with protected path: {dest}, {other}')
    p = R.import_pipeline(args.pipeline)
    info = read_json(args.old_sam / 'meta/sam_info.json')
    signature = R.validate_pipeline(p, info)
    src, old = p.Source(args.raw), p.Source(reference)
    plan, groups = R.make_plan(p, src, old, args.old_sam, info)
    inputs = [Path(importlib.import_module(obj.__module__).__file__) for obj in
              (p.Source, p.MaskStyle, p.QwenMaskJudge, p.extract_operation_objects)]
    inputs += [p.WORKER, project / 'vllm_space/sam3_prompt.py', project / 'sam3/tools/sam3_video_masker.py',
               p.MAPPING, args.raw / 'repair_manifest.json', args.raw / 'repair_report.json']
    inputs += list((args.raw / 'meta').rglob('*')) + list((reference / 'meta').rglob('*'))
    inputs += [args.old_sam / f for f in ('meta/sam_info.json', 'meta/sam_prompts.parquet',
                                        'sam_masks/index.parquet', 'sam_masks/judge_log.parquet')]
    stamp = dict(version=1, nodes=args.nodes, raw=str(args.raw), old_sam=str(args.old_sam),
                 out=str(args.out), signature=signature, alignment_max_mae=args.alignment_max_mae,
                 inputs_sha256={str(f): R.file_hash(f) for f in sorted(set(inputs)) if f.is_file()},
                 code_sha256={name: R.file_hash(Path(__file__).with_name(name)) for name in
                              ('rebuild_robotwin_sam_multinode.py', 'rebuild_robotwin_sam.py',
                               'sam_rebuild_progress.py', 'sam_rebuild_cpu.py', 'sam_rebuild_gpu.py',
                               'start_qwen_gpu0.sh')})
    fingerprint = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    distribution = []
    for rank in range(args.nodes):
        good = shard_episodes(plan['reuse_episodes'], rank, args.nodes)
        bad = shard_episodes(plan['recompute_episodes'], rank, args.nodes)
        distribution.append(dict(rank=rank, reuse=len(good), recompute=len(bad),
                                 reuse_frames=int(src.eps.loc[good, 'length'].sum()),
                                 recompute_frames=int(src.eps.loc[bad, 'length'].sum())))
    print(json.dumps(dict(rank=args.rank, nodes=args.nodes, hostname=socket.gethostname(),
                          distribution=distribution, work=str(args.work), output=str(args.out),
                          coordination='shared filesystem; no MASTER_PORT listener'), indent=2), flush=True)
    return p, info, signature, src, old, plan, groups, stamp, fingerprint


def initialize(args, stamp, plan):
    marker = args.work / 'multinode_manifest.json'
    if args.out.with_name(args.out.name + '.partial').exists() and not marker.exists():
        raise ValueError('Refusing unowned partial output')
    args.work.mkdir(parents=True, exist_ok=True)
    with locked(args.work / '.initialize.lock', blocking=True):
        if marker.exists():
            if read_json(marker) != stamp:
                raise ValueError('Source/code/node count changed; choose a new --work')
        else:
            if any(f.name != '.initialize.lock' for f in args.work.iterdir()):
                raise ValueError('Unrecognized work directory; use a NEW shared --work')
            R.write_json(marker, stamp)
            R.write_json(args.work / 'initial_plan.json', plan)


def import_reuse(args, p, work, signature, src, episodes):
    """Import only hash-verified CPU results, never old damaged masks or active files."""
    if args.reuse_work is None:
        return 0
    source = args.reuse_work
    marker = read_json(source / 'framefixed_sam_manifest.json')
    for key, value in dict(raw=str(args.raw), old_sam=str(args.old_sam), signature=signature,
                           alignment_max_mae=args.alignment_max_mae).items():
        if marker.get(key) != value:
            raise ValueError(f'Old single-node cache differs: {key}')
    with locked(source / '.rebuild.lock', shared=True):
        def compatible_input(f, h):
            if R.file_hash(Path(f)) == h:
                return True
            return (Path(f).name == 'rebuild_robotwin_sam.py'
                    and h == AUDITED_REBUILD.get(marker.get('version'))
                    and R.file_hash(Path(f)) == R.file_hash(Path(R.__file__)))
        if any(not compatible_input(f, h) for f, h in marker['inputs_sha256'].items()):
            raise ValueError('Old single-node cache provenance has changed')
        imported = 0
        for ep in episodes:
            done = source / 'multi/judged/done' / f'{ep:06d}.json'
            if work.done_json(ep).exists() or not done.exists():
                continue
            d = read_json(done)
            hashes = d.get('reuse_file_sha256', {})
            if not hashes or d.get('signature') != signature or d.get('n_frames') != int(src.eps.loc[ep, 'length']):
                continue
            remap = {}
            for f, h in hashes.items():
                oldfile = Path(f).resolve()
                if not oldfile.is_relative_to(source) or R.file_hash(oldfile) != h:
                    raise ValueError(f'Corrupt/unsafe cached segment: {oldfile}')
                newfile = work.root / oldfile.relative_to(source)
                newfile.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(oldfile, newfile)
                remap[f] = str(newfile)
            for group in d['groups']:
                for key in ('file', 'mask_file'):
                    group[key] = remap[group[key]]
            d['reuse_file_sha256'] = {remap[f]: h for f, h in hashes.items()}
            oldq = source / 'multi/qwen' / f'{ep:06d}.json'
            if read_json(oldq) != read_json(work.q(ep)):
                raise ValueError(f'Cached prompt differs: episode {ep}')
            shutil.copyfile(source / 'multi/judged/judge' / f'{ep:06d}.json', work.judge_json(ep))
            R.write_json(work.done_json(ep), d)
            if not R.reusable_done(p, work, ep, signature, int(src.eps.loc[ep, 'length'])):
                raise ValueError(f'Imported episode {ep} is incomplete')
            imported += 1
    print(f'[rank {args.rank}] Imported {imported} completed CPU episodes from {source}', flush=True)
    return imported


def local_complete(p, src, work, signature, episodes):
    for ep in episodes:
        q = p.load_qwen(work.q(ep), src, ep, 'multi')
        if q is None or not p.done_ok(work, ep, q, signature):
            return False
        d = read_json(work.done_json(ep))
        if d['n_frames'] != int(src.eps.loc[ep, 'length']):
            return False
        if any(not Path(g[k]).resolve().is_relative_to(work.root) for g in d['groups'] for k in ('file', 'mask_file')):
            raise ValueError('Segment escaped work directory')
        if not work.judge_json(ep).exists():
            return False
    return True


def clear_promoted_reuse(work, episodes):
    # A crash can occur after recording promotion but before removing old verdicts.
    # Fresh SAM judge documents lack this reuse-only marker and must be preserved.
    for ep in episodes:
        judge = work.judge_json(ep)
        if judge.exists() and 'reused_final_masks_from' in read_json(judge):
            work.done_json(ep).unlink(missing_ok=True)
            judge.unlink()


@contextmanager
def publication(args, fingerprint):
    """Never clean another run's .partial, even when it used a different work root."""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    owner = args.out.with_name(args.out.name + '.multinode-owner.json')
    partial = args.out.with_name(args.out.name + '.partial')
    expected = dict(work=str(args.work), fingerprint=fingerprint)
    with locked(args.out.with_name(args.out.name + '.multinode.lock')):
        if args.out.exists():
            raise ValueError('Output already exists; refusing replacement')
        if owner.exists():
            if read_json(owner) != expected:
                raise ValueError('Output reserved by another rebuild; choose a different --out')
        elif partial.exists():
            raise ValueError('Refusing to clean an unowned partial output')
        else:
            R.write_json(owner, expected)
        yield


def wait_shards(args, fingerprint, progress):
    started = time.time()
    def finished():
        count, states = 0, []
        for rank in range(args.nodes):
            root = args.work / 'nodes' / f'rank{rank}'
            done = root / 'complete.json'
            if done.exists() and read_json(done).get('fingerprint') == fingerprint:
                count += 1
                states.append(f'{rank}:done')
                continue
            fail = root / 'failure.json'
            if fail.exists():
                data = read_json(fail)
                if data.get('fingerprint') == fingerprint and data['time'] >= args.started:
                    raise RuntimeError(f'Rank {rank} failed: {data["error"]}')
            status = root / 'progress.json'
            if status.exists():
                s = read_json(status)
                states.append(f'{rank}:{s["stage"]} {s["completed"]}/{s["total"]}')
            else:
                states.append(f'{rank}:not started')
        progress.detail = '; '.join(states)
        return count
    progress.stage('wait_nodes', args.nodes, refresh=finished)
    while finished() != args.nodes:
        if time.time() - started >= args.wait_timeout:
            raise TimeoutError(f'Timed out waiting for shards: {progress.detail}; resume all nodes with the same command')
        time.sleep(min(2, args.report_every))


def run_rank(args, p, info, signature, src, old, plan, groups, stamp, fingerprint, progress):
    runtime = args.node_dir / 'runtime'
    (runtime / 'tmp').mkdir(parents=True, exist_ok=True)
    for name, subdir in [('TMPDIR', 'tmp'), ('TRITON_CACHE_DIR', 'triton'),
                         ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'), ('XDG_CACHE_HOME', 'cache'),
                         ('TORCH_HOME', 'torch'), ('CUDA_CACHE_PATH', 'cuda'),
                         ('PYTORCH_KERNEL_CACHE_PATH', 'cache/torch/kernels')]:
        (runtime / subdir).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(runtime / subdir)
    tempfile.tempdir = str(runtime / 'tmp')
    p.AUTO_MASK = runtime
    p.concat_segments = R.copy_episode_segment
    work = p.Work(args.work, 'multi', f'rank{args.rank}of{args.nodes}')
    selected = shard_episodes(range(len(src.eps)), args.rank, args.nodes)
    done_marker = args.node_dir / 'complete.json'
    already = done_marker.exists() and read_json(done_marker).get('fingerprint') == fingerprint
    if already and not local_complete(p, src, work, signature, selected):
        raise ValueError('Previously completed shard is corrupt; choose a new work directory')
    if not already and args.stage != 'assemble':
        progress.stage('prepare_prompts', len(selected))
        R.seed_prompts(p, src, work, args.old_sam, info, groups, selected, progress)
        plan_file = args.node_dir / 'alignment_failures.json'
        promoted = read_json(plan_file) if plan_file.exists() else {}
        clear_promoted_reuse(work, map(int, promoted))
        good = [e for e in shard_episodes(plan['reuse_episodes'], args.rank, args.nodes) if str(e) not in promoted]
        bad = sorted(set(shard_episodes(plan['recompute_episodes'], args.rank, args.nodes)) | set(map(int, promoted)))
        launcher = None
        log = None
        try:
            if args.stage in ('all', 'masks') and bad and not local_complete(p, src, work, signature, bad):
                args.sam3_gpus = sam_gpus(args.sam3_gpus, getattr(args, 'share_qwen_gpu', False))
                log = (args.node_dir / 'qwen_start.log').open('a')
                launcher = subprocess.Popen(['bash', str(Path(__file__).with_name('start_qwen_gpu0.sh'))],
                                             stdout=log, stderr=subprocess.STDOUT)
                print(f'[rank {args.rank}] Starting local GPU0 Qwen; log={log.name}', flush=True)
            if args.stage in ('all', 'reuse'):
                progress.stage('import_cpu_cache')
                import_reuse(args, p, work, signature, src, good)
                progress.stage('reuse_judge_logs', len(good))
                R.seed_judges(p, work, args.old_sam, signature, good, progress)
                progress.stage('reuse_render', len(good))
                with closing(reuse_results(args, p, src, old, work, good, groups, signature, runtime)) as completed:
                    for result in completed:
                        ep = result['ep']
                        if 'recompute_reason' in result:
                            promoted[str(ep)] = result['recompute_reason']
                            # Persist promotion BEFORE invalidating old reuse/judge state.
                            R.write_json(plan_file, promoted)
                            work.done_json(ep).unlink(missing_ok=True)
                            work.judge_json(ep).unlink(missing_ok=True)
                            bad = sorted(set(bad) | {ep})
                        progress.advance(cached=int(result.get('cached', False)),
                                         promoted=int('recompute_reason' in result),
                                         detail=f'episode={ep}; worker_pid={result["worker_pid"]}')
                        if launcher is not None and launcher.poll() not in (None, 0):
                            raise RuntimeError(f'Qwen startup failed; see {log.name}')
            if args.stage == 'reuse':
                return
            pending = [ep for ep in bad if not local_complete(p, src, work, signature, [ep])]
            if pending:
                args.sam3_gpus = sam_gpus(args.sam3_gpus, getattr(args, 'share_qwen_gpu', False))
                progress.stage('qwen_start', 1)
                if launcher is None:
                    log = (args.node_dir / 'qwen_start.log').open('a')
                    launcher = subprocess.Popen(['bash', str(Path(__file__).with_name('start_qwen_gpu0.sh'))],
                                                 stdout=log, stderr=subprocess.STDOUT)
                if launcher.wait() != 0:
                    raise RuntimeError(f'Qwen startup failed; see {log.name}')
                progress.advance()
                for ep in pending:
                    if not work.judge_json(ep).exists():
                        R.write_json(work.judge_json(ep), dict(episode_index=ep, phrases={}))
                count = lambda: sum(local_complete(p, src, work, signature, [ep]) for ep in bad)
                cached = count()
                progress.stage('sam_judge_render', len(bad), completed=cached, cached=cached, refresh=count)
                argv = R.pipeline_argv(args, info, 'masks', pending) + ['--shard', f'{args.rank}/{args.nodes}']
                R.call_pipeline(p, argv, args.encoder_threads)
            if not local_complete(p, src, work, signature, selected):
                raise ValueError('Local shard incomplete; run --stage all (CPU reuse and masks both required)')
            R.write_json(done_marker, dict(fingerprint=fingerprint, episodes=selected, hostname=socket.gethostname()))
        finally:
            if log is not None:
                log.close()
    if args.stage == 'reuse':
        return
    if args.rank != 0:
        print(f'[rank {args.rank}] Shard finished; rank 0 will assemble the complete dataset.', flush=True)
        return
    wait_shards(args, fingerprint, progress)
    progress.stage('validate_masks', len(src.eps))
    R.check_complete(p, src, work, signature, progress)
    with publication(args, fingerprint):
        for rank in range(args.nodes):
            pf = args.work / 'nodes' / f'rank{rank}' / 'alignment_failures.json'
            if pf.exists():
                plan['alignment_failures'].update(read_json(pf))
        promoted = set(map(int, plan['alignment_failures']))
        plan['reuse_episodes'] = [ep for ep in plan['reuse_episodes'] if ep not in promoted]
        plan['recompute_episodes'] = sorted(set(plan['recompute_episodes']) | promoted)
        R.update_counts(plan, src, groups)
        report = dict(plan, complete=True, nodes=args.nodes, fingerprint=fingerprint,
                      provenance=stamp, signature=signature)
        other_cameras = sum(v['dtype'] == 'video' and k != p.CAM for k, v in src.info['features'].items())
        args.sam3_gpus = args.sam3_gpus or ''
        with observe_assembly(p, progress, len(src.eps) * (len(p.cam_keys('multi')) + other_cameras)):
            saved_meta = p.write_meta
            def meta_with_report(a, s, w, episodes, done, new_id, where, other_where, out, subset):
                saved_meta(a, s, w, episodes, done, new_id, where, other_where, out, subset)
                # Include provenance BEFORE upstream atomically renames .partial to --out.
                R.write_json(out / 'incremental_sam_report.json', report)
            p.write_meta = meta_with_report
            R.call_pipeline(p, R.pipeline_argv(args, info, 'assemble'))
        progress.stage('publish_report', 1)
        progress.advance(detail=f'dataset={args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'old-sam', 'out', 'work', 'pipeline'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reuse-work', type=Path, help='Read-only import from a STOPPED old single-node rebuild')
    parser.add_argument('--stage', choices=['plan', 'reuse', 'masks', 'assemble', 'all'], default='all')
    parser.add_argument('--sam3-gpus', default=os.environ.get('DATAFIX_SAM_GPUS'), help='Default: all allocated GPUs except 0')
    parser.add_argument('--share-qwen-gpu', action='store_true',
                        help='Include GPU0 in automatic SAM selection, alongside local Qwen (requires 16 GiB free)')
    parser.add_argument('--compose-workers', default='auto', help='CPU processes, or auto from affinity/cgroup quota')
    parser.add_argument('--encoder-threads', type=int, default=2)
    gpu_arguments(parser)
    parser.add_argument('--report-every', type=float, default=30)
    parser.add_argument('--alignment-max-mae', type=float, default=6)
    parser.add_argument('--wait-timeout', type=float, default=86400, help='Rank 0 barrier timeout in seconds')
    args = parser.parse_args()
    native_thread_limits()
    args.compose_workers = resolve_workers(args.compose_workers, args.encoder_threads)
    scheduling(args)
    args.rank = int(os.environ.get('NODE_RANK', '0'))
    args.nodes = int(os.environ.get('NNODES', os.environ.get('HOST_NUM', '1')))
    args.started = time.time()
    if args.nodes < 1 or not 0 <= args.rank < args.nodes:
        parser.error('Require 0 <= NODE_RANK < NNODES')
    if args.compose_workers < 1 or any(not math.isfinite(v) or v <= 0 for v in
                                      (args.report_every, args.wait_timeout, args.alignment_max_mae)):
        parser.error('Workers, report interval, timeout and alignment tolerance must be positive/finite')
    for name in ('raw', 'old_sam', 'out', 'work', 'pipeline', 'reuse_work'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.qwen_url = f'http://127.0.0.1:{os.environ.get("QWEN_PORT", "8000")}/v1/chat/completions'
    print('[prepare] Validating metadata, judged pipeline and shared workspace...', flush=True)
    p, info, signature, src, old, plan, groups, stamp, fingerprint = prepare(args)
    if args.stage == 'plan':
        return
    if args.out.exists():
        report = args.out / 'incremental_sam_report.json'
        if report.exists() and read_json(report).get('complete') and read_json(report).get('fingerprint') == fingerprint:
            print(f'Already published: {args.out}', flush=True)
            return
        raise ValueError('Output already exists; refusing replacement')
    initialize(args, stamp, plan)
    args.node_dir = args.work / 'nodes' / f'rank{args.rank}'
    args.node_dir.mkdir(parents=True, exist_ok=True)
    with locked(args.node_dir / '.rank.lock'):
        (args.node_dir / 'failure.json').unlink(missing_ok=True)
        try:
            with Progress(args.node_dir, args.report_every) as progress:
                run_rank(args, p, info, signature, src, old, plan, groups, stamp, fingerprint, progress)
        except BaseException as exc:
            R.write_json(args.node_dir / 'failure.json', dict(time=time.time(), fingerprint=fingerprint,
                         error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__ == '__main__':
    main()
