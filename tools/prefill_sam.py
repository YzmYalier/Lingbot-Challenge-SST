#!/usr/bin/env python3
"""Compute only missing object masks while Qwen is being provisioned.

Uses the same worker, phrases, thresholds and work/masks_obj cache as the
incremental pipeline. No judged mask or training dataset is published here.
Create work/sam_reuse/prefill.stop to drain the bounded queue and release GPUs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, interpreter, path, runtime_env, writable
from process_data import write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sam-gpus', default=config()['processing']['sam_gpus'])
    args = ap.parse_args()
    gpus = args.sam_gpus.split(',')
    if not gpus or len(gpus) != len(set(gpus)) or any(not g.isdigit() for g in gpus):
        ap.error('Provide distinct physical GPU IDs')
    if os.environ.get('SUBMISSION_PREFILL_SAM') != str(base_path()):
        exe = str(interpreter('data'))
        env = runtime_env('incremental-repair')
        env['SUBMISSION_PREFILL_SAM'] = str(base_path())
        os.execve(exe, [exe, '-B', str(Path(__file__).resolve()), *sys.argv[1:]], env)
    import rebuild_robotwin_sam as R
    from sam_rebuild_gpu import _init_frames, _prepare_frames
    root = writable(path('sam_reuse_work'))
    # The completed reuse report fixes the promotion list before any GPU work.
    reuse = json.loads((root / 'reuse_report.json').read_text())
    if reuse['selected_episodes'] != list(range(2500)):
        raise ValueError('A complete CPU reuse pass is required')
    driver = writable(path('sam_reuse_report').parent / 'driver.lock')
    with driver.open('a') as outer, (root / '.rebuild.lock').open('a') as inner:
        for lock in (outer, inner):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck the existing source/code/signature manifest before reusing masks.
        command = [str(interpreter('data')), '-B', str(base_path() / 'tools/rebuild_robotwin_sam.py'),
                   '--raw', str(path('raw')), '--old-sam', str(path('sam_reuse_source')),
                   '--out', str(path('sam_output')), '--work', str(root),
                   '--pipeline', str(base_path() / 'auto_mask/pipeline_lerobot.py'),
                   '--stage', 'plan', '--allow-source-relocation', '--rerender-style']
        with (root / 'prefill_source_check.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        plan = json.loads((root / 'incremental_plan.json').read_text())
        p = R.import_pipeline(base_path() / 'auto_mask/pipeline_lerobot.py')
        src, work = p.Source(path('raw')), p.Work(root, 'multi')
        p.AUTO_MASK = root / 'runtime'
        pending, cached = [], []
        for ep in plan['recompute_episodes']:
            q = p.load_qwen(work.q(ep), src, ep, 'multi')
            if q is None:
                raise ValueError(f'Missing verified prompt for episode {ep}')
            have = p.mask_index(work, ep, src.n_video_frames(ep))
            missing = [phrase for phrase in p.distinct_phrases(q) if phrase not in have]
            if missing:
                pending.append((ep, missing))
            else:
                cached.append(ep)
        completed, active, futures = list(cached), {}, {}
        stop = root / 'prefill.stop'
        if stop.exists():
            raise ValueError('prefill.stop exists; remove it explicitly to resume prefill')
        stop_requested = False
        def request_stop(*_):
            nonlocal stop_requested
            stop_requested = True
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        workers = []
        started, last_report = time.monotonic(), 0
        try:
            workers = p.launch_sam3_workers(work, gpus)
            with ProcessPoolExecutor(min(4, 2 * len(gpus)), mp_context=mp.get_context('spawn'),
                    initializer=_init_frames, initargs=(str(p.__file__), str(src.root), str(root),
                                                       'local', str(root / 'runtime/prefill-frame-workers'))) as pool:
                while pending or active:
                    draining = stop_requested or stop.exists()
                    while pending and len(active) < 2 * len(gpus) and not draining:
                        ep, phrases = pending.pop(0)
                        for suffix in ('json', 'txt'):
                            (work.failed / f'{ep:06d}.{suffix}').unlink(missing_ok=True)
                        name = f'{ep:06d}_{len(list(work.masks.glob(f"{ep:06d}_*.json"))):02d}'
                        active[ep] = phrases
                        futures[pool.submit(_prepare_frames, ep, phrases, name)] = ep
                    for future in [f for f in futures if f.done()]:
                        future.result()
                        futures.pop(future)
                    for ep, phrases in list(active.items()):
                        failure = work.failed / f'{ep:06d}.txt'
                        if failure.exists():
                            raise RuntimeError(f'SAM episode {ep} failed: {failure}')
                        masks = p.mask_index(work, ep, src.n_video_frames(ep))
                        if all(phrase in masks for phrase in phrases):
                            completed.append(ep)
                            del active[ep]
                            shutil.rmtree(work.frames / f'{ep:06d}', ignore_errors=True)
                    if any(worker.poll() is not None for worker in workers):
                        raise RuntimeError('A SAM worker exited unexpectedly; inspect work/logs')
                    if time.monotonic() - last_report >= 30:
                        report = dict(stage='sam_objects_only', complete=False, completed=len(completed),
                                      total=plan['recompute_count'], active=sorted(active), remaining=len(pending),
                                      draining=draining, cached=len(cached), elapsed_seconds=time.monotonic()-started,
                                      published=False)
                        write_json(root / 'prefill_progress.json', report)
                        print(json.dumps(report), flush=True)
                        last_report = time.monotonic()
                    if draining and not active:
                        break
                    time.sleep(1)
            final = dict(complete=not pending, published=False,
                       stage='sam_objects_only', completed_episodes=sorted(completed),
                       remaining_episodes=[ep for ep, _ in pending], elapsed_seconds=time.monotonic()-started)
            write_json(root / 'prefill_report.json', final)
            write_json(root / 'prefill_progress.json', dict(stage='sam_objects_only', complete=not pending,
                       status='complete' if not pending else 'paused', completed=len(completed),
                       total=plan['recompute_count'], active=[], remaining=len(pending), published=False))
        except BaseException as exc:
            write_json(root / 'prefill_progress.json', dict(stage='sam_objects_only', complete=False,
                       status='failed', error=str(exc), completed=len(completed), total=plan['recompute_count'],
                       active=sorted(active), remaining=len(pending), published=False))
            raise
        finally:
            (work.queue / 'QUEUE_CLOSED').touch()
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
            for worker in workers:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()


if __name__ == '__main__':
    main()
