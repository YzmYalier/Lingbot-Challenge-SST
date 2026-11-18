"""Explicit spawn workers for CPU reuse; no frames or model objects cross IPC."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import signal
import time
import traceback
from types import SimpleNamespace

# Only this audited scheduling-only predecessor can reuse its old v2 manifest.
LEGACY_REBUILD_SHA256 = '897d862af8e018aaecd3d70ab870a675f54070ca367715da2c44a06c788c52fa'


def cpu_capacity():
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    quota = None
    try:
        maximum, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if maximum != 'max':
            quota = int(maximum) / int(period)
    except (OSError, ValueError, ZeroDivisionError):
        try:
            root = Path('/sys/fs/cgroup/cpu')
            maximum = int((root / 'cpu.cfs_quota_us').read_text())
            period = int((root / 'cpu.cfs_period_us').read_text())
            if maximum > 0:
                quota = maximum / period
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return dict(affinity_cpus=affinity, quota_cpus=quota,
                available_cpus=min(affinity, quota) if quota is not None else affinity)


def resolve_workers(requested, encoder_threads=2):
    if encoder_threads < 1:
        raise ValueError('encoder threads must be positive')
    if str(requested).lower() == 'auto':
        # Leave 20% of allocated logical CPU capacity to Qwen, loaders and the parent.
        return max(1, math.floor(cpu_capacity()['available_cpus'] * 0.8 / encoder_threads))
    count = int(requested)
    if count < 1:
        raise ValueError('compose workers must be positive or auto')
    return count


def native_thread_limits():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def compatible_single_manifest(previous, expected):
    if previous == expected:
        return True
    if previous.get('version') != 2 or expected.get('version') != 3:
        return False
    legacy = {**expected, 'version': 2, 'inputs_sha256': dict(expected['inputs_sha256'])}
    helper = Path(__file__).resolve()
    legacy['inputs_sha256'].pop(str(helper), None)
    legacy['inputs_sha256'][str(helper.with_name('rebuild_robotwin_sam.py'))] = LEGACY_REBUILD_SHA256
    return previous == legacy


_STATE = None
_AV = None


def _limited_av_open(*args, **kwargs):
    container = _AV.open(*args, **kwargs)
    if isinstance(container, _AV.container.InputContainer):
        for stream in container.streams.video:
            stream.codec_context.thread_count = 1
    return container


def _initialize_worker(config):
    global _STATE, _AV
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # The parent terminates only its pool on Ctrl-C.
    native_thread_limits()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import rebuild_robotwin_sam as R
    p = R.import_pipeline(Path(config['pipeline']))
    p.MAPPING, p.RAW_ROOT = Path(config['mapping']), Path(config['instructions'])
    p.AUTO_MASK = Path(config['runtime'])
    p.cv2.setNumThreads(1)
    p.pa.set_cpu_count(1)
    p.pa.set_io_thread_count(1)
    # These are in-memory overrides in a private CPU process, not upstream edits.
    _AV = p.av
    p.av = SimpleNamespace(**{**vars(_AV), 'open': _limited_av_open})
    p.ENCODE_OPTIONS = {**p.ENCODE_OPTIONS, 'svtav1-params': f'lp={config["encoder_threads"]}'}
    log = Path(config['logs']) / f'worker-{os.getpid()}.log'
    p.quiet_stderr(str(log))
    source, old = p.Source(Path(config['raw'])), p.Source(Path(config['old_raw']))
    work = p.Work(Path(config['work']), 'multi', config['queue_tag'])
    _STATE = (R, p, source, old, Path(config['old_sam']), work, config)
    R.write_json(log.with_suffix('.json'), dict(pid=os.getpid(), ppid=os.getppid(),
                 backend='spawn', encoder_threads=config['encoder_threads'], decoder_threads=1,
                 **cpu_capacity()))


def _reuse_job(job):
    ep, rows = job
    R, p, src, old, old_sam, work, config = _STATE
    result = R.reuse_episode(p, src, old, old_sam, work, ep, rows,
                             config['signature'], config['max_mae'])
    return dict(result, worker_pid=os.getpid())


def _process_loop(function, initializer, initargs, jobs, results):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if initializer is not None:
            initializer(*initargs)
        while True:
            item = jobs.get()
            if item is None:
                return
            results.put((True, function(item[0])))
    except BaseException:
        results.put((False, traceback.format_exc()))


def spawn_results(function, items, processes, initializer=None, initargs=()):
    """Bounded spawn queue; detect dead workers and terminate only our own children."""
    ctx = mp.get_context('spawn')
    jobs, results = ctx.Queue(maxsize=2 * processes), ctx.Queue()
    workers = [ctx.Process(target=_process_loop, args=(function, initializer, initargs, jobs, results),
                           name=f'sam-cpu-{i}') for i in range(processes)]
    iterator = iter(items)
    complete = False
    try:
        for worker in workers:
            worker.start()
        pending = 0
        def enqueue():
            nonlocal pending
            try:
                item = next(iterator)
            except StopIteration:
                return
            jobs.put((item,))
            pending += 1
        for _ in range(2 * processes):
            enqueue()
        while pending:
            try:
                ok, value = results.get(timeout=1)
            except Empty:
                dead = [(worker.pid, worker.exitcode) for worker in workers if worker.exitcode is not None]
                if dead:
                    raise RuntimeError(f'CPU workers exited unexpectedly (PID, exitcode): {dead}')
                continue
            if not ok:
                raise RuntimeError(f'CPU worker failed:\n{value}')
            pending -= 1
            yield value
            enqueue()
        for _ in workers:
            jobs.put(None)
        complete = True
    finally:
        for worker in workers:
            if worker.pid is not None:
                if not complete and worker.is_alive():
                    worker.terminate()
                worker.join()
        for queue in (jobs, results):
            queue.cancel_join_thread()
            queue.close()


def reuse_results(args, p, src, old, work, episodes, groups, signature, runtime):
    """Yield small result dictionaries; each worker loads/decodes its own videos.

    Process's public terminate/join API promptly stops only these CPU children on an
    exception or interrupt, leaving atomically completed episode markers resumable.
    """
    if not episodes:
        return
    native_thread_limits()
    count = min(args.compose_workers, len(episodes))
    episodes = sorted(episodes, key=lambda ep: (-int(src.eps.loc[ep, 'length']) * groups[ep].file.nunique(), ep))
    logs = Path(runtime) / 'cpu-workers' / f'run-{time.time_ns()}'
    logs.mkdir(parents=True)
    config = dict(pipeline=str(Path(p.__file__).resolve()), mapping=str(p.MAPPING),
                  instructions=str(p.RAW_ROOT), runtime=str(runtime), raw=str(src.root),
                  old_raw=str(old.root), old_sam=str(args.old_sam), work=str(work.root),
                  queue_tag=work.queue.name, signature=signature, max_mae=args.alignment_max_mae,
                  encoder_threads=getattr(args, 'encoder_threads', 2), logs=str(logs))
    print(f'[cpu] backend=spawn processes={count} encoder_threads={config["encoder_threads"]} '
          f'decoder_threads=1 allocated_cpus={cpu_capacity()["available_cpus"]} logs={logs}', flush=True)
    yield from spawn_results(_reuse_job, ((ep, groups[ep]) for ep in episodes), count,
                             initializer=_initialize_worker, initargs=(config,))
