"""Bounded GPU scheduling and process-based frame preparation; upstream stays read-only."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys

from sam_rebuild_cpu import LEGACY_REBUILD_SHA256, cpu_capacity, native_thread_limits

# Audited predecessors: scheduling changed, frame/mask/judge semantics did not.
AUDITED_REBUILD = {2: LEGACY_REBUILD_SHA256,
                   3: 'b5e19af8321389df4cc230e35dfc27bb42a989d50e6dc164a47860e5543b46dd'}


def compatible_single_manifest(previous, expected):
    if previous == expected:
        return True
    version = previous.get('version')
    if version not in AUDITED_REBUILD or expected.get('version') != 4:
        return False
    helper = Path(__file__).resolve()
    legacy = {**expected, 'version': version, 'inputs_sha256': dict(expected['inputs_sha256'])}
    legacy['inputs_sha256'].pop(str(helper), None)
    if version == 2:
        legacy['inputs_sha256'].pop(str(helper.with_name('sam_rebuild_cpu.py')), None)
    legacy['inputs_sha256'][str(helper.with_name('rebuild_robotwin_sam.py'))] = AUDITED_REBUILD[version]
    return previous == legacy


def positive(value):
    value = int(value)
    if value < 1:
        raise ValueError('GPU scheduling values must be positive')
    return value


def add_arguments(parser):
    for option, env, default, help_text in (
        ('prep-workers', 'DATAFIX_PREP_WORKERS', 'auto', 'Frame preparation processes; auto: up to 8'),
        ('max-queued', 'DATAFIX_MAX_QUEUED', 'auto', 'Preparing + queued + active SAM episodes; auto: 4 per GPU'),
        ('gpu-compose-workers', 'DATAFIX_GPU_COMPOSE_WORKERS', 'auto', 'GPU-stage rendering processes; auto: up to 8'),
        ('judge-concurrency', 'DATAFIX_JUDGE_CONCURRENCY', '16', 'Concurrent Qwen judge request chains'),
        ('judge-episodes', 'DATAFIX_JUDGE_EPISODES', '4', 'Episodes undergoing Qwen judge concurrently'),
    ):
        parser.add_argument('--' + option, default=os.environ.get(env, default), help=help_text)


def scheduling(args):
    gpus = [g.strip() for g in (args.sam3_gpus or '').split(',') if g.strip()]
    if len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError('SAM GPU IDs must be distinct physical numeric IDs')
    count = max(1, len(gpus))  # Read-only plans and assembly may omit GPU IDs.
    available = max(1, int(cpu_capacity()['available_cpus']))
    defaults = dict(prep_workers=min(8, count * 2, max(1, available // 4)),
                    max_queued=count * 4,
                    gpu_compose_workers=min(8, positive(args.compose_workers)),
                    judge_concurrency=16, judge_episodes=4)
    return {key: default if str(getattr(args, key, 'auto')).lower() == 'auto'
            else positive(getattr(args, key)) for key, default in defaults.items()}


def check_shared_gpu(gpus):
    if '0' not in gpus:
        return
    # Check after the local Qwen server is ready. A guard, not an OOM guarantee.
    output = subprocess.check_output(
        ['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    free_mib = int(output.strip())
    if free_mib < 16 * 1024:
        raise RuntimeError(f'GPU0 has only {free_mib / 1024:.1f} GiB free; shared SAM requires at least '
                           '16 GiB. Exclude GPU0 or lower Qwen memory allocation before restarting Qwen.')
    print(f'[gpu] GPU0 shared with Qwen: {free_mib / 1024:.1f} GiB free; adding one SAM worker.', flush=True)


_FRAME_STATE = None
_AV = None


def _limited_av_open(*args, **kwargs):
    container = _AV.open(*args, **kwargs)
    if isinstance(container, _AV.container.InputContainer):
        for stream in container.streams.video:
            stream.codec_context.thread_count = 1
    return container


def _cpu_pipeline(pipeline):
    global _AV
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    native_thread_limits()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(Path(pipeline).parent))
    p = importlib.import_module('pipeline_lerobot')
    if Path(p.__file__).resolve() != Path(pipeline).resolve():
        raise ValueError('Unexpected pipeline module in CPU process')
    p.cv2.setNumThreads(1)
    p.pa.set_cpu_count(1)
    p.pa.set_io_thread_count(1)
    _AV = p.av
    from types import SimpleNamespace
    p.av = SimpleNamespace(**{**vars(_AV), 'open': _limited_av_open})
    return p


def _init_frames(pipeline, raw, root, queue_tag, logs):
    global _FRAME_STATE
    p = _cpu_pipeline(pipeline)
    _FRAME_STATE = p, p.Source(Path(raw)), p.Work(Path(root), 'multi', queue_tag)
    Path(logs).mkdir(parents=True, exist_ok=True)
    Path(logs, f'worker-{os.getpid()}.json').write_text(json.dumps(
        dict(pid=os.getpid(), backend='spawn', decoder_threads=1)) + '\n')


def _prepare_frames(ep, phrases, mask_name):
    p, src, work = _FRAME_STATE
    p.write_frames_and_job(src, work, ep, phrases, mask_name)


def _init_compose(pipeline, encoder_threads, log):
    p = _cpu_pipeline(pipeline)
    p.ENCODE_OPTIONS = {**p.ENCODE_OPTIONS, 'svtav1-params': f'lp={encoder_threads}'}
    p.quiet_stderr(log)


@contextmanager
def pipeline_runtime(p, argv, encoder_threads=2):
    """Adapt only this call; threads wait on CPU processes, HTTP remains threaded."""
    def option(name):
        return argv[argv.index('--' + name) + 1]
    if option('stage') != 'masks':
        yield
        return
    gpus = [g for g in option('sam3-gpus').split(',') if g]
    check_shared_gpu(gpus)
    prep = positive(option('prep-threads'))
    queue_tag = 'local'
    if '--shard' in argv:
        rank, nodes = option('shard').split('/')
        queue_tag = f'rank{rank}of{nodes}'
    logs = Path(p.AUTO_MASK) / 'frame-workers'
    print(f'[gpu] sam_workers={len(gpus)} prep_processes={prep} '
          f'max_in_flight={option("max-queued")} compose_processes={option("compose-workers")} '
          f'judge_concurrency={option("judge-concurrency")} judge_episodes={option("judge-episodes")}', flush=True)
    original_frames, original_quiet = p.write_frames_and_job, p.quiet_stderr
    with ProcessPoolExecutor(prep, mp_context=mp.get_context('spawn'), initializer=_init_frames,
                             initargs=(str(p.__file__), option('src'), option('work'), queue_tag, str(logs))) as pool:
        def dispatch_frames(src, work, ep, phrases, mask_name):
            # Metadata is loaded once per process; no video arrays are pickled.
            pool.submit(_prepare_frames, ep, phrases, mask_name).result()
        p.write_frames_and_job = dispatch_frames
        p.quiet_stderr = partial(_init_compose, str(p.__file__), encoder_threads)
        try:
            yield
        finally:
            p.write_frames_and_job, p.quiet_stderr = original_frames, original_quiet
