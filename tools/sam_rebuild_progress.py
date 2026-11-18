"""Progress reporting for the isolated SAM rebuild; does not edit upstream modules."""
from contextlib import contextmanager
import concurrent.futures as cf
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace


def duration(seconds):
    if seconds is None:
        return '?'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


class Progress:
    def __init__(self, root, interval=30):
        self.root = Path(root)
        self.interval = interval
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.phase_started = self.started
        self.phase = 'prepare'
        self.total = None
        self.completed = self.cached = self.promoted = self.failed = 0
        self.baseline = 0
        self.status = 'running'
        self.refresh = None
        self.detail = ''
        self.thread = None

    def __enter__(self):
        self.emit()
        self.thread = threading.Thread(target=self._heartbeat, name='sam-progress', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        with self.lock:
            self.status = 'failed' if kind else 'finished'
            if error is not None:
                self.detail = f'{kind.__name__}: {error}'
        self.emit()

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit()

    def stage(self, phase, total=None, *, completed=0, cached=0, refresh=None):
        self.emit()  # close the previous phase with its actual completed count
        with self.lock:
            self.phase, self.total, self.completed = phase, total, completed
            self.cached, self.baseline = cached, completed
            self.promoted = self.failed = 0
            self.detail = ''
            self.refresh = refresh
            self.phase_started = time.monotonic()
        self.emit()

    def advance(self, n=1, *, cached=0, promoted=0, failed=0, detail=None):
        with self.lock:
            self.completed += n
            self.cached += cached
            self.promoted += promoted
            self.failed += failed
            if detail is not None:
                self.detail = detail

    def future_done(self, future):
        if future.cancelled() or future.exception() is not None:
            self.advance(0, failed=1)
        else:
            self.advance()

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.phase_started
            # Cached work is not throughput. Do not predict a near-zero ETA on resume.
            fresh = self.completed - max(self.baseline, self.cached)
            rate = fresh / elapsed if elapsed > 0 and fresh > 0 else None
            remaining = max(0, self.total - self.completed) if self.total is not None else None
            eta = (remaining / rate if rate else None) if remaining else (0 if remaining == 0 else None)
            return {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': self.status,
                    'stage': self.phase, 'completed': self.completed, 'total': self.total,
                    'percent': 100 * self.completed / self.total if self.total else None,
                    'cached': self.cached, 'promoted_to_sam': self.promoted, 'failed': self.failed,
                    'stage_seconds': elapsed, 'run_seconds': now - self.started,
                    'units_per_second': rate, 'eta_seconds': eta, 'detail': self.detail}

    def emit(self):
        with self.lock:
            if self.refresh is not None:
                self.completed = self.refresh()
            state = self.snapshot()
            count = (f'{state["completed"]}/{state["total"]} '
                     f'({state["percent"] or 0:.1f}%)' if state['total'] is not None else 'working')
            line = (f'[{state["time"]}] [{state["stage"]}] {count} '
                    f'elapsed={duration(state["stage_seconds"])} ETA~={duration(state["eta_seconds"])} '
                    f'cached={state["cached"]} promoted={state["promoted_to_sam"]} '
                    f'failed={state["failed"]} status={state["status"]}')
            if state['detail']:
                line += f' | {state["detail"]}'
            print(line, flush=True)
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / 'progress.json.tmp'
            tmp.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
            os.replace(tmp, self.root / 'progress.json')
            with (self.root / 'progress.log').open('a') as log:
                log.write(line + '\n')


@contextmanager
def observe_assembly(pipeline, progress, total_videos):
    """Count completed futures/copies in the parent, without changing upstream files.

    Proxy module attributes instead of mutating the shared concurrent.futures/shutil
    modules. Submitted worker functions remain the original picklable functions.
    """
    names = ('cf', 'shutil', 'write_sam_masks', 'link_or_copy', 'write_meta')
    saved = {name: getattr(pipeline, name) for name in names}

    class ObservedPool(cf.ProcessPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            future.add_done_callback(progress.future_done)
            return future

    def copy2(src, dst, *args, **kwargs):
        result = saved['shutil'].copy2(src, dst, *args, **kwargs)
        if Path(dst).suffix == '.mp4':
            progress.advance()
        return result

    def link_or_copy(src, dst):
        result = saved['link_or_copy'](src, dst)
        progress.advance()
        return result

    def masks(out, done, new_id):
        progress.stage('assemble_masks', sum(len(d['groups']) for d in done.values()))
        result = saved['write_sam_masks'](out, done, new_id)
        progress.stage('copy_numeric_data')
        return result

    def meta(*args, **kwargs):
        progress.stage('write_metadata', 1)
        result = saved['write_meta'](*args, **kwargs)
        progress.advance()
        return result

    progress.stage('assemble_videos', total_videos)
    pipeline.cf = SimpleNamespace(**{**vars(saved['cf']), 'ProcessPoolExecutor': ObservedPool})
    pipeline.shutil = SimpleNamespace(**{**vars(saved['shutil']), 'copy2': copy2})
    pipeline.write_sam_masks, pipeline.link_or_copy, pipeline.write_meta = masks, link_or_copy, meta
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(pipeline, name, value)
