#!/usr/bin/env python3
"""Drain SAM prefill, own a Qwen service, finish all masks and validate both datasets."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, interpreter, path, runtime_env, writable
from process_data import run, write_json
from services import Service, check_qwen, qwen_command, qwen_url


def judge_probe(url):
    """Exercise two real images and yes/no logprobs before any offline judging."""
    import numpy as np
    import rebuild_robotwin_sam as R
    p = R.import_pipeline(base_path() / 'auto_mask/pipeline_lerobot.py')
    src, work = p.Source(path('raw')), p.Work(path('sam_reuse_work'), 'multi')
    plan = json.loads((work.root / 'incremental_plan.json').read_text())
    for ep in plan['recompute_episodes']:
        q = p.load_qwen(work.q(ep), src, ep, 'multi')
        masks = p.mask_index(work, ep, src.n_video_frames(ep))
        for phrase, (file, key, sidecar) in masks.items():
            with np.load(file) as data:
                labels = data[key]
            occupied = np.flatnonzero(np.any(labels > 0, axis=(1, 2)))
            if not len(occupied):
                continue
            t = int(occupied[0])
            frame = p.source_frames(src, ep)[t]
            mask = labels[t] == int(labels[t].max())
            instruction = next(ins for ins, phrases in zip(q['instructions'], q['phrases']) if phrase in phrases)
            judge = p.QwenMaskJudge(url, model=config()['qwen']['model_name'], timeout=180)
            verdict = judge.ask(judge.frame_image(frame), frame, mask, instruction, phrase, strict=True)
            if verdict.get('error') or verdict.get('p_yes') is None:
                raise RuntimeError(f'Qwen multimodal/logprobs check failed: {verdict}')
            return dict(complete=True, episode=ep, frame=t, phrase=phrase,
                        config=judge.config(), verdict=verdict,
                        note='Protocol check only; this verdict is not injected into training caches.')
    raise ValueError('No existing object mask for a real two-image Qwen probe; run prefill_sam first')


def main():
    if os.environ.get('SUBMISSION_FINISH_REPAIR') != str(base_path()):
        exe = str(interpreter('data'))
        env = runtime_env('incremental-repair')
        env['SUBMISSION_FINISH_REPAIR'] = str(base_path())
        os.execve(exe, [exe, '-B', str(Path(__file__).resolve()), *sys.argv[1:]], env)
    if len(sys.argv) != 1:
        raise ValueError('Configure all resources through base.json; this command takes no arguments')
    audit = writable(path('sam_reuse_report').parent)
    audit.mkdir(parents=True, exist_ok=True)
    work = writable(path('sam_reuse_work'))
    status_file = audit / 'completion_status.json'
    with (audit / 'completion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        started = time.time()
        service = None
        status = dict(complete=False, started_at=started, stage='preflight')
        def stage(name):
            status.update(stage=name, updated_at=time.time())
            write_json(status_file, status)
            print(f'[finish] {name}', flush=True)
        try:
            stage('preflight')
            if not path('sam_output').exists():
                if not json.loads((path('vllm_build_work') / 'report.json').read_text()).get('complete'):
                    raise ValueError('Complete the vLLM source build first')
                downloads = json.loads((path('public_weights_state') / 'last_run.json').read_text())
                expected = set()
                from download_public_weights import load_manifest, destination
                for entry in load_manifest()['files']:
                    if entry['group'] == 'qwen3_8':
                        expected.add(str(destination(entry).relative_to(base_path())))
                verified = {f['path'] for f in downloads['files'] if f.get('ok')}
                if not expected <= verified:
                    raise ValueError('The latest download report must verify all Qwen files')
                stage('drain_sam_prefill')
                (work / 'prefill.stop').touch()
                # Prefill stops accepting jobs and finishes its bounded active queue.
                with (audit / 'driver.lock').open('a') as driver:
                    while True:
                        try:
                            fcntl.flock(driver, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            time.sleep(2)
                    fcntl.flock(driver, fcntl.LOCK_UN)
                stage('start_qwen')
                service = Service('qwen', qwen_command(), config()['qwen']).start()
                check_qwen(qwen_url())
                stage('qwen_two_image_logprobs_check')
                write_json(audit / 'qwen_probe.json', judge_probe(qwen_url()))
                stage('sam_judge_render')
                run([interpreter('data'), '-B', base_path() / 'tools/repair_data_from_masks.py',
                     '--stage', 'masks', '--qwen-url', qwen_url(), '--sam-gpus', config()['processing']['sam_gpus'],
                     '--compose-workers', str(config()['processing']['compose_workers'])],
                    'finish-masks', runtime_env('incremental-repair'))
                stage('assemble_all_2500')
                run([interpreter('data'), '-B', base_path() / 'tools/repair_data_from_masks.py',
                     '--stage', 'assemble', '--compose-workers', str(config()['processing']['compose_workers'])],
                    'finish-assemble', runtime_env('incremental-repair'))
            stage('validate_all_data')
            run([interpreter('data'), '-B', base_path() / 'tools/repair_data_from_masks.py', '--stage', 'validate'],
                'finish-validate', runtime_env('incremental-repair'))
            stage('training_dataset_preflight')
            run([interpreter('data'), '-B', base_path() / 'tools/check_training_data.py'],
                'finish-training-data-check', runtime_env('incremental-repair'))
            status.update(complete=True, raw=str(path('raw')), sam_output=str(path('sam_output')),
                          elapsed_seconds=time.time()-started,
                          limitations=['No VLA training or RoboTwin evaluation was launched.'])
            stage('complete')
        except BaseException:
            status.update(complete=False, error=traceback.format_exc(), elapsed_seconds=time.time()-started)
            stage('failed')
            raise
        finally:
            if service is not None:
                service.stop()


if __name__ == '__main__':
    main()
