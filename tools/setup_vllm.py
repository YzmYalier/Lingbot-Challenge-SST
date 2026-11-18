#!/usr/bin/env python3
"""Build the pinned vLLM CUDA source in a submission-owned Conda environment."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, interpreter, path, runtime_env, writable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    settings = config()['vllm_build']
    source = writable(path('vllm_source'))
    exe = interpreter('vllm')
    prefix = writable(exe.parent.parent)
    work = writable(path('vllm_build_work'))
    if exe.name != 'python' or exe.parent.name != 'bin':
        raise ValueError('vllm interpreter must be <Conda prefix>/bin/python')
    if prefix.exists() and not (prefix / 'conda-meta').is_dir():
        raise ValueError(f'Refusing an existing non-Conda environment: {prefix}')
    cuda = path('cuda_toolkit')
    if not (cuda / 'bin/nvcc').is_file():
        raise FileNotFoundError(f'CUDA toolkit not found: {cuda}')
    commands = []
    if not source.exists():
        commands.append(['git', 'clone', '--depth', '1', '--branch', settings['tag'],
                         'https://github.com/vllm-project/vllm.git', str(source)])
    if not exe.exists():
        commands.append([str(path('conda')), 'create', '-y', '-p', str(prefix),
                         '--override-channels', '-c', 'conda-forge', '--no-default-packages',
                         'python=' + settings['python'], 'pip'])
    install = ['uv', 'pip', 'install', '--python', str(exe), '--editable', str(source),
               '--torch-backend=' + settings['torch_backend'],
               '--constraints', str(path('vllm_runtime_constraints')),
               '--build-constraints', str(path('vllm_build_constraints')), '--verbose']
    for command in commands + [install]:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    work.mkdir(parents=True, exist_ok=True)
    env = runtime_env('setup-vllm')
    env.update(CONDA_REGISTER_ENVS='false', CUDA_HOME=str(cuda),
               TORCH_CUDA_ARCH_LIST=settings['cuda_arch_list'],
               MAX_JOBS=str(settings['max_jobs']), NVCC_THREADS=str(settings['nvcc_threads']),
               VLLM_USE_PRECOMPILED='0', CMAKE_BUILD_TYPE='Release',
               FETCHCONTENT_BASE_DIR=str(work / 'deps'),
               CCACHE_DIR=str(work / 'ccache'), SCCACHE_DIR=str(work / 'sccache'))
    env['PATH'] = os.pathsep.join([str(prefix / 'bin'), str(cuda / 'bin'), env['PATH']])
    with (work / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        logfile = work / 'build.log'
        print(f'Full CUDA source build; log={logfile}', flush=True)
        with logfile.open('a') as log:
            for command in commands:
                subprocess.run(command, cwd=base_path(), env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            if commit != settings['commit']:
                raise ValueError(f'vLLM source commit mismatch: {commit}')
            subprocess.run(install, cwd=base_path(), env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            # Import the compiled extension and execute one CUDA op, not just metadata.
            probe = subprocess.check_output([str(exe), '-B', '-c',
                'import json,torch,vllm,vllm._C; '
                'x=torch.ones(1,device="cuda"); '
                'print(json.dumps({"vllm":vllm.__version__,"torch":torch.__version__, '
                '"cuda":torch.version.cuda,"cuda_test":float(x.sum()), '
                '"extension":vllm._C.__file__}))'], cwd=base_path(), env=env, text=True)
            packages = subprocess.check_output([str(exe), '-m', 'pip', 'freeze'], env=env, text=True)
        (work / 'requirements-resolved.txt').write_text(packages)
        report = dict(complete=True, source=str(source), commit=commit, settings=settings,
                      interpreter=str(exe), elapsed_seconds=time.time()-started, probe=probe,
                      build_log=str(logfile), precompiled_wheel=False)
        (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(probe, flush=True)


if __name__ == '__main__':
    main()
