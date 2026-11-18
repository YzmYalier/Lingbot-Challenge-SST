#!/usr/bin/env python3
"""Read first/last frames through Lingbot's real dataset class, on CPU only."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, interpreter, path, writable


def main():
    if os.environ.get('SUBMISSION_DATA_CHECK') != str(base_path()):
        from run_python import environment
        env = environment('vla', runtime_name='training-data-check')
        env.update(SUBMISSION_DATA_CHECK=str(base_path()), CUDA_VISIBLE_DEVICES='')
        exe = str(interpreter('vla'))
        os.execve(exe, [exe, '-B', str(Path(__file__).resolve()), *sys.argv[1:]], env)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-only', action='store_true')
    args = ap.parse_args()
    import torch
    import yaml
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
    from lingbotvla.data.vla_data.utils import RAW_CAM_HIGH_KEY
    from lingbotvla.data.vla_data.video_utils import get_safe_default_codec
    if not args.raw_only:
        from train import check_dataset, resolve_config
        check_dataset(resolve_config(yaml.safe_load(path('training_recipe').read_text())))
    root = path('raw' if args.raw_only else 'sam_output')
    ds = LeRobotDataset(root.name, root=root, download_videos=False, load_image=True,
                       sam_mode='auto' if args.raw_only else 'multi', aux_raw_image=not args.raw_only,
                       video_integrity='error', video_backend='torchcodec')
    if len(ds) != 548893 or len(ds.meta.episodes) != 2500:
        raise ValueError('Training reader must see all 2500 episodes and 548893 rows')
    keys = ['observation.images.cam_high', 'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist']
    if not args.raw_only:
        keys.append(RAW_CAM_HIGH_KEY)
    random.seed(0)
    samples = []
    for ep in (0, 50, 93, 94, 2499):
        meta = ds.meta.episodes[ep]
        for idx in (int(meta['dataset_from_index']), int(meta['dataset_to_index']) - 1):
            item = ds[idx]
            if int(item['episode_index']) != ep:
                raise ValueError('Training sample crossed an episode boundary')
            for key in keys:
                if item[key].shape != (3, 240, 320) or not torch.isfinite(item[key]).all():
                    raise ValueError(f'Invalid training image: {ep}/{idx}/{key}')
            for key in ('action', 'observation.state'):
                if item[key].shape != (14,) or not torch.isfinite(item[key]).all():
                    raise ValueError(f'Invalid numerical training sample: {ep}/{idx}/{key}')
            if not args.raw_only and item['task'] not in meta['tasks']:
                raise ValueError('Sampled instruction is absent from the episode')
            samples.append(dict(episode=ep, index=idx, task=item['task'], image_keys=keys))
    report = dict(complete=True, dataset=str(root), episodes=2500, rows=len(ds), samples=samples,
                  decoder=get_safe_default_codec(), aux_raw_image=not args.raw_only,
                  note='Full metadata check and 10 actual first/last-frame CPU samples; no GPU training.')
    name = 'raw_loader_smoke.json' if args.raw_only else 'training_loader_smoke.json'
    output = writable(path('sam_reuse_report').parent / name)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
