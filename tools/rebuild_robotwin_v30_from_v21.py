#!/usr/bin/env python3
"""Rebuild v3 videos from a complete v2.1 ZIP, preserving v3 trajectory IDs and labels.

Videos are copied byte-for-byte into one-episode-per-file v3 storage (no concat,
re-encoding, frame interpolation or padding). Every video is actually decoded;
frame count, PTS, shape, content hash and fresh image statistics are recorded.
--smoke-episodes writes ONLY a resumable .partial directory and a timing report.
The completed dataset is published by atomic rename only after a full run.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import zipfile

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = 1
_ZIP = None


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def signature(df):
    """Hash complete chronological numeric trajectories, never task text or IDs."""
    if not np.array_equal(df.frame_index.to_numpy(), np.arange(len(df))):
        raise ValueError('Out-of-order, non-contiguous or duplicate frame indices')
    h = hashlib.sha256()
    for key in ('observation.state', 'action'):
        a = np.stack(df[key]).astype('<f4')
        if not np.isfinite(a).all():
            raise ValueError(f'Non-finite {key}')
        h.update(np.asarray(a.shape, dtype='<i8').tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def episode_mapping(reference, archive, prefix):
    columns = ['episode_index', 'frame_index', 'observation.state', 'action']
    target = {}
    for path in sorted((reference / 'data').glob('*/*.parquet')):
        data = pd.read_parquet(path, columns=columns)
        for ep, df in data.groupby('episode_index'):
            key = signature(df)
            if key in target:
                raise ValueError(f'Ambiguous reference trajectory: {ep}, {target[key]}')
            target[key] = int(ep)
    result = {}
    names = sorted(n for n in archive.namelist() if n.startswith(prefix + 'data/') and n.endswith('.parquet'))
    for i, name in enumerate(names):
        df = pd.read_parquet(io.BytesIO(archive.read(name)), columns=columns)
        if df.episode_index.nunique() != 1:
            raise ValueError(f'Expected one v21 episode in {name}')
        key = signature(df)
        if key not in target or target[key] in result:
            raise ValueError(f'Unmatched or ambiguous v21 trajectory: {name}')
        result[target[key]] = int(df.episode_index.iloc[0])
        if (i + 1) % 500 == 0:
            log(f'Matched {i + 1}/{len(names)} v21 trajectories')
    if len(result) != len(target):
        raise ValueError('The v21 and reference v3 episode sets differ')
    return result


def video_stats(sampled):
    """Same sampling/downsampling convention as LeRobot, with exact sample quantiles."""
    x = np.stack(sampled).astype(np.float32) / 255.0  # N,C,H,W
    stats = {k: fn(x, axis=(0, 2, 3), keepdims=True)[0].tolist()
             for k, fn in [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}
    stats['count'] = [len(sampled)]
    for q, v in zip([1, 10, 50, 90, 99], np.quantile(x, [.01, .10, .50, .90, .99], axis=(0, 2, 3))):
        stats[f'q{q:02d}'] = v[:, None, None].tolist()
    return stats


def validate_video(path, length, fps, shape):
    n_sample = max(min(length, 100), min(int(length ** .75), 10000))
    samples = set(np.rint(np.linspace(0, length - 1, n_sample)).astype(int).tolist())
    sampled, digest, count = [], hashlib.sha256(), 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError(f'{path}: fps {stream.average_rate} != {fps}')
        for i, frame in enumerate(container.decode(stream)):
            ts = float(frame.pts * frame.time_base)
            if abs(ts - i / fps) > 1e-4:
                raise ValueError(f'{path}: frame {i} PTS {ts} != {i / fps}')
            rgb = frame.to_ndarray(format='rgb24')
            if list(rgb.shape) != [shape[1], shape[2], shape[0]]:
                raise ValueError(f'{path}: unexpected frame shape {rgb.shape}')
            digest.update(rgb.tobytes())
            if i in samples:
                # lerobot.auto_downsample_height_width uses integer strides.
                factor = max(shape[1], shape[2]) // 150 if max(shape[1], shape[2]) >= 300 else 1
                sampled.append(rgb[::factor, ::factor].transpose(2, 0, 1))
            count += 1
    if count != length:
        raise ValueError(f'{path}: decoded {count} frames, trajectory has {length}')
    return {'frames': count, 'decoded_rgb_sha256': digest.hexdigest(), 'stats': video_stats(sampled)}


def init_worker(archive):
    global _ZIP
    _ZIP = zipfile.ZipFile(archive)
    av.logging.set_level(av.logging.ERROR)


def rebuild_video(job):
    started = time.monotonic()
    out, record_path = Path(job['output']), Path(job['record'])
    if out.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if record['source_member'] == job['member'] and file_hash(out) == record['sha256']:
            return {**record, 'resumed': True, 'seconds': time.monotonic() - started}
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _ZIP.read(job['member'])  # ZIP CRC is checked by ZipFile.
    tmp = out.with_suffix('.tmp.mp4')
    tmp.write_bytes(data)
    record = validate_video(tmp, job['length'], job['fps'], job['shape'])
    record.update(source_member=job['member'], sha256=hashlib.sha256(data).hexdigest(),
                  bytes=len(data), episode_index=job['episode'], video_key=job['key'])
    if file_hash(tmp) != record['sha256']:
        raise ValueError(f'Copy hash mismatch: {tmp}')
    os.replace(tmp, out)
    write_json(record_path, record)
    return {**record, 'resumed': False, 'seconds': time.monotonic() - started}


def parse_episodes(spec, count):
    if not spec:
        return list(range(count))
    result = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = map(int, part.split('-'))
            result.update(range(a, b + 1))
        else:
            result.add(int(part))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError('Invalid smoke episode selection')
    return sorted(result)


def aggregate_image_stats(items):
    from lerobot.datasets.compute_stats import aggregate_feature_stats
    stats = aggregate_feature_stats([{k: np.asarray(v) for k, v in item.items()} for item in items])
    return {k: v.tolist() for k, v in stats.items()}


def publish_metadata(reference, stage, info, table, keys, records):
    """Keep the existing parquet files and numeric statistics; replace only video metadata."""
    for path in sorted((reference / 'data').glob('*/*.parquet')):
        dest = stage / path.relative_to(reference)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        if file_hash(path) != file_hash(dest):
            raise ValueError(f'Data parquet copy mismatch: {path}')
    ids = table['episode_index'].to_pylist()
    lengths = table['length'].to_pylist()
    stats = json.loads((reference / 'meta/stats.json').read_text())
    chunk_size = int(info['chunks_size'])
    for key in keys:
        values = {
            f'videos/{key}/chunk_index': [ep // chunk_size for ep in ids],
            f'videos/{key}/file_index': [ep % chunk_size for ep in ids],
            f'videos/{key}/from_timestamp': [0.0] * len(ids),
            f'videos/{key}/to_timestamp': [n / info['fps'] for n in lengths],
        }
        per_ep = [records[(ep, key)]['stats'] for ep in ids]
        stats[key] = aggregate_image_stats(per_ep)
        for stat in per_ep[0]:
            values[f'stats/{key}/{stat}'] = [x[stat] for x in per_ep]
        for column, vals in values.items():
            idx = table.schema.get_field_index(column)
            field = table.schema.field(idx)
            table = table.set_column(idx, field, pa.array(vals, type=field.type))
    # Retain the original episode-metadata chunk/file split and its internal references.
    offset = 0
    for path in sorted((reference / 'meta/episodes').glob('*/*.parquet')):
        size = pq.read_metadata(path).num_rows
        dest = stage / path.relative_to(reference)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(offset, size), dest)
        offset += size
    shutil.copyfile(reference / 'meta/tasks.parquet', stage / 'meta/tasks.parquet')
    write_json(stage / 'meta/stats.json', stats)
    write_json(stage / 'meta/info.json', info)


def run(args):
    started = time.monotonic()
    source, reference, output = [Path(p).resolve() for p in (args.source_v21, args.reference_v30, args.out)]
    stage = output.with_name(output.name + '.partial')
    for dest in (output, stage):
        if dest == source or dest.is_relative_to(reference) or reference.is_relative_to(dest) or source.is_relative_to(dest):
            raise ValueError('Output must be separate from both read-only inputs')
    if output.exists():
        raise ValueError(f'Output already exists: {output}; use a new output path')
    info = json.loads((reference / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0' or (reference / 'meta/sam_info.json').exists():
        raise ValueError('--reference-v30 must be the raw v3 dataset, not a SAM derivative')
    keys = [k for k, v in info['features'].items() if v['dtype'] == 'video']
    tables = [pq.read_table(p) for p in sorted((reference / 'meta/episodes').glob('*/*.parquet'))]
    table = pa.concat_tables(tables)
    n = table.num_rows
    if table['episode_index'].to_pylist() != list(range(n)) or n != info['total_episodes']:
        raise ValueError('Reference episode IDs must be contiguous and ordered')
    lengths = table['length'].to_pylist()
    if sum(lengths) != info['total_frames']:
        raise ValueError('Reference total_frames disagrees with lengths')
    source_stat = source.stat()
    fingerprint = {'version': VERSION, 'source': str(source), 'source_size': source_stat.st_size,
                   'source_mtime_ns': source_stat.st_mtime_ns, 'reference': str(reference),
                   'reference_files': {str(p.relative_to(reference)): file_hash(p)
                      for p in sorted(reference.glob('meta/**/*.parquet')) + sorted(reference.glob('meta/*.json'))
                      + sorted(reference.glob('data/*/*.parquet'))}}
    if stage.exists():
        stamp = stage / 'repair_manifest.json'
        if not stamp.exists() or json.loads(stamp.read_text()) != fingerprint:
            raise ValueError(f'Refusing to reuse unrecognized or different-source staging directory: {stage}')
    else:
        stage.mkdir(parents=True)
        write_json(stage / 'repair_manifest.json', fingerprint)
    with zipfile.ZipFile(source) as archive:
        names = [n for n in archive.namelist() if n.endswith('meta/info.json')]
        if len(names) != 1:
            raise ValueError('Expected exactly one v21 dataset in ZIP')
        prefix = names[0][:-len('meta/info.json')]
        old_info = json.loads(archive.read(names[0]))
        if old_info['codebase_version'] != 'v2.1':
            raise ValueError('--source-v21 must contain a v2.1 dataset')
        if old_info['fps'] != info['fps'] or old_info['total_frames'] != info['total_frames']:
            raise ValueError('Source/reference fps or frame totals differ')
        for key in keys:
            if old_info['features'][key]['shape'] != info['features'][key]['shape']:
                raise ValueError(f'Video shape mismatch: {key}')
            # Encoded files come from v21, so codec/pixel-format metadata must follow them.
            if 'info' in old_info['features'][key]:
                info['features'][key]['info'] = old_info['features'][key]['info']
        mapping_path = stage / 'episode_mapping.json'
        if mapping_path.exists():
            mapping = {int(k): v for k, v in json.loads(mapping_path.read_text()).items()}
        else:
            mapping = episode_mapping(reference, archive, prefix)
            write_json(mapping_path, mapping)
        if set(mapping) != set(range(n)) or len(set(mapping.values())) != n:
            raise ValueError('Cached episode mapping is not a one-to-one mapping of all episodes')
        old_eps = {e['episode_index']: e for e in map(json.loads, archive.read(prefix + 'meta/episodes.jsonl').splitlines())}
        for ep, old in mapping.items():
            if lengths[ep] != old_eps[old]['length']:
                raise ValueError(f'Length mismatch at v3 episode {ep}')
        episodes = parse_episodes(args.smoke_episodes, n)
        jobs = []
        for ep in episodes:
            for key in keys:
                if info['features'][key]['shape'] != old_info['features'][key]['shape']:
                    raise ValueError(f'Video shape mismatch: {key}')
                old = mapping[ep]
                member = prefix + old_info['video_path'].format(episode_chunk=old // old_info['chunks_size'],
                                                               episode_index=old, video_key=key)
                archive.getinfo(member)
                relative = info['video_path'].format(video_key=key, chunk_index=ep // info['chunks_size'],
                                                     file_index=ep % info['chunks_size'])
                jobs.append(dict(member=member, output=str(stage / relative),
                                 record=str(stage / 'repair_records' / key / f'{ep:06d}.json'),
                                 length=lengths[ep], fps=info['fps'], shape=info['features'][key]['shape'],
                                 episode=ep, key=key))
    setup_seconds = time.monotonic() - started
    log(f'Validated mapping for {len(mapping)} episodes; processing {len(jobs)} videos with {args.workers} CPU workers')
    video_start, records, fresh = time.monotonic(), {}, []
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                initializer=init_worker, initargs=(str(source),)) as pool:
        for i, record in enumerate(pool.map(rebuild_video, jobs)):
            records[(record['episode_index'], record['video_key'])] = record
            if not record['resumed']:
                fresh.append(record)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                log(f'Validated {i + 1}/{len(jobs)} videos')
    video_seconds = time.monotonic() - video_start
    frames = sum(r['frames'] for r in records.values())
    full_frames = info['total_frames'] * len(keys)
    # Report observed wall-time scaling, not a promise of linear speedup with workers.
    estimated = setup_seconds + video_seconds * full_frames / frames if len(fresh) == len(jobs) else None
    report = {'complete': False, 'smoke_episodes': episodes if args.smoke_episodes else None,
              'workers': args.workers, 'setup_seconds': setup_seconds, 'video_seconds': video_seconds,
              'videos_checked': len(jobs), 'decoded_frames': frames, 'fresh_videos': len(fresh),
              'copied_bytes': sum(r['bytes'] for r in records.values()),
              'estimated_full_video_and_setup_seconds': estimated,
              'estimate_note': 'Same workers and machine load; add metadata/copy overhead. Smoke is not a training dataset.'}
    if not args.smoke_episodes:
        publish_metadata(reference, stage, info, table, keys, records)
        report['complete'] = True
    report['elapsed_seconds'] = time.monotonic() - started
    write_json(stage / 'repair_report.json', report)
    if report['complete']:
        os.rename(stage, output)
        log(f'Complete dataset: {output}')
    else:
        log(f'Smoke only; resumable files and report: {stage}')
    log(json.dumps(report, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-v21', required=True, type=Path, help='Read-only v2.1 ZIP')
    parser.add_argument('--reference-v30', required=True, type=Path, help='Read-only raw v3 dataset')
    parser.add_argument('--out', required=True, type=Path, help='New output directory; never overwrites inputs')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke-episodes', default='', help='E.g. 0,50,93,94; do not publish a full dataset')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    run(args)


if __name__ == '__main__':
    main()
