"""End-to-end converter tests on tiny videos; no production data is written."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rebuild', ROOT / 'tools/rebuild_robotwin_v30_from_v21.py')
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)
KEY = 'observation.images.cam_high'


def make_video(path, n):
    with av.open(str(path), 'w') as c:
        stream = c.add_stream('mpeg4', rate=15)
        stream.width, stream.height, stream.pix_fmt = 16, 16, 'yuv420p'
        for i in range(n):
            rgb = np.full((16, 16, 3), 20 + i * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)


def fixture(root):
    reference = root / 'reference'
    (reference / 'meta/episodes/chunk-000').mkdir(parents=True)
    (reference / 'data/chunk-000').mkdir(parents=True)
    info = dict(codebase_version='v3.0', fps=15, total_episodes=2, total_frames=12, total_tasks=2,
                chunks_size=1000, splits={'train': '0:2'},
                data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
                features={KEY: dict(dtype='video', shape=[3, 16, 16])})
    rows, data, old_data, old_eps = [], [], {}, []
    for ep, n in enumerate([5, 7]):
        d = pd.DataFrame(dict(episode_index=[ep] * n, frame_index=range(n),
                             index=range(sum([5, 7][:ep]), sum([5, 7][:ep + 1])),
                             timestamp=np.arange(n, dtype=np.float32) / 15,
                             task_index=[ep] * n,
                             **{'observation.state': [np.array([ep, t], dtype=np.float32) for t in range(n)],
                                'action': [np.array([ep, t + 1], dtype=np.float32) for t in range(n)]}))
        data.append(d)
        old = 1 - ep
        old_data[old] = d.assign(episode_index=old)
        old_eps.append(dict(episode_index=old, length=n, tasks=[f'old label {old}']))
        row = dict(episode_index=ep, tasks=[f'current label {ep}'], length=n,
                   **{'data/chunk_index': 0, 'data/file_index': 0,
                      'dataset_from_index': sum([5, 7][:ep]), 'dataset_to_index': sum([5, 7][:ep + 1]),
                      'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0,
                      f'videos/{KEY}/chunk_index': 0, f'videos/{KEY}/file_index': 0,
                      f'videos/{KEY}/from_timestamp': ep * 4 / 15,
                      f'videos/{KEY}/to_timestamp': (ep * 4 + n - 1) / 15})
        for k in ['min', 'max', 'mean', 'std', 'q01', 'q10', 'q50', 'q90', 'q99']:
            row[f'stats/{KEY}/{k}'] = [[[0.]] for _ in range(3)]
        row[f'stats/{KEY}/count'] = [n]
        rows.append(row)
    pd.concat(data).to_parquet(reference / 'data/chunk-000/file-000.parquet', index=False)
    pq.write_table(pa.Table.from_pylist(rows), reference / 'meta/episodes/chunk-000/file-000.parquet')
    pd.DataFrame({'task_index': [0, 1]}, index=['current label 0', 'current label 1']).to_parquet(reference / 'meta/tasks.parquet')
    (reference / 'meta/info.json').write_text(json.dumps(info))
    (reference / 'meta/stats.json').write_text(json.dumps({'action': {'mean': [123.]}, KEY: {}}))
    source = root / 'v21.zip'
    old_info = {**info, 'codebase_version': 'v2.1',
                'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}
    with zipfile.ZipFile(source, 'w') as z:
        z.writestr('ds/meta/info.json', json.dumps(old_info))
        z.writestr('ds/meta/episodes.jsonl', '\n'.join(json.dumps(x) for x in old_eps))
        for old, d in old_data.items():
            b = io.BytesIO()
            d.to_parquet(b, index=False)
            z.writestr(f'ds/data/chunk-000/episode_{old:06d}.parquet', b.getvalue())
            path = root / f'{old}.mp4'
            make_video(path, len(d))
            z.write(path, f'ds/videos/chunk-000/{KEY}/episode_{old:06d}.mp4')
    return source, reference


class ConverterTest(unittest.TestCase):
    def test_sam_copy_preserves_frames_and_rejects_concat(self):
        sam_spec = importlib.util.spec_from_file_location('sam_rebuild', ROOT / 'tools/rebuild_robotwin_sam.py')
        sam = importlib.util.module_from_spec(sam_spec)
        sam_spec.loader.exec_module(sam)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dest = root / 'source.mp4', root / 'out.mp4'
            make_video(src, 7)
            self.assertEqual(sam.copy_episode_segment([str(src)], str(dest), 7), 7)
            self.assertEqual(src.read_bytes(), dest.read_bytes())
            with self.assertRaisesRegex(ValueError, 'expected 8'):
                sam.copy_episode_segment([str(src)], str(root / 'bad.mp4'), 8)
            with self.assertRaisesRegex(ValueError, 'one-episode'):
                sam.copy_episode_segment([str(src), str(src)], str(root / 'packed.mp4'), 14)

    def test_full_rebuild_preserves_tables_and_source_video_bytes(self):
        # Run the actual CLI so multiprocessing spawn imports the real worker module.
        import subprocess
        import sys
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, reference = fixture(root)
            before = {str(p): rebuild.file_hash(p) for p in reference.rglob('*') if p.is_file()}
            out = root / 'fixed'
            command = [sys.executable, str(ROOT / 'tools/rebuild_robotwin_v30_from_v21.py'),
                       '--source-v21', str(source), '--reference-v30', str(reference),
                       '--out', str(out), '--workers', '1']
            smoke = subprocess.run(command + ['--smoke-episodes', '0'], capture_output=True, text=True,
                                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            self.assertEqual(smoke.returncode, 0, smoke.stdout + smoke.stderr)
            self.assertFalse(out.exists())
            stage = root / 'fixed.partial'
            self.assertFalse((stage / 'meta/info.json').exists())
            result = subprocess.run(command, capture_output=True, text=True,
                                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(out.exists())
            self.assertFalse(stage.exists())
            self.assertEqual(before, {str(p): rebuild.file_hash(p) for p in reference.rglob('*') if p.is_file()})
            self.assertEqual(rebuild.file_hash(out / 'data/chunk-000/file-000.parquet'),
                             rebuild.file_hash(reference / 'data/chunk-000/file-000.parquet'))
            self.assertEqual(json.loads((out / 'episode_mapping.json').read_text()), {'0': 1, '1': 0})
            metadata = pq.read_table(out / 'meta/episodes/chunk-000/file-000.parquet').to_pydict()
            self.assertEqual(metadata['tasks'], [['current label 0'], ['current label 1']])
            self.assertEqual(metadata[f'videos/{KEY}/from_timestamp'], [0., 0.])
            np.testing.assert_allclose(metadata[f'videos/{KEY}/to_timestamp'], [5 / 15, 7 / 15])
            self.assertEqual(json.loads((out / 'meta/stats.json').read_text())['action'], {'mean': [123.]})
            with zipfile.ZipFile(source) as z:
                for ep, old in [(0, 1), (1, 0)]:
                    self.assertEqual((out / f'videos/{KEY}/chunk-000/file-{ep:03d}.mp4').read_bytes(),
                                     z.read(f'ds/videos/chunk-000/{KEY}/episode_{old:06d}.mp4'))
            # Completed output is never overwritten.
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)

    def test_short_video_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'short.mp4'
            make_video(path, 4)
            with self.assertRaisesRegex(ValueError, 'decoded 4 frames'):
                rebuild.validate_video(path, 5, 15, [3, 16, 16])

    def test_output_cannot_be_inside_original_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, reference = fixture(root)
            args = SimpleNamespace(source_v21=source, reference_v30=reference, out=reference / 'new',
                                   workers=1, smoke_episodes='0')
            with self.assertRaisesRegex(ValueError, 'separate'):
                rebuild.run(args)
            self.assertFalse((reference / 'new.partial').exists())


if __name__ == '__main__':
    unittest.main()
