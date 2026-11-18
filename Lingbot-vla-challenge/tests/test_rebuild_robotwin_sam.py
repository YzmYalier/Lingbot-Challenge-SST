"""Incremental SAM safety/selection tests. No GPU or HTTP requests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import rebuild_robotwin_sam as sam


class IncrementalSAMTests(unittest.TestCase):
    def test_selection_uses_cam_high_and_mask_length_preserves_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'sam_masks').mkdir()
            eps = pd.DataFrame({'length': [5] * 4, 'dataset_from_index': [0, 5, 10, 15],
                                'dataset_to_index': [5, 10, 15, 20]}, index=range(4))
            src = SimpleNamespace(eps=eps, fps=15, n_video_frames=lambda e: 5)
            old = SimpleNamespace(eps=eps.copy(), fps=15, root=root / 'old',
                                  n_video_frames=lambda e: 4 if e == 1 else 5)
            rows = []
            for ep in range(4):
                for key in ('cam_1', 'cam_2'):
                    rows.append(dict(episode_index=ep, source_episode_index=ep, video_key=key,
                                     n_frames=4 if ep in (1, 3) else 5, file=f'{ep}.mkv'))
            pd.DataFrame(rows).to_parquet(root / 'sam_masks/index.parquet')
            p = SimpleNamespace(cam_keys=lambda mode: ['cam_1', 'cam_2'])
            info = dict(source=str(old.root), subset=False, mode='multi', video_keys=p.cam_keys('multi'))
            plan, _ = sam.make_plan(p, src, old, root, info)
            self.assertEqual(plan['reuse_episodes'], [0, 2])
            self.assertEqual(plan['recompute_episodes'], [1, 3])
            self.assertEqual(plan['recompute_frames'], 10)
            self.assertEqual(plan['recompute_group_percent'], 50)
            # A subset/reindexed source must never be accepted.
            info['subset'] = True
            with self.assertRaisesRegex(ValueError, 'original full'):
                sam.make_plan(p, src, old, root, info)

    def test_alignment_accepts_encoding_noise_but_rejects_shift_and_short(self):
        original = np.stack([np.full((8, 8, 3), 20 + 30*t, np.uint8) for t in range(5)])
        result = sam.check_alignment(original, original + np.uint8(1))
        self.assertEqual(result['frames_checked'], 5)
        self.assertEqual(result['mean_mae'], 1)
        with self.assertRaises(sam.AlignmentError):
            sam.check_alignment(original, np.roll(original, 1, axis=0))
        with self.assertRaises(sam.AlignmentError):
            sam.check_alignment(original[:-1], original)

    def test_only_damaged_ids_go_to_models_and_judge_is_explicit(self):
        args = SimpleNamespace(pipeline='pipeline', raw='raw', out='out', work='work',
                               qwen_url='http://host:8000/v1', sam3_gpus='6', compose_workers=2)
        info = {'judge': dict(model='Qwen', threshold=.5, outside_scale=.2, regrow=2., regrow_min_area=.002)}
        argv = sam.pipeline_argv(args, info, 'masks', [93, 94])
        self.assertEqual(argv[argv.index('--episodes') + 1], '93,94')
        self.assertIn('--judge', argv)
        self.assertNotIn('--no-judge', argv)
        self.assertEqual(argv[argv.index('--qwen-url') + 1], 'http://host:8000/v1/chat/completions')
        self.assertNotIn('--episodes', sam.pipeline_argv(args, info, 'assemble'))
        with self.assertRaises(ValueError):
            sam.pipeline_argv(args, info, 'assemble', [93, 94])
        with self.assertRaises(ValueError):
            sam.pipeline_argv(args, info, 'masks', [])

    def test_alignment_failure_promotes_without_copying_old_mask(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            q = root / 'q.json'
            q.write_text(json.dumps({'phrases': [['robot']]}))
            p = SimpleNamespace(done_ok=lambda *a: False,
                                source_frames=lambda source, ep: source.frames,
                                decode_mask_video=Mock())
            old = SimpleNamespace(frames=np.zeros((2, 8, 8, 3), np.uint8))
            src = SimpleNamespace(frames=np.full((2, 8, 8, 3), 50, np.uint8),
                                  eps=pd.DataFrame({'length': [2]}))
            work = SimpleNamespace(q=lambda ep: q)
            result = sam.reuse_episode(p, src, old, root, work, 0, None, {}, 6)
            self.assertIn('recompute_reason', result)
            p.decode_mask_video.assert_not_called()

    def test_cached_reuse_rejects_changed_video(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            q, done, video = root / 'q.json', root / 'done.json', root / 'g.mp4'
            q.write_text('{}')
            video.write_bytes(b'original')
            done.write_text(json.dumps(dict(n_frames=5, reuse_file_sha256={str(video): sam.file_hash(video)})))
            p = SimpleNamespace(done_ok=lambda *a: True)
            work = SimpleNamespace(root=root, q=lambda ep: q, done_json=lambda ep: done)
            self.assertTrue(sam.reusable_done(p, work, 0, {}, 5))
            video.write_bytes(b'changed')
            self.assertFalse(sam.reusable_done(p, work, 0, {}, 5))

    def test_old_judges_seed_only_normal_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'sam_masks').mkdir()
            pd.DataFrame([dict(episode_index=0, source_episode_index=0, phrase='robot',
                               instruction='move', obj_id=7, frame_index=1, area=.2,
                               p_yes=.1, keep=False, error=None)]).to_parquet(root / 'sam_masks/judge_log.parquet')
            q = root / 'q.json'
            q.write_text('{}')
            work = SimpleNamespace(q=lambda ep: q, judge_json=lambda ep: root / f'judge-{ep}.json')
            p = SimpleNamespace(phrase_instructions=lambda q: {'robot': 'move'})
            sam.seed_judges(p, work, root, {'judge': {'threshold': .5}}, [0])
            saved = json.loads(work.judge_json(0).read_text())
            self.assertFalse(saved['phrases']['robot']['verdicts']['7'][0]['keep'])
            self.assertFalse(work.judge_json(1).exists())

    def test_no_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                sam.inside(Path(d), '../mask.mkv')


if __name__ == '__main__':
    unittest.main()
