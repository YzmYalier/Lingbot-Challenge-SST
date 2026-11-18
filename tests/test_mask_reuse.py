"""CPU checks for moving existing masks and changing only their RGB render style."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import av

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, path, writable
sys.path.insert(0, str(base_path() / 'tools'))
import rebuild_robotwin_sam as R


class MaskReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert Path(R.__file__).resolve() == base_path() / 'tools/rebuild_robotwin_sam.py'
        cls.pipeline = R.import_pipeline(base_path() / 'auto_mask/pipeline_lerobot.py')
        cls.smoke = writable(path('sam_reuse_report').parent / 'tests')
        cls.smoke.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.smoke)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        p = self.pipeline
        judge = p.QwenMaskJudge('http://127.0.0.1:8000/v1/chat/completions', model='Qwen/Qwen3.8-27B', threshold=.5, outside_scale=.2)
        self.info = dict(format=p.FORMAT, prompt_version=p.PROMPT_VERSION, sam3=p.SAM3_ARGS,
                         judge=p.judge_signature(judge, SimpleNamespace(judge_regrow=2., judge_regrow_min_area=.002)),
                         style={'name': 'dim', 'outside_scale': .2},
                         masks={'vcodec': 'ffv1', 'pix_fmt': 'gray', 'values': '0 / 255'})

    def test_dim_to_outline_requires_explicit_conversion(self):
        with self.assertRaisesRegex(ValueError, 'outline0.1'):
            R.validate_pipeline(self.pipeline, self.info)
        before = copy.deepcopy(self.info)
        signature = R.validate_pipeline(self.pipeline, self.info, rerender_style=True)
        self.assertEqual(signature['style'], self.pipeline.MaskStyle('outline', .1, 2).to_dict())
        self.assertEqual(self.info, before)
        self.assertEqual(signature['judge'], self.info['judge'])

    def test_style_conversion_rejects_nonbinary_or_changed_judge(self):
        self.info['masks']['vcodec'] = 'h264'
        with self.assertRaisesRegex(ValueError, 'lossless'):
            R.validate_pipeline(self.pipeline, self.info, rerender_style=True)
        self.info['masks']['vcodec'] = 'ffv1'
        self.info['judge']['question_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'semantics|settings'):
            R.validate_pipeline(self.pipeline, self.info, rerender_style=True)

    def source_fixture(self):
        meta = self.root / 'meta'
        (meta / 'episodes/chunk-000').mkdir(parents=True)
        (meta / 'info.json').write_text(json.dumps(dict(codebase_version='v3.0', total_episodes=2, total_frames=10, fps=15)))
        eps = pd.DataFrame(dict(episode_index=[0, 1], length=[5, 5], dataset_from_index=[0, 5], dataset_to_index=[5, 10], tasks=[['one', 'two'], ['three', 'four']]))
        ep_file = meta / 'episodes/chunk-000/file-000.parquet'
        eps.to_parquet(ep_file)
        src = SimpleNamespace(eps=eps.set_index('episode_index', drop=False), fps=15,
                              seen_instructions=lambda ep: [['one', 'two'], ['three', 'four']][ep],
                              n_video_frames=lambda ep: 5)
        return src, eps, ep_file

    def test_relocated_metadata_checks_all_ids_and_instruction_order(self):
        src, eps, file = self.source_fixture()
        self.assertEqual(R.verify_relocated_source(src, self.root), {'episodes_checked': 2, 'ordered_instructions_checked': 4})
        eps.at[0, 'tasks'] = ['two', 'one']
        eps.to_parquet(file)
        with self.assertRaisesRegex(ValueError, 'ordered seen instructions'):
            R.verify_relocated_source(src, self.root)

    def test_relocated_numeric_indices_cannot_change(self):
        src, eps, file = self.source_fixture()
        eps.at[1, 'dataset_from_index'] = 6
        eps.to_parquet(file)
        with self.assertRaisesRegex(ValueError, 'episode metadata'):
            R.verify_relocated_source(src, self.root)

    def test_short_masks_are_never_padded_or_selected_for_reuse(self):
        src, eps, file = self.source_fixture()
        old = SimpleNamespace(eps=src.eps.copy(), fps=15, root=self.root / 'old', n_video_frames=lambda ep: 5)
        (self.root / 'sam_masks').mkdir()
        pd.DataFrame([dict(episode_index=ep, source_episode_index=ep, video_key=key,
                           n_frames=5 if ep==0 else 4, file=f'{ep}.mkv')
                      for ep in range(2) for key in ('cam_1', 'cam_2')]).to_parquet(self.root/'sam_masks/index.parquet')
        p = SimpleNamespace(cam_keys=lambda mode: ['cam_1', 'cam_2'])
        info = dict(source='old-machine-location', subset=False, mode='multi', video_keys=p.cam_keys('multi'))
        with self.assertRaisesRegex(ValueError, 'original full'):
            R.make_plan(p, src, old, self.root, info)
        plan, _ = R.make_plan(p, src, old, self.root, info, allow_source_relocation=True)
        self.assertEqual(plan['reuse_episodes'], [0])
        self.assertEqual(plan['recompute_episodes'], [1])
        self.assertEqual(plan['recompute_frames'], 5)

    def test_full_length_but_shifted_rgb_is_rejected(self):
        frames = np.stack([np.full((8, 8, 3), i*30, dtype=np.uint8) for i in range(6)])
        with self.assertRaises(R.AlignmentError):
            R.check_alignment(frames, np.roll(frames, 1, axis=0))

    def test_lossless_masks_keep_exact_bits_and_millisecond_pts(self):
        masks = np.zeros((3, 16, 16), dtype=bool)
        masks[:, 4:10, 3:8] = True
        file = self.root/'mask.mkv'
        self.pipeline.encode_mask_video(masks, file, 15)
        np.testing.assert_array_equal(R.decode_final_mask(file, 3, 15, (16, 16)), masks)
        with self.assertRaisesRegex(ValueError, 'frame count'):
            R.decode_final_mask(file, 4, 15, (16, 16))

    def test_nonbinary_source_mask_is_rejected(self):
        file = self.root/'invalid.mkv'
        with av.open(str(file), 'w') as output:
            stream = output.add_stream('ffv1', 15)
            stream.width = stream.height = 16
            stream.pix_fmt = 'gray'
            frame = av.VideoFrame.from_ndarray(np.ones((16, 16), dtype=np.uint8), format='gray')
            for packet in stream.encode(frame): output.mux(packet)
            for packet in stream.encode(): output.mux(packet)
        with self.assertRaisesRegex(ValueError, 'nonbinary'):
            R.decode_final_mask(file, 1, 15, (16, 16))


if __name__ == '__main__':
    unittest.main()
