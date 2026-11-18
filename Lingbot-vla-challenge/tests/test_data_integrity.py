"""CPU regression tests for padding supervision and episode-local video reads."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loss_module = load('lingbotvla/models/vla/lingbot_vla/action_loss.py', 'action_loss')
video = load('lingbotvla/data/vla_data/video_integrity.py', 'video_integrity')


def method(relative, class_name, name, **namespace):
    """Exercise actual policy/loader methods without loading multi-GB model dependencies."""
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(torch=torch, Tensor=torch.Tensor, **namespace)
    compiled = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(compiled), str(ROOT / relative), 'exec'), scope)
    return scope[name]


class ActionLossTest(unittest.TestCase):
    def test_valid_element_mean_and_gradient(self):
        losses = torch.tensor([[[2., 4., 999.], [100., 200., 999.]],
                               [[6., 8., 999.], [10., 12., 999.]]], requires_grad=True)
        joint = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
        pad = torch.tensor([[False, True], [False, False]])
        loss, per = loss_module.masked_action_loss(losses, joint, pad, 3)
        self.assertEqual(loss.item(), 7.)  # (2+4+6+8+10+12)/6
        torch.testing.assert_close(per, torch.tensor([3., 9.]))
        loss.backward()
        self.assertEqual(losses.grad[0, 1].abs().sum(), 0.)
        self.assertEqual(losses.grad[..., 2].abs().sum(), 0.)
        self.assertAlmostEqual(losses.grad[1, 1, 1].item(), 1 / 6)

    def test_no_joint_mask_float_padding_and_all_pad(self):
        losses = torch.tensor([[[2., 4., 100.], [200., 200., 100.]]], requires_grad=True)
        loss, _ = loss_module.masked_action_loss(losses, None, torch.tensor([[0., 1.]]), 2)
        self.assertEqual(loss.item(), 3.)
        zero, per = loss_module.masked_action_loss(losses, None, torch.ones(1, 2), 2)
        zero.backward()
        self.assertEqual(zero.item(), 0.)
        self.assertEqual(per.item(), 0.)
        self.assertEqual(losses.grad.abs().sum(), 0.)

    def test_repeat_and_mixed_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            losses = torch.tensor([[[2., 4.], [100., 100.]]], dtype=dtype).repeat(2, 1, 1)
            loss, per = loss_module.masked_action_loss(losses, torch.ones(1, 2, 2),
                                                     torch.tensor([[0, 1]]), 2, repeat=True)
            self.assertEqual(loss.item(), 3.)
            self.assertEqual(per.shape, (2,))
            self.assertEqual(loss.dtype, torch.float32)

    def test_bad_shapes_rejected(self):
        for pad in (torch.zeros(1, 2, 1), torch.zeros(1, 1), torch.zeros(2, 2)):
            with self.assertRaises(ValueError):
                loss_module.masked_action_loss(torch.ones(1, 2, 3), None, pad, 3)

    def test_policy_forward_integration_and_legacy(self):
        for version, cls in [('modeling_lingbot_vla.py', 'LingbotVlaPolicy'),
                             ('modeling_lingbot_vla_v2.py', 'LingbotVlaV2Policy')]:
            forward = method('lingbotvla/models/vla/lingbot_vla/' + version, cls, 'forward',
                             masked_action_loss=loss_module.masked_action_loss)
            losses = torch.tensor([[[2., 4.], [100., 100.]]])
            zero = torch.tensor(0.)
            values = ((losses, zero, None, zero, {}) if version.endswith('vla.py') else
                      (losses, zero, zero, zero, None, zero, zero, {}, None, None, None))
            for enabled, expected in [(False, 51.5), (True, 3.)]:
                obj = SimpleNamespace(config=SimpleNamespace(mask_action_pad=enabled, action_dim=2,
                                      action_fp32=False, align_params={}, loss_type='L1_fm'),
                                      model=SimpleNamespace(forward=lambda *a, **kw: values))
                result = forward(obj, None, None, None, None, None, None,
                                 joint_mask=None, action_is_pad=torch.tensor([[False, True]]))
                self.assertEqual(result[1].item(), expected)


class Columns:
    def __init__(self, data):
        self.data = data

    def select_columns(self, names):
        return Columns({k: self.data[k] for k in names})

    def __getitem__(self, key):
        return self.data if isinstance(key, slice) else self.data[key]


class VideoIntegrityTest(unittest.TestCase):
    def test_metadata_error_and_explicit_clamp(self):
        meta = SimpleNamespace(root='/fixture', fps=15, video_keys=['cam'], episodes=Columns({
            'episode_index': [0, 1], 'length': [76, 80],
            'videos/cam/from_timestamp': [0., 5.], 'videos/cam/to_timestamp': [5., 5. + 80 / 15]}))
        with self.assertRaisesRegex(video.VideoIntegrityError, 'video/trajectory mismatch'):
            video.validate_video_segments(meta)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            video.validate_video_segments(meta, 'clamp')
            self.assertEqual(len(caught), 1)
        meta.episodes.data['videos/cam/to_timestamp'][0] = 76 / 15
        video.validate_video_segments(meta)

    def test_cross_episode_and_future_queries(self):
        ep = {'videos/cam/from_timestamp': 697.0666666666672,
              'videos/cam/to_timestamp': 702.0666666666672}
        with self.assertRaises(video.VideoIntegrityError):
            video.episode_video_timestamps(ep, 'cam', [75 / 15], 15)
        got = video.episode_video_timestamps(ep, 'cam', [0, 74 / 15, 75 / 15], 15, 'clamp')
        self.assertEqual([round(t * 15) for t in got], [10456, 10530, 10530])

    def test_raw_teacher_and_sam_override_use_episode_bounds(self):
        path = 'lingbotvla/data/vla_data/base_dataset.py'
        calls = []
        def decode(path, timestamps, *args):
            calls.append(timestamps)
            return torch.zeros(2, 3, 2, 2)
        ep = {'videos/raw/from_timestamp': 10., 'videos/raw/to_timestamp': 12.,
              'videos/variant/from_timestamp': 10., 'videos/variant/to_timestamp': 12.}
        meta = SimpleNamespace(root=Path('/fixture'), fps=15, episodes=[ep],
                               get_video_file_path=lambda *a: Path('video.mp4'))
        obj = SimpleNamespace(meta=meta, _raw_meta=meta, _sam_video_key='raw', fps=15,
                              video_integrity='clamp', tolerance_s=1e-4, video_backend='torchcodec',
                              root=Path('/fixture'))
        env = dict(LEROBOT_DATASET_API='v3', decode_video_frames=decode, Path=Path,
                   episode_video_timestamps=video.episode_video_timestamps)
        query = method(path, 'LeRobotDataset', '_query_videos', **env)
        raw = method(path, 'LeRobotDataset', '_query_raw_cam_high', **env)
        query(obj, {'raw': [0, 3]}, 0, {'raw': 'variant'})
        raw(obj, [0, 3], 0)
        self.assertEqual(calls, [[10., 179 / 15], [10., 179 / 15]])

    def test_integrity_error_not_randomly_retried(self):
        fn = method('lingbotvla/data/vla_data/multi_vla_dataset.py', 'MultiVLADataset', '__getitem__',
                    VideoIntegrityError=video.VideoIntegrityError, np=np)
        class Broken:
            calls = 0
            def __len__(self):
                return 2
            def getdata(self, idx):
                self.calls += 1
                raise video.VideoIntegrityError('bad frame')
        obj = Broken()
        with self.assertRaises(video.VideoIntegrityError):
            fn(obj, 0)
        self.assertEqual(obj.calls, 1)


if __name__ == '__main__':
    unittest.main()
