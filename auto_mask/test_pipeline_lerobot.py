"""Focused checks for multi-node episode selection and queue ownership."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from auto_mask.lerobot_mask_worker import label_areas
from auto_mask.mask_style import MaskStyle
from auto_mask.pipeline_lerobot import (Work, decode_mask_video, encode_mask_video, kept_mask, parse_shard,
                                        phrase_instructions, select_episodes)
from tools.sam3_mask_judge import judge_track, keep_timeline


class ShardTests(unittest.TestCase):
    def test_parse_shard(self):
        self.assertEqual(parse_shard("2/4"), (2, 4))
        for spec in ("", "0", "0/0", "2/2", "-1/2", "x/2", "1/2/3"):
            with self.subTest(spec=spec), self.assertRaises(SystemExit):
                parse_shard(spec)

    def test_select_episodes_uses_episode_id_after_subset(self):
        src = SimpleNamespace(eps=list(range(12)))
        args = SimpleNamespace(episodes="2-9", per_task=0, shard=(1, 3))
        self.assertEqual(select_episodes(args, src), [4, 7])
        args.shard = (0, 3)
        self.assertEqual(select_episodes(args, src), [3, 6, 9])

    def test_queues_are_disjoint_but_masks_are_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = Work(root, "single", "rank0of2")
            second = Work(root, "multi", "rank1of2")
            self.assertNotEqual(first.jobs, second.jobs)
            self.assertEqual(first.masks, second.masks)
            self.assertEqual(first.frames, second.frames)
            self.assertTrue(first.jobs.is_dir())
            self.assertTrue(second.jobs.is_dir())
            judged, raw = Work(root, "single"), Work(root, "single", variant="nojudge")
            self.assertEqual(judged.qwen, raw.qwen)
            self.assertNotEqual(judged.done, raw.done)


def fake_ask(answers):
    """ask(t) -> verdict from a {frame: keep} script; records the frames asked."""
    asked = []

    def ask(t):
        asked.append(t)
        return {"p_yes": 0.9 if answers[t] else 0.1, "keep": answers[t], "error": None}
    return ask, asked


class JudgeReplayTests(unittest.TestCase):
    def test_first_frame_accept_is_final(self):
        ask, asked = fake_ask({2: True})
        v = judge_track([0, 0, 0.01, 0.05, 0.2], ask)
        self.assertEqual(asked, [2])
        self.assertEqual(keep_timeline(v, 5).tolist(), [False, False, True, True, True])

    def test_regrow_rejudge(self):
        # rejected at area 0.01, re-judged on the first frame >= 2x (0.02), rejected again
        # (new reference 0.025), then accepted at >= 0.05
        areas = [0.01, 0.015, 0.025, 0.03, 0.0, 0.06, 0.07]
        ask, asked = fake_ask({0: False, 2: False, 5: True})
        v = judge_track(areas, ask, regrow=2.0, regrow_min_area=0.002)
        self.assertEqual(asked, [0, 2, 5])
        self.assertEqual(keep_timeline(v, 7).tolist(), [False] * 5 + [True, True])
        self.assertEqual([x["frame_index"] for x in v], [0, 2, 5])

    def test_min_area_floor_and_off(self):
        ask, asked = fake_ask({0: False, 2: False})
        judge_track([0.0001, 0.0003, 0.004], ask, regrow=2.0, regrow_min_area=0.002)
        self.assertEqual(asked, [0, 2])  # 0.0003 >= 2*0.0001 but below 2*floor
        ask, asked = fake_ask({0: False})
        judge_track([0.01, 0.5], ask, regrow=0)
        self.assertEqual(asked, [0])

    def test_never_shown(self):
        ask, asked = fake_ask({})
        self.assertEqual(judge_track([0, 0], ask), [])
        self.assertEqual(keep_timeline([], 2).tolist(), [False, False])


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((3, 6, 8), np.uint8)
        self.labels[:, 1:3, 1:3] = 1   # object id 7
        self.labels[1:, 4:6, 5:8] = 2  # object id 9, from frame 1

    def test_label_areas(self):
        self.assertEqual(label_areas(self.labels, 2), [[4, 0], [4, 6], [4, 6]])

    def test_kept_mask(self):
        ids = [7, 9]
        self.assertTrue((kept_mask(self.labels, ids, None) == (self.labels != 0)).all())
        verdicts = {"7": [{"frame_index": 0, "keep": False}, {"frame_index": 2, "keep": True}],
                    "9": [{"frame_index": 1, "keep": True}]}
        m = kept_mask(self.labels, ids, verdicts)
        self.assertEqual(m[0].sum(), 0)
        self.assertTrue((m[1] == (self.labels[1] == 2)).all())
        self.assertTrue((m[2] == (self.labels[2] != 0)).all())

    def test_phrase_instructions_first_slot(self):
        q = {"instructions": ["a", "b", "c"], "phrases": [["x"], ["x", "y"], []]}
        self.assertEqual(phrase_instructions(q), {"x": "a", "y": "b"})

    def test_mask_video_roundtrip(self):
        rng = np.random.default_rng(0)
        keep = rng.random((5, 24, 32)) > 0.6
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.mkv"
            encode_mask_video(keep, path, 15)
            self.assertTrue((decode_mask_video(path) == keep).all())

    def test_styles(self):
        frames = np.full((1, 9, 9, 3), 200, np.uint8)
        keep = np.zeros((1, 9, 9), bool)
        keep[0, 3:6, 3:6] = True
        dim = MaskStyle("dim", 0.2).render(frames, keep)
        self.assertTrue((dim[keep] == 200).all() and (dim[~keep] == 40).all())
        st = MaskStyle("outline", 0.1, 2)
        out = st.render(frames, keep)
        ring = st.ring(keep)
        self.assertFalse((ring & keep).any())
        self.assertTrue((out[keep] == 200).all())       # mask pixels untouched
        self.assertTrue((out[ring] == 255).all())       # white ring outside the mask
        self.assertTrue((out[~keep & ~ring] == 20).all())
        self.assertEqual(int(ring[0, 3, 1:3].sum()), 2)  # 2 px wide on a straight edge
        self.assertEqual(MaskStyle.from_dict(st.to_dict()), st)
        self.assertEqual(st.tag(), "outline0.1_w2")


if __name__ == "__main__":
    unittest.main()
