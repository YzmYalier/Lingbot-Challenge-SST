"""Unit tests for lingbotvla/data/vla_data/wrist_augment.py.

Run: python -m unittest tests.test_wrist_augment
"""
import unittest

import torch

from lingbotvla.data.vla_data import wrist_augment as wa

LEFT = "observation.images.camera_wrist_left"
RIGHT = "observation.images.camera_wrist_right"
TOP = "observation.images.camera_top"


def _img(seed=0, h=48, w=64):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(3, h, w, generator=g) * 255).round().to(torch.uint8)


def _always(spec=wa.R_REC, **kw):
    out = dict(spec, p_apply=1.0)
    out.update(kw)
    return out


class WristAugmentTest(unittest.TestCase):
    def test_cam_high_untouched_and_wrist_changed(self):
        top = _img(1)
        cur = {TOP: top.clone(), LEFT: _img(2), RIGHT: _img(3)}
        before = {k: v.clone() for k, v in cur.items()}
        wa.augment_wrist_views([cur, {}], [LEFT, RIGHT], _always(), torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(cur[TOP], before[TOP]))
        self.assertFalse(torch.equal(cur[LEFT], before[LEFT]))
        self.assertFalse(torch.equal(cur[RIGHT], before[RIGHT]))

    def test_shape_dtype_range_preserved(self):
        for dtype in (torch.uint8, torch.float32):
            img = _img(4) if dtype == torch.uint8 else _img(4).float() / 255
            p = wa.sample_wrist_augment_params([LEFT], _always(), torch.Generator().manual_seed(1))
            out = wa.apply_wrist_augment(img, p["views"][LEFT])
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, img.dtype)
            hi = 255 if dtype == torch.uint8 else 1.0
            self.assertGreaterEqual(float(out.min()), 0)
            self.assertLessEqual(float(out.max()), hi)

    def test_current_and_future_share_params(self):
        # Without noise the transform is deterministic given params: the same input
        # frame in current and future must come out identical.
        spec = _always(noise=0.0)
        frame = _img(5)
        cur, fut = {LEFT: frame.clone()}, {LEFT: frame.clone()}
        wa.augment_wrist_views([cur, fut], [LEFT], spec, torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(cur[LEFT], fut[LEFT]))

    def test_views_share_scene_light(self):
        p = wa.sample_wrist_augment_params([LEFT, RIGHT], _always(), torch.Generator().manual_seed(3))
        l, r = p["views"][LEFT], p["views"][RIGHT]
        for k in ("contrast", "saturation", "gamma", "tint_alpha"):
            self.assertEqual(l[k], r[k])
        self.assertTrue(torch.equal(l["tint_color"], r["tint_color"]))
        # per-view jitter stays within the configured bounds
        jb, jg = wa.R_REC["view_brightness_jitter"], wa.R_REC["view_gain_jitter"]
        ratio = l["brightness"] / r["brightness"]
        self.assertLessEqual(abs(torch.log(torch.tensor(ratio))), 2 * jb + 1e-6)
        self.assertTrue(bool((torch.log(l["gain"] / r["gain"]).abs() <= 2 * jg + 1e-6).all()))

    def test_p_apply_zero_is_identity(self):
        cur = {LEFT: _img(6), RIGHT: _img(7)}
        before = {k: v.clone() for k, v in cur.items()}
        p = wa.augment_wrist_views([cur], [LEFT, RIGHT], dict(wa.R_REC, p_apply=0.0))
        self.assertFalse(p["apply"])
        for k in cur:
            self.assertTrue(torch.equal(cur[k], before[k]))

    def test_p_apply_rate(self):
        g = torch.Generator().manual_seed(4)
        n = 4000
        hits = sum(wa.sample_wrist_augment_params([LEFT], wa.R_REC, g)["apply"] for _ in range(n))
        self.assertAlmostEqual(hits / n, wa.R_REC["p_apply"], delta=0.02)

    def test_param_ranges(self):
        g = torch.Generator().manual_seed(5)
        s = wa.R_REC
        jb = s["view_brightness_jitter"]
        for _ in range(500):
            v = wa.sample_wrist_augment_params([LEFT], _always(), g)["views"][LEFT]
            b_lo, b_hi = s["brightness"]
            self.assertTrue(b_lo * torch.exp(torch.tensor(-jb)) - 1e-6 <= v["brightness"]
                            <= b_hi * torch.exp(torch.tensor(jb)) + 1e-6)
            self.assertTrue(s["contrast"][0] - 1e-6 <= v["contrast"] <= s["contrast"][1] + 1e-6)
            self.assertTrue(s["saturation"][0] - 1e-6 <= v["saturation"] <= s["saturation"][1] + 1e-6)
            self.assertTrue(s["gamma"][0] - 1e-6 <= v["gamma"] <= s["gamma"][1] + 1e-6)
            self.assertTrue(0 <= v["tint_alpha"] <= s["tint"])
            self.assertTrue(0 <= v["lightfield"] <= s["lightfield"])
            self.assertTrue(0 <= v["noise"] <= s["noise"])

    def test_missing_view_skipped(self):
        cur = {LEFT: _img(8)}
        wa.augment_wrist_views([cur, {}], [LEFT, RIGHT], _always(), torch.Generator().manual_seed(6))
        self.assertEqual(set(cur), {LEFT})

    def test_darkens_bright_clean_frame_on_average(self):
        # A near-white clean wrist frame should come out darker on average (calibration
        # target: luminance 0.81 -> ~0.43).
        white = torch.full((3, 48, 64), 230, dtype=torch.uint8)
        g = torch.Generator().manual_seed(7)
        lum = []
        for _ in range(300):
            d = {LEFT: white.clone()}
            wa.augment_wrist_views([d], [LEFT], wa.R_REC, g)
            lum.append(d[LEFT].float().mean() / 255)
        self.assertLess(float(torch.stack(lum).mean()), 0.75)


if __name__ == "__main__":
    unittest.main()
