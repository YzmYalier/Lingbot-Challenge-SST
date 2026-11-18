"""Tests for lingbotvla/optim/module_lr.py (CPU only, single process).

    python -m unittest tests.test_module_lr -v
"""

import copy
import json
import os
import struct
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from lingbotvla.optim.lr_scheduler import build_lr_scheduler
from lingbotvla.optim.module_lr import (
    MODULE_NAMES,
    classify_param_name,
    freeze_zero_lr_modules,
    module_lr_snapshot,
    resolve_module_lr_mult,
)
from lingbotvla.optim.optimizer import build_muon_optimizer


from submit_paths import path

CKPT_DIR = os.environ.get("LINGBOT_CKPT_DIR", str(path("vla_pretrained")))

EXPECTED = {
    "model.qwenvl_with_expert.qwenvl.model.visual.blocks.3.attn.qkv.weight": "vit",
    "model.qwenvl_with_expert.qwenvl.model.visual.patch_embed.proj.weight": "vit",
    "model.qwenvl_with_expert.qwenvl.model.visual.pos_embed.weight": "vit",
    "model.qwenvl_with_expert.qwenvl.model.visual.merger.linear_fc1.weight": "vit_merger",
    "model.qwenvl_with_expert.qwenvl.model.visual.deepstack_merger_list.2.norm.bias": "vit_merger",
    "model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.weight": "llm_embed",
    "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.mlp.down_proj.weight": "llm",
    "model.qwenvl_with_expert.qwenvl.model.language_model.norm.weight": "llm",
    "model.qwenvl_with_expert.qwen_expert.model.layers.5.mlp.experts.up_proj": "action_expert",
    "model.qwenvl_with_expert.qwen_expert.model.layers.5.mlp.gate.weight": "action_expert",
    "model.qwenvl_with_expert.qwen_expert.model.layers.5.input_layernorm.gamma.weight": "action_expert",
    "model.qwenvl_with_expert.qwen_expert.model.norm.weight": "action_expert",
    "model.state_proj.weight": "action_io",
    "model.action_out_proj.bias": "action_io",
    "model.action_time_mlp_in.weight": "action_io",
    "model.depth_align_embs": "align",
    "model.future_video_align_head.projector.layers.0.attn.to_q.weight": "align",
    "model.current_shared_task_proj.weight": "align",
    # torch.compile prefix must not change the result.
    "_orig_mod.model.qwenvl_with_expert.qwenvl.model.visual.blocks.0.norm1.weight": "vit",
}


def _checkpoint_keys():
    index = os.path.join(CKPT_DIR, "model.safetensors.index.json")
    if not os.path.exists(index):
        return None
    with open(index) as fh:
        weight_map = json.load(fh)["weight_map"]
    keys = []
    for shard in sorted(set(weight_map.values())):
        with open(os.path.join(CKPT_DIR, shard), "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        keys.extend(k for k in header if k != "__metadata__")
    return keys


def _toy_policy():
    """Same FQN layout as LingbotVlaV2Policy, tiny shapes."""

    def lin(i, o, bias=True):
        return nn.Linear(i, o, bias=bias)

    visual = nn.Module()
    visual.blocks = nn.ModuleList([nn.ModuleDict({"qkv": lin(8, 24), "norm1": nn.LayerNorm(8)})])
    visual.merger = nn.ModuleDict({"linear_fc1": lin(8, 16)})
    lm = nn.Module()
    lm.embed_tokens = nn.Embedding(32, 16)
    lm.layers = nn.ModuleList([nn.ModuleDict({"q_proj": lin(16, 16, bias=False)})])
    lm.norm = nn.LayerNorm(16)
    vl_model = nn.Module()
    vl_model.visual = visual
    vl_model.language_model = lm
    qwenvl = nn.Module()
    qwenvl.model = vl_model

    layer = nn.Module()
    layer.self_attn = nn.ModuleDict({"q_proj": lin(12, 12)})
    mlp = nn.Module()
    mlp.experts = nn.Module()
    mlp.experts.up_proj = nn.Parameter(torch.randn(4, 6, 12))
    mlp.gate = lin(12, 4, bias=False)
    layer.mlp = mlp
    expert_model = nn.Module()
    expert_model.layers = nn.ModuleList([layer])
    qwen_expert = nn.Module()
    qwen_expert.model = expert_model

    qwe = nn.Module()
    qwe.qwenvl = qwenvl
    qwe.qwen_expert = qwen_expert
    flow = nn.Module()
    flow.qwenvl_with_expert = qwe
    flow.state_proj = lin(5, 12)
    flow.action_out_proj = lin(12, 5)
    flow.depth_align_embs = nn.Parameter(torch.randn(8, 16))
    flow.depth_align_head = nn.ModuleDict({"proj_in1": lin(16, 16)})
    policy = nn.Module()
    policy.model = flow
    return policy


def _args(module_lr_mult, use_expert_lr=True):
    return SimpleNamespace(
        use_moe=True,
        use_moe_expert_lr=use_expert_lr,
        token_moe_layers=[0],
        token_num_experts=32,
        token_top_k=4,
        module_lr_mult=module_lr_mult,
        muon_exclude_name_patterns=None,
    )


def _step_delta(model, args, lr=1e-3):
    # Zero weights make `after - before` exact, so update ratios can be compared tightly.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_muon_optimizer(model, args, lr=lr)
    opt.step()
    return {n: p.detach() - before[n] for n, p in model.named_parameters()}, opt


class ClassifyTest(unittest.TestCase):
    def test_expected_names(self):
        for name, module in EXPECTED.items():
            self.assertEqual(classify_param_name(name), module, name)

    def test_every_checkpoint_key_is_classified(self):
        keys = _checkpoint_keys()
        if keys is None:
            self.skipTest(f"no checkpoint at {CKPT_DIR}")
        unmatched = [k for k in keys if classify_param_name(k) is None]
        self.assertEqual(unmatched, [])
        seen = {classify_param_name(k) for k in keys}
        self.assertEqual(seen, set(MODULE_NAMES))


class ResolveTest(unittest.TestCase):
    def test_off_when_empty(self):
        self.assertIsNone(resolve_module_lr_mult({}))
        self.assertIsNone(resolve_module_lr_mult(None))

    def test_defaults_and_validation(self):
        mult = resolve_module_lr_mult({"vit": 0.3})
        self.assertEqual(mult["vit"], 0.3)
        self.assertEqual(mult["llm"], 1.0)
        with self.assertRaises(ValueError):
            resolve_module_lr_mult({"vision": 0.3})
        with self.assertRaises(ValueError):
            resolve_module_lr_mult({"vit": -1})


class OptimizerTest(unittest.TestCase):
    def test_disabled_keeps_original_groups(self):
        _, opt = _step_delta(_toy_policy(), _args({}))
        for group in opt.param_groups:
            self.assertNotIn("module", group)
        lrs = sorted(g["lr"] for g in opt.param_groups)
        self.assertAlmostEqual(lrs[-1] / lrs[0], (32 / 4) ** 0.5)

    def test_group_lrs(self):
        mult = {"vit": 0.25, "vit_merger": 0.5, "llm": 0.5, "llm_embed": 0.1, "action_io": 2.0}
        _, opt = _step_delta(_toy_policy(), _args(mult), lr=1e-3)
        base, per_module = module_lr_snapshot(opt)
        self.assertAlmostEqual(base, 1e-3)
        expected = {
            "vit": 2.5e-4,
            "vit_merger": 5e-4,
            "llm": 5e-4,
            "llm_embed": 1e-4,
            "action_expert": 1e-3,
            "action_expert_routed": 1e-3 * (32 / 4) ** 0.5,
            "action_io": 2e-3,
            "align": 1e-3,
        }
        self.assertEqual(set(per_module), set(expected))
        for module, lr in expected.items():
            self.assertAlmostEqual(per_module[module], lr, msg=module)

    def test_update_scales_with_multiplier(self):
        base_model = _toy_policy()
        d_ref, _ = _step_delta(copy.deepcopy(base_model), _args({"vit": 1.0}))
        d_low, _ = _step_delta(copy.deepcopy(base_model), _args({"vit": 0.25}))
        self.assertTrue(all(d.abs().sum() > 0 for d in d_ref.values()))
        for name in d_ref:
            ratio = 0.25 if classify_param_name(name) == "vit" else 1.0
            torch.testing.assert_close(d_low[name], d_ref[name] * ratio, msg=name)

    def test_scheduler_keeps_ratios(self):
        mult = {"vit": 0.1, "llm": 0.5}
        _, opt = _step_delta(_toy_policy(), _args(mult), lr=1e-4)
        sched = build_lr_scheduler(
            opt, train_steps=100, lr=1e-4, lr_min=1e-6, lr_decay_style="cosine", lr_warmup_ratio=0.1
        )
        for step in range(100):
            if step in (5, 50, 99):
                base, per_module = module_lr_snapshot(opt)
                self.assertAlmostEqual(per_module["vit"], base * 0.1, delta=1e-15)
                self.assertAlmostEqual(per_module["llm"], base * 0.5, delta=1e-15)
            opt.step()
            sched.step()
        base, per_module = module_lr_snapshot(opt)
        self.assertAlmostEqual(base, 1e-6, delta=1e-12)
        self.assertAlmostEqual(per_module["vit"], 1e-7, delta=1e-13)

    def test_zero_freezes(self):
        model = _toy_policy()
        mult = resolve_module_lr_mult({"llm_embed": 0.0})
        frozen = freeze_zero_lr_modules(model, mult)
        self.assertEqual(frozen, {"llm_embed": 32 * 16})
        embed = model.model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.weight
        self.assertFalse(embed.requires_grad)
        _, opt = _step_delta(model, _args({"llm_embed": 0.0}))
        self.assertNotIn("llm_embed", module_lr_snapshot(opt)[1])

    def test_zero_without_freeze_raises(self):
        with self.assertRaises(ValueError):
            _step_delta(_toy_policy(), _args({"llm_embed": 0.0}))

    def test_unmatched_param_raises(self):
        model = _toy_policy()
        model.model.mystery = nn.Linear(3, 3)
        with self.assertRaises(ValueError):
            _step_delta(model, _args({"vit": 0.5}))


if __name__ == "__main__":
    unittest.main()
