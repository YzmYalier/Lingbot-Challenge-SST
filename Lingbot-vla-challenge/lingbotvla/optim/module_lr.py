"""Per-module learning-rate multipliers for LingBot-VLA v2.

``train.module_lr_mult`` maps a module name to a multiplier on ``train.lr``:

    vit           Qwen3-VL ViT blocks, patch_embed, pos_embed
    vit_merger    ViT -> LLM bridge (merger, deepstack_merger_list)
    llm           Qwen3-VL language model layers and final norm
    llm_embed     language_model.embed_tokens
    action_expert qwen_expert (attention, adaLN, router, shared + routed experts)
    action_io     state_proj, action_in/out_proj, action_time_mlp_*
    align         depth / video align heads, task projections, align query embs

Missing keys default to 1.0. A multiplier of 0 freezes the module
(``requires_grad=False``) instead of keeping a zero-LR optimizer state.

Each param group carries ``module`` (label) and ``lr_mult`` (module multiplier
times the routed-expert scale, if any), so ``lr = schedule(step) * train.lr *
lr_mult`` holds for every group. ``LambdaLR`` scales all groups by the same
factor, so the ratios also hold for ``lr_min`` and the warmup.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


MODULE_NAMES: Tuple[str, ...] = (
    "vit",
    "vit_merger",
    "llm",
    "llm_embed",
    "action_expert",
    "action_io",
    "align",
)

ROUTED_SUFFIX = "_routed"

# First match wins, so the bridge is tested before the generic `.visual.` rule.
_MODULE_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("vit_merger", re.compile(r"\.visual\.(merger|deepstack_merger_list)\.")),
    ("vit", re.compile(r"\.visual\.")),
    ("llm_embed", re.compile(r"\.language_model\.embed_tokens\.")),
    ("llm", re.compile(r"\.language_model\.")),
    ("action_expert", re.compile(r"\.qwen_expert\.")),
    ("action_io", re.compile(r"\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)\.")),
    ("align", re.compile(r"(_align_head\.|_align_embs$|_shared_task_proj\.)")),
)


def classify_param_name(name: str) -> Optional[str]:
    """Return the module of a parameter FQN, or None if no rule matches."""
    for module, pattern in _MODULE_PATTERNS:
        if pattern.search(name):
            return module
    return None


def resolve_module_lr_mult(module_lr_mult: Optional[Mapping[str, Any]]) -> Optional[Dict[str, float]]:
    """Validate ``train.module_lr_mult``; return None when the feature is off."""
    if not module_lr_mult:
        return None
    unknown = sorted(set(module_lr_mult) - set(MODULE_NAMES))
    if unknown:
        raise ValueError(f"train.module_lr_mult has unknown modules {unknown}; valid: {list(MODULE_NAMES)}")
    resolved = {}
    for module in MODULE_NAMES:
        value = float(module_lr_mult.get(module, 1.0))
        if value < 0:
            raise ValueError(f"train.module_lr_mult.{module} must be >= 0, got {value}")
        resolved[module] = value
    return resolved


def _classify_or_raise(names: Sequence[str]) -> List[str]:
    modules = [classify_param_name(n) for n in names]
    unmatched = [n for n, m in zip(names, modules) if m is None]
    if unmatched:
        preview = ", ".join(unmatched[:10])
        raise ValueError(
            f"train.module_lr_mult: {len(unmatched)} trainable parameters match no module "
            f"(first: {preview}). Extend _MODULE_PATTERNS in lingbotvla/optim/module_lr.py."
        )
    return modules


def freeze_zero_lr_modules(model: nn.Module, module_lr_mult: Dict[str, float]) -> Dict[str, int]:
    """Set requires_grad=False on modules whose multiplier is 0.

    Must run before FSDP wrapping. Returns {module: frozen param count}.
    """
    zero = {m for m, v in module_lr_mult.items() if v == 0.0}
    frozen: Dict[str, int] = {}
    if not zero:
        return frozen
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    modules = _classify_or_raise([n for n, _ in trainable])
    for (_, p), module in zip(trainable, modules):
        if module in zero:
            p.requires_grad_(False)
            frozen[module] = frozen.get(module, 0) + p.numel()
    return frozen


def build_module_param_groups(
    params_and_names: Sequence[Tuple[torch.Tensor, str]],
    base_lr: float,
    module_lr_mult: Dict[str, float],
    layer_to_scale: Dict[int, float],
    layer_re: "re.Pattern[str]",
) -> List[Dict[str, Any]]:
    """Bucket (param, name) pairs by module (and routed-expert scale)."""
    modules = _classify_or_raise([n for _, n in params_and_names])
    buckets: Dict[Tuple[str, float], List[torch.Tensor]] = {}
    for (p, name), module in zip(params_and_names, modules):
        mult = module_lr_mult[module]
        if mult == 0.0:
            raise ValueError(
                f"{name} belongs to module {module!r} with lr multiplier 0 but still requires grad; "
                "call freeze_zero_lr_modules before building the optimizer."
            )
        label = module
        m = layer_re.search(name)
        if m is not None:
            scale = layer_to_scale.get(int(m.group(1)))
            if scale is not None:
                mult = mult * scale
                label = module + ROUTED_SUFFIX
        buckets.setdefault((label, mult), []).append(p)
    return [
        {"params": ps, "lr": base_lr * mult, "module": label, "lr_mult": mult}
        for (label, mult), ps in buckets.items()
    ]


def module_lr_snapshot(optimizer: torch.optim.Optimizer) -> Tuple[Optional[float], Dict[str, float]]:
    """Return (base lr, {module label: current lr}) from tagged param groups."""
    base_lr = None
    per_module: Dict[str, float] = {}
    for group in optimizer.param_groups:
        label = group.get("module")
        if label is None:
            continue
        lr = float(group["lr"])
        per_module[label] = lr
        if base_lr is None and group["lr_mult"] > 0:
            base_lr = lr / group["lr_mult"]
    return base_lr, per_module


def format_module_lr_table(optimizer: torch.optim.Optimizer) -> str:
    """One line per (optimizer, module) group: lr, multiplier, tensor and element counts."""
    inner = getattr(optimizer, "optimizers", [optimizer])
    lines = []
    for opt in inner:
        kind = type(opt).__name__
        for group in opt.param_groups:
            if "module" not in group:
                continue
            numel = sum(p.numel() for p in group["params"])
            lines.append(
                f"  {kind:<16} {group['module']:<22} lr={group['lr']:.3e} "
                f"mult={group['lr_mult']:.4g} tensors={len(group['params'])} params={numel / 1e6:.2f}M"
            )
    return "\n".join(lines)
