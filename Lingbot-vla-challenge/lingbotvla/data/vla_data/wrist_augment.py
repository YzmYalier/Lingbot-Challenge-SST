"""Photometric augmentation for the wrist cameras (data.wrist_augment).

The R_REC preset defines colour and lighting operations for the wrist views:
channel gain, tint, brightness, contrast, saturation, gamma, light fields and noise.
It does not replace textures, copy image regions, rotate hue, blur images or convert
them to grayscale.

Sampling: one draw of the scene light (channel gain, tint, brightness, contrast,
saturation, gamma) per sample, shared by every augmented view. Each view then gets
a small brightness / gain jitter, its own light field and its own noise. The current
and future frames of a sample reuse the same parameters.
"""
import math

import torch
from torch import Tensor

# RGB -> luma weights.
_LUMA = (0.299, 0.587, 0.114)

R_REC = {
    "brightness": (0.30, 1.00),   # log-uniform
    "contrast": (0.70, 1.40),     # log-uniform
    "saturation": (0.80, 2.00),   # uniform
    "gain_spread": 0.15,          # per-channel gain exp(U(-s, s))
    "tint": 0.10,                 # blend toward a random colour, alpha ~ U(0, tint)
    "gamma": (0.80, 1.50),        # log-uniform
    "lightfield": 0.50,           # strength ~ U(0, lightfield)
    "noise": 0.015,               # gaussian std ~ U(0, noise)
    "p_apply": 0.90,
    # per-view jitter on top of the shared scene draw
    "view_brightness_jitter": 0.10,  # exp(U(-j, j))
    "view_gain_jitter": 0.03,        # exp(U(-j, j)) per channel
}

DEFAULT_WRIST_CAMERAS = ("camera_wrist_left", "camera_wrist_right")


def _log_uniform(lo: float, hi: float, gen=None) -> float:
    u = torch.rand((), generator=gen).item()
    return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def _uniform(lo: float, hi: float, shape=(), gen=None) -> Tensor:
    return lo + torch.rand(shape, generator=gen) * (hi - lo)


def sample_wrist_augment_params(view_keys, spec=R_REC, gen=None) -> dict:
    """Draw one replayable parameter set for a sample.

    Returns {"apply": bool, "views": {key: per-view params}}. A view's params hold
    everything needed to replay the transform on another frame (lightfield geometry and
    noise std included); only the noise realisation is redrawn per frame.
    """
    apply = bool(torch.rand((), generator=gen).item() < spec["p_apply"])
    if not apply:
        return {"apply": False, "views": {}}

    s = spec["gain_spread"]
    scene = {
        "gain": torch.exp(_uniform(-s, s, (3,), gen)),
        "tint_color": torch.rand(3, generator=gen),
        "tint_alpha": float(_uniform(0.0, spec["tint"], (), gen)),
        "brightness": _log_uniform(*spec["brightness"], gen=gen),
        "contrast": _log_uniform(*spec["contrast"], gen=gen),
        "saturation": float(_uniform(*spec["saturation"], (), gen)),
        "gamma": _log_uniform(*spec["gamma"], gen=gen),
    }
    views = {}
    jb, jg = spec["view_brightness_jitter"], spec["view_gain_jitter"]
    for key in view_keys:
        views[key] = dict(
            scene,
            gain=scene["gain"] * torch.exp(_uniform(-jg, jg, (3,), gen)),
            brightness=scene["brightness"] * math.exp(float(_uniform(-jb, jb, (), gen))),
            lightfield=float(_uniform(0.0, spec["lightfield"], (), gen)),
            lf_ramp=bool(torch.rand((), generator=gen).item() < 0.5),
            lf_angle=float(_uniform(0.0, 2 * math.pi, (), gen)),
            lf_center=_uniform(-0.25, 0.25, (2,), gen),
            noise=float(_uniform(0.0, spec["noise"], (), gen)),
        )
    return {"apply": True, "views": views}


def _light_field(h: int, w: int, p: dict) -> Tensor:
    """Smooth illumination map in [0, 1]: a linear ramp or an off-centre vignette."""
    yy = torch.linspace(-0.5, 0.5, h).view(h, 1)
    xx = torch.linspace(-0.5, 0.5, w).view(1, w)
    if p["lf_ramp"]:
        return math.cos(p["lf_angle"]) * xx + math.sin(p["lf_angle"]) * yy + 0.5
    cy, cx = float(p["lf_center"][0]), float(p["lf_center"][1])
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.sqrt(0.5)
    return 1 - r.clamp(0, 1) ** 2


def apply_wrist_augment(img: Tensor, p: dict, gen=None) -> Tensor:
    """Apply one view's params to an image (C,H,W) RGB. Keeps shape and dtype."""
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"wrist augment expects a (3,H,W) RGB image, got {tuple(img.shape)}")
    orig_dtype = img.dtype
    x = img.to(torch.float32)
    if orig_dtype == torch.uint8 or x.max() > 1.0:
        x = x / 255.0
    luma = torch.tensor(_LUMA).view(3, 1, 1)

    x = x * p["gain"].view(3, 1, 1)
    a = p["tint_alpha"]
    x = x * (1 - a) + p["tint_color"].view(3, 1, 1) * a
    x = x * p["brightness"]
    mean = (x * luma).sum(0).mean()
    x = (x - mean) * p["contrast"] + mean
    gray = (x * luma).sum(0, keepdim=True)
    x = gray + (x - gray) * p["saturation"]
    x = x.clamp(0, 1) ** p["gamma"]
    if p["lightfield"] > 0:
        field = _light_field(x.shape[1], x.shape[2], p)
        x = x * (1 - p["lightfield"] * (1 - field))
    if p["noise"] > 0:
        x = x + torch.randn(x.shape, generator=gen) * p["noise"]
    x = x.clamp(0, 1)

    if orig_dtype == torch.uint8:
        return (x * 255.0).round().to(torch.uint8)
    if img.is_floating_point() and img.max() > 1.0:
        return (x * 255.0).to(orig_dtype)
    return x.to(orig_dtype)


def augment_wrist_views(image_dicts, view_keys, spec=R_REC, gen=None):
    """Augment `view_keys` in every dict of `image_dicts` (current, future, ...) in place,
    with one shared parameter draw. Keys missing from a dict are skipped; other views
    (cam_high) are never touched. Returns the params for logging / tests."""
    present = [k for k in view_keys if any(k in d for d in image_dicts)]
    params = sample_wrist_augment_params(present, spec, gen)
    if not params["apply"]:
        return params
    for d in image_dicts:
        for key in present:
            if key in d:
                d[key] = apply_wrist_augment(d[key], params["views"][key], gen)
    return params
