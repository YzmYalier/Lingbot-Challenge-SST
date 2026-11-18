"""Reduction of action losses over real timesteps and real joints."""

import torch


def masked_action_loss(losses, joint_mask, action_is_pad, action_dim, *, repeat=False):
    """Return global and per-example means; padding contributes to neither sum nor count.

    The legacy reduction remains in the policies when mask_action_pad=False.
    This only masks the supervision, not the action tokens fed to the transformer.
    """
    if losses.ndim != 3:
        raise ValueError(f"Expected losses (B,T,D), got {tuple(losses.shape)}")
    if joint_mask is None:
        if not 0 < action_dim <= losses.shape[-1]:
            raise ValueError(f"Invalid action_dim={action_dim} for {tuple(losses.shape)}")
        losses = losses[..., :action_dim]
        valid = torch.ones_like(losses, dtype=torch.bool)
    else:
        valid = joint_mask.to(device=losses.device, dtype=torch.bool)
        if valid.ndim == 2:
            valid = valid.unsqueeze(1)
        if valid.ndim != 3:
            raise ValueError("joint_mask must have shape (B,D) or (B,T,D)")
        if repeat and valid.shape[0] * 2 == losses.shape[0]:
            valid = valid.repeat(2, 1, 1)
        if valid.shape[0] != losses.shape[0] or valid.shape[-1] != losses.shape[-1]:
            raise ValueError("joint_mask batch/action dimensions do not match losses")
        if valid.shape[1] not in (1, losses.shape[1]):
            raise ValueError("joint_mask time dimension does not match losses")
        valid = valid.expand_as(losses)
    if action_is_pad is not None:
        pad = action_is_pad.to(device=losses.device, dtype=torch.bool)
        if pad.ndim != 2:
            raise ValueError("action_is_pad must have shape (B,T)")
        if repeat and pad.shape[0] * 2 == losses.shape[0]:
            pad = pad.repeat(2, 1)
        if pad.shape != losses.shape[:2]:
            raise ValueError("action_is_pad shape does not match losses (B,T)")
        valid = valid & ~pad.unsqueeze(-1)
    # Accumulate in fp32 under mixed precision; retain float64 for numerical tests.
    dtype = torch.float32 if losses.dtype in (torch.float16, torch.bfloat16) else losses.dtype
    sums = torch.where(valid, losses, 0).sum(dim=(1, 2), dtype=dtype)
    counts = valid.sum(dim=(1, 2))
    return sums.sum() / counts.sum().clamp(min=1), sums / counts.clamp(min=1)
