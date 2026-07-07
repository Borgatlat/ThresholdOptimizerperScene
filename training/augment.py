"""SpecAugment-style augmentation for spectrogram Ki training (train only)."""

from __future__ import annotations

import torch


def _mask_axis(x: torch.Tensor, axis: int, max_width: int) -> torch.Tensor:
    """Zero out a random contiguous band along freq (axis=-2) or time (axis=-1)."""
    if max_width <= 0:
        return x
    size = x.shape[axis]
    if size <= 1:
        return x
    width = torch.randint(1, min(max_width, size) + 1, (1,)).item()
    start = torch.randint(0, size - width + 1, (1,)).item()
    out = x.clone()
    slc = [slice(None)] * out.dim()
    slc[axis] = slice(start, start + width)
    out[tuple(slc)] = 0.0
    return out


def apply_spec_augment(
    x: torch.Tensor,
    freq_mask_max: int = 12,
    time_mask_max: int = 6,
    noise_std: float = 0.05,
) -> torch.Tensor:
    """Light SpecAugment + noise; helps generalize long-train → short-val shift."""
    out = x
    if freq_mask_max > 0:
        out = _mask_axis(out, axis=-2, max_width=freq_mask_max)
    if time_mask_max > 0:
        out = _mask_axis(out, axis=-1, max_width=time_mask_max)
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out
