from __future__ import annotations

import torch

from vv_wce.losses.self_supervised import ssim


def photometric_error(warped: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    err = (warped - target).abs() * mask
    return float((err.sum() / (mask.sum() * target.shape[1]).clamp_min(1.0)).item())


def warp_ssim(warped: torch.Tensor, target: torch.Tensor) -> float:
    return float(ssim(warped, target).mean().item())
