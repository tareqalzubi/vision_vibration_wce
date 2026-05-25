from __future__ import annotations

import torch
import torch.nn.functional as F


def make_pixel_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    pix = torch.stack([x, y, ones], dim=0).view(1, 3, -1).repeat(batch, 1, 1)
    return pix


def projective_warp(
    source: torch.Tensor,
    depth_ref: torch.Tensor,
    pose_ref_to_source: torch.Tensor,
    intrinsics: torch.Tensor,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp source image into reference coordinates using depth and ref->source pose."""
    b, _, h, w = source.shape
    device, dtype = source.device, source.dtype
    pix = make_pixel_grid(b, h, w, device, dtype)
    k = intrinsics.to(device=device, dtype=dtype)
    k_inv = torch.inverse(k)
    cam = k_inv @ pix
    cam = cam * depth_ref.view(b, 1, -1)
    homo = torch.cat([cam, torch.ones(b, 1, h * w, device=device, dtype=dtype)], dim=1)
    src_cam = pose_ref_to_source.to(device=device, dtype=dtype) @ homo
    xyz = src_cam[:, :3]
    z = xyz[:, 2:3].clamp_min(1e-6)
    proj = k @ (xyz / z)
    x = proj[:, 0].view(b, h, w)
    y = proj[:, 1].view(b, h, w)
    x_norm = 2.0 * (x / max(w - 1, 1)) - 1.0
    y_norm = 2.0 * (y / max(h - 1, 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    valid = ((x_norm > -1.0) & (x_norm < 1.0) & (y_norm > -1.0) & (y_norm < 1.0) & (z.view(b, h, w) > 0)).float().unsqueeze(1)
    projected_depth = z.view(b, 1, h, w)
    return warped, valid, projected_depth
