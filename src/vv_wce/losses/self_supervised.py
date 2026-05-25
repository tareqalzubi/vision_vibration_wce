from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from vv_wce.geometry.pose import pose_vec_to_mat
from vv_wce.geometry.warping import projective_warp


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-7)
    return score.clamp(0, 1)


def affine_brightness_match(src: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(src[:, :1])
    m = mask.expand_as(src)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    src_mean = (src * m).sum(dim=(2, 3), keepdim=True) / denom
    tgt_mean = (tgt * m).sum(dim=(2, 3), keepdim=True) / denom
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    var = ((src_c ** 2) * m).sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    cov = (src_c * tgt_c * m).sum(dim=(2, 3), keepdim=True)
    scale = (cov / var).clamp(0.5, 2.0)
    offset = (tgt_mean - scale * src_mean).clamp(-0.5, 0.5)
    return (scale * src + offset).clamp(0, 1)


class BrightnessAwarePhotometricLoss(nn.Module):
    def __init__(self, kappa: float = 0.85):
        super().__init__()
        self.kappa = kappa

    def forward(self, warped: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        corrected = affine_brightness_match(warped, target, mask)
        ssim_loss = (1.0 - ssim(corrected, target)) / 2.0
        l2 = (corrected - target).pow(2)
        loss = self.kappa * ssim_loss + (1.0 - self.kappa) * l2
        return (loss * mask).sum() / (mask.sum() * target.shape[1]).clamp_min(1.0)


def gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[..., :, 1:] - img[..., :, :-1]


def gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[..., 1:, :] - img[..., :-1, :]


def edge_aware_smoothness_loss(inv_depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    inv_norm = inv_depth / inv_depth.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    dx_depth = gradient_x(inv_norm).abs()
    dy_depth = gradient_y(inv_norm).abs()
    dx_img = gradient_x(image).abs().mean(dim=1, keepdim=True)
    dy_img = gradient_y(image).abs().mean(dim=1, keepdim=True)
    return (dx_depth * torch.exp(-dx_img)).mean() + (dy_depth * torch.exp(-dy_img)).mean()


def geometric_consistency_loss(inv_ref_projected: torch.Tensor, inv_next: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num = (inv_ref_projected - inv_next).abs()
    den = (inv_ref_projected + inv_next).clamp_min(1e-6)
    loss = num / den
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class SelfSupervisedCriterion(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        tcfg = cfg["training"]
        self.weights = tcfg.get("loss_weights", {"intensity": 1.0, "smoothness": 0.001, "geometry": 0.1})
        self.photo = BrightnessAwarePhotometricLoss(kappa=float(tcfg.get("photometric_kappa", 0.85)))

    def forward(self, batch: dict, outputs: dict, next_depth_outputs: dict | None = None) -> dict[str, torch.Tensor]:
        depth = outputs["depth"]
        pose_next = pose_vec_to_mat(outputs["pose_next"])
        warped_next, valid, projected_depth = projective_warp(batch["frame_next"], depth, pose_next, batch["intrinsics"])
        l_int = self.photo(warped_next, batch["frame_ref"], valid)
        l_reg = edge_aware_smoothness_loss(outputs["inv_depth"], batch["frame_ref"])
        l_geo = depth.new_tensor(0.0)
        if next_depth_outputs is not None:
            # compare inverse projected range against direct next-frame inverse range at corresponding pixels
            inv_projected = 1.0 / projected_depth.clamp_min(1e-6)
            inv_next = next_depth_outputs["inv_depth"].detach()
            l_geo = geometric_consistency_loss(inv_projected, inv_next, valid)
        total = (
            float(self.weights.get("intensity", 1.0)) * l_int
            + float(self.weights.get("smoothness", 0.001)) * l_reg
            + float(self.weights.get("geometry", 0.1)) * l_geo
        )
        return {"loss": total, "loss_intensity": l_int, "loss_smoothness": l_reg, "loss_geometry": l_geo}
