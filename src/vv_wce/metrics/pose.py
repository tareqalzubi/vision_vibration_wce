from __future__ import annotations

import numpy as np
import torch


def relative_pose_error(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Compute one-step translational and rotational error for Bx4x4 relative transforms."""
    err = torch.linalg.inv(gt) @ pred
    t = torch.linalg.norm(err[:, :3, 3], dim=-1)
    trace = err[:, 0, 0] + err[:, 1, 1] + err[:, 2, 2]
    angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)) * 180.0 / torch.pi
    return {"rpe_t": float(t.mean().item()), "rpe_r_deg": float(angle.mean().item())}


def absolute_trajectory_error(pred_poses: np.ndarray, gt_poses: np.ndarray) -> float:
    """Translation RMSE after centering. This is lightweight; use evo for publication-grade alignment."""
    p = pred_poses[:, :3, 3]
    g = gt_poses[:, :3, 3]
    p = p - p.mean(axis=0, keepdims=True)
    g = g - g.mean(axis=0, keepdims=True)
    scale = np.sum(g * p) / max(np.sum(p * p), 1e-12)
    p = p * scale
    return float(np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1))))
