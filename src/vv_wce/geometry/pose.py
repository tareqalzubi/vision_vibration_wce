from __future__ import annotations

import torch


def skew(v: torch.Tensor) -> torch.Tensor:
    b = v.shape[0]
    z = torch.zeros(b, device=v.device, dtype=v.dtype)
    return torch.stack([
        z, -v[:, 2], v[:, 1],
        v[:, 2], z, -v[:, 0],
        -v[:, 1], v[:, 0], z,
    ], dim=-1).view(b, 3, 3)


def axis_angle_to_matrix(vec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-9)
    axis = vec / angle
    k = skew(axis)
    eye = torch.eye(3, device=vec.device, dtype=vec.dtype).unsqueeze(0).expand(vec.shape[0], 3, 3)
    sin = torch.sin(angle).view(-1, 1, 1)
    cos = torch.cos(angle).view(-1, 1, 1)
    return eye + sin * k + (1.0 - cos) * (k @ k)


def pose_vec_to_mat(pose: torch.Tensor) -> torch.Tensor:
    """Convert Bx6 vector [tx,ty,tz,rx,ry,rz] to Bx4x4 transform."""
    t = pose[:, :3]
    r = pose[:, 3:]
    rot = axis_angle_to_matrix(r)
    b = pose.shape[0]
    mat = torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(b, 1, 1)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = t
    return mat


def invert_transform(t: torch.Tensor) -> torch.Tensor:
    r = t[:, :3, :3]
    tr = t[:, :3, 3:4]
    out = torch.eye(4, device=t.device, dtype=t.dtype).unsqueeze(0).repeat(t.shape[0], 1, 1)
    out[:, :3, :3] = r.transpose(1, 2)
    out[:, :3, 3:4] = -r.transpose(1, 2) @ tr
    return out
