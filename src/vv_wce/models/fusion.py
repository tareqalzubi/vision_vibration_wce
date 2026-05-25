from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RhoEstimator(nn.Module):
    def __init__(self, descriptor_dim: int, mode: str = "adaptive", fixed_rho: float = 1.0):
        super().__init__()
        self.mode = mode
        self.fixed_rho = float(fixed_rho)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, max(descriptor_dim // 2, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(descriptor_dim // 2, 8), 1),
        )
        self.global_log_rho = nn.Parameter(torch.zeros(1))

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        b = descriptor.shape[0]
        if self.mode == "fixed":
            return descriptor.new_full((b, 1), self.fixed_rho)
        if self.mode == "global":
            return F.softplus(self.global_log_rho).expand(b, 1) + 1e-4
        if self.mode in {"adaptive", "direct"}:
            return F.softplus(self.mlp(descriptor)) + 1e-4
        raise ValueError(f"Unknown rho mode: {self.mode}")


class NullFusion(nn.Module):
    def forward(self, x: torch.Tensor, descriptor: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return x


class FrequencyReliabilityFusion(nn.Module):
    """Wiener-style spectral purification controlled by vibration-derived reliability rho."""

    def __init__(self, channels: int, descriptor_dim: int, rho_mode: str = "adaptive", eps: float = 1e-4):
        super().__init__()
        self.rho_mode = rho_mode
        self.eps = eps
        self.transfer = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.direct = nn.Linear(descriptor_dim, channels) if rho_mode == "direct" else None
        self.out = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.GroupNorm(num_groups=min(8, channels), num_channels=channels), nn.ELU(inplace=True))

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x_freq = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
        a = F.softplus(self.transfer).float()
        if self.direct is not None:
            # Direct vibration injection ablation: descriptor drives channel-wise spectral denominator.
            channel_rho = F.softplus(self.direct(descriptor)).view(x.shape[0], x.shape[1], 1, 1).float() + self.eps
            denom = a.pow(2) + 1.0 / channel_rho
        else:
            denom = a.pow(2) + 1.0 / rho.view(-1, 1, 1, 1).float().clamp_min(self.eps)
        purified = x_freq * a / (denom + self.eps)
        y = torch.fft.irfft2(purified, s=(h, w), dim=(-2, -1), norm="ortho").to(dtype=x.dtype)
        return self.out(y + x)


class ConcatFusion(nn.Module):
    def __init__(self, channels: int, descriptor_dim: int):
        super().__init__()
        self.proj = nn.Linear(descriptor_dim, channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        g = self.proj(descriptor).view(x.shape[0], x.shape[1], 1, 1).expand_as(x)
        return self.conv(torch.cat([x, g], dim=1))


class FiLMFusion(nn.Module):
    def __init__(self, channels: int, descriptor_dim: int):
        super().__init__()
        self.affine = nn.Linear(descriptor_dim, channels * 2)

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(descriptor).chunk(2, dim=-1)
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta = beta.view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


class SpatialGateFusion(nn.Module):
    def __init__(self, channels: int, descriptor_dim: int):
        super().__init__()
        self.proj = nn.Linear(descriptor_dim, channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.ELU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        g = self.proj(descriptor).view(x.shape[0], x.shape[1], 1, 1).expand_as(x)
        gate = self.gate(torch.cat([x, g], dim=1))
        return x * gate


def build_fusion(kind: str, channels: int, descriptor_dim: int, rho_mode: str) -> nn.Module:
    kind = kind.lower()
    if kind in {"none", "vision_only", "null"}:
        return NullFusion()
    if kind in {"frequency", "freq", "spectral"}:
        return FrequencyReliabilityFusion(channels, descriptor_dim, rho_mode=rho_mode)
    if kind in {"concat", "concat_conv", "naive"}:
        return ConcatFusion(channels, descriptor_dim)
    if kind == "film":
        return FiLMFusion(channels, descriptor_dim)
    if kind in {"spatial_gate", "spatial", "gate"}:
        return SpatialGateFusion(channels, descriptor_dim)
    raise ValueError(f"Unknown fusion type: {kind}")
