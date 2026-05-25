from __future__ import annotations

import torch
from torch import nn

from .common import ResidualBlock
from .fusion import build_fusion


class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], descriptor_dim: int, fusion_type: str, rho_mode: str):
        super().__init__()
        stages = []
        fusions = []
        prev = in_channels
        for ch in channels:
            stages.append(nn.Sequential(ResidualBlock(prev, ch, stride=2), ResidualBlock(ch, ch, stride=1)))
            fusions.append(build_fusion(fusion_type, ch, descriptor_dim, rho_mode))
            prev = ch
        self.stages = nn.ModuleList(stages)
        self.fusions = nn.ModuleList(fusions)
        self.out_channels = channels

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor, rho: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage, fusion in zip(self.stages, self.fusions):
            x = stage(x)
            x = fusion(x, descriptor, rho)
            feats.append(x)
        return feats
