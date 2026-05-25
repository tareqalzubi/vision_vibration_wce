from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common import ConvNormAct, UpsampleBlock


class InverseRangeDecoder(nn.Module):
    def __init__(self, encoder_channels: list[int], min_depth: float = 0.05, max_depth: float = 20.0):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.up3 = UpsampleBlock(c4, c3, c3)
        self.up2 = UpsampleBlock(c3, c2, c2)
        self.up1 = UpsampleBlock(c2, c1, c1)
        self.up0 = nn.Sequential(ConvNormAct(c1, max(c1 // 2, 8)), ConvNormAct(max(c1 // 2, 8), max(c1 // 2, 8)))
        self.pred = nn.Conv2d(max(c1 // 2, 8), 1, 3, padding=1)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, feats: list[torch.Tensor], image_size: tuple[int, int]) -> torch.Tensor:
        f1, f2, f3, f4 = feats
        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = F.interpolate(x, size=image_size, mode="bilinear", align_corners=False)
        x = self.up0(x)
        inv_min = 1.0 / self.max_depth
        inv_max = 1.0 / self.min_depth
        inv_depth = inv_min + (inv_max - inv_min) * torch.sigmoid(self.pred(x))
        return inv_depth


class PoseDecoder(nn.Module):
    def __init__(self, in_channels: int = 9, channels: list[int] | None = None, pose_scale: float = 0.01):
        super().__init__()
        channels = channels or [16, 32, 64, 128]
        layers = []
        prev = in_channels
        for ch in channels:
            layers += [nn.Conv2d(prev, ch, 7 if ch == channels[0] else 3, stride=2, padding=3 if ch == channels[0] else 1), nn.ReLU(inplace=True)]
            prev = ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(prev, 128), nn.ReLU(inplace=True), nn.Linear(128, 12))
        self.pose_scale = pose_scale

    def forward(self, frame_prev: torch.Tensor, frame_ref: torch.Tensor, frame_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([frame_prev, frame_ref, frame_next], dim=1)
        pose = self.fc(self.conv(x)).view(x.shape[0], 2, 6) * self.pose_scale
        return pose
