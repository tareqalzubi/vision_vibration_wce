from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, norm: bool = True):
        super().__init__()
        pad = kernel // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=not norm)]
        if norm:
            layers.append(nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch))
        layers.append(nn.ELU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvNormAct(in_ch + skip_ch, out_ch, 3, 1),
            ConvNormAct(out_ch, out_ch, 3, 1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)
