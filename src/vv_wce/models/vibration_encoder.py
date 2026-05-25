from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SequenceSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,C
        pooled = x.mean(dim=1)
        weights = self.fc(pooled).unsqueeze(1)
        return x * weights


class TwoStreamMLSTMSEEncoder(nn.Module):
    """Practical MLSTM-style encoder using multiplicative input gating and two LSTM streams."""

    def __init__(self, in_channels: int = 6, hidden: int = 64, descriptor_dim: int = 128, use_se: bool = True):
        super().__init__()
        self.input_gate = nn.Sequential(nn.Conv1d(in_channels, in_channels, 1), nn.Sigmoid())
        self.base = nn.LSTM(in_channels, hidden, batch_first=True)
        self.attn = nn.LSTM(in_channels, hidden, batch_first=True)
        self.se = SequenceSE(hidden) if use_se else nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, descriptor_dim),
            nn.LayerNorm(descriptor_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, vibration: torch.Tensor) -> torch.Tensor:
        # vibration: B,C,T
        gated = vibration * self.input_gate(vibration)
        x = gated.transpose(1, 2)  # B,T,C
        _, (h_base, _) = self.base(x)
        attn_out, _ = self.attn(x)
        attn_out = self.se(attn_out)
        h_attn = attn_out.mean(dim=1)
        h = torch.cat([h_base[-1], h_attn], dim=-1)
        return self.proj(h)


class SingleMLSTMEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden: int = 64, descriptor_dim: int = 128):
        super().__init__()
        self.input_gate = nn.Sequential(nn.Conv1d(in_channels, in_channels, 1), nn.Sigmoid())
        self.rnn = nn.LSTM(in_channels, hidden, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden, descriptor_dim), nn.LayerNorm(descriptor_dim), nn.ReLU(inplace=True))

    def forward(self, vibration: torch.Tensor) -> torch.Tensor:
        x = (vibration * self.input_gate(vibration)).transpose(1, 2)
        _, (h, _) = self.rnn(x)
        return self.proj(h[-1])


class BiGRUEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden: int = 64, descriptor_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, descriptor_dim), nn.LayerNorm(descriptor_dim), nn.ReLU(inplace=True))

    def forward(self, vibration: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(vibration.transpose(1, 2))
        return self.proj(out.mean(dim=1))


class TCNEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden: int = 64, descriptor_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 5, padding=2, dilation=1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(nn.Linear(hidden, descriptor_dim), nn.LayerNorm(descriptor_dim), nn.ReLU(inplace=True))

    def forward(self, vibration: torch.Tensor) -> torch.Tensor:
        x = self.net(vibration).mean(dim=-1)
        return self.proj(x)


def build_vibration_encoder(name: str, in_channels: int, hidden: int, descriptor_dim: int) -> nn.Module:
    name = name.lower()
    if name in {"two_stream_mlstm_se", "proposed", "mlstm_se"}:
        return TwoStreamMLSTMSEEncoder(in_channels, hidden, descriptor_dim, use_se=True)
    if name in {"mlstm_no_se", "two_stream_mlstm"}:
        return TwoStreamMLSTMSEEncoder(in_channels, hidden, descriptor_dim, use_se=False)
    if name in {"single_mlstm", "1s_mlstm"}:
        return SingleMLSTMEncoder(in_channels, hidden, descriptor_dim)
    if name == "bigru":
        return BiGRUEncoder(in_channels, hidden, descriptor_dim)
    if name == "tcn":
        return TCNEncoder(in_channels, hidden, descriptor_dim)
    raise ValueError(f"Unknown vibration encoder: {name}")
