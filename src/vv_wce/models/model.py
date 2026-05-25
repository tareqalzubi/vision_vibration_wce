from __future__ import annotations

import torch
from torch import nn

from .decoders import InverseRangeDecoder, PoseDecoder
from .fusion import RhoEstimator
from .vibration_encoder import build_vibration_encoder
from .vision_encoder import VisionEncoder


class VisionVibrationModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        channels = list(mcfg.get("encoder_channels", [32, 64, 128, 256]))
        descriptor_dim = int(mcfg.get("descriptor_dim", 128))
        vib_hidden = max(descriptor_dim // 2, 16)
        self.vibration_encoder = build_vibration_encoder(
            mcfg.get("vibration_encoder", "two_stream_mlstm_se"),
            in_channels=int(cfg["data"].get("vibration_channels", 6)),
            hidden=vib_hidden,
            descriptor_dim=descriptor_dim,
        )
        self.rho = RhoEstimator(descriptor_dim, mode=mcfg.get("rho_mode", "adaptive"), fixed_rho=float(mcfg.get("fixed_rho", 1.0)))
        self.visual_encoder = VisionEncoder(
            in_channels=int(mcfg.get("image_channels", 3)),
            channels=channels,
            descriptor_dim=descriptor_dim,
            fusion_type=mcfg.get("fusion_type", "frequency"),
            rho_mode=mcfg.get("rho_mode", "adaptive"),
        )
        self.depth_decoder = InverseRangeDecoder(channels, min_depth=float(mcfg.get("min_depth", 0.05)), max_depth=float(mcfg.get("max_depth", 20.0)))
        self.pose_decoder = PoseDecoder(in_channels=9, pose_scale=float(mcfg.get("pose_scale", 0.01)))

    def encode_vibration(self, vibration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor = self.vibration_encoder(vibration)
        rho = self.rho(descriptor)
        return descriptor, rho

    def predict_depth(self, frame_ref: torch.Tensor, vibration: torch.Tensor) -> dict[str, torch.Tensor]:
        descriptor, rho = self.encode_vibration(vibration)
        feats = self.visual_encoder(frame_ref, descriptor, rho)
        inv_depth = self.depth_decoder(feats, image_size=frame_ref.shape[-2:])
        return {"inv_depth": inv_depth, "depth": 1.0 / inv_depth.clamp_min(1e-6), "rho": rho, "descriptor": descriptor}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        depth_out = self.predict_depth(batch["frame_ref"], batch["vibration"])
        pose = self.pose_decoder(batch["frame_prev"], batch["frame_ref"], batch["frame_next"])
        depth_out["pose_prev"] = pose[:, 0]
        depth_out["pose_next"] = pose[:, 1]
        return depth_out
