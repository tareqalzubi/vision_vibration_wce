from __future__ import annotations

import torch

from vv_wce.models.model import VisionVibrationModel


def test_forward_smoke():
    cfg = {
        "data": {"vibration_channels": 6, "vibration_length": 240},
        "model": {
            "image_channels": 3,
            "encoder_channels": [8, 16, 32, 64],
            "descriptor_dim": 32,
            "fusion_type": "frequency",
            "rho_mode": "adaptive",
            "vibration_encoder": "two_stream_mlstm_se",
            "min_depth": 0.05,
            "max_depth": 10.0,
        },
    }
    model = VisionVibrationModel(cfg)
    batch = {
        "frame_prev": torch.rand(2, 3, 64, 64),
        "frame_ref": torch.rand(2, 3, 64, 64),
        "frame_next": torch.rand(2, 3, 64, 64),
        "vibration": torch.rand(2, 6, 240),
    }
    out = model(batch)
    assert out["depth"].shape == (2, 1, 64, 64)
    assert out["pose_next"].shape == (2, 6)
