from __future__ import annotations

import torch

from vv_wce.losses.self_supervised import edge_aware_smoothness_loss


def test_smoothness_finite():
    inv = torch.rand(2, 1, 16, 16)
    img = torch.rand(2, 3, 16, 16)
    loss = edge_aware_smoothness_loss(inv, img)
    assert torch.isfinite(loss)
