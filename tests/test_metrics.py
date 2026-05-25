from __future__ import annotations

import torch

from vv_wce.metrics.depth import depth_metrics


def test_depth_metrics_perfect():
    gt = torch.ones(1, 1, 4, 4)
    pred = torch.ones(1, 1, 4, 4)
    m = depth_metrics(pred, gt)
    assert m["abs_rel"] == 0.0
    assert m["delta1"] == 1.0
