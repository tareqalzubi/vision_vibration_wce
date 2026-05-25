from __future__ import annotations

import numpy as np
import torch


def vibration_rms(vibration: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(vibration.pow(2), dim=(1, 2)))


def tertile_groups(values: list[float]) -> list[str]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    q1, q2 = np.quantile(arr, [1 / 3, 2 / 3])
    labels = []
    for v in arr:
        if v <= q1:
            labels.append("low")
        elif v <= q2:
            labels.append("medium")
        else:
            labels.append("high")
    return labels
