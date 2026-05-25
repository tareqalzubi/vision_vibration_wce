from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def resolve_path(root: str | Path, value: str | float | None) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(root) / p


def load_rgb(path: str | Path, size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def load_depth(path: str | Path | None, size: tuple[int, int] | None) -> torch.Tensor | None:
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p).astype("float32")
    elif p.suffix.lower() == ".npz":
        data = np.load(p)
        key = "depth" if "depth" in data else list(data.keys())[0]
        arr = data[key].astype("float32")
    else:
        arr = np.asarray(Image.open(p)).astype("float32")
        if arr.max() > 255:
            arr = arr / 1000.0
    if arr.ndim == 3:
        arr = arr[..., 0]
    t = torch.from_numpy(arr)[None]
    if size is not None and tuple(t.shape[-2:]) != tuple(size):
        t = torch.nn.functional.interpolate(t[None], size=size, mode="nearest")[0]
    return t


def load_vibration(path: str | Path | None, channels: int = 6, length: int = 240) -> torch.Tensor:
    if path is None or not Path(path).exists():
        return torch.zeros(channels, length, dtype=torch.float32)
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p).astype("float32")
    elif p.suffix.lower() == ".npz":
        data = np.load(p)
        key = "vibration" if "vibration" in data else list(data.keys())[0]
        arr = data[key].astype("float32")
    else:
        arr = np.loadtxt(p, delimiter="," if p.suffix.lower() == ".csv" else None).astype("float32")
    if arr.ndim != 2:
        raise ValueError(f"Vibration must be 2D, got shape {arr.shape} from {p}")
    if arr.shape[0] != channels and arr.shape[1] == channels:
        arr = arr.T
    if arr.shape[0] != channels:
        raise ValueError(f"Expected {channels} vibration channels, got {arr.shape} from {p}")
    if arr.shape[1] != length:
        # linear resample to target length
        x_old = np.linspace(0, 1, arr.shape[1])
        x_new = np.linspace(0, 1, length)
        arr = np.stack([np.interp(x_new, x_old, arr[c]) for c in range(channels)], axis=0).astype("float32")
    return torch.from_numpy(arr)


def load_pose(path: str | Path | None) -> torch.Tensor | None:
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p).astype("float32")
    else:
        arr = np.loadtxt(p).astype("float32")
    arr = arr.reshape(4, 4) if arr.size == 16 else arr
    if arr.shape != (4, 4):
        raise ValueError(f"Pose must be 4x4 or 16 values, got {arr.shape} from {p}")
    return torch.from_numpy(arr)
