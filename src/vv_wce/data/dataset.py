from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io import load_depth, load_pose, load_rgb, load_vibration, resolve_path


class EndoscopyTripletDataset(Dataset):
    """Manifest-based dataset for self-supervised WCE range and motion learning."""

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (320, 320),
        vibration_channels: int = 6,
        vibration_length: int = 240,
        normalize_vibration: bool = True,
    ) -> None:
        self.root = Path(root)
        self.manifest = Path(manifest)
        self.split = split
        self.image_size = tuple(image_size)
        self.vibration_channels = vibration_channels
        self.vibration_length = vibration_length
        self.normalize_vibration = normalize_vibration

        df = pd.read_csv(self.manifest)
        if "split" in df.columns:
            df = df[df["split"].astype(str) == str(split)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {manifest}")
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _intrinsics(self, row: pd.Series) -> torch.Tensor:
        h, w = self.image_size
        fx = float(row.get("fx", w))
        fy = float(row.get("fy", h))
        cx = float(row.get("cx", w / 2))
        cy = float(row.get("cy", h / 2))
        k = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
        return k

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        prev_path = resolve_path(self.root, row["frame_prev"])
        ref_path = resolve_path(self.root, row["frame_ref"])
        next_path = resolve_path(self.root, row["frame_next"])
        vib_path = resolve_path(self.root, row.get("vibration"))

        frame_prev = load_rgb(prev_path, self.image_size)
        frame_ref = load_rgb(ref_path, self.image_size)
        frame_next = load_rgb(next_path, self.image_size)
        vibration = load_vibration(vib_path, self.vibration_channels, self.vibration_length)
        if self.normalize_vibration:
            mean = vibration.mean(dim=1, keepdim=True)
            std = vibration.std(dim=1, keepdim=True).clamp_min(1e-6)
            vibration = (vibration - mean) / std

        depth_ref = load_depth(resolve_path(self.root, row.get("depth_ref")), self.image_size)
        depth_next = load_depth(resolve_path(self.root, row.get("depth_next")), self.image_size)
        pose_ref = load_pose(resolve_path(self.root, row.get("pose_ref")))
        pose_next = load_pose(resolve_path(self.root, row.get("pose_next")))

        sample: Dict[str, Any] = {
            "frame_prev": frame_prev,
            "frame_ref": frame_ref,
            "frame_next": frame_next,
            "vibration": vibration,
            "intrinsics": self._intrinsics(row),
            "sequence_id": str(row.get("sequence_id", "unknown")),
            "index": int(row.get("index", idx)),
        }
        if depth_ref is not None:
            sample["depth_ref"] = depth_ref
        if depth_next is not None:
            sample["depth_next"] = depth_next
        if pose_ref is not None:
            sample["pose_ref"] = pose_ref
        if pose_next is not None:
            sample["pose_next"] = pose_next
        return sample


def collate_triplets(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        vals = [b[key] for b in batch if key in b]
        if len(vals) != len(batch):
            continue
        if torch.is_tensor(vals[0]):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out
