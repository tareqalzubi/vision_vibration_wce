from __future__ import annotations

from torch.utils.data import DataLoader

from .dataset import EndoscopyTripletDataset, collate_triplets


def build_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    data_cfg = cfg["data"]
    split_name = data_cfg.get(f"{split}_split", split)
    ds = EndoscopyTripletDataset(
        root=data_cfg["root"],
        manifest=data_cfg["manifest"],
        split=split_name,
        image_size=tuple(data_cfg.get("image_size", [320, 320])),
        vibration_channels=int(data_cfg.get("vibration_channels", 6)),
        vibration_length=int(data_cfg.get("vibration_length", 240)),
    )
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    if split != "train":
        batch_size = int(cfg.get("evaluation", {}).get("batch_size", batch_size))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=__import__("torch").cuda.is_available(),
        collate_fn=collate_triplets,
        drop_last=(split == "train"),
    )
