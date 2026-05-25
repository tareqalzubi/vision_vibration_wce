from __future__ import annotations

import argparse
import torch

from vv_wce.data.build import build_loader
from vv_wce.engine.trainer import Trainer
from vv_wce.models.model import VisionVibrationModel
from vv_wce.utils.config import load_config, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    model = VisionVibrationModel(cfg)
    Trainer(model, train_loader, val_loader, cfg, device).fit()


if __name__ == "__main__":
    main()
