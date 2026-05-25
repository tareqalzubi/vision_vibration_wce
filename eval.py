from __future__ import annotations

import argparse
from pathlib import Path
import torch

from vv_wce.data.build import build_loader
from vv_wce.engine.checkpoint import load_checkpoint
from vv_wce.engine.evaluator import evaluate
from vv_wce.models.model import VisionVibrationModel
from vv_wce.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(cfg, args.split, shuffle=False)
    model = VisionVibrationModel(cfg)
    load_checkpoint(args.checkpoint, model, map_location=device)
    out = args.out or str(Path(cfg.get("output_dir", "outputs/default")) / f"eval_{args.split}.json")
    metrics = evaluate(model, loader, cfg, device, out)
    print(metrics)


if __name__ == "__main__":
    main()
