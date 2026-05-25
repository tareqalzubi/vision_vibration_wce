from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vv_wce.data.build import build_loader
from vv_wce.engine.evaluator import evaluate
from vv_wce.engine.trainer import Trainer
from vv_wce.models.model import VisionVibrationModel
from vv_wce.utils.config import deep_update, load_yaml, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument("--train", action="store_true", help="Train each ablation before evaluation.")
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for path in sorted(Path(args.ablation_dir).glob("*.yaml")):
        cfg = deep_update(base, load_yaml(path))
        cfg["output_dir"] = str(Path(base.get("output_dir", "outputs/ablations")) / path.stem)
        seed_everything(int(cfg.get("seed", 42)))
        train_loader = build_loader(cfg, "train", True)
        val_loader = build_loader(cfg, "val", False)
        model = VisionVibrationModel(cfg)
        if args.train:
            Trainer(model, train_loader, val_loader, cfg, device).fit()
        metrics = evaluate(model, val_loader, cfg, device)
        results[path.stem] = metrics
        print(path.stem, metrics)
    out_path = Path(base.get("output_dir", "outputs/ablations")) / "ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
