from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vv_wce.data.io import load_rgb, load_vibration
from vv_wce.engine.checkpoint import load_checkpoint
from vv_wce.models.model import VisionVibrationModel
from vv_wce.utils.config import load_config, ensure_dir


def save_depth_png(depth: torch.Tensor, path: Path) -> None:
    arr = depth.squeeze().detach().cpu().numpy()
    arr = arr / max(np.percentile(arr, 95), 1e-6)
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype("uint8")).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-prev", required=True)
    parser.add_argument("--frame-ref", required=True)
    parser.add_argument("--frame-next", required=True)
    parser.add_argument("--vibration", required=True)
    parser.add_argument("--out", default="outputs/inference")
    args = parser.parse_args()
    cfg = load_config(args.config)
    size = tuple(cfg["data"].get("image_size", [320, 320]))
    batch = {
        "frame_prev": load_rgb(args.frame_prev, size).unsqueeze(0),
        "frame_ref": load_rgb(args.frame_ref, size).unsqueeze(0),
        "frame_next": load_rgb(args.frame_next, size).unsqueeze(0),
        "vibration": load_vibration(args.vibration, cfg["data"].get("vibration_channels", 6), cfg["data"].get("vibration_length", 240)).unsqueeze(0),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionVibrationModel(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
    out_dir = ensure_dir(args.out)
    torch.save({k: v.cpu() for k, v in out.items() if torch.is_tensor(v)}, out_dir / "prediction.pt")
    save_depth_png(out["depth"], out_dir / "depth_preview.png")
    print(f"Saved inference outputs to {out_dir}")


if __name__ == "__main__":
    main()
