from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make_pose(i: int) -> np.ndarray:
    t = np.eye(4, dtype=np.float32)
    t[0, 3] = i * 0.002
    t[1, 3] = np.sin(i / 5) * 0.001
    t[2, 3] = 0.01 + i * 0.0005
    return t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/smoke")
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--frames-per-sequence", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    h = w = args.image_size
    yy, xx = np.mgrid[0:h, 0:w]
    for s in range(args.num_sequences):
        seq = f"seq{s:03d}"
        for sub in ["rgb", "depth", "vibration", "pose"]:
            (root / seq / sub).mkdir(parents=True, exist_ok=True)
        for i in range(args.frames_per_sequence):
            base = (xx + yy + i * 4 + s * 10) % 255
            img = np.stack([base, np.roll(base, i, axis=0), np.roll(base, i, axis=1)], axis=-1).astype(np.uint8)
            Image.fromarray(img).save(root / seq / "rgb" / f"{i:06d}.png")
            depth = 1.0 + 0.2 * np.sin(xx / 10 + i / 3) + 0.1 * np.cos(yy / 9)
            np.save(root / seq / "depth" / f"{i:06d}.npy", depth.astype(np.float32))
            pose = make_pose(i)
            np.savetxt(root / seq / "pose" / f"{i:06d}.txt", pose.reshape(4, 4))
            t = np.linspace(0, 1, 240)
            vib = np.stack([np.sin(2 * np.pi * (c + 1) * t + i / 5) for c in range(6)], axis=0)
            vib += 0.05 * np.random.randn(6, 240)
            np.save(root / seq / "vibration" / f"{i:06d}.npy", vib.astype(np.float32))
        for i in range(1, args.frames_per_sequence - 1):
            split = "train" if i < args.frames_per_sequence - 3 else ("val" if i == args.frames_per_sequence - 3 else "test")
            rows.append({
                "sequence_id": seq,
                "index": i,
                "frame_prev": f"{seq}/rgb/{i-1:06d}.png",
                "frame_ref": f"{seq}/rgb/{i:06d}.png",
                "frame_next": f"{seq}/rgb/{i+1:06d}.png",
                "vibration": f"{seq}/vibration/{i:06d}.npy",
                "depth_ref": f"{seq}/depth/{i:06d}.npy",
                "depth_next": f"{seq}/depth/{i+1:06d}.npy",
                "pose_ref": f"{seq}/pose/{i:06d}.txt",
                "pose_next": f"{seq}/pose/{i+1:06d}.txt",
                "fx": 60.0,
                "fy": 60.0,
                "cx": w / 2,
                "cy": h / 2,
                "split": split,
            })
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    print(f"Wrote synthetic smoke dataset to {root}")


if __name__ == "__main__":
    main()
