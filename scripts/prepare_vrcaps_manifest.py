from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_rows(root: Path, split: str, fx: float, fy: float, cx: float, cy: float) -> list[dict]:
    rows = []
    for seq_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        rgb_dir = seq_dir / "rgb"
        if not rgb_dir.exists():
            continue
        frames = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
        for i in range(1, len(frames) - 1):
            stem = frames[i].stem
            def rel(p: Path) -> str:
                return str(p.relative_to(root)) if p.exists() else ""
            rows.append({
                "sequence_id": seq_dir.name,
                "index": i,
                "frame_prev": rel(frames[i - 1]),
                "frame_ref": rel(frames[i]),
                "frame_next": rel(frames[i + 1]),
                "vibration": rel(seq_dir / "vibration" / f"{stem}.npy"),
                "depth_ref": rel(seq_dir / "depth" / f"{stem}.npy"),
                "depth_next": rel(seq_dir / "depth" / f"{frames[i+1].stem}.npy"),
                "pose_ref": rel(seq_dir / "pose" / f"{stem}.txt"),
                "pose_next": rel(seq_dir / "pose" / f"{frames[i+1].stem}.txt"),
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "split": split,
            })
    return rows


def main(name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Prepare a manifest for {name}-style folders.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--fx", type=float, default=250.0)
    parser.add_argument("--fy", type=float, default=250.0)
    parser.add_argument("--cx", type=float, default=160.0)
    parser.add_argument("--cy", type=float, default=160.0)
    args = parser.parse_args()
    rows = infer_rows(Path(args.root), args.split, args.fx, args.fy, args.cx, args.cy)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main("VR-Caps")
