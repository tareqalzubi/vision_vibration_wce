from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from vv_wce.engine.trainer import move_to_device
from vv_wce.geometry.pose import pose_vec_to_mat
from vv_wce.geometry.warping import projective_warp
from vv_wce.metrics.depth import depth_metrics
from vv_wce.metrics.pose import relative_pose_error
from vv_wce.metrics.robustness import vibration_rms
from vv_wce.metrics.synthesis import photometric_error, warp_ssim


@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: torch.device, out_path: str | Path | None = None) -> dict:
    model.eval().to(device)
    rows = []
    for batch in tqdm(loader, desc="eval"):
        batch = move_to_device(batch, device)
        outputs = model(batch)
        pose_next = pose_vec_to_mat(outputs["pose_next"])
        warped, valid, _ = projective_warp(batch["frame_next"], outputs["depth"], pose_next, batch["intrinsics"])
        row = {
            "photo_err": photometric_error(warped, batch["frame_ref"], valid),
            "ssim_warp": warp_ssim(warped, batch["frame_ref"]),
            "rho": float(outputs["rho"].mean().item()),
            "vibration_rms": float(vibration_rms(batch["vibration"]).mean().item()),
        }
        if "depth_ref" in batch:
            row.update(depth_metrics(outputs["depth"], batch["depth_ref"], median_scale=bool(cfg.get("evaluation", {}).get("median_scale", True))))
        if "pose_ref" in batch and "pose_next" in batch:
            gt_rel = torch.linalg.inv(batch["pose_ref"]) @ batch["pose_next"]
            row.update(relative_pose_error(pose_next, gt_rel))
        rows.append(row)
    metrics = {}
    if rows:
        for k in rows[0].keys():
            vals = [r[k] for r in rows if k in r]
            metrics[k] = sum(vals) / max(len(vals), 1)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")
    return metrics
