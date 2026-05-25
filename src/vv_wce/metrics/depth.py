from __future__ import annotations

import torch


def median_scale_prediction(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = pred.clone()
    for i in range(pred.shape[0]):
        m = mask[i, 0]
        if m.sum() > 10:
            scale = torch.median(gt[i, 0][m]) / torch.median(pred[i, 0][m]).clamp_min(1e-6)
            out[i] = pred[i] * scale
    return out


def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 100.0, median_scale: bool = False) -> dict[str, float]:
    mask = torch.isfinite(gt) & (gt > min_depth) & (gt < max_depth) & torch.isfinite(pred) & (pred > min_depth)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if median_scale:
        pred = median_scale_prediction(pred, gt, mask)
    pred_v = pred[mask].clamp(min_depth, max_depth)
    gt_v = gt[mask].clamp(min_depth, max_depth)
    if gt_v.numel() == 0:
        return {k: float("nan") for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]}
    abs_rel = torch.mean(torch.abs(gt_v - pred_v) / gt_v)
    sq_rel = torch.mean((gt_v - pred_v).pow(2) / gt_v)
    rmse = torch.sqrt(torch.mean((gt_v - pred_v).pow(2)))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt_v) - torch.log(pred_v)).pow(2)))
    ratio = torch.maximum(gt_v / pred_v, pred_v / gt_v)
    d1 = torch.mean((ratio < 1.25).float())
    d2 = torch.mean((ratio < 1.25 ** 2).float())
    d3 = torch.mean((ratio < 1.25 ** 3).float())
    return {
        "abs_rel": float(abs_rel.item()),
        "sq_rel": float(sq_rel.item()),
        "rmse": float(rmse.item()),
        "rmse_log": float(rmse_log.item()),
        "delta1": float(d1.item()),
        "delta2": float(d2.item()),
        "delta3": float(d3.item()),
    }
