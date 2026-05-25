from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from vv_wce.engine.checkpoint import save_checkpoint
from vv_wce.losses.self_supervised import SelfSupervisedCriterion
from vv_wce.metrics.depth import depth_metrics
from vv_wce.utils.config import ensure_dir


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: dict, device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        tcfg = cfg["training"]
        self.criterion = SelfSupervisedCriterion(cfg)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.0)))
        self.use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.out = ensure_dir(cfg.get("output_dir", "outputs/default"))
        self.ckpt_dir = ensure_dir(self.out / "checkpoints")
        self.best = float("inf")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, float] = {}
        pbar = tqdm(self.train_loader, desc=f"train {epoch}")
        for step, batch in enumerate(pbar, 1):
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(batch)
                next_outputs = self.model.predict_depth(batch["frame_next"], batch["vibration"])
                loss_dict = self.criterion(batch, outputs, next_outputs)
            self.scaler.scale(loss_dict["loss"]).backward()
            if float(self.cfg["training"].get("grad_clip", 0)) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["training"].get("grad_clip", 1.0)))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for k, v in loss_dict.items():
                meters[k] = meters.get(k, 0.0) + float(v.detach().item())
            pbar.set_postfix({k: f"{meters[k]/step:.4f}" for k in meters if k.startswith("loss")})
        return {k: v / max(len(self.train_loader), 1) for k, v in meters.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        losses = []
        depth_scores = []
        for batch in tqdm(self.val_loader, desc=f"val {epoch}"):
            batch = move_to_device(batch, self.device)
            outputs = self.model(batch)
            next_outputs = self.model.predict_depth(batch["frame_next"], batch["vibration"])
            loss_dict = self.criterion(batch, outputs, next_outputs)
            losses.append(float(loss_dict["loss"].item()))
            if "depth_ref" in batch:
                depth_scores.append(depth_metrics(outputs["depth"], batch["depth_ref"], median_scale=bool(self.cfg.get("evaluation", {}).get("median_scale", True))))
        metrics = {"val_loss": sum(losses) / max(len(losses), 1)}
        if depth_scores:
            for k in depth_scores[0].keys():
                vals = [d[k] for d in depth_scores]
                metrics[k] = sum(vals) / max(len(vals), 1)
        if metrics["val_loss"] < self.best:
            self.best = metrics["val_loss"]
            save_checkpoint(self.ckpt_dir / "best.pt", self.model, self.optimizer, epoch, metrics, self.cfg)
        save_checkpoint(self.ckpt_dir / "last.pt", self.model, self.optimizer, epoch, metrics, self.cfg)
        return metrics

    def fit(self) -> None:
        epochs = int(self.cfg["training"].get("epochs", 1))
        for epoch in range(1, epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)
            print({**train_m, **val_m})
