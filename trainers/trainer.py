"""
trainer.py
----------
Full training pipeline for Multi-Teacher Knowledge Distillation IQA.

Supports:
  - Multiple teacher models (black-box)
  - Confidence-based gating with logging
  - Relational KD (optional, toggled via config)
  - Proper val loop with SRCC / PLCC
  - Gradient clipping
  - CSV logging
  - Mixed precision (optional)
"""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from losses.kd_losses import KDLossComposer
from losses.relational import build_relational_loss
from models.gating import DisagreementGating, build_gating
from models.student import build_student
from models.teacher_wrapper import TeacherWrapper
from utils.logger import CSVLogger
from utils.metrics import MetricAccumulator


class Trainer:
    """
    Orchestrates the full multi-teacher KD training loop.

    Args:
        cfg      : Full config dict
        teachers : Dict of {name: TeacherWrapper}
        train_loader : DataLoader yielding (images, gt_scores)
        val_loader   : DataLoader yielding (images, gt_scores)
        device   : torch.device
    """

    def __init__(
        self,
        cfg: dict,
        teachers: Dict[str, TeacherWrapper],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.cfg = cfg
        self.teachers = teachers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        train_cfg = cfg.get("training", {})
        self.num_epochs = train_cfg.get("epochs", 30)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.use_amp = train_cfg.get("mixed_precision", False)
        self.log_gating = train_cfg.get("log_gating_weights", True)
        self.use_relational = cfg.get("losses", {}).get("use_relational_kd", False)

        # --- Build models ---
        self.student = build_student(cfg).to(device)

        num_teachers = len(teachers)
        embed_dim = cfg.get("student", {}).get("embed_dim", 256)
        self.gating = build_gating(cfg, embed_dim, num_teachers).to(device)

        # --- Build losses ---
        self.kd_composer = KDLossComposer(cfg)
        if self.use_relational:
            self.relational_loss = build_relational_loss(cfg)
            self.lambda_rel = cfg.get("losses", {}).get("lambda_relational", 0.5)

        # --- Optimizer ---
        opt_cfg = cfg.get("optimizer", {})
        params = list(self.student.parameters()) + list(self.gating.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )

        # LR scheduler: cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )

        # --- AMP scaler ---
        self.scaler = GradScaler(enabled=self.use_amp)

        # --- Logging ---
        log_dir = cfg.get("logging", {}).get("log_dir", "./logs")
        self.logger = CSVLogger(log_dir=log_dir)
        self.teacher_names = list(teachers.keys())

        # Track best val SRCC for checkpointing
        self.best_srcc = -1.0
        self.ckpt_dir = Path(log_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_teacher_scores(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run all teachers and stack outputs.

        Returns:
            teacher_scores : (B, num_teachers)
        """
        scores = []
        for name in self.teacher_names:
            t_score = self.teachers[name](images)  # (B,) — no_grad inside wrapper
            scores.append(t_score)
        return torch.stack(scores, dim=-1)  # (B, num_teachers)

    def _forward_student(self, images: torch.Tensor):
        """
        Run student forward pass.

        Returns:
            score     : (B,)
            embedding : (B, 256)
        """
        return self.student(images)  # returns (score, embedding) if return_features=True

    def _compute_gating(
        self,
        embedding: torch.Tensor,
        teacher_scores: torch.Tensor,
        student_score: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gating weights — handles both MLPGating and DisagreementGating."""
        if isinstance(self.gating, DisagreementGating):
            return self.gating(embedding, teacher_scores, student_score)
        else:
            return self.gating(embedding)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        self.student.train()
        self.gating.train()

        total_losses = {
            "total": 0.0, "gt": 0.0,
            "kd_standard": 0.0, "kd_confidence": 0.0, "relational": 0.0,
        }
        gating_log = {name: 0.0 for name in self.teacher_names}
        n_batches = 0

        for images, gt_scores in self.train_loader:
            images = images.to(self.device)
            gt_scores = gt_scores.to(self.device).float()

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # Teacher inference (no_grad inside wrapper)
                teacher_scores = self._get_teacher_scores(images)  # (B, T)

                # Student forward
                student_score, embedding = self._forward_student(images)

                # Gating
                gating_weights = self._compute_gating(
                    embedding, teacher_scores, student_score
                )  # (B, T)

                # KD losses
                losses = self.kd_composer(
                    student_score, gt_scores, teacher_scores, gating_weights
                )
                total = losses["total"]

                # Optional relational KD
                if self.use_relational:
                    rel_loss = self.relational_loss(student_score, teacher_scores)
                    total = total + self.lambda_rel * rel_loss
                    losses["relational"] = rel_loss

            self.scaler.scale(total).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.gating.parameters()),
                self.grad_clip,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            for k in total_losses:
                if k in losses:
                    total_losses[k] += losses[k].item()

            # Accumulate gating weights for logging
            if self.log_gating:
                mean_weights = gating_weights.detach().mean(dim=0)  # (T,)
                for i, name in enumerate(self.teacher_names):
                    gating_log[name] += mean_weights[i].item()

            n_batches += 1

        # Average over batches
        avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
        avg_gating = {f"gate_{k}": v / max(n_batches, 1) for k, v in gating_log.items()}

        return {**avg_losses, **avg_gating}

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    def _val_epoch(self) -> dict:
        self.student.eval()
        acc = MetricAccumulator()

        with torch.no_grad():
            for images, gt_scores in self.val_loader:
                images = images.to(self.device)
                gt_scores = gt_scores.to(self.device).float()

                student_score, _ = self._forward_student(images)
                acc.update(student_score, gt_scores)

        return acc.compute()  # {'srcc': ..., 'plcc': ...}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run full training loop."""
        print(f"\n{'='*60}")
        print(f"  Starting training — {self.num_epochs} epochs")
        print(f"  Teachers: {self.teacher_names}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._val_epoch()

            # LR step
            self.scheduler.step()

            # Combine metrics for logging
            row = {
                "epoch": epoch,
                "lr": self.scheduler.get_last_lr()[0],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            self.logger.log(**row)
            self.logger.print_row(epoch, row)

            # Checkpoint best model
            val_srcc = val_metrics.get("srcc", 0.0)
            if val_srcc > self.best_srcc:
                self.best_srcc = val_srcc
                self._save_checkpoint(epoch, val_srcc)

        print(f"\n Training complete. Best Val SRCC: {self.best_srcc:.4f}")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, srcc: float):
        path = self.ckpt_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "srcc": srcc,
                "student_state": self.student.state_dict(),
                "gating_state": self.gating.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  [Checkpoint] Saved best model (SRCC={srcc:.4f}) → {path}")