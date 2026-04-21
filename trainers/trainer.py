"""
trainers/trainer.py
-------------------
Full training pipeline for Multi-Teacher Knowledge Distillation IQA.

Changes from v1:
  - tqdm progress bar: one bar per epoch, live running-average loss postfix
  - DataParallel: optional toggle via cfg['training']['data_parallel']
  - 3-way split: accepts train / val / test loaders; test runs once at end
  - Early stopping: patience-based, monitors val SRCC
  - Saves both best checkpoint AND last checkpoint
  - Scheduler toggle: 'cosine' (default) or 'plateau'
  - Bug fixes:
      * Checkpoint keys unified to 'student' / 'gating' everywhere
      * _forward_student handles return_features=True and False safely
      * DataParallel-safe state_dict via .module unwrapping
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        cfg          : Full config dict
        teachers     : Dict of {name: TeacherWrapper}
        train_loader : DataLoader yielding (images, gt_scores)
        val_loader   : DataLoader yielding (images, gt_scores)
        test_loader  : DataLoader yielding (images, gt_scores) -- run once at end
        device       : torch.device
    """

    def __init__(
        self,
        cfg: dict,
        teachers: Dict[str, TeacherWrapper],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ):
        self.cfg          = cfg
        self.teachers     = teachers
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device       = device

        train_cfg = cfg.get("training", {})
        self.num_epochs     = train_cfg.get("epochs", 30)
        self.grad_clip      = train_cfg.get("grad_clip", 1.0)
        self.use_amp        = train_cfg.get("mixed_precision", False)
        self.log_gating     = train_cfg.get("log_gating_weights", True)
        self.use_relational = cfg.get("losses", {}).get("use_relational_kd", False)
        self.use_dp         = train_cfg.get("data_parallel", False)

        # Early stopping config
        es_cfg             = train_cfg.get("early_stopping", {})
        self.es_patience   = es_cfg.get("patience", 10)
        self.es_min_delta  = es_cfg.get("min_delta", 1e-4)
        self._es_counter   = 0
        self._es_triggered = False

        # -- Build student ------------------------------------------------
        self.student = build_student(cfg).to(device)

        # DataParallel wraps student only (teachers are frozen/black-box)
        if self.use_dp and torch.cuda.device_count() > 1:
            print(f"  [DataParallel] Using {torch.cuda.device_count()} GPUs")
            self.student = nn.DataParallel(self.student)
        elif self.use_dp:
            print("  [DataParallel] Requested but only 1 GPU found -- single GPU used")

        # -- Build gating -------------------------------------------------
        num_teachers = len(teachers)
        embed_dim    = cfg.get("student", {}).get("embed_dim", 256)
        self.gating  = build_gating(cfg, embed_dim, num_teachers).to(device)

        # -- Build losses -------------------------------------------------
        self.kd_composer = KDLossComposer(cfg)
        if self.use_relational:
            self.relational_loss = build_relational_loss(cfg)
            self.lambda_rel      = cfg.get("losses", {}).get("lambda_relational", 0.5)

        # -- Optimiser ----------------------------------------------------
        # nn.DataParallel.parameters() correctly yields all wrapped params
        opt_cfg = cfg.get("optimizer", {})
        params  = list(self.student.parameters()) + list(self.gating.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )

        # -- LR Scheduler -------------------------------------------------
        sched_mode = train_cfg.get("scheduler", "cosine")
        if sched_mode == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
            self._plateau_scheduler = False
        elif sched_mode == "plateau":
            # Monitors val SRCC (higher = better), steps when it plateaus
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )
            self._plateau_scheduler = True
        else:
            raise ValueError(
                f"Unknown scheduler '{sched_mode}'. Choose 'cosine' or 'plateau'."
            )

        # -- AMP scaler ---------------------------------------------------
        self.scaler = GradScaler(enabled=self.use_amp)

        # -- Logging / checkpointing --------------------------------------
        log_dir        = cfg.get("logging", {}).get("log_dir", "./logs")
        self.logger    = CSVLogger(log_dir=log_dir)
        self.teacher_names = list(teachers.keys())

        self.best_srcc = -1.0
        self.ckpt_dir  = Path(log_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _unwrapped_student(self) -> nn.Module:
        """Return raw student module, unwrapping DataParallel if present."""
        return (
            self.student.module
            if isinstance(self.student, nn.DataParallel)
            else self.student
        )

    def _get_teacher_scores(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run all teachers and stack outputs.

        Returns:
            teacher_scores : (B, num_teachers)
        """
        scores = [self.teachers[n](images) for n in self.teacher_names]
        return torch.stack(scores, dim=-1)

    def _forward_student(self, images: torch.Tensor):
        """
        Run student forward pass safely regardless of return_features setting.

        Returns:
            score     : (B,)
            embedding : (B, embed_dim)  or  None if return_features=False
        """
        out = self.student(images)
        if isinstance(out, tuple):
            return out[0], out[1]   # (score, embedding)
        return out, None            # (score, None)

    def _compute_gating(
        self,
        embedding: Optional[torch.Tensor],
        teacher_scores: torch.Tensor,
        student_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-teacher gating weights.

        Falls back to uniform weights when no embedding is available.
        """
        if embedding is None:
            B, T = teacher_scores.shape
            return torch.full((B, T), 1.0 / T, device=teacher_scores.device)
        if isinstance(self.gating, DisagreementGating):
            return self.gating(embedding, teacher_scores, student_score)
        return self.gating(embedding)

    # ─────────────────────────────────────────────────────────────────────
    # Training epoch
    # ─────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.student.train()
        self.gating.train()

        # Running sums -- divided by n_batches for smooth postfix values
        running    = {"total": 0.0, "gt": 0.0, "kd_std": 0.0, "kd_conf": 0.0}
        gating_sum = {name: 0.0 for name in self.teacher_names}
        n_batches  = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Ep {epoch:03d}/{self.num_epochs:03d} [Train]",
            leave=True,
            dynamic_ncols=True,
        )

        for images, gt_scores in pbar:
            images    = images.to(self.device)
            gt_scores = gt_scores.to(self.device).float()

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                teacher_scores           = self._get_teacher_scores(images)
                student_score, embedding = self._forward_student(images)
                gating_weights           = self._compute_gating(
                    embedding, teacher_scores, student_score
                )
                losses = self.kd_composer(
                    student_score, gt_scores, teacher_scores, gating_weights
                )
                total = losses["total"]

                if self.use_relational:
                    rel   = self.relational_loss(student_score, teacher_scores)
                    total = total + self.lambda_rel * rel
                    losses["relational"] = rel

            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.gating.parameters()),
                self.grad_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate running averages
            n_batches        += 1
            running["total"] += losses["total"].item()
            running["gt"]    += losses.get("gt",            torch.zeros(1)).item()
            running["kd_std"]+= losses.get("kd_standard",   torch.zeros(1)).item()
            running["kd_conf"]+= losses.get("kd_confidence",torch.zeros(1)).item()

            if self.log_gating:
                mean_w = gating_weights.detach().mean(dim=0)  # (T,)
                for i, name in enumerate(self.teacher_names):
                    gating_sum[name] += mean_w[i].item()

            # Live running-average postfix (much less noisy than per-batch)
            pbar.set_postfix({
                "loss" : f"{running['total']   / n_batches:.4f}",
                "gt"   : f"{running['gt']      / n_batches:.4f}",
                "kd"   : f"{(running['kd_std'] + running['kd_conf']) / n_batches:.4f}",
            })

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        avg.update({
            f"gate_{n}": gating_sum[n] / max(n_batches, 1)
            for n in self.teacher_names
        })
        return avg

    # ─────────────────────────────────────────────────────────────────────
    # Eval epoch  (shared by val and test)
    # ─────────────────────────────────────────────────────────────────────

    def _eval_epoch(self, loader: DataLoader, label: str) -> dict:
        """
        One evaluation pass over `loader`.

        Args:
            loader : val_loader or test_loader
            label  : display label for tqdm ('Val' or 'Test')

        Returns:
            {'srcc': float, 'plcc': float}
        """
        self.student.eval()
        acc  = MetricAccumulator()

        pbar = tqdm(
            loader,
            desc=f"{'':>21}[{label:>5}]",
            leave=False,        # Val bar disappears cleanly after each epoch
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for images, gt_scores in pbar:
                images    = images.to(self.device)
                gt_scores = gt_scores.to(self.device).float()
                score, _  = self._forward_student(images)
                acc.update(score, gt_scores)

        return acc.compute()

    # ─────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, srcc: float, tag: str = "best"):
        """
        Save checkpoint to  <ckpt_dir>/<tag>_model.pt

        Always unwraps DataParallel before saving so the checkpoint
        is loadable on any number of GPUs (or CPU) without modification.

        State dict keys are always 'student' and 'gating'.

        Args:
            tag : 'best' or 'last'
        """
        path = self.ckpt_dir / f"{tag}_model.pt"
        torch.save(
            {
                "epoch"     : epoch,
                "srcc"      : srcc,
                "student"   : self._unwrapped_student().state_dict(),
                "gating"    : self.gating.state_dict(),
                "optimizer" : self.optimizer.state_dict(),
            },
            path,
        )
        if tag == "best":
            tqdm.write(
                f"  [Ckpt] Best saved  epoch={epoch}  SRCC={srcc:.4f}  -> {path}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Early stopping
    # ─────────────────────────────────────────────────────────────────────

    def _check_early_stopping(self, val_srcc: float) -> bool:
        """
        Returns True if training should stop.

        Counter increments when improvement < min_delta.
        Resets to 0 on any meaningful improvement.
        """
        if val_srcc > self.best_srcc + self.es_min_delta:
            self._es_counter = 0
        else:
            self._es_counter += 1
            if self._es_counter >= self.es_patience:
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────

    def train(self):
        """
        Run full training + validation loop, then evaluate on the test set.

        Per-epoch flow:
          1. _train_epoch   -- tqdm bar with running-average loss postfix
          2. _eval_epoch    -- brief val bar (disappears after)
          3. Scheduler step
          4. CSV log
          5. Save best + last checkpoints
          6. Early stopping check

        After training:
          7. Load best checkpoint -> run test evaluation -> log test results
        """
        dp_info = (
            f"DataParallel x{torch.cuda.device_count()}"
            if self.use_dp and torch.cuda.device_count() > 1
            else str(self.device)
        )
        print(f"\n{'='*65}")
        print(f"  Training  {self.num_epochs} epochs  |  {dp_info}")
        print(f"  Teachers  : {self.teacher_names}")
        print(f"  Scheduler : {'plateau' if self._plateau_scheduler else 'cosine'}")
        print(f"  Early stop: patience={self.es_patience}  min_delta={self.es_min_delta}")
        print(f"{'='*65}\n")

        for epoch in range(1, self.num_epochs + 1):

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._eval_epoch(self.val_loader, "Val")
            val_srcc      = val_metrics["srcc"]

            # Scheduler step
            if self._plateau_scheduler:
                self.scheduler.step(val_srcc)
                current_lr = self.optimizer.param_groups[0]["lr"]
            else:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

            # CSV log
            row = {
                "epoch": epoch,
                "lr"   : current_lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}":   v for k, v in val_metrics.items()},
            }
            self.logger.log(**row)

            # Per-epoch summary line (printed after tqdm bars)
            gate_str = "  ".join(
                f"gate_{n}={train_metrics.get(f'gate_{n}', 0):.3f}"
                for n in self.teacher_names
            )
            tqdm.write(
                f"  >> Ep {epoch:03d}  "
                f"loss={train_metrics['total']:.4f}  "
                f"val_SRCC={val_srcc:.4f}  "
                f"val_PLCC={val_metrics['plcc']:.4f}  "
                f"lr={current_lr:.2e}  "
                + gate_str
            )

            # Checkpoints
            if val_srcc > self.best_srcc:
                self.best_srcc = val_srcc
                self._save_checkpoint(epoch, val_srcc, tag="best")
            self._save_checkpoint(epoch, val_srcc, tag="last")

            # Early stopping
            if self._check_early_stopping(val_srcc):
                tqdm.write(
                    f"\n  Early stopping after epoch {epoch} "
                    f"({self.es_patience} epochs without improvement >= {self.es_min_delta})."
                )
                break

        print(f"\n  Training complete.  Best Val SRCC = {self.best_srcc:.4f}")
        print(f"  Logs      -> {self.logger.log_path}")
        print(f"  Best ckpt -> {self.ckpt_dir / 'best_model.pt'}")
        print(f"  Last ckpt -> {self.ckpt_dir / 'last_model.pt'}")

        # Final test pass
        self._run_test()

    # ─────────────────────────────────────────────────────────────────────
    # Test evaluation
    # ─────────────────────────────────────────────────────────────────────

    def _run_test(self):
        """
        Load best checkpoint and evaluate on the held-out test set.

        Test is never used for checkpointing or early stopping --
        it is a one-time final evaluation after all training is done.

        Test metrics are printed and appended to the CSV log as a
        row with epoch=-1 so it is easy to filter programmatically.
        """
        best_path = self.ckpt_dir / "best_model.pt"
        if not best_path.exists():
            print("  [Test] No best checkpoint found -- skipping test evaluation.")
            return

        print("\n  Loading best checkpoint for final test evaluation...")
        ckpt = torch.load(best_path, map_location=self.device)
        self._unwrapped_student().load_state_dict(ckpt["student"])
        print(f"  (Best: epoch {ckpt['epoch']}, Val SRCC = {ckpt['srcc']:.4f})\n")

        test_metrics = self._eval_epoch(self.test_loader, "Test")

        print(
            f"  *** FINAL TEST RESULTS ***\n"
            f"      SRCC = {test_metrics['srcc']:.4f}\n"
            f"      PLCC = {test_metrics['plcc']:.4f}\n"
        )

        # Append test row to the same CSV (epoch=-1 distinguishes it)
        self.logger.log(
            epoch=-1,
            lr=0.0,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        )