"""MAE pretraining trainer for MISFIT.

Supports single-GPU, multi-GPU (single node), and multi-node distributed
training via torchrun. The same code runs in all three configurations:

    # Single GPU
    misfit_train --index index.parquet --results /runs/exp1 ...

    # 4-GPU single node
    torchrun --nproc_per_node=4 $(which misfit_train) ...

    # 4 nodes x 8 GPUs = 32 GPUs
    torchrun --nproc_per_node=8 --nnodes=4 \\
             --node_rank=<rank> --master_addr=<addr> --master_port=29500 \\
             $(which misfit_train) ...

Distributed setup is torchrun-native: RANK, LOCAL_RANK, and WORLD_SIZE are
read from environment variables set by torchrun. No mp.spawn is used, making
multi-node training straightforward.
"""
import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import misfit.loss_functions  # noqa: F401 — trigger loss registrations
import misfit.models  # noqa: F401 — trigger model registrations
from misfit.data_loading.dataloader import (
    get_training_dataloader,
    get_validation_dataloader,
)
from misfit.loss_functions.loss_registry import get_loss
from misfit.models.model_registry import get_model_from_registry
from misfit.training.lr_schedulers.lr_scheduler_registry import get_lr_scheduler
from misfit.training.optimizers.optimizer_registry import get_optimizer
from misfit.training.trainer_constants import tc
from misfit.training.training_utils import (
    RunningMean,
    build_accumulation_plan,
    set_seed,
)
from misfit.utils import (
    console,
    get_progress_bar,
    print_warning,
    read_json_file,
    write_json_file,
)


class MAETrainer:
    """Masked Autoencoder pretraining trainer.

    Reads distributed context from torchrun environment variables so that
    the same class runs on 1 GPU or N×M GPUs without code changes.

    Args:
        args: Parsed CLI arguments from the misfit_train entrypoint.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # Read distributed context set by torchrun (default to single-GPU).
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_main = self.rank == 0

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.amp = True  # Always on by default; can be changed in config.json.

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_distributed(self) -> None:
        """Initialise the NCCL process group (torchrun-native)."""
        if self.is_distributed:
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.local_rank)

    def _enable_cudnn_optimisations(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def _warn_if_underutilising_gpus(self) -> None:
        """Warn when multiple GPUs are visible but we are not under torchrun.

        MISFIT has no mp.spawn fallback: without torchrun, WORLD_SIZE is 1 and
        training silently runs on a single GPU. Surface that rather than letting
        it pass unnoticed (a sharp edge vs. frameworks that auto-spawn DDP).
        """
        if (
            not self.is_distributed
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            n = torch.cuda.device_count()
            print_warning(
                f"{n} GPUs are visible but misfit_train was not launched with "
                f"torchrun; only cuda:0 will be used. For multi-GPU training, "
                f"run:\n  torchrun --nproc_per_node={n} $(which misfit_train) ..."
            )

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        model = get_model_from_registry(
            self.args.model,
            in_channels=1,
            img_size=tuple(self.args.patch_size),
            mask_patch_size=self.args.mask_patch_size,
            mask_ratio=self.args.mask_ratio,
        )
        model = model.to(self.device)
        if self.is_distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.local_rank], bucket_cap_mb=self.args.bucket_cap_mb)
        return model

    def _build_loss(self) -> nn.Module:
        loss_cls = get_loss(self.args.loss)
        if self.args.loss == "normalized_masked_mse":
            return loss_cls(patch_size=self.args.mask_patch_size)
        return loss_cls()

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return get_optimizer(
            name=self.args.optimizer,
            params=model.parameters(),
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=tc.NO_AMP_EPS,
        )

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return get_lr_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            epochs=self.args.epochs,
            warmup_epochs=self.args.warmup_epochs,
        )

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def _training_step(
        self,
        model: nn.Module,
        batch: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        window_size: int = 1,
        is_last_accum: bool = True,
    ) -> float:
        """Forward + backward for one micro-batch; optimizer step on last accumulation.

        Args:
            model: The (possibly DDP-wrapped) SwinMAE model.
            batch: Dict with "image" (B, 1, D, H, W) and "spacing" (B, 3).
            criterion: Reconstruction loss function.
            optimizer: Optimizer instance.
            window_size: Number of micro-steps in the current accumulation
                window; the loss is divided by this to normalise gradient
                magnitude. Use the window's actual size (which may be smaller
                than accum_steps for the trailing window of an epoch).
            is_last_accum: When True, performs the optimizer step and gradient
                clipping (the micro-step also runs with DDP gradient sync).
                When False, only accumulates gradients under no_sync(). The
                caller must call optimizer.zero_grad() before the first
                micro-step of each accumulation window.

        Returns:
            Un-scaled scalar loss for this micro-batch (before 1/window_size
            normalisation), for logging purposes.
        """
        images = batch["image"].to(self.device, non_blocking=True)
        spacing = batch["spacing"].to(self.device, non_blocking=True)

        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()
        )
        # Skip all-reduce on non-final micro-steps to avoid premature synchronisation.
        sync_ctx = (
            model.no_sync()
            if (self.is_distributed and not is_last_accum)
            else nullcontext()
        )
        with sync_ctx, amp_ctx:
            output = model(images, spacing=spacing)
            loss = criterion(
                reconstruction=output["reconstruction"],
                target=images,
                mask=output["mask"],
            ) / window_size

        loss.backward()
        if is_last_accum:
            nn.utils.clip_grad_norm_(model.parameters(), tc.GRAD_CLIP_VALUE)
            optimizer.step()

        return loss.item() * window_size  # un-scaled for logging

    def _validation_step(
        self,
        model: nn.Module,
        batch: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        """Forward pass only — no gradient computation.

        Args:
            model: The (possibly DDP-wrapped) SwinMAE model in eval mode.
            batch: Dict with "image" (B, 1, D, H, W) and "spacing" (B, 3).
            criterion: Reconstruction loss function.

        Returns:
            Scalar loss value for this batch.
        """
        images = batch["image"].to(self.device, non_blocking=True)
        spacing = batch["spacing"].to(self.device, non_blocking=True)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            output = model(images, spacing=spacing)
            loss = criterion(
                reconstruction=output["reconstruction"],
                target=images,
                mask=output["mask"],
            )
        return loss.item()

    # ------------------------------------------------------------------
    # Distributed loss aggregation
    # ------------------------------------------------------------------

    def _aggregate_loss(self, loss_value: float) -> float:
        """All-reduce a scalar loss across ranks and return the mean.

        No-op when not running in distributed mode.
        """
        if not self.is_distributed:
            return loss_value
        t = torch.tensor(loss_value, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() / self.world_size

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        global_step: int,
        best_val_loss: float,
        path: Path,
    ) -> None:
        """Atomically write a training checkpoint (rank 0 only).

        Writes to a .tmp file first then renames, so the checkpoint file is
        never in a partially-written state.
        """
        raw_model = model.module if self.is_distributed else model
        checkpoint = {
            "epoch":         epoch,
            "global_step":   global_step,
            "best_val_loss": best_val_loss,
            "model":         raw_model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
        }
        tmp = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp)
        tmp.rename(path)

    def _load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        path: Path,
    ) -> tuple[int, int, float]:
        """Load a checkpoint and restore all training state.

        Args:
            path: Checkpoint file path.

        Returns:
            Tuple of (start_epoch, global_step, best_val_loss).
            Returns (0, 0, inf) if the checkpoint does not exist.
        """
        if not path.exists():
            return 0, 0, float("inf")

        checkpoint = torch.load(
            path, map_location=self.device, weights_only=True
        )
        raw_model = model.module if self.is_distributed else model
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        return (
            checkpoint["epoch"],
            checkpoint["global_step"],
            checkpoint["best_val_loss"],
        )

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def _build_config(self) -> dict:
        """Serialise current training args to a reproducibility config dict."""
        import misfit
        from misfit.metrics.metrics_registry import list_registered_metrics
        return {
            "misfit_version": misfit.__version__,
            "data": {
                "index": str(self.args.index),
            },
            "model": {
                "architecture":    self.args.model,
                "patch_size":      list(self.args.patch_size),
                "mask_patch_size": self.args.mask_patch_size,
                "mask_ratio":      self.args.mask_ratio,
            },
            "training": {
                "epochs":        self.args.epochs,
                "batch_size":    self.args.batch_size,
                "optimizer":     self.args.optimizer,
                "learning_rate": self.args.learning_rate,
                "weight_decay":  self.args.weight_decay,
                "lr_scheduler":  self.args.lr_scheduler,
                "warmup_epochs": self.args.warmup_epochs,
                "loss":          self.args.loss,
                "amp":                        self.amp,
                "seed":                       self.args.seed,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "bucket_cap_mb":              self.args.bucket_cap_mb,
            },
            "evaluation": {
                metric: {} for metric in list_registered_metrics()
            },
        }

    def _validate_resume(self, saved_config: dict) -> None:
        """Check that the current args are compatible with a saved config.

        Hard errors (raises :class:`ValueError`) if architecture-defining
        fields changed — resuming with a different model or patch size would
        produce nonsensical results.  Soft mismatches (different
        hyperparameters) emit warnings but allow training to continue.
        """
        # Fields that are not safe to change on resume.
        immutable = [
            ("model", "architecture",    self.args.model),
            ("model", "patch_size",      list(self.args.patch_size)),
            ("model", "mask_patch_size", self.args.mask_patch_size),
        ]
        for section, key, current in immutable:
            saved = saved_config.get(section, {}).get(key)
            if saved is not None and saved != current:
                raise ValueError(
                    f"Cannot resume: '{section}.{key}' changed from "
                    f"{saved!r} to {current!r}.  "
                    "Use --overwrite to start fresh."
                )

        # Fields that are allowed to change but deserve a warning.
        soft = [
            ("training", "epochs",        self.args.epochs),
            ("training", "batch_size",    self.args.batch_size),
            ("training", "optimizer",     self.args.optimizer),
            ("training", "learning_rate", self.args.learning_rate),
            ("training", "weight_decay",  self.args.weight_decay),
            ("training", "lr_scheduler",  self.args.lr_scheduler),
            ("training", "warmup_epochs", self.args.warmup_epochs),
            ("training", "loss",                         self.args.loss),
            ("training", "gradient_accumulation_steps",  self.args.gradient_accumulation_steps),
            ("training", "bucket_cap_mb",                self.args.bucket_cap_mb),
        ]
        for section, key, current in soft:
            saved = saved_config.get(section, {}).get(key)
            if saved is not None and saved != current:
                print_warning(
                    f"Hyperparameter '{section}.{key}' changed from "
                    f"{saved!r} to {current!r}."
                )

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------

    def _make_progress(self):
        return get_progress_bar()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full MAE pretraining loop.

        Single entry point — safe to call from any rank. All rank-0-only
        operations (logging, checkpointing, console output) are guarded by
        ``self.is_main``.
        """
        # --- Config preconditions ---
        # Run on EVERY rank before distributed init. If only rank 0 checked and
        # raised, the other ranks would proceed into init_process_group and hang
        # on NCCL rendezvous waiting for a rank that already aborted. Checking
        # on all ranks makes a precondition failure exit the whole job cleanly.
        results_dir = Path(self.args.results)
        config_path = results_dir / "config.json"
        if (config_path.exists()
                and not self.args.resume
                and not self.args.overwrite):
            raise RuntimeError(
                f"Output directory '{results_dir}' already contains a "
                "config.json.  Use --resume to continue training or "
                "--overwrite to start fresh."
            )
        if self.args.resume and config_path.exists():
            self._validate_resume(read_json_file(config_path))

        # Read amp setting from saved config on resume (all ranks).
        if self.args.resume and config_path.exists():
            saved_training = read_json_file(config_path).get("training", {})
            self.amp = saved_training.get("amp", True)

        self._warn_if_underutilising_gpus()
        self._setup_distributed()
        self._enable_cudnn_optimisations()
        set_seed(self.args.seed, self.rank)

        # --- Build components ---
        model = self._build_model()
        criterion = self._build_loss().to(self.device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        # --- Data loaders ---
        train_loader = get_training_dataloader(
            index_path=self.args.index,
            patch_size=tuple(self.args.patch_size),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_cpu_workers,
            distributed=self.is_distributed,
            seed=self.args.seed,
        )
        val_loader = get_validation_dataloader(
            index_path=self.args.index,
            patch_size=tuple(self.args.patch_size),
            batch_size=self.args.batch_size,
            num_workers=max(self.args.num_cpu_workers // 2, 1),
            distributed=self.is_distributed,
            seed=self.args.seed,
        )

        # --- Output directories (rank 0 creates, then barrier) ---
        checkpoint_dir = results_dir / "checkpoints"
        models_dir = results_dir / "models"
        logs_dir = results_dir / "logs"
        if self.is_main:
            for d in (checkpoint_dir, models_dir, logs_dir):
                d.mkdir(parents=True, exist_ok=True)
            if not self.args.resume:
                write_json_file(config_path, self._build_config())
        if self.is_distributed:
            dist.barrier()

        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        best_model_path = models_dir / "best_model.pt"

        # --- Optionally resume ---
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        if self.args.resume:
            start_epoch, global_step, best_val_loss = self._load_checkpoint(
                model, optimizer, scheduler, checkpoint_path
            )
            if self.is_main:
                console.print(
                    f"[bold]Resumed from epoch {start_epoch} "
                    f"(best val loss: {best_val_loss:.4f})[/bold]"
                )

        # --- TensorBoard (rank 0 only) ---
        writer: SummaryWriter | None = (
            SummaryWriter(str(logs_dir)) if self.is_main else None
        )

        # --- Epoch loop ---
        if self.is_main:
            console.print(
                f"\n[bold green]Starting MAE pretraining[/bold green]  "
                f"model={self.args.model}  "
                f"world_size={self.world_size}  "
                f"epochs={self.args.epochs}  "
                f"amp={self.amp}\n"
            )

        for epoch in range(start_epoch, self.args.epochs):
            # Required for DistributedSampler to re-shuffle each epoch.
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)

            # ---- Training ----
            model.train()
            train_meter = RunningMean()

            accum_steps = self.args.gradient_accumulation_steps
            # Per-batch (window_size, is_window_end). The plan forces the final
            # batch of the epoch to close its window, so its gradients are
            # synced and stepped instead of being left under no_sync() (which
            # silently desynchronises ranks); it also scales the trailing
            # partial window by its actual size.
            plan = build_accumulation_plan(len(train_loader), accum_steps)
            progress_ctx = self._make_progress() if self.is_main else nullcontext()
            with progress_ctx as progress:
                task = (
                    progress.add_task(
                        f"Epoch {epoch + 1}/{self.args.epochs} [train]",
                        total=len(train_loader),
                    )
                    if self.is_main else None
                )
                optimizer.zero_grad()
                accum_loss = 0.0
                for (window_size, is_window_end), batch in zip(plan, train_loader):
                    step_loss = self._training_step(
                        model, batch, criterion, optimizer,
                        window_size=window_size, is_last_accum=is_window_end,
                    )
                    accum_loss += step_loss
                    if self.is_main and progress is not None:
                        progress.advance(task)
                    if is_window_end:
                        agg_loss = self._aggregate_loss(accum_loss / window_size)
                        train_meter.update(agg_loss)
                        global_step += 1
                        accum_loss = 0.0
                        optimizer.zero_grad()

            scheduler.step()

            # ---- Validation ----
            if self.is_distributed:
                dist.barrier()

            model.eval()
            val_meter = RunningMean()

            for batch in val_loader:
                step_loss = self._validation_step(model, batch, criterion)
                step_loss = self._aggregate_loss(step_loss)
                val_meter.update(step_loss)

            # ---- Logging & checkpointing (rank 0 only) ----
            if self.is_main:
                lr = optimizer.param_groups[0]["lr"]

                if writer is not None:
                    writer.add_scalar("loss/train", train_meter.value, epoch)
                    writer.add_scalar("loss/val",   val_meter.value,   epoch)
                    writer.add_scalar("lr",          lr,                epoch)
                    writer.flush()

                improved = val_meter.value < best_val_loss
                status = (
                    f"[green]↓ {best_val_loss:.4f} → {val_meter.value:.4f}[/green]"
                    if improved
                    else f"[dim](best: {best_val_loss:.4f})[/dim]"
                )
                console.print(
                    f"  train_loss={train_meter.value:.4f}  "
                    f"val_loss={val_meter.value:.4f}  "
                    f"lr={lr:.2e}  {status}"
                )

                # Update best_val_loss before saving rolling checkpoint so that
                # a resume always restores the correct best-so-far value.
                if improved:
                    best_val_loss = val_meter.value

                # Save rolling checkpoint every epoch.
                self._save_checkpoint(
                    model, optimizer, scheduler,
                    epoch + 1, global_step, best_val_loss, checkpoint_path,
                )

                # Save best model when validation loss improves.
                if improved:
                    self._save_checkpoint(
                        model, optimizer, scheduler,
                        epoch + 1, global_step, best_val_loss, best_model_path,
                    )
                    # Export encoder weights in MIST-compatible format so the
                    # pretrained encoder can be passed directly to mist_train
                    # via --pretrained-weights.
                    raw_model = model.module if self.is_distributed else model
                    encoder_weights_path = models_dir / "encoder_weights.pt"
                    torch.save(raw_model.get_encoder_state_dict(), encoder_weights_path)

            if self.is_distributed:
                dist.barrier()

        # ---- Cleanup ----
        if writer is not None:
            writer.close()
        if self.is_distributed:
            dist.destroy_process_group()
        if self.is_main:
            console.print(
                f"\n[bold green]Training complete.[/bold green]  "
                f"Best val loss: {best_val_loss:.4f}\n"
                f"Best model saved to: {best_model_path}"
            )
