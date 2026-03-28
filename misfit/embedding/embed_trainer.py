"""EmbedTrainer — trains an aggregator on cached per-crop feature files."""
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from misfit.embedding.aggregators.aggregator_registry import get_aggregator
from misfit.embedding.objectives.base import crop_collate_fn
from misfit.embedding.objectives.objective_registry import get_objective
from misfit.utils import console, get_progress_bar


class EmbedTrainer:
    """Train an aggregator module on cached per-crop ``.npz`` feature files.

    The encoder is **not** touched — it is assumed that per-volume feature
    files (produced by ``misfit_embed``) already exist in
    *features_dir*.

    The training loop is intentionally simple:

    - One aggregator forward pass per volume.
    - Optional output head (classification: linear; contrastive: MLP).
    - Loss computed by the objective.
    - Adam optimiser with cosine LR decay.

    Args:
        features_dir: Directory of ``{volume_id}.npz`` files.
        labels_csv: Path to CSV with ``volume_id`` + label columns.
        label_col: Column to use as the training target.
        objective_name: ``"classification"`` or ``"contrastive"``.
        aggregator_name: ``"mean_pool"`` or ``"attention_pool"``.
        embed_dim: Dimensionality ``C`` of each crop feature vector.
        output_dir: Where to save ``aggregator.pt`` and training logs.
        num_epochs: Number of training epochs.
        batch_size: Number of volumes per batch (for classification) or
            ``M × 2`` pairs (for contrastive).
        learning_rate: Initial learning rate for Adam.
        num_workers: DataLoader worker count.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        use_position_encoding: Passed to ``AttentionPoolAggregator`` when
            ``aggregator_name="attention_pool"``.
    """

    def __init__(
        self,
        features_dir: Path,
        labels_csv: Path,
        label_col: str,
        objective_name: str,
        aggregator_name: str,
        embed_dim: int,
        output_dir: Path,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        num_workers: int = 4,
        device: str = "cuda",
        use_position_encoding: bool = True,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.labels_csv = Path(labels_csv)
        self.label_col = label_col
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_position_encoding = use_position_encoding

        # Instantiate objective and aggregator.
        objective_cls = get_objective(objective_name)
        self.objective = objective_cls()

        aggregator_cls = get_aggregator(aggregator_name)
        agg_kwargs = {"embed_dim": embed_dim}
        if aggregator_name == "attention_pool":
            agg_kwargs["use_position_encoding"] = use_position_encoding
        self.aggregator: nn.Module = aggregator_cls(**agg_kwargs).to(self.device)

        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full training loop and save the trained aggregator."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        labels_df = pd.read_csv(self.labels_csv)
        labels_df = self.objective.validate_labels(labels_df, self.label_col)

        dataset = self.objective.build_dataset(
            self.features_dir, labels_df, self.label_col
        )
        if len(dataset) == 0:
            raise RuntimeError(
                "No feature files matched the label CSV.  "
                "Run `misfit_embed` first."
            )

        batch_sampler = self.objective.build_batch_sampler(dataset, self.batch_size)

        if batch_sampler is not None:
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=crop_collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.device.type == "cuda",
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=crop_collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.device.type == "cuda",
            )

        head = self.objective.build_head(self.embed_dim).to(self.device)

        params = list(self.aggregator.parameters()) + list(head.parameters())
        optimiser = torch.optim.Adam(params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.num_epochs
        )

        console.print(
            f"[bold]Training aggregator[/bold] — {self.num_epochs} epochs, "
            f"{len(dataset)} volumes"
        )

        with get_progress_bar() as progress:
            epoch_task = progress.add_task(
                "Training", total=self.num_epochs
            )
            for epoch in range(1, self.num_epochs + 1):
                avg_loss = self._train_epoch(loader, head, optimiser)
                scheduler.step()
                progress.advance(epoch_task)

                if epoch % max(1, self.num_epochs // 10) == 0:
                    console.print(
                        f"  epoch {epoch:4d}/{self.num_epochs}  "
                        f"loss={avg_loss:.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

        self._save(head)
        console.print(f"[green]Aggregator saved to {self.output_dir}[/green]")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        head: nn.Module,
        optimiser: torch.optim.Optimizer,
    ) -> float:
        self.aggregator.train()
        head.train()
        total_loss = 0.0
        n_batches = 0

        for features, positions, padding_mask, labels in loader:
            features = features.to(self.device)      # (B, N_max, C)
            positions = positions.to(self.device)    # (B, N_max, 3)
            padding_mask = padding_mask.to(self.device)  # (B, N_max)
            labels = labels.to(self.device)          # (B,)

            # Aggregate each volume in the batch independently.
            embeddings = self._batch_aggregate(features, positions, padding_mask)  # (B, C)

            head_output = head(embeddings)           # (B, num_classes) or (B, proj_dim)
            loss = self.objective.compute_loss(head_output, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _batch_aggregate(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the aggregator over each volume in the batch.

        Args:
            features:     ``(B, N_max, C)``
            positions:    ``(B, N_max, 3)``
            padding_mask: ``(B, N_max)`` True = padded

        Returns:
            ``(B, C)`` volume embeddings.
        """
        B = features.shape[0]
        out = []
        for i in range(B):
            emb = self.aggregator(
                crop_features=features[i],
                positions=positions[i],
                padding_mask=padding_mask[i],
            )  # (C,)
            out.append(emb)
        return torch.stack(out, dim=0)  # (B, C)

    def _save(self, head: nn.Module) -> None:
        torch.save(
            {
                "aggregator_state": self.aggregator.state_dict(),
                "head_state": head.state_dict(),
                "embed_dim": self.embed_dim,
                "objective": self.objective.name,
                "aggregator": self.aggregator.name
                if hasattr(self.aggregator, "name")
                else type(self.aggregator).__name__,
            },
            self.output_dir / "aggregator.pt",
        )
