Advanced Topics
===

## Reproducibility and Resumption

MISFIT writes a `config.json` file to the `--results` directory at the start of
training. This file records the MISFIT version, model architecture, patch size,
and all training hyperparameters. It is the **single source of truth** for
model architecture — `misfit_evaluate`, `misfit_inspect`, and `misfit_embed` all
require it via `--config` rather than re-accepting architecture flags.

### config.json structure

Below is an example `config.json` produced by `misfit_train`.

```json
{
  "misfit_version": "0.1.0-alpha",

  "data": {
    "index": "/data/index.parquet"
  },

  "model": {
    "name": "swinmae-base",
    "patch_size": [96, 96, 96],
    "mask_patch_size": 16,
    "mask_ratio": 0.75
  },

  "training": {
    "epochs": 200,
    "batch_size": 2,
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "weight_decay": 0.05,
    "lr_scheduler": "cosine",
    "warmup_epochs": 20,
    "loss": "normalized_masked_mse",
    "amp": true,
    "seed": 42
  },

  "evaluation": {
    "masked_mae":  {},
    "masked_mse":  {},
    "masked_psnr": {},
    "ssim":        {}
  }
}
```

### Resuming a run

Pass `--resume` to continue from the latest checkpoint in `--results`. MISFIT
will read `config.json` and validate that the architecture and patch size are
unchanged. If either has changed, training exits with an error rather than
silently producing an incompatible checkpoint.

Changes to other hyperparameters (learning rate, epochs, optimizer, etc.) are
allowed and emit a warning so you are aware of the discrepancy.

```console
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --resume
```

### Starting fresh

If you want to discard a previous run and start from scratch, pass `--overwrite`.
This deletes the existing `config.json` and checkpoints before training begins.

```console
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --overwrite
```

!!!warning
    `--resume` and `--overwrite` are mutually exclusive. If neither is passed and
    `config.json` already exists in `--results`, `misfit_train` refuses to run.
    This is intentional — it prevents accidentally overwriting a completed run.

### Immutable vs. mutable parameters

The following parameters are **immutable** — changing them while resuming raises
a hard error:

| Parameter | Reason |
|---|---|
| `model.name` | Checkpoint weights are architecture-specific. |
| `model.patch_size` | Determines the spatial dimension of all model tensors. |
| `model.mask_patch_size` | Determines the masking grid structure inside the encoder. |

All other parameters (learning rate, epochs, optimizer, loss, etc.) are
**mutable** — changes produce a warning and training continues.

---

## Model Variants

MISFIT provides three SwinMAE variants corresponding to different
encoder capacities. All variants share the same SwinUNETR backbone architecture
but differ in the width of the feature maps (`feature_size`).

| Variant | `--model` | `feature_size` | Parameters (approx.) | Recommended for |
|---|---|---|---|---|
| Small | `swinmae-small` | 24 | ~14M | Rapid prototyping, small datasets |
| Base | `swinmae-base` | 48 | ~55M | Standard pretraining (default) |
| Large | `swinmae-large` | 96 | ~210M | Large-scale datasets, maximum capacity |

!!!note
    The model variant is locked into `config.json` at the start of training and
    cannot be changed when resuming. To use a different variant, start a new run
    with `--overwrite` or in a new `--results` directory.

---

## Patch Size Selection

The `--patch-size` argument controls the spatial crop fed to the model during
training and inference. Every dimension must be divisible by 32 (due to the
SwinUNETR downsampling stages).

A few practical guidelines:

- **GPU memory** is the primary constraint. A 32 GB GPU with batch size 2 can
  comfortably fit `96 96 96`. Reduce to `64 64 64` if you run out of memory.
  Mixed-precision (AMP) is always enabled on new runs. To disable it, let
  training run for at least one epoch (so `config.json` is written), then set
  `"amp": false` in the `training` section of `config.json` and restart with
  `--resume`.

- **Voxel spacing matters.** If your data has 5 mm slice thickness (thick-slice
  CT or MRI), a `96 96 96` crop at 1 mm isotropic covers much more anatomy than
  the raw voxel count suggests. Consider resampling to isotropic spacing as a
  preprocessing step.

- **The patch size is fixed at inference time.** `misfit_inspect`, `misfit_evaluate`,
  and `misfit_embed` all read `patch_size` from `config.json` via `--config`.
  You do not need to specify it again on the command line.

---

## Mask Ratio

The `--mask-ratio` controls what fraction of patch tokens are hidden from the
encoder during pretraining. The default of 0.75 (75%) follows the original MAE
paper and works well for 3D medical images.

Higher mask ratios force the model to learn longer-range spatial dependencies;
lower ratios make the reconstruction task easier and may be preferable for
datasets with complex, fine-grained anatomy where local context is important.

---

## Loss Functions

| Loss | `--loss` | Description |
|---|---|---|
| Normalized Masked MSE | `normalized_masked_mse` | MSE computed on masked patches, normalized by patch variance. Recommended for mixed-modality datasets (CT + MRI). |
| Masked MSE | `masked_mse` | Standard MSE on masked patches without variance normalization. |
| Masked MAE | `masked_mae` | Mean absolute error on masked patches. More robust to intensity outliers than MSE. |
| Masked L1 | `masked_l1` | Alias for masked MAE. |

The `normalized_masked_mse` loss is recommended as the default because it
normalizes each patch's contribution by its local variance, preventing
high-contrast regions (e.g., bone in CT) from dominating the gradient signal.
This is especially useful when pretraining on datasets that mix modalities with
very different intensity distributions.

---

## Optimizers

| Optimizer | `--optimizer` | Notes |
|---|---|---|
| AdamW | `adamw` | Default. Best general-purpose choice for ViT-based architectures. |
| Adam | `adam` | No weight decay. Use `adamw` for better regularization. |
| SGD | `sgd` | Requires careful learning rate tuning. Not recommended for MAE pretraining. |

---

## Learning Rate Schedulers

| Scheduler | `--lr-scheduler` | Description |
|---|---|---|
| Cosine | `cosine` | Cosine annealing from `--learning-rate` to 0. Default. |
| Polynomial | `polynomial` | Polynomial decay. |
| Constant | `constant` | No decay; learning rate stays at `--learning-rate`. |

All schedulers support a **linear warmup** phase controlled by `--warmup-epochs`.
During warmup the learning rate increases linearly from 0 to `--learning-rate`.
A warmup of 20 epochs is recommended for SwinMAE — skipping warmup can cause
instability in the early stages of training.

---

## Multi-Node Training

MISFIT uses `torch.distributed` and can be launched with `torchrun` on any
cluster that supports NCCL. Distributed setup is handled automatically when
`torchrun` sets the `LOCAL_RANK` environment variable.

### Single node, multiple GPUs

```console
torchrun --nproc_per_node=4 \
    $(which misfit_train) \
        --index      /data/index.parquet \
        --results    /runs/exp1 \
        --batch-size 2
```

!!!note
    `--batch-size` is the **per-GPU** batch size. The effective global batch
    size is `--batch-size × number of GPUs`. Adjust `--learning-rate` accordingly
    (linear scaling rule: multiply LR by the number of GPUs when scaling up).

### Multiple nodes

On a SLURM cluster or similar, set `MASTER_ADDR` and `MASTER_PORT` and use
`torchrun --nnodes` and `--node_rank`:

```console
torchrun --nnodes=2 \
         --nproc_per_node=4 \
         --node_rank=$SLURM_NODEID \
         --master_addr=$MASTER_ADDR \
         --master_port=29500 \
    $(which misfit_train) \
        --index   /data/index.parquet \
        --results /runs/exp1
```

---

## Embedding Aggregators

When running `misfit_embed`, the encoder produces a feature map for each
cubic crop extracted from the volume. An **aggregator** pools these per-crop
feature vectors into a single volume-level embedding.

### Mean Pool (`mean_pool`)

The simplest aggregator: computes the unweighted mean of all crop feature
vectors. No training is required — it works zero-shot directly after
pretraining.

Use `mean_pool` when you want a quick, training-free embedding for retrieval
or visualization (e.g., UMAP of a cohort).

### Attention Pool (`attention_pool`)

A learnable aggregator that uses a multi-head cross-attention mechanism to
weight crop contributions, conditioned on their 3D spatial positions. It can
learn to focus on diagnostically relevant regions for a given task.

Requires training with `misfit_embed_train` before use. Pass the resulting
`aggregator.pt` to `misfit_embed` via `--aggregator-checkpoint`.

---

## Embedding Training Objectives

Both objectives train only the aggregator — the pretrained encoder weights are
frozen. This makes embedding training fast and memory-efficient even on large
feature sets.

### Classification (`classification`)

Optimizes a cross-entropy loss for multi-class label prediction. The aggregator
learns to produce discriminative embeddings for the `label` column in your
`--input` CSV. Labels are treated as strings and mapped to integer indices
lexicographically; the mapping is saved in `aggregator.pt` for
inference-time decoding.

### Contrastive (`contrastive`)

Optimizes a Supervised Contrastive loss (SupCon). Volumes sharing the same
label are pulled together in embedding space; volumes with different labels
are pushed apart. Uses K=2 pairs per group.

Contrastive training generally produces more generalizable embeddings than
classification training, at the cost of requiring balanced sampling.
`--batch-size` must be even.

---

## Preparing the Embedding Training Input CSV

`misfit_embed_train` accepts a single unified CSV with four required columns:

| Column | Description |
|---|---|
| `volume_id` | Volume identifier — used for logging only. |
| `split` | Dataset split. Only `split='train'` rows are used for training. |
| `features_path` | Absolute path to the `.npz` file produced by `misfit_embed`. |
| `label` | String label for the training objective. |

A minimal example:

```csv
volume_id,split,features_path,label
CT_001,train,/data/embeddings/CT_001.npz,adenocarcinoma
CT_002,train,/data/embeddings/CT_002.npz,squamous_cell
CT_003,val,/data/embeddings/CT_003.npz,adenocarcinoma
CT_004,val,/data/embeddings/CT_004.npz,squamous_cell
```

You can include val and test rows in the same file — only `split='train'` rows
are loaded for aggregator training.
