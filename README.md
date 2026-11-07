# MISFIT: Medical Imaging Semantic Foundation Toolkit

[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/misfit-medical)](https://pypi.org/project/misfit-medical/)
[![Docker](https://img.shields.io/docker/v/mistmedical/misfit?label=docker&sort=semver)](https://hub.docker.com/r/mistmedical/misfit)
[![Docs](https://img.shields.io/readthedocs/misfit)](https://misfit.readthedocs.io/en/latest/)
[![Tests](https://img.shields.io/github/actions/workflow/status/mist-medical/MISFIT/test.yml?branch=main&label=tests)](https://github.com/mist-medical/MISFIT/actions/workflows/test.yml)
![Coverage](coverage.svg)

MISFIT is a simple, scalable, end-to-end framework for pretraining 3D medical
imaging foundation models using masked autoencoders (MAE). Give it a directory
of unlabeled NIfTI files and it produces a pretrained encoder — no labels
required.

---

## What it does

MISFIT trains a masked autoencoder on 3D medical images, using a
[SwinUNETR](https://arxiv.org/abs/2201.01266) or Primus transformer or a ResEnc
CNN built from [MIST](https://github.com/mist-medical/MIST)'s own models. At
each training step, 75% of the image patches are randomly hidden, and the model
learns to reconstruct them from the remaining context. This forces the encoder
to build rich spatial representations of anatomy — representations that transfer
well to downstream tasks like classification, segmentation, and retrieval.

```
Unlabeled NIfTIs  →  misfit_index  →  misfit_train  →  Pretrained Encoder
                                                               ↓
                                             misfit_encode  →  Raw Spatial Features (N_crops, C, D', H', W')
                                                               ↓
                                        misfit_embed_train  →  Trained Aggregator (optional)
                                                               ↓
                                              misfit_embed  →  Global Embedding (C,)  →  Retrieval / Classifier
```

---

## Key Features

- **No labels required** — pretrains entirely on unlabeled NIfTI volumes
- **Built for MIST** — encoders are MIST's own SwinUNETR-V2, ResEnc, and Primus,
  so pretrained weights and `config.json` load directly into `mist_train`
- **Starts from nnFoundation** — the openly released nnFoundationCNN and
  nnFoundationViT work in every command, as a baseline encoder or a starting
  point for pretraining
- **End-to-end pipeline** — seven CLI commands take you from raw files to
  downstream-ready embeddings
- **3D-native** — operates on full volumetric data, not 2D slices
- **Mixed modality** — the `normalized_masked_mse` loss normalizes per-patch
  variance, handling CT and MRI in the same training run
- **Scalable** — single-GPU to multi-node training via `torchrun`; step-based
  epochs, a fixed validation subset, and mid-epoch checkpoints keep long runs on
  large corpora practical; the same command runs everywhere (CPU too, for
  testing)
- **BF16 AMP** — automatic `bfloat16` mixed precision on GPUs with BF16 matrix
  hardware (NVIDIA Ampere+ — A100, H100, RTX 30xx+ — and AMD CDNA / RDNA3+),
  with an automatic FP32 fallback on older GPUs and CPU
- **Reproducible** — `config.json` captures every architecture and training
  hyperparameter; downstream commands require it rather than re-accepting flags
- **100% test coverage**

---

## Installation

From PyPI:

```console
pip install misfit-medical
```

Or run the container (CUDA 12.8, torch 2.9.1):

```console
docker pull mistmedical/misfit:latest
```

From source (for development, or to add a new model / loss):

```console
git clone https://github.com/mist-medical/MISFIT.git
cd MISFIT
pip install -e .
```

On an **AMD ROCm** machine, install a ROCm-enabled PyTorch build first (PyPI's
default `torch` wheel is CUDA-only; the Docker image is CUDA-only too), matching
your driver's ROCm version, then MISFIT on top with nothing extra:

```console
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install misfit-medical
```

**Requirements:** Python ≥ 3.10. A GPU with BF16 matrix hardware — NVIDIA Ampere
or newer (A100, H100, RTX 30xx+) or AMD CDNA / RDNA3+ (MI200/MI300, RX 7000+) —
is recommended, since it enables BF16 mixed precision. Pre-Ampere NVIDIA GPUs,
older AMD GPUs (RDNA1/2), and CPU-only machines work too; MISFIT automatically
falls back to FP32.

---

## Quick Start

```console
# 1. Build an index from a CSV of NIfTI paths
misfit_index --input  paths.csv \
             --output index.parquet

# 2. Pretrain a SwinMAE encoder
misfit_train --index   index.parquet \
             --results /runs/exp1

# 3. Evaluate reconstruction quality
misfit_evaluate --checkpoint /runs/exp1/models/best_model.pt \
                --index      index.parquet \
                --config     /runs/exp1/config.json \
                --output-csv /runs/exp1/eval_results.csv

# 4. Extract volume embeddings
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index               index.parquet \
             --config              /runs/exp1/config.json \
             --output-dir          /data/embeddings
```

---

## Pipeline

### Stage 1 — Indexing (`misfit_index`)

Scans your NIfTI files in parallel and computes per-volume intensity statistics
(p1, p99, foreground mean/std, bounding box) and voxel spacing. Assigns each
volume to a `train`, `val`, or `test` split. The resulting Parquet index is the
single input to all downstream commands.

```console
misfit_index --input  /data/paths.csv \
             --output /data/index.parquet
```

The `--input` CSV must have a `path` column with absolute paths to `.nii` or
`.nii.gz` files. Val and test each get 5% of the volumes, capped at 500, and
train gets the rest; change this in the auto-generated `index_config.json`
sidecar. Add an optional `group` column (e.g. a subject ID) to keep all of a
patient's volumes in the same split.

### Stage 2 — Pretraining (`misfit_train`)

Trains a masked autoencoder. At each step, 75% of the volume's patches are
masked and the model reconstructs them from visible context. Supports
single-GPU, multi-GPU, and multi-node training out of the box.

```console
# Single GPU
misfit_train --index   /data/index.parquet \
             --results /runs/exp1

# 4 GPUs — batch size scales automatically
torchrun --nproc_per_node=4 -m misfit.cli.train_entrypoint \
    --index   /data/index.parquet \
    --results /runs/exp1

# Resume an interrupted run
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --resume
```

Key options:

| Flag                     | Default                 | Description                                        |
| ------------------------ | ----------------------- | -------------------------------------------------- |
| `--model`                | `swinunetr-base`        | `swinunetr-*`, `resenc`, or `primus-*` (see below) |
| `--patch-size X Y Z`     | `96 96 96`              | Spatial crop size (must be divisible by 32)        |
| `--target-spacing X Y Z` | off (native)            | Resample volumes to this spacing (e.g. `1 1 1`)    |
| `--epochs`               | `1000`                  | Training epochs of `--steps-per-epoch` steps each  |
| `--steps-per-epoch`      | `250`                   | Optimizer steps per epoch                          |
| `--batch-size`           | `2`                     | Per-GPU batch size                                 |
| `--loss`                 | `normalized_masked_mse` | Loss function                                      |
| `--normalization-policy` | `clip_and_z_score`      | Intensity normalization (`z_score_only` for nnssl) |

Training writes a `config.json` to `--results` that captures every architecture
and hyperparameter decision. All downstream commands read this file — you never
have to re-specify model flags.

### Stage 3 — Evaluation (`misfit_evaluate`) and Inspection (`misfit_inspect`)

`misfit_evaluate` computes reconstruction metrics (MAE, MSE, PSNR, SSIM) and
writes a per-volume CSV. It defaults to the `val` split; pass `--split test` to
evaluate on the test set, or `--split ""` for all rows.

```console
misfit_evaluate --checkpoint /runs/exp1/models/best_model.pt \
                --index      /data/index.parquet \
                --config     /runs/exp1/config.json \
                --output-csv /runs/exp1/eval_results.csv

# Evaluate on the test split
misfit_evaluate --checkpoint /runs/exp1/models/best_model.pt \
                --index      /data/index.parquet \
                --config     /runs/exp1/config.json \
                --output-csv /runs/exp1/test_results.csv \
                --split      test
```

`misfit_inspect` reconstructs volumes and saves outputs under two subdirectories
— `reconstructions/` (denormalized NIfTIs) and `masks/` (binary masks,
1=masked/reconstructed, 0=visible). Load both in ITK-SNAP or 3D Slicer and
overlay the mask (1=reconstructed, 0=visible) to highlight exactly which regions
the model had to fill in from context. Defaults to all rows; use `--split val`
to restrict to the validation set.

```console
misfit_inspect --checkpoint /runs/exp1/models/best_model.pt \
               --index      /data/index.parquet \
               --config     /runs/exp1/config.json \
               --output-dir /runs/exp1/inspect

# Inspect only the validation split
misfit_inspect --checkpoint /runs/exp1/models/best_model.pt \
               --index      /data/index.parquet \
               --config     /runs/exp1/config.json \
               --output-dir /runs/exp1/inspect \
               --split      val
```

Output structure:

```
inspect/
    reconstructions/   <volume_id>.nii.gz   — full-volume reconstruction
    masks/             <volume_id>.nii.gz   — 1=masked (reconstructed), 0=visible
```

### Stage 4 — Encoding & Embedding (`misfit_encode`, `misfit_embed` + `misfit_embed_train`)

`misfit_encode` caches the **full spatial bottleneck feature map**
`(N_crops, C, D', H', W')` for every crop to disk — useful for fast aggregator
training without re-running the encoder.

```console
misfit_encode --encoder-checkpoint /runs/exp1/models/best_model.pt \
              --index               /data/index.parquet \
              --config              /runs/exp1/config.json \
              --output-dir          /data/encodings
```

`misfit_embed` runs the full encode-and-aggregate pipeline end-to-end, saving a
single global `(C,)` embedding vector per volume. With `--aggregator mean_pool`
(default), no aggregator training is needed — embeddings are ready immediately
for zero-shot retrieval or UMAP visualization. Use `--split` to restrict to a
specific split.

```console
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index               /data/index.parquet \
             --config              /runs/exp1/config.json \
             --output-dir          /data/embeddings

# Embed only the test split
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index               /data/index.parquet \
             --config              /runs/exp1/config.json \
             --output-dir          /data/embeddings \
             --split               test
```

`misfit_embed_train` fine-tunes a lightweight aggregator on the **cached
features from `misfit_encode`** for a downstream task. Supports `classification`
(cross-entropy) and `contrastive` (Supervised Contrastive) objectives.

```console
misfit_embed_train --input      /data/train_manifest.csv \
                   --output-dir /runs/agg \
                   --embed-dim  768
```

The input CSV has four columns: `volume_id`, `split`, `features_path`, `label`.
`features_path` points to the `.npz` files produced by `misfit_encode`. Only
`split='train'` rows are used for training.

---

## Model Variants

| Variant         | `--model`         | Parameters            | Recommended for                        |
| --------------- | ----------------- | --------------------- | -------------------------------------- |
| SwinUNETR Small | `swinunetr-small` | ~7M (4.8M encoder)    | Rapid prototyping, small datasets      |
| SwinUNETR Base  | `swinunetr-base`  | ~28M (18.6M encoder)  | Standard pretraining (default)         |
| SwinUNETR Large | `swinunetr-large` | ~110M (73.9M encoder) | Large-scale datasets, maximum capacity |
| ResEnc          | `resenc`          | ~102M (90M encoder)¹  | nnU-Net/nnssl-style CNN pretraining    |
| Primus Medium   | `primus-medium`   | ~166M (146M encoder)² | nnssl-style transformer pretraining    |
| Primus Large    | `primus-large`    | ~381M (324M encoder)² | Larger Primus                          |
| PrimusX         | `primus-x`        | ~649M (539M encoder)² | The nnFoundationViT network            |

¹ At a patch size of 128³ or larger. ResEnc plans its depth from `--patch-size`;
see
[Model Variants](https://misfit-medical.readthedocs.io/en/latest/advanced_topics/#model-variants).

² At a patch size of 96³; the position embeddings grow with `--patch-size`.

Every architecture is built from MIST, so `--model` names match the MIST model
each encoder fine-tunes into.

`resenc` and `primus-x` are the same networks as the openly released
[nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) and
[nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT): their
weights work directly in every MISFIT command, as a baseline encoder or as the
starting point for `--init-weights`. See
[Starting from nnFoundation](https://misfit-medical.readthedocs.io/en/latest/advanced_topics/#starting-from-nnfoundation).

---

## Output Structure

```text
results/
    checkpoints/
        checkpoint.pt       Latest checkpoint (each epoch, and every 30 min)
    models/
        best_model.pt       Checkpoint with the lowest validation loss
    logs/                   TensorBoard event files
    config.json             Reproducibility record — architecture, patch size,
                            all hyperparameters, and MISFIT version
```

---

## Reproducibility

`config.json` is the single source of truth for model architecture. Every
downstream command (`misfit_evaluate`, `misfit_inspect`, `misfit_embed`)
requires it via `--config` rather than re-accepting architecture flags. This
prevents silent mismatches between training and inference.

When resuming with `--resume`, MISFIT validates that the model name and patch
size are unchanged (hard error if not). Changes to other hyperparameters emit a
warning but are allowed.
