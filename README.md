# MISFIT
### Medical Imaging Semantic Foundation Toolkit

[![PyPI version](https://img.shields.io/pypi/v/misfit-medical.svg)](https://pypi.org/project/misfit-medical/)
[![Python](https://img.shields.io/pypi/pyversions/misfit-medical.svg)](https://pypi.org/project/misfit-medical/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)]()

MISFIT is a simple, scalable, end-to-end framework for pretraining 3D medical imaging foundation models using masked autoencoders (MAE). Give it a directory of unlabeled NIfTI files and it produces a pretrained encoder — no labels required.

Developed in collaboration with **Rice University** and **The University of Texas MD Anderson Cancer Center**.

---

## What it does

MISFIT trains a [SwinUNETR](https://arxiv.org/abs/2201.01266)-based masked autoencoder on 3D medical images. At each training step, 75% of the image patches are randomly hidden, and the model learns to reconstruct them from the remaining context. This forces the encoder to build rich spatial representations of anatomy — representations that transfer well to downstream tasks like classification, segmentation, and retrieval.

```
Unlabeled NIfTIs  →  misfit_index  →  misfit_train  →  Pretrained Encoder
                                                               ↓
                                              misfit_embed  →  Volume Embeddings
                                                               ↓
                                        misfit_embed_train  →  Classifier / Retrieval
```

---

## Key Features

- **No labels required** — pretrains entirely on unlabeled NIfTI volumes
- **End-to-end pipeline** — six CLI commands take you from raw files to downstream-ready embeddings
- **3D-native** — operates on full volumetric data, not 2D slices
- **Mixed modality** — the `normalized_masked_mse` loss normalizes per-patch variance, handling CT and MRI in the same training run
- **Scalable** — single-GPU to multi-node training via `torchrun`; the same command runs everywhere
- **BF16 / FP16 AMP** — `--amp-dtype bf16` for Ampere+ GPUs (A100, H100, RTX 30xx+); `fp16` for all CUDA GPUs
- **Reproducible** — `config.json` captures every architecture and training hyperparameter; downstream commands require it rather than re-accepting flags
- **100% test coverage**

---

## Installation

```console
pip install misfit-medical
```

To install from source:

```console
git clone https://github.com/mist-medical/MISFIT.git
cd MISFIT
pip install -e .
```

**Requirements:** Python ≥ 3.10, at least one NVIDIA GPU.

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

Scans your NIfTI files in parallel and computes per-volume intensity statistics (p1, p99, foreground mean/std, bounding box) and voxel spacing. Assigns each volume to a `train`, `val`, or `test` split. The resulting Parquet index is the single input to all downstream commands.

```console
misfit_index --input  /data/paths.csv \
             --output /data/index.parquet
```

The `--input` CSV must have a `path` column with absolute paths to `.nii` or `.nii.gz` files. Split ratios (default 80/10/10) are controlled via the auto-generated `index_config.json` sidecar.

### Stage 2 — Pretraining (`misfit_train`)

Trains a SwinUNETR masked autoencoder. At each step, 75% of patch tokens are masked and the model reconstructs them from visible context. Supports single-GPU, multi-GPU, and multi-node training out of the box.

```console
# Single GPU
misfit_train --index   /data/index.parquet \
             --results /runs/exp1

# 4 GPUs — batch size scales automatically
torchrun --nproc_per_node=4 $(which misfit_train) \
    --index   /data/index.parquet \
    --results /runs/exp1

# Resume an interrupted run
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --resume
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `swinmae-base` | `swinmae-small` / `swinmae-base` / `swinmae-large` |
| `--patch-size D H W` | `96 96 96` | Spatial crop size (must be divisible by 32) |
| `--epochs` | `200` | Total training epochs |
| `--batch-size` | `2` | Per-GPU batch size |
| `--amp-dtype` | `fp16` | `fp16` (all GPUs) or `bf16` (Ampere+, more stable) |
| `--loss` | `normalized_masked_mse` | Loss function |

Training writes a `config.json` to `--results` that captures every architecture and hyperparameter decision. All downstream commands read this file — you never have to re-specify model flags.

### Stage 3 — Evaluation (`misfit_evaluate`) and Inspection (`misfit_inspect`)

`misfit_evaluate` computes reconstruction metrics (MAE, MSE, PSNR, SSIM) on the validation split and writes a per-volume CSV.

```console
misfit_evaluate --checkpoint /runs/exp1/models/best_model.pt \
                --index      /data/index.parquet \
                --config     /runs/exp1/config.json \
                --output-csv /runs/exp1/eval_results.csv
```

`misfit_inspect` reconstructs every volume in the index and saves the output as NIfTI files — open them side-by-side with the originals in ITK-SNAP or 3D Slicer to visually assess pretraining quality.

```console
misfit_inspect --checkpoint /runs/exp1/models/best_model.pt \
               --index      /data/index.parquet \
               --config     /runs/exp1/config.json \
               --output-dir /runs/exp1/reconstructions
```

### Stage 4 — Embedding (`misfit_embed` + `misfit_embed_train`)

`misfit_embed` tiles each volume into non-overlapping crops, encodes each crop with the pretrained encoder, and saves per-crop features as `.npz` files. With `--aggregator mean_pool` (default), no additional training is needed — embeddings are ready for zero-shot retrieval or UMAP visualization.

```console
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index               /data/index.parquet \
             --config              /runs/exp1/config.json \
             --output-dir          /data/embeddings
```

`misfit_embed_train` fine-tunes a lightweight aggregator head on top of the frozen embeddings for a downstream task. Supports `classification` (cross-entropy) and `contrastive` (Supervised Contrastive) objectives.

```console
misfit_embed_train --input      /data/train_manifest.csv \
                   --output-dir /runs/agg \
                   --embed-dim  768
```

The input CSV has four columns: `volume_id`, `split`, `features_path`, `label`. Only `split='train'` rows are used for training.

---

## Model Variants

| Variant | `--model` | Parameters | Recommended for |
|---------|-----------|------------|-----------------|
| Small | `swinmae-small` | ~14M | Rapid prototyping, small datasets |
| Base | `swinmae-base` | ~55M | Standard pretraining (default) |
| Large | `swinmae-large` | ~210M | Large-scale datasets, maximum capacity |

---

## Output Structure

```text
results/
    checkpoints/
        checkpoint.pt       Latest checkpoint (overwritten each epoch)
    models/
        best_model.pt       Checkpoint with the lowest validation loss
    logs/                   TensorBoard event files
    config.json             Reproducibility record — architecture, patch size,
                            all hyperparameters, and MISFIT version
```

---

## Reproducibility

`config.json` is the single source of truth for model architecture. Every downstream command (`misfit_evaluate`, `misfit_inspect`, `misfit_embed`) requires it via `--config` rather than re-accepting architecture flags. This prevents silent mismatches between training and inference.

When resuming with `--resume`, MISFIT validates that the model name and patch size are unchanged (hard error if not). Changes to other hyperparameters emit a warning but are allowed.

---

## Citation

If you use MISFIT in your research, please cite:

```bibtex
@software{misfit2026,
  title  = {{MISFIT}: Medical Imaging Semantic Foundation Toolkit},
  author = {Celaya, Adrian and Fuentes, David and Riviere, Beatrice},
  year   = {2026},
  url    = {https://github.com/mist-medical/MISFIT}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
