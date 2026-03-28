Usage
===

## Overview

MISFIT is a **command-line tool** for pretraining and deploying 3D medical
imaging foundation models. The core pipeline consists of four stages:

1. **Indexing** — Scans a manifest of NIfTI files, computes per-volume intensity
statistics and spacing, assigns train/val/test splits, and produces a Parquet
index that all downstream commands consume.

2. **Pretraining** — Trains a SwinUNETR masked autoencoder (MAE) on the indexed
dataset. The encoder learns to reconstruct randomly masked patches of each
volume, producing generalizable semantic representations without any labels.

3. **Evaluation / Inspection** — Commands for measuring reconstruction quality
on the validation set and generating full-volume reconstruction NIfTIs for
visual inspection.

4. **Embedding** — Extracts per-volume feature vectors from the pretrained
encoder for downstream tasks such as classification, retrieval, and anomaly
detection. A lightweight aggregator can be fine-tuned on labeled data with
`misfit_embed_train`.

---

## Indexing

The **indexing step** scans your NIfTI files and records the path, intensity
statistics (p1, p99, foreground mean, foreground std), voxel spacing, image
shape, and affine transform for each volume. It also assigns each volume a
`split` label (`train`, `val`, or `test`) using configurable ratios. The output
is a single Parquet file consumed by all downstream MISFIT commands.

Run indexing with `misfit_index`:

- `--input FILE` (**required**): CSV or Parquet file with a `path` column
  listing absolute paths to NIfTI files.
- `--output PARQUET` (**required**): Destination path for the output Parquet
  index.
- `--num-workers-index N`: Number of parallel worker processes. *(default: 32)*

### Split configuration

On first run, `misfit_index` writes a companion `<output_stem>_config.json`
file alongside the Parquet index. This file records the split ratios and
random seed used to assign the `split` column. Edit it and re-run
`misfit_index` to change the split proportions.

### Example

Index from a CSV manifest.

```console
misfit_index --input  /data/paths.csv \
             --output /data/index.parquet
```

Index with 64 parallel workers.

```console
misfit_index --input             /data/paths.csv \
             --output            /data/index.parquet \
             --num-workers-index 64
```

### Output

A single Parquet file at the path given by `--output`. Each row corresponds to
one NIfTI volume and contains the following columns:

| Column | Description |
|---|---|
| `volume_id` | Unique identifier derived from the filename (no extension). |
| `path` | Absolute path to the NIfTI file. |
| `split` | Dataset split: `train`, `val`, or `test`. |
| `shape_d` / `shape_h` / `shape_w` | Voxel dimensions (depth, height, width). |
| `spacing_d` / `spacing_h` / `spacing_w` | Voxel spacing in mm. |
| `affine` | JSON-encoded 4×4 affine transform matrix. |
| `fg_x_start` / `fg_x_end` | Foreground bounding box extent along x. |
| `fg_y_start` / `fg_y_end` | Foreground bounding box extent along y. |
| `fg_z_start` / `fg_z_end` | Foreground bounding box extent along z. |
| `p1` | 1st-percentile foreground intensity (lower clip bound for normalization). |
| `p99` | 99th-percentile foreground intensity (upper clip bound). |
| `fg_mean` | Foreground mean intensity after clipping. |
| `fg_std` | Foreground standard deviation after clipping. |

---

## Training

The **training step** pretrains a SwinUNETR masked autoencoder on the indexed
dataset. At each iteration a random 75% of patch tokens are masked, and the
model learns to reconstruct the missing voxels from the visible context.
Train/val split is determined by the `split` column in the index.

Run training with `misfit_train`:

### Data

- `--index PARQUET` (**required**): Parquet index produced by `misfit_index`.
  Rows with `split='train'` are used for training; rows with `split='val'` for
  validation.
- `--num-cpu-workers N`: CPU worker processes for data loading. *(default: 8)*

### Output

- `--results DIR` (**required**): Directory for checkpoints, `config.json`, and
  TensorBoard logs.

### Model

- `--model NAME`: SwinMAE variant to train. *(default: `swinmae-base`)*
  See [Model Variants](advanced_topics.md#model-variants) for details.
- `--patch-size D H W`: Spatial crop size fed to the model in voxels. Must be
  divisible by 32. *(default: `96 96 96`)*
- `--mask-patch-size P`: Edge length of each masked 3D cube in voxels.
  *(default: 16)*
- `--mask-ratio R`: Fraction of patch tokens to mask. *(default: 0.75)*

### Loss

- `--loss NAME`: Reconstruction loss function. *(default: `normalized_masked_mse`)*
  See [Loss Functions](advanced_topics.md#loss-functions) for available options.

### Optimisation

- `--epochs N`: Total training epochs. *(default: 200)*
- `--batch-size N`: Batch size per GPU. *(default: 2)*
- `--optimizer NAME`: Optimizer. *(default: `adamw`)*
- `--learning-rate LR`: Initial learning rate. *(default: 1e-4)*
- `--weight-decay WD`: L2 weight decay. *(default: 0.05)*
- `--lr-scheduler NAME`: Learning rate schedule. *(default: `cosine`)*
- `--warmup-epochs N`: Linear warmup epochs before cosine decay. *(default: 20)*

### Miscellaneous

- `--seed N`: Random seed for reproducibility. *(default: 42)*
- `--resume`: Resume from the latest checkpoint in `--results`. The model
  architecture and patch size must match the saved `config.json`; changes to
  other hyperparameters emit warnings but are allowed.
- `--overwrite`: Discard any existing checkpoint and `config.json` in
  `--results` and start fresh.

!!!note
    `--resume` and `--overwrite` are mutually exclusive. If neither is passed and
    a `config.json` already exists in `--results`, `misfit_train` will exit with
    an error rather than silently overwriting your run.

### Example

Train a base SwinMAE for 200 epochs.

```console
misfit_train --index   /data/index.parquet \
             --results /runs/exp1
```

Train a small model on 4 GPUs using `torchrun`.

```console
torchrun --nproc_per_node=4 \
    $(which misfit_train) \
        --index   /data/index.parquet \
        --results /runs/exp1 \
        --model   swinmae-small
```

Resume a run that was interrupted.

```console
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --resume
```

### Output

```text
results/
    checkpoints/
        checkpoint.pt       Latest checkpoint (overwritten each epoch).
    models/
        best_model.pt       Checkpoint with the lowest validation loss.
    logs/                   TensorBoard event files.
    config.json             Reproducibility config (architecture, patch size,
                            hyperparameters, and MISFIT version).
```

`config.json` is the **single source of truth** for model architecture. It is
required by `misfit_evaluate`, `misfit_inspect`, and `misfit_embed` to
reconstruct the correct model without re-specifying any flags.

---

## Evaluation

The **evaluation step** measures reconstruction quality on the validation split.
For each volume, the evaluator tiles the full volume into non-overlapping
patches, runs a complete MAE forward pass on each patch (with a fresh random
mask), computes masked reconstruction metrics per patch, and averages the
results across all patches to produce one score per volume.

!!!note
    Metrics are computed on the **masked patches only** — the voxels the encoder
    never saw. This directly measures the MAE pretraining objective.

Run evaluation with `misfit_evaluate`:

- `--checkpoint PT` (**required**): Path to a checkpoint produced by
  `misfit_train`.
- `--index PARQUET` (**required**): Parquet index. Only rows with `split='val'`
  are evaluated.
- `--config JSON` (**required**): Path to the `config.json` produced by
  `misfit_train`. Model architecture and metrics to compute are read from this
  file.
- `--output-csv CSV` (**required**): Path where the evaluation results CSV will
  be written.
- `--device DEVICE`: Torch device (e.g. `cuda:0`, `cpu`). *(default: auto)*

### Example

Evaluate a checkpoint.

```console
misfit_evaluate --checkpoint  /runs/exp1/models/best_model.pt \
                --index       /data/index.parquet \
                --config      /runs/exp1/config.json \
                --output-csv  /runs/exp1/eval_results.csv
```

### Output

A single CSV file at the path given by `--output-csv`. Each row is one volume.
Five summary rows are appended at the bottom of the file.

| `volume_id` | `masked_mae` | `masked_mse` | `masked_psnr` | `ssim` |
|---|---|---|---|---|
| CT_001 | 0.312 | 0.187 | 23.4 | 0.821 |
| CT_002 | 0.298 | 0.163 | 24.1 | 0.845 |
| ... | | | | |
| Mean | 0.317 | 0.190 | 23.4 | 0.821 |
| Std | 0.019 | 0.025 | 0.7 | 0.020 |
| 25th Percentile | 0.302 | 0.171 | 22.9 | 0.808 |
| Median | 0.316 | 0.188 | 23.5 | 0.822 |
| 75th Percentile | 0.331 | 0.207 | 23.9 | 0.834 |

All metrics operate on z-score normalised intensities (not original HU or signal units).

| Metric | Description | Direction |
|---|---|---|
| `masked_mae` | Mean absolute error on masked voxels. | Lower is better |
| `masked_mse` | Mean squared error on masked voxels. | Lower is better |
| `masked_psnr` | Peak signal-to-noise ratio on masked voxels (dB). | Higher is better |
| `ssim` | Structural similarity over the full patch (range −1 to 1). | Higher is better |

---

## Inspection

The **inspection step** produces a full-volume reconstruction NIfTI for every
volume in an index. This is the primary tool for visually assessing pretraining
quality — open the input and output side-by-side in a viewer such as ITK-SNAP
or 3D Slicer to see where reconstruction succeeds and fails.

The reconstruction pipeline:

1. Loads and z-score normalises the volume.
2. Zero-pads to the nearest multiple of the patch size in every dimension.
3. Tiles the padded volume into non-overlapping patches and runs MAE
   reconstruction on each.
4. Stitches the reconstructed patches back into the full padded volume.
5. Trims padding to restore the original voxel dimensions.
6. Denormalises intensities back to the original intensity space
   (`reconstruction × fg_std + fg_mean`).
7. Saves a NIfTI file using the original affine transform from the index.

Run inspection with `misfit_inspect`:

- `--checkpoint PT` (**required**): Path to a pretrained MISFIT checkpoint.
- `--index PARQUET` (**required**): Parquet index of volumes to reconstruct.
- `--config JSON` (**required**): Path to the `config.json` produced by
  `misfit_train`. Model architecture and patch size are read from this file.
- `--output-dir DIR` (**required**): Directory where `<volume_id>.nii.gz` files
  are written.
- `--device DEVICE`: Torch device. *(default: auto)*

### Example

Reconstruct all volumes in the index.

```console
misfit_inspect --checkpoint  /runs/exp1/models/best_model.pt \
               --index       /data/index.parquet \
               --config      /runs/exp1/config.json \
               --output-dir  /runs/exp1/reconstructions
```

!!!note
    The output NIfTIs are in the **original coordinate space** with
    **denormalized intensities** so they can be directly compared to the input
    volumes in any NIfTI viewer.

### Output

One NIfTI file per volume at `<output-dir>/<volume_id>.nii.gz`, in the original
image space with denormalized intensities.

---

## Embedding

The **embedding step** uses the pretrained encoder to extract per-crop feature
vectors for every volume in an index. These embeddings can be used directly for
zero-shot retrieval (using `mean_pool` aggregation) or fine-tuned for a
downstream task with `misfit_embed_train`.

Run embedding extraction with `misfit_embed`:

- `--encoder-checkpoint PT` (**required**): Path to a pretrained MISFIT encoder
  checkpoint.
- `--index PARQUET` (**required**): Parquet index of volumes to embed.
- `--config JSON` (**required**): Path to the `config.json` produced by
  `misfit_train`. Model architecture and patch size are read from this file.
- `--output-dir DIR` (**required**): Directory where `.npz` files are saved.
- `--aggregator NAME`: Aggregation strategy for combining patch-level features
  into a single volume-level embedding. *(default: `mean_pool`)*
  Options: `mean_pool`, `attention_pool`.
- `--aggregator-checkpoint PT`: Path to a trained aggregator checkpoint produced
  by `misfit_embed_train`. Required when `--aggregator attention_pool`.
- `--device DEVICE`: Torch device. *(default: auto)*

### Example

Extract mean-pooled embeddings (zero-shot, no aggregator training required).

```console
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index              /data/index.parquet \
             --config             /runs/exp1/config.json \
             --output-dir         /data/embeddings
```

Extract embeddings with a trained attention-pooling aggregator.

```console
misfit_embed --encoder-checkpoint  /runs/exp1/models/best_model.pt \
             --index               /data/index.parquet \
             --config              /runs/exp1/config.json \
             --output-dir          /data/embeddings \
             --aggregator          attention_pool \
             --aggregator-checkpoint /runs/agg/aggregator.pt
```

### Output

One `.npz` file per volume at `<output-dir>/<volume_id>.npz`, containing:

| Key | Shape | Description |
|---|---|---|
| `features` | `(N_crops, C)` | Per-crop encoder feature vectors. |
| `positions` | `(N_crops, 3)` | Voxel-space centre coordinates of each crop. |

---

## Embedding Training

The **embedding training step** fine-tunes a lightweight aggregator head on top
of frozen embeddings extracted by `misfit_embed`. The aggregator learns to pool
the per-crop feature vectors into a single discriminative volume-level embedding.

Run aggregator training with `misfit_embed_train`:

### Input

- `--input CSV` (**required**): Unified CSV with columns:

  | Column | Description |
  |---|---|
  | `volume_id` | Volume identifier. |
  | `split` | Dataset split. Only rows where `split='train'` are used. |
  | `features_path` | Absolute path to the `.npz` file produced by `misfit_embed`. |
  | `label` | String label for the training objective. |

  Labels are always treated as strings. Integer or boolean labels should be
  converted to strings before passing. The mapping from label strings to
  integer indices is saved in the output `aggregator.pt` checkpoint for
  inference-time decoding.

### Output

- `--output-dir DIR` (**required**): Directory for the trained `aggregator.pt`
  and training logs.

### Model

- `--aggregator NAME`: Aggregator architecture. *(default: `attention_pool`)*
- `--objective NAME`: Training objective. *(default: `classification`)*
  Options: `classification` (cross-entropy), `contrastive` (Supervised
  Contrastive with K=2 pairs per group).
- `--embed-dim C` (**required**): Dimensionality of the encoder bottleneck
  features. Must match the feature files from `misfit_embed`.
- `--no-position-encoding`: Disable learned 3D position encoding in
  `AttentionPoolAggregator`. *(default: off)*

### Training

- `--epochs N`: Training epochs. *(default: 50)*
- `--batch-size N`: Batch size. For contrastive training this must be even.
  *(default: 32)*
- `--learning-rate LR`: Initial learning rate. *(default: 1e-3)*
- `--num-workers-embed N`: DataLoader worker processes. *(default: 4)*
- `--device DEVICE`: Torch device. *(default: `cuda`)*

### Example

Train an attention-pooling aggregator for classification.

```console
misfit_embed_train --input      /data/train_manifest.csv \
                   --output-dir /runs/agg \
                   --embed-dim  768
```

Train with a contrastive objective.

```console
misfit_embed_train --input      /data/train_manifest.csv \
                   --output-dir /runs/agg \
                   --embed-dim  768 \
                   --objective  contrastive
```

### Output

```text
output-dir/
    aggregator.pt   Trained aggregator weights and label_to_idx mapping.
                    Pass to misfit_embed via --aggregator-checkpoint to
                    use at inference time.
```
