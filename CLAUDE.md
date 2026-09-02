# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

MISFIT (Medical Imaging Semantic Foundation Toolkit) trains 3D medical imaging
foundation models using masked autoencoders (MAE). It reads unlabeled NIfTI
files and produces a pretrained SwinUNETR-V2 encoder that transfers directly to
MIST for segmentation fine-tuning. No labels required.

## Running tests

Always use the `mist` mamba environment — MISFIT and MIST are both installed
editably there:

```bash
mamba run -n mist pytest
```

Never use plain `pytest` or `python -m pytest`.

## Formatting

Python: `ruff check` (lint + import order). All Markdown: Prettier
(`proseWrap: always`, 80 cols) — run `npm run format` / `npm run format:check`
after `npm install`. The Prettier tooling (`package.json`, `.prettierrc.json`,
`node_modules/`) is git-ignored; it is not part of the Python package. Each
mkdocs admonition in `docs/` is preceded by a `<!-- prettier-ignore -->` comment
so Prettier leaves the `!!! note` block and its indented body intact — keep that
comment when adding a new admonition.

## Pipeline overview

```
misfit_index  →  misfit_train  →  misfit_evaluate / misfit_inspect
                                        ↓
                               misfit_encode (raw spatial features)
                                        ↓
                            misfit_embed_train (optional aggregator)
                                        ↓
                               misfit_embed (global vector)
```

### CLI entry points (pyproject.toml)

| Command              | Module                          | Purpose                                              |
| -------------------- | ------------------------------- | ---------------------------------------------------- |
| `misfit_index`       | `cli/index_entrypoint.py`       | Build Parquet index from CSV of NIfTI paths          |
| `misfit_train`       | `cli/train_entrypoint.py`       | MAE pretraining (single-GPU to multi-node)           |
| `misfit_evaluate`    | `cli/evaluate_entrypoint.py`    | Reconstruction metrics → CSV                         |
| `misfit_inspect`     | `cli/inspect_entrypoint.py`     | Full-volume reconstruction → NIfTI                   |
| `misfit_encode`      | `cli/encode_entrypoint.py`      | Raw spatial features (N_crops, C, D', H', W')        |
| `misfit_embed`       | `cli/embed_entrypoint.py`       | Global embedding vector (C,) per volume              |
| `misfit_embed_train` | `cli/embed_train_entrypoint.py` | Train crop aggregator (classification / contrastive) |

All argument parsing lives in `cli/args.py`. The `ArgParser` subclass adds
`.arg()` and `.flag()` shorthands. `add_*_args` functions are shared across the
individual entrypoints.

## Module map

```
misfit/
  cli/                  Entry points + shared ArgParser / add_*_args
  preprocessing/        NIfTI indexer → Parquet (parallel, ProcessPoolExecutor)
  data_loading/         MISFITDataset + DataLoader; on-the-fly clip+z-score normalization
  models/               MISFITModel base class; SwinMAE (SwinUNETR-V2 + MAE head)
  training/             MAETrainer; optimizer/LR-scheduler registries; training_utils
  loss_functions/       ReconstructionLoss base; masked_mse, masked_l1, normalized_mse
  metrics/              ReconstructionMetric base; masked_mae/mse/psnr (defaults) + ssim (opt-in)
  evaluation/           ReconstructionEvaluator; tiled full-volume inference + CSV output
  inference/            InferenceRunners; tiled reconstruct pipeline (pad→tile→stitch)
  embedding/            Embedder; EmbedTrainer; aggregators (mean_pool, attention_pool);
                        objectives (classification, contrastive)
  utils/                console (Rich), io (read/write JSON), progress_bar,
                        hardware (bf16_supported / resolve_amp / autocast_context),
                        normalization (normalize_patchwise / denormalize_patchwise)
```

## Key design decisions

### Registry pattern

Models, losses, metrics, aggregators, and objectives all use a registry
(`@register_*` decorator, `get_*` lookup, `list_*` for CLI choices).
Registrations are triggered by importing the module (e.g.,
`import misfit.loss_functions`). New implementations only need to inherit the
base class and apply the decorator.

### Normalization

Per-volume clip to [p1, p99] then z-score with foreground mean/std — computed
once at index time, stored in the Parquet index, applied on the fly at load
time. This handles CT and MRI in the same batch without dataset-level
statistics.

### `normalized_masked_mse` loss (default)

Normalizes the target within each `mask_patch_size` cube (zero mean, unit
variance) before computing MSE on masked voxels. This equalizes loss scale
across CT (HU values, large range) and MRI (arbitrary units), enabling
mixed-modality pretraining in one run. The per-cube transform lives in
`misfit.utils.normalization.normalize_patchwise` (with its inverse
`denormalize_patchwise`) — the loss, `misfit_evaluate`, and `misfit_inspect` all
call it, so metrics/reconstructions are in the space the model was optimized in.

### Evaluation metric space (`misfit_evaluate`)

Metrics are computed on masked voxels in the **training loss's space**
(per-`mask_patch_size`-cube normalized target for `normalized_masked_mse`), so
`masked_mse` is directly comparable to `best_val_loss`. The evaluator crops each
volume to its foreground bbox (matching `MISFITDataset`) and draws deterministic
masks from `--seed`. Every metric is also reported for a naive baseline (impute
masked voxels with the visible-region mean) with a `_skill` column (`<metric>` /
`<metric>_naive` / `<metric>_skill` in the CSV); positive skill = beats trivial.
Defaults: `masked_mae`, `masked_mse`, `masked_psnr` (`DEFAULT_METRICS`). `ssim`
is registered but opt-in via `--metrics ssim` — it has no consistent space here;
use `misfit_inspect` for a viewer-space read.

### Model architecture (`models/swinunetr/misfit_swinunetr_mae.py`)

`SwinMAE(MISFITModel)` wraps `SwinUNETR.swinViT` as `self.encoder` (UNet decoder
discarded). The MAE decoder is a lightweight `ConvTranspose3d` stack
(`MAEDecoder`). Masking is applied at the image level before encoding
(SimMIM-style) — required because Swin windowed attention breaks with irregular
token counts.

Constraints:

- All `img_size` dimensions must be divisible by 32 (SwinUNETR-V2 downsamples
  32×).
- All `img_size` dimensions must be divisible by `mask_patch_size` (default 16).
- Default patch size for training: `96×96×96`.

Model sizes (`feature_size`): small=24, base=48 (default), large=96.

### `config.json` as source of truth

`misfit_train` writes `config.json` to the results directory. All downstream
commands (`misfit_evaluate`, `misfit_inspect`, `misfit_encode`, `misfit_embed`)
require `--config` and load architecture from it — no re-specifying model flags.
Resume validation checks model name and patch size (hard error on mismatch).

### Distributed training

`MAETrainer` reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from torchrun environment
variables. The same class runs on 1 GPU or N×M GPUs (NCCL), and also on CPU /
multi-process CPU (gloo) for testing — `self.use_cuda` gates device placement,
backend, and cuDNN tuning; CPU pretraining is very slow and warns once.

AMP is BF16-only — there is no fp16 path and no GradScaler, since BF16 has
float32's dynamic range. It is _requested_ on by default, then resolved against
the actual hardware by `misfit.utils.hardware.resolve_amp`: BF16 needs an
Ampere+ GPU (compute capability ≥ 8.0), so pre-Ampere GPUs (V100, T4) and CPU
fall back to FP32 with a warning. `MAETrainer.train()` resolves once (from the
CLI default, or from the saved config on `--resume`) and persists the effective
value to `config.json`; `misfit_evaluate` and `misfit_inspect` re-resolve the
config value against their own hardware. Disable AMP entirely by setting
`"amp": false` in `config.json`. Shared helper:
`hardware.autocast_context(enabled)`.

### Transfer learning to MIST

`get_encoder_state_dict()` remaps keys `encoder.<name>` → `model.swinViT.<name>`
to match MIST's `MistSwinUNETR` checkpoint format. Its output is saved to
`models/encoder_weights.pt` whenever validation loss improves (alongside
`best_model.pt`), so the MIST-ready encoder is always available after training.
Pass it to `mist_train --pretrained-weights models/encoder_weights.pt` along
with `--pretrained-config <misfit results>/config.json` (MIST uses the source
config to validate encoder compatibility; omitting it only warns and skips that
check). Channel mismatch (MISFIT single-channel → MIST multi-channel) is handled
by MIST's `--input-channel-strategy` (default: average).

## Adding new components

**New loss:** subclass `ReconstructionLoss`, apply `@register_loss(name="...")`,
place under `loss_functions/reconstruction/`. Import in
`loss_functions/__init__.py`.

**New metric:** subclass `ReconstructionMetric`, apply
`@register_metric(name="...")`, place under `metrics/`. Import in
`metrics/__init__.py` (or `metrics_registry.py`).

**New model:** subclass `MISFITModel`, implement `get_encoder_state_dict()`,
apply `@register_model(name="...")`, place under `models/<name>/`. Import in
`models/__init__.py`.

**New aggregator:** subclass `AbstractAggregator`, apply
`@register_aggregator(name="...")`, place under `embedding/aggregators/`. Import
in `embedding/aggregators/__init__.py`.

## Output structure (misfit_train)

```
results/
    checkpoints/checkpoint.pt    Latest checkpoint (overwritten each epoch)
    models/best_model.pt         Lowest validation loss
    models/encoder_weights.pt    Encoder-only weights remapped for MIST (model.swinViT.*); refreshed with best_model.pt
    logs/                        TensorBoard event files
    config.json                  Architecture + hyperparameters (required by downstream commands)
```

## Planned next features (not yet implemented)

- `misfit_anomaly` — reconstruction error as anomaly score; zero-label anomaly
  detection
- `misfit_search` — FAISS-backed nearest-neighbor retrieval over embedding
  corpus
- `misfit_visualize` — UMAP of embedding space, attention weight heatmaps
