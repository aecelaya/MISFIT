# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

MISFIT (Medical Imaging Semantic Foundation Toolkit) trains 3D medical imaging
foundation models using masked autoencoders (MAE). It reads unlabeled NIfTI
files and produces a pretrained encoder (SwinUNETR-V2 or ResEnc, built from
MIST's own models) that transfers directly to MIST for segmentation fine-tuning.
No labels required.

## Running tests

Always use the `mist` mamba environment — MISFIT and MIST are both installed
editably there:

```bash
mamba run -n mist pytest
```

Never use plain `pytest` or `python -m pytest`.

## Continuous integration

Two workflows run on every push and PR to `main` (and via `workflow_dispatch`):

- **`test.yml`** — the pytest suite on Python 3.10 / 3.11 / 3.12, gated at
  `--cov-fail-under=100` (the suite is 100%-covered; a regression fails the
  build). A separate `distributed` leg runs the gloo DDP tests
  (`pytest -o addopts= -m distributed`), which the default run skips. On push to
  `main` the 3.12 leg regenerates `coverage.svg` (the README badge) and commits
  it back with `[skip ci]`.
- **`lint.yml`** — `ruff check .` (pinned to the `mist` env's ruff) and
  `codespell` (config in `[tool.codespell]`). `ruff format` is intentionally not
  enforced.

`coverage.svg` is committed to the repo (not git-ignored, unlike `coverage.xml`
/ `.coverage*`). Add domain abbreviations that trip codespell to
`ignore-words-list` in `pyproject.toml` rather than scattering inline ignores.

## Formatting

Python: `ruff check` (lint + import order). All Markdown: Prettier
(`proseWrap: always`, 80 cols) — run `npm run format` / `npm run format:check`
after `npm install`. The Prettier tooling (`package.json`, `.prettierrc.json`,
`node_modules/`) is git-ignored; it is not part of the Python package. Each
mkdocs admonition in `docs/` is preceded by a `<!-- prettier-ignore -->` comment
so Prettier leaves the `!!! note` block and its indented body intact — keep that
comment when adding a new admonition.

## Releasing

`misfit-medical` (PyPI) and `mistmedical/misfit` (Docker Hub) publish on a
GitHub Release, mirroring MIST:

1. Bump the version in **both** `pyproject.toml` and `misfit/__init__.py` (they
   must agree — `tests/unit/test_packaging.py` enforces it; `_build_config`
   writes `misfit.__version__` into `config.json`, the Docker workflow reads
   `pyproject.toml`).
2. Merge to `main` with `test.yml` + `lint.yml` green (they run on the PR, but
   no workflow _gates_ the Release itself — the tag is what ships). Run `pytest`
   locally first regardless.
3. Create a GitHub Release tagged `v<version>` targeting `main`.
   - `.github/workflows/python-publish.yml` builds the sdist+wheel and uploads
     to PyPI using the `PYPI_API_TOKEN` repo secret.
   - `.github/workflows/docker-publish.yml` then fires (`workflow_run`),
     rebuilds from `./Dockerfile` (which `pip install`s the just-published PyPI
     package), and pushes `mistmedical/misfit:<version>` + `:latest`. Needs repo
     secrets `DOCKER_USERNAME` / `DOCKER_PASSWORD`.

The `-alpha`/`-beta`/`-rc` suffix in `pyproject.toml` is transformed to the PEP
440 form (`a0`/`b0`/`rc0`) for the Docker tag.

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

Every `*_entrypoint.py` ends with `if __name__ == "__main__": <x>_entry()` so it
is runnable as `python -m misfit.cli.<x>_entrypoint` — this is the form
`torchrun -m misfit.cli.train_entrypoint` and shell-less container/Kubernetes
manifests use instead of `$(which misfit_train)`. `tests/unit/test_packaging.py`
guards both the guard block and the `-m` path.

## Module map

```
misfit/
  cli/                  Entry points + shared ArgParser / add_*_args
  preprocessing/        NIfTI indexer → Parquet (parallel, ProcessPoolExecutor)
  data_loading/         MISFITDataset + DataLoader; on-the-fly normalization (policy);
                        EpochSampler (fixed samples per epoch for step-based training)
  models/               MISFITModel base class; SwinMAE (MIST SwinUNETR-V2 + MAE head);
                        ResEncMAE (MIST ResEnc UNet as the MAE network);
                        masking (shared image-level patch masking)
  training/             MAETrainer; optimizer/LR-scheduler registries; training_utils
  loss_functions/       ReconstructionLoss base; masked_mse, masked_l1, normalized_mse
  metrics/              ReconstructionMetric base; masked_mae/mse/psnr (defaults) + ssim (opt-in)
  evaluation/           ReconstructionEvaluator; tiled full-volume inference + CSV output
  inference/            InferenceRunners; tiled reconstruct pipeline (pad→tile→stitch)
  embedding/            Embedder; EmbedTrainer; aggregators (mean_pool, attention_pool);
                        objectives (classification, contrastive)
  normalization/        NormalizationPolicy registry: clip_and_z_score (default), z_score_only
  utils/                console (Rich), io (read/write JSON),
                        progress_bar (get_progress_bar; TrainProgressBar /
                        ValidationProgressBar — mirror mist.utils.progress_bar),
                        hardware (get_accelerator_type / bf16_supported /
                        resolve_amp / autocast_context),
                        normalization (normalize_patchwise / denormalize_patchwise),
                        resampling (--target-spacing helpers)
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

### Index splits (`preprocessing/indexer.py`)

Val and test default to 5% of the corpus each, capped at 500 volumes
(`DEFAULT_SPLIT_CONFIG`); train takes the rest. Validation only tracks
reconstruction loss and picks `best_model.pt`, and real encoder evaluation is
downstream, so held-out sets shouldn't grow with the corpus — 80/10/10 withheld
20k volumes from a 100k pretraining run. Integer sizes are exact counts and are
not capped. A legacy config (has a `train` key) keeps its old ratio meaning.

An optional `group` column in the `--input` file (e.g. subject ID) keeps all of
a group's volumes in one split: `assign_splits` shuffles groups and deals them
out whole (val, then test, then train), so a split may overshoot its target by
less than one group. Without groups, every row is its own group. Records are
sorted back into input order after the parallel stats pass, since `as_completed`
order would otherwise make the seeded split vary run to run. `split_targets` is
called before the stats pass so a bad config fails immediately rather than hours
into indexing.

### Load-failure handling (`MISFITDataset.__getitem__`)

Every path in the index was already confirmed loadable by `misfit_index` (bad
files are dropped there, not passed through), so a load failure inside
`MISFITDataset.__getitem__` means something changed since indexing — deleted,
moved, a transient filesystem error. `Dataset.__getitem__` has no way to "skip"
an index (unlike `misfit_evaluate`/`misfit_encode`/`misfit_embed`, which warn
and `continue`), so an isolated failure substitutes a zero volume and **warns**
(`UserWarning`, matches the existing 4D-volume-fallback pattern) rather than
failing silently — a bare `except Exception: return zeros` with no warning at
all was the original implementation, and it's a real footgun: a systemic problem
(a mount gone away, permissions revoked) would silently train on an ever-growing
fraction of zero-filled "volumes" with zero visibility. `max_load_failures`
_consecutive_ failures (no successful load in between; counter lives on the
`Dataset` instance, which `persistent_workers=True` keeps alive for the whole
run) raises `RuntimeError` instead, so a systemic issue fails the run loudly
rather than being tolerated indefinitely. Defaults to 3 — small enough to catch
a sustained problem quickly, large enough not to trip over one-off, isolated bad
files spread across a long run (the counter resets to 0 on every successful
load).

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

### Model architectures (`models/swinunetr/`, `models/resenc/`)

MISFIT depends on MIST (`mist-medical>=2.4.0` in `pyproject.toml`; ResEnc first
shipped in 2.4.0) for model architectures: encoders are built from
`mist.models`, never reimplemented, so a pretrained encoder is the exact module
MIST fine-tunes. MISFIT adds only the MAE-specific parts (masking, decoder,
masked loss).

`SwinMAE(MISFITModel)` builds MIST's `MistSwinUNETR` and keeps its
`.model.swinViT` as `self.encoder` (UNet decoder discarded). The MAE decoder is
a lightweight `ConvTranspose3d` stack (`MAEDecoder`). Masking is applied at the
image level before encoding (SimMIM-style) — required because Swin windowed
attention breaks with irregular token counts. The masking lives in
`models/masking.py` (`generate_patch_mask`, `apply_mask`), not in `SwinMAE`, so
CNN backbones (which have no tokens to drop) reuse the same scheme;
`SwinMAE.generate_mask` delegates to it with identical RNG calls, so seeded
masks (e.g. `misfit_evaluate --seed`) are unchanged.

Constraints:

- All `img_size` dimensions must be divisible by 32 (SwinUNETR-V2 downsamples
  32×).
- All `img_size` dimensions must be divisible by `mask_patch_size` (default 16).
- Default patch size for training: `96×96×96`.

Model sizes (`feature_size`): small=24, base=48 (default), large=96.

Every model's contract with the rest of the pipeline is the same, so the
trainer, evaluator, inference, and encode/embed code are architecture-agnostic:
`forward(x, mask_ratio=None)` returns `{"reconstruction", "mask"}` in both train
and eval mode, `model.encoder(x)[-1]` is the bottleneck (`misfit_encode` /
`misfit_embed`), and every parameter must receive a gradient (DDP runs with
`find_unused_parameters=False`; `test_ddp_synchronizes_gradients_on_window_end`
is parametrized over architectures and asserts it).

`ResEncMAE(MISFITModel)` (`--model resenc`, the same name as MIST's model) is
MIST's full `ResEncUNet` used as the MAE network, the nnssl recipe: dense
masking, then encoder + UNet decoder with skips; masked regions are blank at
every level, so the skips can't leak the target. `out_channels=in_channels`
turns the decoder's full-resolution seg layer into the reconstruction head
(`self.head`); the lower-resolution seg layers (deep supervision only) are
deleted because they'd never get gradients. `self.encoder` is MIST's
`ResidualEncoder`, so `get_encoder_state_dict()` returns MIST's `encoder.*` keys
unchanged (no remap). Topology (stages/strides/kernels) comes from
`get_unet_params(img_size, spacing)`: the run's `--target-spacing` when set,
otherwise `ISOTROPIC_PLANNING_SPACING = (1, 1, 1)` (native, mixed-resolution
data has no single spacing; only axis ratios matter, so this just means
"isotropic topology"). Every builder (trainer, evaluator,
`build_model_from_checkpoint`) passes `target_spacing` through the registry;
SwinMAE ignores it via `**kwargs`. Isotropically, the patch size sets depth: 96³
→ 5 stages (~63M params), ≥128³ → 6 stages (~102M; nnssl's default MAE trainer
uses 160³, nnFoundationCNN used 192³). A MIST `resenc` planned with more stages
or anisotropic kernels still loads every matching tensor; MIST skips the rest
and warns. Same `img_size` constraints as SwinMAE (÷32, ÷`mask_patch_size`).

### `config.json` as source of truth

`misfit_train` writes `config.json` to the results directory. All downstream
commands (`misfit_evaluate`, `misfit_inspect`, `misfit_encode`, `misfit_embed`)
require `--config` and load architecture from it — no re-specifying model flags.
Resume validation checks model name, patch size, mask patch size, and target
spacing (hard error on mismatch), and rejects configs without
`training.steps_per_epoch` (pre-step-based runs, whose scheduler state counted
epochs). The target spacing is compared even when one side is `None` (native ↔
resampled is a change), and a config without `spatial_config`
(pre-`--target-spacing`) counts as native.

`spatial_config: {patch_size, target_spacing}` duplicates `model.patch_size` on
purpose: it is MIST's own section/key layout, so MIST's
`validate_encoder_compatibility` reads a MISFIT config passed as
`mist_train --pretrained-config` directly. Keep `model.patch_size` too (older
configs and every MISFIT reader use it). MIST (≥ 2.4.0) treats a null
`target_spacing`, or a missing `spatial_config`, as "pretrained without
resampling": it warns that spacing can't be checked instead of raising.

### Step-based training (`mae_trainer.py`, `data_loading/samplers.py`)

An epoch is `--steps-per-epoch` optimizer steps (default 1000 × 250), not a pass
over the data: with 100k+ volumes a data pass takes hours, and validation,
checkpoints, logging, and the LR schedule would all run that rarely and scale
with corpus size. `EpochSampler` draws exactly `steps × accum × batch_size`
samples per rank per epoch from a seeded stream of back-to-back permutations
(seeded by `seed + epoch`, identical on every rank; each rank takes a strided,
disjoint slice). Being deterministic, `set_epoch(epoch, start=k)` skips the
samples a preempted epoch already consumed, so mid-epoch resume continues the
same sequence. The scheduler is stepped per optimizer step
(`total_steps = epochs × steps_per_epoch`, warmup = `warmup_epochs × steps`).

Validation runs every `--val-every` epochs and after the last, on the first
`--val-volumes` val rows (`Subset`), inside `_fixed_validation_rng`
(`fork_rng` + reseed): every pass scores the same masks on the same volumes, so
the curve is comparable, and the training RNG is restored afterwards.

Checkpoints store `(epoch, step_in_epoch)`: `(e + 1, 0)` at epoch end, `(e, s)`
mid-epoch. Rank 0 alone saves mid-epoch when `--checkpoint-minutes` have passed
(no collective needed — other ranks just continue). TensorBoard scalars are all
logged against `global_step`: per step `loss/train_step`, `lr` (the LR the step
used, read before `scheduler.step()`), `grad_norm` (pre-clip total norm from
`clip_grad_norm_`); per epoch `loss/train`, `loss/val`,
`throughput/{steps,samples}_per_sec` (samples counts all ranks).

Optimizer defaults are the same for every model (AdamW 1e-4, wd 0.05, cosine, 50
warmup epochs). nnssl's ResEnc recipe uses SGD (Nesterov 0.99, 1e-2, PolyLR),
but its paper (Wald et al. 2025, "Revisiting MAE pre-training") takes that from
nnU-Net without ablating optimizers, so it isn't evidence SGD beats AdamW; one
shared recipe also keeps architecture comparisons clean.

### Normalization policies (`normalization/`, `--normalization-policy`)

Registry like losses/optimizers:
`NormalizationPolicy.normalize(volume, row) -> (normalized, IntensityScale(shift, scale))`.
Every input path calls the run's policy: `MISFITDataset._normalize`,
`inference_utils.load_and_normalize(row, normalization_policy, target_spacing, return_scale)`
(encode, embed, inspect), and `ReconstructionEvaluator._load_and_normalize`.
Inspect inverts the returned `IntensityScale` (`recon × scale + shift`), so a
new policy needs no inspect changes. Stored as
`config["data"]["normalization_policy"]`; `get_normalization_policy_name`
defaults missing entries to `clip_and_z_score` (pre-option configs), and resume
rejects a change.

`z_score_only` reproduces nnssl: z-score over the bounding box of nonzero voxels
(nnssl fills holes first, which can't change the box), no clipping,
`use_mask_for_norm` false. Stats are computed at load time from the volume, so
no index columns. For nnFoundation also use `--loss masked_mse` (nnssl's
`MAEMSELoss`): `normalized_masked_mse` standardizes each target cube, which
nnFoundation's head never learned — with it, `misfit_evaluate` reports
nnFoundation as worse than the naive baseline.

### Pretrained weights (`models/pretrained.py`, `--init-weights`)

`load_model_weights` is the one loader for full-network weights: the trainer's
`--init-weights` and every inference path (`build_model_from_checkpoint`,
`ReconstructionEvaluator`) use it, so all commands accept a MISFIT checkpoint
(`"model"`), an nnssl checkpoint (`"network_weights"`, e.g. nnFoundationCNN), or
a plain state dict. Each model maps foreign names in
`adapt_pretrained_state_dict` (ResEncMAE: last `decoder.seg_layers.N` → `head`,
drop `decoder.encoder.*` aliases and deep-supervision layers; SwinMAE: drop the
legacy `spacing_embed.*`). Loading is strict — only `mask_token` may be absent
(nnssl masks with zeros, the token's init) — because a partial load would
silently mix pretrained and random weights.

nnFoundation facts (verified from the checkpoints' `nnssl_adaptation_plan`,
nnssl's trainer source, and Harsy et al. 2026): MAE (16³ blocks, 75% masked,
zero-filled, `MAEMSELoss`), 192³ patches, 2.1M CT/MRI/PET volumes, nnssl's
`noresample` config (native resolution, median spacing 1 mm; NOT resampled to 1
mm), crop to nonzero (holes filled), z-score without mask or clipping. So the
closest MISFIT match is
`--model resenc --patch-size 192 192 192 --normalization-policy z_score_only --loss masked_mse`
with NO `--target-spacing` (native, isotropic planning = nnFoundationCNN's
topology); `--target-spacing 1 1 1` builds the same network but resamples.

nnFoundationCNN equals `resenc` with an isotropic topology and a patch ≥128³;
verified against nnU-Net's `ResidualEncoderUNet` with the real checkpoint (max
output difference 0). MIST pins the same layout with a name/shape fingerprint
test (`test_resenc_layout_matches_nnfoundation_cnn_checkpoint`).

### Target spacing (`--target-spacing X Y Z`, `utils/resampling.py`)

Off by default (native resolution, the original behavior). When set, every path
from NIfTI to model input resamples with the same helpers in
`misfit.utils.resampling` (trilinear `F.interpolate`, shape =
`round(n * spacing / target)` per axis): `MISFITDataset` (normalize → fg crop →
resample → pad/crop/augment, so only the foreground is interpolated), the
evaluator (same order), and `inference_utils.load_and_normalize` (no fg crop)
for inspect/encode/embed. Spacing comes from the index's `spacing_d/h/w` columns
(x, y, z — nibabel and MIST's transposed SimpleITK arrays share (x, y, z) order,
which is why `--patch-size` is labelled `X Y Z` like MIST's). Downstream
commands read it with `get_target_spacing(config)`. `misfit_inspect` resamples
reconstructions and masks back to the native grid (re-binarising the mask at
0.5) so outputs keep the input affine.

### Distributed training

`MAETrainer` reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from torchrun environment
variables. The same class runs on 1 GPU or N×M GPUs (NCCL — RCCL on AMD ROCm,
same backend name), and also on CPU / multi-process CPU (gloo) for testing —
`self.use_cuda` gates device placement, backend, and cuDNN tuning; CPU
pretraining is very slow and warns once. ROCm needs no special-casing: PyTorch's
ROCm build reuses the `torch.cuda` namespace and the `cuda:<rank>` device string
as a shim, so `use_cuda` is `True` and every CUDA code path already works.

`_setup_distributed` calls `torch.cuda.set_device(local_rank)` **before**
`dist.init_process_group`, and passes
`device_id=torch.device("cuda", local_rank)`. Both are load-bearing on
multi-GPU: a NCCL group created while every rank is still on the default
`cuda:0` binds DDP's construction-time param-shape allgather to device 0 on all
ranks, so the first collective hangs (`rank 0 has inconsistent 0 params`, 10-min
watchdog timeout). Guarded by
`test_setup_distributed_sets_cuda_device_before_init_process_group`.

AMP is BF16-only — there is no fp16 path and no GradScaler, since BF16 has
float32's dynamic range. It is _requested_ on by default, then resolved against
the actual hardware by `misfit.utils.hardware.resolve_amp` → `bf16_supported()`,
which branches on `get_accelerator_type()` (cuda / rocm / cpu, via
`torch.version.hip`):

- **CUDA:** compute capability ≥ 8.0 (Ampere+: A100, H100, RTX 30xx+). Uses the
  capability, not `torch.cuda.is_bf16_supported()`, which reports True on
  V100/T4 via emulation.
- **ROCm:** `gcnArchName` against `_ROCM_BF16_ACCELERATED_ARCHES` — an
  allow-list of CDNA (MFMA) and RDNA3+ (WMMA) arches. The capability check is
  useless here (ROCm reports ≥ (9,0) for every AMD GPU) and
  `is_bf16_supported()` lies the same way it does on pre-Ampere NVIDIA — it
  returns True on RDNA1/2 (gfx103x), which runs BF16 on shader ALUs with no
  speedup. Unrecognized arch → unsupported (conservative default).
- **CPU:** always False.

`MAETrainer.train()` resolves once (from `--no-amp`/the CLI default, or from the
saved config on `--resume` — `--no-amp` is not consulted there, the saved value
wins) and persists the effective value to `config.json`; `misfit_evaluate` and
`misfit_inspect` re-resolve the config value against their own hardware. Shared
helper: `hardware.autocast_context(enabled)` (device type `"cuda"` for both CUDA
and ROCm, `"cpu"` otherwise).

`--init-only` writes `config.json` + the `--results` skeleton
(`checkpoints/`/`models/`/`logs/`) and returns before `_setup_distributed`,
model, or data loader construction — `_build_config()` only needs `self.args` +
the already-resolved `self.amp`, so nothing else is required. Shared between
that path and the real run by `_write_results_skeleton()`. Combine with
`--no-amp` to get an AMP-off config with no run/kill/edit/`--resume` dance;
combine with `--resume` to just recreate the skeleton dirs (existing
`config.json` is left alone, same as a real resume).

### Training progress bar

`utils.progress_bar.TrainProgressBar` / `ValidationProgressBar` mirror
`mist.utils.progress_bar`'s classes (same columns, same `loss: ` / `lr: `
formatting) so `misfit_train` output looks like MIST's — no `fold` argument,
since MISFIT pretraining has no cross-validation folds. `TrainProgressBar`
diverges from MIST in one way: `update(loss=None, lr=None)` — both optional,
where MIST's are always passed — because gradient accumulation means a
cross-rank-aggregated loss only exists at a window end (`is_window_end` in
`train()`); mid-window micro-steps call `update()` with no arguments to advance
the bar without touching the displayed text (Rich's `Progress.update()` leaves
omitted fields at their last value). At `accum_steps=1` every step is a window
end, so the displayed loss is identical to MIST's: a running mean of the
all-reduced per-step loss (`train_meter.value`), refreshed every step.

### Transfer learning to MIST

`get_encoder_state_dict()` returns keys in the matching MIST model's format:
`SwinMAE` remaps `encoder.<name>` → `model.swinViT.<name>` (MIST's
`MistSwinUNETR`), and `ResEncMAE` keeps `encoder.<name>` as is (MIST's
`ResEncUNet`). Its output is saved to `models/encoder_weights.pt` whenever
validation loss improves (alongside `best_model.pt`), so the MIST-ready encoder
is always available after training. Pass it to
`mist_train --pretrained-weights models/encoder_weights.pt` along with
`--pretrained-config <misfit results>/config.json` (MIST uses the source config
to validate encoder compatibility; omitting it only warns and skips that check).
Channel mismatch (MISFIT single-channel → MIST multi-channel) is handled by
MIST's `--input-channel-strategy` (default: average).
`test_misfit_encoder_matches_mist_encoder_exactly` (every `swinunetr-*` variant)
and `test_misfit_resenc_encoder_matches_mist_resenc_exactly` guard this: keys
and shapes must equal MIST's own `get_encoder_state_dict()`, and MIST's
`load_pretrained_encoder` must transfer every tensor with nothing skipped.

## Adding new components

**New loss:** subclass `ReconstructionLoss`, apply `@register_loss(name="...")`,
place under `loss_functions/reconstruction/`. Import in
`loss_functions/__init__.py`.

**New metric:** subclass `ReconstructionMetric`, apply
`@register_metric(name="...")`, place under `metrics/`. Import in
`metrics/__init__.py` (or `metrics_registry.py`).

**New model:** build the network from `mist.models` (don't reimplement it),
subclass `MISFITModel`, implement `get_encoder_state_dict()` in the MIST model's
key format, apply `@register_model(name="<MIST model name>")`, place under
`models/<name>/`. Import in `models/__init__.py`. Reuse `models/masking.py`,
follow the forward/encoder contract above, and add an exact key/shape
compatibility test against the MIST model plus an entry in the parametrized DDP
test.

**New aggregator:** subclass `AbstractAggregator`, apply
`@register_aggregator(name="...")`, place under `embedding/aggregators/`. Import
in `embedding/aggregators/__init__.py`.

## Output structure (misfit_train)

```
results/
    checkpoints/checkpoint.pt    Latest checkpoint (overwritten each epoch)
    models/best_model.pt         Lowest validation loss
    models/encoder_weights.pt    Encoder-only weights in the MIST model's key format; refreshed with best_model.pt
    logs/                        TensorBoard event files
    config.json                  Architecture + hyperparameters (required by downstream commands)
```

## Planned next features (not yet implemented)

- `misfit_anomaly` — reconstruction error as anomaly score; zero-label anomaly
  detection
- `misfit_search` — FAISS-backed nearest-neighbor retrieval over embedding
  corpus
- `misfit_visualize` — UMAP of embedding space, attention weight heatmaps
