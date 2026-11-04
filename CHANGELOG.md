# Changelog

All notable changes to MISFIT are listed here, newest first. Releases before
0.2.0-alpha are described on the
[GitHub releases page](https://github.com/mist-medical/MISFIT/releases).

## 0.2.0-alpha — 2026-11-03

MISFIT now builds every encoder from MIST, adds a CNN, and trains in fixed-size
steps so long runs on large corpora are practical. Requires
`mist-medical>=2.4.0`.

### Added

- **ResEnc** (`--model resenc`) — MIST's Residual Encoder UNet pretrained as a
  masked autoencoder the way nnssl does it: dense masking, then the full UNet
  with skip connections reconstructs the image. Its encoder transfers to MIST's
  `resenc` with no key remapping.
- **nnFoundation and other nnssl checkpoints** — every command that takes a
  checkpoint (`misfit_evaluate`, `misfit_inspect`, `misfit_encode`,
  `misfit_embed`) accepts nnFoundationCNN's `checkpoint_final.pth` directly, and
  `misfit_train --init-weights` starts pretraining from it (or from any MISFIT
  checkpoint). Loading is strict, so a mismatched architecture stops with an
  error instead of mixing pretrained and random weights.
- **Normalization policies** (`--normalization-policy`) — `clip_and_z_score`
  (the default and previous behavior) or `z_score_only`, which reproduces
  nnssl's preprocessing for nnFoundation weights. The policy is saved to
  `config.json` and applied by every command.
- **Target spacing** (`--target-spacing X Y Z`) — optionally resample every
  volume before cropping patches. Off by default (native resolution).
- **Grouped splits** — an optional `group` column in the `misfit_index` input
  (e.g. subject ID) keeps all of a group's volumes in the same split.
- **Training controls** — `--steps-per-epoch`, `--val-every`, `--val-volumes`,
  and `--checkpoint-minutes`; per-step TensorBoard logging of loss, learning
  rate, and gradient norm, plus per-epoch throughput.

### Changed

- **Step-based training.** An epoch is now a fixed number of optimizer steps
  (default 1000 epochs × 250 steps), not a pass over the data, so validation,
  checkpoints, and the learning-rate schedule no longer scale with corpus size.
  The scheduler steps every optimizer step, and warmup defaults to 50 epochs.
- **Fixed validation.** Validation scores the same masks on the same subset of
  validation volumes every time (the first 128 by default), so the curve is
  comparable across epochs and runs.
- **Mid-epoch checkpoints.** A resumable checkpoint is also saved at most every
  30 minutes by default, and `--resume` continues mid-epoch.
- **Smaller held-out splits.** `misfit_index` now gives val and test 5% each,
  capped at 500 volumes (previously 10% each, uncapped). Split files from older
  versions keep their original meaning.
- **SwinUNETR encoder built from MIST.** `SwinMAE` uses MIST's own SwinUNETR-V2
  backbone, so the pretrained encoder is exactly the module MIST fine-tunes. Its
  spacing embedding was removed; older checkpoints still load.
- `config.json` gains MIST's `spatial_config` section, so
  `mist_train --pretrained-config` reads a MISFIT config directly.
- `--patch-size` is labeled `X Y Z`, matching MIST.
- American spelling throughout (`normalize`, `optimize`, `initialize`).

### Breaking

- Runs started before step-based training cannot be resumed; start a new run.
- Resuming now also rejects a changed `--target-spacing` or
  `--normalization-policy`.
- New indexes use the smaller default val/test splits.

### Fixed

- `misfit_train` no longer hangs at exit after training finishes.
