Getting Started
===

### System Requirements

MISFIT requires at least one NVIDIA **Ampere or newer GPU** (A100, H100, RTX
30xx+) for training. BF16 automatic mixed precision is always enabled during
training and is not supported on older architectures (Volta, Turing). Multi-GPU
and multi-node training are supported via `torchrun`. For large datasets a
high-core-count CPU is recommended for the indexing step.

### Install

To install the latest release of MISFIT, use

```console
pip install misfit-medical
```

To install from source and customize the underlying code (e.g., to add a new
model or loss function), clone the repository and install in editable mode:

```console
git clone https://github.com/mist-medical/MISFIT.git
cd MISFIT
pip install -e .
```

### Data Format

MISFIT works with 3D NIfTI files (`.nii` or `.nii.gz`). There are no
restrictions on how your files are organized on disk — a single flat directory,
per-patient subdirectories, or any other layout all work equally well.

```console
data/
    patient_001.nii.gz
    patient_002.nii.gz
    ...
    patient_N.nii.gz
```

or

```console
data/
    patient_001/
        ct.nii.gz
    patient_002/
        ct.nii.gz
    ...
```

!!!note
    MISFIT is a **single-channel** framework. Each NIfTI file must contain one
    3D volume. If your file has a fourth dimension (e.g., a time series), only
    the first volume is used.

### Building an Index

Before training, you must build a **Parquet index** with `misfit_index`. The
index records the path, intensity statistics, voxel spacing, and affine
transform for every volume, and assigns each volume to a `train`, `val`, or
`test` split. These statistics are computed once and reused throughout the rest
of the pipeline.

`misfit_index` accepts a CSV or Parquet file with a `path` column listing the
absolute paths to your NIfTI files:

```console
misfit_index --input  /path/to/paths.csv \
             --output /path/to/index.parquet
```

On first run, a companion split-config file is written alongside the index
(e.g. `index_config.json`). It records the train/val/test ratios and random
seed used to assign the `split` column. Edit it and re-run `misfit_index` to
change the proportions.

Once your index is built, you are ready to start pretraining. See
[Usage](usage.md) for details on all available commands.
