Getting Started
===

### System Requirements

MISFIT requires at least one NVIDIA GPU. Multi-GPU and multi-node training are
supported via `torchrun`. For large datasets a high-core-count CPU is recommended
for the indexing step.

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
index records the path, intensity statistics, voxel spacing, and affine transform
for every volume in your dataset. These statistics are computed once and reused
throughout the rest of the pipeline.

```console
misfit_index --data-dir /path/to/niftis \
             --output /path/to/index.parquet
```

Alternatively, if your files span multiple directories, provide a CSV manifest
with a `path` column:

```console
misfit_index --manifest /path/to/paths.csv \
             --output /path/to/index.parquet
```

It is good practice to build separate train and validation indexes:

```console
misfit_index --data-dir /data/train --output /data/train.parquet
misfit_index --data-dir /data/val   --output /data/val.parquet
```

Once your indexes are built, you are ready to start pretraining. See
[Usage](usage.md) for details on all available commands.
