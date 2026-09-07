Getting Started
===

### System Requirements

A **GPU with BF16 matrix hardware** is recommended for training — it enables
BF16 automatic mixed precision. That means an NVIDIA **Ampere or newer GPU**
(A100, H100, RTX 30xx+) or an AMD **CDNA / RDNA3+ GPU** (MI200/MI300 series, RX
7000 series and newer). On pre-Ampere NVIDIA architectures (Volta, Turing), on
older AMD GPUs (RDNA1/2 — RX 5000/6000 series), and on CPU-only machines, MISFIT
automatically falls back to FP32 with a warning; CPU training runs but is only
practical for tests and small debugging runs. Multi-GPU and multi-node training
are supported via `torchrun` (NCCL/RCCL on GPU, gloo on CPU). For large datasets
a high-core-count CPU is recommended for the indexing step.

### Install

From PyPI:

```console
pip install misfit-medical
```

Or use the Docker image (CUDA 12.8, torch 2.9.1):

```console
docker pull mistmedical/misfit:latest
```

Clone and install in editable mode instead if you want to customize the
underlying code (e.g., to add a new model or loss function):

```console
git clone https://github.com/mist-medical/MISFIT.git
cd MISFIT
pip install -e .
```

#### AMD ROCm GPU

PyPI's default `torch` wheel has no ROCm support (and the Docker image is
CUDA-only), so install a ROCm-enabled PyTorch build _first_ — matching the ROCm
version on your machine, which you can check with `cat /opt/rocm/.info/version`
— then install MISFIT on top of it. No install extra is needed; ROCm reuses the
same code paths as CUDA (`torch.cuda` is a compatibility shim on ROCm builds):

```console
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install misfit-medical
```

Swap `rocm6.4` for whichever ROCm release matches your driver. To confirm the
ROCm build of PyTorch is the one that got installed:

```console
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

You want a version string ending in `+rocmX.Y`, a non-`None`
`torch.version.hip`, and `True` — that combination is what MISFIT's hardware
detection checks for. BF16 AMP is then enabled automatically on CDNA
(MI100/200/300) and RDNA3+ (RX 7000+) cards, and skipped with a warning on
RDNA1/2.

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

<!-- prettier-ignore -->
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

On first run, a companion split-config file is written alongside the index (e.g.
`index_config.json`). It records the train/val/test ratios and random seed used
to assign the `split` column. Edit it and re-run `misfit_index` to change the
proportions.

Once your index is built, you are ready to start pretraining. See
[Usage](usage.md) for details on all available commands.
