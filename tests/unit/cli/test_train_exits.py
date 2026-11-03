"""Regression test: misfit_train must exit once training completes.

On macOS (spawn start method + PyTorch's file_system sharing strategy), a
DataLoader worker's first shared-memory batch launches a torch_shm_manager
daemon that used to inherit the resource tracker's pipe, so the
``ResourceTracker.__del__`` waitpid at interpreter exit blocked forever after
"Training complete." (see ``seal_resource_tracker_fd``). The run must use real
worker processes (``--num-cpu-workers`` >= 1) to exercise that path.
"""
import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pandas as pd


def _write_index(tmp_path):
    rows = []
    for i, split in enumerate(["train", "train", "val"]):
        data = np.random.randn(32, 32, 32).astype(np.float32)
        nifti_path = tmp_path / f"vol{i}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(nifti_path))
        rows.append({
            "volume_id": f"vol{i}",
            "path": str(nifti_path),
            "split": split,
            "shape_d": 32, "shape_h": 32, "shape_w": 32,
            "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
            "affine": json.dumps(np.eye(4).tolist()),
            "fg_x_start": 0, "fg_x_end": 31,
            "fg_y_start": 0, "fg_y_end": 31,
            "fg_z_start": 0, "fg_z_end": 31,
            "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
        })
    index_path = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(index_path, index=False)
    return index_path


def test_misfit_train_exits_after_training_with_workers(tmp_path):
    """misfit_train with DataLoader workers exits instead of hanging."""
    index_path = _write_index(tmp_path)
    results = tmp_path / "results"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "misfit.cli.train_entrypoint",
            "--index", str(index_path),
            "--results", str(results),
            "--model", "swinunetr-small",
            "--patch-size", "32", "32", "32",
            "--epochs", "1",
            "--batch-size", "2",
            "--num-cpu-workers", "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        raise AssertionError(
            "misfit_train did not exit after training:\n" + output
        ) from None
    assert proc.returncode == 0, output
    assert "Training complete." in output
    assert (results / "models" / "best_model.pt").exists()
