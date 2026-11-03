"""The dataloader factories forward --target-spacing to MISFITDataset."""
import json

import numpy as np
import pandas as pd
import pytest

from misfit.data_loading.dataloader import get_training_dataloader, get_validation_dataloader


@pytest.fixture
def index_path(tmp_path):
    row = {
        "volume_id": "vol", "path": str(tmp_path / "vol.nii.gz"),
        "shape_d": 32, "shape_h": 32, "shape_w": 32,
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": 0, "fg_x_end": 31, "fg_y_start": 0, "fg_y_end": 31,
        "fg_z_start": 0, "fg_z_end": 31,
        "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
    }
    df = pd.DataFrame([{**row, "split": "train"}, {**row, "split": "val"}])
    p = tmp_path / "index.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.mark.parametrize("factory", [get_training_dataloader, get_validation_dataloader])
@pytest.mark.parametrize("spacing", [None, (1.0, 1.0, 2.5)])
def test_dataloader_forwards_target_spacing(index_path, factory, spacing):
    loader = factory(index_path, patch_size=(32, 32, 32), num_workers=0,
                     target_spacing=spacing)
    assert loader.dataset.target_spacing == spacing
