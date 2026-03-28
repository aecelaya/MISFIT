"""Tests for misfit.embedding.objectives.contrastive."""
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from misfit.embedding.objectives.contrastive import (
    ContrastiveObjective,
    GroupedBatchSampler,
    SupervisedContrastiveLoss,
    _L2NormLayer,
)


# ---------------------------------------------------------------------------
# GroupedBatchSampler
# ---------------------------------------------------------------------------

def test_grouped_batch_sampler_batch_size():
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    sampler = GroupedBatchSampler(groups, batch_size=4, drop_last=True)
    torch.manual_seed(0)
    batches = list(sampler)
    for batch in batches:
        assert len(batch) == 4  # pairs_per_batch=2 → 2*2=4


def test_grouped_batch_sampler_remainder_not_dropped():
    # 3 groups, batch_size=4 → pairs_per_batch=2
    # First 2 groups → 1 batch; 1 group left
    groups = [[0, 1], [2, 3], [4, 5]]
    sampler = GroupedBatchSampler(groups, batch_size=4, drop_last=False)
    torch.manual_seed(0)
    batches = list(sampler)
    # 1 full batch (4) + 1 remainder batch (2)
    assert len(batches) == 2
    assert len(batches[-1]) == 2


def test_grouped_batch_sampler_drop_last():
    groups = [[0, 1], [2, 3], [4, 5]]
    sampler = GroupedBatchSampler(groups, batch_size=4, drop_last=True)
    torch.manual_seed(0)
    batches = list(sampler)
    assert len(batches) == 1


def test_grouped_batch_sampler_len():
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    sampler = GroupedBatchSampler(groups, batch_size=4, drop_last=True)
    assert len(sampler) == 2


def test_grouped_batch_sampler_batch_size_too_small():
    with pytest.raises(ValueError, match="batch_size must be"):
        GroupedBatchSampler([[0, 1]], batch_size=1)


# ---------------------------------------------------------------------------
# SupervisedContrastiveLoss
# ---------------------------------------------------------------------------

def test_supcon_loss_scalar():
    loss_fn = SupervisedContrastiveLoss()
    projections = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(projections, labels)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_supcon_loss_no_positives():
    """Each sample has a unique label → no positives → differentiable zero."""
    loss_fn = SupervisedContrastiveLoss()
    projections = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
    labels = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(projections, labels)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ContrastiveObjective
# ---------------------------------------------------------------------------

@pytest.fixture
def cont_obj():
    return ContrastiveObjective()


def test_contrastive_validate_labels_basic(cont_obj):
    df = pd.DataFrame({
        "volume_id": [f"v{i}" for i in range(6)],
        "group": ["a", "a", "b", "b", "c", "c"],
    })
    cleaned = cont_obj.validate_labels(df, "group")
    assert len(cleaned) == 6
    assert len(cont_obj.label_to_idx) == 3


def test_contrastive_validate_labels_drops_small_groups(cont_obj):
    # 'a' is a singleton (1 sample), 'b' and 'c' each have 2 samples.
    # After dropping singletons, 2 valid groups remain → should succeed.
    df = pd.DataFrame({
        "volume_id": ["v1", "v2", "v3", "v4", "v5"],
        "group": ["a", "b", "b", "c", "c"],
    })
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cleaned = cont_obj.validate_labels(df, "group")
        assert any("Dropping" in str(warning.message) for warning in w)
    # group 'a' dropped (only 1 sample)
    assert "a" not in cont_obj.label_to_idx
    # 'b' and 'c' kept
    assert "b" in cont_obj.label_to_idx
    assert "c" in cont_obj.label_to_idx


def test_contrastive_validate_labels_too_few_groups_raises(cont_obj):
    df = pd.DataFrame({"volume_id": ["v1"], "group": ["a"]})
    with pytest.warns(UserWarning, match="Dropping 1 group"):
        with pytest.raises(ValueError, match="at least 2"):
            cont_obj.validate_labels(df, "group")


def test_build_batch_sampler_returns_grouped_sampler(cont_obj, tmp_path):
    """build_batch_sampler returns a GroupedBatchSampler when groups have ≥2 files."""
    import numpy as np
    import pandas as pd
    from misfit.embedding.objectives.base import CropFeaturesDataset

    # Create 4 .npz files: 2 per group
    for vid in ("v1", "v2", "v3", "v4"):
        np.savez(tmp_path / f"{vid}.npz",
                 features=np.zeros((3, 16), dtype=np.float32),
                 positions=np.zeros((3, 3), dtype=np.float32))

    df = pd.DataFrame({
        "volume_id": ["v1", "v2", "v3", "v4"],
        "features_path": [str(tmp_path / f"{v}.npz") for v in ("v1", "v2", "v3", "v4")],
        "group": ["A", "A", "B", "B"],
    })
    df = cont_obj.validate_labels(df, "group")
    dataset = cont_obj.build_dataset(df, "group")

    sampler = cont_obj.build_batch_sampler(dataset, batch_size=4)
    assert isinstance(sampler, GroupedBatchSampler)


def test_build_batch_sampler_returns_none_when_all_singletons(cont_obj, tmp_path):
    """build_batch_sampler returns None when no group has ≥2 cached files (lines 233-239)."""
    import numpy as np
    import pandas as pd
    from misfit.embedding.objectives.base import CropFeaturesDataset

    # validate_labels requires ≥2 valid groups — give it two groups with 2 samples each
    # but only write ONE .npz file per group so build_batch_sampler sees singletons
    np.savez(tmp_path / "v1.npz",
             features=np.zeros((3, 16), dtype=np.float32),
             positions=np.zeros((3, 3), dtype=np.float32))
    np.savez(tmp_path / "v3.npz",
             features=np.zeros((3, 16), dtype=np.float32),
             positions=np.zeros((3, 3), dtype=np.float32))
    # v2 and v4 intentionally missing → each group ends up with only 1 cached file

    df = pd.DataFrame({
        "volume_id": ["v1", "v2", "v3", "v4"],
        "features_path": [str(tmp_path / f"{v}.npz") for v in ("v1", "v2", "v3", "v4")],
        "group": ["A", "A", "B", "B"],
    })
    df = cont_obj.validate_labels(df, "group")
    dataset = cont_obj.build_dataset(df, "group")

    with pytest.warns(UserWarning, match="Falling back"):
        sampler = cont_obj.build_batch_sampler(dataset, batch_size=4)

    assert sampler is None


def test_contrastive_build_head(cont_obj):
    import torch.nn as nn
    head = cont_obj.build_head(32)
    assert isinstance(head, nn.Sequential)


def test_contrastive_compute_loss(cont_obj):
    from misfit.embedding.embedding_constants import ec
    proj = torch.nn.functional.normalize(torch.randn(4, ec.PROJECTION_DIM), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    loss = cont_obj.compute_loss(proj, labels)
    assert loss.ndim == 0


def test_l2_norm_layer():
    layer = _L2NormLayer()
    x = torch.randn(4, 16)
    out = layer(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)
