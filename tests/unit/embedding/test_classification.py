"""Tests for misfit.embedding.objectives.classification.ClassificationObjective."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from misfit.embedding.objectives.classification import ClassificationObjective


@pytest.fixture
def obj():
    return ClassificationObjective()


def test_validate_labels_sets_num_classes(obj, tmp_path):
    df = pd.DataFrame({"volume_id": ["v1", "v2", "v3"], "label": ["a", "b", "a"]})
    cleaned = obj.validate_labels(df, "label")
    assert obj.num_classes == 2
    assert len(cleaned) == 3


def test_validate_labels_string_mapping(obj):
    df = pd.DataFrame({"volume_id": ["v1", "v2"], "label": ["cat", "dog"]})
    obj.validate_labels(df, "label")
    assert "cat" in obj.label_to_idx
    assert "dog" in obj.label_to_idx


def test_validate_labels_drops_nan(obj):
    df = pd.DataFrame({"volume_id": ["v1", "v2", "v3"], "label": ["a", None, "b"]})
    cleaned = obj.validate_labels(df, "label")
    assert len(cleaned) == 2


def test_validate_labels_empty_raises(obj):
    df = pd.DataFrame({"volume_id": [], "label": []})
    with pytest.raises(ValueError, match="No valid samples"):
        obj.validate_labels(df, "label")


def test_build_head_is_linear(obj):
    df = pd.DataFrame({"volume_id": ["v1", "v2"], "label": ["a", "b"]})
    obj.validate_labels(df, "label")
    head = obj.build_head(16)
    assert isinstance(head, nn.Linear)
    assert head.out_features == 2


def test_build_batch_sampler_returns_none(obj, tmp_path):
    df = pd.DataFrame({"volume_id": ["v1"], "label": ["a"]})
    obj.validate_labels(df, "label")
    from misfit.embedding.objectives.base import CropFeaturesDataset
    ds = CropFeaturesDataset(tmp_path, df, "label", obj.label_to_idx)
    sampler = obj.build_batch_sampler(ds, 4)
    assert sampler is None


def test_compute_loss(obj):
    df = pd.DataFrame({"volume_id": ["v1", "v2"], "label": ["a", "b"]})
    obj.validate_labels(df, "label")
    logits = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loss = obj.compute_loss(logits, labels)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_build_dataset_returns_dataset(obj, tmp_path):
    # Need at least one npz to have a non-empty dataset
    np.savez(tmp_path / "v1.npz",
             features=np.random.randn(3, 16).astype(np.float32),
             positions=np.random.rand(3, 3).astype(np.float32))
    df = pd.DataFrame({"volume_id": ["v1"], "label": ["a"]})
    obj.validate_labels(df, "label")
    from misfit.embedding.objectives.base import CropFeaturesDataset
    ds = obj.build_dataset(tmp_path, df, "label")
    assert isinstance(ds, CropFeaturesDataset)
