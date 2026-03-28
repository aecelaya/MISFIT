"""Tests for misfit.embedding.embed_trainer.EmbedTrainer."""
import numpy as np
import pandas as pd
import pytest

import misfit.embedding  # noqa — trigger registrations


def _write_npz(path, n_crops=5, C=16):
    np.savez(
        path,
        features=np.random.randn(n_crops, C).astype(np.float32),
        positions=np.random.rand(n_crops, 3).astype(np.float32),
    )


def _make_labels_csv(tmp_path, volume_ids, labels, label_col="label"):
    df = pd.DataFrame({"volume_id": volume_ids, label_col: labels})
    p = tmp_path / "labels.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_embed_trainer_classification_runs(tmp_path):
    from misfit.embedding.embed_trainer import EmbedTrainer

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for i in range(4):
        _write_npz(features_dir / f"vol{i}.npz", n_crops=3, C=16)

    labels_csv = _make_labels_csv(
        tmp_path,
        [f"vol{i}" for i in range(4)],
        ["a", "b", "a", "b"],
    )
    output_dir = tmp_path / "out"

    trainer = EmbedTrainer(
        features_dir=features_dir,
        labels_csv=labels_csv,
        label_col="label",
        objective_name="classification",
        aggregator_name="mean_pool",
        embed_dim=16,
        output_dir=output_dir,
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        num_workers=0,
        device="cpu",
    )
    trainer.run()
    assert (output_dir / "aggregator.pt").exists()


def test_embed_trainer_contrastive_runs(tmp_path):
    from misfit.embedding.embed_trainer import EmbedTrainer

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for i in range(4):
        _write_npz(features_dir / f"vol{i}.npz", n_crops=3, C=16)

    labels_csv = _make_labels_csv(
        tmp_path,
        [f"vol{i}" for i in range(4)],
        ["grp_a", "grp_a", "grp_b", "grp_b"],
        label_col="group",
    )
    output_dir = tmp_path / "out_cont"

    trainer = EmbedTrainer(
        features_dir=features_dir,
        labels_csv=labels_csv,
        label_col="group",
        objective_name="contrastive",
        aggregator_name="mean_pool",
        embed_dim=16,
        output_dir=output_dir,
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        num_workers=0,
        device="cpu",
    )
    trainer.run()
    assert (output_dir / "aggregator.pt").exists()


def test_embed_trainer_no_feature_files_raises(tmp_path):
    from misfit.embedding.embed_trainer import EmbedTrainer

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    # No npz files written

    labels_csv = _make_labels_csv(tmp_path, ["vol1", "vol2"], ["a", "b"])
    output_dir = tmp_path / "out_err"

    trainer = EmbedTrainer(
        features_dir=features_dir,
        labels_csv=labels_csv,
        label_col="label",
        objective_name="classification",
        aggregator_name="mean_pool",
        embed_dim=16,
        output_dir=output_dir,
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="No feature files"):
        trainer.run()


def test_embed_trainer_attention_pool_with_position_encoding(tmp_path):
    from misfit.embedding.embed_trainer import EmbedTrainer

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for i in range(4):
        _write_npz(features_dir / f"vol{i}.npz", n_crops=3, C=16)

    labels_csv = _make_labels_csv(
        tmp_path, [f"vol{i}" for i in range(4)], ["a", "b", "a", "b"]
    )
    output_dir = tmp_path / "out_atten"

    trainer = EmbedTrainer(
        features_dir=features_dir,
        labels_csv=labels_csv,
        label_col="label",
        objective_name="classification",
        aggregator_name="attention_pool",
        embed_dim=16,
        output_dir=output_dir,
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        use_position_encoding=True,
    )
    trainer.run()
    assert (output_dir / "aggregator.pt").exists()
