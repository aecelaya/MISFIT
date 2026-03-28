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


def _make_input_csv(tmp_path, volume_ids, labels, split="train"):
    """Write a unified input CSV with features_path and label columns."""
    features_dir = tmp_path / "features"
    features_dir.mkdir(exist_ok=True)
    rows = []
    for vid, label in zip(volume_ids, labels):
        npz = features_dir / f"{vid}.npz"
        _write_npz(npz, n_crops=3, C=16)
        rows.append({
            "volume_id": vid,
            "split": split,
            "features_path": str(npz),
            "label": label,
        })
    p = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_embed_trainer_classification_runs(tmp_path):
    from misfit.embedding.embed_trainer import EmbedTrainer

    input_csv = _make_input_csv(
        tmp_path, [f"vol{i}" for i in range(4)], ["a", "b", "a", "b"]
    )
    output_dir = tmp_path / "out"

    trainer = EmbedTrainer(
        input_csv=input_csv,
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

    input_csv = _make_input_csv(
        tmp_path,
        [f"vol{i}" for i in range(4)],
        ["grp_a", "grp_a", "grp_b", "grp_b"],
    )
    output_dir = tmp_path / "out_cont"

    trainer = EmbedTrainer(
        input_csv=input_csv,
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

    # CSV points to non-existent .npz files
    rows = [
        {"volume_id": "vol1", "split": "train", "features_path": str(tmp_path / "vol1.npz"), "label": "a"},
        {"volume_id": "vol2", "split": "train", "features_path": str(tmp_path / "vol2.npz"), "label": "b"},
    ]
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    trainer = EmbedTrainer(
        input_csv=input_csv,
        objective_name="classification",
        aggregator_name="mean_pool",
        embed_dim=16,
        output_dir=tmp_path / "out_err",
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="No feature files"):
        trainer.run()


def test_embed_trainer_attention_pool_with_position_encoding(tmp_path):
    from misfit.embedding.embed_trainer import EmbedTrainer

    input_csv = _make_input_csv(
        tmp_path, [f"vol{i}" for i in range(4)], ["a", "b", "a", "b"]
    )
    output_dir = tmp_path / "out_atten"

    trainer = EmbedTrainer(
        input_csv=input_csv,
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


def test_embed_trainer_only_train_split_used(tmp_path):
    """Rows with split != 'train' are ignored."""
    from misfit.embedding.embed_trainer import EmbedTrainer

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    rows = []
    for i, (split, label) in enumerate([("train", "a"), ("train", "b"),
                                         ("val", "a"), ("test", "b")]):
        npz = features_dir / f"vol{i}.npz"
        _write_npz(npz, n_crops=3, C=16)
        rows.append({"volume_id": f"vol{i}", "split": split,
                     "features_path": str(npz), "label": label})
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    trainer = EmbedTrainer(
        input_csv=input_csv,
        objective_name="classification",
        aggregator_name="mean_pool",
        embed_dim=16,
        output_dir=tmp_path / "out_split",
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    trainer.run()
    assert (tmp_path / "out_split" / "aggregator.pt").exists()


def test_embed_trainer_checkpoint_contains_label_to_idx(tmp_path):
    """Saved checkpoint includes label_to_idx for inference-time decoding."""
    import torch
    from misfit.embedding.embed_trainer import EmbedTrainer

    input_csv = _make_input_csv(
        tmp_path, [f"vol{i}" for i in range(4)], ["healthy", "disease", "healthy", "disease"]
    )
    output_dir = tmp_path / "out_labels"

    trainer = EmbedTrainer(
        input_csv=input_csv,
        objective_name="classification",
        aggregator_name="mean_pool",
        embed_dim=16,
        output_dir=output_dir,
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    trainer.run()

    ckpt = torch.load(output_dir / "aggregator.pt", map_location="cpu", weights_only=True)
    assert "label_to_idx" in ckpt
    assert ckpt["label_to_idx"] == {"disease": 0, "healthy": 1}
