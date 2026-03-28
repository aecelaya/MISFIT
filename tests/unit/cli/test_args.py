"""Tests for misfit.cli.args."""
import argparse
import pytest

from misfit.cli.args import (
    ArgParser,
    add_evaluate_args,
    add_index_args,
    add_inspect_args,
    add_train_args,
    add_embed_args,
    add_embed_train_args,
    float_0_1,
    non_negative_int,
    positive_float,
    positive_int,
)


# ---------------------------------------------------------------------------
# Validator types
# ---------------------------------------------------------------------------

def test_positive_int_valid():
    assert positive_int("5") == 5


def test_positive_int_zero_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")


def test_positive_int_negative_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("-1")


def test_positive_float_valid():
    assert positive_float("0.5") == pytest.approx(0.5)


def test_positive_float_zero_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("0.0")


def test_positive_float_negative_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("-1.0")


def test_non_negative_int_zero():
    assert non_negative_int("0") == 0


def test_non_negative_int_negative_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")


def test_float_0_1_valid():
    assert float_0_1("0.5") == pytest.approx(0.5)
    assert float_0_1("0.0") == pytest.approx(0.0)
    assert float_0_1("1.0") == pytest.approx(1.0)


def test_float_0_1_out_of_range():
    with pytest.raises(argparse.ArgumentTypeError):
        float_0_1("1.1")
    with pytest.raises(argparse.ArgumentTypeError):
        float_0_1("-0.1")


# ---------------------------------------------------------------------------
# ArgParser helpers
# ---------------------------------------------------------------------------

def test_arg_parser_arg():
    p = ArgParser()
    p.arg("--foo", type=str, default="bar")
    ns = p.parse_args([])
    assert ns.foo == "bar"


def test_arg_parser_flag():
    p = ArgParser()
    p.flag("--verbose")
    ns = p.parse_args([])
    assert ns.verbose is False
    ns = p.parse_args(["--verbose"])
    assert ns.verbose is True


# ---------------------------------------------------------------------------
# Argument group builders
# ---------------------------------------------------------------------------

def test_add_index_args_input_csv(tmp_path):
    p = ArgParser()
    add_index_args(p)
    ns = p.parse_args(["--input", "paths.csv", "--output", "index.parquet"])
    assert ns.input == "paths.csv"
    assert ns.output == "index.parquet"
    assert ns.num_workers_index == 32


def test_add_index_args_input_parquet(tmp_path):
    p = ArgParser()
    add_index_args(p)
    ns = p.parse_args(["--input", "paths.parquet", "--output", "index.parquet"])
    assert ns.input == "paths.parquet"


def test_add_index_args_missing_input_raises():
    p = ArgParser()
    add_index_args(p, input_required=True)
    with pytest.raises(SystemExit):
        p.parse_args(["--output", "index.parquet"])


def test_add_train_args_defaults():
    p = ArgParser()
    add_train_args(p)
    ns = p.parse_args([
        "--index", "index.parquet",
        "--results", "/runs/exp1",
    ])
    assert ns.model == "swinmae-base"
    assert ns.epochs == 200
    assert ns.batch_size == 2
    assert ns.amp_dtype == "fp16"


def test_add_train_args_custom():
    p = ArgParser()
    add_train_args(p)
    ns = p.parse_args([
        "--index", "index.parquet",
        "--results", "/runs/exp1",
        "--model", "swinmae-small",
        "--epochs", "10",
    ])
    assert ns.model == "swinmae-small"
    assert ns.epochs == 10


def test_add_evaluate_args_defaults():
    p = ArgParser()
    add_evaluate_args(p)
    ns = p.parse_args([
        "--checkpoint", "best.pt",
        "--config", "/runs/exp1/config.json",
        "--index", "index.parquet",
        "--output-csv", "/runs/eval/results.csv",
    ])
    assert ns.checkpoint == "best.pt"
    assert ns.config == "/runs/exp1/config.json"
    assert ns.device is None
    assert ns.split == "val"


def test_add_inspect_args_defaults():
    p = ArgParser()
    add_inspect_args(p)
    ns = p.parse_args([
        "--checkpoint", "best.pt",
        "--config", "config.json",
        "--index", "index.parquet",
        "--output-dir", "/out",
    ])
    assert ns.checkpoint == "best.pt"
    assert ns.config == "config.json"
    assert ns.device is None
    assert ns.split is None


def test_add_embed_args_defaults():
    p = ArgParser()
    add_embed_args(p)
    ns = p.parse_args([
        "--encoder-checkpoint", "best.pt",
        "--config", "config.json",
        "--index", "index.parquet",
        "--output-dir", "/out",
    ])
    assert ns.encoder_checkpoint == "best.pt"
    assert ns.aggregator == "mean_pool"
    assert ns.config == "config.json"
    assert ns.split is None


def test_add_embed_train_args_defaults():
    p = ArgParser()
    add_embed_train_args(p)
    ns = p.parse_args([
        "--input", "input.csv",
        "--output-dir", "/out",
        "--embed-dim", "128",
    ])
    assert ns.input == "input.csv"
    assert ns.objective == "classification"
    assert ns.aggregator == "attention_pool"
    assert ns.embed_dim == 128
