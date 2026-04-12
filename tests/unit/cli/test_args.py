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

@pytest.mark.parametrize("fn,value,expected", [
    pytest.param(positive_int,     "5",   5,                   id="positive_int"),
    pytest.param(positive_float,   "0.5", pytest.approx(0.5),  id="positive_float"),
    pytest.param(non_negative_int, "0",   0,                   id="non_negative_int_zero"),
    pytest.param(float_0_1,        "0.5", pytest.approx(0.5),  id="float_0_1_mid"),
    pytest.param(float_0_1,        "0.0", pytest.approx(0.0),  id="float_0_1_lower_bound"),
    pytest.param(float_0_1,        "1.0", pytest.approx(1.0),  id="float_0_1_upper_bound"),
])
def test_validator_accepts_valid_input(fn, value, expected):
    assert fn(value) == expected


@pytest.mark.parametrize("fn,value", [
    pytest.param(positive_int,     "0",    id="positive_int_zero"),
    pytest.param(positive_int,     "-1",   id="positive_int_negative"),
    pytest.param(positive_float,   "0.0",  id="positive_float_zero"),
    pytest.param(positive_float,   "-1.0", id="positive_float_negative"),
    pytest.param(non_negative_int, "-1",   id="non_negative_int_negative"),
    pytest.param(float_0_1,        "1.1",  id="float_0_1_above_max"),
    pytest.param(float_0_1,        "-0.1", id="float_0_1_below_min"),
])
def test_validator_rejects_invalid_input(fn, value):
    with pytest.raises(argparse.ArgumentTypeError):
        fn(value)


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
    assert ns.model == "swinunetr-base"
    assert ns.epochs == 200
    assert ns.batch_size == 2
    assert ns.amp_dtype == "fp16"


def test_add_train_args_custom():
    p = ArgParser()
    add_train_args(p)
    ns = p.parse_args([
        "--index", "index.parquet",
        "--results", "/runs/exp1",
        "--model", "swinunetr-small",
        "--epochs", "10",
    ])
    assert ns.model == "swinunetr-small"
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
