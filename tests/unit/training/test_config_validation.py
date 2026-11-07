"""Tests for misfit.training.config_validation."""
from unittest.mock import patch

from misfit.training import config_validation
from misfit.training.config_validation import (
    find_config_problems,
    warn_about_config_problems,
)
from misfit.training.trainers.mae_trainer import MAETrainer
from tests.unit.training.test_mae_trainer import _make_args


def _flatten(section, prefix=""):
    paths = set()
    for key, value in section.items():
        if isinstance(value, dict):
            paths |= _flatten(value, f"{prefix}{key}.")
        else:
            paths.add(f"{prefix}{key}")
    return paths


def _config(tmp_path):
    return MAETrainer(_make_args(tmp_path))._build_config()


def test_settings_match_what_the_trainer_writes(tmp_path):
    """A new config key must be added to the checks, or it would warn on every run."""
    written = _config(tmp_path)
    del written["evaluation"]
    assert _flatten(written) == set(config_validation._SETTINGS)


def test_config_from_the_trainer_has_no_problems(tmp_path):
    assert find_config_problems(_config(tmp_path)) == []


def test_missing_keys_are_not_problems():
    """Older configs don't have the newest settings."""
    assert find_config_problems({"training": {"amp": True}}) == []


def test_unknown_key_suggests_the_likely_spelling():
    problems = find_config_problems({"training": {"learning_rat": 0.1}})
    assert len(problems) == 1
    assert "'training.learning_rat'" in problems[0]
    assert "Did you mean 'training.learning_rate'?" in problems[0]


def test_unknown_key_with_nothing_close():
    problems = find_config_problems({"zzz": 1})
    assert problems == ["MISFIT doesn't use 'zzz', so it has no effect."]


def test_evaluation_section_is_free_form():
    assert find_config_problems({"evaluation": {"anything": {"at": "all"}}}) == []


def test_setting_where_a_section_belongs():
    assert find_config_problems({"training": 3}) == ["'training' should be a section, but is 3."]


def _problem_for(section, key, value):
    return find_config_problems({section: {key: value}})


def test_string_false_is_not_a_boolean():
    assert "true or false" in _problem_for("training", "compile", "false")[0]
    assert "true or false" in _problem_for("training", "amp", 1)[0]
    assert _problem_for("training", "amp", False) == []


def test_numbers():
    assert "a number" in _problem_for("training", "learning_rate", "1e-4")[0]
    assert "a number" in _problem_for("training", "learning_rate", True)[0]
    assert _problem_for("training", "learning_rate", 1e-4) == []


def test_whole_numbers():
    assert "whole number" in _problem_for("training", "epochs", 1.5)[0]
    assert _problem_for("training", "epochs", 10) == []


def test_fractions():
    assert "from 0 to 1" in _problem_for("model", "mask_ratio", 75)[0]
    assert _problem_for("model", "mask_ratio", 0.75) == []


def test_text():
    assert "text" in _problem_for("training", "loss", 3)[0]
    assert "text or null" in _problem_for("training", "init_weights", 3)[0]
    assert _problem_for("training", "init_weights", None) == []
    assert _problem_for("training", "init_weights", "a.pth") == []


def test_three_numbers():
    assert "three whole numbers" in _problem_for("model", "patch_size", [96, 96])[0]
    assert "three whole numbers" in _problem_for("model", "patch_size", 96)[0]
    assert "three numbers or null" in _problem_for("spatial_config", "target_spacing", [1, "1", 1])[0]
    assert _problem_for("spatial_config", "target_spacing", None) == []
    assert _problem_for("spatial_config", "target_spacing", [1, 1, 2.5]) == []


def test_patch_size_copies_that_differ():
    problems = find_config_problems({
        "model": {"patch_size": [96, 96, 96]},
        "spatial_config": {"patch_size": [128, 128, 128]},
    })
    assert len(problems) == 1
    assert "differ" in problems[0]


def test_patch_size_copies_that_match_or_are_missing():
    same = {"patch_size": [96, 96, 96]}
    assert find_config_problems({"model": same, "spatial_config": same}) == []
    assert find_config_problems({"spatial_config": same}) == []
    assert find_config_problems({"model": same}) == []


def test_patch_size_check_tolerates_malformed_sections():
    problems = find_config_problems({"model": 3, "spatial_config": 3})
    assert len(problems) == 2


def test_warn_prints_each_problem():
    with patch.object(config_validation, "print_warning") as warn:
        warn_about_config_problems({"a": 1, "b": 2})
    assert warn.call_count == 2
    assert warn.call_args.args[0].startswith("config.json: ")


def test_warn_is_silent_without_problems():
    with patch.object(config_validation, "print_warning") as warn:
        warn_about_config_problems({})
    warn.assert_not_called()
