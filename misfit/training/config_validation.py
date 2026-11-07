"""Warnings for mistakes in config.json, which people sometimes edit by hand.

A misspelled key would otherwise be ignored without a word, and a value of the
wrong type can fail far from its cause (the string "false" counts as true). These
checks only warn, so a config that works today keeps working. Keys missing from a
config aren't reported, since older configs don't have the newest settings.
"""

import difflib
from typing import Any

from misfit.utils.console import print_warning

# Every setting in config.json and the kind of value it takes. A test keeps this in
# step with MAETrainer._build_config.
_SETTINGS = {
    "misfit_version": "text",
    "data.index": "text",
    "data.normalization_policy": "text",
    "model.architecture": "text",
    "model.patch_size": "three_whole_numbers",
    "model.mask_patch_size": "whole_number",
    "model.mask_ratio": "fraction",
    "spatial_config.patch_size": "three_whole_numbers",
    "spatial_config.target_spacing": "optional_three_numbers",
    "training.epochs": "whole_number",
    "training.steps_per_epoch": "whole_number",
    "training.batch_size": "whole_number",
    "training.optimizer": "text",
    "training.learning_rate": "number",
    "training.weight_decay": "number",
    "training.lr_scheduler": "text",
    "training.warmup_epochs": "whole_number",
    "training.loss": "text",
    "training.amp": "boolean",
    "training.compile": "boolean",
    "training.seed": "whole_number",
    "training.gradient_accumulation_steps": "whole_number",
    "training.bucket_cap_mb": "whole_number",
    "training.val_every": "whole_number",
    "training.val_volumes": "whole_number",
    "training.checkpoint_minutes": "number",
    "training.init_weights": "optional_text",
}

# Sections whose contents vary: the metrics to evaluate, and their options.
_FREE_FORM_SECTIONS = frozenset({"evaluation"})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_whole_number(value: Any) -> bool:
    return _is_number(value) and float(value).is_integer()


def _is_three(value: Any, is_item) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(is_item(v) for v in value)


_KINDS = {
    "text": ("text", lambda v: isinstance(v, str)),
    "optional_text": ("text or null", lambda v: v is None or isinstance(v, str)),
    "boolean": ("true or false", lambda v: isinstance(v, bool)),
    "number": ("a number", _is_number),
    "whole_number": ("a whole number", _is_whole_number),
    "fraction": ("a number from 0 to 1", lambda v: _is_number(v) and 0 <= v <= 1),
    "three_whole_numbers": ("three whole numbers", lambda v: _is_three(v, _is_whole_number)),
    "optional_three_numbers": (
        "three numbers or null",
        lambda v: v is None or _is_three(v, _is_number),
    ),
}


def _schema() -> dict[str, Any]:
    """Return _SETTINGS as nested sections, with each setting's kind at its leaf."""
    schema: dict[str, Any] = {section: {} for section in _FREE_FORM_SECTIONS}
    for path, kind in _SETTINGS.items():
        *sections, key = path.split(".")
        node = schema
        for section in sections:
            node = node.setdefault(section, {})
        node[key] = kind
    return schema


def _check_section(
    section: dict[str, Any], schema: dict[str, Any], prefix: str, problems: list[str]
) -> None:
    """Add the problems in one section of the config, and in those inside it."""
    for key, value in section.items():
        path = f"{prefix}{key}"
        if path in _FREE_FORM_SECTIONS:
            continue
        if key not in schema:
            problem = f"MISFIT doesn't use '{path}', so it has no effect."
            close = difflib.get_close_matches(key, schema, n=1)
            if close:
                problem += f" Did you mean '{prefix}{close[0]}'?"
            problems.append(problem)
        elif isinstance(schema[key], dict):
            if isinstance(value, dict):
                _check_section(value, schema[key], f"{path}.", problems)
            else:
                problems.append(f"'{path}' should be a section, but is {value!r}.")
        else:
            description, is_valid = _KINDS[schema[key]]
            if not is_valid(value):
                problems.append(f"'{path}' should be {description}, but is {value!r}.")


def find_config_problems(config: dict[str, Any]) -> list[str]:
    """Find unknown keys and invalid values in a MISFIT config.

    Args:
        config: The contents of config.json.

    Returns:
        A description of each problem. Empty if there are none.
    """
    problems: list[str] = []
    _check_section(config, _schema(), "", problems)

    # The patch size is stored twice so MIST can read it; every MISFIT command
    # uses model.patch_size, so editing only the other copy has no effect here.
    model_patch = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    spatial = config.get("spatial_config")
    spatial_patch = spatial.get("patch_size") if isinstance(spatial, dict) else None
    if spatial_patch is not None and model_patch.get("patch_size") not in (None, spatial_patch):
        problems.append(
            f"'model.patch_size' ({model_patch['patch_size']}) and "
            f"'spatial_config.patch_size' ({spatial_patch}) differ. MISFIT uses the "
            "first and MIST reads the second, so they should match."
        )
    return problems


def warn_about_config_problems(config: dict[str, Any]) -> None:
    """Print a warning for each problem in a MISFIT config.

    Args:
        config: The contents of config.json.
    """
    for problem in find_config_problems(config):
        print_warning(f"config.json: {problem}")
