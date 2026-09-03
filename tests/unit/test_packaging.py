"""Packaging guards for the release process.

The version lives in two hand-edited files — ``pyproject.toml`` (what PyPI and
the Docker publish workflow read) and ``misfit/__init__.py`` (what
``_build_config`` writes into every ``config.json``). Cutting a release bumps
both; this keeps them from drifting. Also checks that every declared console
script still resolves.
"""
import re
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from packaging.version import Version

import misfit

_PYPROJECT = Path(misfit.__file__).resolve().parent.parent / "pyproject.toml"

# One console script per CLI command — keep in sync with [project.scripts].
_EXPECTED_SCRIPTS = {
    "misfit_index",
    "misfit_train",
    "misfit_evaluate",
    "misfit_inspect",
    "misfit_encode",
    "misfit_embed",
    "misfit_embed_train",
}


def _pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    # First `version = "..."` line (under [project]).
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, 'no `version = "..."` in pyproject.toml'
    return match.group(1)


@pytest.mark.skipif(not _PYPROJECT.exists(), reason="not a source checkout")
def test_version_matches_pyproject():
    assert Version(misfit.__version__) == Version(_pyproject_version())


def test_console_scripts_resolve():
    scripts = {
        ep.name: ep
        for ep in entry_points(group="console_scripts")
        if ep.name.startswith("misfit_")
    }
    assert set(scripts) == _EXPECTED_SCRIPTS
    for ep in scripts.values():
        assert callable(ep.load())  # imports the module + resolves the attr
