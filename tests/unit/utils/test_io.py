"""Tests for misfit.utils.io."""
import json
from pathlib import Path

import pytest

from misfit.utils.io import read_json_file, write_json_file


def test_write_and_read_roundtrip(tmp_path):
    data = {"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}
    path = tmp_path / "data.json"
    write_json_file(path, data)
    result = read_json_file(path)
    assert result == data


def test_write_creates_file(tmp_path):
    path = tmp_path / "out.json"
    write_json_file(path, {"x": 1})
    assert path.exists()


def test_read_json_file_with_string_path(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"a": 1}))
    result = read_json_file(str(path))
    assert result == {"a": 1}


def test_write_json_file_with_string_path(tmp_path):
    path = tmp_path / "out.json"
    write_json_file(str(path), {"b": 2})
    assert json.loads(path.read_text()) == {"b": 2}


def test_write_json_indentation(tmp_path):
    path = tmp_path / "pretty.json"
    write_json_file(path, {"x": 1})
    content = path.read_text()
    # indent=2 means there are newlines in the file
    assert "\n" in content


def test_read_nonexistent_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json_file(tmp_path / "missing.json")
