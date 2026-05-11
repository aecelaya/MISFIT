"""IO utilities for MISFIT runtime operations."""
import json
from pathlib import Path
from typing import Any


def read_json_file(json_file: str | Path) -> dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary.

    Args:
        json_file: Path to the JSON file.

    Returns:
        Dictionary with the JSON file data.
    """
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


def write_json_file(
    json_file: str | Path,
    data: dict[str, Any],
) -> None:
    """Write a dictionary to a JSON file.

    Args:
        json_file: Destination path.
        data: Dictionary to serialise.
    """
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
