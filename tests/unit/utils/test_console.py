"""Tests for misfit.utils.console."""
from unittest.mock import patch

import pytest
from rich.console import Console

from misfit.utils.console import (
    console,
    print_error,
    print_info,
    print_section_header,
    print_success,
    print_warning,
)


def test_console_singleton():
    """console is a rich Console instance."""
    assert isinstance(console, Console)


@pytest.mark.parametrize("fn,keyword", [
    (print_section_header, "bold"),
    (print_info, "hello"),
    (print_warning, "Warning"),
    (print_error, "Error"),
    (print_success, "green"),
])
def test_print_functions_call_console(fn, keyword):
    """Each print helper calls console.print without raising."""
    with patch.object(console, "print") as mock_print:
        fn("hello")
        mock_print.assert_called_once()
        call_str = str(mock_print.call_args)
        assert keyword in call_str or "hello" in call_str


def test_print_section_header_content():
    outputs = []
    with patch.object(console, "print", side_effect=lambda msg: outputs.append(msg)):
        print_section_header("MyTitle")
    assert "MyTitle" in outputs[0]
    assert "bold" in outputs[0]


def test_print_warning_content():
    outputs = []
    with patch.object(console, "print", side_effect=lambda msg: outputs.append(msg)):
        print_warning("some warning")
    assert "Warning" in outputs[0]
    assert "some warning" in outputs[0]


def test_print_error_content():
    outputs = []
    with patch.object(console, "print", side_effect=lambda msg: outputs.append(msg)):
        print_error("some error")
    assert "Error" in outputs[0]
    assert "some error" in outputs[0]


def test_print_success_content():
    outputs = []
    with patch.object(console, "print", side_effect=lambda msg: outputs.append(msg)):
        print_success("done")
    assert "green" in outputs[0]
    assert "done" in outputs[0]
