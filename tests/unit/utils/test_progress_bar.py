"""Tests for misfit.utils.progress_bar."""
from rich.progress import Progress

from misfit.utils.progress_bar import get_progress_bar


def test_get_progress_bar_returns_progress():
    pb = get_progress_bar()
    assert isinstance(pb, Progress)


def test_progress_bar_usable_as_context_manager():
    """The progress bar should work as a context manager."""
    with get_progress_bar() as progress:
        task = progress.add_task("test", total=5)
        for _ in range(5):
            progress.advance(task)
    # No exception raised means it works.


def test_progress_bar_uses_shared_console():
    """Progress bar should use the misfit console."""
    from misfit.utils.console import console
    pb = get_progress_bar()
    assert pb.console is console
