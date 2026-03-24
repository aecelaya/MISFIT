"""Shared Rich progress bar helpers for MISFIT.

Provides a single ``get_progress_bar()`` factory used throughout the codebase
so that every long-running loop (indexing, training, evaluation) shares the
same visual style.
"""
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from misfit.utils.console import console


def get_progress_bar() -> Progress:
    """Return a rich :class:`~rich.progress.Progress` bar.

    The returned object can be used as a context manager::

        with get_progress_bar() as progress:
            task = progress.add_task("Doing work", total=n)
            for item in items:
                do_work(item)
                progress.advance(task)

    Returns:
        A configured :class:`~rich.progress.Progress` instance that writes
        to the shared MISFIT console.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
