"""Shared Rich progress bar helpers for MISFIT.

Provides a single ``get_progress_bar()`` factory used throughout the codebase
so that every long-running loop (indexing, evaluation, inspection, embedding)
shares the same visual style, plus ``TrainProgressBar`` / ``ValidationProgressBar``
for ``misfit_train``'s epoch loop. The latter two mirror
``mist.utils.progress_bar`` (MISFIT has no cross-validation folds, so there's
no ``fold`` argument) so pretraining output looks the same as MIST training
output: a live loss and learning rate on every step, not just an epoch-end
summary.
"""
import numpy as np
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


def format_loss(value: float, precision: int = 4) -> str:
    """Format a loss value, switching to scientific notation once fixed-point
    display would round it away to all zeros.

    Loss values here are non-negative (MSE-family losses), so once training
    converges well past ``precision`` decimal places, ``f"{value:.4f}"``
    prints a flat, uninformative ``"0.0000"`` no matter how much further the
    loss actually drops. Switch to scientific notation in that case; a true
    zero is left as fixed-point (it isn't "too small to show", it *is* zero).
    Used by ``TrainProgressBar`` / ``ValidationProgressBar`` and by
    ``MAETrainer``'s epoch-summary console output, so both stay consistent.

    Args:
        value: The loss value to format.
        precision: Decimal places for the fixed-point form. Defaults to 4,
            matching the progress bar and epoch-summary formatting.

    Returns:
        ``f"{value:.{precision}f}"``, or ``f"{value:.3e}"`` if that would be
        indistinguishable from zero.
    """
    text = f"{value:.{precision}f}"
    if value != 0.0 and float(text) == 0.0:
        return f"{value:.3e}"
    return text


class TrainProgressBar:
    """Progress bar for the training loop with live loss and learning rate.

    Mirrors ``mist.utils.progress_bar.TrainProgressBar``'s columns and
    ``update()`` formatting. Use as a context manager::

        with TrainProgressBar(current_epoch=epoch + 1, epochs=epochs,
                               train_steps=len(train_loader)) as pb:
            for batch in train_loader:
                loss = training_step(batch)
                pb.update(loss=loss, lr=optimizer.param_groups[0]["lr"])
    """

    def __init__(self, current_epoch: int, epochs: int, train_steps: int) -> None:
        epoch_width = len(str(epochs))
        self.progress = Progress(
            TextColumn(f"Epoch{current_epoch: {epoch_width}}/{epochs}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[loss]}"),
            TextColumn("•"),
            TextColumn("{task.fields[lr]}"),
        )
        self.task = self.progress.add_task(
            description="Training (loss)",
            total=train_steps,
            loss="loss: ",
            lr="lr: ",
        )

    def update(self, loss: float | None = None, lr: float | None = None) -> None:
        """Advance one step, refreshing whichever of loss/lr are given.

        Both are optional (unlike MIST's, which are always available every
        step) so that MISFIT's gradient-accumulation micro-steps can still
        advance the bar between window ends, where there is no freshly
        aggregated loss yet — call with no arguments to just advance. A field
        left unset keeps showing its last value, per Rich's normal
        ``Progress.update()`` semantics.
        """
        fields = {}
        if loss is not None:
            fields["loss"] = f"loss: {format_loss(loss)}"
        if lr is not None:
            fields["lr"] = f"lr: {np.format_float_scientific(lr, precision=3)}"
        self.progress.update(self.task, advance=1, **fields)

    def __enter__(self) -> "TrainProgressBar":
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.progress.stop()


class ValidationProgressBar:
    """Progress bar for the validation loop with a live running loss.

    Mirrors ``mist.utils.progress_bar.ValidationProgressBar``. Use as a
    context manager::

        with ValidationProgressBar(val_steps=len(val_loader)) as pb:
            for batch in val_loader:
                loss = validation_step(batch)
                pb.update(loss=loss)
    """

    def __init__(self, val_steps: int) -> None:
        self.progress = Progress(
            TextColumn("Validating"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[loss]}"),
        )
        self.task = self.progress.add_task(
            description="Validation", total=val_steps, loss="val_loss: "
        )

    def update(self, loss: float) -> None:
        """Advance one step and refresh the displayed validation loss."""
        self.progress.update(
            self.task, advance=1, loss=f"val_loss: {format_loss(loss)}"
        )

    def __enter__(self) -> "ValidationProgressBar":
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.progress.stop()


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
