"""Tests for misfit.utils.progress_bar."""
from typing import Any
from unittest.mock import patch

import numpy as np
from rich.progress import Progress

from misfit.utils.progress_bar import (
    TrainProgressBar,
    ValidationProgressBar,
    get_progress_bar,
)


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


# ---------------------------------------------------------------------------
# TrainProgressBar / ValidationProgressBar — ported from MIST's SpyProgress
# pattern (mist.utils.progress_bar), minus the `fold` argument: MISFIT
# pretraining has no cross-validation folds.
# ---------------------------------------------------------------------------


class SpyProgress:
    """Spy replacement for rich.progress.Progress used for tests.

    Captures constructor columns, add_task calls, update calls, and
    start/stop state without performing any terminal output.
    """

    def __init__(self, *columns: Any, **kwargs: Any):
        self.columns = columns
        self.started = False
        self.stopped = False
        self._next_task_id = 1
        self.tasks: list[int] = []
        self.add_task_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []

    def add_task(self, description: str, total: int, **fields: Any) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        self.tasks.append(task_id)
        self.add_task_calls.append(
            {"description": description, "total": total, "fields": fields}
        )
        return task_id

    def update(self, task_id: int, **kwargs: Any) -> None:
        self.update_calls.append({"task_id": task_id, **kwargs})

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@patch("misfit.utils.progress_bar.Progress", new=SpyProgress)
def test_train_progressbar_initialization_creates_task():
    """TrainProgressBar creates a task with loss/lr fields initialised."""
    pb = TrainProgressBar(current_epoch=3, epochs=10, train_steps=123)

    assert isinstance(pb.progress, SpyProgress)
    assert len(pb.progress.add_task_calls) == 1

    call = pb.progress.add_task_calls[0]
    assert call["description"] == "Training (loss)"
    assert call["total"] == 123
    assert call["fields"]["loss"] == "loss: "
    assert call["fields"]["lr"] == "lr: "
    assert len(pb.progress.columns) > 0


@patch("misfit.utils.progress_bar.Progress", new=SpyProgress)
def test_train_progressbar_update_formats_loss_and_lr():
    """update(loss=, lr=) advances by 1 and formats both fields."""
    pb = TrainProgressBar(current_epoch=1, epochs=5, train_steps=10)

    loss, lr = 0.123456, 1e-4
    pb.update(loss=loss, lr=lr)

    assert len(pb.progress.update_calls) == 1
    update = pb.progress.update_calls[0]
    assert update["advance"] == 1
    assert update["loss"] == f"loss: {loss:.4f}"
    assert update["lr"] == f"lr: {np.format_float_scientific(lr, precision=3)}"


def test_train_progressbar_update_with_no_args_advances_only():
    """update() with neither loss nor lr still advances — the gradient-
    accumulation mid-window case, where there's no fresh aggregated loss yet.

    Uses the real Rich Progress (not the spy) to confirm the *actual*
    Progress.update() semantics: an omitted field keeps its previous value
    rather than being cleared.
    """
    pb = TrainProgressBar(current_epoch=1, epochs=5, train_steps=10)
    with pb:
        pb.update()
        task = next(t for t in pb.progress.tasks if t.id == pb.task)
        assert task.completed == 1
        # Fields keep their initial placeholder text — nothing was refreshed.
        assert task.fields["loss"] == "loss: "
        assert task.fields["lr"] == "lr: "

        pb.update(loss=0.5, lr=1e-3)
        pb.update()  # advance-only again, after a real refresh
        task = next(t for t in pb.progress.tasks if t.id == pb.task)
        assert task.completed == 3
        assert task.fields["loss"] == "loss: 0.5000"
        assert task.fields["lr"] == f"lr: {np.format_float_scientific(1e-3, precision=3)}"


@patch("misfit.utils.progress_bar.Progress", new=SpyProgress)
def test_train_progressbar_multiple_updates():
    pb = TrainProgressBar(current_epoch=1, epochs=5, train_steps=3)

    values = [(0.2, 1e-3), (0.19, 9.5e-4), (0.181, 8.7e-4)]
    for loss, lr in values:
        pb.update(loss=loss, lr=lr)

    assert len(pb.progress.update_calls) == len(values)
    for (loss, lr), update in zip(values, pb.progress.update_calls):
        assert update["advance"] == 1
        assert update["loss"] == f"loss: {loss:.4f}"
        assert update["lr"] == f"lr: {np.format_float_scientific(lr, precision=3)}"


@patch("misfit.utils.progress_bar.Progress", new=SpyProgress)
def test_train_progressbar_context_manager_starts_and_stops():
    pb = TrainProgressBar(current_epoch=1, epochs=5, train_steps=1)

    assert not pb.progress.started
    assert not pb.progress.stopped
    with pb as ctx:
        assert ctx is pb
        assert pb.progress.started
        assert not pb.progress.stopped
    assert pb.progress.stopped


@patch("misfit.utils.progress_bar.Progress", new=SpyProgress)
def test_validation_progressbar_initialization_creates_task():
    """ValidationProgressBar creates a task with a val_loss field."""
    pb = ValidationProgressBar(val_steps=42)

    assert isinstance(pb.progress, SpyProgress)
    assert len(pb.progress.add_task_calls) == 1

    call = pb.progress.add_task_calls[0]
    assert call["description"] == "Validation"
    assert call["total"] == 42
    assert call["fields"]["loss"] == "val_loss: "
    assert len(pb.progress.columns) > 0


@patch("misfit.utils.progress_bar.Progress", new=SpyProgress)
def test_validation_progressbar_update_formats_loss():
    pb = ValidationProgressBar(val_steps=5)

    loss = 0.987654
    pb.update(loss=loss)

    assert len(pb.progress.update_calls) == 1
    update = pb.progress.update_calls[0]
    assert update["advance"] == 1
    assert update["loss"] == f"val_loss: {loss:.4f}"


@patch("misfit.utils.progress_bar.Progress", new=SpyProgress)
def test_validation_progressbar_multiple_updates():
    pb = ValidationProgressBar(val_steps=3)
    losses = [0.33, 0.31, 0.3051]

    for val in losses:
        pb.update(loss=val)

    assert len(pb.progress.update_calls) == len(losses)
    for val, update in zip(losses, pb.progress.update_calls):
        assert update["advance"] == 1
        assert update["loss"] == f"val_loss: {val:.4f}"


@patch("misfit.utils.progress_bar.Progress", new=SpyProgress)
def test_validation_progressbar_context_manager_starts_and_stops():
    pb = ValidationProgressBar(val_steps=2)

    assert not pb.progress.started
    assert not pb.progress.stopped
    with pb as ctx:
        assert ctx is pb
        assert pb.progress.started
        assert not pb.progress.stopped
    assert pb.progress.stopped
