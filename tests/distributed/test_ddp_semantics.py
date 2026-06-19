"""Multi-process DDP integration tests using the CPU ``gloo`` backend.

These spawn real torch.distributed process groups (no GPU needed) to exercise
behaviour that single-process, ``world_size==1`` unit tests cannot reach:

* gradient synchronisation semantics that the gradient-accumulation fix relies
  on (the trailing-window ``no_sync()`` divergence bug);
* the config precondition guard failing fast on *every* rank instead of
  leaving worker ranks hung on rendezvous.

They are marked ``distributed`` and excluded from the default run (which keeps
the fast suite CPU/single-process). Run them with::

    pytest -m distributed
"""
import os
import socket
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

pytestmark = pytest.mark.distributed

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("torch.distributed gloo backend unavailable", allow_module_level=True)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Workers (must be top-level so the spawn start method can pickle them)
# ---------------------------------------------------------------------------


def _grad_sync_worker(rank: int, world_size: int, port: int, is_last_accum: bool) -> None:
    """Run one _training_step under DDP and compare gradients across ranks."""
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)  # identical parameter init on every rank
    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    ddp = DDP(model)

    trainer = MAETrainer.__new__(MAETrainer)
    trainer.device = torch.device("cpu")
    trainer.amp = False
    trainer.is_distributed = True

    criterion = MaskedMSELoss()
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.0)  # inspect grads, don't move params

    # Rank-specific data so that, without synchronisation, gradients differ.
    torch.manual_seed(100 + rank)
    batch = {"image": torch.randn(1, 1, 32, 32, 32) * (rank + 1),
             "spacing": torch.ones(1, 3)}
    optimizer.zero_grad()
    trainer._training_step(ddp, batch, criterion, optimizer,
                           window_size=1, is_last_accum=is_last_accum)

    checksum = float(sum(p.grad.double().sum() for p in ddp.parameters()
                         if p.grad is not None))
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, checksum)
    dist.destroy_process_group()

    spread = max(gathered) - min(gathered)
    if is_last_accum:
        # Window end → gradients are all-reduced → identical on every rank.
        assert spread < 1e-6, f"gradients not synchronised across ranks: {gathered}"
    else:
        # no_sync() → each rank keeps its own (different) gradients.
        assert spread > 1e-9, f"gradients unexpectedly identical under no_sync: {gathered}"


def _guard_worker(rank: int, world_size: int, results_dir: str) -> None:
    """train() must raise on every rank when the results dir already has a config."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"

    args = SimpleNamespace(results=results_dir, resume=False, overwrite=False)
    trainer = MAETrainer(args)
    try:
        trainer.train()
    except RuntimeError:
        return  # expected: precondition guard fired on this rank
    raise AssertionError(f"rank {rank} did not raise on an existing config.json")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ddp_synchronises_gradients_on_window_end():
    """is_last_accum=True → DDP all-reduces, so gradients match across ranks."""
    mp.spawn(_grad_sync_worker, args=(2, _free_port(), True), nprocs=2, join=True)


def test_ddp_keeps_gradients_local_under_no_sync():
    """is_last_accum=False → no_sync(), so each rank keeps distinct gradients.

    This is the exact semantic the trailing-flush bug violated: stepping after
    a no_sync() backward let ranks diverge silently.
    """
    mp.spawn(_grad_sync_worker, args=(2, _free_port(), False), nprocs=2, join=True)


def test_config_guard_fails_fast_on_all_ranks_without_hanging(tmp_path):
    """Every rank raises on an existing config — no rank hangs on rendezvous."""
    (tmp_path / "config.json").write_text("{}")
    ctx = mp.spawn(_guard_worker, args=(2, str(tmp_path)), nprocs=2, join=False)
    # ctx.join() reaps one process per call (raising if a worker errored) and
    # returns True only once every rank has exited. Loop with an overall
    # deadline so a genuine hang fails the test instead of blocking forever.
    deadline = time.time() + 120
    while not ctx.join(timeout=5):
        assert time.time() < deadline, "a rank hung instead of failing fast"
