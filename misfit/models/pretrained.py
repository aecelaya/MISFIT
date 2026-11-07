"""Load full-network weights from MISFIT and nnssl/nnFoundation checkpoints.

Used both to initialize pretraining (``misfit_train --init-weights``) and to
rebuild a model for evaluation, inspection, encoding, and embedding, so every
command accepts the same checkpoint formats:

* a MISFIT checkpoint (``checkpoint.pt`` / ``best_model.pt``; weights under
  ``"model"``),
* an nnssl checkpoint such as nnFoundationCNN's or nnFoundationViT's
  ``checkpoint_final.pth``
  (weights under ``"network_weights"``, nnU-Net parameter names), or
* a plain state dict.

Each model maps foreign parameter names onto its own with
``adapt_pretrained_state_dict``. Loading is strict: every parameter must be
present with the right shape, except ``mask_token``, which nnssl's MAE does
not have (it masks with zeros, which is the token's initial value). A partial
load would silently mix pretrained and random weights.
"""
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

# Keys under which checkpoints nest the state dict: MISFIT ("model"),
# nnssl/nnU-Net ("network_weights"), PyTorch Lightning ("state_dict").
_CHECKPOINT_CONTAINER_KEYS = ("model", "network_weights", "state_dict", "model_state_dict")

# Parameters a foreign checkpoint may lack; they keep their initialization.
_OPTIONAL_KEYS = frozenset({"mask_token"})


def extract_state_dict(checkpoint: dict) -> OrderedDict:
    """Return the state dict inside a checkpoint, without a DDP ``module.`` prefix.

    Args:
        checkpoint: A loaded checkpoint: a plain state dict, or a dict that
            nests one under a known container key.

    Returns:
        The state dict.

    Raises:
        ValueError: If no state dict can be found.
    """
    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        state_dict = next(
            (checkpoint[k] for k in _CHECKPOINT_CONTAINER_KEYS
             if isinstance(checkpoint.get(k), dict)),
            None,
        )
        if state_dict is None:
            raise ValueError(
                "No model weights found in the checkpoint. Expected a state "
                f"dict or one of the keys {list(_CHECKPOINT_CONTAINER_KEYS)}; "
                f"found {list(checkpoint)[:10]}."
            )
    return OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
    )


def load_model_weights(model: nn.Module, checkpoint: dict) -> None:
    """Load a checkpoint's weights into *model*, strictly.

    Args:
        model: The MISFIT model to load into.
        checkpoint: A loaded checkpoint (see :func:`extract_state_dict`).

    Raises:
        ValueError: If any parameter is missing (other than ``mask_token``),
            unexpected, or has the wrong shape — the checkpoint is for a
            different architecture, patch size, or target spacing.
    """
    state_dict = extract_state_dict(checkpoint)
    adapt = getattr(model, "adapt_pretrained_state_dict", None)
    if adapt is not None:
        state_dict = adapt(state_dict)

    target = model.state_dict()
    missing = [k for k in target if k not in state_dict and k not in _OPTIONAL_KEYS]
    unexpected = [k for k in state_dict if k not in target]
    mismatched = [
        f"{k}: checkpoint {tuple(state_dict[k].shape)} vs model {tuple(v.shape)}"
        for k, v in target.items()
        if k in state_dict and state_dict[k].shape != v.shape
    ]
    if missing or unexpected or mismatched:
        details = "; ".join(
            f"{label} ({len(keys)}): {keys[:3]}"
            for label, keys in (
                ("missing", missing),
                ("unexpected", unexpected),
                ("shape mismatch", mismatched),
            )
            if keys
        )
        raise ValueError(
            f"Checkpoint weights don't match {type(model).__name__}: {details}. "
            "Check --model, --patch-size, and --target-spacing; for example, "
            "nnFoundationCNN needs --model resenc --patch-size 192 192 192 "
            "(any patch of at least 128^3 gives the same network) and either "
            "no --target-spacing (native resolution, as nnssl pretrained it) "
            "or an isotropic one such as --target-spacing 1 1 1. "
            "nnFoundationViT needs --model primus-x."
        )
    model.load_state_dict(state_dict, strict=False)


def load_pretrained_weights(model: nn.Module, path: str | Path) -> None:
    """Load a checkpoint file's weights into *model* (see :func:`load_model_weights`).

    The file is read with torch's weights-only loader, so it cannot execute
    code.

    Args:
        model: The MISFIT model to load into.
        path: Path to the checkpoint file.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    load_model_weights(model, checkpoint)
