"""Patch-wise intensity normalization shared by the loss and the evaluators.

``normalized_masked_mse`` normalizes its reconstruction target within each
non-overlapping ``mask_patch_size³`` cube (zero mean, unit variance) before
computing MSE. The evaluator and the ``misfit_inspect`` de-normalization step
need the *same* transform to score / visualize the model in the space it was
actually trained in — so the implementation lives here, once, and both the loss
and the inference code call it.

Getting the cube size wrong (e.g. normalizing per whole crop instead of per
mask cube) silently rescales every cube and makes the reported error
uninterpretable, so there is exactly one implementation.
"""

import torch


def _fold_cubes(x: torch.Tensor, p: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Reshape ``(B, C, D, H, W)`` into ``(B, C, n_cubes, p**3)``.

    Returns the folded tensor and the ``(B, C, gd, gh, gw)`` grid shape needed
    to invert the fold.
    """
    B, C, D, H, W = x.shape
    for name, size in zip(("D", "H", "W"), (D, H, W), strict=True):
        if size % p != 0:
            raise ValueError(
                f"Spatial dimension {name}={size} is not divisible by "
                f"patch_size={p}. Ensure img_size and mask_patch_size are "
                f"consistent."
            )
    gd, gh, gw = D // p, H // p, W // p
    t = x.reshape(B, C, gd, p, gh, p, gw, p)
    t = t.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, C, gd * gh * gw, p * p * p)
    return t, (B, C, gd, gh, gw)


def _unfold_cubes(t: torch.Tensor, grid: tuple[int, ...], p: int) -> torch.Tensor:
    """Invert :func:`_fold_cubes` — ``(B, C, n_cubes, p**3)`` → ``(B, C, D, H, W)``."""
    B, C, gd, gh, gw = grid
    t = t.reshape(B, C, gd, gh, gw, p, p, p)
    t = t.permute(0, 1, 2, 5, 3, 6, 4, 7)
    return t.reshape(B, C, gd * p, gh * p, gw * p)


def _as_5d(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Promote a 3D/4D/5D tensor to ``(B, C, D, H, W)``; return it and the original ndim."""
    ndim = x.ndim
    if ndim == 3:
        return x[None, None], ndim
    if ndim == 4:
        return x[None], ndim
    if ndim == 5:
        return x, ndim
    raise ValueError(f"Expected a 3D, 4D, or 5D tensor, got {ndim}D.")


def _restore_ndim(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Undo :func:`_as_5d`."""
    if ndim == 3:
        return x[0, 0]
    if ndim == 4:
        return x[0]
    return x


def normalize_patchwise(
    x: torch.Tensor, patch_size: int, eps: float = 1e-6
) -> torch.Tensor:
    """Normalize each ``patch_size³`` cube of *x* to zero mean and unit variance.

    Args:
        x: Tensor of shape ``(D, H, W)``, ``(C, D, H, W)``, or
            ``(B, C, D, H, W)``. Every spatial dimension must be divisible by
            ``patch_size``.
        patch_size: Edge length of each cube (voxels).
        eps: Added to each cube's std before dividing.

    Returns:
        Tensor of the same shape as *x*.

    Raises:
        ValueError: If a spatial dimension is not divisible by ``patch_size``
            or *x* is not 3D/4D/5D.
    """
    x5d, ndim = _as_5d(x)
    t, grid = _fold_cubes(x5d, patch_size)
    mean = t.mean(dim=-1, keepdim=True)
    std = t.std(dim=-1, keepdim=True)
    t = (t - mean) / (std + eps)
    return _restore_ndim(_unfold_cubes(t, grid, patch_size), ndim)


def denormalize_patchwise(
    x: torch.Tensor,
    reference: torch.Tensor,
    patch_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Invert :func:`normalize_patchwise` using *reference*'s per-cube statistics.

    Rescales each ``patch_size³`` cube of *x* by the corresponding cube's
    ``std + eps`` and ``mean`` computed from *reference*. Use this to bring a
    cube-normalized reconstruction back into the intensity space of the target
    volume.

    Args:
        x: Cube-normalized tensor, shape ``(D, H, W)`` / ``(C, D, H, W)`` /
            ``(B, C, D, H, W)``.
        reference: Tensor of the same shape supplying the per-cube mean/std
            (typically the un-normalized target).
        patch_size: Edge length of each cube (voxels).
        eps: Must match the value used in :func:`normalize_patchwise`.

    Returns:
        Tensor of the same shape as *x*.
    """
    x5d, ndim = _as_5d(x)
    ref5d, _ = _as_5d(reference)
    if x5d.shape != ref5d.shape:
        raise ValueError(
            f"x and reference must have the same shape, got "
            f"{tuple(x.shape)} and {tuple(reference.shape)}."
        )
    xt, grid = _fold_cubes(x5d, patch_size)
    rt, _ = _fold_cubes(ref5d, patch_size)
    mean = rt.mean(dim=-1, keepdim=True)
    std = rt.std(dim=-1, keepdim=True)
    xt = xt * (std + eps) + mean
    return _restore_ndim(_unfold_cubes(xt, grid, patch_size), ndim)
