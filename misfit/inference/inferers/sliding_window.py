"""Sliding-window inferer for MISFIT using MONAI.

Use this when the volume is larger than the model's native patch size.
MONAI tiles the volume into overlapping patches, runs each through the
model, and blends the outputs back into a full-resolution prediction.

Typical use case: extracting encoder features over an entire, un-cropped
volume by running ``model.encoder`` patch-by-patch.
"""
from typing import Callable, Optional, Tuple, Union

import monai
import torch

from misfit.inference.inference_constants import ic
from misfit.inference.inferers.base import AbstractInferer
from misfit.inference.inferers.inferer_registry import register_inferer
from misfit.inference.inference_utils import get_default_device


@register_inferer("sliding_window")
class SlidingWindowInferer(AbstractInferer):
    """Sliding-window inference via MONAI's built-in API.

    Args:
        patch_size: Size of each inference patch (D, H, W).
        patch_overlap: Fractional overlap between adjacent patches in
            ``[0, 1)``. Higher overlap → smoother blending but slower.
            Defaults to :attr:`InferenceConstants.DEFAULT_PATCH_OVERLAP`.
        patch_blend_mode: How overlapping patch predictions are blended.
            ``"gaussian"`` weights the centre voxels of each patch more
            heavily; ``"constant"`` weights all voxels equally.
            Defaults to :attr:`InferenceConstants.DEFAULT_BLEND_MODE`.
        device: Inference device. Defaults to :func:`get_default_device`.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        patch_overlap: float = ic.DEFAULT_PATCH_OVERLAP,
        patch_blend_mode: str = ic.DEFAULT_BLEND_MODE,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if len(patch_size) != 3 or not all(isinstance(d, int) and d > 0 for d in patch_size):
            raise ValueError(
                f"patch_size must be three positive ints, got: {patch_size}"
            )
        if not 0 <= patch_overlap < 1:
            raise ValueError(
                f"patch_overlap must be in [0, 1), got: {patch_overlap}"
            )
        if patch_blend_mode not in ic.SLIDING_WINDOW_PATCH_BLEND_MODES:
            raise ValueError(
                f"Unsupported blend mode '{patch_blend_mode}'. "
                f"Supported: {sorted(ic.SLIDING_WINDOW_PATCH_BLEND_MODES)}"
            )

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_blend_mode = patch_blend_mode
        self.device = device or get_default_device()

    def infer(
        self,
        image: torch.Tensor,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply sliding-window inference.

        Args:
            image: Input tensor of shape ``(1, C, D, H, W)``.
            model_fn: Callable mapping a patch tensor to an output tensor.

        Returns:
            Blended output tensor matching the spatial dims of *image*.
        """
        image = image.to(self.device)
        with torch.no_grad():
            return monai.inferers.sliding_window_inference(  # type: ignore[attr-defined]
                inputs=image,
                roi_size=self.patch_size,
                sw_batch_size=ic.SLIDING_WINDOW_BATCH_SIZE,
                predictor=model_fn,
                overlap=self.patch_overlap,
                mode=self.patch_blend_mode,
                device=self.device,
            )
