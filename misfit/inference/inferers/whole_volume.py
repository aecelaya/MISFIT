"""Whole-volume inferer for MISFIT.

Pads / centre-crops the input to the model's native patch size and runs a
single forward pass.  Use this when the volume already fits in GPU memory
at the required resolution.  For volumes larger than the patch size, prefer
:class:`~misfit.inference.inferers.sliding_window.SlidingWindowInferer`.
"""
from contextlib import nullcontext
from typing import Callable, Optional, Tuple, Union

import torch

from misfit.inference.inferers.base import AbstractInferer
from misfit.inference.inferers.inferer_registry import register_inferer
from misfit.inference.inference_utils import get_default_device


@register_inferer("whole_volume")
class WholeVolumeInferer(AbstractInferer):
    """Single-pass inference on a whole (possibly cropped) volume.

    The input is expected to have been padded / cropped to *patch_size*
    before calling this inferer.  It simply forwards the tensor through
    *model_fn* under an optional AMP context.

    Args:
        patch_size: Expected spatial dimensions (D, H, W).
        amp: Use ``torch.amp.autocast`` during inference. Defaults to False.
        device: Inference device. Defaults to :func:`get_default_device`.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        amp: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if len(patch_size) != 3 or not all(isinstance(d, int) and d > 0 for d in patch_size):
            raise ValueError(
                f"patch_size must be three positive ints, got: {patch_size}"
            )
        self.patch_size = patch_size
        self.amp = amp
        self.device = device or get_default_device()

    def infer(
        self,
        image: torch.Tensor,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run a single forward pass.

        Args:
            image: Input tensor of shape ``(1, C, D, H, W)``.
            model_fn: Callable mapping the tensor to an output tensor.

        Returns:
            Output tensor from *model_fn*.
        """
        image = image.to(self.device)
        amp_ctx = torch.amp.autocast("cuda") if self.amp else nullcontext()
        with torch.no_grad(), amp_ctx:
            return model_fn(image)
