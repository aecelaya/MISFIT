"""Predictor: chains an inferer, TTA transforms, and an ensembler.

This is the core inference object in MISFIT, mirroring MIST's Predictor.
It is agnostic to the output type — wrap the model callable appropriately
before passing it in:

    # Reconstruction
    model_fn = lambda x: model(x)["reconstruction"]
    predictor = Predictor(model_fn=model_fn, inferer=..., ...)

    # Encoder bottleneck features
    model_fn = lambda x: model.encoder(x)[-1]
    predictor = Predictor(model_fn=model_fn, inferer=..., ...)

TTA transforms are applied to the *input* and their inverses are applied to
the *output* so the final tensor is always in the original coordinate frame.
The ensembler then averages across all TTA variants.
"""
from typing import Callable, List, Optional, Union

import torch

from misfit.inference.ensemblers.base import AbstractEnsembler
from misfit.inference.inferers.base import AbstractInferer
from misfit.inference.inference_utils import get_default_device
from misfit.inference.tta.transforms import AbstractTransform


class Predictor:
    """Performs inference with optional TTA and ensembling.

    Args:
        model_fn: Callable that maps a ``(1, C, D, H, W)`` tensor to an
            output tensor.  Wrap the model as needed (e.g.
            ``lambda x: model(x)["reconstruction"]``).
        inferer: An :class:`~misfit.inference.inferers.base.AbstractInferer`
            instance that controls *how* the model is applied
            (whole-volume or sliding window).
        ensembler: An
            :class:`~misfit.inference.ensemblers.base.AbstractEnsembler`
            instance that aggregates the TTA predictions.
        tta_transforms: List of TTA
            :class:`~misfit.inference.tta.transforms.AbstractTransform`
            instances.  Pass ``[IdentityTransform()]`` for no TTA.
        device: Inference device. Defaults to :func:`get_default_device`.
    """

    def __init__(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        inferer: AbstractInferer,
        ensembler: AbstractEnsembler,
        tta_transforms: List[AbstractTransform],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model_fn = model_fn
        self.inferer = inferer
        self.ensembler = ensembler
        self.tta_transforms = tta_transforms
        self.device = device or get_default_device()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Invoke :meth:`predict`."""
        return self.predict(image)

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference with TTA and ensembling.

        For each TTA transform:
            1. Apply the forward transform to the image.
            2. Run the inferer.
            3. Apply the inverse transform to the output.

        Then ensemble all TTA outputs into a single tensor.

        Args:
            image: Input tensor of shape ``(1, C, D, H, W)``.

        Returns:
            Ensembled output tensor in the original coordinate frame.
        """
        image = image.to(self.device)
        predictions = []

        for transform in self.tta_transforms:
            augmented = transform(image)
            output = self.inferer(augmented, self.model_fn)
            predictions.append(transform.inverse(output))

        return self.ensembler(predictions)
