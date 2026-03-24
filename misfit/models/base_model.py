"""Base class for all MISFIT model wrappers."""
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn


class MISFITModel(nn.Module, ABC):
    """Abstract base class for all MISFIT model wrappers.

    All MISFIT model wrappers must inherit from this class and implement
    get_encoder_state_dict(). This interface is the contract between MISFIT
    pretraining and downstream transfer learning (e.g., MIST fine-tuning).
    """

    @abstractmethod
    def get_encoder_state_dict(self) -> OrderedDict:
        """Return the encoder weights as a state dict.

        Keys must match those returned by self.state_dict() exactly, with
        the 'encoder.' prefix stripped so that the returned names correspond
        directly to the sub-module parameter names (e.g., 'patch_embed.proj.weight').
        Only encoder weights are included — SSL heads and decoders are excluded.

        Returns:
            OrderedDict of encoder parameter name → tensor.
        """
