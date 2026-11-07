"""Primus with a 3D Masked Autoencoder (MAE) objective.

Architecture overview:

  1. Tokenize  — MIST's Primus tokenizer turns the volume into one token per
                 8x8x8 voxels.
  2. Masking   — mask_ratio of the 3D cubes (size mask_patch_size^3) are chosen
                 and their tokens are removed, so the encoder only sees visible
                 tokens.
  3. Encoder   — MIST's Primus transformer processes the visible tokens.
  4. Decoder   — mask tokens are put back in place of the removed tokens, and a
                 small transformer plus MIST's Primus patch decoder reconstruct
                 the image.
  5. Loss      — computed by the training loop over masked voxels only.

This is the recipe nnssl uses for Primus pretraining (``EvaMAE``). Unlike the
image-level masking of SwinMAE and ResEncMAE, masked regions never reach the
encoder. That needs non-overlapping tokens, which is why only Primus (V1) is
registered: PrimusV2's tokenizer has overlapping receptive fields, so visible
tokens would leak masked voxels.

Transfer learning compatibility:
    Parameter names match nnssl's ``EvaMAE``, and the encoder is held under
    MIST's own names (``down_projection.*`` and ``eva.*``), so
    get_encoder_state_dict() can be saved with torch.save and passed directly
    to ``mist_train --pretrained-weights``. In the other direction, nnssl PrimusX
    checkpoints such as nnFoundationViT load into the whole network (see
    :meth:`PrimusMAE.adapt_pretrained_state_dict` and
    :mod:`misfit.models.pretrained`).
"""

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mist.models.primus.eva_blocks import Eva
from mist.models.primus.mist_primus import Primus
from mist.models.primus.primus_constants import PrimusConstants

from misfit.models.base_model import MISFITModel
from misfit.models.masking import generate_patch_mask

# Encoder submodules, named as in MIST's Primus and nnssl's EvaMAE.
ENCODER_PREFIXES = ("down_projection.", "eva.")
POSITION_EMBEDDING_KEYS = ("eva.pos_embed", "decoder.pos_embed")


def resize_position_embedding(
    embedding: torch.Tensor, grid_shape: tuple[int, int, int]
) -> torch.Tensor:
    """Trilinearly resize a learned position embedding to a new token grid.

    Args:
        embedding: Position embedding of shape (1, num_tokens, dim).
        grid_shape: Target token grid.

    Returns:
        The resized embedding, or the input unchanged if it already has the
        right number of tokens or its source grid isn't cubic (the source grid
        can't be recovered from the token count, so the strict loader reports
        the shape mismatch).
    """
    num_tokens = grid_shape[0] * grid_shape[1] * grid_shape[2]
    side = round(embedding.shape[1] ** (1 / 3))
    if embedding.shape[1] == num_tokens or side**3 != embedding.shape[1]:
        return embedding
    grid = embedding.reshape(1, side, side, side, -1).permute(0, 4, 1, 2, 3)
    resized = F.interpolate(
        grid.float(), size=grid_shape, mode="trilinear", align_corners=False
    )
    return resized.permute(0, 2, 3, 4, 1).reshape(1, num_tokens, -1).to(embedding.dtype)


class PrimusMAE(MISFITModel):
    """Primus with a 3D Masked Autoencoder pretraining objective.

    Spatial input requirements:
        - mask_patch_size must be a multiple of the 8-voxel token size, so that
          each masked cube covers whole tokens.
        - All img_size dimensions must be divisible by mask_patch_size (and so
          by the token size).
        - The position embeddings are sized for img_size, so the model only
          accepts inputs of exactly that size.

    Args:
        embed_dim: Transformer width.
        depth: Number of encoder transformer blocks.
        num_heads: Number of encoder attention heads.
        decoder_depth: Number of decoder transformer blocks.
        decoder_num_heads: Number of decoder attention heads.
        in_channels: Number of input image channels. Defaults to 1.
        img_size: Spatial dimensions of the input volume. Defaults to
            (96, 96, 96).
        mask_patch_size: Edge length (in voxels) of each masked 3D cube.
            Defaults to 16.
        mask_ratio: Fraction of cubes to mask during training. Defaults to 0.75.

    Raises:
        ValueError: If mask_patch_size isn't a multiple of the token size, or
            any img_size dimension isn't divisible by mask_patch_size.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        decoder_depth: int,
        decoder_num_heads: int,
        in_channels: int = 1,
        img_size: tuple[int, int, int] = (96, 96, 96),
        mask_patch_size: int = 16,
        mask_ratio: float = 0.75,
        **kwargs: Any,
    ):
        super().__init__()

        self.token_size = PrimusConstants.TOKEN_SIZE
        if mask_patch_size % self.token_size != 0:
            raise ValueError(
                f"mask_patch_size must be a multiple of the {self.token_size}-voxel "
                f"Primus token size. Got mask_patch_size={mask_patch_size}."
            )
        for dim in img_size:
            if dim % mask_patch_size != 0:
                raise ValueError(
                    f"All img_size dimensions must be divisible by "
                    f"mask_patch_size={mask_patch_size}. Got img_size={img_size}."
                )

        self.in_channels = in_channels
        self.img_size = img_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio

        # Build MIST's Primus with one output channel per input channel, so its
        # patch decoder reconstructs the image. Its mask_token buffer is an
        # unused placeholder; this model's is learned.
        _mist_primus = Primus(
            in_channels=in_channels,
            out_channels=in_channels,
            patch_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.grid_shape = _mist_primus.grid_shape
        self.down_projection = _mist_primus.down_projection
        self.eva = _mist_primus.eva
        self.up_projection = _mist_primus.up_projection

        self.decoder = Eva(
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            grid_shape=self.grid_shape,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)

    def generate_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> torch.Tensor:
        """Generate a random patch-level binary mask.

        Args:
            x: Input tensor of shape (B, C, D, H, W). Only batch size,
                spatial shape, and device are used.
            mask_ratio: Fraction of patches to mask. If None, uses the value
                set at construction (``self.mask_ratio``). Defaults to None.

        Returns:
            Float mask tensor of shape (B, 1, D, H, W), where 1.0 = masked
            and 0.0 = visible.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        return generate_patch_mask(x, self.mask_patch_size, mask_ratio)

    def _visible_token_indices(self, mask: torch.Tensor) -> torch.Tensor:
        """Return the visible tokens' indices, shape (B, num_visible).

        Every sample masks the same number of cubes, so the result is regular.

        Raises:
            ValueError: If every token is masked; the encoder needs at least one.
        """
        step = self.token_size
        token_is_visible = mask[:, 0, ::step, ::step, ::step].flatten(1) == 0
        batch_size = mask.shape[0]
        indices = token_is_visible.nonzero()[:, 1].reshape(batch_size, -1)
        if indices.shape[1] == 0:
            raise ValueError("mask_ratio masks every token; nothing is left to encode.")
        return indices

    def _encode_visible(self, tokens: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """Run the encoder on the visible tokens only.

        MIST's Eva.forward has no token dropping, so this runs its pieces in the
        same order as the Eva nnssl uses: position embedding, drop tokens, then
        blocks with each sample's rotary embeddings, then the final norm.

        Args:
            tokens: All tokens, shape (B, N, C).
            visible: Visible token indices, shape (B, num_visible).

        Returns:
            Encoded visible tokens, shape (B, num_visible, C).
        """
        x = tokens + self.eva.pos_embed
        x = x.gather(1, visible.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        # (B, num_visible, E) -> (B, 1, num_visible, E), broadcast over heads.
        rope = self.eva.rope.get_embed()[visible].unsqueeze(1)
        for block in self.eva.blocks:
            x = block(x, rope=rope)
        return self.eva.norm(x)

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W). All spatial dimensions
                must match img_size.
            mask_ratio: Override the instance mask_ratio for this call. If
                None, uses the value set at construction. Defaults to None.

        Returns:
            Dict with:
                ``"reconstruction"``: Reconstructed volume (B, C, D, H, W).
                ``"mask"``: Binary mask (B, 1, D, H, W); 1 = masked voxels.
        """
        mask = self.generate_mask(x, mask_ratio=mask_ratio)
        visible = self._visible_token_indices(mask)

        tokens = self.down_projection(x)
        b, c, d, h, w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        encoded = self._encode_visible(tokens, visible)

        # Put the encoded tokens back in place, with mask tokens everywhere else.
        restored = self.mask_token.expand(b, tokens.shape[1], -1).scatter(
            1, visible.unsqueeze(-1).expand(-1, -1, c), encoded
        )
        decoded = self.decoder(restored).transpose(1, 2).reshape(b, c, d, h, w)
        return {"reconstruction": self.up_projection(decoded), "mask": mask}

    def encoder(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode a whole volume, with no masking.

        A method rather than a module, as parameter names must stay those of
        MIST and nnssl. It has the same contract as the other models' encoders:
        the last element is the spatial bottleneck.

        Args:
            x: Input tensor of shape (B, C, D, H, W). All spatial dimensions
                must match img_size.

        Returns:
            A one-element list holding the token feature map
            (B, embed_dim, D/8, H/8, W/8).
        """
        tokens = self.down_projection(x)
        b, c, d, h, w = tokens.shape
        encoded = self.eva(tokens.flatten(2).transpose(1, 2))
        return [encoded.transpose(1, 2).reshape(b, c, d, h, w)]

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder-only weights for downstream transfer learning.

        Keys are ``down_projection.<name>`` and ``eva.<name>``, exactly as in
        MIST's ``Primus``, so the state dict can be saved with ``torch.save``
        and passed to ``mist_train --pretrained-weights`` without key
        manipulation.

        Returns:
            OrderedDict mapping parameter name → tensor.
        """
        return OrderedDict(
            (k, v) for k, v in self.state_dict().items() if k.startswith(ENCODER_PREFIXES)
        )

    def adapt_pretrained_state_dict(self, state_dict: OrderedDict) -> OrderedDict:
        """Resize learned position embeddings pretrained at another patch size.

        Other parameter names and shapes don't depend on the patch size and
        already match nnssl's, so nothing else needs mapping.

        Args:
            state_dict: Checkpoint state dict.

        Returns:
            The state dict with its position embeddings fitted to this model.
        """
        adapted = OrderedDict(state_dict)
        for key in POSITION_EMBEDDING_KEYS:
            if key in adapted:
                adapted[key] = resize_position_embedding(adapted[key], self.grid_shape)
        return adapted
