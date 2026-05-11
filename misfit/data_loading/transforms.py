"""MONAI transform pipelines for MISFIT data loading.

Two pipelines are provided:
    build_train_transforms — random crop + flips + light intensity augmentation.
    build_val_transforms   — deterministic center crop, no augmentation.

Both pipelines operate on a single-channel 4D tensor (1, D, H, W) that has
already been clip-normalized by the Dataset.__getitem__ method.

Spatial padding is applied before cropping so that volumes smaller than
patch_size (e.g., small organ ROIs) are handled without errors.
"""

from monai.transforms import (
    CenterSpatialCrop,
    Compose,
    EnsureType,
    RandFlip,
    RandGaussianNoise,
    RandScaleIntensity,
    RandSpatialCrop,
    SpatialPad,
)

from misfit.data_loading.data_loading_constants import dc


def build_train_transforms(
    patch_size: tuple[int, int, int] = (96, 96, 96),
) -> Compose:
    """Build the training transform pipeline.

    Operations (in order):
        1. SpatialPad     — ensure volume is at least patch_size in each dim.
        2. RandSpatialCrop — randomly sample a patch_size sub-volume.
        3. RandFlip (x3)  — independent 50% flips along each spatial axis.
        4. RandGaussianNoise — light scanner-noise simulation (15% prob).
        5. RandScaleIntensity — mild brightness variation (15% prob).
        6. EnsureType     — cast to float32 torch.Tensor.

    Args:
        patch_size: Output spatial dimensions (D, H, W). Must match
            SwinMAE.img_size. Defaults to (96, 96, 96).

    Returns:
        MONAI Compose transform applied to a (1, D, H, W) tensor.
    """
    return Compose([
        # Pad volumes smaller than the crop size.
        SpatialPad(spatial_size=patch_size, mode="constant", constant_values=0),

        # Random 3D crop to fixed patch_size.
        RandSpatialCrop(roi_size=patch_size, random_size=False),

        # Spatial flips — cheap, modality-agnostic.
        RandFlip(prob=dc.FLIP_PROB, spatial_axis=0),
        RandFlip(prob=dc.FLIP_PROB, spatial_axis=1),
        RandFlip(prob=dc.FLIP_PROB, spatial_axis=2),

        # Light intensity augmentation.
        RandGaussianNoise(
            prob=dc.NOISE_PROB,
            mean=0.0,
            std=dc.NOISE_STD_MAX,
        ),
        RandScaleIntensity(
            factors=dc.BRIGHTNESS_FACTOR,
            prob=dc.BRIGHTNESS_PROB,
        ),

        EnsureType(dtype="float32"),
    ])


def build_val_transforms(
    patch_size: tuple[int, int, int] = (96, 96, 96),
) -> Compose:
    """Build the validation transform pipeline.

    Operations (in order):
        1. SpatialPad      — ensure volume is at least patch_size in each dim.
        2. CenterSpatialCrop — deterministic center crop to patch_size.
        3. EnsureType      — cast to float32 torch.Tensor.

    Args:
        patch_size: Output spatial dimensions (D, H, W). Must match
            SwinMAE.img_size. Defaults to (96, 96, 96).

    Returns:
        MONAI Compose transform applied to a (1, D, H, W) tensor.
    """
    return Compose([
        SpatialPad(spatial_size=patch_size, mode="constant", constant_values=0),
        CenterSpatialCrop(roi_size=patch_size),
        EnsureType(dtype="float32"),
    ])
