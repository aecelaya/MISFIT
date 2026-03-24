"""Constants for reconstruction metric computation."""


class ReconstructionMetricsConstants:
    """Constants for reconstruction metric computation.

    All metrics operate on z-score normalised volumes (mean 0, std 1 over
    the foreground), so intensity values are not bounded to [0, 1].  The
    SSIM data range is therefore set to the expected spread of normalised
    intensities rather than a fixed [0, 1] window.
    """

    # Expected intensity range for SSIM after clip + z-score normalisation.
    # With p1/p99 clipping the range is approximately [-3, 3].
    SSIM_DATA_RANGE: float = 6.0

    # Small epsilon to prevent log(0) in PSNR.
    PSNR_EPS: float = 1e-8

    # Small epsilon to prevent division by zero in masked metrics.
    MASK_EPS: float = 1e-8
