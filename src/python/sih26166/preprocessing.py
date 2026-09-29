# [ANNOTATION] Module docstring describing configurable image preprocessing methods.
"""Configurable image representations for correspondence experiments."""

# [ANNOTATION] Enable modern type annotations.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclass primitives and typing literals.
from dataclasses import dataclass
from typing import Literal

# [ANNOTATION] Import OpenCV for CLAHE local contrast enhancement and NumPy for percentile operations.
import cv2
import numpy as np

# [ANNOTATION] Define supported local contrast method literals.
LocalContrast = Literal["NONE", "CLAHE"]


# [ANNOTATION] Immutable dataclass configuring image normalization and CLAHE parameters.
@dataclass(frozen=True)
class PreprocessingConfig:
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    local_contrast: LocalContrast = "NONE"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # [ANNOTATION] Post-initialization validation hook enforcing parameter limits.
    def __post_init__(self) -> None:
        if not 0.0 <= self.percentile_low < self.percentile_high <= 100.0:
            raise ValueError("Invalid percentile range.")

        if self.local_contrast not in {"NONE", "CLAHE"}:
            raise ValueError("Unsupported local contrast method.")


# [ANNOTATION] Helper function mapping a numerical image array to 8-bit unsigned integer using percentile stretching.
def percentile_normalize(
    image: np.ndarray,
    *,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Map a numeric image to uint8 using percentile clipping."""
    print(f"[PREPROCESS] Normalizing image dynamic range using percentiles [{low}%, {high}%]...")
    if image.ndim != 2:
        raise ValueError("Expected a single-channel 2-D image.")

    if not 0.0 <= low < high <= 100.0:
        raise ValueError("Invalid percentile range.")

    image_f = np.asarray(image, dtype=np.float32)

    lo, hi = np.percentile(image_f, [low, high])

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Image percentiles are not finite.")

    if hi <= lo:
        print("[PREPROCESS] Flat intensity distribution detected. Returning zeros array.")
        return np.zeros(image.shape, dtype=np.uint8)

    clipped = np.clip(image_f, lo, hi)
    normalized = (clipped - lo) / (hi - lo)

    return np.rint(normalized * 255.0).astype(np.uint8)


# [ANNOTATION] Primary function creating preprocessed image representation optimized for feature extraction.
def preprocess_for_features(
    image: np.ndarray,
    *,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Create an explicit feature-detection representation.

    The input image is never modified.
    """
    config = config or PreprocessingConfig()

    print(f"[PREPROCESS] Preparing feature representation (Local Contrast: {config.local_contrast})...")

    result = percentile_normalize(
        image,
        low=config.percentile_low,
        high=config.percentile_high,
    )

    # [ANNOTATION] Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) if requested.
    if config.local_contrast == "CLAHE":
        print(
            f"[PREPROCESS] Applying CLAHE contrast enhancement "
            f"(Clip limit: {config.clahe_clip_limit}, Grid: {config.clahe_tile_grid})..."
        )
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_grid,
        )
        result = clahe.apply(result)

    print("[PREPROCESS] Image preprocessing complete.")

    return result