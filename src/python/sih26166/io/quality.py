# [ANNOTATION] Module docstring describing quality diagnostics for extracted real image windows.
"""
Quality diagnostics for extracted real Chandrayaan-2 image windows.

These diagnostics are descriptive only. They do not claim that a patch
is scientifically suitable for correspondence; they provide measurable
properties that can be used to select and document candidate windows.
"""

# [ANNOTATION] Enable modern type hint annotations.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclass decorator.
from dataclasses import dataclass

# [ANNOTATION] Import NumPy for statistical and matrix operations.
import numpy as np


# [ANNOTATION] Dataclass storing image window quality metrics (means, standard deviations, texture gradients).
@dataclass(frozen=True)
class ImageQualityMetrics:
    """Descriptive statistics for one grayscale image window."""

    width: int
    height: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    median: float
    nonzero_fraction: float
    gradient_mean: float
    gradient_standard_deviation: float

    # [ANNOTATION] Helper property returning intensity dynamic range (max - min).
    @property
    def dynamic_range(self) -> float:
        """Return maximum minus minimum."""
        return self.maximum - self.minimum


# [ANNOTATION] Function computing statistical and texture quality metrics for a grayscale image window.
def analyze_window(pixels: np.ndarray) -> ImageQualityMetrics:
    """
    Compute descriptive quality statistics for a grayscale image window.

    The input is not modified.

    Gradient statistics use simple first-order finite differences in
    the horizontal and vertical directions. They are intended as
    texture/edge diagnostics, not as a scientific image-quality metric.
    """
    print("[QUALITY] Computing quality diagnostics and gradient statistics for image window...")
    array = np.asarray(pixels)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale image, got shape {array.shape}."
        )

    if array.size == 0:
        raise ValueError("Image window must not be empty.")

    values = array.astype(np.float32, copy=False)

    dx = np.diff(values, axis=1)
    dy = np.diff(values, axis=0)

    if dx.size == 0 and dy.size == 0:
        gradient_mean = 0.0
        gradient_std = 0.0
    else:
        gradients = np.concatenate(
            (
                np.abs(dx).ravel(),
                np.abs(dy).ravel(),
            )
        )

        gradient_mean = float(np.mean(gradients))
        gradient_std = float(np.std(gradients))

    metrics = ImageQualityMetrics(
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values)),
        median=float(np.median(values)),
        nonzero_fraction=float(np.count_nonzero(values) / values.size),
        gradient_mean=gradient_mean,
        gradient_standard_deviation=gradient_std,
    )

    print(
        f"[QUALITY] Quality analysis complete:\n"
        f"          Range: [{metrics.minimum:.1f}, {metrics.maximum:.1f}], Mean: {metrics.mean:.2f}, Std: {metrics.standard_deviation:.2f}\n"
        f"          Gradient Mean: {metrics.gradient_mean:.2f}"
    )

    return metrics