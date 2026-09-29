# [ANNOTATION] Module docstring describing sub-pixel correspondence refinement via quadratic surface fitting.
"""
Sub-pixel correspondence refinement.

This module refines already established image correspondences using local
image structure. It does not detect features, match descriptors, or perform
geometric verification.

Coordinates follow image convention:
    x = column
    y = row

The initial implementation uses a local quadratic response surface around
each correspondence. The refinement is intentionally conservative: if the
local neighborhood does not support a stable sub-pixel estimate, the original
coordinate is retained.
"""

# [ANNOTATION] Enable future annotation syntax support.
from __future__ import annotations

# [ANNOTATION] Import standard dataclasses and literal typing helpers.
from dataclasses import dataclass
from typing import Literal

# [ANNOTATION] Import NumPy for mathematical operations and response surface fitting.
import numpy as np

# [ANNOTATION] Import CorrespondenceSet container from local types module.
from sih26166.correspondence.types import CorrespondenceSet

# [ANNOTATION] Define supported refinement method literals.
RefinementMethod = Literal["QUADRATIC"]


# [ANNOTATION] Dataclass configuring parameters for local quadratic sub-pixel refinement.
@dataclass(frozen=True)
class SubpixelRefinementConfig:
    """Configuration for local sub-pixel refinement."""

    method: RefinementMethod = "QUADRATIC"
    window_radius: int = 2
    max_offset: float = 0.75
    minimum_curvature: float = 1e-6

    def __post_init__(self) -> None:
        if self.method not in {"QUADRATIC"}:
            raise ValueError(
                f"Unsupported refinement method: {self.method!r}."
            )

        if self.window_radius < 1:
            raise ValueError("window_radius must be at least 1.")

        if self.max_offset <= 0.0:
            raise ValueError("max_offset must be positive.")

        if self.max_offset > self.window_radius:
            raise ValueError(
                "max_offset cannot exceed window_radius."
            )

        if self.minimum_curvature <= 0.0:
            raise ValueError(
                "minimum_curvature must be positive."
            )


# [ANNOTATION] Dataclass storing results of sub-pixel refinement (refined set, mask, offsets).
@dataclass(frozen=True)
class SubpixelRefinementResult:
    """Result of sub-pixel correspondence refinement."""

    correspondences: CorrespondenceSet
    refined_mask: np.ndarray
    offsets: np.ndarray

    @property
    def refined_count(self) -> int:
        """Return the number of correspondences that were refined."""
        return int(np.count_nonzero(self.refined_mask))

    @property
    def total_count(self) -> int:
        """Return the total number of correspondences."""
        return self.correspondences.count


# [ANNOTATION] Helper function validating image array dimensions and converting to float64 grayscale.
def _validate_image(
    image: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Validate and convert an image to floating-point grayscale.

    Supported layouts:
        H x W
        H x W x 1
        H x W x 3
    """
    array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[..., 0]
        elif array.shape[2] == 3:
            array = np.mean(
                array.astype(np.float64),
                axis=2,
            )
        else:
            raise ValueError(
                f"{name} must have 1 or 3 channels."
            )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D grayscale image or an HxWx1/HxWx3 image."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    array = array.astype(
        np.float64,
        copy=False,
    )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


# [ANNOTATION] Helper function estimating peak offset along 1D axis through 3 discrete samples using quadratic fit.
def _quadratic_vertex(
    values: np.ndarray,
    minimum_curvature: float,
) -> float | None:
    """
    Estimate the vertex of a 1-D quadratic through three samples.

    Samples are assumed to lie at x = -1, 0, +1.

    Returns None when the fitted curvature is too small or the result is not
    finite.
    """
    left, center, right = (
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )

    denominator = left - 2.0 * center + right

    if abs(denominator) < minimum_curvature:
        return None

    offset = 0.5 * (left - right) / denominator

    if not np.isfinite(offset):
        return None

    return float(offset)


# [ANNOTATION] Helper function attempting local 3x3 quadratic surface peak location around coordinate (x, y).
def _refine_peak(
    image: np.ndarray,
    x: float,
    y: float,
    config: SubpixelRefinementConfig,
) -> tuple[float, float, bool]:
    """
    Refine a local intensity peak around (x, y).

    The quadratic refinement currently uses a 3x3 neighborhood. The
    window_radius remains part of the public configuration so that future
    refinement strategies can use larger neighborhoods without changing
    the API.
    """
    height, width = image.shape

    center_x = int(round(x))
    center_y = int(round(y))

    if (
        center_x < 1
        or center_x >= width - 1
        or center_y < 1
        or center_y >= height - 1
    ):
        return x, y, False

    patch = image[
        center_y - 1:center_y + 2,
        center_x - 1:center_x + 2,
    ]

    center_value = float(patch[1, 1])

    if not (
        center_value >= float(patch[0, 1])
        and center_value >= float(patch[2, 1])
        and center_value >= float(patch[1, 0])
        and center_value >= float(patch[1, 2])
    ):
        return x, y, False

    dx = _quadratic_vertex(
        patch[1, :],
        config.minimum_curvature,
    )

    dy = _quadratic_vertex(
        patch[:, 1],
        config.minimum_curvature,
    )

    if dx is None or dy is None:
        return x, y, False

    dx = float(
        np.clip(
            dx,
            -config.max_offset,
            config.max_offset,
        )
    )

    dy = float(
        np.clip(
            dy,
            -config.max_offset,
            config.max_offset,
        )
    )

    refined_x = float(center_x + dx)
    refined_y = float(center_y + dy)

    if not (
        np.isfinite(refined_x)
        and np.isfinite(refined_y)
    ):
        return x, y, False

    if (
        refined_x < 0.0
        or refined_x >= width
        or refined_y < 0.0
        or refined_y >= height
    ):
        return x, y, False

    return refined_x, refined_y, True


# [ANNOTATION] Helper function iterating through point array to refine coordinates sub-pixel wise.
def _refine_points(
    image: np.ndarray,
    points: np.ndarray,
    config: SubpixelRefinementConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine an array of image coordinates."""
    refined = np.array(
        points,
        dtype=np.float64,
        copy=True,
    )

    offsets = np.zeros_like(refined)

    refined_mask = np.zeros(
        points.shape[0],
        dtype=bool,
    )

    for index, (x, y) in enumerate(points):
        refined_x, refined_y, success = _refine_peak(
            image,
            float(x),
            float(y),
            config,
        )

        refined[index] = [
            refined_x,
            refined_y,
        ]

        if success:
            offsets[index] = [
                refined_x - float(x),
                refined_y - float(y),
            ]

            refined_mask[index] = True

    return (
        refined,
        offsets,
        refined_mask,
    )


# [ANNOTATION] Primary function refining coordinates on both reference and source sides of correspondence set.
def refine_correspondences(
    reference_image: np.ndarray,
    source_image: np.ndarray,
    correspondences: CorrespondenceSet,
    config: SubpixelRefinementConfig | None = None,
) -> SubpixelRefinementResult:
    """
    Refine both sides of an existing correspondence set.

    Each reference coordinate is refined using local structure in the
    reference image, and each source coordinate is independently refined
    using local structure in the source image.

    A correspondence is marked as refined only when both coordinates were
    successfully refined. If either side cannot be refined safely, the
    original correspondence is retained for that side and the correspondence
    is marked as not fully refined.

    No descriptor matching or geometric verification is performed here.
    """
    if config is None:
        config = SubpixelRefinementConfig()

    print(f"[SUBPIXEL] Refining {correspondences.count} correspondences using strategy '{config.method}'...")

    reference = _validate_image(
        reference_image,
        "reference_image",
    )

    source = _validate_image(
        source_image,
        "source_image",
    )

    if correspondences.count == 0:
        print("[SUBPIXEL] Input correspondence set is empty. Returning empty sub-pixel result.")
        return SubpixelRefinementResult(
            correspondences=CorrespondenceSet.empty(),
            refined_mask=np.empty(
                0,
                dtype=bool,
            ),
            offsets=np.empty(
                (0, 2),
                dtype=np.float64,
            ),
        )

    reference_refined, reference_offsets, reference_mask = _refine_points(
        reference,
        correspondences.reference_points,
        config,
    )

    source_refined, source_offsets, source_mask = _refine_points(
        source,
        correspondences.source_points,
        config,
    )

    refined_mask = reference_mask & source_mask

    offsets = np.zeros(
        (correspondences.count, 2),
        dtype=np.float64,
    )

    offsets[refined_mask] = (
        reference_offsets[refined_mask]
        + source_offsets[refined_mask]
    ) / 2.0

    refined_correspondences = CorrespondenceSet(
        reference_points=reference_refined,
        source_points=source_refined,
        scores=correspondences.scores,
    )

    print(f"[SUBPIXEL] Refinement complete. Successfully refined {np.count_nonzero(refined_mask)} / {correspondences.count} points.")

    return SubpixelRefinementResult(
        correspondences=refined_correspondences,
        refined_mask=refined_mask,
        offsets=offsets,
    )