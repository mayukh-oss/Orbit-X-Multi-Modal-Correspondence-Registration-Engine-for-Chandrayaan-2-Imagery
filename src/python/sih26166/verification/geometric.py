# [ANNOTATION] Module docstring describing RANSAC robust geometric verification.
"""
Robust geometric verification of image correspondences.

This module estimates a geometric transformation from descriptor-level
correspondences and identifies geometrically consistent inliers using
RANSAC.

Supported models:
    - AFFINE
    - HOMOGRAPHY

The implementation deliberately keeps geometric verification separate from
descriptor matching. A descriptor match is not considered a valid geometric
correspondence until it passes this stage.
"""

# [ANNOTATION] Enable modern type hint syntax.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclasses and typing literals.
from dataclasses import dataclass
from typing import Literal

# [ANNOTATION] Import OpenCV for RANSAC estimation and NumPy for array math.
import cv2
import numpy as np

# [ANNOTATION] Import CorrespondenceSet dataclass from local types module.
from ..correspondence.types import CorrespondenceSet

# [ANNOTATION] Define supported geometric model literals.
GeometricModel = Literal["AFFINE", "HOMOGRAPHY"]


# [ANNOTATION] Immutable dataclass configuring RANSAC thresholds and geometric model parameters.
@dataclass(frozen=True)
class GeometricVerificationConfig:
    """Configuration for robust geometric verification."""

    model: GeometricModel = "AFFINE"
    reprojection_threshold: float = 3.0
    confidence: float = 0.995
    max_iterations: int = 5000
    min_inliers: int = 4

    # [ANNOTATION] Post-initialization validation hook enforcing parameter bounds.
    def __post_init__(self) -> None:
        if self.model not in {"AFFINE", "HOMOGRAPHY"}:
            raise ValueError(
                f"Unsupported geometric model: {self.model!r}."
            )

        if not isinstance(
            self.reprojection_threshold,
            (int, float),
        ):
            raise TypeError(
                "reprojection_threshold must be numeric."
            )

        if self.reprojection_threshold <= 0:
            raise ValueError(
                "reprojection_threshold must be greater than zero."
            )

        if not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be numeric.")

        if not 0.0 < float(self.confidence) < 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if not isinstance(self.max_iterations, int) or isinstance(
            self.max_iterations,
            bool,
        ):
            raise TypeError(
                "max_iterations must be an integer."
            )

        if self.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero."
            )

        if not isinstance(self.min_inliers, int) or isinstance(
            self.min_inliers,
            bool,
        ):
            raise TypeError(
                "min_inliers must be an integer."
            )

        minimum_required = 3 if self.model == "AFFINE" else 4

        if self.min_inliers < minimum_required:
            raise ValueError(
                f"min_inliers must be at least {minimum_required} "
                f"for the {self.model} model."
            )


# [ANNOTATION] Immutable dataclass storing RANSAC estimation outputs, inlier masks, and residual errors.
@dataclass(frozen=True)
class GeometricVerificationResult:
    """Result of robust geometric verification."""

    model: GeometricModel
    transformation: np.ndarray | None
    inlier_mask: np.ndarray
    residuals: np.ndarray
    total_matches: int

    # [ANNOTATION] Property returning total geometrically verified inlier count.
    @property
    def inlier_count(self) -> int:
        """Return the number of geometrically consistent matches."""
        return int(np.count_nonzero(self.inlier_mask))

    # [ANNOTATION] Property returning inlier ratio.
    @property
    def inlier_ratio(self) -> float:
        """Return the fraction of matches classified as inliers."""
        if self.total_matches == 0:
            return 0.0

        return self.inlier_count / self.total_matches

    # [ANNOTATION] Property indicating overall verification success.
    @property
    def success(self) -> bool:
        """Return whether a valid model with sufficient inliers was found."""
        return (
            self.transformation is not None
            and self.inlier_count > 0
        )


# [ANNOTATION] Helper function determining minimum point correspondence count needed for model fitting.
def _minimum_points(model: GeometricModel) -> int:
    """Return the minimum number of point correspondences required."""

    if model == "AFFINE":
        return 3

    return 4


# [ANNOTATION] Helper function validating correspondence input matrices.
def _validate_correspondences(
    correspondences: CorrespondenceSet,
) -> None:
    """Validate correspondence data before model estimation."""

    if not isinstance(correspondences, CorrespondenceSet):
        raise TypeError(
            "correspondences must be a CorrespondenceSet."
        )

    if correspondences.reference_points.shape != (
        correspondences.count,
        2,
    ):
        raise ValueError(
            "reference_points must have shape (N, 2)."
        )

    if correspondences.source_points.shape != (
        correspondences.count,
        2,
    ):
        raise ValueError(
            "source_points must have shape (N, 2)."
        )


# [ANNOTATION] Helper function estimating 2x3 affine matrix via OpenCV estimateAffine2D RANSAC.
def _estimate_affine(
    reference_points: np.ndarray,
    source_points: np.ndarray,
    config: GeometricVerificationConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate an affine transformation using RANSAC."""

    print("[GEOMETRIC] Estimating AFFINE matrix using cv2.estimateAffine2D (RANSAC)...")
    transformation, mask = cv2.estimateAffine2D(
        reference_points,
        source_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.reprojection_threshold),
        maxIters=int(config.max_iterations),
        confidence=float(config.confidence),
        refineIters=10,
    )

    if transformation is None or mask is None:
        print("[GEOMETRIC] Warning: RANSAC failed to find an affine model.")
        return None, None

    return (
        np.asarray(transformation, dtype=np.float64),
        np.asarray(mask, dtype=bool).reshape(-1),
    )


# [ANNOTATION] Helper function estimating 3x3 homography matrix via OpenCV findHomography RANSAC.
def _estimate_homography(
    reference_points: np.ndarray,
    source_points: np.ndarray,
    config: GeometricVerificationConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate a projective homography using RANSAC."""

    print("[GEOMETRIC] Estimating HOMOGRAPHY matrix using cv2.findHomography (RANSAC)...")
    transformation, mask = cv2.findHomography(
        reference_points,
        source_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.reprojection_threshold),
        maxIters=int(config.max_iterations),
        confidence=float(config.confidence),
    )

    if transformation is None or mask is None:
        print("[GEOMETRIC] Warning: RANSAC failed to find a homography model.")
        return None, None

    return (
        np.asarray(transformation, dtype=np.float64),
        np.asarray(mask, dtype=bool).reshape(-1),
    )


# [ANNOTATION] Helper function transforming reference 2D points to source coordinates using estimated model.
def _transform_points(
    points: np.ndarray,
    transformation: np.ndarray,
    model: GeometricModel,
) -> np.ndarray:
    """Transform points using an affine matrix or homography."""

    points = np.asarray(points, dtype=np.float64)

    if model == "AFFINE":
        homogeneous = np.column_stack(
            (
                points,
                np.ones(points.shape[0], dtype=np.float64),
            )
        )

        return (homogeneous @ transformation.T)[:, :2]

    transformed = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2),
        transformation,
    )

    return transformed.reshape(-1, 2)


# [ANNOTATION] Helper function calculating reprojection residual errors for every correspondence point pair.
def _compute_residuals(
    reference_points: np.ndarray,
    source_points: np.ndarray,
    transformation: np.ndarray | None,
    model: GeometricModel,
) -> np.ndarray:
    """Compute Euclidean reprojection residuals in source-image pixels."""

    if transformation is None:
        return np.full(
            reference_points.shape[0],
            np.inf,
            dtype=np.float64,
        )

    try:
        predicted = _transform_points(
            reference_points,
            transformation,
            model,
        )
    except (cv2.error, ValueError, FloatingPointError):
        return np.full(
            reference_points.shape[0],
            np.inf,
            dtype=np.float64,
        )

    residuals = np.linalg.norm(
        predicted - source_points,
        axis=1,
    )

    residuals[~np.isfinite(residuals)] = np.inf

    return residuals


# [ANNOTATION] Primary function executing RANSAC robust geometric verification and inlier filtering.
def verify_geometry(
    correspondences: CorrespondenceSet,
    *,
    config: GeometricVerificationConfig | None = None,
) -> GeometricVerificationResult:
    """Estimate a robust geometric model and classify inliers.

    Parameters
    ----------
    correspondences:
        Descriptor-level correspondences between reference and source images.
    config:
        RANSAC and geometric-model configuration.

    Returns
    -------
    GeometricVerificationResult
        Estimated transformation, inlier mask, residuals, and summary
        statistics.

    Notes
    -----
    The transformation maps reference-image coordinates to
    source-image coordinates.

    The returned residuals are Euclidean reprojection errors measured in
    source-image pixels.
    """

    if config is None:
        config = GeometricVerificationConfig()

    _validate_correspondences(correspondences)

    reference_points = correspondences.reference_points
    source_points = correspondences.source_points

    total_matches = correspondences.count
    minimum_required = _minimum_points(config.model)

    print(
        f"[GEOMETRIC] Verifying geometry for {total_matches} matches "
        f"(Model: {config.model}, Threshold: {config.reprojection_threshold}px)..."
    )

    empty_mask = np.zeros(
        total_matches,
        dtype=bool,
    )

    empty_residuals = np.full(
        total_matches,
        np.inf,
        dtype=np.float64,
    )

    if total_matches < minimum_required:
        print(
            f"[GEOMETRIC] Insufficient matches ({total_matches}) for {config.model} "
            f"model (requires at least {minimum_required})."
        )
        return GeometricVerificationResult(
            model=config.model,
            transformation=None,
            inlier_mask=empty_mask,
            residuals=empty_residuals,
            total_matches=total_matches,
        )

    if config.model == "AFFINE":
        transformation, ransac_mask = _estimate_affine(
            reference_points,
            source_points,
            config,
        )
    else:
        transformation, ransac_mask = _estimate_homography(
            reference_points,
            source_points,
            config,
        )

    if transformation is None or ransac_mask is None:
        return GeometricVerificationResult(
            model=config.model,
            transformation=None,
            inlier_mask=empty_mask,
            residuals=empty_residuals,
            total_matches=total_matches,
        )

    residuals = _compute_residuals(
        reference_points,
        source_points,
        transformation,
        config.model,
    )

    inlier_mask = (
        ransac_mask
        & np.isfinite(residuals)
        & (residuals <= config.reprojection_threshold)
    )

    inlier_count = int(np.count_nonzero(inlier_mask))

    if inlier_count < config.min_inliers:
        print(
            f"[GEOMETRIC] RANSAC inlier count ({inlier_count}) below min_inliers threshold ({config.min_inliers})."
        )
        return GeometricVerificationResult(
            model=config.model,
            transformation=transformation,
            inlier_mask=inlier_mask,
            residuals=residuals,
            total_matches=total_matches,
        )

    print(
        f"[GEOMETRIC] Robust verification succeeded! Identified {inlier_count}/{total_matches} "
        f"inliers ({inlier_count/total_matches*100:.1f}%)."
    )

    return GeometricVerificationResult(
        model=config.model,
        transformation=transformation,
        inlier_mask=inlier_mask,
        residuals=residuals,
        total_matches=total_matches,
    )