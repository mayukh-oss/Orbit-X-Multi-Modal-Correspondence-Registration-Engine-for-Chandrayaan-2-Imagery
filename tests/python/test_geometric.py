# [ANNOTATION] Unit tests for RANSAC robust geometric verification.
"""
Tests for robust geometric verification.
"""

from __future__ import annotations

# [ANNOTATION] Import OpenCV, NumPy, and PyTest primitives.
import cv2
import numpy as np
import pytest

# [ANNOTATION] Import CorrespondenceSet and geometric verification utilities under test.
from sih26166.correspondence.types import CorrespondenceSet
from sih26166.verification.geometric import (
    GeometricVerificationConfig,
    verify_geometry,
)


# [ANNOTATION] Helper function generating synthetic correspondences with known affine matrix and outlier noise.
def _make_affine_correspondences(
    *,
    seed: int = 26166,
    inlier_count: int = 80,
    outlier_count: int = 20,
) -> tuple[CorrespondenceSet, np.ndarray]:
    """Create correspondences from a known affine transformation."""

    rng = np.random.default_rng(seed)

    reference_points = rng.uniform(
        low=[20.0, 20.0],
        high=[236.0, 236.0],
        size=(inlier_count, 2),
    )

    true_transform = np.array(
        [
            [1.08, -0.12, 18.0],
            [0.09, 1.04, -11.0],
        ],
        dtype=np.float64,
    )

    homogeneous = np.column_stack(
        (
            reference_points,
            np.ones(inlier_count, dtype=np.float64),
        )
    )

    source_inliers = homogeneous @ true_transform.T

    source_inliers += rng.normal(
        loc=0.0,
        scale=0.15,
        size=source_inliers.shape,
    )

    reference_outliers = rng.uniform(
        low=[0.0, 0.0],
        high=[256.0, 256.0],
        size=(outlier_count, 2),
    )

    source_outliers = rng.uniform(
        low=[0.0, 0.0],
        high=[256.0, 256.0],
        size=(outlier_count, 2),
    )

    reference_points = np.vstack(
        (
            reference_points,
            reference_outliers,
        )
    )

    source_points = np.vstack(
        (
            source_inliers,
            source_outliers,
        )
    )

    correspondences = CorrespondenceSet(
        reference_points=reference_points,
        source_points=source_points,
    )

    return correspondences, true_transform


# [ANNOTATION] Helper function generating synthetic correspondences with known homography matrix.
def _make_homography_correspondences(
    *,
    seed: int = 26166,
    count: int = 60,
) -> tuple[CorrespondenceSet, np.ndarray]:
    """Create correspondences from a known projective transformation."""

    rng = np.random.default_rng(seed)

    reference_points = rng.uniform(
        low=[20.0, 20.0],
        high=[236.0, 236.0],
        size=(count, 2),
    )

    true_transform = np.array(
        [
            [1.04, 0.015, 12.0],
            [-0.01, 1.02, 8.0],
            [0.00008, 0.00012, 1.0],
        ],
        dtype=np.float64,
    )

    source_points = cv2.perspectiveTransform(
        reference_points.reshape(-1, 1, 2),
        true_transform,
    ).reshape(-1, 2)

    source_points += rng.normal(
        loc=0.0,
        scale=0.1,
        size=source_points.shape,
    )

    correspondences = CorrespondenceSet(
        reference_points=reference_points,
        source_points=source_points,
    )

    return correspondences, true_transform


# [ANNOTATION] Test verifying RANSAC recovers ground-truth 2x3 affine transformation matrix.
def test_affine_ransac_recovers_known_geometry() -> None:
    print("\n[TEST] Executing test_affine_ransac_recovers_known_geometry...")
    correspondences, true_transform = _make_affine_correspondences()

    result = verify_geometry(
        correspondences,
        config=GeometricVerificationConfig(
            model="AFFINE",
            reprojection_threshold=1.5,
            confidence=0.999,
            max_iterations=5000,
            min_inliers=50,
        ),
    )

    assert result.transformation is not None
    assert result.transformation.shape == (2, 3)

    print(f"[TEST] Affine RANSAC identified {result.inlier_count} inliers.")
    assert result.inlier_count >= 70
    assert result.inlier_ratio >= 0.70

    assert result.residuals.shape == (correspondences.count,)

    assert np.isfinite(
        result.residuals[result.inlier_mask]
    ).all()

    assert np.allclose(
        result.transformation,
        true_transform,
        atol=0.5,
    )


# [ANNOTATION] Test verifying RANSAC correctly rejects random outlier matches.
def test_affine_outliers_are_rejected() -> None:
    print("\n[TEST] Executing test_affine_outliers_are_rejected...")
    correspondences, _ = _make_affine_correspondences(
        inlier_count=80,
        outlier_count=40,
    )

    result = verify_geometry(
        correspondences,
        config=GeometricVerificationConfig(
            model="AFFINE",
            reprojection_threshold=1.5,
            min_inliers=50,
        ),
    )

    assert result.transformation is not None
    assert result.inlier_count >= 70
    assert result.inlier_count < correspondences.count


# [ANNOTATION] Test verifying RANSAC recovers ground-truth 3x3 homography matrix.
def test_homography_ransac_recovers_known_geometry() -> None:
    print("\n[TEST] Executing test_homography_ransac_recovers_known_geometry...")
    correspondences, true_transform = _make_homography_correspondences()

    result = verify_geometry(
        correspondences,
        config=GeometricVerificationConfig(
            model="HOMOGRAPHY",
            reprojection_threshold=1.5,
            confidence=0.999,
            max_iterations=5000,
            min_inliers=40,
        ),
    )

    assert result.transformation is not None
    assert result.transformation.shape == (3, 3)

    print(f"[TEST] Homography RANSAC identified {result.inlier_count} inliers.")
    assert result.inlier_count >= 50
    assert result.inlier_ratio >= 0.80

    estimated = result.transformation / result.transformation[2, 2]
    expected = true_transform / true_transform[2, 2]

    assert np.allclose(
        estimated,
        expected,
        atol=0.25,
    )


# [ANNOTATION] Test verifying verification fails gracefully when input points are fewer than minimum required.
def test_insufficient_correspondences_return_failure() -> None:
    print("\n[TEST] Executing test_insufficient_correspondences_return_failure...")
    correspondences = CorrespondenceSet(
        reference_points=np.array(
            [
                [10.0, 10.0],
                [20.0, 20.0],
            ]
        ),
        source_points=np.array(
            [
                [15.0, 12.0],
                [25.0, 22.0],
            ]
        ),
    )

    result = verify_geometry(
        correspondences,
        config=GeometricVerificationConfig(
            model="AFFINE",
        ),
    )

    assert result.transformation is None
    assert result.inlier_count == 0
    assert result.inlier_ratio == 0.0
    assert not result.success


# [ANNOTATION] Test verifying verification fails gracefully on empty input set.
def test_empty_correspondences_return_failure() -> None:
    print("\n[TEST] Executing test_empty_correspondences_return_failure...")
    result = verify_geometry(
        CorrespondenceSet.empty(),
        config=GeometricVerificationConfig(
            model="AFFINE",
        ),
    )

    assert result.transformation is None
    assert result.inlier_count == 0
    assert result.inlier_ratio == 0.0
    assert result.residuals.size == 0
    assert not result.success


# [ANNOTATION] Test verifying invalid geometric model name raises ValueError.
def test_invalid_model_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_model_is_rejected...")
    with pytest.raises(ValueError, match="Unsupported geometric model"):
        GeometricVerificationConfig(
            model="PROJECTIVE"  # type: ignore[arg-type]
        )


# [ANNOTATION] Test verifying non-positive reprojection threshold raises ValueError.
def test_invalid_reprojection_threshold_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_reprojection_threshold_is_rejected...")
    with pytest.raises(ValueError, match="greater than zero"):
        GeometricVerificationConfig(
            reprojection_threshold=0.0
        )


# [ANNOTATION] Test verifying out-of-range confidence parameter raises ValueError.
def test_invalid_confidence_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_confidence_is_rejected...")
    with pytest.raises(ValueError, match="between 0 and 1"):
        GeometricVerificationConfig(
            confidence=1.0
        )


# [ANNOTATION] Test verifying invalid RANSAC max_iterations count raises errors.
def test_invalid_iteration_count_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_iteration_count_is_rejected...")
    with pytest.raises(ValueError, match="greater than zero"):
        GeometricVerificationConfig(
            max_iterations=0
        )

    with pytest.raises(TypeError, match="must be an integer"):
        GeometricVerificationConfig(
            max_iterations=100.5  # type: ignore[arg-type]
        )


# [ANNOTATION] Test verifying min_inliers thresholds below model degrees of freedom are rejected.
def test_invalid_minimum_inliers_are_rejected() -> None:
    print("\n[TEST] Executing test_invalid_minimum_inliers_are_rejected...")
    with pytest.raises(ValueError, match="at least 3"):
        GeometricVerificationConfig(
            model="AFFINE",
            min_inliers=2,
        )

    with pytest.raises(ValueError, match="at least 4"):
        GeometricVerificationConfig(
            model="HOMOGRAPHY",
            min_inliers=3,
        )


# [ANNOTATION] Test confirming reprojection residuals are zero for exact noiseless geometric matches.
def test_residuals_are_zero_for_exact_geometry() -> None:
    print("\n[TEST] Executing test_residuals_are_zero_for_exact_geometry...")
    reference_points = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
            [50.0, 50.0],
            [25.0, 75.0],
        ],
        dtype=np.float64,
    )

    true_transform = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, -5.0],
        ],
        dtype=np.float64,
    )

    homogeneous = np.column_stack(
        (
            reference_points,
            np.ones(reference_points.shape[0]),
        )
    )

    source_points = homogeneous @ true_transform.T

    correspondences = CorrespondenceSet(
        reference_points=reference_points,
        source_points=source_points,
    )

    result = verify_geometry(
        correspondences,
        config=GeometricVerificationConfig(
            model="AFFINE",
            reprojection_threshold=0.01,
            min_inliers=6,
        ),
    )

    assert result.transformation is not None
    assert result.inlier_count == correspondences.count
    assert np.max(result.residuals) < 1e-8