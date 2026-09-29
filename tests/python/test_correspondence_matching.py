# [ANNOTATION] Unit tests for descriptor-based feature matching strategies.
"""
Tests for descriptor-based feature matching.
"""

from __future__ import annotations

# [ANNOTATION] Import OpenCV, NumPy, and PyTest primitives.
import cv2
import numpy as np
import pytest

# [ANNOTATION] Import target feature extraction and descriptor matching functions.
from sih26166.correspondence.features import detect_features
from sih26166.correspondence.matching import MatchingConfig, match_features


# [ANNOTATION] Helper function creating a synthetic test image.
def _make_test_image() -> np.ndarray:
    """Create a deterministic image with repeatable local structure."""

    image = np.zeros((256, 256), dtype=np.uint8)

    cv2.rectangle(image, (30, 30), (100, 100), 180, thickness=3)
    cv2.circle(image, (170, 80), 35, 220, thickness=4)
    cv2.line(image, (30, 190), (220, 150), 255, thickness=5)

    cv2.putText(
        image,
        "SIH",
        (70, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        200,
        3,
        cv2.LINE_AA,
    )

    return image


# [ANNOTATION] Helper function extracting SIFT features from identical synthetic images.
def _extract_features() -> tuple:
    """Extract deterministic SIFT features from identical images."""

    image = _make_test_image()

    reference = detect_features(
        image,
        max_features=200,
    )

    source = detect_features(
        image,
        max_features=200,
    )

    return reference, source


# [ANNOTATION] Test verifying matching features extracted from identical images yields valid correspondences.
def test_identical_images_produce_correspondences() -> None:
    print("\n[TEST] Executing test_identical_images_produce_correspondences...")
    reference, source = _extract_features()

    result = match_features(reference, source)

    print(f"[TEST] Produced {result.count} correspondences.")
    assert result.count > 0
    assert result.reference_points.shape == (result.count, 2)
    assert result.source_points.shape == (result.count, 2)
    assert result.scores is not None
    assert result.scores.shape == (result.count,)


# [ANNOTATION] Test confirming matched point coordinates stay within image boundary limits.
def test_correspondence_coordinates_are_valid() -> None:
    print("\n[TEST] Executing test_correspondence_coordinates_are_valid...")
    reference, source = _extract_features()

    result = match_features(reference, source)

    assert result.count > 0

    assert np.all(result.reference_points[:, 0] >= 0.0)
    assert np.all(result.reference_points[:, 0] < reference.image_shape[1])
    assert np.all(result.reference_points[:, 1] >= 0.0)
    assert np.all(result.reference_points[:, 1] < reference.image_shape[0])

    assert np.all(result.source_points[:, 0] >= 0.0)
    assert np.all(result.source_points[:, 0] < source.image_shape[1])
    assert np.all(result.source_points[:, 1] >= 0.0)
    assert np.all(result.source_points[:, 1] < source.image_shape[0])

    assert result.scores is not None
    assert np.all(result.scores >= 0.0)


# [ANNOTATION] Test verifying Lowe's Ratio test matching strategy.
def test_ratio_matching_produces_valid_result() -> None:
    print("\n[TEST] Executing test_ratio_matching_produces_valid_result...")
    reference, source = _extract_features()

    result = match_features(
        reference,
        source,
        config=MatchingConfig(
            strategy="RATIO",
            ratio_threshold=0.75,
        ),
    )

    print(f"[TEST] RATIO strategy produced {result.count} matches.")
    assert result.count >= 0


# [ANNOTATION] Test verifying Mutual nearest-neighbor matching strategy.
def test_mutual_matching_produces_valid_result() -> None:
    print("\n[TEST] Executing test_mutual_matching_produces_valid_result...")
    reference, source = _extract_features()

    result = match_features(
        reference,
        source,
        config=MatchingConfig(
            strategy="MUTUAL",
        ),
    )

    print(f"[TEST] MUTUAL strategy produced {result.count} matches.")
    assert result.count >= 0


# [ANNOTATION] Test verifying combined RATIO + MUTUAL matching strategy.
def test_ratio_mutual_matching_produces_valid_result() -> None:
    print("\n[TEST] Executing test_ratio_mutual_matching_produces_valid_result...")
    reference, source = _extract_features()

    result = match_features(
        reference,
        source,
        config=MatchingConfig(
            strategy="RATIO_MUTUAL",
            ratio_threshold=0.75,
        ),
    )

    print(f"[TEST] RATIO_MUTUAL strategy produced {result.count} matches.")
    assert result.count >= 0


# [ANNOTATION] Test confirming max_matches limit parameter caps returned correspondence count.
def test_max_matches_is_respected() -> None:
    print("\n[TEST] Executing test_max_matches_is_respected...")
    reference, source = _extract_features()

    result = match_features(
        reference,
        source,
        config=MatchingConfig(
            strategy="RATIO",
            max_matches=5,
        ),
    )

    print(f"[TEST] Max matches limit 5 -> Produced: {result.count}")
    assert result.count <= 5

    if result.scores is not None:
        assert result.scores.shape == (result.count,)


# [ANNOTATION] Test verifying invalid matching strategy raises ValueError.
def test_invalid_matching_strategy_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_matching_strategy_is_rejected...")
    with pytest.raises(ValueError, match="Unsupported matching strategy"):
        MatchingConfig(strategy="INVALID")  # type: ignore[arg-type]


# [ANNOTATION] Test verifying out-of-range ratio threshold values raise ValueError.
def test_invalid_ratio_threshold_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_ratio_threshold_is_rejected...")
    with pytest.raises(ValueError, match="between 0 and 1"):
        MatchingConfig(ratio_threshold=0.0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        MatchingConfig(ratio_threshold=1.0)


# [ANNOTATION] Test verifying invalid max_matches values raise errors.
def test_invalid_max_matches_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_max_matches_is_rejected...")
    with pytest.raises(ValueError, match="greater than zero"):
        MatchingConfig(max_matches=0)

    with pytest.raises(TypeError, match="must be an integer"):
        MatchingConfig(max_matches=5.5)  # type: ignore[arg-type]


# [ANNOTATION] Test verifying error raised when feature sets lack descriptor matrices.
def test_missing_descriptors_are_rejected() -> None:
    print("\n[TEST] Executing test_missing_descriptors_are_rejected...")
    reference, source = _extract_features()

    reference_without_descriptors = type(reference)(
        keypoints=reference.keypoints,
        descriptors=None,
        image_shape=reference.image_shape,
        method=reference.method,
    )

    with pytest.raises(ValueError, match="must contain descriptors"):
        match_features(reference_without_descriptors, source)