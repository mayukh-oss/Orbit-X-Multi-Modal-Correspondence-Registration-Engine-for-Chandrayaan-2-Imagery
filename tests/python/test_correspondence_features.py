# [ANNOTATION] Unit tests for SIFT feature detection and descriptor extraction.
"""
Tests for SIFT feature detection and description.
"""

from __future__ import annotations

# [ANNOTATION] Import OpenCV, NumPy, and PyTest primitives.
import cv2
import numpy as np
import pytest

# [ANNOTATION] Import target function under test.
from sih26166.correspondence.features import detect_features


# [ANNOTATION] Helper function creating a synthetic high-contrast test image with geometric structures.
def _make_test_image() -> np.ndarray:
    """Create a deterministic synthetic image with useful SIFT structure."""

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


# [ANNOTATION] Test verifying SIFT features are detected and descriptors generated.
def test_sift_features_are_detected() -> None:
    print("\n[TEST] Executing test_sift_features_are_detected...")
    image = _make_test_image()

    result = detect_features(image)

    print(f"[TEST] Detected {result.count} SIFT features.")
    assert result.method == "SIFT"
    assert result.image_shape == image.shape
    assert result.count > 0
    assert result.descriptors is not None


# [ANNOTATION] Test verifying SIFT descriptor array dimensions match keypoint count (N x 128).
def test_descriptor_shape_matches_keypoints() -> None:
    print("\n[TEST] Executing test_descriptor_shape_matches_keypoints...")
    image = _make_test_image()

    result = detect_features(image)

    assert result.descriptors is not None
    print(f"[TEST] Descriptors shape: {result.descriptors.shape}")
    assert result.descriptors.shape[0] == result.count
    assert result.descriptors.shape[1] == 128


# [ANNOTATION] Test confirming max_features threshold parameter limits output count.
def test_feature_limit_is_respected() -> None:
    print("\n[TEST] Executing test_feature_limit_is_respected...")
    image = _make_test_image()

    result = detect_features(image, max_features=10)

    print(f"[TEST] Limited feature count: {result.count} (<= 10)")
    assert result.count <= 10
    assert result.descriptors is not None
    assert result.descriptors.shape[0] == result.count


# [ANNOTATION] Test verifying 3-channel color image inputs are handled gracefully.
def test_color_images_are_supported() -> None:
    print("\n[TEST] Executing test_color_images_are_supported...")
    grayscale = _make_test_image()

    color = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    result = detect_features(color)

    assert result.image_shape == grayscale.shape
    assert result.count > 0


# [ANNOTATION] Test verifying unsupported feature detection methods raise ValueError.
def test_unsupported_method_is_rejected() -> None:
    print("\n[TEST] Executing test_unsupported_method_is_rejected...")
    image = _make_test_image()

    with pytest.raises(ValueError, match="Unsupported feature method"):
        detect_features(image, method="ORB")  # type: ignore[arg-type]


# [ANNOTATION] Test verifying invalid max_features values raise errors.
def test_invalid_max_features_are_rejected() -> None:
    print("\n[TEST] Executing test_invalid_max_features_are_rejected...")
    image = _make_test_image()

    with pytest.raises(ValueError, match="greater than zero"):
        detect_features(image, max_features=0)

    with pytest.raises(TypeError, match="must be an integer"):
        detect_features(image, max_features=10.5)  # type: ignore[arg-type]


# [ANNOTATION] Test verifying empty image inputs raise ValueError.
def test_empty_images_are_rejected() -> None:
    print("\n[TEST] Executing test_empty_images_are_rejected...")
    image = np.empty((0, 0), dtype=np.uint8)

    with pytest.raises(ValueError, match="must not be empty"):
        detect_features(image)