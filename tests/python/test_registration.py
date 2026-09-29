# [ANNOTATION] Unit tests for image warping and geometric registration algorithms.
from __future__ import annotations

# [ANNOTATION] Import OpenCV, NumPy, and PyTest modules.
import cv2
import numpy as np
import pytest

# [ANNOTATION] Import registration module functions and configuration dataclass under test.
from sih26166.registration.register import (
    RegistrationConfig,
    register_image,
)


# [ANNOTATION] Helper function creating a synthetic grayscale test image.
def _make_test_image(
    height: int = 64,
    width: int = 80,
) -> np.ndarray:
    """Create a deterministic image with easily identifiable structure."""
    image = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    image[10:20, 15:30] = 1.0
    image[30:45, 40:60] = 0.6
    image[48:55, 8:18] = 0.8

    return image


# [ANNOTATION] Test verifying identity affine transformation preserves image pixel values.
def test_affine_identity_registration() -> None:
    print("\n[TEST] Executing test_affine_identity_registration...")
    image = _make_test_image()

    transformation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = register_image(
        source_image=image,
        reference_shape=image.shape,
        transformation=transformation,
        config=RegistrationConfig(model="AFFINE"),
    )

    assert result.model == "AFFINE"
    assert result.output_shape == image.shape
    assert result.registered_image.shape == image.shape
    assert np.allclose(
        result.registered_image,
        image,
    )


# [ANNOTATION] Test verifying reference-to-source affine translation convention.
def test_affine_translation_uses_reference_to_source_convention() -> None:
    print("\n[TEST] Executing test_affine_translation_uses_reference_to_source_convention...")
    image = np.zeros(
        (40, 50),
        dtype=np.float32,
    )

    # A point at source coordinate (20, 15).
    image[15, 20] = 1.0

    # Reference -> source:
    # source_x = reference_x + 5
    # source_y = reference_y + 3
    # Therefore the source point (20, 15) should appear at reference coordinate (15, 12).
    transformation = np.array(
        [
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )

    result = register_image(
        source_image=image,
        reference_shape=image.shape,
        transformation=transformation,
        config=RegistrationConfig(
            model="AFFINE",
            interpolation=cv2.INTER_NEAREST,
        ),
    )

    peak_y, peak_x = np.unravel_index(
        np.argmax(result.registered_image),
        result.registered_image.shape,
    )

    print(f"[TEST] Translated peak position -> (x={peak_x}, y={peak_y})")
    assert (peak_x, peak_y) == (15, 12)


# [ANNOTATION] Test verifying 3x3 identity homography matrix preserves image.
def test_homography_identity_registration() -> None:
    print("\n[TEST] Executing test_homography_identity_registration...")
    image = _make_test_image()

    transformation = np.eye(
        3,
        dtype=np.float64,
    )

    result = register_image(
        source_image=image,
        reference_shape=image.shape,
        transformation=transformation,
        config=RegistrationConfig(
            model="HOMOGRAPHY",
        ),
    )

    assert result.model == "HOMOGRAPHY"
    assert result.output_shape == image.shape
    assert result.registered_image.shape == image.shape
    assert np.allclose(
        result.registered_image,
        image,
    )


# [ANNOTATION] Test verifying 3-channel color image shape and values are preserved during warping.
def test_color_image_is_preserved() -> None:
    print("\n[TEST] Executing test_color_image_is_preserved...")
    image = np.zeros(
        (32, 40, 3),
        dtype=np.uint8,
    )

    image[10:15, 12:18, 0] = 255
    image[20:25, 25:30, 1] = 180
    image[5:10, 30:35, 2] = 120

    transformation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = register_image(
        source_image=image,
        reference_shape=image.shape[:2],
        transformation=transformation,
        config=RegistrationConfig(
            model="AFFINE",
            interpolation=cv2.INTER_NEAREST,
        ),
    )

    assert result.registered_image.shape == image.shape
    assert result.registered_image.dtype == image.dtype
    assert np.array_equal(
        result.registered_image,
        image,
    )


# [ANNOTATION] Parameterized test checking rejection of invalid matrix dimensions according to model.
@pytest.mark.parametrize(
    "model,shape",
    [
        ("AFFINE", (3, 3)),
        ("HOMOGRAPHY", (2, 3)),
        ("AFFINE", (2, 2)),
        ("HOMOGRAPHY", (4, 4)),
    ],
)
def test_invalid_transformation_shape_is_rejected(
    model: str,
    shape: tuple[int, int],
) -> None:
    print(f"\n[TEST] Executing test_invalid_transformation_shape_is_rejected ({model}, {shape})...")
    image = _make_test_image()

    transformation = np.zeros(
        shape,
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="shape"):
        register_image(
            source_image=image,
            reference_shape=image.shape,
            transformation=transformation,
            config=RegistrationConfig(model=model),  # type: ignore[arg-type]
        )


# [ANNOTATION] Test verifying non-finite values (NaN/Inf) in transformation matrix raise ValueError.
def test_nonfinite_transformation_is_rejected() -> None:
    print("\n[TEST] Executing test_nonfinite_transformation_is_rejected...")
    image = _make_test_image()

    transformation = np.array(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="finite"):
        register_image(
            source_image=image,
            reference_shape=image.shape,
            transformation=transformation,
        )


# [ANNOTATION] Test verifying non-positive reference dimensions raise ValueError.
def test_invalid_reference_shape_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_reference_shape_is_rejected...")
    image = _make_test_image()

    transformation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="reference_shape",
    ):
        register_image(
            source_image=image,
            reference_shape=(0, 50),
            transformation=transformation,
        )


# [ANNOTATION] Test verifying empty source image input raises ValueError.
def test_invalid_image_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_image_is_rejected...")
    transformation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        register_image(
            source_image=np.empty(
                (0, 20),
                dtype=np.float32,
            ),
            reference_shape=(20, 20),
            transformation=transformation,
        )