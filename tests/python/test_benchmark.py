# [ANNOTATION] Enable modern type hint annotations.
from __future__ import annotations

# [ANNOTATION] Import NumPy for array math and PyTest for unit testing assertions.
import numpy as np
import pytest

# [ANNOTATION] Import evaluation metrics containers.
from sih26166.evaluation.metrics import (
    CorrespondenceMetrics,
    SpatialCoverageMetrics,
)
# [ANNOTATION] Import transformation error internal function under test.
from sih26166.experiments.benchmark import (
    _transformation_error,
)


# [ANNOTATION] Test verifying zero transformation error for identical 2x3 affine matrices.
def test_affine_transformation_error_is_zero_for_identical_matrices() -> None:
    print("\n[TEST] Executing test_affine_transformation_error_is_zero_for_identical_matrices...")
    matrix = np.array(
        [
            [1.0, 0.0, 12.0],
            [0.0, 1.0, -7.0],
        ],
        dtype=np.float64,
    )

    rmse, maximum = _transformation_error(
        matrix,
        matrix,
        "AFFINE",
    )

    print(f"[TEST] Resulting Affine RMSE: {rmse}, Max: {maximum}")
    assert rmse == pytest.approx(0.0)
    assert maximum == pytest.approx(0.0)


# [ANNOTATION] Test calculating non-zero affine matrix element-wise transformation errors.
def test_affine_transformation_error() -> None:
    print("\n[TEST] Executing test_affine_transformation_error...")
    estimated = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
        ],
        dtype=np.float64,
    )

    ground_truth = np.array(
        [
            [1.0, 0.0, 13.0],
            [0.0, 1.0, 16.0],
        ],
        dtype=np.float64,
    )

    rmse, maximum = _transformation_error(
        estimated,
        ground_truth,
        "AFFINE",
    )

    expected_rmse = np.sqrt(
        (3.0**2 + 4.0**2) / 6.0
    )

    print(f"[TEST] Affine Error -> RMSE: {rmse} (Expected: {expected_rmse:.6f}), Max: {maximum}")
    assert rmse == pytest.approx(
        expected_rmse,
    )
    assert maximum == pytest.approx(
        4.0,
    )


# [ANNOTATION] Test confirming homography scale normalization (H[2, 2] = 1) ignores global scaling.
def test_homography_scale_normalization() -> None:
    print("\n[TEST] Executing test_homography_scale_normalization...")
    ground_truth = np.array(
        [
            [2.0, 0.0, 10.0],
            [0.0, 2.0, 20.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    scaled_estimate = ground_truth * 5.0

    rmse, maximum = _transformation_error(
        scaled_estimate,
        ground_truth,
        "HOMOGRAPHY",
    )

    print(f"[TEST] Scaled Homography Normalized RMSE: {rmse}")
    assert rmse == pytest.approx(0.0)
    assert maximum == pytest.approx(0.0)


# [ANNOTATION] Test calculating homography matrix transformation error between non-identical 3x3 matrices.
def test_homography_transformation_error() -> None:
    print("\n[TEST] Executing test_homography_transformation_error...")
    estimated = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ground_truth = np.array(
        [
            [1.0, 0.0, 12.0],
            [0.0, 1.0, 24.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    rmse, maximum = _transformation_error(
        estimated,
        ground_truth,
        "HOMOGRAPHY",
    )

    expected_rmse = np.sqrt(
        (2.0**2 + 4.0**2) / 9.0
    )

    print(f"[TEST] Homography Error -> RMSE: {rmse} (Expected: {expected_rmse:.6f}), Max: {maximum}")
    assert rmse == pytest.approx(
        expected_rmse,
    )
    assert maximum == pytest.approx(
        4.0,
    )


# [ANNOTATION] Parameterized test checking rejection of invalid matrix dimensions for specified model types.
@pytest.mark.parametrize(
    "model,shape",
    [
        ("AFFINE", (3, 3)),
        ("HOMOGRAPHY", (2, 3)),
    ],
)
def test_invalid_ground_truth_shape_is_rejected(
    model: str,
    shape: tuple[int, int],
) -> None:
    print(f"\n[TEST] Executing test_invalid_ground_truth_shape_is_rejected for {model} with shape {shape}...")
    matrix = np.zeros(
        shape,
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="shape"):
        _transformation_error(
            matrix,
            matrix,
            model,  # type: ignore[arg-type]
        )


# [ANNOTATION] Test verifying ValueError raised when homography scale term H[2, 2] == 0.
def test_invalid_homography_normalization_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_homography_normalization_is_rejected...")
    estimated = np.eye(
        3,
        dtype=np.float64,
    )

    ground_truth = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="normalization",
    ):
        _transformation_error(
            estimated,
            ground_truth,
            "HOMOGRAPHY",
        )