# [ANNOTATION] Unit tests for correspondence evaluation and spatial coverage metric utilities.
from __future__ import annotations

import numpy as np
import pytest

from sih26166.correspondence.types import CorrespondenceSet
from sih26166.evaluation.metrics import (
    correspondence_metrics,
    spatial_coverage,
    transformation_residual_metrics,
)


# [ANNOTATION] Helper function creating test CorrespondenceSet instance with known offsets.
def _correspondences() -> CorrespondenceSet:
    reference = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    source = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 9.0],
            [13.0, 14.0],
        ],
        dtype=np.float64,
    )

    return CorrespondenceSet(
        reference_points=reference,
        source_points=source,
    )


# [ANNOTATION] Test calculating Euclidean correspondence error metrics (RMSE, mean, median).
def test_correspondence_metrics() -> None:
    print("\n[TEST] Executing test_correspondence_metrics...")
    correspondences = _correspondences()

    result = correspondence_metrics(
        correspondences,
    )

    expected_errors = np.array(
        [
            0.0,
            0.0,
            1.0,
            5.0,
        ],
        dtype=np.float64,
    )

    expected_rmse = float(
        np.sqrt(
            np.mean(expected_errors**2)
        )
    )

    print(f"[TEST] Correspondence Metrics -> RMSE: {result.rmse}, Mean: {result.mean_error}")
    assert result.count == 4
    assert result.rmse == pytest.approx(
        expected_rmse,
    )
    assert result.mean_error == pytest.approx(
        1.5,
    )
    assert result.median_error == pytest.approx(
        0.5,
    )
    assert result.inlier_count == 4
    assert result.inlier_ratio == pytest.approx(
        1.0,
    )


# [ANNOTATION] Test evaluating correspondence metrics filtering via inlier boolean mask.
def test_correspondence_metrics_with_inlier_mask() -> None:
    print("\n[TEST] Executing test_correspondence_metrics_with_inlier_mask...")
    correspondences = _correspondences()

    mask = np.array(
        [True, True, True, False],
        dtype=bool,
    )

    result = correspondence_metrics(
        correspondences,
        inlier_mask=mask,
    )

    assert result.count == 4
    assert result.inlier_count == 3
    assert result.inlier_ratio == pytest.approx(
        0.75,
    )


# [ANNOTATION] Test evaluating correspondence metrics on an empty CorrespondenceSet.
def test_empty_correspondence_metrics() -> None:
    print("\n[TEST] Executing test_empty_correspondence_metrics...")
    correspondences = CorrespondenceSet.empty()

    result = correspondence_metrics(
        correspondences,
    )

    assert result.count == 0
    assert result.rmse == 0.0
    assert result.mean_error == 0.0
    assert result.median_error == 0.0
    assert result.inlier_count == 0
    assert result.inlier_ratio == 0.0
    assert result.has_correspondences is False


# [ANNOTATION] Test evaluating transformation reprojection residuals against predicted source points.
def test_transformation_residual_metrics() -> None:
    print("\n[TEST] Executing test_transformation_residual_metrics...")
    correspondences = _correspondences()

    predicted_source = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    result = transformation_residual_metrics(
        correspondences,
        predicted_source,
    )

    expected_errors = np.array(
        [
            0.0,
            0.0,
            1.0,
            5.0,
        ],
        dtype=np.float64,
    )

    expected_rmse = float(
        np.sqrt(
            np.mean(expected_errors**2)
        )
    )

    assert result.count == 4
    assert result.rmse == pytest.approx(
        expected_rmse,
    )
    assert result.mean_error == pytest.approx(
        1.5,
    )
    assert result.median_error == pytest.approx(
        0.5,
    )


# [ANNOTATION] Test evaluating transformation residuals with inlier mask filtering.
def test_transformation_residual_with_inlier_mask() -> None:
    print("\n[TEST] Executing test_transformation_residual_with_inlier_mask...")
    correspondences = _correspondences()

    predicted_source = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    mask = np.array(
        [True, True, False, False],
        dtype=bool,
    )

    result = transformation_residual_metrics(
        correspondences,
        predicted_source,
        inlier_mask=mask,
    )

    assert result.inlier_count == 2
    assert result.inlier_ratio == pytest.approx(
        0.5,
    )


# [ANNOTATION] Test calculating spatial grid coverage ratio (4/4 cells occupied = 1.0).
def test_spatial_coverage() -> None:
    print("\n[TEST] Executing test_spatial_coverage...")
    reference = np.array(
        [
            [5.0, 5.0],
            [15.0, 5.0],
            [5.0, 15.0],
            [15.0, 15.0],
        ],
        dtype=np.float64,
    )

    source = reference.copy()

    correspondences = CorrespondenceSet(
        reference_points=reference,
        source_points=source,
    )

    result = spatial_coverage(
        correspondences,
        image_shape=(20, 20),
        grid_rows=2,
        grid_cols=2,
    )

    print(f"[TEST] Spatial coverage ratio: {result.coverage_ratio}")
    assert result.occupied_cells == 4
    assert result.total_cells == 4
    assert result.coverage_ratio == pytest.approx(
        1.0,
    )


# [ANNOTATION] Test calculating spatial grid coverage ratio on partially occupied grid (2/4 cells occupied = 0.5).
def test_spatial_coverage_partial() -> None:
    print("\n[TEST] Executing test_spatial_coverage_partial...")
    reference = np.array(
        [
            [5.0, 5.0],
            [15.0, 5.0],
        ],
        dtype=np.float64,
    )

    source = reference.copy()

    correspondences = CorrespondenceSet(
        reference_points=reference,
        source_points=source,
    )

    result = spatial_coverage(
        correspondences,
        image_shape=(20, 20),
        grid_rows=2,
        grid_cols=2,
    )

    assert result.occupied_cells == 2
    assert result.total_cells == 4
    assert result.coverage_ratio == pytest.approx(
        0.5,
    )


# [ANNOTATION] Test evaluating spatial coverage on empty CorrespondenceSet.
def test_empty_spatial_coverage() -> None:
    print("\n[TEST] Executing test_empty_spatial_coverage...")
    correspondences = CorrespondenceSet.empty()

    result = spatial_coverage(
        correspondences,
        image_shape=(100, 100),
    )

    assert result.occupied_cells == 0
    assert result.total_cells == 64
    assert result.coverage_ratio == 0.0


# [ANNOTATION] Parameterized test checking rejection of mismatched shape arrays for predicted_source.
@pytest.mark.parametrize(
    "predicted_source",
    [
        np.zeros((4, 2)),
        np.zeros((3, 3)),
    ],
)
def test_invalid_predicted_source_shapes_are_rejected(
    predicted_source: np.ndarray,
) -> None:
    print("\n[TEST] Executing test_invalid_predicted_source_shapes_are_rejected...")
    correspondences = CorrespondenceSet(
        reference_points=np.zeros((3, 2)),
        source_points=np.zeros((3, 2)),
    )

    with pytest.raises(ValueError, match="same shape"):
        transformation_residual_metrics(
            correspondences,
            predicted_source,
        )


# [ANNOTATION] Test verifying error raised when inlier mask length does not match correspondence count.
def test_invalid_inlier_mask_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_inlier_mask_is_rejected...")
    correspondences = _correspondences()

    with pytest.raises(
        ValueError,
        match="inlier_mask",
    ):
        correspondence_metrics(
            correspondences,
            inlier_mask=np.array(
                [True, False],
            ),
        )


# [ANNOTATION] Test verifying invalid image_shape dimensions raise ValueError.
def test_invalid_image_shape_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_image_shape_is_rejected...")
    correspondences = _correspondences()

    with pytest.raises(
        ValueError,
        match="image_shape",
    ):
        spatial_coverage(
            correspondences,
            image_shape=(100,), # type: ignore
        )


# [ANNOTATION] Test verifying non-positive grid row count raises ValueError.
def test_invalid_grid_is_rejected() -> None:
    print("\n[TEST] Executing test_invalid_grid_is_rejected...")
    correspondences = _correspondences()

    with pytest.raises(
        ValueError,
        match="grid_rows",
    ):
        spatial_coverage(
            correspondences,
            image_shape=(100, 100),
            grid_rows=0,
        )