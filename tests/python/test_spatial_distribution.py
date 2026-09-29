# [ANNOTATION] Unit tests for spatially distributed correspondence selection across grid cells.
from __future__ import annotations

# [ANNOTATION] Import NumPy and PyTest.
import numpy as np
import pytest

# [ANNOTATION] Import CorrespondenceSet container and spatial distribution functions under test.
from sih26166.correspondence.types import CorrespondenceSet
from sih26166.spatial.distribution import (
    SpatialSelectionConfig,
    select_spatially_distributed,
)


# [ANNOTATION] Helper function creating CorrespondenceSet with identity reference-to-source mapping.
def _make_correspondences(
    points: np.ndarray,
    scores: np.ndarray | None = None,
) -> CorrespondenceSet:
    """Create a correspondence set with an identity reference-to-source map."""
    points = np.asarray(points, dtype=np.float64)

    return CorrespondenceSet(
        reference_points=points,
        source_points=points.copy(),
        scores=scores,
    )


# [ANNOTATION] Test confirming selected point coordinates match original input points.
def test_selection_preserves_correspondence_coordinates() -> None:
    print("\n[TEST] Executing test_selection_preserves_correspondence_coordinates...")
    points = np.array(
        [
            [10.0, 10.0],
            [90.0, 10.0],
            [10.0, 90.0],
            [90.0, 90.0],
        ]
    )

    correspondences = _make_correspondences(points)

    result = select_spatially_distributed(
        correspondences,
        image_shape=(100, 100),
        config=SpatialSelectionConfig(
            grid_rows=2,
            grid_cols=2,
            max_points=4,
        ),
    )

    assert result.selected_count == 4

    np.testing.assert_allclose(
        result.correspondences.reference_points,
        points,
    )

    np.testing.assert_allclose(
        result.correspondences.source_points,
        points,
    )


# [ANNOTATION] Test confirming round-robin selection selects points from sparsely populated cells before clustering.
def test_round_robin_favors_spatial_coverage() -> None:
    print("\n[TEST] Executing test_round_robin_favors_spatial_coverage...")
    # Six points are concentrated in the upper-left cell. Three additional points occupy the other three cells.
    points = np.array(
        [
            [5.0, 5.0],
            [10.0, 10.0],
            [15.0, 15.0],
            [20.0, 20.0],
            [25.0, 25.0],
            [30.0, 30.0],
            [75.0, 10.0],
            [10.0, 75.0],
            [75.0, 75.0],
        ]
    )

    scores = np.arange(len(points), dtype=np.float64)

    correspondences = _make_correspondences(
        points,
        scores=scores,
    )

    result = select_spatially_distributed(
        correspondences,
        image_shape=(100, 100),
        config=SpatialSelectionConfig(
            grid_rows=2,
            grid_cols=2,
            max_points=4,
        ),
    )

    print(f"[TEST] Selected {result.selected_count} points occupying {result.occupied_cells} cells.")
    assert result.selected_count == 4
    assert result.occupied_cells == 4
    assert result.coverage_ratio == pytest.approx(1.0)

    selected = result.correspondences.reference_points

    assert any(np.allclose(point, [5.0, 5.0]) for point in selected)
    assert any(np.allclose(point, [75.0, 10.0]) for point in selected)
    assert any(np.allclose(point, [10.0, 75.0]) for point in selected)
    assert any(np.allclose(point, [75.0, 75.0]) for point in selected)


# [ANNOTATION] Test verifying points within the same cell are selected in order of best match scores (lowest first).
def test_scores_are_used_within_each_cell() -> None:
    print("\n[TEST] Executing test_scores_are_used_within_each_cell...")
    points = np.array(
        [
            [10.0, 10.0],
            [20.0, 20.0],
            [80.0, 80.0],
        ]
    )

    scores = np.array(
        [
            0.8,
            0.1,
            0.5,
        ]
    )

    correspondences = _make_correspondences(
        points,
        scores=scores,
    )

    result = select_spatially_distributed(
        correspondences,
        image_shape=(100, 100),
        config=SpatialSelectionConfig(
            grid_rows=1,
            grid_cols=1,
            max_points=2,
        ),
    )

    # All points are in the same spatial cell, so the two lowest scores should be selected: 0.1 followed by 0.5.
    np.testing.assert_allclose(
        result.correspondences.reference_points,
        np.array(
            [
                [20.0, 20.0],
                [80.0, 80.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        result.correspondences.scores, # type: ignore
        np.array([0.1, 0.5]),
    ) # type: ignore


# [ANNOTATION] Test confirming max_points selection budget parameter limits output count.
def test_max_points_is_respected() -> None:
    print("\n[TEST] Executing test_max_points_is_respected...")
    points = np.array(
        [
            [float(x), float(y)]
            for y in range(0, 100, 10)
            for x in range(0, 100, 10)
        ]
    )

    correspondences = _make_correspondences(points)

    result = select_spatially_distributed(
        correspondences,
        image_shape=(100, 100),
        config=SpatialSelectionConfig(
            grid_rows=5,
            grid_cols=5,
            max_points=12,
        ),
    )

    assert result.selected_count == 12
    assert len(result.selected_indices) == 12


# [ANNOTATION] Test confirming all points are returned when total count is less than max_points budget.
def test_all_points_are_returned_when_below_limit() -> None:
    print("\n[TEST] Executing test_all_points_are_returned_when_below_limit...")
    points = np.array(
        [
            [10.0, 10.0],
            [50.0, 50.0],
            [90.0, 90.0],
        ]
    )

    correspondences = _make_correspondences(points)

    result = select_spatially_distributed(
        correspondences,
        image_shape=(100, 100),
        config=SpatialSelectionConfig(
            grid_rows=2,
            grid_cols=2,
            max_points=100,
        ),
    )

    assert result.selected_count == 3

    np.testing.assert_array_equal(
        result.selected_indices,
        np.array([0, 1, 2]),
    )


# [ANNOTATION] Test verifying empty CorrespondenceSet input yields empty SpatialSelectionResult.
def test_empty_input_returns_empty_result() -> None:
    print("\n[TEST] Executing test_empty_input_returns_empty_result...")
    result = select_spatially_distributed(
        CorrespondenceSet.empty(),
        image_shape=(100, 100),
    )

    assert result.selected_count == 0
    assert result.occupied_cells == 0
    assert result.coverage_ratio == 0.0
    assert result.selected_indices.size == 0
    assert result.cell_indices.size == 0


# [ANNOTATION] Test confirming points outside image boundaries are clipped to valid cell indices.
def test_points_outside_image_are_clipped_to_valid_cells() -> None:
    print("\n[TEST] Executing test_points_outside_image_are_clipped_to_valid_cells...")
    points = np.array(
        [
            [-10.0, -10.0],
            [110.0, 110.0],
        ]
    )

    correspondences = _make_correspondences(points)

    result = select_spatially_distributed(
        correspondences,
        image_shape=(100, 100),
        config=SpatialSelectionConfig(
            grid_rows=2,
            grid_cols=2,
            max_points=2,
        ),
    )

    assert result.selected_count == 2
    assert np.all(result.cell_indices >= 0)
    assert np.all(result.cell_indices < 4)


# [ANNOTATION] Parameterized test verifying invalid configuration parameters raise ValueError.
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid_rows": 0}, "grid_rows"),
        ({"grid_cols": 0}, "grid_cols"),
        ({"max_points": 0}, "max_points"),
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, int],
    message: str,
) -> None:
    print(f"\n[TEST] Executing test_invalid_configuration_is_rejected ({kwargs})...")
    with pytest.raises(ValueError, match=message):
        SpatialSelectionConfig(**kwargs)


# [ANNOTATION] Parameterized test verifying invalid image_shape dimensions raise ValueError.
@pytest.mark.parametrize(
    "image_shape",
    [
        (0, 100),
        (100, 0),
        (-1, 100),
        (100, -1),
        (100,),
        (100, 100, 3),
    ],
)
def test_invalid_image_shape_is_rejected(
    image_shape: tuple[int, ...],
) -> None:
    print(f"\n[TEST] Executing test_invalid_image_shape_is_rejected ({image_shape})...")
    correspondences = _make_correspondences(
        np.array([[10.0, 10.0]])
    )

    with pytest.raises(ValueError, match="image_shape"):
        select_spatially_distributed(
            correspondences,
            image_shape=image_shape,  # type: ignore[arg-type]
        )