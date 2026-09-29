# [ANNOTATION] Unit tests for fundamental correspondence data structures and conversion helpers.
import numpy as np
import pytest

from sih26166.correspondence.types import (
    Correspondence,
    CorrespondenceSet,
    correspondence_arrays,
)


# [ANNOTATION] Test instantiating single Correspondence dataclass.
def test_single_correspondence():
    print("\n[TEST] Executing test_single_correspondence...")
    item = Correspondence(
        reference_xy=(10.5, 20.25),
        source_xy=(15.75, 25.5),
    )

    assert item.reference_xy == (10.5, 20.25)
    assert item.source_xy == (15.75, 25.5)


# [ANNOTATION] Test instantiating CorrespondenceSet with parallel point arrays and scores.
def test_correspondence_set():
    print("\n[TEST] Executing test_correspondence_set...")
    result = CorrespondenceSet(
        reference_points=np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        ),
        source_points=np.array(
            [
                [15.0, 25.0],
                [35.0, 45.0],
            ]
        ),
        scores=np.array([0.9, 0.8]),
    )

    print(f"[TEST] CorrespondenceSet count: {result.count}")
    assert result.count == 2
    assert result.reference_points.shape == (2, 2)
    assert result.source_points.shape == (2, 2)
    assert result.scores is not None
    assert np.allclose(result.scores, [0.9, 0.8])


# [ANNOTATION] Test helper converting list of individual Correspondence items into CorrespondenceSet.
def test_conversion_from_individual_correspondences():
    print("\n[TEST] Executing test_conversion_from_individual_correspondences...")
    items = [
        Correspondence(
            reference_xy=(1.0, 2.0),
            source_xy=(3.0, 4.0),
        ),
        Correspondence(
            reference_xy=(5.0, 6.0),
            source_xy=(7.0, 8.0),
        ),
    ]

    result = correspondence_arrays(items)

    assert result.count == 2
    assert np.allclose(
        result.reference_points,
        [[1.0, 2.0], [5.0, 6.0]],
    )
    assert np.allclose(
        result.source_points,
        [[3.0, 4.0], [7.0, 8.0]],
    )


# [ANNOTATION] Test instantiating empty CorrespondenceSet.
def test_empty_correspondence_set():
    print("\n[TEST] Executing test_empty_correspondence_set...")
    result = CorrespondenceSet.empty()

    assert result.count == 0
    assert result.reference_points.shape == (0, 2)
    assert result.source_points.shape == (0, 2)


# [ANNOTATION] Test verifying error raised when reference and source point counts do not match.
def test_mismatched_point_counts_are_rejected():
    print("\n[TEST] Executing test_mismatched_point_counts_are_rejected...")
    with pytest.raises(ValueError, match="same number of points"):
        CorrespondenceSet(
            reference_points=np.array([[1.0, 2.0]]),
            source_points=np.array(
                [
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ),
        )


# [ANNOTATION] Test verifying error raised when point coordinates contain NaN values.
def test_non_finite_coordinates_are_rejected():
    print("\n[TEST] Executing test_non_finite_coordinates_are_rejected...")
    with pytest.raises(ValueError, match="finite"):
        Correspondence(
            reference_xy=(np.nan, 2.0),
            source_xy=(3.0, 4.0),
        )