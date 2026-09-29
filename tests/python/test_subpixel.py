# [ANNOTATION] Unit tests for local quadratic sub-pixel correspondence refinement.
from __future__ import annotations

# [ANNOTATION] Import NumPy and PyTest.
import numpy as np
import pytest

# [ANNOTATION] Import CorrespondenceSet and sub-pixel refinement utilities under test.
from sih26166.correspondence.types import CorrespondenceSet
from sih26166.refinement.subpixel import (
    SubpixelRefinementConfig,
    refine_correspondences,
)


# [ANNOTATION] Helper function creating a synthetic 2D Gaussian peak at sub-pixel offset coordinates.
def _gaussian_peak(
    size: int = 41,
    center_x: float = 20.35,
    center_y: float = 19.70,
    sigma: float = 2.5,
) -> np.ndarray:
    """Create a synthetic image containing a sub-pixel Gaussian peak."""
    y, x = np.mgrid[0:size, 0:size]

    image = np.exp(
        -(
            (x - center_x) ** 2
            + (y - center_y) ** 2
        )
        / (2.0 * sigma**2)
    )

    return image.astype(np.float64)


# [ANNOTATION] Helper function creating a single-point CorrespondenceSet.
def _correspondence_at(
    x: float,
    y: float,
) -> CorrespondenceSet:
    point = np.array([[x, y]], dtype=np.float64)

    return CorrespondenceSet(
        reference_points=point,
        source_points=point.copy(),
    )


# [ANNOTATION] Test verifying sub-pixel refinement accurately estimates peak center (x=20.35, y=19.70).
def test_recovers_subpixel_peak() -> None:
    print("\n[TEST] Executing test_recovers_subpixel_peak...")
    image = _gaussian_peak(
        center_x=20.35,
        center_y=19.70,
    )

    correspondences = _correspondence_at(
        20.0,
        20.0,
    )

    result = refine_correspondences(
        image,
        image,
        correspondences,
    )

    refined = result.correspondences.reference_points[0]

    print(f"[TEST] Refined peak coordinates -> x={refined[0]:.3f}, y={refined[1]:.3f}")
    assert result.refined_mask[0]

    assert refined[0] == pytest.approx(
        20.35,
        abs=0.05,
    )

    assert refined[1] == pytest.approx(
        19.70,
        abs=0.05,
    )


# [ANNOTATION] Test verifying reference and source point coordinates are refined independently.
def test_reference_and_source_are_refined_independently() -> None:
    print("\n[TEST] Executing test_reference_and_source_are_refined_independently...")
    reference = _gaussian_peak(
        center_x=20.30,
        center_y=19.75,
    )

    source = _gaussian_peak(
        center_x=21.20,
        center_y=18.65,
    )

    correspondences = CorrespondenceSet(
        reference_points=np.array(
            [[20.0, 20.0]],
            dtype=np.float64,
        ),
        source_points=np.array(
            [[21.0, 19.0]],
            dtype=np.float64,
        ),
    )

    result = refine_correspondences(
        reference,
        source,
        correspondences,
    )

    assert result.refined_mask[0]

    reference_refined = (
        result.correspondences.reference_points[0]
    )
    source_refined = (
        result.correspondences.source_points[0]
    )

    print(f"[TEST] Refined Ref -> ({reference_refined[0]:.2f}, {reference_refined[1]:.2f})")
    print(f"[TEST] Refined Src -> ({source_refined[0]:.2f}, {source_refined[1]:.2f})")

    assert reference_refined[0] == pytest.approx(
        20.30,
        abs=0.05,
    )
    assert reference_refined[1] == pytest.approx(
        19.75,
        abs=0.05,
    )

    assert source_refined[0] == pytest.approx(
        21.20,
        abs=0.05,
    )
    assert source_refined[1] == pytest.approx(
        18.65,
        abs=0.05,
    )


# [ANNOTATION] Test confirming flat image regions with insufficient curvature are not refined.
def test_flat_region_is_not_falsely_refined() -> None:
    print("\n[TEST] Executing test_flat_region_is_not_falsely_refined...")
    image = np.ones(
        (41, 41),
        dtype=np.float64,
    )

    correspondences = _correspondence_at(
        20.0,
        20.0,
    )

    result = refine_correspondences(
        image,
        image,
        correspondences,
    )

    assert not result.refined_mask[0]

    np.testing.assert_allclose(
        result.correspondences.reference_points,
        correspondences.reference_points,
    )

    np.testing.assert_allclose(
        result.correspondences.source_points,
        correspondences.source_points,
    )


# [ANNOTATION] Test confirming points at image boundaries are retained unrefined without throwing errors.
def test_boundary_point_is_retained() -> None:
    print("\n[TEST] Executing test_boundary_point_is_retained...")
    image = _gaussian_peak(
        center_x=1.0,
        center_y=1.0,
    )

    correspondences = _correspondence_at(
        0.0,
        0.0,
    )

    result = refine_correspondences(
        image,
        image,
        correspondences,
    )

    assert not result.refined_mask[0]

    np.testing.assert_allclose(
        result.correspondences.reference_points,
        correspondences.reference_points,
    )


# [ANNOTATION] Test verifying empty CorrespondenceSet returns empty SubpixelRefinementResult.
def test_empty_correspondences_return_empty_result() -> None:
    print("\n[TEST] Executing test_empty_correspondences_return_empty_result...")
    image = np.ones(
        (20, 20),
        dtype=np.float64,
    )

    result = refine_correspondences(
        image,
        image,
        CorrespondenceSet.empty(),
    )

    assert result.total_count == 0
    assert result.refined_count == 0
    assert result.correspondences.count == 0
    assert result.refined_mask.size == 0
    assert result.offsets.shape == (0, 2)


# [ANNOTATION] Test confirming match scores are preserved across sub-pixel refinement.
def test_scores_are_preserved() -> None:
    print("\n[TEST] Executing test_scores_are_preserved...")
    image = _gaussian_peak()

    correspondences = CorrespondenceSet(
        reference_points=np.array(
            [[20.0, 20.0]],
            dtype=np.float64,
        ),
        source_points=np.array(
            [[20.0, 20.0]],
            dtype=np.float64,
        ),
        scores=np.array(
            [0.123],
            dtype=np.float64,
        ),
    )

    result = refine_correspondences(
        image,
        image,
        correspondences,
    )

    np.testing.assert_allclose(
        result.correspondences.scores, # type: ignore
        np.array([0.123]),
    ) # type: ignore


# [ANNOTATION] Parameterized test checking rejection of invalid configuration settings.
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"method": "INVALID"},
            "Unsupported refinement method",
        ),
        (
            {"window_radius": 0},
            "window_radius",
        ),
        (
            {"max_offset": 0.0},
            "max_offset",
        ),
        (
            {"window_radius": 1, "max_offset": 1.5},
            "max_offset",
        ),
        (
            {"minimum_curvature": 0.0},
            "minimum_curvature",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    print(f"\n[TEST] Executing test_invalid_configuration_is_rejected ({kwargs})...")
    with pytest.raises(ValueError, match=message):
        SubpixelRefinementConfig(**kwargs) # type: ignore


# [ANNOTATION] Parameterized test verifying invalid image inputs raise ValueError.
@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 10)),
        np.empty((10, 0)),
        np.ones((10, 10, 2)),
        np.full((10, 10), np.nan),
        np.full((10, 10), np.inf),
    ],
)
def test_invalid_images_are_rejected(
    image: np.ndarray,
) -> None:
    print("\n[TEST] Executing test_invalid_images_are_rejected...")
    correspondences = _correspondence_at(
        5.0,
        5.0,
    )

    with pytest.raises(ValueError):
        refine_correspondences(
            image,
            image,
            correspondences,
        )