# [ANNOTATION] Integration tests for the complete end-to-end processing pipeline.
from __future__ import annotations

# [ANNOTATION] Import NumPy for image array generation.
import numpy as np

# [ANNOTATION] Import pipeline configuration and entry function under test.
from sih26166.pipeline import PipelineConfig, run_pipeline
from sih26166.registration.register import RegistrationConfig
from sih26166.verification.geometric import GeometricVerificationConfig


# [ANNOTATION] Helper function generating synthetic feature-rich test image.
def _make_feature_rich_image(
    size: int = 256,
) -> np.ndarray:
    """Create a deterministic image with many stable structures."""
    image = np.zeros(
        (size, size),
        dtype=np.uint8,
    )

    # Large rectangles.
    image[20:70, 20:80] = 220
    image[100:150, 35:95] = 150
    image[175:225, 25:85] = 200

    image[30:85, 150:215] = 180
    image[115:170, 135:205] = 230
    image[190:235, 155:220] = 130

    # Internal geometric structures.
    cv2 = __import__("cv2")

    cv2.circle(
        image,
        (120, 55),
        18,
        255,
        2,
    )

    cv2.circle(
        image,
        (115, 190),
        22,
        245,
        3,
    )

    cv2.line(
        image,
        (10, 240),
        (240, 10),
        190,
        2,
    )

    cv2.line(
        image,
        (20, 120),
        (230, 120),
        100,
        3,
    )

    # Small high-contrast features.
    for x, y in [
        (30, 30),
        (60, 45),
        (45, 125),
        (75, 140),
        (175, 50),
        (195, 70),
        (160, 130),
        (190, 150),
        (45, 205),
        (75, 215),
        (175, 205),
        (205, 220),
    ]:
        cv2.circle(
            image,
            (x, y),
            5,
            255,
            -1,
        )

    return image


# [ANNOTATION] Integration test running pipeline on identical reference/source images.
def test_pipeline_runs_on_identical_images() -> None:
    print("\n[TEST] Executing test_pipeline_runs_on_identical_images...")
    image = _make_feature_rich_image()

    config = PipelineConfig(
        feature_max_features=1000,
        geometric=GeometricVerificationConfig(
            model="AFFINE",
            reprojection_threshold=3.0,
            confidence=0.995,
            max_iterations=5000,
            min_inliers=3,
        ),
        registration=RegistrationConfig(
            model="AFFINE",
        ),
    )

    result = run_pipeline(
        reference_image=image,
        source_image=image.copy(),
        config=config,
    )

    print(f"[TEST] Pipeline completed. Final residual RMSE: {result.residual_metrics.rmse:.4f} px")
    assert result.reference_features.count > 0
    assert result.source_features.count > 0
    assert result.initial_matches.count > 0

    assert result.geometric_verification.success
    assert result.geometric_verification.inlier_count >= 3

    assert result.spatial_selection.selected_count > 0

    assert (
        result.subpixel_refinement.total_count
        == result.spatial_selection.selected_count
    )

    assert result.registration.registered_image.shape == image.shape

    assert result.residual_metrics.count == (
        result.subpixel_refinement.total_count
    )

    assert result.spatial_coverage.total_cells > 0
    assert result.success


# [ANNOTATION] Test confirming registered output image dimensions match reference image dimensions.
def test_pipeline_preserves_reference_dimensions() -> None:
    print("\n[TEST] Executing test_pipeline_preserves_reference_dimensions...")
    reference = _make_feature_rich_image(
        size=256,
    )

    source = reference.copy()

    config = PipelineConfig(
        feature_max_features=1000,
        geometric=GeometricVerificationConfig(
            model="AFFINE",
            min_inliers=3,
        ),
        registration=RegistrationConfig(
            model="AFFINE",
        ),
    )

    result = run_pipeline(
        reference_image=reference,
        source_image=source,
        config=config,
    )

    assert result.registration.output_shape == reference.shape
    assert result.registration.registered_image.shape == reference.shape


# [ANNOTATION] Test verifying pipeline rejects empty reference image input.
def test_pipeline_rejects_empty_reference() -> None:
    print("\n[TEST] Executing test_pipeline_rejects_empty_reference...")
    source = _make_feature_rich_image()

    try:
        run_pipeline(
            reference_image=np.empty(
                (0, 0),
                dtype=np.uint8,
            ),
            source_image=source,
        )
    except ValueError as exc:
        assert "reference_image" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for empty reference image."
        )


# [ANNOTATION] Test verifying pipeline rejects empty source image input.
def test_pipeline_rejects_empty_source() -> None:
    print("\n[TEST] Executing test_pipeline_rejects_empty_source...")
    reference = _make_feature_rich_image()

    try:
        run_pipeline(
            reference_image=reference,
            source_image=np.empty(
                (0, 0),
                dtype=np.uint8,
            ),
        )
    except ValueError as exc:
        assert "source_image" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for empty source image."
        )