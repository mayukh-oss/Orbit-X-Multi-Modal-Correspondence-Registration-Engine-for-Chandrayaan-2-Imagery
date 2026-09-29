# [ANNOTATION] Module docstring describing end-to-end correspondence and registration pipeline execution.
"""
End-to-end correspondence and registration pipeline.

This module connects the independently tested SIH26166 processing stages:

    feature detection
        -> descriptor matching
        -> geometric verification
        -> spatial selection
        -> sub-pixel refinement
        -> image registration
        -> quantitative evaluation

The pipeline does not assume that a particular geometric model is universally
correct. The selected model is explicitly configured by the caller.

The current implementation operates on already-loaded images. Data ingestion,
PDS4 metadata handling, and real Chandrayaan-2 product discovery remain
separate concerns.

SIH 2026 | PS ID: SIH26166
"""

# [ANNOTATION] Enable modern type hint annotations.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclass primitives.
from dataclasses import dataclass

# [ANNOTATION] Import OpenCV for coordinate transformations and NumPy for matrix manipulation.
import cv2
import numpy as np

# [ANNOTATION] Import feature extraction utilities.
from sih26166.correspondence.features import (
    FeatureSet,
    detect_features,
)
# [ANNOTATION] Import descriptor matching algorithms and configurations.
from sih26166.correspondence.matching import (
    MatchingConfig,
    match_features,
)
# [ANNOTATION] Import core correspondence set data structure.
from sih26166.correspondence.types import CorrespondenceSet
# [ANNOTATION] Import metric calculation functions for evaluation.
from sih26166.evaluation.metrics import (
    CorrespondenceMetrics,
    SpatialCoverageMetrics,
    spatial_coverage,
    transformation_residual_metrics,
)
# [ANNOTATION] Import image registration functions and configuration.
from sih26166.registration.register import (
    RegistrationConfig,
    RegistrationResult,
    register_image,
)
# [ANNOTATION] Import local sub-pixel refinement functions.
from sih26166.refinement.subpixel import (
    SubpixelRefinementConfig,
    SubpixelRefinementResult,
    refine_correspondences,
)
# [ANNOTATION] Import spatial grid selection module.
from sih26166.spatial.distribution import (
    SpatialSelectionConfig,
    SpatialSelectionResult,
    select_spatially_distributed,
)
# [ANNOTATION] Import robust RANSAC geometric verification module.
from sih26166.verification.geometric import (
    GeometricVerificationConfig,
    GeometricVerificationResult,
    verify_geometry,
)


# [ANNOTATION] Immutable dataclass encapsulating end-to-end pipeline parameter settings.
@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the end-to-end correspondence pipeline."""

    feature_max_features: int = 2000
    matching: MatchingConfig = MatchingConfig()
    geometric: GeometricVerificationConfig = (
        GeometricVerificationConfig()
    )
    spatial: SpatialSelectionConfig = (
        SpatialSelectionConfig()
    )
    subpixel: SubpixelRefinementConfig = (
        SubpixelRefinementConfig()
    )
    registration: RegistrationConfig = (
        RegistrationConfig()
    )

    # [ANNOTATION] Validation hook for pipeline parameters.
    def __post_init__(self) -> None:
        if self.feature_max_features <= 0:
            raise ValueError(
                "feature_max_features must be positive."
            )


# [ANNOTATION] Immutable dataclass packaging all intermediate outputs and metrics from pipeline run.
@dataclass(frozen=True)
class PipelineResult:
    """Complete result produced by the end-to-end pipeline."""

    reference_features: FeatureSet
    source_features: FeatureSet
    initial_matches: CorrespondenceSet
    geometric_verification: GeometricVerificationResult
    spatial_selection: SpatialSelectionResult
    subpixel_refinement: SubpixelRefinementResult
    registration: RegistrationResult
    residual_metrics: CorrespondenceMetrics
    spatial_coverage: SpatialCoverageMetrics

    # [ANNOTATION] Property indicating overall pipeline registration success.
    @property
    def success(self) -> bool:
        """Return whether registration produced a valid result."""
        return bool(
            self.geometric_verification.success
            and self.registration.registered_image.size > 0
        )


# [ANNOTATION] Helper function applying geometric matrix transformation to reference point arrays.
def _apply_transformation(
    reference_points: np.ndarray,
    transformation: np.ndarray,
    model: str,
) -> np.ndarray:
    """
    Transform reference coordinates into source coordinates.

    The geometric verification stage stores transformations using the project
    convention:

        reference -> source
    """
    print(f"[PIPELINE] Transforming point coordinates using {model} matrix...")
    points = np.asarray(
        reference_points,
        dtype=np.float64,
    )

    if points.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    if model == "AFFINE":
        transformed = cv2.transform(
            points.reshape(-1, 1, 2),
            np.asarray(
                transformation,
                dtype=np.float64,
            ),
        )

        return transformed.reshape(-1, 2)

    if model == "HOMOGRAPHY":
        transformed = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            np.asarray(
                transformation,
                dtype=np.float64,
            ),
        )

        return transformed.reshape(-1, 2)

    raise ValueError(
        f"Unsupported geometric model: {model!r}."
    )


# [ANNOTATION] Main function executing all 9 stages of the correspondence and registration pipeline sequentially.
def run_pipeline(
    reference_image: np.ndarray,
    source_image: np.ndarray,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Run the complete correspondence and registration pipeline.

    Parameters
    ----------
    reference_image:
        Fixed/reference image.

    source_image:
        Moving/source image.

    config:
        Optional pipeline configuration.

    Returns
    -------
    PipelineResult
        Complete intermediate and final processing results.

    Notes
    -----
    The current pipeline uses SIFT as the feature-detection baseline.
    SIFT is treated as a baseline component, not as a claim that it solves
    cross-modal Chandrayaan-2 correspondence by itself.
    """
    if config is None:
        config = PipelineConfig()

    print("\n" + "=" * 60)
    print("[PIPELINE] STARTING END-TO-END CORRESPONDENCE PIPELINE")
    print("=" * 60)

    reference_array = np.asarray(
        reference_image,
    )

    source_array = np.asarray(
        source_image,
    )

    if reference_array.size == 0:
        raise ValueError(
            "reference_image must not be empty."
        )

    if source_array.size == 0:
        raise ValueError(
            "source_image must not be empty."
        )

    # ---------------------------------------------------------------
    # 1. Feature detection
    # ---------------------------------------------------------------
    print("[PIPELINE] Stage 1/9: Extracting local features (SIFT)...")
    reference_features = detect_features(
        reference_array,
        max_features=config.feature_max_features,
    )

    source_features = detect_features(
        source_array,
        max_features=config.feature_max_features,
    )

    # ---------------------------------------------------------------
    # 2. Descriptor matching
    # ---------------------------------------------------------------
    print(f"[PIPELINE] Stage 2/9: Matching descriptors ({config.matching.strategy})...")
    initial_matches = match_features(
        reference_features,
        source_features,
        config=config.matching,
    )

    if initial_matches.count < config.geometric.min_inliers:
        raise ValueError(
            "Insufficient matches for geometric verification: "
            f"{initial_matches.count} available, "
            f"{config.geometric.min_inliers} required."
        )

    # ---------------------------------------------------------------
    # 3. Geometric verification
    # ---------------------------------------------------------------
    print(f"[PIPELINE] Stage 3/9: Robust geometric verification ({config.geometric.model})...")
    geometric_result = verify_geometry(
        initial_matches,
        config=config.geometric,
    )

    if not geometric_result.success:
        raise ValueError(
            "Geometric verification failed."
        )

    # ---------------------------------------------------------------
    # 4. Keep only geometrically verified correspondences
    # ---------------------------------------------------------------
    print("[PIPELINE] Stage 4/9: Filtering matches to retain geometrically verified inliers...")
    verified_mask = geometric_result.inlier_mask

    verified_correspondences = CorrespondenceSet(
        reference_points=(
            initial_matches.reference_points[verified_mask]
        ),
        source_points=(
            initial_matches.source_points[verified_mask]
        ),
        scores=(
            None
            if initial_matches.scores is None
            else initial_matches.scores[verified_mask]
        ),
    )

    # ---------------------------------------------------------------
    # 5. Spatially distribute the verified correspondences
    # ---------------------------------------------------------------
    print("[PIPELINE] Stage 5/9: Selecting spatially distributed correspondences across grid...")
    spatial_result = select_spatially_distributed(
        verified_correspondences,
        image_shape=reference_array.shape[:2],
        config=config.spatial,
    )

    if spatial_result.selected_count == 0:
        raise ValueError(
            "Spatial correspondence selection produced no points."
        )

    # ---------------------------------------------------------------
    # 6. Sub-pixel refinement
    # ---------------------------------------------------------------
    print("[PIPELINE] Stage 6/9: Performing quadratic sub-pixel refinement...")
    subpixel_result = refine_correspondences(
        reference_array,
        source_array,
        spatial_result.correspondences,
        config=config.subpixel,
    )

    # ---------------------------------------------------------------
    # 7. Registration
    #
    # Geometric verification estimates the transformation from the
    # reference frame to the source frame. Registration uses the same
    # transformation convention and maps the source into the reference
    # image dimensions.
    # ---------------------------------------------------------------
    print("[PIPELINE] Stage 7/9: Registering source image into reference frame...")
    registration_result = register_image(
        source_image=source_array,
        reference_shape=reference_array.shape[:2],
        transformation=geometric_result.transformation, # type: ignore
        config=config.registration,
    )

    # ---------------------------------------------------------------
    # 8. Transformation-aware residual evaluation
    #
    # The refined correspondences are compared against the source
    # coordinates predicted by the verified transformation.
    # ---------------------------------------------------------------
    print("[PIPELINE] Stage 8/9: Evaluating transformation-aware residual metrics...")
    refined_correspondences = (
        subpixel_result.correspondences
    )

    predicted_source = _apply_transformation(
        refined_correspondences.reference_points,
        geometric_result.transformation, # type: ignore
        geometric_result.model,
    )

    residual_metrics = transformation_residual_metrics(
        refined_correspondences,
        predicted_source,
    )

    # ---------------------------------------------------------------
    # 9. Spatial coverage evaluation
    # ---------------------------------------------------------------
    print("[PIPELINE] Stage 9/9: Evaluating spatial grid coverage ratio...")
    coverage = spatial_coverage(
        refined_correspondences,
        image_shape=reference_array.shape[:2],
        grid_rows=config.spatial.grid_rows,
        grid_cols=config.spatial.grid_cols,
    )

    print("=" * 60)
    print(f"[PIPELINE] PIPELINE EXECUTION COMPLETE! RMSE: {residual_metrics.rmse:.4f} px")
    print("=" * 60 + "\n")

    return PipelineResult(
        reference_features=reference_features,
        source_features=source_features,
        initial_matches=initial_matches,
        geometric_verification=geometric_result,
        spatial_selection=spatial_result,
        subpixel_refinement=subpixel_result,
        registration=registration_result,
        residual_metrics=residual_metrics,
        spatial_coverage=coverage,
    )