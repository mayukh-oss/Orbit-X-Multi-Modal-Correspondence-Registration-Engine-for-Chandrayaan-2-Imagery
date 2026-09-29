# [ANNOTATION] Module docstring describing controlled validation benchmark utilities.
"""
Controlled-validation benchmark utilities.

This module evaluates the end-to-end SIH26166 pipeline against a known
ground-truth transformation.

IMPORTANT:
    Results produced by this module belong to CONTROLLED_VALIDATION mode.
    They must not be presented as measured performance on real Chandrayaan-2
    imagery.

The benchmark reports:
    - transformation error
    - correspondence residual RMSE
    - inlier count
    - inlier ratio
    - spatial coverage
    - sub-pixel refinement count
"""

# [ANNOTATION] Enable modern type hinting behavior across module annotations.
from __future__ import annotations

# [ANNOTATION] Import standard dataclass primitives and typing literals.
from dataclasses import dataclass
from typing import Literal

# [ANNOTATION] Import NumPy for mathematical operations and array manipulations.
import numpy as np

# [ANNOTATION] Import pipeline utilities for running end-to-end registration execution.
from sih26166.pipeline import PipelineConfig, PipelineResult, run_pipeline

# [ANNOTATION] Define supported geometric benchmark model literals.
BenchmarkModel = Literal["AFFINE", "HOMOGRAPHY"]


# [ANNOTATION] Immutable dataclass encapsulating quantitative metrics from one benchmark run.
@dataclass(frozen=True)
class BenchmarkMetrics:
    """Quantitative results from one controlled benchmark run."""

    mode: str
    model: BenchmarkModel

    transformation_rmse: float
    transformation_max_error: float

    residual_rmse: float
    residual_mean_error: float
    residual_median_error: float

    inlier_count: int
    inlier_ratio: float

    spatial_coverage_ratio: float
    occupied_cells: int
    total_cells: int

    matched_count: int
    selected_count: int
    refined_count: int

    registration_height: int
    registration_width: int


# [ANNOTATION] Immutable dataclass packaging the final benchmark result metrics and pipeline outputs.
@dataclass(frozen=True)
class BenchmarkResult:
    """Complete controlled-validation benchmark result."""

    metrics: BenchmarkMetrics
    pipeline: PipelineResult


# [ANNOTATION] Internal helper function normalizing a 3x3 homography matrix so H[2, 2] == 1.
def _normalize_homography(
    matrix: np.ndarray,
) -> np.ndarray:
    """Normalize a homography so that H[2, 2] equals one."""
    print("[BENCHMARK] Normalizing homography matrix scaling factor H[2, 2] -> 1.0...")
    normalized = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if normalized.shape != (3, 3):
        raise ValueError(
            "HOMOGRAPHY matrix must have shape (3, 3)."
        )

    denominator = float(normalized[2, 2])

    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError(
            "HOMOGRAPHY matrix has an invalid normalization term."
        )

    return normalized / denominator


# [ANNOTATION] Internal helper function validating ground-truth matrix shape according to model.
def _validate_ground_truth(
    matrix: np.ndarray,
    model: BenchmarkModel,
) -> np.ndarray:
    """Validate a ground-truth transformation matrix."""
    print(f"[BENCHMARK] Validating ground truth matrix for model type: {model}...")
    ground_truth = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(ground_truth)):
        raise ValueError(
            "ground_truth must contain only finite values."
        )

    if model == "AFFINE" and ground_truth.shape == (3, 3):
        if np.allclose(ground_truth[2, :], [0.0, 0.0, 1.0]):
            ground_truth = ground_truth[:2, :]

    if model == "HOMOGRAPHY" and ground_truth.shape == (2, 3):
        ground_truth = np.vstack([ground_truth, [0.0, 0.0, 1.0]])

    expected_shape = (
        (2, 3)
        if model == "AFFINE"
        else (3, 3)
    )

    if ground_truth.shape != expected_shape:
        raise ValueError(
            f"{model} ground-truth transformation must have "
            f"shape {expected_shape}."
        )

    if model == "HOMOGRAPHY":
        ground_truth = _normalize_homography(
            ground_truth,
        )

    return ground_truth


# [ANNOTATION] Helper function computing element-wise matrix transformation error metrics.
def _transformation_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    model: BenchmarkModel,
) -> tuple[float, float]:
    """
    Compute element-wise transformation error.

    For homographies, both matrices are normalized before comparison because
    a homography is defined only up to a non-zero scale factor.
    """
    print(f"[BENCHMARK] Calculating element-wise transformation matrix errors ({model})...")
    estimated_array = np.asarray(
        estimated,
        dtype=np.float64,
    )

    ground_truth_array = _validate_ground_truth(
        ground_truth,
        model,
    )

    if model == "HOMOGRAPHY":
        estimated_array = _normalize_homography(
            estimated_array,
        )

    if estimated_array.shape != ground_truth_array.shape:
        raise ValueError(
            "estimated and ground_truth transformations must have "
            "the same shape."
        )

    difference = np.abs(
        estimated_array - ground_truth_array
    )

    rmse = float(
        np.sqrt(
            np.mean(difference**2)
        )
    )

    maximum = float(
        np.max(difference)
    )

    print(f"[BENCHMARK] Transformation Matrix Error -> RMSE: {rmse:.6f}, Max: {maximum:.6f}")
    return rmse, maximum


# [ANNOTATION] Primary entry function running controlled synthetic benchmark comparison against ground truth.
def run_controlled_benchmark(
    reference_image: np.ndarray,
    source_image: np.ndarray,
    ground_truth_transformation: np.ndarray,
    config: PipelineConfig | None = None,
    model: BenchmarkModel = "AFFINE",
) -> BenchmarkResult:
    """
    Run the pipeline and compare its estimated transformation with ground truth.

    Parameters
    ----------
    reference_image:
        Fixed/reference image.

    source_image:
        Controlled-validation moving/source image.

    ground_truth_transformation:
        Known reference -> source transformation used to create the controlled
        source image.

    config:
        Optional end-to-end pipeline configuration.

    model:
        Ground-truth transformation model.

    Returns
    -------
    BenchmarkResult
        Pipeline output together with controlled-validation metrics.
    """
    print(f"\n[BENCHMARK] Starting controlled validation run using model: {model}")

    if model not in {"AFFINE", "HOMOGRAPHY"}:
        raise ValueError(
            f"Unsupported benchmark model: {model!r}."
        )

    ground_truth = _validate_ground_truth(
        ground_truth_transformation,
        model,
    )

    if config is None:
        config = PipelineConfig()

    if config.geometric.model != model:
        raise ValueError(
            "Pipeline geometric model and benchmark model must match."
        )

    print("[BENCHMARK] Executing end-to-end pipeline execution loop...")
    pipeline_result = run_pipeline(
        reference_image=reference_image,
        source_image=source_image,
        config=config,
    )

    if not pipeline_result.geometric_verification.success:
        raise ValueError(
            "Controlled benchmark requires successful geometric verification."
        )

    estimated = pipeline_result.geometric_verification.transformation

    transformation_rmse, transformation_max_error = (
        _transformation_error(
            estimated, # type: ignore
            ground_truth,
            model,
        )
    )

    residual = pipeline_result.residual_metrics
    coverage = pipeline_result.spatial_coverage

    metrics = BenchmarkMetrics(
        mode="CONTROLLED_VALIDATION",
        model=model,
        transformation_rmse=transformation_rmse,
        transformation_max_error=transformation_max_error,
        residual_rmse=residual.rmse,
        residual_mean_error=residual.mean_error,
        residual_median_error=residual.median_error,
        inlier_count=(
            pipeline_result.geometric_verification.inlier_count
        ),
        inlier_ratio=(
            pipeline_result.geometric_verification.inlier_ratio
        ),
        spatial_coverage_ratio=coverage.coverage_ratio,
        occupied_cells=coverage.occupied_cells,
        total_cells=coverage.total_cells,
        matched_count=pipeline_result.initial_matches.count,
        selected_count=(
            pipeline_result.spatial_selection.selected_count
        ),
        refined_count=(
            pipeline_result.subpixel_refinement.refined_count
        ),
        registration_height=(
            pipeline_result.registration.height
        ),
        registration_width=(
            pipeline_result.registration.width
        ),
    )

    print("[BENCHMARK] Controlled validation evaluation completed successfully.\n")

    return BenchmarkResult(
        metrics=metrics,
        pipeline=pipeline_result,
    )