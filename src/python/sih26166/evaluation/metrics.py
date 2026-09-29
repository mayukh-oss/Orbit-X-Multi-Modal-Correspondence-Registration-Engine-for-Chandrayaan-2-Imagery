# [ANNOTATION] Module docstring describing the quantitative evaluation metrics for image correspondence and registration.
"""
Quantitative evaluation metrics for image correspondence and registration.

The metrics in this module are deliberately independent of the feature
detector, matcher, geometric verifier, and registration implementation.

Supported measurements include:

    - correspondence RMSE
    - correspondence mean error
    - correspondence median error
    - inlier count
    - inlier ratio
    - spatial coverage

These metrics are intended for both controlled validation experiments and
real Chandrayaan-2 processing. No metric assumes that a particular geometric
model is universally correct.
"""

# [ANNOTATION] Enable modern type annotations for clean runtime type evaluation.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclass decorator for structured immutability.
from dataclasses import dataclass

# [ANNOTATION] Import NumPy for vector arithmetic and mathematical computations.
import numpy as np

# [ANNOTATION] Import the core CorrespondenceSet type container from local correspondence module.
from sih26166.correspondence.types import CorrespondenceSet


# [ANNOTATION] Immutable dataclass encapsulating quantitative metrics computed over point correspondences.
@dataclass(frozen=True)
class CorrespondenceMetrics:
    """Quantitative metrics for a correspondence set."""

    # [ANNOTATION] Total number of correspondences evaluated.
    count: int
    # [ANNOTATION] Root Mean Square Error across evaluated correspondences.
    rmse: float
    # [ANNOTATION] Arithmetic mean error value.
    mean_error: float
    # [ANNOTATION] Median error value.
    median_error: float
    # [ANNOTATION] Total count of geometrically verified inliers.
    inlier_count: int
    # [ANNOTATION] Inlier ratio calculated as inlier_count / total count.
    inlier_ratio: float

    # [ANNOTATION] Helper property determining whether any correspondences exist in this metric set.
    @property
    def has_correspondences(self) -> bool:
        """Return whether at least one correspondence is present."""
        return self.count > 0


# [ANNOTATION] Immutable dataclass representing spatial distribution coverage statistics across an image grid.
@dataclass(frozen=True)
class SpatialCoverageMetrics:
    """Spatial coverage statistics for reference-image points."""

    # [ANNOTATION] Count of grid cells containing at least one reference point.
    occupied_cells: int
    # [ANNOTATION] Total number of grid cells defined in the spatial layout (rows * cols).
    total_cells: int
    # [ANNOTATION] Spatial coverage ratio computed as occupied_cells / total_cells.
    coverage_ratio: float


# [ANNOTATION] Internal validation helper verifying coordinate structure, dimensions, and finite values.
def _validate_point_arrays(
    reference_points: np.ndarray,
    source_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired point arrays."""

    # [ANNOTATION] Print diagnostic message logging point array validation initialization.
    print("[METRICS] Validating reference and source point coordinate arrays...")

    # [ANNOTATION] Convert reference points to float64 NumPy array.
    reference = np.asarray(
        reference_points,
        dtype=np.float64,
    )

    # [ANNOTATION] Convert source points to float64 NumPy array.
    source = np.asarray(
        source_points,
        dtype=np.float64,
    )

    # [ANNOTATION] Ensure reference points array is 2D with exactly two coordinate columns (x, y).
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError(
            "reference_points must have shape (N, 2)."
        )

    # [ANNOTATION] Ensure source points array is 2D with exactly two coordinate columns (x, y).
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(
            "source_points must have shape (N, 2)."
        )

    # [ANNOTATION] Ensure point count equality across reference and source arrays.
    if reference.shape[0] != source.shape[0]:
        raise ValueError(
            "reference_points and source_points must contain "
            "the same number of points."
        )

    # [ANNOTATION] Verify that reference points contain only finite numerical values.
    if not np.all(np.isfinite(reference)):
        raise ValueError(
            "reference_points must contain only finite values."
        )

    # [ANNOTATION] Verify that source points contain only finite numerical values.
    if not np.all(np.isfinite(source)):
        raise ValueError(
            "source_points must contain only finite values."
        )

    # [ANNOTATION] Return validated coordinate arrays.
    return reference, source


# [ANNOTATION] Internal validation helper ensuring boolean mask shape matches total point count.
def _validate_inlier_mask(
    inlier_mask: np.ndarray,
    count: int,
) -> np.ndarray:
    """Validate an inlier mask."""

    # [ANNOTATION] Print diagnostic message when validating inlier boolean mask.
    print(f"[METRICS] Validating boolean inlier mask against expected length = {count}...")

    # [ANNOTATION] Convert input mask to a boolean NumPy array.
    mask = np.asarray(
        inlier_mask,
        dtype=bool,
    )

    # [ANNOTATION] Validate 1D shape and match with expected point count.
    if mask.ndim != 1 or mask.shape[0] != count:
        raise ValueError(
            "inlier_mask must have shape (N,), matching the "
            "number of correspondences."
        )

    # [ANNOTATION] Return validated boolean mask array.
    return mask


# [ANNOTATION] Compute direct Euclidean point-to-point correspondence metrics.
def correspondence_metrics(
    correspondences: CorrespondenceSet,
    inlier_mask: np.ndarray | None = None,
) -> CorrespondenceMetrics:
    """
    Compute correspondence error and inlier statistics.

    Error is the Euclidean distance between each reference point and its
    corresponding source point:

        e_i = ||reference_i - source_i||_2

    This metric is directly meaningful when the two point sets are expressed
    in the same coordinate frame, such as after applying a known alignment
    transformation in a controlled experiment.

    For a general registration problem, transformation-aware residuals should
    be used instead of interpreting raw coordinate differences as alignment
    error.
    """

    # [ANNOTATION] Print initialization log entry.
    print("[METRICS] Calculating direct coordinate Euclidean correspondence metrics...")

    # [ANNOTATION] Validate input point arrays.
    reference, source = _validate_point_arrays(
        correspondences.reference_points,
        correspondences.source_points,
    )

    # [ANNOTATION] Extract point count.
    count = int(reference.shape[0])

    # [ANNOTATION] Handle empty point sets gracefully.
    if count == 0:
        print("[METRICS] Warning: Correspondence count is zero. Returning zeroed metrics.")
        if inlier_mask is not None:
            _validate_inlier_mask(
                inlier_mask,
                0,
            )

        return CorrespondenceMetrics(
            count=0,
            rmse=0.0,
            mean_error=0.0,
            median_error=0.0,
            inlier_count=0,
            inlier_ratio=0.0,
        )

    # [ANNOTATION] Compute Euclidean distance errors for each point pair: e_i = ||ref_i - src_i||_2
    errors = np.linalg.norm(
        reference - source,
        axis=1,
    )

    # [ANNOTATION] Compute Root Mean Square Error (RMSE).
    rmse = float(
        np.sqrt(
            np.mean(errors**2)
        )
    )

    # [ANNOTATION] Compute arithmetic mean error.
    mean_error = float(
        np.mean(errors)
    )

    # [ANNOTATION] Compute median error.
    median_error = float(
        np.median(errors)
    )

    # [ANNOTATION] Determine inlier counts using optional boolean mask or default to total point count.
    if inlier_mask is None:
        inlier_count = count
    else:
        mask = _validate_inlier_mask(
            inlier_mask,
            count,
        )
        inlier_count = int(
            np.count_nonzero(mask)
        )

    # [ANNOTATION] Compute inlier ratio.
    inlier_ratio = float(
        inlier_count / count
    )

    # [ANNOTATION] Log computed metrics summary.
    print(
        f"[METRICS] Computed metrics for N={count} points:\n"
        f"          RMSE = {rmse:.4f} px, Mean = {mean_error:.4f} px, Median = {median_error:.4f} px\n"
        f"          Inliers = {inlier_count}/{count} ({inlier_ratio*100:.2f}%)"
    )

    # [ANNOTATION] Return populated CorrespondenceMetrics object.
    return CorrespondenceMetrics(
        count=count,
        rmse=rmse,
        mean_error=mean_error,
        median_error=median_error,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
    )


# [ANNOTATION] Compute transformation-aware residual metrics against predicted source point locations.
def transformation_residual_metrics(
    correspondences: CorrespondenceSet,
    predicted_source_points: np.ndarray,
    inlier_mask: np.ndarray | None = None,
) -> CorrespondenceMetrics:
    """
    Compute residual errors against predicted source coordinates.

    This is the transformation-aware metric intended for registration
    evaluation.

    Parameters
    ----------
    correspondences:
        Original reference/source correspondences.

    predicted_source_points:
        Source coordinates predicted by the estimated transformation from
        the reference coordinates.

    inlier_mask:
        Optional externally supplied inlier mask.

    Returns
    -------
    CorrespondenceMetrics
        RMSE and related residual statistics.

    Notes
    -----
    The residual for correspondence i is:

        r_i = ||source_i - predicted_source_i||_2
    """

    # [ANNOTATION] Print entry diagnostic message.
    print("[METRICS] Calculating transformation-aware residual metrics...")

    # [ANNOTATION] Validate reference and source point arrays.
    reference, source = _validate_point_arrays(
        correspondences.reference_points,
        correspondences.source_points,
    )

    # [ANNOTATION] Convert predicted source coordinates to float64 NumPy array.
    predicted = np.asarray(
        predicted_source_points,
        dtype=np.float64,
    )

    # [ANNOTATION] Ensure predicted source matrix dimensions match observed source points.
    if predicted.shape != source.shape:
        raise ValueError(
            "predicted_source_points must have the same shape as "
            "correspondence source_points."
        )

    # [ANNOTATION] Verify finiteness of predicted coordinates.
    if not np.all(np.isfinite(predicted)):
        raise ValueError(
            "predicted_source_points must contain only finite values."
        )

    # [ANNOTATION] Extract correspondence count.
    count = int(reference.shape[0])

    # [ANNOTATION] Handle empty point sets gracefully.
    if count == 0:
        print("[METRICS] Warning: Correspondence count is zero. Returning empty residual metrics.")
        if inlier_mask is not None:
            _validate_inlier_mask(
                inlier_mask,
                0,
            )

        return CorrespondenceMetrics(
            count=0,
            rmse=0.0,
            mean_error=0.0,
            median_error=0.0,
            inlier_count=0,
            inlier_ratio=0.0,
        )

    # [ANNOTATION] Delete reference variable as calculations use source and predicted arrays.
    del reference

    # [ANNOTATION] Compute residual error norm: r_i = ||source_i - predicted_source_i||_2
    errors = np.linalg.norm(
        source - predicted,
        axis=1,
    )

    # [ANNOTATION] Compute Root Mean Square Residual Error.
    rmse = float(
        np.sqrt(
            np.mean(errors**2)
        )
    )

    # [ANNOTATION] Compute mean residual error.
    mean_error = float(
        np.mean(errors)
    )

    # [ANNOTATION] Compute median residual error.
    median_error = float(
        np.median(errors)
    )

    # [ANNOTATION] Evaluate inlier counts using mask or default to count.
    if inlier_mask is None:
        inlier_count = count
    else:
        mask = _validate_inlier_mask(
            inlier_mask,
            count,
        )
        inlier_count = int(
            np.count_nonzero(mask)
        )

    # [ANNOTATION] Compute inlier ratio.
    inlier_ratio = float(
        inlier_count / count
    )

    # [ANNOTATION] Log computed residual metrics summary.
    print(
        f"[METRICS] Residual analysis complete for N={count} points:\n"
        f"          Residual RMSE = {rmse:.4f} px, Mean = {mean_error:.4f} px, Median = {median_error:.4f} px\n"
        f"          Inliers = {inlier_count}/{count} ({inlier_ratio*100:.2f}%)"
    )

    # [ANNOTATION] Return populated CorrespondenceMetrics record.
    return CorrespondenceMetrics(
        count=count,
        rmse=rmse,
        mean_error=mean_error,
        median_error=median_error,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
    )


# [ANNOTATION] Compute spatial coverage ratio across a uniform 2D spatial grid.
def spatial_coverage(
    correspondences: CorrespondenceSet,
    image_shape: tuple[int, int],
    grid_rows: int = 8,
    grid_cols: int = 8,
) -> SpatialCoverageMetrics:
    """
    Compute spatial coverage of reference-image correspondences.

    The reference image is divided into a regular grid. A cell is considered
    occupied when at least one reference point falls inside it.
    """

    # [ANNOTATION] Print entry diagnostic message.
    print(f"[METRICS] Computing spatial coverage across grid ({grid_rows}x{grid_cols})...")

    # [ANNOTATION] Validate image shape dimensions tuple length.
    if len(image_shape) != 2:
        raise ValueError(
            "image_shape must contain (height, width)."
        )

    # [ANNOTATION] Extract height and width dimensions.
    height, width = (
        int(image_shape[0]),
        int(image_shape[1]),
    )

    # [ANNOTATION] Validate positive height and width.
    if height <= 0 or width <= 0:
        raise ValueError(
            "image_shape dimensions must be positive."
        )

    # [ANNOTATION] Validate positive grid rows.
    if grid_rows <= 0:
        raise ValueError(
            "grid_rows must be positive."
        )

    # [ANNOTATION] Validate positive grid columns.
    if grid_cols <= 0:
        raise ValueError(
            "grid_cols must be positive."
        )

    # [ANNOTATION] Convert reference points to float64 array.
    points = np.asarray(
        correspondences.reference_points,
        dtype=np.float64,
    )

    # [ANNOTATION] Validate points array shape.
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "reference points must have shape (N, 2)."
        )

    # [ANNOTATION] Calculate total cells in grid.
    total_cells = int(
        grid_rows * grid_cols
    )

    # [ANNOTATION] Handle empty reference point sets.
    if points.shape[0] == 0:
        print("[METRICS] Reference points array is empty. Spatial coverage is 0.0.")
        return SpatialCoverageMetrics(
            occupied_cells=0,
            total_cells=total_cells,
            coverage_ratio=0.0,
        )

    # [ANNOTATION] Verify that reference coordinates contain only finite values.
    if not np.all(np.isfinite(points)):
        raise ValueError(
            "reference points must contain only finite values."
        )

    # [ANNOTATION] Clip x coordinates within image boundary [0, width).
    x = np.clip(
        points[:, 0],
        0.0,
        np.nextafter(float(width), 0.0),
    )

    # [ANNOTATION] Clip y coordinates within image boundary [0, height).
    y = np.clip(
        points[:, 1],
        0.0,
        np.nextafter(float(height), 0.0),
    )

    # [ANNOTATION] Map x coordinate to column index.
    cols = np.floor(
        x / width * grid_cols
    ).astype(np.int64)

    # [ANNOTATION] Map y coordinate to row index.
    rows = np.floor(
        y / height * grid_rows
    ).astype(np.int64)

    # [ANNOTATION] Enforce column bounds [0, grid_cols - 1].
    cols = np.clip(
        cols,
        0,
        grid_cols - 1,
    )

    # [ANNOTATION] Enforce row bounds [0, grid_rows - 1].
    rows = np.clip(
        rows,
        0,
        grid_rows - 1,
    )

    # [ANNOTATION] Compute 1D cell indices from row and column locations.
    cells = rows * grid_cols + cols

    # [ANNOTATION] Count unique occupied cells.
    occupied_cells = int(
        np.unique(cells).size
    )

    # [ANNOTATION] Compute coverage ratio.
    coverage_ratio = float(
        occupied_cells / total_cells
    )

    # [ANNOTATION] Log spatial distribution summary.
    print(
        f"[METRICS] Spatial grid evaluation complete: Occupied {occupied_cells}/{total_cells} cells "
        f"({coverage_ratio*100:.1f}% coverage)."
    )

    # [ANNOTATION] Return SpatialCoverageMetrics record.
    return SpatialCoverageMetrics(
        occupied_cells=occupied_cells,
        total_cells=total_cells,
        coverage_ratio=coverage_ratio,
    )