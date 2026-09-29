# [ANNOTATION] Module docstring describing spatially distributed correspondence selection across grid cells.
"""
Spatially distributed correspondence selection.

This module selects geometrically valid correspondences while encouraging
spatial coverage across the reference image.

The selector is deliberately independent of feature detection, descriptor
matching, and geometric verification. It operates only on an already
verified CorrespondenceSet.

Selection is deterministic:
    - correspondences are partitioned into spatial grid cells;
    - candidates inside each cell are ordered by match score;
    - candidates are selected in round-robin passes across occupied cells.

Lower scores are considered better when match scores are supplied.
"""

# [ANNOTATION] Enable modern type hint annotations.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclass decorator.
from dataclasses import dataclass

# [ANNOTATION] Import NumPy for spatial grid computations.
import numpy as np

# [ANNOTATION] Import CorrespondenceSet container from local types module.
from sih26166.correspondence.types import CorrespondenceSet


# [ANNOTATION] Immutable dataclass configuring grid partitioning and point budget for spatial selection.
@dataclass(frozen=True)
class SpatialSelectionConfig:
    """Configuration for spatially distributed correspondence selection."""

    grid_rows: int = 8
    grid_cols: int = 8
    max_points: int = 500

    def __post_init__(self) -> None:
        if self.grid_rows <= 0:
            raise ValueError("grid_rows must be positive.")

        if self.grid_cols <= 0:
            raise ValueError("grid_cols must be positive.")

        if self.max_points <= 0:
            raise ValueError("max_points must be positive.")


# [ANNOTATION] Immutable dataclass containing selected correspondences and grid occupancy statistics.
@dataclass(frozen=True)
class SpatialSelectionResult:
    """Result of spatially distributed correspondence selection."""

    correspondences: CorrespondenceSet
    selected_indices: np.ndarray
    cell_indices: np.ndarray
    occupied_cells: int
    total_cells: int

    # [ANNOTATION] Property returning total selected correspondence count.
    @property
    def selected_count(self) -> int:
        """Return the number of selected correspondences."""
        return self.correspondences.count

    # [ANNOTATION] Property calculating grid cell coverage ratio.
    @property
    def coverage_ratio(self) -> float:
        """Return the fraction of grid cells containing a selected point."""
        if self.total_cells == 0:
            return 0.0

        return float(self.occupied_cells / self.total_cells)


# [ANNOTATION] Helper function validating image dimensions.
def _validate_image_shape(
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    """Validate and normalize an image shape as (height, width)."""
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain (height, width).")

    height, width = image_shape

    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive.")

    return int(height), int(width)


# [ANNOTATION] Helper function mapping 2D point coordinates to 1D grid cell indices.
def _compute_cell_indices(
    points: np.ndarray,
    image_shape: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    """
    Map reference-image coordinates to grid-cell indices.

    Coordinates use image convention:
        x = column
        y = row

    Points outside the image are clipped to the nearest valid cell.
    """
    height, width = image_shape

    x = np.clip(
        points[:, 0],
        0.0,
        np.nextafter(float(width), 0.0),
    )
    y = np.clip(
        points[:, 1],
        0.0,
        np.nextafter(float(height), 0.0),
    )

    col = np.floor(x / width * grid_cols).astype(np.int64)
    row = np.floor(y / height * grid_rows).astype(np.int64)

    col = np.clip(col, 0, grid_cols - 1)
    row = np.clip(row, 0, grid_rows - 1)

    return row * grid_cols + col


# [ANNOTATION] Helper function ordering candidate points within each cell by match score.
def _candidate_order(
    indices: np.ndarray,
    scores: np.ndarray | None,
) -> list[int]:
    """
    Return deterministic candidate ordering.

    When scores are available, lower scores are preferred. Without scores,
    original correspondence order is retained.
    """
    if scores is None:
        return [int(index) for index in indices]

    return sorted(
        (int(index) for index in indices),
        key=lambda index: (
            float(scores[index]),
            index,
        ),
    )


# [ANNOTATION] Primary function executing round-robin spatial selection across grid cells.
def select_spatially_distributed(
    correspondences: CorrespondenceSet,
    image_shape: tuple[int, int],
    config: SpatialSelectionConfig | None = None,
) -> SpatialSelectionResult:
    """
    Select a spatially distributed subset of correspondences.

    Parameters
    ----------
    correspondences:
        Geometrically verified correspondences. This function does not perform
        geometric verification itself.

    image_shape:
        Reference image shape as ``(height, width)``.

    config:
        Spatial selection configuration.

    Returns
    -------
    SpatialSelectionResult
        Selected correspondences and spatial coverage information.

    Notes
    -----
    Selection proceeds in deterministic round-robin passes over occupied
    spatial cells. This prevents a dense cluster of matches from consuming
    the entire output budget before sparsely populated regions are considered.
    """
    if config is None:
        config = SpatialSelectionConfig()

    _validate_image_shape(image_shape)

    print(
        f"[SPATIAL] Selecting spatially distributed points across "
        f"{config.grid_rows}x{config.grid_cols} grid (Target max: {config.max_points})..."
    )

    if correspondences.count == 0:
        print("[SPATIAL] Input correspondence set is empty. Returning empty spatial result.")
        return SpatialSelectionResult(
            correspondences=CorrespondenceSet.empty(),
            selected_indices=np.empty(0, dtype=np.int64),
            cell_indices=np.empty(0, dtype=np.int64),
            occupied_cells=0,
            total_cells=config.grid_rows * config.grid_cols,
        )

    reference_points = correspondences.reference_points

    cell_indices = _compute_cell_indices(
        reference_points,
        image_shape,
        config.grid_rows,
        config.grid_cols,
    )

    cell_candidates: dict[int, list[int]] = {}

    for index, cell in enumerate(cell_indices):
        cell_candidates.setdefault(int(cell), []).append(index)

    ordered_cells = sorted(cell_candidates)

    for cell in ordered_cells:
        cell_candidates[cell] = _candidate_order(
            np.asarray(
                cell_candidates[cell],
                dtype=np.int64,
            ),
            correspondences.scores,
        )

    selected: list[int] = []
    round_index = 0

    # [ANNOTATION] Round-robin selection loop over grid cells.
    while len(selected) < config.max_points:
        added_this_round = False

        for cell in ordered_cells:
            candidates = cell_candidates[cell]

            if round_index >= len(candidates):
                continue

            selected.append(candidates[round_index])
            added_this_round = True

            if len(selected) >= config.max_points:
                break

        if not added_this_round:
            break

        round_index += 1

    selected_indices = np.asarray(
        selected,
        dtype=np.int64,
    )

    selected_correspondences = CorrespondenceSet(
        reference_points=correspondences.reference_points[selected_indices],
        source_points=correspondences.source_points[selected_indices],
        scores=(
            None
            if correspondences.scores is None
            else correspondences.scores[selected_indices]
        ),
    )

    selected_cells = cell_indices[selected_indices]
    occupied_count = int(np.unique(selected_cells).size)
    total_grid_cells = config.grid_rows * config.grid_cols

    print(
        f"[SPATIAL] Selected {len(selected_indices)} points occupying "
        f"{occupied_count}/{total_grid_cells} cells."
    )

    return SpatialSelectionResult(
        correspondences=selected_correspondences,
        selected_indices=selected_indices,
        cell_indices=selected_cells,
        occupied_cells=occupied_count,
        total_cells=total_grid_cells,
    )