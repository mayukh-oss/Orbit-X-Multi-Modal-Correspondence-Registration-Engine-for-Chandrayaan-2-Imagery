# [ANNOTATION] Module docstring defining core image correspondence data structures.
"""
Core data structures for image correspondence.

This module contains only typed containers and validation helpers.
It intentionally does not implement a matching algorithm.
"""

# [ANNOTATION] Enable modern type annotations for clean type checking.
from __future__ import annotations

# [ANNOTATION] Import standard dataclass primitives and sequence typing wrappers.
from dataclasses import dataclass
from typing import Sequence

# [ANNOTATION] Import NumPy for vector and matrix coordinate operations.
import numpy as np


# [ANNOTATION] Immutable dataclass encapsulating a single point correspondence record.
@dataclass(frozen=True)
class Correspondence:
    """A single correspondence between reference and source images.

    Coordinates follow image convention:
        x = column
        y = row

    reference_xy and source_xy therefore have the form:
        [x, y]
    """

    # [ANNOTATION] Reference image point coordinates tuple (x, y).
    reference_xy: tuple[float, float]
    # [ANNOTATION] Source image point coordinates tuple (x, y).
    source_xy: tuple[float, float]

    # [ANNOTATION] Post-initialization validation ensuring coordinate lengths and finite values.
    def __post_init__(self) -> None:
        if len(self.reference_xy) != 2:
            raise ValueError("reference_xy must contain exactly two coordinates.")

        if len(self.source_xy) != 2:
            raise ValueError("source_xy must contain exactly two coordinates.")

        values = (*self.reference_xy, *self.source_xy)

        if not all(np.isfinite(value) for value in values):
            raise ValueError("Correspondence coordinates must be finite.")


# [ANNOTATION] Immutable dataclass managing aligned sets of point correspondences between images.
@dataclass(frozen=True)
class CorrespondenceSet:
    """A collection of correspondences between two images."""

    # [ANNOTATION] Reference point coordinate array (N, 2).
    reference_points: np.ndarray
    # [ANNOTATION] Source point coordinate array (N, 2).
    source_points: np.ndarray
    # [ANNOTATION] Optional correspondence match distance or confidence scores array (N,).
    scores: np.ndarray | None = None

    # [ANNOTATION] Validation hook executing after instance creation to enforce matrix shape and finiteness constraints.
    def __post_init__(self) -> None:
        reference_points = np.asarray(self.reference_points, dtype=np.float64)
        source_points = np.asarray(self.source_points, dtype=np.float64)

        # [ANNOTATION] Verify matrix dimensionality and shape bounds for reference coordinates.
        if reference_points.ndim != 2 or reference_points.shape[1] != 2:
            raise ValueError(
                "reference_points must have shape (N, 2)."
            )

        # [ANNOTATION] Verify matrix dimensionality and shape bounds for source coordinates.
        if source_points.ndim != 2 or source_points.shape[1] != 2:
            raise ValueError(
                "source_points must have shape (N, 2)."
            )

        # [ANNOTATION] Ensure both point sets contain equal row counts.
        if reference_points.shape[0] != source_points.shape[0]:
            raise ValueError(
                "reference_points and source_points must contain the same "
                "number of points."
            )

        # [ANNOTATION] Validate finiteness of coordinates.
        if not np.all(np.isfinite(reference_points)):
            raise ValueError("reference_points must contain only finite values.")

        if not np.all(np.isfinite(source_points)):
            raise ValueError("source_points must contain only finite values.")

        # [ANNOTATION] Validate scores vector shape and values if provided.
        if self.scores is not None:
            scores = np.asarray(self.scores, dtype=np.float64)

            if scores.ndim != 1:
                raise ValueError("scores must have shape (N,).")

            if scores.shape[0] != reference_points.shape[0]:
                raise ValueError(
                    "scores must contain one value for every correspondence."
                )

            if not np.all(np.isfinite(scores)):
                raise ValueError("scores must contain only finite values.")

            object.__setattr__(self, "scores", scores)

        # [ANNOTATION] Safely assign validated float64 NumPy matrices to frozen dataclass fields.
        object.__setattr__(self, "reference_points", reference_points)
        object.__setattr__(self, "source_points", source_points)

    # [ANNOTATION] Return total count of point correspondences contained in this set.
    @property
    def count(self) -> int:
        """Return the number of correspondences."""
        return int(self.reference_points.shape[0])

    # [ANNOTATION] Convert structured array matrices into a tuple of individual immutable Correspondence objects.
    def as_pairs(self) -> tuple[Correspondence, ...]:
        """Return the correspondences as individual immutable records."""
        return tuple(
            Correspondence(
                reference_xy=(float(reference[0]), float(reference[1])),
                source_xy=(float(source[0]), float(source[1])),
            )
            for reference, source in zip(
                self.reference_points,
                self.source_points,
            )
        )

    # [ANNOTATION] Factory method generating an empty correspondence set.
    @classmethod
    def empty(cls) -> "CorrespondenceSet":
        """Create an empty correspondence set."""
        print("[TYPES] Instantiating empty CorrespondenceSet instance.")
        return cls(
            reference_points=np.empty((0, 2), dtype=np.float64),
            source_points=np.empty((0, 2), dtype=np.float64),
        )


# [ANNOTATION] Helper function constructing a CorrespondenceSet from a sequence of Correspondence items.
def correspondence_arrays(
    correspondences: Sequence[Correspondence],
) -> CorrespondenceSet:
    """Convert individual correspondences into a CorrespondenceSet."""

    # [ANNOTATION] Handle empty sequences directly.
    if not correspondences:
        return CorrespondenceSet.empty()

    print(f"[TYPES] Converting sequence of {len(correspondences)} Correspondence objects into CorrespondenceSet matrix...")

    # [ANNOTATION] Extract reference points list to float64 array.
    reference_points = np.asarray(
        [item.reference_xy for item in correspondences],
        dtype=np.float64,
    )

    # [ANNOTATION] Extract source points list to float64 array.
    source_points = np.asarray(
        [item.source_xy for item in correspondences],
        dtype=np.float64,
    )

    # [ANNOTATION] Return constructed CorrespondenceSet instance.
    return CorrespondenceSet(
        reference_points=reference_points,
        source_points=source_points,
    )