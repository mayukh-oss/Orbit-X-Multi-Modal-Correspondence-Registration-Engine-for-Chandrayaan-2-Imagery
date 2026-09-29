# [ANNOTATION] Module docstring detailing local feature descriptor matching strategies.
"""
Feature matching for image correspondence.

This module provides a deterministic baseline matcher for local feature
descriptors. Matching strategy is kept separate from feature detection so
that different correspondence strategies can be benchmarked later for
multi-modal Chandrayaan-2 imagery.
"""

# [ANNOTATION] Enable modern typing annotations.
from __future__ import annotations

# [ANNOTATION] Import standard dataclass primitives and literal typing hints.
from dataclasses import dataclass
from typing import Literal

# [ANNOTATION] Import OpenCV for distance metric calculation and NumPy for array operations.
import cv2
import numpy as np

# [ANNOTATION] Import local module types for feature representations and correspondence outcomes.
from .features import FeatureSet
from .types import CorrespondenceSet

# [ANNOTATION] Define supported strategy literals for descriptor filtering.
MatchStrategy = Literal["RATIO", "MUTUAL", "RATIO_MUTUAL"]


# [ANNOTATION] Configuration data structure defining matching strategy and thresholds.
@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for descriptor-based feature matching."""

    strategy: MatchStrategy = "RATIO_MUTUAL"
    ratio_threshold: float = 0.75
    max_matches: int = 2000

    # [ANNOTATION] Post-initialization validation hook ensuring parameter constraints.
    def __post_init__(self) -> None:
        if self.strategy not in {"RATIO", "MUTUAL", "RATIO_MUTUAL"}:
            raise ValueError(f"Unsupported matching strategy: {self.strategy!r}.")

        if not isinstance(self.ratio_threshold, (int, float)):
            raise TypeError("ratio_threshold must be numeric.")

        if not 0.0 < float(self.ratio_threshold) < 1.0:
            raise ValueError("ratio_threshold must be between 0 and 1.")

        if not isinstance(self.max_matches, int) or isinstance(
            self.max_matches, bool
        ):
            raise TypeError("max_matches must be an integer.")

        if self.max_matches <= 0:
            raise ValueError("max_matches must be greater than zero.")


# [ANNOTATION] Internal validation helper ensuring both reference and source feature sets are valid for matching.
def _validate_feature_sets(
    reference: FeatureSet,
    source: FeatureSet,
) -> None:
    """Validate feature sets before descriptor matching."""

    print("[MATCHING] Validating input feature sets and descriptor matrices...")

    if not isinstance(reference, FeatureSet):
        raise TypeError("reference must be a FeatureSet.")

    if not isinstance(source, FeatureSet):
        raise TypeError("source must be a FeatureSet.")

    if reference.descriptors is None or source.descriptors is None:
        raise ValueError("Both feature sets must contain descriptors.")

    if reference.count != reference.descriptors.shape[0]:
        raise ValueError(
            "Reference descriptor count does not match its keypoint count."
        )

    if source.count != source.descriptors.shape[0]:
        raise ValueError(
            "Source descriptor count does not match its keypoint count."
        )

    if reference.descriptors.ndim != 2 or source.descriptors.ndim != 2:
        raise ValueError("Feature descriptors must be two-dimensional.")

    if reference.descriptors.shape[1] != source.descriptors.shape[1]:
        raise ValueError("Reference and source descriptors must have equal dimensions.")

    if reference.descriptors.shape[0] == 0 or source.descriptors.shape[0] == 0:
        raise ValueError("Both feature sets must contain at least one descriptor.")


# [ANNOTATION] Helper function implementing Lowe's ratio test for nearest neighbor filtering.
def _ratio_matches(
    reference_descriptors: np.ndarray,
    source_descriptors: np.ndarray,
    ratio_threshold: float,
) -> list[cv2.DMatch]:
    """Apply Lowe's nearest-neighbor ratio test."""

    print(f"[MATCHING] Running k-NN matcher with Lowe's ratio threshold = {ratio_threshold}...")

    # [ANNOTATION] Instantiate brute-force matcher utilizing L2 Euclidean distance.
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # [ANNOTATION] Perform k-nearest neighbor search with k=2.
    knn_matches = matcher.knnMatch(
        reference_descriptors,
        source_descriptors,
        k=2,
    )

    accepted: list[cv2.DMatch] = []

    # [ANNOTATION] Filter matches based on relative distance ratio between first and second nearest neighbors.
    for pair in knn_matches:
        if len(pair) < 2:
            continue

        first, second = pair

        if second.distance <= 0.0:
            continue

        if first.distance < ratio_threshold * second.distance:
            accepted.append(first)

    print(f"[MATCHING] Ratio test accepted {len(accepted)} matches.")
    return accepted


# [ANNOTATION] Helper function identifying mutual nearest neighbor matches between descriptors.
def _mutual_matches(
    reference_descriptors: np.ndarray,
    source_descriptors: np.ndarray,
) -> list[cv2.DMatch]:
    """Return descriptor matches that are mutual nearest neighbours."""

    print("[MATCHING] Running mutual nearest-neighbor search...")

    forward_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    reverse_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # [ANNOTATION] Perform forward match from reference to source descriptors.
    forward = forward_matcher.match(
        reference_descriptors,
        source_descriptors,
    )

    # [ANNOTATION] Perform backward match from source to reference descriptors.
    reverse = reverse_matcher.match(
        source_descriptors,
        reference_descriptors,
    )

    # [ANNOTATION] Map reverse match lookup table for index verification.
    reverse_best = {
        match.queryIdx: match.trainIdx
        for match in reverse
    }

    accepted: list[cv2.DMatch] = []

    # [ANNOTATION] Keep only matches that are bi-directionally consistent.
    for match in forward:
        if reverse_best.get(match.trainIdx) == match.queryIdx:
            accepted.append(match)

    print(f"[MATCHING] Mutual filtering accepted {len(accepted)} matches.")
    return accepted


# [ANNOTATION] Combined filtering strategy executing ratio test followed by mutual consistency validation.
def _ratio_mutual_matches(
    reference_descriptors: np.ndarray,
    source_descriptors: np.ndarray,
    ratio_threshold: float,
) -> list[cv2.DMatch]:
    """Apply the ratio test followed by mutual-nearest-neighbour filtering."""

    print(f"[MATCHING] Executing combined RATIO_MUTUAL match filtering...")

    ratio_matches = _ratio_matches(
        reference_descriptors,
        source_descriptors,
        ratio_threshold,
    )

    reverse_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    reverse_matches = reverse_matcher.match(
        source_descriptors,
        reference_descriptors,
    )

    reverse_best = {
        match.queryIdx: match.trainIdx
        for match in reverse_matches
    }

    filtered = [
        match
        for match in ratio_matches
        if reverse_best.get(match.trainIdx) == match.queryIdx
    ]

    print(f"[MATCHING] RATIO_MUTUAL filtering yielded {len(filtered)} matches.")
    return filtered


# [ANNOTATION] Deterministic match sorting order based on descriptor distance and query/train indices.
def _sort_matches(matches: list[cv2.DMatch]) -> list[cv2.DMatch]:
    """Sort matches deterministically by descriptor distance and indices."""

    return sorted(
        matches,
        key=lambda match: (
            float(match.distance),
            int(match.queryIdx),
            int(match.trainIdx),
        ),
    )


# [ANNOTATION] Primary feature matching public interface.
def match_features(
    reference: FeatureSet,
    source: FeatureSet,
    *,
    config: MatchingConfig | None = None,
) -> CorrespondenceSet:
    """Match local descriptors between reference and source images.

    Parameters
    ----------
    reference:
        Feature set extracted from the fixed/reference image.
    source:
        Feature set extracted from the moving/source image.
    config:
        Matching strategy and filtering parameters.

    Returns
    -------
    CorrespondenceSet
        Geometric point correspondences. The ``scores`` field contains
        descriptor distances, where lower values indicate closer matches.

    Notes
    -----
    The matcher operates on descriptor similarity only. It does not perform
    geometric verification. RANSAC and spatial-distribution filtering belong
    to later pipeline stages.
    """

    # [ANNOTATION] Default configuration instantiation if none provided.
    if config is None:
        config = MatchingConfig()

    print(f"[MATCHING] Starting feature matching using strategy: {config.strategy}...")

    # [ANNOTATION] Validate descriptor compatibility across both feature sets.
    _validate_feature_sets(reference, source)

    assert reference.descriptors is not None
    assert source.descriptors is not None

    # [ANNOTATION] Branch execution according to chosen matching strategy.
    if config.strategy == "RATIO":
        matches = _ratio_matches(
            reference.descriptors,
            source.descriptors,
            config.ratio_threshold,
        )

    elif config.strategy == "MUTUAL":
        matches = _mutual_matches(
            reference.descriptors,
            source.descriptors,
        )

    else:
        matches = _ratio_mutual_matches(
            reference.descriptors,
            source.descriptors,
            config.ratio_threshold,
        )

    # [ANNOTATION] Sort matches deterministically.
    matches = _sort_matches(matches)

    # [ANNOTATION] Limit matches to the specified maximum match limit.
    if len(matches) > config.max_matches:
        print(f"[MATCHING] Truncating total matches from {len(matches)} to max_matches = {config.max_matches}.")
    matches = matches[: config.max_matches]

    # [ANNOTATION] Extract 2D floating-point coordinate pairs for reference keypoints.
    reference_points = np.asarray(
        [
            reference.keypoints[match.queryIdx].pt
            for match in matches
        ],
        dtype=np.float64,
    ).reshape(-1, 2)

    # [ANNOTATION] Extract 2D floating-point coordinate pairs for source keypoints.
    source_points = np.asarray(
        [
            source.keypoints[match.trainIdx].pt
            for match in matches
        ],
        dtype=np.float64,
    ).reshape(-1, 2)

    # [ANNOTATION] Extract corresponding descriptor distance scores.
    scores = np.asarray(
        [float(match.distance) for match in matches],
        dtype=np.float64,
    )

    print(f"[MATCHING] Successfully constructed CorrespondenceSet with {len(reference_points)} correspondences.")

    # [ANNOTATION] Return verified geometric point correspondence set container.
    return CorrespondenceSet(
        reference_points=reference_points,
        source_points=source_points,
        scores=scores,
    )