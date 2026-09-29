# [ANNOTATION] Module docstring detailing the feature detection and description process for image correspondence.
"""
Feature detection and description for image correspondence.

The implementation is intentionally detector-agnostic at the public API
level so that different feature methods can be benchmarked later under
scale, illumination, viewpoint, and cross-modal conditions.
"""

# [ANNOTATION] Enable modern type annotations for type hinting without evaluating them at runtime.
from __future__ import annotations

# [ANNOTATION] Import standard dataclass primitives and typing constructs.
from dataclasses import dataclass
from typing import Literal

# [ANNOTATION] Import OpenCV for computer vision tasks and NumPy for multi-dimensional matrix operations.
import cv2
import numpy as np

# [ANNOTATION] Define supported feature extraction method literal types.
FeatureMethod = Literal["SIFT"]


# [ANNOTATION] Define an immutable data class to encapsulate detected keypoints, descriptors, and image shape.
@dataclass(frozen=True)
class FeatureSet:
    """Detected local features and their descriptors."""

    # [ANNOTATION] Tuple storing OpenCV KeyPoint objects.
    keypoints: tuple[cv2.KeyPoint, ...]
    # [ANNOTATION] 2D NumPy array storing descriptor vectors or None if no features found.
    descriptors: np.ndarray | None
    # [ANNOTATION] Height and width of the processed image as a tuple.
    image_shape: tuple[int, int]
    # [ANNOTATION] Method identifier string (e.g. "SIFT").
    method: FeatureMethod

    # [ANNOTATION] Property helper returning the total count of keypoints in this feature set.
    @property
    def count(self) -> int:
        """Return the number of detected keypoints."""
        return len(self.keypoints)


# [ANNOTATION] Internal validation and normalization helper function for input image arrays.
def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate and normalize an input image for feature extraction."""

    # [ANNOTATION] Trace message when image array validation begins.
    print("[FEATURES] Validating input image array type and dimensions...")

    # [ANNOTATION] Ensure the input is a valid NumPy ndarray instance.
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray.")

    # [ANNOTATION] Validate image matrix dimensions (must be 2D grayscale or 3D color).
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2-D grayscale or 3-D color array.")

    # [ANNOTATION] Ensure array contains at least one pixel element.
    if image.size == 0:
        raise ValueError("image must not be empty.")

    # [ANNOTATION] Verify that the data type is numerical.
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image must contain numeric pixel values.")

    # [ANNOTATION] Check color channel layout for 3D image arrays.
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError(
            "Color images must have 3 channels (RGB/BGR) or 4 channels (RGBA/BGRA)."
        )

    # [ANNOTATION] Ensure image matrix does not contain NaN or infinity values.
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values.")

    # [ANNOTATION] Return immediately if the array is already 8-bit unsigned integer type.
    if image.dtype == np.uint8:
        print("[FEATURES] Input image is valid uint8 format.")
        return image

    # [ANNOTATION] Calculate minimum and maximum intensity values for scaling.
    minimum = float(np.min(image))
    maximum = float(np.max(image))

    # [ANNOTATION] If image intensity is constant, return an all-zero matrix.
    if maximum <= minimum:
        print("[FEATURES] Image intensity range is flat; returning zeroed array.")
        return np.zeros(image.shape, dtype=np.uint8)

    # [ANNOTATION] Perform linear dynamic range normalization into uint8 scale [0, 255].
    print(f"[FEATURES] Normalizing array intensity from [{minimum:.2f}, {maximum:.2f}] to [0, 255] uint8...")
    normalized = (image.astype(np.float32) - minimum) / (maximum - minimum)
    return np.rint(normalized * 255.0).astype(np.uint8)


# [ANNOTATION] Helper function converting a validated image array to single-channel grayscale.
def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a validated image to 8-bit grayscale."""

    # [ANNOTATION] Validate and obtain uint8 normalized image.
    normalized = _validate_image(image)

    # [ANNOTATION] If already 2D grayscale, return normalized array directly.
    if normalized.ndim == 2:
        return normalized

    # [ANNOTATION] Convert 4-channel BGRA images to grayscale using OpenCV.
    if normalized.shape[2] == 4:
        print("[FEATURES] Converting 4-channel BGRA image to grayscale...")
        return cv2.cvtColor(normalized, cv2.COLOR_BGRA2GRAY)

    # [ANNOTATION] Convert 3-channel BGR images to grayscale using OpenCV.
    print("[FEATURES] Converting 3-channel BGR image to grayscale...")
    return cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)


# [ANNOTATION] Primary feature extraction entry function using specified detector and descriptor algorithm.
def detect_features(
    image: np.ndarray,
    *,
    method: FeatureMethod = "SIFT",
    max_features: int = 2000,
) -> FeatureSet:
    """Detect local features and compute their descriptors.

    Parameters
    ----------
    image:
        Input grayscale or color image.
    method:
        Feature detector/descriptor implementation.
    max_features:
        Maximum number of retained keypoints.

    Returns
    -------
    FeatureSet
        Keypoints, descriptors, image dimensions, and method metadata.

    Notes
    -----
    SIFT is used as the first baseline because it provides scale-aware
    local features and descriptors. It is a baseline, not a claim that
    SIFT is the final solution for cross-modal Chandrayaan-2 imagery.
    """

    # [ANNOTATION] Validate requested feature extraction method parameter.
    if method != "SIFT":
        raise ValueError(f"Unsupported feature method: {method!r}.")

    # [ANNOTATION] Validate target maximum feature count parameter type and bounds.
    if not isinstance(max_features, int) or isinstance(max_features, bool):
        raise TypeError("max_features must be an integer.")

    if max_features <= 0:
        raise ValueError("max_features must be greater than zero.")

    # [ANNOTATION] Convert input image matrix to standard grayscale representation.
    grayscale = _to_grayscale(image)
    height, width = grayscale.shape

    # [ANNOTATION] Trace feature extraction startup parameters.
    print(f"[FEATURES] Detecting up to {max_features} SIFT features on image dimensions ({width}x{height})...")

    # [ANNOTATION] Instantiate OpenCV SIFT detector object.
    detector = cv2.SIFT_create(nfeatures=max_features) # type: ignore

    # [ANNOTATION] Execute detection and descriptor extraction pipeline on grayscale image.
    keypoints, descriptors = detector.detectAndCompute(grayscale, None)

    # [ANNOTATION] Handle cases where no keypoints are detected gracefully.
    if keypoints is None:
        print("[FEATURES] Warning: No keypoints were detected in the input image.")
        keypoints = []

    # [ANNOTATION] Cast keypoints list into an immutable tuple for thread safety.
    keypoints = tuple(keypoints)

    # [ANNOTATION] Validate and convert descriptor matrix shape if descriptors were found.
    if descriptors is not None:
        descriptors = np.asarray(descriptors, dtype=np.float32)

        if descriptors.ndim != 2:
            raise RuntimeError("Feature descriptors must have shape (N, D).")

        if descriptors.shape[0] != len(keypoints):
            raise RuntimeError(
                "The number of descriptors must equal the number of keypoints."
            )

    # [ANNOTATION] Log total extracted features summary.
    print(f"[FEATURES] Feature extraction complete. Extracted {len(keypoints)} keypoints with descriptors.")

    # [ANNOTATION] Return typed FeatureSet dataclass output.
    return FeatureSet(
        keypoints=keypoints,
        descriptors=descriptors,
        image_shape=(height, width),
        method=method,
    )