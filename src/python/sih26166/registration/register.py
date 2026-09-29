# [ANNOTATION] Module docstring describing image warping and registration utilities.
"""
Image registration utilities.

This module applies an already-estimated geometric transformation to a source
image so that it aligns with a reference image.

Transformation estimation is deliberately kept separate from image warping:
the verification stage determines the transformation, while this module
performs the registration itself.

Supported models:
    - AFFINE
    - HOMOGRAPHY

The transformation convention is:

    reference coordinates -> source coordinates

Therefore, when producing a registered source image in the reference frame,
the inverse mapping is used internally by OpenCV's warp functions.
"""

# [ANNOTATION] Enable modern type hint annotations.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclass primitives and typing literals.
from dataclasses import dataclass
from typing import Literal

# [ANNOTATION] Import OpenCV for image warping algorithms and NumPy for matrix manipulation.
import cv2
import numpy as np

# [ANNOTATION] Define supported registration model literal types.
RegistrationModel = Literal["AFFINE", "HOMOGRAPHY"]


# [ANNOTATION] Immutable dataclass configuring image registration and warping parameters.
@dataclass(frozen=True)
class RegistrationConfig:
    """Configuration for image registration."""

    model: RegistrationModel = "AFFINE"
    interpolation: int = cv2.INTER_LINEAR
    border_mode: int = cv2.BORDER_CONSTANT
    border_value: float = 0.0

    # [ANNOTATION] Post-initialization validation ensuring parameter constraints.
    def __post_init__(self) -> None:
        if self.model not in {"AFFINE", "HOMOGRAPHY"}:
            raise ValueError(
                f"Unsupported registration model: {self.model!r}."
            )

        if self.interpolation < 0:
            raise ValueError(
                "interpolation must be a valid OpenCV interpolation flag."
            )

        if self.border_mode < 0:
            raise ValueError(
                "border_mode must be a valid OpenCV border mode."
            )

        if not np.isfinite(self.border_value):
            raise ValueError(
                "border_value must be finite."
            )


# [ANNOTATION] Immutable dataclass containing registered output image array and associated metadata.
@dataclass(frozen=True)
class RegistrationResult:
    """Result of applying a geometric registration."""

    registered_image: np.ndarray
    transformation: np.ndarray
    model: RegistrationModel
    output_shape: tuple[int, int]

    # [ANNOTATION] Property returning output registered image height.
    @property
    def height(self) -> int:
        """Return the registered image height."""
        return int(self.registered_image.shape[0])

    # [ANNOTATION] Property returning output registered image width.
    @property
    def width(self) -> int:
        """Return the registered image width."""
        return int(self.registered_image.shape[1])


# [ANNOTATION] Helper function validating image array structure and values before warping.
def _validate_image(
    image: np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate an image supplied for registration."""
    print(f"[REGISTER] Validating input image '{name}' for registration warping...")
    array = np.asarray(image)

    if array.ndim not in {2, 3}:
        raise ValueError(
            f"{name} must be a 2-D or 3-D image."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(
            f"{name} must have positive spatial dimensions."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"{name} must contain numeric values."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


# [ANNOTATION] Helper function validating and normalizing geometric transformation matrix shapes.
def _validate_transformation(
    transformation: np.ndarray,
    model: RegistrationModel,
) -> np.ndarray:
    """Validate and normalize a transformation matrix."""
    print(f"[REGISTER] Validating transformation matrix for model '{model}'...")
    matrix = np.asarray(
        transformation,
        dtype=np.float64,
    )

    expected_shape = (
        (2, 3)
        if model == "AFFINE"
        else (3, 3)
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"{model} transformation must have shape "
            f"{expected_shape}, got {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "transformation must contain only finite values."
        )

    if model == "HOMOGRAPHY":
        if abs(float(matrix[2, 2])) < np.finfo(float).eps:
            raise ValueError(
                "HOMOGRAPHY transformation has an invalid scale term."
            )

        matrix = matrix / matrix[2, 2]

    return matrix


# [ANNOTATION] Primary function warping moving source image into fixed reference coordinate frame.
def register_image(
    source_image: np.ndarray,
    reference_shape: tuple[int, int],
    transformation: np.ndarray,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    """
    Warp a source image into the reference-image coordinate frame.

    Parameters
    ----------
    source_image:
        Moving/source image to be registered.

    reference_shape:
        Target image shape as ``(height, width)``.

    transformation:
        Transformation mapping reference coordinates to source coordinates.

        For AFFINE:
            shape (2, 3)

        For HOMOGRAPHY:
            shape (3, 3)

    config:
        Registration configuration.

    Returns
    -------
    RegistrationResult
        Registered image and transformation metadata.

    Notes
    -----
    OpenCV's warpAffine/warpPerspective functions normally interpret the
    supplied matrix as a forward mapping from source to destination unless
    WARP_INVERSE_MAP is specified.

    Our project convention is reference -> source, so WARP_INVERSE_MAP is
    used directly.
    """
    if config is None:
        config = RegistrationConfig()

    print(f"[REGISTER] Registering source image into target frame {reference_shape[1]}x{reference_shape[0]} ({config.model})...")

    source = _validate_image(
        source_image,
        "source_image",
    )

    if len(reference_shape) != 2:
        raise ValueError(
            "reference_shape must contain (height, width)."
        )

    reference_height, reference_width = (
        int(reference_shape[0]),
        int(reference_shape[1]),
    )

    if reference_height <= 0 or reference_width <= 0:
        raise ValueError(
            "reference_shape dimensions must be positive."
        )

    matrix = _validate_transformation(
        transformation,
        config.model,
    )

    # [ANNOTATION] Apply appropriate warping function depending on geometric model type.
    if config.model == "AFFINE":
        print("[REGISTER] Applying cv2.warpAffine with WARP_INVERSE_MAP...")
        registered = cv2.warpAffine(
            source,
            matrix,
            (reference_width, reference_height),
            flags=config.interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=config.border_mode,
            borderValue=config.border_value,
        )
    else:
        print("[REGISTER] Applying cv2.warpPerspective with WARP_INVERSE_MAP...")
        registered = cv2.warpPerspective(
            source,
            matrix,
            (reference_width, reference_height),
            flags=config.interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=config.border_mode,
            borderValue=config.border_value,
        )

    print(f"[REGISTER] Image registration complete. Result shape: {registered.shape}.")

    return RegistrationResult(
        registered_image=registered,
        transformation=matrix.copy(),
        model=config.model,
        output_shape=(
            reference_height,
            reference_width,
        ),
    )