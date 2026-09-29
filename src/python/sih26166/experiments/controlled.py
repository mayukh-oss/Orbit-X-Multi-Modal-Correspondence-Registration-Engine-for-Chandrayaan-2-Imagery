# [ANNOTATION] Module docstring describing controlled validation pair generation utilities.
"""
Controlled validation experiment generation.

This module creates synthetic reference/source image pairs with known
geometric and configurable photometric changes.

IMPORTANT:
    Outputs from this module are explicitly labeled
    CONTROLLED_VALIDATION. They are not real Chandrayaan-2 products.

The known transformation is stored as:

    reference -> source

SIH 2026 | PS ID: SIH26166
"""

# [ANNOTATION] Enable future annotation syntax support.
from __future__ import annotations

# [ANNOTATION] Import standard library modules for JSON handling, dataclasses, and path operations.
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# [ANNOTATION] Import NumPy for mathematical grid calculations and array operations.
import numpy as np
# [ANNOTATION] Import Pillow Image class for image synthesis and transformation operations.
from PIL import Image


# [ANNOTATION] Dataclass configuring synthetic photometric variations (gamma, contrast, illumination gradients).
@dataclass(frozen=True)
class PhotometricConfig:
    """Configurable photometric variation."""

    gamma: float = 0.95
    contrast: float = 1.05
    gradient_strength: float = 0.12
    gradient_angle_deg: float = 35.0

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive.")

        if self.contrast <= 0.0:
            raise ValueError("contrast must be positive.")

        if not np.isfinite(self.gradient_strength):
            raise ValueError(
                "gradient_strength must be finite."
            )

        if not np.isfinite(self.gradient_angle_deg):
            raise ValueError(
                "gradient_angle_deg must be finite."
            )


# [ANNOTATION] Dataclass configuring parameters for a synthetic controlled experiment run.
@dataclass(frozen=True)
class ControlledExperimentConfig:
    """Configuration for one controlled validation experiment."""

    reference_image: str | Path
    output_directory: str | Path
    run_id: str = "controlled_001"

    seed: int = 26166

    scale: float = 1.15
    rotation_deg: float = 12.5
    translation_x: float = 35.0
    translation_y: float = -22.0

    photometric: PhotometricConfig = PhotometricConfig()

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise ValueError(
                "run_id must not be empty."
            )

        if self.scale <= 0.0:
            raise ValueError(
                "scale must be positive."
            )

        if not np.isfinite(self.scale):
            raise ValueError(
                "scale must be finite."
            )

        if not np.isfinite(self.rotation_deg):
            raise ValueError(
                "rotation_deg must be finite."
            )

        if not np.isfinite(self.translation_x):
            raise ValueError(
                "translation_x must be finite."
            )

        if not np.isfinite(self.translation_y):
            raise ValueError(
                "translation_y must be finite."
            )


# [ANNOTATION] Dataclass output storing result file paths and ground truth metadata.
@dataclass(frozen=True)
class ControlledExperimentResult:
    """Paths and ground truth for a generated experiment."""

    mode: str
    run_id: str
    output_directory: Path
    reference_image: Path
    source_image: Path
    ground_truth: dict
    config: ControlledExperimentConfig


# [ANNOTATION] Helper function constructing forward reference-to-source affine transformation matrix.
def _forward_matrix(
    width: int,
    height: int,
    scale: float,
    rotation_deg: float,
    translation_x: float,
    translation_y: float,
) -> np.ndarray:
    """
    Build the reference -> source affine transformation.

    Rotation and scaling occur around the image center, followed by the
    requested translation.
    """
    print(f"[CONTROLLED] Constructing 3x3 forward affine matrix (Scale: {scale}, Rot: {rotation_deg}°)...")
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    theta = np.deg2rad(rotation_deg)
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))

    rotation_scale = np.array(
        [
            [
                scale * cosine,
                -scale * sine,
            ],
            [
                scale * sine,
                scale * cosine,
            ],
        ],
        dtype=np.float64,
    )

    center = np.array(
        [center_x, center_y],
        dtype=np.float64,
    )

    translation = np.array(
        [translation_x, translation_y],
        dtype=np.float64,
    )

    offset = center + translation - rotation_scale @ center

    matrix = np.eye(
        3,
        dtype=np.float64,
    )

    matrix[:2, :2] = rotation_scale
    matrix[:2, 2] = offset

    return matrix


# [ANNOTATION] Helper function calculating inverse affine coefficients required by Pillow image transform.
def _inverse_affine_coefficients(
    matrix: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Convert a reference->source affine matrix to Pillow inverse coefficients.

    Pillow's affine transform maps output coordinates back to input
    coordinates.
    """
    print("[CONTROLLED] Inverting affine matrix for Pillow resampling transform...")
    inverse = np.linalg.inv(matrix)

    return (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2]),
    )


# [ANNOTATION] Helper function applying gamma, contrast, and illumination gradient changes to an image array.
def _apply_photometric(
    image: np.ndarray,
    config: PhotometricConfig,
) -> np.ndarray:
    """Apply configurable photometric variation."""
    print(f"[CONTROLLED] Applying synthetic photometric variation (Gamma: {config.gamma}, Contrast: {config.contrast})...")
    result = image.astype(
        np.float64,
        copy=True,
    )

    result = result / 255.0

    result = np.power(
        np.clip(result, 0.0, 1.0),
        config.gamma,
    )

    result = (
        (result - 0.5) * config.contrast
        + 0.5
    )

    height, width = result.shape

    y, x = np.mgrid[
        0:height,
        0:width,
    ]

    angle = np.deg2rad(
        config.gradient_angle_deg
    )

    direction_x = np.cos(angle)
    direction_y = np.sin(angle)

    normalized = (
        x * direction_x
        + y * direction_y
    )

    minimum = float(normalized.min())
    maximum = float(normalized.max())

    if maximum > minimum:
        normalized = (
            normalized - minimum
        ) / (maximum - minimum)
    else:
        normalized = np.zeros_like(
            normalized,
            dtype=np.float64,
        )

    result = result * (
        1.0
        + config.gradient_strength
        * (normalized - 0.5)
    )

    result = np.clip(
        result,
        0.0,
        1.0,
    )

    return np.rint(
        result * 255.0
    ).astype(np.uint8)


# [ANNOTATION] Helper function serializing dataclasses into JSON-compatible dictionaries.
def _serialize_config(
    config: ControlledExperimentConfig,
) -> dict:
    """Serialize experiment configuration into JSON-compatible data."""
    data = asdict(config)

    data["reference_image"] = str(
        config.reference_image
    )

    data["output_directory"] = str(
        config.output_directory
    )

    return data


# [ANNOTATION] Primary function executing the creation of a controlled benchmark experiment dataset.
def generate_controlled_experiment(
    config: ControlledExperimentConfig,
) -> ControlledExperimentResult:
    """
    Generate one controlled validation experiment.

    Outputs:

        reference.png
        source.png
        ground_truth.json
        config.json

    The generated source image contains the configured geometric and
    photometric variation relative to the reference image.
    """
    print(f"\n[CONTROLLED] Generating synthetic experiment pair '{config.run_id}'...")

    reference_path = Path(
        config.reference_image
    )

    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference image does not exist: {reference_path}"
        )

    output_directory = (
        Path(config.output_directory)
        / config.run_id
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_image = Image.open(
        reference_path
    ).convert("L")

    reference_array = np.asarray(
        reference_image,
        dtype=np.uint8,
    )

    if reference_array.size == 0:
        raise ValueError(
            "Reference image must not be empty."
        )

    height, width = reference_array.shape

    matrix = _forward_matrix(
        width=width,
        height=height,
        scale=config.scale,
        rotation_deg=config.rotation_deg,
        translation_x=config.translation_x,
        translation_y=config.translation_y,
    )

    coefficients = _inverse_affine_coefficients(
        matrix
    )

    print("[CONTROLLED] Transforming reference image array to generate synthetic source...")
    source_image = reference_image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )

    source_array = np.asarray(
        source_image,
        dtype=np.uint8,
    )

    source_array = _apply_photometric(
        source_array,
        config.photometric,
    )

    source_image = Image.fromarray(
        source_array,
        mode="L",
    )

    reference_output = (
        output_directory / "reference.png"
    )

    source_output = (
        output_directory / "source.png"
    )

    ground_truth_output = (
        output_directory / "ground_truth.json"
    )

    config_output = (
        output_directory / "config.json"
    )

    print(f"[CONTROLLED] Saving generated experiment files to: {output_directory}")
    reference_image.save(
        reference_output
    )

    source_image.save(
        source_output
    )

    ground_truth = {
        "mode": "CONTROLLED_VALIDATION",
        "reference_to_source_matrix": (
            matrix.tolist()
        ),
        "scale": config.scale,
        "rotation_deg": config.rotation_deg,
        "translation_x": config.translation_x,
        "translation_y": config.translation_y,
        "image_width": width,
        "image_height": height,
        "photometric": asdict(
            config.photometric
        ),
    }

    ground_truth_output.write_text(
        json.dumps(
            ground_truth,
            indent=2,
        ),
        encoding="utf-8",
    )

    config_output.write_text(
        json.dumps(
            _serialize_config(config),
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[CONTROLLED] Controlled experiment generation completed successfully.\n")

    return ControlledExperimentResult(
        mode="CONTROLLED_VALIDATION",
        run_id=config.run_id,
        output_directory=output_directory,
        reference_image=reference_output,
        source_image=source_output,
        ground_truth=ground_truth,
        config=config,
    )