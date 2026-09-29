# [ANNOTATION] Module docstring describing extraction utilities for large PDS4 raster image windows.
"""
Real Chandrayaan-2 image extraction utilities.

This module extracts spatially referenced windows from large PDS4
raster products without loading the complete product into memory.

The extracted pixel array remains the original raster values. Any
normalization is applied only to the visualization/processing copy.

Outputs are explicitly marked as REAL_CHANDRAYAAN_2.
"""

# [ANNOTATION] Enable future annotation evaluation support.
from __future__ import annotations

# [ANNOTATION] Import standard dataclasses and path helpers.
from dataclasses import dataclass
from pathlib import Path

# [ANNOTATION] Import NumPy for matrix operations and Pillow for image file saving.
import numpy as np
from PIL import Image

# [ANNOTATION] Import PDS4 label reader and raw binary window reader utilities.
from sih26166.io.pds4 import PDS4ImageMetadata, read_pds4_label
from sih26166.io.raw_image import RawImageReader


# [ANNOTATION] Dataclass representing an extracted PDS4 raster window and its spatial coordinates.
@dataclass(frozen=True)
class ExtractedWindow:
    """A real PDS4 image window with its original image coordinates."""

    product_id: str
    x: int
    y: int
    width: int
    height: int
    pixels: np.ndarray
    mode: str = "REAL_CHANDRAYAAN_2"

    @property
    def shape(self) -> tuple[int, int]:
        """Return the extracted pixel shape as (height, width)."""
        return self.pixels.shape


# [ANNOTATION] Main function extracting a rectangular pixel window from a PDS4 raw product without full load.
def extract_window(
    image_path: str | Path,
    label_path: str | Path,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[PDS4ImageMetadata, ExtractedWindow]:
    """
    Extract a window from a real PDS4 raster product.

    The complete source image is never loaded into memory.

    Coordinates are in source-image pixels:
        x = sample / column
        y = line / row

    Returns
    -------
    metadata, ExtractedWindow
        PDS4 metadata and the extracted real-data window.
    """
    print(f"[EXTRACT] Reading PDS4 label: {label_path}...")
    metadata = read_pds4_label(label_path)
    
    print(f"[EXTRACT] Reading raw window at x={x}, y={y}, {width}x{height} from: {image_path}...")
    reader = RawImageReader(image_path, metadata)

    pixels = reader.read_window(
        x=x,
        y=y,
        width=width,
        height=height,
    )

    extracted = ExtractedWindow(
        product_id=metadata.product_id, # type: ignore
        x=x,
        y=y,
        width=width,
        height=height,
        pixels=pixels,
    )

    print(f"[EXTRACT] Successfully extracted window with shape {pixels.shape}.")
    return metadata, extracted


# [ANNOTATION] Helper function applying percentile dynamic range stretching to produce an 8-bit visualization copy.
def normalize_for_visualization(
    pixels: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """
    Create an 8-bit visualization copy using percentile stretching.

    This does NOT modify the original extracted pixel values.

    Percentile stretching is intended only for visualization and
    downstream experimentation. It should not be interpreted as a
    scientifically validated radiometric correction.
    """
    print(f"[EXTRACT] Normalizing pixels for visualization using percentiles [{lower_percentile}%, {upper_percentile}%]...")
    array = np.asarray(pixels)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale array, got shape {array.shape}."
        )

    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower_percentile < upper_percentile <= 100."
        )

    finite = array[np.isfinite(array)]

    if finite.size == 0:
        raise ValueError("Image contains no finite pixels.")

    low = float(np.percentile(finite, lower_percentile))
    high = float(np.percentile(finite, upper_percentile))

    if high <= low:
        print("[EXTRACT] Warning: Flat intensity range detected. Returning zeros.")
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = (array.astype(np.float32) - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    return np.rint(normalized * 255.0).astype(np.uint8)


# [ANNOTATION] Helper function saving a normalized visualization image file as PNG.
def save_visualization(
    extracted: ExtractedWindow,
    output_path: str | Path,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Path:
    """
    Save a visualization copy of a real extracted window as PNG.

    The PNG contains the normalized visualization pixels, while the
    filename and console metadata make its real-data provenance clear.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    visualization = normalize_for_visualization(
        extracted.pixels,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    print(f"[EXTRACT] Saving normalized visualization PNG to: {output}")
    Image.fromarray(visualization, mode="L").save(output)

    return output