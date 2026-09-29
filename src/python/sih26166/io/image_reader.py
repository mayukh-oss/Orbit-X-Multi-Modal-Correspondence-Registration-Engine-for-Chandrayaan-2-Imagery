# [ANNOTATION] Module docstring describing binary image reader supporting PDS4 image data types.
"""PDS4-compatible binary image reader.

Supports the two image encodings currently encountered in SIH26166:
- UnsignedByte / 8-bit
- UnsignedLSB2 / little-endian unsigned 16-bit

The reader preserves the original radiometric representation.
"""

# [ANNOTATION] Enable modern type hint annotations.
from __future__ import annotations

# [ANNOTATION] Import standard dataclass primitives and Path handling utilities.
from dataclasses import dataclass
from pathlib import Path

# [ANNOTATION] Import NumPy for binary mapping and data type resolution.
import numpy as np

ImageDType = np.dtype


# [ANNOTATION] Immutable dataclass specifying image dimensions and numerical data type parameters.
@dataclass(frozen=True)
class ImageSpec:
    width: int
    height: int
    dtype: str = "uint8"

    # [ANNOTATION] Property mapping text data type representations to NumPy dtype objects.
    @property
    def numpy_dtype(self) -> np.dtype:
        key = self.dtype.lower().replace(" ", "")
        if key in {"uint8", "unsignedbyte", "unsignedbyte1"}:
            return np.dtype("u1")
        if key in {
            "uint16",
            "unsignedlsb2",
            "unsignedlittleendian2",
        }:
            return np.dtype("<u2")
        raise ValueError(f"Unsupported image dtype: {self.dtype}")

    # [ANNOTATION] Property calculating expected binary file size in bytes based on shape and dtype size.
    @property
    def expected_bytes(self) -> int:
        return self.width * self.height * self.numpy_dtype.itemsize


# [ANNOTATION] Core function reading a 2D row-major binary image matrix via direct file reading or memory mapping.
def read_image(
    path: str | Path,
    spec: ImageSpec,
    *,
    mmap: bool = True,
    copy: bool = False,
) -> np.ndarray:
    """Read a row-major PDS4 Array_2D_Image.

    The file is validated against the expected byte count before
    exposing it as an ndarray.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    actual = path.stat().st_size
    expected = spec.expected_bytes

    print(f"[IMAGE_READER] Reading image '{path.name}' ({spec.width}x{spec.height}, dtype={spec.dtype})...")

    if actual != expected:
        raise ValueError(
            f"Image size mismatch for {path}: "
            f"actual={actual}, expected={expected}"
        )

    mode = "r" if mmap else "r+"

    if mmap:
        print("[IMAGE_READER] Memory mapping file buffer via np.memmap...")
    
    arr = np.memmap(
        path,
        dtype=spec.numpy_dtype,
        mode=mode,
        shape=(spec.height, spec.width),
        order="C",
    )

    if copy:
        print("[IMAGE_READER] Copying memory-mapped buffer into RAM array...")
        return np.asarray(arr).copy()

    return arr


# [ANNOTATION] Helper function validating raw image binary file byte size without loading pixel data.
def validate_image_file(path: str | Path, spec: ImageSpec) -> None:
    """Validate a binary image without loading it."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    actual = path.stat().st_size
    expected = spec.expected_bytes

    print(f"[IMAGE_READER] Validating file size for '{path.name}' (Expected: {expected} bytes)...")

    if actual != expected:
        raise ValueError(
            f"Image size mismatch: actual={actual}, expected={expected}"
        )
    print("[IMAGE_READER] File size validation successful.")