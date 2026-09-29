# [ANNOTATION] Module docstring describing random-access reader for large binary PDS4 raster files.
"""
Windowed readers for raw PDS4 raster image data.

The reader is intentionally designed for large Chandrayaan-2 products:
it reads only the requested image window instead of loading the entire
binary product into memory.
"""

# [ANNOTATION] Enable future type hint syntax.
from __future__ import annotations

# [ANNOTATION] Import standard dataclass primitives and Path utilities.
from dataclasses import dataclass
from pathlib import Path

# [ANNOTATION] Import NumPy for raw buffer loading.
import numpy as np

# [ANNOTATION] Import PDS4ImageMetadata type container from local module.
from sih26166.io.pds4 import PDS4ImageMetadata


# [ANNOTATION] Dataclass representing a bounding box window definition (x, y, width, height).
@dataclass(frozen=True)
class ImageWindow:
    """A rectangular image window in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


# [ANNOTATION] Class implementing random-access file streaming to read specific row-major pixel windows.
class RawImageReader:
    """
    Random-access reader for a PDS4 2-D raw image.

    Coordinates use image convention:
        x = Sample / column
        y = Line / row

    The returned array has shape:
        (height, width)
    """

    def __init__(
        self,
        image_path: str | Path,
        metadata: PDS4ImageMetadata,
    ) -> None:
        self.image_path = Path(image_path)
        self.metadata = metadata

        if not self.image_path.is_file():
            raise FileNotFoundError(
                f"Raw image not found: {self.image_path}"
            )

        if metadata.width is None or metadata.height is None:
            raise ValueError(
                "PDS4 metadata does not contain image dimensions."
            )

        if metadata.offset_bytes is None:
            raise ValueError(
                "PDS4 metadata does not contain image offset."
            )

        if metadata.data_type != "UnsignedByte":
            raise ValueError(
                "RawImageReader currently supports only "
                f"UnsignedByte data, got {metadata.data_type!r}."
            )

        self.width = metadata.width
        self.height = metadata.height
        self.offset_bytes = metadata.offset_bytes
        self.dtype = np.dtype(np.uint8)

        self._bytes_per_pixel = self.dtype.itemsize
        print(f"[RAW_IMAGE] Initialized RawImageReader for '{self.image_path.name}' ({self.width}x{self.height}).")

    @property
    def shape(self) -> tuple[int, int]:
        """Return image shape as (height, width)."""
        return self.height, self.width

    @property
    def size_bytes(self) -> int:
        """Expected binary image size from metadata."""
        return (
            self.width
            * self.height
            * self._bytes_per_pixel
        )

    # [ANNOTATION] Helper function verifying window coordinates stay strictly within image bounds.
    def _validate_window(self, window: ImageWindow) -> None:
        """Validate an image window against image bounds."""
        if window.width <= 0:
            raise ValueError("Window width must be positive.")

        if window.height <= 0:
            raise ValueError("Window height must be positive.")

        if window.x < 0 or window.y < 0:
            raise ValueError(
                "Window origin must be non-negative."
            )

        if window.x + window.width > self.width:
            raise ValueError(
                "Window extends beyond image width: "
                f"x={window.x}, width={window.width}, "
                f"image_width={self.width}"
            )

        if window.y + window.height > self.height:
            raise ValueError(
                "Window extends beyond image height: "
                f"y={window.y}, height={window.height}, "
                f"image_height={self.height}"
            )

    # [ANNOTATION] Method seeking and reading only requested row bytes from disk without full product memory load.
    def read_window(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Read a rectangular image window.

        Only the requested rows are read from disk.

        Parameters
        ----------
        x, y:
            Top-left pixel coordinate.
        width, height:
            Window dimensions.

        Returns
        -------
        numpy.ndarray
            uint8 array with shape (height, width).
        """
        window = ImageWindow(
            x=x,
            y=y,
            width=width,
            height=height,
        )

        self._validate_window(window)

        print(f"[RAW_IMAGE] Streaming window {width}x{height} starting at row {y}, col {x}...")

        result = np.empty(
            (window.height, window.width),
            dtype=self.dtype,
        )

        row_bytes = window.width * self._bytes_per_pixel

        with self.image_path.open("rb") as handle:
            for row in range(window.height):
                absolute_row = window.y + row

                byte_offset = (
                    self.offset_bytes
                    + (
                        absolute_row
                        * self.width
                        * self._bytes_per_pixel
                    )
                    + (
                        window.x
                        * self._bytes_per_pixel
                    )
                )

                handle.seek(byte_offset)

                data = handle.read(row_bytes)

                if len(data) != row_bytes:
                    raise IOError(
                        "Unexpected end of image file while reading "
                        f"row {absolute_row}."
                    )

                result[row, :] = np.frombuffer(
                    data,
                    dtype=self.dtype,
                    count=window.width,
                )

        return result

    # [ANNOTATION] Reads full image matrix from start coordinate (0, 0).
    def read_full(self) -> np.ndarray:
        """
        Read the entire image.

        This method is provided explicitly for controlled use and may
        require approximately the full image size in RAM.
        """
        print("[RAW_IMAGE] Warning: Reading full binary image into memory...")
        return self.read_window(
            x=0,
            y=0,
            width=self.width,
            height=self.height,
        )

    # [ANNOTATION] Validates physical file size on disk against offset and declared dimensions.
    def validate_file_size(self) -> bool:
        """
        Validate that the physical file is large enough for the
        PDS4-declared raster.

        Returns True when the file size is exactly the expected size
        after accounting for the declared offset.
        """
        actual_size = self.image_path.stat().st_size

        expected_size = (
            self.offset_bytes
            + self.size_bytes
        )

        return actual_size == expected_size