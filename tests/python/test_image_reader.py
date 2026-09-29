# [ANNOTATION] Unit tests for PDS4 binary image reader and ImageSpec configurations.
from pathlib import Path

import numpy as np
import pytest

from sih26166.io.image_reader import ImageSpec, read_image


# [ANNOTATION] Test verifying uint8 UnsignedByte ImageSpec data type mapping and expected byte calculations.
def test_uint8_spec():
    print("\n[TEST] Executing test_uint8_spec...")
    spec = ImageSpec(10, 20, "UnsignedByte")
    assert spec.numpy_dtype == np.dtype("u1")
    assert spec.expected_bytes == 200


# [ANNOTATION] Test verifying uint16 UnsignedLSB2 ImageSpec data type mapping.
def test_uint16_spec():
    print("\n[TEST] Executing test_uint16_spec...")
    spec = ImageSpec(10, 20, "UnsignedLSB2")
    assert spec.numpy_dtype == np.dtype("<u2")
    assert spec.expected_bytes == 400


# [ANNOTATION] Test verifying unsupported data type string raises ValueError.
def test_unsupported_dtype():
    print("\n[TEST] Executing test_unsupported_dtype...")
    with pytest.raises(ValueError):
        ImageSpec(10, 20, "Float32").numpy_dtype


# [ANNOTATION] Test reading binary uint16 raster file array from disk using read_image.
def test_read_uint16(tmp_path: Path):
    print("\n[TEST] Executing test_read_uint16...")
    data = np.arange(20, dtype="<u2").reshape(4, 5)
    path = tmp_path / "test.img"
    data.tofile(path)

    result = read_image(
        path,
        ImageSpec(5, 4, "UnsignedLSB2"),
        mmap=False,
    )

    assert result.dtype == np.dtype("<u2")
    assert np.array_equal(result, data)