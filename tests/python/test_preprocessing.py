# [ANNOTATION] Enable modern type annotations and import testing primitives.
import numpy as np
import pytest

# [ANNOTATION] Import preprocessing module functions and configuration dataclass under test.
from sih26166.preprocessing import (
    PreprocessingConfig,
    percentile_normalize,
    preprocess_for_features,
)


# [ANNOTATION] Test verifying percentile_normalize maps uint16 numeric image arrays to uint8.
def test_percentile_normalize_returns_uint8():
    print("\n[TEST] Executing test_percentile_normalize_returns_uint8...")
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    result = percentile_normalize(image)
    print(f"[TEST] Output array dtype: {result.dtype}, shape: {result.shape}")
    assert result.dtype == np.uint8
    assert result.shape == image.shape


# [ANNOTATION] Test verifying preprocess_for_features with CLAHE preserves 2D image shape.
def test_preprocessing_preserves_shape():
    print("\n[TEST] Executing test_preprocessing_preserves_shape...")
    image = np.arange(400, dtype=np.uint16).reshape(20, 20)

    result = preprocess_for_features(
        image,
        config=PreprocessingConfig(local_contrast="CLAHE"),
    )

    print(f"[TEST] Preprocessed shape: {result.shape}")
    assert result.shape == image.shape
    assert result.dtype == np.uint8


# [ANNOTATION] Test verifying invalid percentile range (low >= high) raises ValueError.
def test_invalid_percentiles():
    print("\n[TEST] Executing test_invalid_percentiles...")
    with pytest.raises(ValueError):
        PreprocessingConfig(
            percentile_low=99,
            percentile_high=1,
        )