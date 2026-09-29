# [ANNOTATION] Unit tests for TMC-2 stereoscopic dataset discovery and PDS4 XML label parsing.
"""
Tests for TMC-2 and PDS4 data discovery, metadata validation, and loading.
"""

from __future__ import annotations

# [ANNOTATION] Import PyTest and TMC-2/PDS4 module functions.
import pytest
from sih26166.io.tmc2 import get_tmc2_20200804_pair, _resolve_tmc2_dirs
from sih26166.io.pds4 import read_pds4_label
from sih26166.paths import PROJECT_ROOT, TMC_RAW_DIR, DATA_DIR


# [ANNOTATION] Test resolving relative filesystem directories for NCA and NCF camera pairs.
def test_tmc2_dir_resolution():
    print("\n[TEST] Executing test_tmc2_dir_resolution...")
    nca_dir, ncf_dir = _resolve_tmc2_dirs(PROJECT_ROOT)
    assert nca_dir.is_dir()
    assert ncf_dir.is_dir()

    # Pass subfolder directly
    nca_dir2, ncf_dir2 = _resolve_tmc2_dirs(TMC_RAW_DIR / "20200804")
    assert nca_dir2 == nca_dir
    assert ncf_dir2 == ncf_dir


# [ANNOTATION] Test verifying TMC2Product object specs for 2020-08-04 acquisition.
def test_tmc2_product_metadata():
    print("\n[TEST] Executing test_tmc2_product_metadata...")
    nca, ncf = get_tmc2_20200804_pair(PROJECT_ROOT, validate_files=False)
    assert nca.name == "NCA"
    assert ncf.name == "NCF"
    assert nca.image_spec.width == 4000
    assert nca.image_spec.height == 168383
    assert nca.image_spec.dtype == "uint16"
    assert ncf.image_spec.width == 4000
    assert ncf.image_spec.height == 168385


# [ANNOTATION] Test parsing PDS4 XML labels to verify extracted metadata fields (width, height, data_type).
def test_pds4_metadata_parsing():
    print("\n[TEST] Executing test_pds4_metadata_parsing...")
    nca, ncf = get_tmc2_20200804_pair(PROJECT_ROOT, validate_files=False)
    assert nca.label_path.is_file()
    assert ncf.label_path.is_file()

    nca_meta = read_pds4_label(nca.label_path)
    print(f"[TEST] Parsed NCA label: width={nca_meta.width}, height={nca_meta.height}, dtype={nca_meta.data_type}")
    assert nca_meta.width == 4000
    assert nca_meta.height == 168383
    assert nca_meta.data_type == "UnsignedLSB2"
    assert nca_meta.bit_depth == 16
    assert nca_meta.product_id is not None
    assert "ch2_tmc_nca" in nca_meta.product_id.lower()

    ncf_meta = read_pds4_label(ncf.label_path)
    print(f"[TEST] Parsed NCF label: width={ncf_meta.width}, height={ncf_meta.height}, dtype={ncf_meta.data_type}")
    assert ncf_meta.width == 4000
    assert ncf_meta.height == 168385
    assert ncf_meta.data_type == "UnsignedLSB2"
    assert ncf_meta.bit_depth == 16
    assert ncf_meta.product_id is not None
    assert "ch2_tmc_ncf" in ncf_meta.product_id.lower()