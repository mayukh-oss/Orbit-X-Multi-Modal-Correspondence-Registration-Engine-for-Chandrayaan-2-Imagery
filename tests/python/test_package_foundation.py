# [ANNOTATION] Sanity tests for core package foundation and directory structure verification.
"""
Sanity tests for the sih26166 package foundation: packaging,
importability, and path configuration.

These do not test any correspondence/registration logic - there
isn't any yet. They only prove the project skeleton (Files #1-3)
is sound.
"""

# [ANNOTATION] Import main package and paths module.
import sih26166
from sih26166 import paths


# [ANNOTATION] Test verifying package version string match ("0.1.0").
def test_package_version():
    print("\n[TEST] Executing test_package_version...")
    print(f"[TEST] Installed sih26166 version: {sih26166.__version__}")
    assert sih26166.__version__ == "0.1.0"


# [ANNOTATION] Test verifying project root directory exists on disk.
def test_project_root_exists():
    print("\n[TEST] Executing test_project_root_exists...")
    print(f"[TEST] Project root path: {paths.PROJECT_ROOT}")
    assert paths.PROJECT_ROOT.exists()
    assert paths.PROJECT_ROOT.is_dir()


# [ANNOTATION] Test verifying required data directories exist.
def test_data_directories_exist():
    print("\n[TEST] Executing test_data_directories_exist...")
    for directory in [
        paths.DATA_DIR,
        paths.RAW_DIR,
        paths.INTERIM_DIR,
        paths.PROCESSED_DIR,
        paths.SAMPLES_DIR,
        paths.OHRC_RAW_DIR,
        paths.TMC_RAW_DIR,
        paths.IIRS_RAW_DIR,
    ]:
        print(f"[TEST] Verifying data directory: {directory.name}")
        assert directory.exists(), f"{directory} does not exist"
        assert directory.is_dir(), f"{directory} is not a directory"


# [ANNOTATION] Test verifying project configuration and output directories exist.
def test_project_directories_exist():
    print("\n[TEST] Executing test_project_directories_exist...")
    for directory in [
        paths.METADATA_DIR,
        paths.PDS4_METADATA_DIR,
        paths.EXPERIMENTS_DIR,
        paths.EVALUATION_DIR,
        paths.MODELS_DIR,
    ]:
        print(f"[TEST] Verifying project directory: {directory.name}")
        assert directory.exists(), f"{directory} does not exist"
        assert directory.is_dir(), f"{directory} is not a directory"