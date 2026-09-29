# [ANNOTATION] Module docstring describing central path configuration for the project layout.
"""
Central path configuration for sih26166.

Other modules should import paths from here instead of hardcoding
relative paths, so the project layout is described in one place.
"""

# [ANNOTATION] Import Path class from pathlib for clean cross-platform path resolution.
from pathlib import Path

# [ANNOTATION] Print diagnostic log entry indicating central path resolution initialization.
print("[PATHS] Resolving central project directories...")

# [ANNOTATION] Determine project root directory dynamically by walking 3 parent levels up from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# [ANNOTATION] Define core dataset paths relative to project root directory.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

# [ANNOTATION] Define instrument-specific raw data paths.
OHRC_RAW_DIR = RAW_DIR / "OHRC"
TMC_RAW_DIR = RAW_DIR / "TMC"
IIRS_RAW_DIR = RAW_DIR / "IIRS"

# [ANNOTATION] Define PDS4 metadata paths.
METADATA_DIR = PROJECT_ROOT / "metadata"
PDS4_METADATA_DIR = METADATA_DIR / "pds4"

# [ANNOTATION] Define experiment, evaluation, and trained model directory paths.
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
MODELS_DIR = PROJECT_ROOT / "models"

# [ANNOTATION] Log project root path verification.
print(f"[PATHS] Project root resolved at: {PROJECT_ROOT}")