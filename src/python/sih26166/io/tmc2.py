# [ANNOTATION] Module docstring describing Chandrayaan-2 TMC-2 stereoscopic pair dataset resolution.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclasses and Path helpers.
from dataclasses import dataclass
from pathlib import Path

# [ANNOTATION] Import ImageSpec dataclass from local image_reader module.
from .image_reader import ImageSpec


# [ANNOTATION] Dataclass representing a TMC-2 product entity containing image/label paths and ImageSpec details.
@dataclass(frozen=True)
class TMC2Product:
    name: str
    image_path: Path
    label_path: Path
    image_spec: ImageSpec


# [ANNOTATION] Internal helper function locating relative dataset directories for NCA/NCF camera pairs.
def _resolve_tmc2_dirs(root: Path | str | None = None) -> tuple[Path, Path]:
    if root is None:
        from ..paths import PROJECT_ROOT
        root = PROJECT_ROOT
    root = Path(root)

    print(f"[TMC2] Resolving TMC-2 dataset paths relative to root: {root}")

    # 1. Direct subdirectories (e.g. root is data/raw/TMC/20200804)
    if (root / "NCA").is_dir() and (root / "NCF").is_dir():
        return root / "NCA", root / "NCF"

    # 2. Under data/raw/TMC/20200804 (e.g. root is PROJECT_ROOT)
    if (root / "data" / "raw" / "TMC" / "20200804" / "NCA").is_dir():
        return (
            root / "data" / "raw" / "TMC" / "20200804" / "NCA",
            root / "data" / "raw" / "TMC" / "20200804" / "NCF",
        )

    # 3. Under raw/TMC/20200804 (e.g. root is DATA_DIR)
    if (root / "raw" / "TMC" / "20200804" / "NCA").is_dir():
        return (
            root / "raw" / "TMC" / "20200804" / "NCA",
            root / "raw" / "TMC" / "20200804" / "NCF",
        )

    # 4. Under 20200804 (e.g. root is TMC_RAW_DIR)
    if (root / "20200804" / "NCA").is_dir():
        return (
            root / "20200804" / "NCA",
            root / "20200804" / "NCF",
        )

    return (
        root / "data" / "raw" / "TMC" / "20200804" / "NCA",
        root / "data" / "raw" / "TMC" / "20200804" / "NCF",
    )


# [ANNOTATION] Primary function returning TMC2Product objects for 2020-08-04 acquisition.
def get_tmc2_20200804_pair(
    root: Path | str | None = None,
    validate_files: bool = True,
) -> tuple[TMC2Product, TMC2Product]:
    """Return TMC2Product descriptors for the 2020-08-04 NCA / NCF stereo pair.

    Parameters
    ----------
    root:
        Project root or TMC dataset root directory. If None, uses paths.PROJECT_ROOT.
    validate_files:
        If True, validates that both .img and .xml files exist on disk.
    """
    nca_dir, ncf_dir = _resolve_tmc2_dirs(root)

    print("[TMC2] Instantiating TMC2Product descriptors for 2020-08-04 NCA/NCF pair...")

    nca = TMC2Product(
        name="NCA",
        image_path=nca_dir / "ch2_tmc_nca_20200804T1310425663_d_img_d18.img",
        label_path=nca_dir / "ch2_tmc_nca_20200804T1310425663_d_img_d18.xml",
        image_spec=ImageSpec(
            width=4000,
            height=168383,
            dtype="uint16",
        ),
    )

    ncf = TMC2Product(
        name="NCF",
        image_path=ncf_dir / "ch2_tmc_ncf_20200804T1310425663_d_img_d18.img",
        label_path=ncf_dir / "ch2_tmc_ncf_20200804T1310425663_d_img_d18.xml",
        image_spec=ImageSpec(
            width=4000,
            height=168385,
            dtype="uint16",
        ),
    )

    if validate_files:
        print("[TMC2] Validating file existence on disk for NCA and NCF products...")
        for product in (nca, ncf):
            if not product.image_path.is_file():
                raise FileNotFoundError(
                    f"Missing TMC-2 image: {product.image_path}"
                )

            if not product.label_path.is_file():
                raise FileNotFoundError(
                    f"Missing TMC-2 PDS4 label: {product.label_path}"
                )

    return nca, ncf


# [ANNOTATION] Wrapper function returning strictly validated TMC-2 product pair.
def validated_tmc2_20200804_pair(
    root: Path | str | None = None,
) -> tuple[TMC2Product, TMC2Product]:
    """Return validated TMC-2 product pair, raising FileNotFoundError if files are missing."""
    return get_tmc2_20200804_pair(root, validate_files=True)