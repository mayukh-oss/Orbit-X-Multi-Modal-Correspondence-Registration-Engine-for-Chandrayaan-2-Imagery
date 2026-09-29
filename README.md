# Orbit X — Multi-Modal Correspondence & Registration Engine for Chandrayaan-2 Imagery

**SIH Problem Statement: SIH26166**

Orbit X is a Python-based correspondence and image-registration engine for Chandrayaan-2 optical imagery. It is designed to establish reliable correspondence between a **source (moving) image** and a **reference (fixed) image** despite differences in scale, geometry, viewpoint and illumination.

The current prototype works with real **Chandrayaan-2 TMC-2 PDS4 products** and also provides a controlled known-transform validation workflow. The architecture is intended to support heterogeneous Chandrayaan-2 optical sources including **OHRC, TMC-2 and IIRS**.

> **Current status:** Working desktop software prototype with real-data demonstration and controlled validation. Full cross-sensor OHRC ↔ TMC-2/IIRS multimodal validation remains an extension of the current experimental scope.

---

## Overview

SIH26166 requires a software solution that finds correspondence between Chandrayaan-2 optical imagery and lunar reference imagery, producing a registered result while maintaining spatially distributed matches.

The project therefore focuses on more than simple visual image matching. The pipeline combines:

- PDS4-aware data ingestion
- Image preprocessing and normalization
- SIFT-based feature detection
- Descriptor ratio filtering
- Mutual correspondence checks
- RANSAC-based geometric verification
- Spatially distributed correspondence selection using an **8×8 grid**
- Local sub-pixel refinement
- Image registration
- Residual and correspondence metrics
- Visualization and evidence/provenance
- Real-data and controlled-validation workflows

The architecture is modular so that the current SIFT baseline or individual processing stages can be improved or replaced without redesigning the complete registration framework.

---

## Pipeline

```text
                 Chandrayaan-2 / Reference Imagery
                              │
                              ▼
                    ┌───────────────────┐
                    │  INGEST & VALIDATE │
                    │ PDS4 + metadata    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │      PREPARE      │
                    │ Normalize / window │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     CORRESPOND     │
                    │ SIFT + ratio +     │
                    │ mutual matching    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ GEOMETRIC VERIFY  │
                    │ RANSAC + inliers   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ SPATIAL SELECTION │
                    │      8 × 8 grid   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ SUB-PIXEL REFIN.  │
                    │ Local refinement  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ REGISTER & EVAL.  │
                    │ Image + residuals │
                    │ + quantitative     │
                    │ diagnostics        │
                    └───────────────────┘
```

---

## Key Design Goals

### 1. Geometrically reliable correspondence

Descriptor similarity alone is not treated as sufficient. Candidate matches are filtered and then evaluated using geometric consistency and RANSAC-based inlier estimation.

### 2. Spatially distributed matches

The system explicitly monitors correspondence coverage through an **8×8 spatial grid**. This prevents the final correspondence set from being dominated by a single highly textured region.

### 3. Sub-pixel refinement

After robust correspondence selection, local refinement is used to improve transformation recovery and enable quantitative residual measurement.

### 4. Evidence-oriented processing

Intermediate results are exposed rather than hidden behind a black-box alignment step. The application can show correspondence, inliers, spatial coverage, registration output and residual information.

### 5. Real-data + controlled validation

The repository separates:

- **Real Chandrayaan-2 validation**, where the available ground truth transformation is not necessarily known.
- **Controlled known-transform validation**, where transformation-recovery error can be measured directly.

This distinction is important when interpreting registration metrics.

---

## Current Data Scope

### Real Chandrayaan-2 data

The current real-data demonstration uses **TMC-2 PDS4 imagery**.

The repository contains PDS4 XML metadata for selected Chandrayaan-2 products and sample TMC-2 imagery used by the application and validation workflows.

OHRC products have also been inspected during data reconnaissance. Full OHRC ↔ TMC-2 / IIRS cross-sensor correspondence validation remains part of the intended multimodal extension.

### Controlled benchmark

A separate controlled benchmark uses a known transformation between a source and reference image. This provides a ground-truth basis for measuring transformation-recovery error independently of the uncertainties of real lunar imagery.

---

## Quantitative Evaluation

The project tracks metrics relevant to SIH26166, including:

- Match count
- Inlier count
- Inlier ratio
- Spatial correspondence coverage
- Transformation residuals
- Mean residual error
- Maximum residual error

For controlled validation, the known transformation provides a direct reference for assessing recovery accuracy.

For real Chandrayaan-2 imagery, the diagnostics are used to characterize correspondence and registration quality rather than being interpreted as a direct substitute for known ground truth.

---

## Desktop Application

The project includes a **PySide6 desktop application** with separate workflows for:

### Real Chandrayaan-2 Workflow

Processes available real TMC-2 imagery through the correspondence and registration pipeline and provides visual and quantitative diagnostics.

### Controlled Validation Workflow

Runs the same core processing architecture against a controlled known-transform dataset, allowing transformation-recovery performance to be measured directly.

### Visualization

The workstation provides views for:

- Source/reference imagery
- Registered imagery
- Registration difference/residual visualization
- Correspondence inspection
- Spatial coverage
- Pipeline stage status
- Quantitative metrics
- Evidence/provenance information

---

## Repository Structure

```text
SIH 2026/
│
├── src/
│   └── python/
│       └── sih26166/
│           ├── app/
│           │   ├── __main__.py
│           │   └── gui.py
│           │
│           ├── correspondence/
│           │   ├── features.py
│           │   ├── matching.py
│           │   └── types.py
│           │
│           ├── evaluation/
│           │   └── metrics.py
│           │
│           ├── experiments/
│           │   ├── benchmark.py
│           │   └── controlled.py
│           │
│           ├── io/
│           │   ├── extract.py
│           │   ├── image_reader.py
│           │   ├── pds4.py
│           │   ├── quality.py
│           │   ├── raw_image.py
│           │   └── tmc2.py
│           │
│           ├── refinement/
│           │   └── subpixel.py
│           │
│           ├── registration/
│           │   └── register.py
│           │
│           ├── spatial/
│           │   └── distribution.py
│           │
│           ├── verification/
│           │   └── geometric.py
│           │
│           ├── paths.py
│           ├── pipeline.py
│           └── preprocessing.py
│
├── data/
│   ├── raw/
│   │   ├── OHRC/
│   │   └── TMC/
│   ├── samples/
│   │   ├── controlled_reference.png
│   │   ├── real_ohrc_*.png
│       └── real_tmc2_*.png
│
├── experiments/
│   └── controlled_final_verification/
│
├── tests/
│   └── python/
│       ├── test_benchmark.py
│       ├── test_correspondence_features.py
│       ├── test_correspondence_matching.py
│       ├── test_correspondence_types.py
│       ├── test_geometric.py
│       ├── test_gui.py
│       ├── test_image_reader.py
│       ├── test_metrics.py
│       ├── test_package_foundation.py
│       ├── test_pipeline.py
│       ├── test_preprocessing.py
│       ├── test_registration.py
│       ├── test_spatial_distribution.py
│       ├── test_subpixel.py
│       ├── test_tmc2_loading.py
│       └── test_visualizations.py
│
├── pyproject.toml
├── requirements.txt
├── .gitignore
```

---

## Requirements

- **Python 3.12+**
- NumPy
- Pandas
- OpenCV
- Pillow
- PySide6
- shiboken6

Development/testing additionally uses:

- pytest
- matplotlib
- ruff
- jupyter

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

For an editable development installation using the project's packaging configuration:

```bash
pip install -e .
```

---

## Running the Application

From the repository root:

```bash
python -m sih26166.app
```

Alternatively, after installing the package:

```bash
sih26166-gui
```

The repository also contains `execute.txt` with the module-based launch command.

If running directly from a source checkout without an editable installation, ensure the `src/python` directory is on `PYTHONPATH`.

---

## Running Tests

Run the complete test suite with:

```bash
pytest -q
```

The repository contains dedicated tests covering:

- Package foundation
- Image reading
- PDS4/TMC-2 loading
- Preprocessing
- Feature detection
- Correspondence matching
- Correspondence data types
- Geometric verification
- Spatial distribution
- Sub-pixel refinement
- Registration
- Metrics
- End-to-end pipeline
- Controlled benchmarking
- GUI behavior
- Visualization behavior

---

## Technical Architecture

The main processing pipeline is implemented as independent stages:

```text
detect_features()
        │
        ▼
match_features()
        │
        ▼
verify_geometry()
        │
        ▼
select_spatially_distributed()
        │
        ▼
refine_correspondences()
        │
        ▼
register_image()
        │
        ▼
transformation / spatial / residual metrics
```

The end-to-end pipeline exposes intermediate results through a structured `PipelineResult`, allowing the application and evaluation layer to inspect each stage.

The geometric model is explicitly configurable rather than assuming one model is universally correct.

---

## Why the 8×8 Spatial Grid Matters

A large number of matches does not automatically imply a reliable registration.

If most correspondences are concentrated in one textured region, the estimated transformation may be poorly constrained across the rest of the image.

Orbit X therefore treats spatial distribution as a first-class objective:

```text
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ • │   │ • │   │   │ • │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │ • │   │ • │   │   │ • │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ • │   │   │   │ • │   │   │ • │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │ • │ • │   │   │ • │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ ... spatial correspondence coverage ...             │
└───┴───┴───┴───┴───┴───┴───┴───┘
```

The implementation uses this coverage constraint to avoid selecting a final correspondence set that is dominated by one local region.

---

## Validation Philosophy

Orbit X distinguishes between **engineering validation** and **scientific interpretation**.

### Controlled validation

Known transformations make it possible to directly compare the recovered transformation against ground truth.

### Real-data validation

Real Chandrayaan-2 products provide evidence that the complete ingestion → correspondence → verification → registration workflow operates on actual mission data. Without a known ground-truth transformation for every real acquisition pair, the resulting residual and inlier metrics are treated as diagnostic evidence rather than absolute truth.

The project therefore exposes intermediate evidence and provenance instead of presenting registration as an unexplained black-box result.

---

## Current Limitations & Future Scope

The current implementation is intentionally positioned as a working engineering prototype.

Planned/ongoing areas include:

- Full OHRC ↔ TMC-2 ↔ IIRS cross-sensor correspondence validation
- Larger-scale production workflows
- Tiled and coarse-to-fine processing for larger products
- Further improvement of cross-modal correspondence robustness
- Additional feature/correspondence strategies beyond the current SIFT baseline
- Expanded real-data validation across acquisition conditions
- Continued strengthening of quantitative evidence and reproducibility

SIFT is currently used as the baseline correspondence method and is not presented as the final or exclusive solution.

---

## SIH26166 Alignment

| SIH26166 requirement | Orbit X implementation |
|---|---|
| Source/reference correspondence | Feature detection + descriptor matching |
| Illumination variation | Preprocessing and robust correspondence/verification |
| Scale variation | SIFT scale-space features |
| Geometric/viewpoint variation | RANSAC-based geometric verification |
| False correspondence rejection | Ratio + mutual matching + geometric inlier filtering |
| Spatially distributed matches | Explicit 8×8 spatial selection |
| Sub-pixel accuracy | Local sub-pixel refinement |
| Registered output | Configurable image registration |
| Quantitative evaluation | Match, inlier, ratio, coverage and residual metrics |
| Real Chandrayaan-2 data | TMC-2 PDS4 workflow |
| Multimodal architecture | OHRC, TMC-2 and IIRS extension path |
| Reproducibility/evidence | Controlled benchmark + diagnostic/provenance outputs |

---

## Project Status

**Working engineering prototype**

Current implementation includes:

- Real PDS4 data ingestion
- TMC-2 real-data workflow
- Preprocessing
- SIFT correspondence
- Geometric verification
- 8×8 spatial correspondence selection
- Sub-pixel refinement
- Registration
- Quantitative diagnostics
- Controlled known-transform validation
- Desktop visualization
- Evidence/provenance handling
- Automated software tests

The current real-data demonstration has been performed using verified Chandrayaan-2 TMC-2 PDS4 products. OHRC and IIRS remain part of the intended multimodal extension and validation scope.

---

## Team

**Orbit X**

**Smart India Hackathon 2026 — SIH26166**

> Multi-Modal Correspondence & Registration Engine for Chandrayaan-2 Imagery

