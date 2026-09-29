# Orbit X
## Multi-Modal Correspondence & Registration Engine for Chandrayaan-2 Imagery

**SIH26166 · Smart India Hackathon 2026**

Orbit X is a multi-modal correspondence and image registration engine for Chandrayaan-2 optical imagery. It is designed to find reliable correspondence between a **source (moving)** image and a **reference (fixed)** image under differences in scale, geometry, viewpoint and illumination, and to produce a quantitatively evaluated registered result.

The current implementation is a **working Python desktop prototype** with a PySide6 workstation, a real-data workflow using Chandrayaan-2 **TMC-2 PDS4 products**, and a separate controlled known-transform validation workflow.

---

## What the System Does

```text
PDS4 Ingestion → Preprocessing → SIFT Features
       → Ratio + Mutual Matching → RANSAC Verification
       → 8×8 Spatial Selection → Sub-pixel Refinement
       → Registration → Quantitative Evaluation
```

The system does not treat image matching as a simple descriptor-similarity problem. Candidate correspondences are filtered, geometrically verified, spatially distributed, refined and then used for registration.

---

## SIH26166 Problem

SIH26166 requires a software solution for finding correspondence between Chandrayaan-2 optical imagery and lunar reference imagery while remaining robust to:

- Illumination / Sun-angle variation
- Viewpoint and geometric variation
- Scale variation
- False or visually plausible correspondences
- Uneven spatial distribution of matches

The registration output is evaluated using measures such as **RMSE / residual error, inlier match count and inlier ratio**, while correspondence points should remain spatially distributed across the image.

Orbit X addresses these requirements through a modular correspondence and registration pipeline.

---

## Core Pipeline

### 1. PDS4 Ingestion & Validation

The system reads Chandrayaan-2 PDS4-labelled products and uses their associated metadata and acquisition context before correspondence processing.

Current real-data support is centered on **TMC-2** products. OHRC and IIRS are represented in the architecture and data reconnaissance scope for future multimodal validation.

### 2. Preprocessing

Input imagery is normalized and prepared for correspondence processing. The implementation supports working-window / memory-aware processing so large products do not necessarily need to be loaded as a single full-resolution array.

### 3. Feature Correspondence

The current baseline uses **SIFT scale-space features** followed by descriptor ratio filtering and mutual correspondence checks.

### 4. Geometric Verification

Candidate correspondences are evaluated using **RANSAC-based geometric estimation**, separating geometrically consistent inliers from rejected correspondences.

### 5. Spatial Correspondence Selection

Orbit X uses an explicit **8 × 8 spatial grid** to monitor correspondence coverage. This prevents the selected correspondence set from being dominated by one highly textured region.

### 6. Sub-pixel Refinement

Selected correspondences are locally refined for improved transformation recovery and residual measurement.

### 7. Registration & Evaluation

The verified and refined correspondences are used for image registration. The system exposes correspondence points, inliers, spatial coverage, registered imagery, registration differences/residuals, match count, inlier count, inlier ratio and residual diagnostics.

---

## Two Validation Modes

### Real Chandrayaan-2 Workflow

The real-data workflow operates on Chandrayaan-2 PDS4 imagery. The current demonstrated real-data path uses **TMC-2 NCA/NCF products**.

Because a real acquisition pair does not necessarily provide a known ground-truth transformation, real-data metrics are treated as diagnostic evidence rather than direct ground-truth error measurements.

### Controlled Validation Workflow

A separate controlled dataset uses a **known transformation** between source and reference imagery. This makes it possible to directly measure transformation-recovery error and validate the registration pipeline independently of uncertainties in real lunar imagery.

---

## Desktop Workstation

The project includes a **PySide6 desktop GUI** exposed through the `sih26166-gui` entry point. The workstation provides separate real-data and controlled-validation workflows together with visual diagnostics for correspondence, coverage, registration and residuals.

Main GUI files:

```text
src/python/sih26166/app/gui.py
src/python/sih26166/app/__main__.py
```

---

## Repository Structure

```text
Orbit-X-Multi-Modal-Correspondence-Registration-Engine-for-Chandrayaan-2-Imagery/
│
├── data                  
│   ├── raw/
│   │   ├── IIRS/
│   │   ├── OHRC/
│   │   └── TMC/
│   └── samples/                         # lightweight sample imagery
│
├── experiments/
│
├── src/
│   └── python/
│       └── sih26166/
│           ├── app/
│           ├── correspondence/
│           ├── evaluation/
│           ├── experiments/
│           ├── io/
│           ├── refinement/
│           ├── registration/
│           ├── spatial/
│           ├── verification/
│           ├── paths.py
│           ├── pipeline.py
│           └── preprocessing.py
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
├── .gitignore
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

Generated caches and intermediate data are excluded through `.gitignore`.

---

## Installation

### Requirements

- **Python >= 3.12**
- NumPy
- Pandas
- OpenCV
- Pillow
- PySide6
- shiboken6

The current `requirements.txt` contains the runtime dependency ranges.

### Environment

Using Conda:

```bash
conda create -n sih26166 python=3.12
conda activate sih26166
pip install -r requirements.txt
```

Or use a Python virtual environment:

```bash
python -m venv .venv
```

Activate it and run:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Running the Application

After editable installation:

```bash
sih26166-gui
```

Equivalent module entry point:

```bash
python -m sih26166.app
```

The package uses a `src/python` layout and the executable is declared by `pyproject.toml`.

---

## Running Tests

Run the complete test suite with:

```bash
pytest -q
```

The tests cover feature detection, correspondence matching, geometric verification, image reading, PDS4/TMC-2 loading, metrics, preprocessing, registration, spatial distribution, sub-pixel refinement, controlled benchmarking, GUI behavior and visualizations.

---

## Data

The repository contains lightweight samples and PDS4 metadata for development and demonstration.

The current real-data workflow is based on Chandrayaan-2 **TMC-2** products, with NCA and NCF products represented in the data layout. OHRC metadata/samples are also present as part of the broader multimodal scope.

Expected raw-data organization follows the sensor/date/product structure:

```text
data/raw/
├── OHRC/
│   └── <acquisition-date>/
├── TMC/
│   └── <acquisition-date>/
│       ├── NCA/
│       └── NCF/
└── IIRS/
```

Large raw mission image products (`.img`) may exceed GitHub's normal web-upload limit. These files should be handled separately through Git LFS or another suitable data-distribution mechanism; the corresponding PDS4 XML labels remain part of the data organization.

---

## Controlled Validation

The controlled validation experiment is stored under:

```text
experiments/controlled_final_verification/final_verification/
```

It contains:

```text
config.json
ground_truth.json
reference.png
source.png
```

The known transformation in `ground_truth.json` provides the basis for direct transformation-recovery evaluation.

---

## Design Highlights

### PDS4-aware processing

Orbit X works with real Chandrayaan-2 product structure and metadata rather than treating the problem as generic PNG/JPEG image matching.

### Geometrically verified correspondence

Descriptor similarity is not sufficient. Ratio filtering, mutual matching and RANSAC-based geometric verification are used before registration.

### Spatial distribution as a first-class objective

The 8×8 spatial selection stage prevents correspondence quality from being represented only by the total number of matches.

### Evidence-oriented registration

The system exposes intermediate correspondence, inlier, spatial-coverage, registration and residual information so the result can be inspected.

### Modular architecture

I/O, preprocessing, correspondence, verification, spatial selection, refinement, registration and evaluation are separated into independent modules, allowing individual methods to be improved without redesigning the whole pipeline.

---

## SIH26166 Requirement Mapping

| Requirement / challenge | Orbit X approach |
|---|---|
| Source/reference correspondence | Feature correspondence pipeline |
| Scale variation | SIFT scale-space features |
| Illumination variation | Image preprocessing + robust correspondence |
| Geometric/viewpoint variation | RANSAC-based geometric verification |
| False correspondences | Ratio + mutual matching + geometric filtering |
| Spatially distributed matches | 8×8 spatial correspondence selection |
| Sub-pixel requirement | Local sub-pixel refinement |
| Registered imagery | Registration module |
| Quantitative evaluation | Match, inlier, ratio and residual metrics |
| Real Chandrayaan-2 data | TMC-2 PDS4 workflow |
| Multimodal architecture | OHRC / TMC-2 / IIRS support path |
| Reproducibility | Controlled known-transform benchmark |
| Traceability | Evidence/provenance workflow |

---

## Current Status

### Implemented

- PDS4-aware ingestion
- Chandrayaan-2 TMC-2 real-data workflow
- Image preprocessing
- SIFT feature correspondence
- Ratio filtering
- Mutual correspondence checks
- RANSAC geometric verification
- 8×8 spatial correspondence selection
- Sub-pixel refinement
- Image registration
- Quantitative diagnostics
- Controlled known-transform validation
- PySide6 desktop workstation
- Correspondence and spatial-coverage visualization
- Registration/residual visualization
- Automated test suite
- Evidence/provenance handling

### Extension Scope

- Full cross-sensor OHRC ↔ TMC-2 ↔ IIRS correspondence validation
- Broader real-data validation across acquisition conditions
- Larger-scale production processing
- Tiled / coarse-to-fine processing for very large products
- Additional correspondence strategies beyond the current SIFT baseline
- Further robustness improvements for heterogeneous sensor imagery

SIFT is the current baseline and is not intended to be the only correspondence strategy.

---

## Interpreting Results

Orbit X separates **real-data demonstration** from **controlled quantitative validation**.

A real-data run demonstrates that the pipeline can ingest actual Chandrayaan-2 products and produce correspondence and registration diagnostics. The controlled benchmark is used when a known transformation is required to measure recovery error directly.

This distinction prevents diagnostic metrics from real mission data from being presented as ground-truth accuracy when an independent ground truth is unavailable.

---

## License

This project is released under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

## Project

**Orbit X**  
**Smart India Hackathon 2026 — SIH26166**  
**Multi-Modal Correspondence & Registration Engine for Chandrayaan-2 Imagery**
