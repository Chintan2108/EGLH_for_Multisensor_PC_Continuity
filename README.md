# EGLH for Multisensor PC Continuity

**A cross-calibrated phycocyanin index for Landsat and ocean color sensors for long-term monitoring of cyanobacteria in inland waterbodies**

[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)](#citation)
[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code and shareable data to reproduce the analyses, tables and figures of **"A Cross-calibrated Phycocyanin Index for Landsat and Ocean Color Sensors for Long-Term Monitoring of Cyanobacteria in Inland Waterbodies"** (Kumar, Maniyar and Mishra, *Harmful Algae*).

## 📄 Abstract
Cyanobacterial harmful algal blooms (CyanoHABs) need synoptic monitoring at high revisit frequency and fine spatial resolution, a combination that no single satellite provides. This study harmonizes Sentinel-3 OLCI, Landsat 8/9 OLI/OLI-2 and Envisat MERIS observations with phycocyanin-sensitive indices built from true and virtual orange-band (620 nm) configurations. Using 770 in situ Rrs spectra (328 with paired phycocyanin), we evaluate common-band consistency, six cyanobacteria indices, matchup performance and sensor-to-sensor transferability. OLCI and MERIS are near-identical; OLI needs an empirical transfer. Enhanced Green Line Height (EGLH) and Green Line Height give the lowest errors under blocked cross-validation in western Lake Erie, an OLI-to-OLCI transfer is derived from one near-coincident overpass, and an independent 2024–2025 multi-lake test shows that EGLH agrees between satellite and in situ data and between sensors, while absolute PC outside Lake Erie needs regional calibration.

## 🗂️ Repository layout

```
notebooks/   10 notebooks, run in order (each is self-contained)
src/         eglh_utils.py: indices, PC models, metrics, bootstrap, blocked CV, plotting helpers
data/        raw RSRs, processed tables, one example scene, and the (empty) embargoed folder
acolite/     ACOLITE settings used for the Landsat example
results/     figures and tables written by the notebooks
EQUATIONS.md quick reference of all equations and coefficients
```

| Notebook | Reproduces |
|---|---|
| `01_sensor_band_simulation` | Figure 2, band simulation with RSRs, check against the open GLORIA spectra |
| `02_cross_sensor_consistency` | Figures 3–4, Table S3 |
| `03_olci_matchup_validation` | Figures 5–6 |
| `04_pc_models_calibration_validation` | Table 2, Table S4–S7, Figure S3 |
| `05_blocked_cross_validation` | Table 3, Tables S8–S9, Figure S4 |
| `06_eglh_valid_domain_and_uncertainty` | Tables S10–S13, Figure S5 |
| `07_oli_olci_cross_calibration` | Eq. 3, Table 4, Tables S15–S17, Figure S6 |
| `08_independent_validation_2024_2025` | Table 5, Tables S14, S18–S21, Figure 9 (needs embargoed data) |
| `09_dataset_differences_and_recalibration` | Table S3 extended, Tables S22–S25, Figures S7–S8 (needs embargoed data) |
| `10_apply_eglh_to_acolite_scene` | applying EGLH → PC to an ACOLITE-corrected Landsat 8 scene |

## 🚀 Getting Started

### Prerequisites
The analysis was performed with Python 3.10. Create the environment and start Jupyter:

```
conda env create -f environment.yml
conda activate eglh
jupyter lab
```

Open the notebooks in `notebooks/` and run them from top to bottom. Every notebook starts with a short description of its inputs and outputs; the shared helper functions are in `src/eglh_utils.py`.
Results are written to `results/figures/` and `results/tables/`.

### Data Availability
- **Processed data** needed for notebooks 01–07 and 10 are included in `data/` (see `data/README.md` for the data dictionary).
- **Open spectra:** notebook 01 downloads the GLORIA dataset (Lehmann et al., 2023, CC-BY 4.0) into `data/external/` to simulate sensor bands from hyperspectral Rrs.
- **Satellite data:** Sentinel-3 OLCI and Landsat scenes are available from the [Copernicus Data Space](https://dataspace.copernicus.eu/) and [USGS EarthExplorer](https://earthexplorer.usgs.gov/); atmospheric correction used [ACOLITE](https://github.com/acolite/acolite) (settings in `acolite/`). Spectral response functions are from the [NASA Ocean Biology RSR tables](https://oceandata.sci.gsfc.nasa.gov/rsr_tables/).
- **2024–2025 field data (embargoed):** the in situ phycocyanin, radiometry and OLCI matchups from eight lakes are under a NASA and University of Georgia embargo and are not part of this repository. They are planned for release on [NASA SeaBASS](https://seabass.gsfc.nasa.gov/) when the project ends. Notebooks 08 and 09 run as soon as the three files described in `data/embargoed/README.md` are placed in `data/embargoed/`; until then they stop at the first cell that needs them. Collaborators can request the files from the authors.

## 🔗 Citation
If you use this code or data in your research, please cite the paper:
```
@article{Kumar2026_EGLH,
  title   = {A Cross-calibrated Phycocyanin Index for Landsat and Ocean Color Sensors for Long-Term Monitoring of Cyanobacteria in Inland Waterbodies},
  author  = {Kumar, Abhishek and Maniyar, Chintan B. and Mishra, Deepak R.},
  journal = {Harmful Algae},
  year    = {2026}
}
```
Update the volume, pages and DOI (and the DOI badge above) once the paper is published.

---

*Center for Geospatial Research, Department of Geography, University of Georgia*
