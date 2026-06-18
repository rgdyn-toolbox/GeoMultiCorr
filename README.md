<div align="center">
  <img src="gmc_logo/GMC_logo.svg" width="320px" alt="GeoMultiCorr logo"/>
</div>

# GeoMultiCorr

A Python framework for measuring earth-surface displacements from multi-sensor optical remote sensing archives.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17079420.svg)](https://doi.org/10.5281/zenodo.17079420)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE.md)

---

# Contents
1. [GeoMultiCorr philosophy](#geomulticorr-philosophy)
2. [Processing pipeline](#processing-pipeline)
3. [Developer team](#developer-team)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Documentation](#documentation)
7. [Contact and citation](#contact-and-citation)

---

# GeoMultiCorr philosophy

By the end of my PhD, we had accumulated a remarkably diverse optical remote sensing archive — images spanning multiple sensors, spatial resolutions, and decades. Yet exploiting all of that data together remained challenging. The question that motivated this project is simple: *why go to the trouble of combining heterogeneous datasets?* The answer is equally clear — combining sensors allows you to build more robust displacement time series, extend the record further back in time using older archives (Landsat 4/5/7), and sharpen uncertainty estimates by integrating newer, higher-resolution sensors alongside the coarser ones. We accepted the challenge of making multi-sensor image pairs work together consistently, and **GeoMultiCorr** is the result.

GeoMultiCorr is an open-source Python pipeline designed to exploit heterogeneous optical remote sensing archives — Landsat, Sentinel-2, SPOT, Pléiades, PlanetScope, WorldView, and more — to measure time series of ground deformation in high-mountain environments. The pipeline interactively manages any image database, harmonises and prepares the data, and measures surface displacement via image correlation using NASA's Ames Stereo Pipeline ([ASP](https://stereopipeline.readthedocs.io/en/latest/introduction.html)). Displacement time series are then produced through time-series inversion (TIO) ([Bontemps et al., 2018](https://doi.org/10.1016/j.rse.2018.02.023)), a robust methodology that works even with coarse-resolution imagery (15 m) such as Landsat 4/5/7/8 ([Cusicanqui et al., 2025](https://tc.copernicus.org/articles/19/2559/2025/)).

The project is freely available to the scientific community (GNU AGPL v3) via the Rock Glacier Dynamics Toolbox ([RGDyn toolbox](https://github.com/rgdyn-toolbox)).

## Matrioshka object hierarchy

GeoMultiCorr organises a project as a set of nested spatial objects — like Russian matrioshka dolls, each layer contains and operates on the one inside it:

```
Session  (project root — single entry point)
└── Pzone  (processing zone — geographic area of interest)
    ├── Thumb    — single optical image (date + sensor + file)
    ├── Pair     — two Thumbs combined for image correlation  [Thumb + Thumb]
    ├── Spine    — linear geomorphic feature; generates perpendicular ribs
    ├── Geomorph — geomorphological unit; aggregates spines and pairs
    └── Xzone    — exclusion zone; masks irrelevant areas
```

All spatial objects are stored as rows in a GeoPackage (`.gpkg`) geodatabase. `Session` is the single entry point — every object is accessed and created through it.

---

# Processing pipeline

GeoMultiCorr covers the full workflow from raw archive to calibrated displacement time series in three stages:

## 1 — Database & archive management

`Session`, `Pzone`, and `Thumb` provide interactive tools to ingest, organise, and filter multi-sensor imagery regardless of resolution or acquisition geometry. Sensor-specific pairing strategies (e.g., redundancy, sequential) are applied independently per sensor family to avoid resolution mismatches.

## 2 — Image correlation & corrections

Each `Pair` launches ASP's `parallel_stereo` to compute raw `EW` and `NS` displacement maps and correlation quality map (`CC`). The output then flows through a **modular, composable corrections pipeline**:

```
raw displacement maps (EW, NS, NMAD)
    ↓  OutlierFilter       — remove pixel-level outliers
    ↓  CCFilter            — apply correlation-coefficient quality threshold
    ↓  MedianCentering     — centre displacement around zero
    ↓  RampCorrection      — remove systematic orbital/atmospheric tilts
    ↓  TopoCorrection      — correct for topographic slope effects
    ↓  AlongTrackDestriping  — remove along-track scanner artifacts
    ↓  AcrossTrackDestriping — remove across-track scanner artifacts
    ↓  Spatial Masks       — SnowMask · CloudMask · SlopeMask · ShadowMask
    ↓
corrected displacement maps
```

Filters are optional and composable — apply only what your data requires.

## 3 — Time-series inversion (TIO)

`TIOInversion` prepares and launches the *invers_pixel_omp* Fortran solver: it exports corrected pairs to binary, writes the `liste_image`, `liste_couple` (with weighting), and `input_tio` files, generates cluster bash scripts, and post-processes binary outputs back to GeoTIFF.

---

# Developer team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/cusicand">
        <img src="https://github.com/cusicand.png" width="80px;" alt="Diego CUSICANQUI"/><br/>
        <b>Diego CUSICANQUI</b>
      </a><br/>
      <sub>Lead developer</sub><br/>
      <sub>ISTerre · Université Grenoble Alpes</sub>
    </td>
    <td align="center">
      <a href="https://github.com/duvanelt">
        <img src="https://github.com/duvanelt.png" width="80px;" alt="Thibaut DUVANEL"/><br/>
        <b>Thibaut DUVANEL</b>
      </a><br/>
      <sub>Developer</sub><br/>
      <sub>Université de Lausanne</sub>
    </td>
  </tr>
</table>

Contributions are welcome — feel free to open an issue or submit a pull request.

---

# Requirements

Although GeoMultiCorr is written in Python, it relies on the Ames Stereo Pipeline ([ASP](https://stereopipeline.readthedocs.io/en/latest/introduction.html)), which runs on **Linux and macOS only**. Windows is not supported.

Key Python dependencies:

| Package | Version |
|---|---|
| python | ≥ 3.11 |
| geoutils | = 0.2.5 |
| geopandas | ≥ 1.0.0 |
| rasterio | ≥ 1.3, < 2 |
| numpy | ≥ 1, < 3 |
| numba | 0.* |
| scipy | 1.* |
| matplotlib | 3.* |
| pyproj | ≥ 3.4, < 4 |
| xarray / rioxarray | latest |
| gdal / libgdal | latest |
| scikit-image | latest |
| scikit-gstat | latest |
| rasterstats | latest |
| tqdm | latest |

Full dependency list is in [`gmc_env.yml`](gmc_env.yml).

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/rgdyn-toolbox/GeoMultiCorr.git
cd GeoMultiCorr
```

## 2. Create the Python environment

**Recommended — micromamba** (fastest):

```bash
micromamba env create -f gmc_env.yml
micromamba activate gmc_env
```

**Alternative — conda / mamba:**

```bash
conda env create -f gmc_env.yml
conda activate gmc_env
```

## 3. Install GeoMultiCorr

```bash
pip install -e .
```

## 4. Install ASP (Ames Stereo Pipeline)

ASP must be installed separately (Linux / macOS only):

```bash
python install_ASP.py
```

> [!NOTE]
> Make sure ASP binaries are on your `PATH` after installation.

---

# Documentation

Documentation is under development. Please be patient . . .

In the meantime, explore the worked examples in [`notebooks/`](notebooks/):

| Notebook | Description |
|---|---|
| [GMC_SinglePair.ipynb](notebooks/GMC_SinglePair.ipynb) | Single-pair correlation: load a pair, run ASP, visualise raw results |
| [GMC_Corrections.ipynb](notebooks/GMC_Corrections.ipynb) | Apply the corrections pipeline; before/after comparison |
| [GMC_project.ipynb](notebooks/GMC_project.ipynb) | Full project workflow: initialise session → create pairs → invert time series |
| [GMC_ProjectContinuation.ipynb](notebooks/GMC_ProjectContinuation.ipynb) | Continue an existing project: load session, add new pairs, update results |
| [GMC_project_sensor_mixed.ipynb](notebooks/GMC_project_sensor_mixed.ipynb) | Multi-sensor heterogeneous archive workflow |

Launch with:

```bash
micromamba activate gmc_env
jupyter lab notebooks/
```

---

# Contact and citation ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17079420.svg)

For any question, bug, or issue regarding this tool, please report it in the [Issues](../../issues) section or contact the developer team:
- [diego.cusicanqui@univ-grenoble-alpes.fr](mailto:diego.cusicanqui@univ-grenoble-alpes.fr)
- [thibaut.duvanel@unil.ch](mailto:thibaut.duvanel@unil.ch)

> [!IMPORTANT]
> If you use this tool, please cite it using the following [DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17079420.svg). This will help recognise the time invested and keep the tool open and accessible.

![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17079420.svg)
