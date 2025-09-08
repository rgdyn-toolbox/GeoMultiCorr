# GeoMultiCorr
A framework for study earth-surface displacements from optical images.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17079420.svg)](https://doi.org/10.5281/zenodo.17079420)

# Contents
1. [GeoMultiCorr philosophy](#geomulticorr-philosophy)
2. [Developers team](#geomulticorr-developer-team)
3. [Requirements](#requirements)
4. [How to use](#how-to-use)
5. [Installation](#installation)
6. [Contact and citation](#contact-and-citation)

# _GeoMultiCorr_ philosophy

The GeoMultiCorr project arose from the need to be able to exploit heterogeneous optical remote sensing archives, derived from multiple sensors, to measure time series of ground deformation (i.e. surface displacement) in high mountains environments. This open-source python pipeline, allows any image database to be manipulated interactively, harmonizing and preparing the data to measure ground deformation via image correlation. For the latter, we use optimized software Ames Stereo Pipeline ([ASP](https://stereopipeline.readthedocs.io/en/latest/introduction.html)), developed by NASA, to estimate surface displacement.

In recent developments, we coupled time series inversion (TIO) ([Bontemps et al., 2018](https://doi.org/10.1016/j.rse.2018.02.023)) to produce filtered and cleaned displacement series. This robust methodology allowed me to further diversify the data potentially usable for monitoring mass movements in mountainous area, even with coarse resolution images (15 m), such as those from Landsat 4/5/7/8 satellites ([Cusicanqui et al., 2025](https://tc.copernicus.org/articles/19/2559/2025/)).

This project is freely available to the scientific community (License Apache 2.0) on GitHub via the Rock Glacier Dynamics Toolbox ([RGDyn toolbox](https://github.com/rgdyn-toolbox)) to make it open and collaborative. 

# _GeoMultiCorr_ developer team
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/cusicand">
        <img src="https://github.com/cusicand.png" width="60px;" /><br />
        <sub><b>Diego CUSICANQUI</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pyzak117">
        <img src="https://github.com/pyzak117.png" width="60px;" /><br />
        <sub><b>Thibaut DUVANEL</b></sub>
      </a>
    </td>
  </tr>
</table>

# Requirements
Although _GeoMultiCorr_ pipeline is written in python, it relies on Ames Stereo Pipeline ([ASP](https://stereopipeline.readthedocs.io/en/latest/introduction.html)) software, who only runs on Unix & Mac OS. So for the moment, it will only run on both platforms.

Python dependencies:
- python>=3.11
  - geopandas>=1.0.0
  - geoutils=0.1.17
  - numba=0.*
  - numpy>=1,<3
  - matplotlib=3.*
  - pyproj>=3.4,<4
  - rasterio>=1.3,<2
  - scipy=1.*
  - alive-progress
  - ipython
  - ipykernel
  - xarray
  - rioxarray
  - gdal
  - geocube
  - scikit-image
  - scikit-gstat
  - pytransform3d
  - pillow
  - affine
  - pandas
  - pyogrio
  - shapely
  - geemap
  - earthengine-api
  - geodatasets
  - contextily

# Installation

Most of the libraries required for this script are standard and are often pre-installed in conda Python environments. Please follow the instructions below based on your requirements.

If you already have a conda Python environment pre-installed, please follow the instructions in the section [Install packages on an existing conda Python environment](#install-packages-on-an-existing-python-environment). Otherwise, you will need to install a conda Python environment to use this script. Instructions are provided in the section [Install packages on a new conda Python environment](#install-packages-on-a-new-python-environment).

## Install packages on an existing python environment

Run the next command lines in your command-line prompt:

```bash
conda activate <your-env-name>
```
or 
```bash
conda install -c conda-forge geopandas>=1.0.0 geoutils=0.1.17 numba=0.* numpy>=1,<3 matplotlib=3.* pyproj>=3.4,<4 rasterio>=1.3,<2 scipy=1.* alive-progress
```

We encourage the use of `mamba` since this library as is faster than conda. 
If you want to use `mamba`, run the following lines:

```bash
conda activate <your-env-name>
```

```bash
mamba install -c conda-forge geopandas>=1.0.0 geoutils=0.1.17 numba=0.* numpy>=1,<3 matplotlib=3.* pyproj>=3.4,<4 rasterio>=1.3,<2 scipy=1.* alive-progress
```

## Install packages on a new python environment

If you want to create a specific python environment, please follow the instructions below.

### Python environment with Miniconda

Go to the [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) website and download the lastest version of Miniconda. Detailed instructions on how to install conda python environments for your operating system are available on the [Anaconda website](https://docs.anaconda.com/free/miniconda/).

Once conda installed, you can 

### Create the new environment using `mamba`

First, install `mamba`
```bash
conda install -n base -c conda-forge mamba 
```
then, install all packages using the `gmc_env.yml` file provided.
```bash
mamba env create -f gmc_env.yml
```
To active the environment, type `conda activate gmc_env`

### Create the new environment using `conda`

```bash
conda env create -f pdal_env.yml
```
To active the environment, type:

```bash
conda activate pdal_env
```

## Make python script executables

If you want to run the script anywhere in your computer from CLI, you need to add the following lines to your `.bashrc` file to have full access to scripts. 
Open your `.bashrc` file using vi `~/.bashrc` or `nano ~/.bashrc` and copy the following lines at the end.

```bash
export GEOMULTICORR_PATH=$HOME/geomulticorr
export PATH=$GEOMULTICORR_PATH:$PATH            
export PYTHONPATH=$GEOMULTICORR_PATH:$PYTHONPATH
```
> [!NOTE]   
> If your installation directory is different than `$HOME`, Replace `$HOME` by the full directory path.

Use `source ~/.bashrc` to reload changes.

# How to use

Documentation is under development. Please be patient . . .

# Contact and citation ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17079420.svg)
For any question/bug/issue regarding this tool, please report it on issues section or contact the developer team:
- [diego.cusicanqui@univ-grenoble-alpes.fr](mailto:diego.cusicanqui@univ-grenoble-alpes.fr).
- [thibaut.duvanel@unil.ch](mailto:thibaut.duvanel@unil.ch)

> [!IMPORTANT]   
> If you use this tool, please cite using the following [DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17079420.svg). This will allow some recognition of the time invested and open access to this tool.
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17079420.svg)