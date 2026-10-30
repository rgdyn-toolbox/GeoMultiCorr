#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# session.py
# creation date: 2026-05-07.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------- #
"""
Session management for GeoMultiCorr
====================================
This module defines the Session class, which encapsulates the data and operations related to a GeoMultiCorr project.
It provides methods to initialize a session, access project data such as processing zones, thumbnails, pairs, geomorphs, xzones, and spines, and interact with the associated QGIS project.
"""

from __future__ import annotations

import collections
import copy
import pathlib
import warnings
from collections import abc
from contextlib import ExitStack
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    overload,
)

import re
from shapely.geometry import Polygon, MultiPolygon
import tqdm
import shutil
import subprocess
from datetime import datetime

# Importing third-party libraries
import itertools
import numpy as np

# from osgeo import gdal
import geoutils as gu
import pandas as pd
import geopandas as gpd
import rasterio.crs as rio_crs
import rasterio.transform as rio_transform
import shapely.wkt as _wkt
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from alive_progress import alive_bar
from rich.console import Group as _RichGroup
from rich.live import Live as _RichLive
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text as _RichText

# Interactive tools
from ipyleaflet import Map, DrawControl
from IPython.display import display

# Importing GeoMultiCorr modules
import geomulticorr.core.geomorph as gmc_geomorph
import geomulticorr.core.pzone as gmc_pzone
import geomulticorr.core.pair as gmc_pair
import geomulticorr.core.spine as gmc_spine
import geomulticorr.core.thumb as gmc_thumb
import geomulticorr.core.xzone as gmc_xzone
import geomulticorr.stats.stats as gmc_stats
import geomulticorr.correlation.supported_sensors as gmc_sensors

from geomulticorr.core.console import (
    _rich_console,
    BatchProgress,
    print_register_thumbs_summary,
    print_sieve_bulk_summary,
    print_prepare_correlation_summary,
    print_launch_correlation_summary,
    print_sync_after_cluster_summary,
    print_reset_outputs_summary,
    print_extract_displacements_summary,
    print_apply_corrections_summary,
)

from geomulticorr._logging import logger
from geomulticorr._profiling import profile as gmc_profile
from geomulticorr.utils._image_types import (
    ELEVATION_DIRNAME,
    ELEVATION_IMAGE_TYPES,
    IMAGE_TYPE_OPTICAL,
    IMAGE_TYPES,
    normalise_image_type,
    pair_subdir,
    type_column,
)
from geomulticorr.utils._terrain import (
    DEFAULT_MD_AZIMUTHS,
    TERRAIN_PRODUCTS,
    shade_to_uint8,
    terrain_product,
)
from geomulticorr.utils._durations import (
    DT_UNIT_TO_DAYS,
    format_dt_days,
    parse_dt_days,
)
from geomulticorr._typing import DTypeLike, MArrayNum, NDArrayBool, NDArrayNum, Number

file_location = pathlib.Path(__file__)
package_root = file_location.parent.parent  # geomulticorr/core/ -> geomulticorr/

resources_location = package_root / "resources"

project_template_location = resources_location / "project_template"

# Pzone subdirectory kinds (new layout constants)
PZ_KIND_OPTICAL = "optical"
PZ_KIND_IMAGE_CORRELATION = "image_correlation"
PZ_KIND_REFERENCE_DEM = "reference_dem"
PZ_KIND_REFERENCE_RASTER = "reference_raster"
PZ_KIND_VECTOR = "vector"
PZ_KIND_INVERSION = "inversion"
PZ_KIND_FIGURES = "figures"
#: The similarity trace of ``compute_thumbs_similarity`` (JSON, one per type).
PZ_KIND_SIMILARITY = "similarity"
#: DEM-derived thumbs, one subfolder per image type (elevation/<type>/).
PZ_KIND_ELEVATION = ELEVATION_DIRNAME

NEW_LAYOUT_PZ_SUBDIRS = (
    PZ_KIND_OPTICAL,
    PZ_KIND_IMAGE_CORRELATION,
    PZ_KIND_REFERENCE_DEM,
    PZ_KIND_REFERENCE_RASTER,
    PZ_KIND_VECTOR,
    PZ_KIND_INVERSION,
    PZ_KIND_FIGURES,
    PZ_KIND_ELEVATION,
    PZ_KIND_SIMILARITY,
)

#: Identity columns of a ``compute_thumbs_similarity`` frame; the metric columns
#: (``similarity.raster.REPORT_KEYS``) follow them.  ``params_hash`` is what
#: lets a re-run reuse a stored row only when it was computed the same way.
SIMILARITY_ID_COLUMNS = (
    "pz_name",
    "image_type",
    "left_key",
    "right_key",
    "left_date",
    "right_date",
    "left_sensor",
    "right_sensor",
    "cross_sensor",
    "dt_days",
    "doy_diff",
    "params_hash",
)

#: Backwards-compatible aliases — the implementation now lives in
#: :mod:`geomulticorr.utils._durations` so ``pzone`` and the explorers can
#: share it without importing ``session`` (which would be circular).
_DT_UNIT_TO_DAYS = DT_UNIT_TO_DAYS
_parse_dt_days = parse_dt_days

# Rendering cost presets for the control figures written by
# ``apply_pairs_corrections``.  Correlation output is routinely 10+ Mpx while the
# largest panel renders at well under 1000 px, so "light" decimates for display,
# shrinks the hexbin grid and drops the dpi — same figures, ~9x cheaper.
# ``bbox_inches`` is deliberately absent: "tight" forces a second full renderer
# pass to measure the bounding box, and every figure already calls tight_layout().
PLOT_LEVELS: dict[str, dict] = {
    "light": {"preview_px": 1400, "dpi": 110, "hexbin_gridsize": 200},
    "full": {"preview_px": None, "dpi": 150, "hexbin_gridsize": 1000},
}

# Metric maps for syncing per-pair stats JSON into the Pairs geodatabase layer.
# Each entry is ``pairs_column: (stats_layer, stat_key)`` — see
# ``geomulticorr.stats.stats.resolve_stat_columns``. Column names are defined
# here so they live in a single place (the sync is a derived cache; these columns
# are intentionally NOT part of ``Pair.to_pdserie()``).
#: ASP input-encoding defaults per image type, used by
#: ``prepare_pairs_correlation`` when the caller passes ``None``.  Optical keeps
#: ASP's own defaults (no line written), so its settings files are unchanged.
#: Elevation products carry an explicit nodata (see ``sieve_bulk``'s encoding)
#: and are normalised per image: slope and shading ranges differ between dates.
CORR_INPUT_DEFAULTS: dict[str, dict[str, Any]] = {
    "opt": {
        "nodata_value": None,
        "individual_normalization": False,
        "pre_normalization": False,
    },
    "hillshade": {
        "nodata_value": 0,
        "individual_normalization": True,
        "pre_normalization": False,
    },
    "mdhillshade": {
        "nodata_value": 0,
        "individual_normalization": True,
        "pre_normalization": False,
    },
    "slope": {
        "nodata_value": -9999.0,
        "individual_normalization": True,
        "pre_normalization": False,
    },
    "dem": {
        "nodata_value": -9999.0,
        "individual_normalization": True,
        "pre_normalization": False,
    },
}


def _corr_input_params(pa_type: str, explicit: Mapping[str, Any]) -> dict:
    """Input-encoding settings for one pair: its type's defaults, explicit values winning."""
    params = dict(CORR_INPUT_DEFAULTS.get(pa_type or "opt", CORR_INPUT_DEFAULTS["opt"]))
    params.update({k: v for k, v in explicit.items() if v is not None})
    return params


def _asp_param_kwargs(effective: Mapping[str, Any]) -> dict:
    """Rename resolved parameters to ``build_correlation_params``' argument names.

    Only ``subpixel_mode`` actually differs — the ASP wrapper calls it
    ``subpixel_refinement_mode`` — but routing every key through one table means
    a future rename cannot silently drop a parameter at one of the two call
    sites.
    """
    kwargs = {k: v for k, v in dict(effective).items() if k != "subpixel_mode"}
    if "subpixel_mode" in effective:
        kwargs["subpixel_refinement_mode"] = effective["subpixel_mode"]
    return kwargs


RAW_STATS_SYNC: dict[str, tuple[str, str]] = {
    "pa_ew_nmad_raw": ("ew", "nmad"),
    "pa_ns_nmad_raw": ("ns", "nmad"),
    "pa_cc_quality050": ("cc", "cc_quality_gte_050"),
}
CORR_STATS_SYNC: dict[str, tuple[str, str]] = {
    "pa_ew_nmad_corr": ("ew", "nmad"),
    "pa_ns_nmad_corr": ("ns", "nmad"),
}


def is_conform_to_gmc_template(target_root_path: str | pathlib.Path) -> bool:
    """Check GeoMultiCorr structure (both new and legacy layouts).

    Checks that the target_root_path contains the two required project files:
    GMC_mapset_{name}.qgz and GMC_geodatabase_{name}.gpkg. Accepts both the new
    (pzone-direct) and legacy (raster_data_wrapper) layouts.

    Args:
        target_root_path (str | pathlib.Path): Path to check for GeoMultiCorr project.

    Returns:
        bool: True if both required files exist. False otherwise.
    """
    target_root_path = pathlib.Path(target_root_path)
    target_name = target_root_path.name

    # Check for the QGIS mapset file
    target_map_expected_path = pathlib.Path(
        target_root_path, f"GMC_mapset_{target_name}.qgz"
    )
    if not target_map_expected_path.exists():
        return False

    # Check for the geodatabase file
    target_geodb_expected_path = pathlib.Path(
        target_root_path, f"GMC_geodatabase_{target_name}.gpkg"
    )
    if not target_geodb_expected_path.exists():
        return False

    return True


# * Works properly
def re_searcher(string: str, pattern: str) -> str:
    """Search for a pattern in a string and return the first match.

    Args:
        string (str): The string to search in.
        pattern (str): The pattern to search for.

    Returns:
        str: The first match within the string.
    """
    try:
        return re.search(re.compile(pattern), string).group()
    except AttributeError:
        return "unknown"
    # END try


# END def


# * Works properly
def open_gmc_session(
    target_directory_path: str | pathlib.Path,
    epsg_code: str | int | None = None,
    empty_geodatabase: str | pathlib.Path = None,
) -> "Session":
    """
    Open or create a GeoMultiCorr (GMC) session at the specified directory.

    If the directory does not exist, it will be created and initialized with the GMC project structure.
    If the directory exists and is empty, it will be initialized as a new GMC project.
    If the directory exists and matches the GMC project structure, it will be opened as a session.
    If the directory exists but does not match the GMC structure, an error is raised.

    Args:
        target_directory_path (str | pathlib.Path): Path to the GMC project directory.
        epsg_code (str | int | None, optional): EPSG code for the project CRS. Defaults to 4326 if not provided.
        empty_geodatabase (str | pathlib.Path, optional): Path to an empty geodatabase template to use. If None, the default template is used.

    Returns:
        Session: An initialized Session object for the GMC project.

    Raises:
        FileNotFoundError: If the parent directory does not exist.
        ValueError: If the directory exists but does not conform to the GMC project structure.
    """
    if epsg_code is None:
        epsg_code = 4326
    # END if

    if not isinstance(target_directory_path, pathlib.Path):
        target_directory_path = pathlib.Path(target_directory_path)
    # END if
    return Session(target_directory_path, epsg_code, empty_geodatabase)


class Session:
    """
    Manipulate data specific to a sample of sites related to an earth surface displacement study.
    """

    # * It works properly
    def __init__(
        self,
        target_root_path: str | pathlib.Path,
        epsg_code: str | int | None = None,
        empty_geodatabase: bool | str | pathlib.Path = None,
    ):
        """Initialize a new Session.

        Args:
            target_root_path (str | pathlib.Path): The root path for the target data.
            epsg_code (str | int | None, optional): The EPSG code for the target data. Defaults to None.

        Raises:
            FileNotFoundError: If the target root path does not exist.
            ValueError: If the target root path is not a valid GeoMultiCorr project.
        """

        # pathlib.Path conversion of the string target_root_path
        target_root_path = pathlib.Path(target_root_path).resolve()

        # If EPSG code is not provided, EPSG:4326 is set by default.
        if epsg_code is None:
            epsg_code = 4326

        # Check the address validity
        if not target_root_path.parent.exists():
            msg = f"Invalid path: {target_root_path}"
            logger.folder(msg)
            raise FileNotFoundError(msg)
        # Check if the target root path exists and is conform to GMC template, if not create the GMC structure
        if target_root_path.exists():
            if not any(target_root_path.iterdir()):
                # Folder exists and is empty: create GMC structure
                project_name = target_root_path.name
                logger.launch(f"New GMC project: '{project_name}'")
                logger.success(f"Creating new GMC structure in: {target_root_path}")

                if empty_geodatabase:
                    # Copy GMC-geodatabase template
                    logger.file(f"Empty geodatabase template created")
                    raw_gdb_path = (
                        project_template_location / "GMC_geodatabase_empty.gpkg"
                    )
                    shutil.copy(raw_gdb_path, target_root_path)
                    new_gdb = target_root_path / "GMC_geodatabase_empty.gpkg"
                    new_gdb.rename(
                        target_root_path / f"GMC_geodatabase_{project_name}.gpkg"
                    )
                else:
                    logger.file(
                        f"Default GMC-geodatabase template copied into {target_root_path}"
                    )
                    raw_gdb_path = (
                        project_template_location / "GMC_geodatabase_project.gpkg"
                    )
                    shutil.copy(raw_gdb_path, target_root_path)
                    new_gdb = target_root_path / "GMC_geodatabase_project.gpkg"
                    new_gdb.rename(
                        target_root_path / f"GMC_geodatabase_{project_name}.gpkg"
                    )

                logger.info(f"Setting GMC folder structure")
            elif is_conform_to_gmc_template(target_root_path):
                logger.success("GMC project structure found")
                pass
            else:
                msg = f"Something else exists at {target_root_path}. Please clean it or choose another path."
                raise ValueError(msg)
        else:
            # Folder does not exist: create and initialize GMC structure
            project_name = target_root_path.name
            logger.success(f"New GMC project: '{project_name}'")
            target_root_path.mkdir(parents=True)
            logger.success(f"Creating new GMC structure in: {target_root_path}")

            if empty_geodatabase:
                # Copy GMC-geodatabase template
                logger.file(f"Empty geodatabase template created")
                raw_gdb_path = project_template_location / "GMC_geodatabase_empty.gpkg"
                shutil.copy(raw_gdb_path, target_root_path)
                new_gdb = target_root_path / "GMC_geodatabase_empty.gpkg"
                new_gdb.rename(
                    target_root_path / f"GMC_geodatabase_{project_name}.gpkg"
                )
            else:
                logger.file(
                    f"Default GMC-geodatabase template copied into {target_root_path}"
                )
                raw_gdb_path = (
                    project_template_location / "GMC_geodatabase_project.gpkg"
                )
                shutil.copy(raw_gdb_path, target_root_path)
                new_gdb = target_root_path / "GMC_geodatabase_project.gpkg"
                new_gdb.rename(
                    target_root_path / f"GMC_geodatabase_{project_name}.gpkg"
                )

            # Copy GMC-Mapset template
            logger.file(
                f"Making a copy of 'GMC-QGIS-Mapset' template into {target_root_path}"
            )
            raw_mapset_path = project_template_location / "GMC_mapset_project.qgz"
            shutil.copy(raw_mapset_path, target_root_path)
            new_mapset = target_root_path / "GMC_mapset_project.qgz"
            new_mapset.rename(target_root_path / f"GMC_mapset_{project_name}.qgz")

            logger.success(f"Setting GMC folder structure")
        # END if

        # Load project data into the current session — runs for all code paths
        self.path_root = target_root_path
        self.project_name = target_root_path.name
        self.path_geodb = target_root_path / f"GMC_geodatabase_{self.project_name}.gpkg"
        self.path_qgis_project = (
            target_root_path / f"GMC_mapset_{self.project_name}.qgz"
        )
        self.epsg = epsg_code

        # Detect layout: legacy if raster_data_<project_name> wrapper folder exists
        self._legacy_layout = (
            target_root_path / f"raster_data_{self.project_name}"
        ).is_dir()

        # Backward-compatibility path attributes (same for both layouts, computed here)
        self.path_raster_data = target_root_path / f"raster_data_{self.project_name}"
        self.path_img_correlation_outs = (
            target_root_path / f"img_correlation_outs_{self.project_name}"
        )
        self.path_tio_inversion_outs = (
            target_root_path / f"tio_inversion_outs_{self.project_name}"
        )
        # For new layout: figures live in project root; for legacy: figures_<project_name>
        self.path_figures = target_root_path / (
            f"figures_{self.project_name}" if self._legacy_layout else "figures"
        )

        # There is underscore before the attribute name because user have to access to this data from the getters to ensure that the data is up to date
        self._pzones = gpd.read_file(
            self.path_geodb, engine="pyogrio", layer="Pzones"
        ).to_crs(epsg=epsg_code)
        self._thumbs = gpd.read_file(
            self.path_geodb, engine="pyogrio", layer="Thumbs"
        ).to_crs(epsg=epsg_code)
        self._geomorphs = gpd.read_file(
            self.path_geodb, engine="pyogrio", layer="Geomorphs"
        ).to_crs(epsg=epsg_code)
        self._pairs = gpd.read_file(
            self.path_geodb, engine="pyogrio", layer="Pairs"
        ).to_crs(epsg=epsg_code)
        self._xzones = gpd.read_file(
            self.path_geodb, engine="pyogrio", layer="Xzones"
        ).to_crs(epsg=epsg_code)
        self._spines = gpd.read_file(
            self.path_geodb, engine="pyogrio", layer="Spines"
        ).to_crs(epsg=epsg_code)

        # List the processing zones names
        self.pz_names = list(self._pzones.pz_name.unique())
        logger.launch(f"Session '{self.project_name}' initialized.")

    # END def

    def pz_dir(self, pz_name: str, kind: str | None = None) -> pathlib.Path:
        """Directory for `kind` inside a pzone, resolved per this session's layout.

        For the new layout, subdirectories live under pzone root:
        - pz_dir(name, "optical") → path_root/name/optical
        - pz_dir(name, None) → path_root/name

        For the legacy layout, all subdirectories map to the old structure:
        - pz_dir(name, "optical") → path_raster_data/name/opticals
        - pz_dir(name, "image_correlation") → path_raster_data/name/image_correlation
        - pz_dir(name, "reference_dem") → path_raster_data/name (DEM file in root)
        - pz_dir(name, "reference_raster") → path_raster_data/name (grid file in root)
        - pz_dir(name, "vector") → path_raster_data/name (vectors in root)
        - pz_dir(name, "inversion") → path_raster_data/name (inversion_* subdirs in root)
        - pz_dir(name, "figures") → path_figures (project-wide)
        - pz_dir(name, "similarity") → path_raster_data/name/similarity
        - pz_dir(name, None) → path_raster_data/name

        Args:
            pz_name: Processing zone name.
            kind: Subdirectory kind (one of PZ_KIND_*); None returns the pzone root.

        Returns:
            Resolved pathlib.Path for the requested location.
        """
        if self._legacy_layout:
            pz_root = self.path_raster_data / pz_name
            if kind is None:
                return pz_root
            elif kind == PZ_KIND_OPTICAL:
                return pz_root / "opticals"
            elif kind == PZ_KIND_IMAGE_CORRELATION:
                return pz_root / "image_correlation"
            elif kind in (
                PZ_KIND_REFERENCE_DEM,
                PZ_KIND_REFERENCE_RASTER,
                PZ_KIND_VECTOR,
                PZ_KIND_INVERSION,
            ):
                return pz_root
            elif kind == PZ_KIND_FIGURES:
                return self.path_figures
            elif kind == PZ_KIND_SIMILARITY:
                # A real folder even in legacy: a loose JSON in the pzone root
                # would be the one file migrate_to_new_structure misses.
                return pz_root / "similarity"
            elif kind == PZ_KIND_ELEVATION:
                # Deliberately no fallback to the pzone root: DEM-derived
                # thumbs need the per-type subfolders of the new layout.
                raise ValueError(
                    "DEM-derived thumbs (elevation/<type>/) need the new folder "
                    "layout; run session.migrate_to_new_structure() first"
                )
            else:
                raise ValueError(f"Unknown pzone subdirectory kind: {kind}")
        else:
            pz_root = self.path_root / pz_name
            if kind is None:
                return pz_root
            elif kind in NEW_LAYOUT_PZ_SUBDIRS:
                return pz_root / kind
            else:
                raise ValueError(f"Unknown pzone subdirectory kind: {kind}")

    def thumb_dir(
        self, pz_name: str, image_type: str = IMAGE_TYPE_OPTICAL
    ) -> pathlib.Path:
        """Folder holding the thumbs of *image_type* in pzone *pz_name*.

        ``"opt"`` → ``pz_dir(pz, "optical")`` (both layouts); an elevation type
        → ``pz_dir(pz, "elevation") / <type>`` (new layout only — the legacy
        layout raises, pointing to ``migrate_to_new_structure()``).
        """
        t = normalise_image_type(image_type)
        if t == IMAGE_TYPE_OPTICAL:
            return self.pz_dir(pz_name, PZ_KIND_OPTICAL)
        return self.pz_dir(pz_name, PZ_KIND_ELEVATION) / t

    def pair_dir(
        self, pz_name: str, image_type: str = IMAGE_TYPE_OPTICAL
    ) -> pathlib.Path:
        """Folder holding the pair directories of *image_type* in pzone *pz_name*.

        ``"opt"`` → ``image_correlation/`` (unchanged); any other type →
        ``image_correlation/<type>/``. Mirrors how ``Pair`` builds ``pa_path``.
        """
        t = normalise_image_type(image_type)
        if t != IMAGE_TYPE_OPTICAL:
            self.thumb_dir(pz_name, t)  # legacy layout raises here, not later
        return self.pz_dir(pz_name, PZ_KIND_IMAGE_CORRELATION) / pair_subdir(t)

    # * Works properly
    def find_qgis_binary(self, qgis_binary_path: str | None = None) -> str:
        """Find the QGIS binary executable.

        Args:
            qgis_binary_path (str | None, optional): Path to the QGIS binary. Defaults to None.

        Raises:
            FileNotFoundError: If the QGIS binary is not found.

        Returns:
            str: Path to the QGIS binary.
        """
        # Try to find 'qgis' in PATH
        qgis_bin = shutil.which("qgis")
        if qgis_bin:
            return qgis_bin
        # Try common install locations
        common_paths = [
            "/usr/bin/qgis",
            "/usr/local/bin/qgis",
            "/opt/qgis/bin/qgis",
        ]
        for path in common_paths:
            if pathlib.Path(path).exists():
                return path
        raise FileNotFoundError(
            "QGIS binary not found. Please install QGIS or specify its path."
        )

    # * Works properly
    # ! This function may overload the system if the project as QGIs is invoked from terminal. We can add a warning in the docstring to inform the user about this point. We can also add an optional argument to choose if the project should be opened or not.
    def open_qgis_project(
        self,
        project_path: str | pathlib.Path | None = None,
        qgis_binary_path: str | None = None,
    ) -> None:
        """Open the QGIS project

        Args:
            project_path (str | pathlib.Path | None, optional): Path to the QGIS project file. Defaults to None.
            qgis_binary_path (str | None, optional): Path to the QGIS binary. Defaults to None.

        Raises:
            FileNotFoundError: If the QGIS binary or project file is not found.
        """
        if qgis_binary_path is None:
            qgis_binary_path = self.find_qgis_binary()

        if not qgis_binary_path:
            raise FileNotFoundError(
                "QGIS binary not found. Please install QGIS or specify its path."
            )
        else:
            call = []
            call.extend([f"{qgis_binary_path}"])
            call.extend(
                ["--nologo", "--noplugins", "--noversioncheck", "--hide-browser"]
            )
            if project_path is not None:
                call.extend(["--project", f"{project_path}"])
            else:
                call.extend(["--project", f"{self.path_qgis_project}"])
            call.extend([f"{self.path_geodb}"])
        subprocess.run(call, check=True)
        logger.info(f"Opening QGIS project {self.path_qgis_project}.")

    ###################################################
    # -------------------- GETTERS --------------------#
    # * Works properly
    def get_thumbs_overview(self, criterias: str | list[str] = "") -> gpd.GeoDataFrame:
        """Get thumbnails overview informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            gpd.GeoDataFrame: Send a dataframe with informations about each thumb raster file meeting the criteria. If there's no criteria, send informations about all the thumbs of the project.
        # TODO: Implement unittests.
        """
        return self._search_engine("Thumbs", criterias)

    #! ask Thibaut for homogenesation.
    def get_thumbs(self, criterias: str | list[str] = "") -> list[gmc_thumb.Thumb]:
        """Get thumbnails

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list: Send a list of thumb.Thumb objects meeting the criterias.
        """
        selected_thumbs = self.get_thumbs_overview(criterias)
        return [gmc_thumb.Thumb(x.th_path) for x in selected_thumbs.iloc]
        # selected_thumbs = self.get_thumbs_overview(criterias)
        # return [gmc_thumb.Thumb(r.th_path) for r in selected_thumbs.itertuples(index=False)]

    # * Works properly
    def get_pairs_overview(
        self, criterias: str | list[str] = "", image_type: str | None = None
    ) -> gpd.GeoDataFrame:
        """Get pairs overview informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".
            image_type (str | None, optional): keep only pairs of this image type
                (``pa_type`` column; rows predating it read as ``"opt"``).
                ``None`` (default) keeps every type.

        Returns:
            gpd.GeoDataFrame: Send a dataframe with each possible Pair according to the Thumbs
        # TODO: Implement unittests.
        """
        overview = self._search_engine("Pairs", criterias)
        if image_type is not None and overview is not None:
            overview = overview[
                type_column(overview, "pa_type") == normalise_image_type(image_type)
            ]
        return overview

    # * Works properly
    def get_pairs(
        self, criterias: str | list[str] = "", image_type: str | None = None
    ) -> list[gmc_pair.Pair]:
        """Get pairs informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".
            image_type (str | None, optional): keep only pairs of this image type;
                ``None`` (default) keeps every type.

        Returns:
            pd.DataFrame: Send a list of pair.Pairs objects meeting the criterias
        # TODO: solve issue regarding order of return pairs. When session.get_pairs(['ribon', '2014', '2018']), pair 2018-2014 is returned first.

        """
        # selected_pairs = self.get_pairs_overview(criterias)
        # return [gmc_pair.Pair(self, target_path=x.pa_path) for x in selected_pairs.iloc]
        selected_pairs = self.get_pairs_overview(criterias, image_type=image_type)
        # return [gmc_pair.Pair(self, target_path=x.pa_path) for x in selected_pairs]
        return [
            gmc_pair.Pair(self, target_path=row["pa_path"])
            for _, row in selected_pairs.iterrows()
        ]

    def get_failed_pairs(
        self, criterias: str | list[str] = "", image_type: str | None = None
    ) -> list[gmc_pair.Pair]:
        """Return pairs where one or more ASP output files are missing.

        Checks all files listed in ``ASP._KEEP_SUFFIXES`` inside each pair's
        ``pa_path``. Logs the missing files for each failed pair.

        Args:
            criterias: Same filter syntax as :meth:`get_pairs`.
            image_type: Keep only pairs of this image type (``"opt"``, ``"hillshade"``, ``"mdhillshade"``, ``"slope"``, ``"dem"``); ANDed with *criterias*. Prefer it over a criterias string: ``"hillshade"`` is also a substring of ``"mdhillshade"``.

        Returns:
            List of ``Pair`` objects with at least one missing output file.
        """
        pairs = self.get_pairs(criterias, image_type=image_type)
        failed = []
        for pair in pairs:
            outputs = pair.check_corr_outputs()
            missing = [s for s, exists in outputs.items() if not exists]
            if missing:
                logger.warning(
                    f"Pair '{pair.pa_key}' failed — missing: {', '.join(missing)}"
                )
                failed.append(pair)
        logger.statistics(
            f"{len(failed)} / {len(pairs)} pair(s) have incomplete correlation outputs"
        )
        return failed

    # * Works properly
    def get_pzones_overview(self, pz_name: str | list[str] = "") -> pd.DataFrame:
        """Get pzones overview informations

        Args:
            pz_name (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a dataframe with each pzone
        # TODO: Implement unittests.
        """
        return self._search_engine("Pzones", pz_name)

    def get_pzones(self, pz_name: str = "") -> list[gmc_pzone.Pzone]:
        """Get pzones informations

        Args:
            pz_name (str, optional): pzone name. Defaults to "".

        Returns:
            list: Send a list of pzone.Pzones objects.
        """
        # selected_pzones = self.get_pzones_overview(pz_name)
        selected_pzones = self.get_pzones_overview(pz_name)
        # return [gmc_pzone.Pzone(x.pz_name, self) for x in selected_pzones.iloc]
        return [
            gmc_pzone.Pzone(x.pz_name, self)
            for x in selected_pzones.itertuples(index=False)
        ]

    def get_geomorphs_overview(self, criterias: str | list[str] = "") -> pd.DataFrame:
        """Get geomorphos overview informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a dataframe with each geomorph.
        """
        return self._search_engine("Geomorphs", criterias)

    def get_geomorphs(
        self, criterias: str | list[str] = ""
    ) -> list[gmc_geomorph.Geomorph]:
        """Get geomorphs informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list: Send a list of geomorph.Geomorphs objects.
        """
        selected_geomorphs = self.get_geomorphs_overview(criterias)
        return [
            gmc_geomorph.Geomorph(self, x.ge_frogi_id)
            for x in selected_geomorphs.itertuples(index=False)
        ]

    def get_xzone(self, xz_id: str | int | float) -> gmc_xzone.Xzone:
        """Get a xzone

        Args:
            xz_id (str | int | float): xzone id.

        Returns:
            gmc_xzone.Xzone: Send a xzones.Xzones object
        """
        return gmc_xzone.Xzone(self, xz_id)

    def get_pairs_overview_on_period(
        self,
        ymin: str | pd.Timestamp,
        ymax: str | pd.Timestamp,
        criterias: str | list[str] = "",
    ) -> pd.DataFrame:
        """Get pairs overview informations on a given period

        Args:
            ymin (str | pd.Timestamp): minimun year of the given period.
            ymax (str | pd.Timestamp): maximum year of the given period.
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Return a dataframe with each pair completely included in the period [yMin;yMax]
        """
        # def is_valid_timestamp(date_str):
        #     try:
        #         pd.to_datetime(date_str)
        #         return True
        #     except ValueError:
        #         return False
        # # Convert ymin and ymax to pd.Timestamp if they are strings
        # if isinstance(ymin, str) and is_valid_timestamp(ymin):
        #     ymin = pd.to_datetime(ymin)
        # if isinstance(ymax, str) and is_valid_timestamp(ymax):
        #     ymax = pd.to_datetime(ymax)

        pairs = self.get_pairs_overview(criterias).copy()
        pairs["chrono_min"] = pairs.apply(
            lambda row: min(
                int(row.pa_left_date.split("-")[0]),
                int(row.pa_right_date.split("-")[0]),
            ),
            axis=1,
        )
        pairs["chrono_max"] = pairs.apply(
            lambda row: max(
                int(row.pa_left_date.split("-")[0]),
                int(row.pa_right_date.split("-")[0]),
            ),
            axis=1,
        )
        # Possible error on text and number
        pairs = pairs[(pairs.chrono_min >= ymin) & (pairs.chrono_max <= ymax)]
        return pairs

    def get_pairs_on_period(
        self,
        ymin: str | pd.Timestamp,
        ymax: str | pd.Timestamp,
        criterias: str | list[str] = "",
    ) -> list[gmc_pair.Pair]:
        """Get pairs on a given period

        Args:
            ymin (str | pd.Timestamp): minimun year of the given period.
            ymax (str | pd.Timestamp): maximun year of the given period.
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list[gmc_pair.Pair]: Returns a list of gcm_pair.Pair objects with the pairs completely included in the period [yMin;yMax]
        """
        board = self.get_pairs_overview_on_period(ymin, ymax, criterias)
        pairs = [gmc_pair.Pair(self, p.pa_path) for p in board.iloc]
        return pairs

    def get_spine(self, sp_id: str | int | float) -> gmc_spine.Spine:
        """Get spine

        Args:
            sp_id (str | int | float): Spine ID.

        Returns:
            gmc_spine.Spine: Send a spine.Spine object
        """
        return gmc_spine.Spine(self, sp_id)

    def get_dems(self, criterias: str | list[str] = "") -> list:
        """Get a list of DEMs

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list: Return a list of DEMs
        """
        # Read each DEM once. The previous form called pz.get_dem() twice per
        # pzone — once in the filter, once in the body — and get_dem() loads
        # the full array, so every DEM was read off disk twice.
        dems = (pz.get_dem() for pz in self.get_pzones(criterias))
        return [dem for dem in dems if dem is not False]

    # ------------------------------------------------------------------ #
    # Pzone reference grid / DEM / moving areas
    # ------------------------------------------------------------------ #

    def _pzone_vector(self, pz_name: str) -> gu.Vector:
        """Return a single pzone's AOI as a :class:`geoutils.Vector`.

        Wraps the one-row-slice idiom used by ``sieve_bulk`` and
        ``extract_pairs_raw_displacements``.  ``self._pzones`` is already in
        the session CRS, so no reprojection happens here.

        Args:
            pz_name: Exact processing-zone name.

        Returns:
            gu.Vector: Single-feature vector in the session CRS.

        Raises:
            KeyError: If *pz_name* matches no pzone.
        """
        df = self.get_pzones_overview(pz_name)
        exact = df[df["pz_name"] == pz_name] if "pz_name" in df.columns else df
        if exact.empty:
            raise KeyError(
                f"No pzone named {pz_name!r}. Known pzones: {sorted(self.pz_names)}"
            )
        return gu.Vector(
            gpd.GeoDataFrame([exact.iloc[0]], crs=df.crs, geometry="geometry")
        )

    def _normalize_vector_source(
        self,
        source,
        layer: str | None = None,
        bounds: "tuple[float, float, float, float] | None" = None,
    ) -> gu.Vector:
        """Normalise any AOI/polygon *source* into a session-CRS ``gu.Vector``.

        Accepts a :class:`geoutils.Vector`, a :class:`geoutils.Raster` (its
        footprint is used), or a path to either a raster or a vector file.

        Args:
            source: Vector, raster, or path to one of them.
            layer: Layer name for multi-layer vector files. Ignored otherwise.
            bounds: Optional ``(left, bottom, right, top)`` in the **session CRS**.
                Only the features whose bounding box meets it are kept — and,
                for a file path, only those are read from disk — *before* the
                reprojection, so a continental inventory costs what the area of
                interest costs. Features are selected, never clipped. An empty
                selection returns an empty vector instead of raising.

        Returns:
            gu.Vector: Features reprojected to ``self.epsg``.

        Raises:
            TypeError: If *source* is not a supported type.
            ValueError: If the source contains no features (and *bounds* is None).
        """
        raster_suffixes = (".tif", ".tiff")
        bounds_requested = bounds is not None
        if isinstance(source, gu.Vector):
            vec = source
        elif isinstance(source, gu.Raster):
            vec = gu.Vector(source.footprint.ds)
        elif isinstance(source, gpd.GeoDataFrame):
            vec = gu.Vector(source)
        elif isinstance(source, (str, pathlib.Path)):
            path = pathlib.Path(source)
            if path.suffix.lower() in raster_suffixes:
                vec = gu.Vector(gu.Raster(str(path), load_data=False).footprint.ds)
            else:
                read_kwargs = {"engine": "pyogrio"}
                if layer is not None:
                    read_kwargs["layer"] = layer
                if bounds is not None:
                    import pyogrio
                    from rasterio.warp import transform_bounds

                    info = pyogrio.read_info(str(path), layer=layer)
                    src_crs = info.get("crs")
                    read_kwargs["bbox"] = (
                        tuple(
                            transform_bounds(
                                rio_crs.CRS.from_epsg(self.epsg),
                                src_crs,
                                *bounds,
                                densify_pts=21,
                            )
                        )
                        if src_crs
                        else tuple(bounds)
                    )
                vec = gu.Vector(gpd.read_file(str(path), **read_kwargs))
                bounds = None  # already applied on read
        else:
            raise TypeError(
                "source must be a file path, gu.Vector, gu.Raster, or GeoDataFrame "
                f"(got {type(source).__name__})."
            )

        if bounds is not None and not vec.ds.empty:
            from rasterio.warp import transform_bounds
            from shapely.geometry import box

            src_bounds = (
                transform_bounds(
                    rio_crs.CRS.from_epsg(self.epsg), vec.crs, *bounds, densify_pts=21
                )
                if vec.crs is not None
                else tuple(bounds)
            )
            hit = vec.ds.sindex.query(box(*src_bounds), predicate="intersects")
            vec = gu.Vector(vec.ds.iloc[np.sort(hit)])

        if vec.ds.empty:
            if bounds_requested:
                return gu.Vector(gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{self.epsg}"))
            raise ValueError(f"No features found in vector source: {source!r}")

        # A CRS-less source cannot be reprojected — it can only be declared.
        # Assume it is already in the session CRS and say so loudly.
        if vec.crs is None:
            logger.warning(
                f"Vector source {source!r} has no CRS — assuming EPSG:{self.epsg}."
            )
            vec = gu.Vector(vec.ds.set_crs(epsg=self.epsg))
        elif vec.crs.to_epsg() != self.epsg:
            vec = vec.reproject(crs=self.epsg)

        return vec

    def reference_grid_path(self, pz_name: str) -> pathlib.Path:
        """Path of a pzone's reference-grid raster (may not exist yet)."""
        return self.pz_dir(pz_name, PZ_KIND_REFERENCE_RASTER) / f"{pz_name}_grid.tif"

    def build_reference_grid(
        self,
        pz_name: str,
        resolution: float | None = None,
        from_pair=None,
        overwrite: bool = False,
    ) -> pathlib.Path:
        """Create the pzone's reference grid and persist it as a 1-bit GeoTIFF.

        The reference grid is the single source of truth for a pzone's CRS,
        extent and pixel dimensions.  Everything else in the pzone — the DEM,
        the moving-area polygons, any raster you need to crop — is placed on
        it, so nothing has to hunt for "some displacement raster" to match
        against.

        The file holds all ones and is packed one pixel per bit, so it costs a
        few kB no matter how large the grid is.  Only its georeferencing
        matters.

        Exactly one of *resolution* or *from_pair* must be given:

        - ``resolution=`` derives the grid from the pzone polygon via
          :meth:`_compute_canonical_grid` — bounds snapped outward to a
          lattice anchored at the CRS origin, with an exact width/height.
          This is reproducible from the geodatabase alone and is the same
          grid ``extract_pairs_raw_displacements`` snaps pairs to.
        - ``from_pair=`` copies the grid off an existing pair's EW raster.
          Use it when rasters are already on disk on a grid you must match.

        Args:
            pz_name: Processing-zone name.
            resolution: Target pixel size in CRS units (metres).
            from_pair: A :class:`~geomulticorr.core.pair.Pair` whose EW raster
                defines the grid.
            overwrite: Rebuild even if the grid file already exists.

        Returns:
            pathlib.Path: Path to the written grid raster.

        Raises:
            ValueError: If neither or both of *resolution* / *from_pair* are
                given.
            FileNotFoundError: If *from_pair*'s EW raster does not exist.
        """
        from geomulticorr.utils._grid import describe_grid, write_binary_raster

        if (resolution is None) == (from_pair is None):
            raise ValueError(
                "Provide exactly one of resolution= or from_pair= "
                "(resolution derives the grid from the pzone AOI; from_pair "
                "copies it from an existing pair's EW raster)."
            )

        out_path = self.reference_grid_path(pz_name)
        if out_path.exists() and not overwrite:
            logger.info(f"Reference grid already exists: {out_path} (overwrite=False)")
            return out_path

        if from_pair is not None:
            ew_path = pathlib.Path(from_pair.pa_ew_path)
            if not ew_path.exists():
                raise FileNotFoundError(
                    f"Pair {from_pair.pa_key!r} has no EW raster at {ew_path}. "
                    "Run extract_pairs_raw_displacements() first."
                )
            ref = gu.Raster(str(ew_path), load_data=False)
            crs, transform = ref.crs, ref.transform
            width, height = ref.width, ref.height
        else:
            pz_vec = self._pzone_vector(pz_name)
            bounds, (width, height) = self._compute_canonical_grid(
                pz_vec, self.epsg, resolution
            )
            crs = rio_crs.CRS.from_epsg(self.epsg)
            transform = rio_transform.from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_binary_raster(
            np.ones((height, width), dtype="uint8"), transform, crs, out_path
        )

        written = gu.Raster(str(out_path), load_data=False)
        logger.success(
            f"Reference grid for '{pz_name}': {describe_grid(written)} → {out_path}"
        )
        return out_path

    def get_reference_grid(self, pz_name: str) -> gu.Raster:
        """Load a pzone's reference grid, without reading any pixels.

        Opened with ``load_data=False``, so this costs one metadata read and
        allocates nothing.  ``.crs``, ``.transform``, ``.res``, ``.bounds``,
        ``.shape``, ``.width`` and ``.height`` are all available.

        To place a raster on this grid, destructure it — do **not** pass it as
        ``ref=``.  ``reproject(ref=…)`` takes only CRS, resolution and bounds
        from the reference and then *infers* the size, which can round to a
        different shape::

            g = session.get_reference_grid("MyPzone")
            out = r.reproject(crs=g.crs, bounds=g.bounds,
                              grid_size=(g.width, g.height), resampling="bilinear")

        Args:
            pz_name: Processing-zone name.

        Returns:
            gu.Raster: Lazily-opened grid raster.

        Raises:
            FileNotFoundError: If the grid has not been built yet.
        """
        path = self.reference_grid_path(pz_name)
        if not path.exists():
            raise FileNotFoundError(
                f"No reference grid for pzone {pz_name!r} at {path}. "
                f"Build it first: session.build_reference_grid({pz_name!r}, resolution=…)"
            )
        return gu.Raster(str(path), load_data=False)

    @gmc_profile("gmc.session.build_reference_dem", memprof=True)
    def build_reference_dem(
        self,
        pz_name: str,
        source,
        resampling: str = "bilinear",
        overwrite: bool = False,
    ) -> pathlib.Path:
        """Crop and regrid a DEM onto the pzone reference grid, and persist it.

        Writes ``<pzone>/reference_dem/<pz_name>_dem.tif`` — the exact path
        :meth:`~geomulticorr.core.pzone.Pzone.get_dem` reads, so that method
        starts working with no change to it.

        This is what stops the corrections pipeline re-warping the DEM: once
        the stored DEM sits on the same grid as the displacement rasters, the
        alignment check in every ``fit()`` is a no-op.

        The source is cropped to the pzone AOI *before* reprojection, so only
        the overlapping window is read from disk.

        Args:
            pz_name: Processing-zone name.
            source: DEM as a path, a :class:`geoutils.Raster`, or the ``Path``
                returned by
                :meth:`~geomulticorr.data_sources.global_dem_download.DEMFinder.download_dem`.
            resampling: Resampling method for the regrid. ``"bilinear"`` suits
                continuous elevation; use ``"cubic"`` for smoother slopes.
            overwrite: Rebuild even if the DEM already exists.

        Returns:
            pathlib.Path: Path to the written DEM.

        Raises:
            FileNotFoundError: If the reference grid has not been built, or the
                source path does not exist.
            ValueError: If the source does not overlap the pzone.
        """
        from geomulticorr.utils._grid import describe_grid, grids_match

        grid = self.get_reference_grid(pz_name)

        out_path = self.pz_dir(pz_name, PZ_KIND_REFERENCE_DEM) / f"{pz_name}_dem.tif"
        if out_path.exists() and not overwrite:
            logger.info(f"Reference DEM already exists: {out_path} (overwrite=False)")
            return out_path

        if isinstance(source, gu.Raster):
            dem = source
        else:
            src_path = pathlib.Path(source)
            if not src_path.exists():
                raise FileNotFoundError(f"DEM source not found: {src_path}")
            dem = gu.Raster(str(src_path), load_data=False)

        # Reject a disjoint source up front — reprojecting it would otherwise
        # produce a full grid of nodata with no explanation.
        gxmin, gymin, gxmax, gymax = grid.get_bounds_projected(out_crs=dem.crs)
        dxmin, dymin, dxmax, dymax = (
            dem.bounds.left,
            dem.bounds.bottom,
            dem.bounds.right,
            dem.bounds.top,
        )
        if gxmax <= dxmin or gxmin >= dxmax or gymax <= dymin or gymin >= dymax:
            raise ValueError(
                f"DEM source does not overlap pzone {pz_name!r}.\n"
                f"  pzone (in DEM CRS): ({gxmin:.1f}, {gymin:.1f}, {gxmax:.1f}, {gymax:.1f})\n"
                f"  DEM:                ({dxmin:.1f}, {dymin:.1f}, {dxmax:.1f}, {dymax:.1f})"
            )

        # Windowed crop in the source CRS first: reads only the overlap.
        # Padded by a few pixels so the resampling kernel has neighbours at
        # the edge instead of inventing nodata there.
        pad = 4 * max(dem.res)
        try:
            dem = dem.crop(
                [gxmin - pad, gymin - pad, gxmax + pad, gymax + pad],
                mode="match_pixel",
            )
        except Exception as exc:  # pragma: no cover - geoutils edge cases
            logger.warning(f"Could not pre-crop DEM ({exc}); reprojecting in full.")

        dem_on_grid = dem.reproject(
            crs=grid.crs,
            bounds=grid.bounds,
            grid_size=(grid.width, grid.height),
            resampling=resampling,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        dem_on_grid.to_file(str(out_path))

        if not grids_match(gu.Raster(str(out_path), load_data=False), grid):
            logger.warning(
                f"Written DEM grid does not match the reference grid for {pz_name!r} — "
                "this should not happen; inspect the output."
            )
        logger.success(
            f"Reference DEM for '{pz_name}': {describe_grid(dem_on_grid)} → {out_path}"
        )
        return out_path

    def moving_areas_path(self, pz_name: str) -> pathlib.Path:
        """Path of a pzone's moving-area polygons (may not exist yet)."""
        return self.pz_dir(pz_name, PZ_KIND_VECTOR) / f"{pz_name}_moving-areas.gpkg"

    def load_moving_areas(
        self,
        pz_name: str,
        source,
        layer: str | None = None,
        buffer: float | None = None,
        clip: bool = True,
        overwrite: bool = False,
    ) -> pathlib.Path:
        """Import moving-area polygons, place them on the pzone grid, and store them.

        Moving areas are the terrain the corrections must **not** be fitted on
        — rock glaciers, landslides, active glacier tongues.  This reprojects
        them into the session CRS, trims them to the reference grid, and
        persists them at ``<pzone>/vector/<pz_name>_moving-areas.gpkg`` so the
        polygons live with the project instead of at some absolute path in a
        notebook.

        Distinct from the generated ``<pz>_moving-areas_round-0.gpkg``, which
        is the *output* of the clustering workflow — these are user-supplied
        *input*.

        Args:
            pz_name: Processing-zone name.
            source: Polygons as a path, :class:`geoutils.Vector`, or
                :class:`geopandas.GeoDataFrame`.
            layer: Layer name, for multi-layer GeoPackages.
            buffer: Optional buffer in CRS units (metres) applied after
                reprojection — widens the exclusion around each feature.
            clip: Cut geometries at the grid boundary. ``False`` keeps whole
                features that merely intersect it.
            overwrite: Replace an existing file.

        Returns:
            pathlib.Path: Path to the written GeoPackage.

        Raises:
            FileNotFoundError: If the reference grid has not been built.
            ValueError: If no polygon survives the clip to the pzone.
        """
        grid = self.get_reference_grid(pz_name)

        out_path = self.moving_areas_path(pz_name)
        if out_path.exists() and not overwrite:
            logger.info(f"Moving areas already exist: {out_path} (overwrite=False)")
            return out_path

        vec = self._normalize_vector_source(source, layer=layer)
        n_input = len(vec.ds)

        if buffer:
            vec = gu.Vector(vec.ds.assign(geometry=vec.ds.geometry.buffer(buffer)))

        vec = vec.crop(grid, clip=clip)
        if vec.ds.empty:
            raise ValueError(
                f"No moving-area polygon overlaps pzone {pz_name!r}. "
                "Check that the source covers the right area."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        layer_name = f"{pz_name}_moving-areas"
        vec.ds.to_file(str(out_path), layer=layer_name, driver="GPKG")

        logger.success(
            f"Moving areas for '{pz_name}': {len(vec.ds)}/{n_input} features "
            f"(EPSG:{self.epsg}) → {out_path}"
        )
        return out_path

    def get_moving_areas(self, pz_name: str) -> gu.Vector:
        """Load a pzone's moving-area polygons.

        Returns a :class:`geoutils.Vector`, which is the CRS-safe input to
        :class:`~geomulticorr.corrections.masks.StableAreaMask` — it carries
        its CRS, so the mask reprojects rather than assuming.

        Args:
            pz_name: Processing-zone name.

        Returns:
            gu.Vector: Moving-area polygons in the session CRS.

        Raises:
            FileNotFoundError: If they have not been imported yet.
        """
        path = self.moving_areas_path(pz_name)
        if not path.exists():
            raise FileNotFoundError(
                f"No moving areas for pzone {pz_name!r} at {path}. "
                f"Import them first: session.load_moving_areas({pz_name!r}, <source>)"
            )
        return gu.Vector(
            gpd.read_file(str(path), engine="pyogrio", layer=f"{pz_name}_moving-areas")
        )

    def similarity_path(
        self, pz_name: str, image_type: str = IMAGE_TYPE_OPTICAL
    ) -> pathlib.Path:
        """JSON trace of :meth:`compute_thumbs_similarity` for one pzone and type.

        ``<pzone>/similarity/thumb_similarity_<pz>[_<type>].json``. The type
        suffix is omitted for ``"opt"``. Layout:
        :mod:`geomulticorr.similarity.trace`.
        """
        t = normalise_image_type(image_type)
        suffix = "" if t == IMAGE_TYPE_OPTICAL else f"_{t}"
        return (
            self.pz_dir(pz_name, PZ_KIND_SIMILARITY)
            / f"thumb_similarity_{pz_name}{suffix}.json"
        )

    def _similarity_trace_path(self, pz_name: str, image_type: str) -> pathlib.Path:
        """:meth:`similarity_path`, after moving a trace left in ``image_correlation/``.

        Traces written before the ``similarity/`` folder existed sit in
        ``image_correlation/`` under the same name; the first read or write
        moves such a file once. A failed move is logged and the old file is
        left where it is (the new path is then simply absent).
        """
        path = self.similarity_path(pz_name, image_type)
        old = self.pz_dir(pz_name, PZ_KIND_IMAGE_CORRELATION) / path.name
        if not path.exists() and old.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(path))
                logger.info(f"Moved the similarity trace {old} → {path}")
            except OSError as exc:
                logger.warning(f"Could not move the similarity trace {old}: {exc}")
        return path

    @gmc_profile("gmc.session.compute_thumbs_similarity", memprof=True)
    def compute_thumbs_similarity(
        self,
        pz_name: str,
        image_type: str = IMAGE_TYPE_OPTICAL,
        couples: str = "all",
        max_step: int | None = None,
        sensor_filter: str | None = None,
        cross_sensor_only: bool = False,
        moving_areas: Any = "auto",
        moving_buffer: float = 0.0,
        grid: str = "coarser",
        save: bool = True,
        overwrite: bool = False,
        progress: bool = True,
        plots: bool = True,
        formats: Sequence[str] = ("png",),
        **report_kwargs,
    ) -> pd.DataFrame:
        """Measure how alike the thumbs of a pzone are, couple by couple.

        A predictor of correlability computed from the images alone, before
        anything is sent to ASP — see ``docs/source/theory/image_similarity.md``
        for what each metric means. Every couple is put on a common grid and
        measured **on stable ground** (outside the pzone's moving areas): on
        moving terrain two images are *supposed* to differ.

        Couples are **unordered**, left = earlier date; similarity is (nearly)
        symmetric, so a forward and a backward pair share one row.

        Thumb files are located through :meth:`thumb_dir`, not the stored
        ``th_path``, so a project that was moved still resolves.

        Results go to the JSON trace :meth:`similarity_path`, one *run* per set
        of parameters (``params_hash``). With ``overwrite=False`` a couple
        already stored in the same run is reused, so adding thumbs only measures
        the new couples; changing any parameter starts another run (earlier runs
        are kept). A write failure is logged, never raised.

        Every row gets a ``score``: the mean percentile rank of
        ``trial_peak_median``, ``coh_mean``, ``trial_valid_frac`` and (inverted)
        ``trial_sigma_px`` **over the whole run**, so ``score >= 0.8`` means
        "better than 80 % of this dataset's couples". It is recomputed whenever
        couples are added to the run.

        With ``plots=True`` the dataset overview is written to
        ``<pzone>/figures/similarity/``: thumb × thumb matrices, indices per
        sensor combination and, once at least five of the couples have been
        correlated and corrected, the validation of every index against what
        ASP achieved (re-run after correlation: the couples are reused, only
        the figure is new).

        Args:
            pz_name: Processing zone.
            image_type: Thumb type, as everywhere else (``"opt"``, ``"hillshade"``…).
            couples: ``"all"`` (every couple) or a pairing strategy name
                (``"consecutive"``, ``"step"``, ``"redundancy"``,
                ``"forward-backward"``) to measure only the couples it would pair.
            max_step: For ``"step"`` / ``"redundancy"``.
            sensor_filter: Case-insensitive substring of the thumb file name.
            cross_sensor_only: Keep only couples whose sensor **families** differ
                (``spot6``↔``spot7`` is not cross-sensor).
            moving_areas: ``"auto"`` (the pzone's polygons from
                :meth:`get_moving_areas`, with a warning when none are loaded),
                ``None`` (all terrain), or a path / GeoDataFrame / ``gu.Vector``.
            moving_buffer: Margin around the polygons, in metres.
            grid: ``"coarser"`` (each couple on its coarser image's grid — what a
                cross-sensor pair is correlated on) or ``"reference"`` (the
                pzone reference grid, for every couple).
            save: Write the JSON trace.
            overwrite: Recompute couples even when a matching row is stored.
            progress: Show a progress bar.
            plots: Write the overview figures.
            formats: Figure formats (``"png"``, ``"pdf"``, ``"svg"``, ``"jpg"``).
            **report_kwargs: Metric settings — ``kernel`` (default 21, as
                :meth:`prepare_pairs_correlation`), ``search``,
                ``prefilter_mode``, ``prefilter_width``, ``n_tiles``,
                ``min_peak``, ``tile``, ``bins``, ``rng``. Use the values the
                pairs are (or will be) correlated with.

        Returns:
            One row per requested couple: :data:`SIMILARITY_ID_COLUMNS`, then
            ``similarity.raster.REPORT_KEYS``, then ``score``; sorted by dates.
        """
        # Imported here: scikit-image is only needed by this method.
        import hashlib
        import json

        from geomulticorr.similarity import raster as sim_raster
        from geomulticorr.similarity import trace as sim_trace
        from geomulticorr.similarity.selection import sensor_family

        t = normalise_image_type(image_type)
        params = sim_raster.report_params(**report_kwargs)

        thumbs = self._thumbs
        if thumbs is None or len(thumbs) == 0:
            raise ValueError("The Thumbs layer is empty; run update_thumbs() first")
        thumbs = thumbs[
            (thumbs["th_pz_name"] == pz_name) & (type_column(thumbs, "th_type") == t)
        ].copy()
        thumbs["file"] = [
            self.thumb_dir(pz_name, t) / pathlib.Path(str(p)).name
            for p in thumbs["th_path"]
        ]
        if sensor_filter:
            thumbs = thumbs[
                thumbs["file"].map(lambda p: sensor_filter.lower() in p.name.lower())
            ]
        missing = ~thumbs["file"].map(lambda p: p.exists())
        if missing.any():
            logger.warning(
                f"{int(missing.sum())} thumb(s) of '{pz_name}' not found in "
                f"{self.thumb_dir(pz_name, t)}; skipped"
            )
            thumbs = thumbs[~missing]
        thumbs = thumbs.sort_values("th_date").reset_index(drop=True)
        if len(thumbs) < 2:
            raise ValueError(
                f"Need at least two {t!r} thumbs in '{pz_name}', found {len(thumbs)}"
            )

        if couples == "all":
            idx = list(itertools.combinations(range(len(thumbs)), 2))
        else:
            raw = gmc_pzone._strategy_pair_indices(len(thumbs), couples, max_step)
            idx = sorted({(min(i, j), max(i, j)) for i, j in raw})
        families = [sensor_family(x) for x in thumbs["th_sensor"]]
        if cross_sensor_only:
            idx = [(i, j) for i, j in idx if families[i] != families[j]]

        if isinstance(moving_areas, str) and moving_areas == "auto":
            if self.moving_areas_path(pz_name).exists():
                moving_source = f"auto:{self.moving_areas_path(pz_name).name}"
                moving_areas = self.get_moving_areas(pz_name)
            else:
                logger.warning(
                    f"No moving areas loaded for '{pz_name}': similarity is measured "
                    f"on all terrain, moving areas included. Load them with "
                    f"session.load_moving_areas({pz_name!r}, <source>)"
                )
                moving_source, moving_areas = "none", None
        elif moving_areas is None:
            moving_source = "none"
        else:
            moving_source = (
                f"custom:{moving_areas}"
                if isinstance(moving_areas, (str, pathlib.Path))
                else "custom"
            )

        if grid == "reference":
            target = sim_raster.Grid.of(self.reference_grid_path(pz_name))
        elif grid == "coarser":
            target = "coarser"
        else:
            raise ValueError(f"grid must be 'coarser' or 'reference', got {grid!r}")

        fingerprint = dict(
            params, grid=grid, moving=moving_source, moving_buffer=moving_buffer
        )
        params_hash = hashlib.sha1(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest()[:10]

        path = self._similarity_trace_path(pz_name, t)
        legacy_csv = (
            self.pz_dir(pz_name, PZ_KIND_IMAGE_CORRELATION)
            / path.with_suffix(".csv").name
        )
        if legacy_csv.exists() and not path.exists():
            logger.info(
                f"{legacy_csv.name} (CSV, before the JSON trace) is not read; "
                f"its couples are measured again into {path.name}"
            )
        doc = sim_trace.empty_trace(pz_name, t)
        if path.exists():
            try:
                doc = sim_trace.read_trace(path)
            except (OSError, ValueError) as exc:
                logger.warning(f"Could not read {path}: {exc}; starting a new trace")
        stored = pd.DataFrame()
        if params_hash in doc["runs"]:
            stored = sim_trace.run_frame(doc, params_hash)
        reusable = set()
        if not overwrite and not stored.empty:
            reusable = set(zip(stored["left_key"], stored["right_key"]))

        keys = [p.stem for p in thumbs["file"]]
        todo = [(i, j) for i, j in idx if (keys[i], keys[j]) not in reusable]
        logger.info(
            f"Similarity '{pz_name}' ({t}): {len(idx)} couple(s), "
            f"{len(idx) - len(todo)} reused, {len(todo)} to measure "
            f"(params {params_hash})"
        )

        cache: dict = {}
        rows = []
        for i, j in tqdm.tqdm(todo, disable=not progress, desc="similarity"):
            a, b = thumbs.iloc[i], thumbs.iloc[j]
            try:
                report = sim_raster.compare_rasters(
                    a["file"],
                    b["file"],
                    moving_areas=moving_areas,
                    buffer=moving_buffer,
                    grid=target,
                    cache=cache,
                    **params,
                )
            except Exception as exc:  # one unreadable file must not stop a batch
                logger.warning(f"Similarity {keys[i]} / {keys[j]} failed: {exc}")
                continue
            da = pd.Timestamp(a["th_date"])
            db = pd.Timestamp(b["th_date"])
            rows.append(
                {
                    "pz_name": pz_name,
                    "image_type": t,
                    "left_key": keys[i],
                    "right_key": keys[j],
                    "left_date": str(a["th_date"]),
                    "right_date": str(b["th_date"]),
                    "left_sensor": a["th_sensor"],
                    "right_sensor": b["th_sensor"],
                    "cross_sensor": families[i] != families[j],
                    "dt_days": abs((db - da).days),
                    "doy_diff": abs(da.dayofyear - db.dayofyear),
                    "params_hash": params_hash,
                    **report,
                }
            )
        columns = list(SIMILARITY_ID_COLUMNS) + list(sim_raster.REPORT_KEYS) + ["score"]
        new = pd.DataFrame(rows, columns=columns[:-1])

        # The whole run is rescored on every merge (score is a percentile).
        run = sim_trace.merge_run(doc, params_hash, fingerprint, new)
        if run.empty:
            run = pd.DataFrame(columns=columns)
        wanted = {(keys[i], keys[j]) for i, j in idx}
        in_request = np.array(
            [k in wanted for k in zip(run["left_key"], run["right_key"])], dtype=bool
        )
        result = (
            run[in_request]
            .reindex(columns=columns)
            .sort_values(["left_date", "right_date"], ignore_index=True)
        )
        result["cross_sensor"] = result["cross_sensor"].astype(bool)

        if save:  # also when all reused: the run becomes the latest (seq)
            if sim_trace.write_trace(path, doc) is None:
                logger.warning(f"Could not write thumb similarity to {path}")
            else:
                logger.file(f"Thumb similarity written: {path}")
        if plots and not result.empty:
            self._write_similarity_figures(pz_name, t, result, formats)
        return result

    def load_thumbs_similarity(
        self,
        pz_name: str,
        image_type: str = IMAGE_TYPE_OPTICAL,
        params_hash: str | None = None,
    ) -> pd.DataFrame:
        """Read one run of the similarity trace back, without recomputing.

        What :meth:`explore_similarity_pairs` and
        ``update_pairs(strategy="similarity")`` read.

        Args:
            pz_name: Processing zone.
            image_type: Thumb type.
            params_hash: Run to read; ``None`` → the most recently updated run.

        Returns:
            The run's couples, with ``score``.

        Raises:
            FileNotFoundError: No trace yet — run :meth:`compute_thumbs_similarity`.
            KeyError: No such run in the trace.
        """
        from geomulticorr.similarity import trace as sim_trace

        t = normalise_image_type(image_type)
        path = self._similarity_trace_path(pz_name, t)
        if not path.exists():
            raise FileNotFoundError(
                f"No similarity trace for '{pz_name}' ({t}) at {path}. "
                f"Run session.compute_thumbs_similarity({pz_name!r}, ...) first."
            )
        frame = sim_trace.run_frame(sim_trace.read_trace(path), params_hash)
        if "cross_sensor" in frame.columns:
            frame["cross_sensor"] = frame["cross_sensor"].astype(bool)
        return frame.sort_values(["left_date", "right_date"], ignore_index=True)

    def _similarity_outcomes(
        self, pz_name: str, image_type: str = IMAGE_TYPE_OPTICAL
    ) -> pd.DataFrame:
        """What ASP achieved, per correlated thumb couple, from the pair stats JSON.

        Keyed by the **sorted pair of thumb stems** parsed from each pair key
        (``couple_key_from_pair_name``), the same key the similarity rows carry;
        pairs whose key predates the ``__`` format fall back on
        ``<pz>_<date>_<sensor>``. Forward and backward pairs are averaged.

        Reads the stats JSON (the source the Pairs-layer stats columns are
        synced from), so it works whether or not those columns exist yet; a
        pair with no JSON contributes nothing.

        Returns:
            Indexed by couple: ``cc_median``, ``cc_q50`` (fraction of CC ≥ 0.5)
            and ``nmad_m`` (``hypot`` of the corrected EW/NS NMAD on stable
            ground, metres; NaN until corrections ran). Empty when nothing is
            correlated.
        """
        import json

        from geomulticorr.similarity.validation import couple_key_from_pair_name

        t = normalise_image_type(image_type)
        cols = ["cc_median", "cc_q50", "nmad_m"]
        pairs = self._pairs
        if pairs is None or len(pairs) == 0:
            return pd.DataFrame(columns=cols)
        pairs = pairs[
            (pairs["pa_pz_name"] == pz_name) & (type_column(pairs, "pa_type") == t)
        ]
        root = self.pair_dir(pz_name, t)
        rows = []
        for _, p in pairs.iterrows():
            name = pathlib.Path(str(p["pa_path"])).name
            try:
                js = json.loads((root / name / f"{name}.json").read_text())
            except (OSError, ValueError):
                continue
            couple = couple_key_from_pair_name(name) or tuple(
                sorted(
                    (
                        f"{pz_name}_{p['pa_left_date']}_{p['pa_left_sensor']}",
                        f"{pz_name}_{p['pa_right_date']}_{p['pa_right_sensor']}",
                    )
                )
            )
            cc = js.get("raw_corr_stats", {}).get("cc", {})
            fin = js.get("final_corrected_stats", {})
            nmad = np.nan
            if "ew" in fin and "ns" in fin:
                nmad = float(np.hypot(fin["ew"].get("nmad"), fin["ns"].get("nmad")))
            rows.append(
                {
                    "couple": couple,
                    "cc_median": cc.get("median", np.nan),
                    "cc_q50": cc.get("cc_quality_gte_050", np.nan),
                    "nmad_m": nmad,
                }
            )
        if not rows:
            return pd.DataFrame(columns=cols)
        frame = pd.DataFrame(rows)
        frame[cols] = frame[cols].apply(pd.to_numeric, errors="coerce")
        return frame.groupby("couple")[cols].mean()

    def similarity_validation(
        self,
        pz_name: str,
        image_type: str = IMAGE_TYPE_OPTICAL,
        sim: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compare similarity with what ASP achieved on the correlated couples.

        Args:
            pz_name: Processing zone.
            image_type: Thumb type.
            sim: A similarity frame; default the latest stored run.

        Returns:
            ``(matched, table)``: the similarity rows of the correlated couples
            with their outcomes (``cc_median``, ``cc_q50``, ``nmad_m``,
            ``nmad_px``), and the Spearman ρ of every index against every
            outcome (empty below
            :data:`~geomulticorr.similarity.validation.MIN_VALIDATION_COUPLES`
            matched couples).
        """
        from geomulticorr.similarity import validation as sim_val

        if sim is None:
            sim = self.load_thumbs_similarity(pz_name, image_type)
        outcomes = self._similarity_outcomes(pz_name, image_type)
        if outcomes.empty:
            return pd.DataFrame(), pd.DataFrame()
        matched = sim_val.match_outcomes(sim, outcomes)
        if len(matched) < sim_val.MIN_VALIDATION_COUPLES:
            return matched, pd.DataFrame()
        return matched, sim_val.spearman_table(matched)

    def _write_similarity_figures(
        self,
        pz_name: str,
        image_type: str,
        frame: pd.DataFrame,
        formats: Sequence[str] = ("png",),
    ) -> list[pathlib.Path]:
        """Write the dataset overview of a similarity run. Never raises.

        ``<pzone>/figures/similarity/<pz>[_<type>]_similarity_{matrices,
        by_sensor,validation}.<fmt>``. The validation figure needs at least
        :data:`~geomulticorr.similarity.validation.MIN_VALIDATION_COUPLES`
        correlated couples; below that it is skipped with one info line.
        File names carry no parameter: re-running refreshes the same files.
        """
        import matplotlib.pyplot as plt

        from geomulticorr.similarity import validation as sim_val
        from geomulticorr.utils import gmc_functions as gmc_fn

        formats = [f for f in formats if f in ("png", "pdf", "svg", "jpg")]
        suffix = "" if image_type == IMAGE_TYPE_OPTICAL else f"_{image_type}"
        stem = f"{pz_name}{suffix}_similarity"
        out_dir = self.pz_dir(pz_name, PZ_KIND_FIGURES) / "similarity"
        written: list[pathlib.Path] = []

        def _save(build, name):
            fig = None
            try:
                fig = build()
                if isinstance(fig, tuple):
                    fig = fig[0]
                out_dir.mkdir(parents=True, exist_ok=True)
                for fmt in formats:
                    path = out_dir / f"{stem}_{name}.{fmt}"
                    fig.savefig(path, dpi=120)
                    written.append(path)
            except Exception as exc:  # a figure must never cost the computation
                logger.warning(f"Could not write similarity figure '{name}': {exc}")
            finally:
                if fig is not None:
                    plt.close(fig)

        title = f"{pz_name} — {len(frame)} couples"
        _save(
            lambda: gmc_fn.plot_similarity_matrices(frame, fig_name=title), "matrices"
        )
        _save(
            lambda: gmc_fn.plot_similarity_by_sensor(frame, fig_name=title),
            "by_sensor",
        )

        matched, table = self.similarity_validation(pz_name, image_type, frame)
        if len(matched) >= sim_val.MIN_VALIDATION_COUPLES:
            _save(
                lambda: gmc_fn.plot_similarity_validation(table, n=len(matched)),
                "validation",
            )
        else:
            logger.info(
                f"Similarity validation skipped for '{pz_name}': {len(matched)} "
                f"correlated couple(s), {sim_val.MIN_VALIDATION_COUPLES} needed. "
                f"Re-run compute_thumbs_similarity after correlation and corrections."
            )
        if written:
            logger.file(f"Similarity figures written to {out_dir}")
        return written

    #: ``**defaults`` accepted by :meth:`explore_similarity_pairs`; with
    #: ``strategy``, ``pz_name`` and ``image_type`` they are exactly the
    #: ``update_pairs`` keywords the explorer stashes.
    _SIMILARITY_EXPLORER_DEFAULTS = frozenset(
        {
            "similarity_metric",
            "min_similarity",
            "per_image",
            "cross_sensor_only",
            "max_dt_days",
            "min_dt_days",
            "both_directions",
            "sensor_filter",
        }
    )

    @staticmethod
    def _similarity_sensor_filter(sim: pd.DataFrame, sensor_filter) -> pd.DataFrame:
        """Couples whose two thumb names both contain *sensor_filter* (any case)."""
        if not sensor_filter:
            return sim
        f = str(sensor_filter).lower()
        keep = sim["left_key"].str.lower().str.contains(f, regex=False) & sim[
            "right_key"
        ].str.lower().str.contains(f, regex=False)
        return sim[keep]

    def explore_similarity_pairs(
        self,
        pz_name: str,
        image_type: str = IMAGE_TYPE_OPTICAL,
        view: str = "matrix",
        interactive: bool = True,
        params_hash: str | None = None,
        **defaults,
    ):
        """Interactively choose couples from the similarity trace, then commit them.

        Reads the run written by :meth:`compute_thumbs_similarity` **once**; no
        raster is opened and no ``Thumb`` is built, so every control change is
        a pandas filter over at most n(n−1)/2 rows plus a figure redraw —
        milliseconds even for hundreds of thumbs.

        Controls: the metric (``score`` or a raw index) and its threshold
        (``>=``, or ``<=`` for ``trial_sigma_px``), cross-sensor only, a cap
        per image (0 = none; per minority-sensor image when cross-sensor),
        Δt window (duration strings, ``"2Y"``), direction, and the view:

        ==========  ==============================================================
        ``matrix``  thumb × thumb heatmap of the metric, selected couples outlined
        ``network`` the selected pairs on the timeline (coloured by sensor)
        ==========  ==============================================================

        The **Save figure** button writes the view on screen to
        ``<pzone>/figures/similarity/`` in the chosen formats (``html`` via
        plotly, ``png``/``jpg``/``pdf``/``svg`` via matplotlib), named after the
        selection; :meth:`save_similarity_figure` is the scripted equivalent.

        The current selection is stashed on ``self._last_similarity_params`` as
        ``update_pairs`` keywords, so committing it is one line::

            session.explore_similarity_pairs("PasDeLours")
            session.update_pairs(**session._last_similarity_params, append=True)

        ``interactive=False`` skips the widgets and returns
        ``(selected_frame, fig)`` for the arguments given (stashing the same
        way); it imports no ipywidgets.

        :param pz_name: Processing zone.
        :param image_type: Thumb type.
        :param view: Initial view (see above).
        :param interactive: Build the widget (default) or return results.
        :param params_hash: Similarity run to read; default the latest.
        :param defaults: Initial values / actual parameters:
            ``similarity_metric`` (``"score"``), ``min_similarity``,
            ``per_image``, ``cross_sensor_only``, ``max_dt_days``,
            ``min_dt_days``, ``both_directions`` (``True``), ``sensor_filter``.
        :returns: An ``ipywidgets.VBox`` when *interactive* (returned, not
            displayed: Jupyter shows the value of the last expression, and
            displaying it here too drew the explorer twice), else
            ``(selected_frame, fig)``.
        """
        from geomulticorr.similarity import selection as sim_selection
        from geomulticorr.utils import _similarity_plotly as sim_plotly

        if view not in sim_plotly.SIMILARITY_VIEWS:
            raise ValueError(
                f"Unknown view {view!r}. Valid options: {list(sim_plotly.SIMILARITY_VIEWS)}."
            )
        unknown = set(defaults) - self._SIMILARITY_EXPLORER_DEFAULTS
        if unknown:
            raise TypeError(
                f"Unexpected argument(s) {sorted(unknown)}. "
                f"Valid options: {sorted(self._SIMILARITY_EXPLORER_DEFAULTS)}."
            )
        t = normalise_image_type(image_type)
        sim = self.load_thumbs_similarity(pz_name, t, params_hash)
        metrics = [m for m in sim_selection.METRIC_HIGHER_IS_BETTER if m in sim.columns]
        metric0 = defaults.get("similarity_metric", "score")
        if metric0 not in metrics:
            raise ValueError(
                f"similarity_metric must be one of {metrics}, got {metric0!r}"
            )

        def _select(metric, threshold, cross, per_image, max_dt, min_dt, sfilter):
            sub = self._similarity_sensor_filter(sim, sfilter)
            # Resolved as a module attribute so tests can count recomputes.
            return sim_selection.select_couples(
                sub,
                metric=metric,
                threshold=threshold,
                cross_sensor_only=cross,
                per_image=per_image or None,
                max_dt_days=max_dt,
                min_dt_days=min_dt,
            )

        def _stash(metric, threshold, cross, per_image, max_dt, min_dt, both, sfilter):
            self._last_similarity_params = {
                "strategy": "similarity",
                "pz_name": pz_name,
                "image_type": t,
                "similarity_metric": metric,
                "min_similarity": None if threshold is None else float(threshold),
                "per_image": per_image or None,
                "cross_sensor_only": bool(cross),
                "max_dt_days": max_dt,
                "min_dt_days": min_dt,
                "both_directions": bool(both),
                "sensor_filter": sfilter or None,
            }

        def _summary(selected, both, cross) -> str:
            n = len(selected)
            text = (
                f"{n} couple(s) selected of {len(sim)} → "
                f"{n * (2 if both else 1)} pair(s)"
            )
            if cross and n:
                cov = sim_selection.bridge_coverage(sim, selected)
                lonely = cov[cov["n_bridges"] == 0]["anchor_key"].tolist()
                text += f"; {len(cov) - len(lonely)}/{len(cov)} anchor image(s) bridged"
                if lonely:
                    text += " — no bridge: " + ", ".join(
                        k.split("_", 1)[-1] for k in lonely
                    )
            return text

        def _title(metric, threshold) -> str:
            op = ">=" if sim_selection.METRIC_HIGHER_IS_BETTER[metric] else "<="
            thr = "any" if threshold is None else f"{op} {threshold:.3g}"
            return f"{pz_name} — {metric} {thr}"

        # ---- headless --------------------------------------------------------
        if not interactive:
            args = (
                metric0,
                defaults.get("min_similarity"),
                bool(defaults.get("cross_sensor_only", False)),
                defaults.get("per_image"),
                _parse_dt_days(defaults.get("max_dt_days")),
                _parse_dt_days(defaults.get("min_dt_days")),
                defaults.get("sensor_filter"),
            )
            both = bool(defaults.get("both_directions", True))
            selected = _select(*args)
            _stash(*args[:6], both, args[6])
            fig = sim_plotly.figure_similarity_view(
                view,
                sim,
                selected,
                metric=metric0,
                both_directions=both,
                title=_title(metric0, args[1]),
            )
            logger.info(_summary(selected, both, args[2]))
            return selected, fig

        # ---- widgets ---------------------------------------------------------
        import ipywidgets as widgets
        from IPython.display import clear_output, display

        from geomulticorr.utils import _similarity_export as sim_export
        from geomulticorr.utils._pairs_export import FIGURE_FORMATS

        def _range(metric):
            v = pd.to_numeric(sim[metric], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
            if metric == "score":
                lo, hi = 0.0, 1.0
            if hi <= lo:
                hi = lo + 1e-6
            return lo, hi

        def _permissive(metric):
            lo, hi = _range(metric)
            return lo if sim_selection.METRIC_HIGHER_IS_BETTER[metric] else hi

        lo, hi = _range(metric0)
        thr0 = defaults.get("min_similarity")
        w_metric = widgets.Dropdown(
            options=metrics, value=metric0, description="metric"
        )
        w_thr = widgets.FloatSlider(
            value=_permissive(metric0) if thr0 is None else min(max(thr0, lo), hi),
            min=lo,
            max=hi,
            step=(hi - lo) / 100,
            description="threshold",
            readout_format=".3f",
            continuous_update=False,
        )
        w_cross = widgets.Checkbox(
            value=bool(defaults.get("cross_sensor_only", False)),
            description="cross-sensor only",
        )
        n_thumbs = len(sim_selection.thumb_order(sim))
        w_per = widgets.IntSlider(
            value=int(defaults.get("per_image") or 0),
            min=0,
            max=max(n_thumbs - 1, 1),
            description="per image",
            continuous_update=False,
        )
        w_max_dt = widgets.Text(
            value=str(defaults.get("max_dt_days") or ""),
            description="max Δt",
            placeholder="e.g. 2Y, 400D",
        )
        w_min_dt = widgets.Text(
            value=str(defaults.get("min_dt_days") or ""),
            description="min Δt",
            placeholder="e.g. 30D",
        )
        w_both = widgets.Checkbox(
            value=bool(defaults.get("both_directions", True)),
            description="both directions",
        )
        w_view = widgets.Dropdown(
            options=[(label, k) for k, label in sim_plotly.SIMILARITY_VIEWS.items()],
            value=view,
            description="view",
        )
        w_formats = widgets.SelectMultiple(
            options=list(FIGURE_FORMATS),
            value=("html", "png"),
            rows=len(FIGURE_FORMATS),
            description="formats",
        )
        b_save = widgets.Button(description="Save figure", icon="save")
        status = widgets.HTML()
        save_status = widgets.HTML()
        summary = widgets.HTML()
        out = widgets.Output()
        state = {"syncing": False, "dt": (None, None), "selected": None}

        def _read_dt():
            try:
                dts = (
                    _parse_dt_days(w_max_dt.value.strip() or None),
                    _parse_dt_days(w_min_dt.value.strip() or None),
                )
            except ValueError as exc:
                status.value = f"<span style='color:#c0392b'>{exc}</span>"
                return state["dt"]
            status.value = ""
            state["dt"] = dts
            return dts

        def _current():
            max_dt, min_dt = _read_dt()
            thr = w_thr.value
            if np.isclose(thr, _permissive(w_metric.value)):
                thr = None
            return (
                w_metric.value,
                thr,
                w_cross.value,
                w_per.value,
                max_dt,
                min_dt,
                defaults.get("sensor_filter"),
            )

        def _draw():
            args = _current()
            fig = sim_plotly.figure_similarity_view(
                w_view.value,
                sim,
                state["selected"],
                metric=args[0],
                both_directions=w_both.value,
                title=_title(args[0], args[1]),
            )
            with out:
                clear_output(wait=True)
                display(fig)

        def _recompute(_change=None):
            if state["syncing"]:
                return
            args = _current()
            state["selected"] = _select(*args)
            _stash(*args[:6], w_both.value, args[6])
            summary.value = _summary(state["selected"], w_both.value, args[2])
            _draw()

        def _on_metric(change):
            if state["syncing"]:
                return
            state["syncing"] = True
            try:
                lo, hi = _range(change["new"])
                # widen first so neither bound clamps the value mid-update
                w_thr.min, w_thr.max = min(lo, w_thr.min), max(hi, w_thr.max)
                w_thr.value = _permissive(change["new"])
                w_thr.min, w_thr.max, w_thr.step = lo, hi, (hi - lo) / 100
            finally:
                state["syncing"] = False
            _recompute()

        def _on_view(_change):
            if not state["syncing"]:
                _draw()

        w_metric.observe(_on_metric, names="value")
        for w in (w_thr, w_cross, w_per, w_both):
            w.observe(_recompute, names="value")
        for w in (w_max_dt, w_min_dt):
            w.observe(_recompute, names="value")

        def _on_save(_button):
            # Never raise: an exception in a widget callback only reaches the
            # kernel log, which VSCode hides. Saving draws what is selected
            # already; it does not recompute the selection.
            b_save.disabled = True
            save_status.value = "<i>saving…</i>"
            try:
                params = dict(self._last_similarity_params)
                paths = sim_export.save_similarity_figure(
                    sim,
                    state["selected"],
                    self._similarity_figures_dir(pz_name),
                    view=w_view.value,
                    formats=tuple(w_formats.value) or ("html",),
                    stem=sim_export.similarity_figure_stem(w_view.value, **params),
                    title=_title(params["similarity_metric"], params["min_similarity"]),
                    metric=params["similarity_metric"],
                    both_directions=w_both.value,
                    pz_name=pz_name,
                )
                names = " ".join(f"<code>{p.name}</code>" for p in paths.values())
                folder = next(iter(paths.values())).parent
                save_status.value = (
                    f"<span style='color:#2a7'>✔ saved to</span> "
                    f"<code>{folder}</code><br>{names}"
                )
            except Exception as exc:
                logger.error(f"save_similarity_figure failed: {exc!r}")
                save_status.value = (
                    f"<span style='color:#c33'>✘ save failed — "
                    f"{type(exc).__name__}: {exc}</span>"
                )
            finally:
                b_save.disabled = False

        w_view.observe(_on_view, names="value")
        b_save.on_click(_on_save)

        _recompute()
        box = widgets.VBox(
            [
                widgets.HBox([w_metric, w_thr, w_view]),
                widgets.HBox([w_cross, w_per, w_both]),
                widgets.HBox([w_max_dt, w_min_dt]),
                status,
                summary,
                widgets.HTML(
                    "Commit with <code>session.update_pairs("
                    "**session._last_similarity_params, append=True)</code>"
                ),
                out,
                save_status,
                widgets.HBox([w_formats, b_save]),
            ]
        )
        return box

    def _similarity_figures_dir(self, pz_name: str) -> pathlib.Path:
        """``<pzone>/figures/similarity/`` (project-wide figures in legacy)."""
        return self.pz_dir(pz_name, PZ_KIND_FIGURES) / "similarity"

    def save_similarity_figure(
        self,
        view: str = "matrix",
        pz_name: str | None = None,
        *,
        formats: str | Sequence[str] = ("html", "png"),
        stem: str | None = None,
        title: str = "",
        overwrite: bool = True,
        dpi: int = 300,
        plotlyjs: bool | str = True,
        params_hash: str | None = None,
        **selection,
    ) -> dict:
        """Save a view of a similarity selection — the explorer's button, scripted.

        With no *selection* keywords, the last selection of
        :meth:`explore_similarity_pairs` (``self._last_similarity_params``) is
        drawn; otherwise *selection* takes the explorer's keywords
        (``similarity_metric``, ``min_similarity``, ``per_image``,
        ``cross_sensor_only``, ``max_dt_days``, ``min_dt_days``,
        ``both_directions``, ``sensor_filter``, ``image_type``).
        Files go to ``<pzone>/figures/similarity/``, named after the selection
        (:func:`~geomulticorr.utils._similarity_export.similarity_figure_stem`)::

            session.explore_similarity_pairs("PasDeLours", cross_sensor_only=True, per_image=2)
            session.save_similarity_figure("network", formats=("png", "pdf"))

        :param view: ``"matrix"`` or ``"network"``.
        :param pz_name: Pzone; default the one of the stashed selection.
        :param formats: Any of ``html``, ``png``, ``jpg``, ``pdf``, ``svg``.
        :param stem: File stem; derived from the selection when omitted.
        :param title: Figure title; derived from the selection when omitted.
        :param overwrite: Replace a file of the same name (default).
        :param dpi: Raster resolution of the matplotlib output.
        :param plotlyjs: ``True`` inlines plotly.js in the html (offline).
        :param params_hash: Similarity run to read; default the latest.
        :returns: ``{format: Path}`` for every file written.
        :raises ValueError: Neither a selection nor a pzone to take one from.
        """
        from geomulticorr.similarity.selection import METRIC_HIGHER_IS_BETTER
        from geomulticorr.utils import _similarity_export as sim_export

        stash = dict(getattr(self, "_last_similarity_params", None) or {})
        if not selection:
            selection = {
                k: v
                for k, v in stash.items()
                if k in self._SIMILARITY_EXPLORER_DEFAULTS or k == "image_type"
            }
            pz_name = pz_name or stash.get("pz_name")
        if not pz_name:
            raise ValueError(
                "pz_name is needed: no selection from explore_similarity_pairs yet"
            )
        image_type = selection.pop("image_type", IMAGE_TYPE_OPTICAL)
        selected, _ = self.explore_similarity_pairs(
            pz_name,
            image_type,
            view=view,
            interactive=False,
            params_hash=params_hash,
            **selection,
        )
        params = self._last_similarity_params
        metric, threshold = params["similarity_metric"], params["min_similarity"]
        op = ">=" if METRIC_HIGHER_IS_BETTER[metric] else "<="
        thr = "any" if threshold is None else f"{op} {threshold:.3g}"
        return sim_export.save_similarity_figure(
            self.load_thumbs_similarity(pz_name, image_type, params_hash),
            selected,
            self._similarity_figures_dir(pz_name),
            view=view,
            formats=formats,
            stem=stem or sim_export.similarity_figure_stem(view, **params),
            title=title or f"{pz_name} — {metric} {thr}",
            metric=metric,
            both_directions=params["both_directions"],
            pz_name=pz_name,
            dpi=dpi,
            plotlyjs=plotlyjs,
            overwrite=overwrite,
        )

    @staticmethod
    def _protomap_layer(image_type: str | Sequence[str] = "opt") -> str:
        """Geodatabase layer holding the raster-bank map of *image_type*.

        ``"opt"`` keeps the historical ``Protomap`` name, so existing projects
        read unchanged; ``"dem"`` gets ``Protomap_dem``, so mapping a DEM bank no
        longer overwrites the optical map.
        """
        types = (
            {str(image_type).lower()}
            if isinstance(image_type, str)
            else {str(t).lower() for t in image_type}
        )
        if types == {"dem"}:
            return "Protomap_dem"
        if types <= {"opt"}:
            return "Protomap"
        raise ValueError(
            f"Unsupported raster-bank image_type {image_type!r}; use 'opt' or 'dem' "
            f"(hillshade/slope are derived from 'dem' by sieve_bulk)"
        )

    def get_georasters_map(
        self, image_type: str | list[str] = "opt", suffix: str = ""
    ) -> gpd.GeoDataFrame:
        layername = self._protomap_layer(image_type)
        try:
            return gpd.read_file(self.path_geodb, layer=layername, engine="pyogrio")
        except ValueError:
            logger.error(
                f"No georasters map named {layername} in the session geodatabase"
            )
            return None

    # ------------------ END GETTERS ------------------#
    ###################################################

    ###################################################
    # -------------------- SETTERS --------------------#
    def insert_pzone(
        self,
        coordinates: list[tuple[float, float]] | Polygon,
        pz_name: str,
        pz_shortname: str,
        crs: str | int | None = None,
    ) -> gmc_pzone.Pzone:
        """Insert a new processing zone in the session geodatabase.

        Args:
            coordinates (list[tuple[float, float]] | Polygon): List of (x, y) coordinate tuples
                defining the polygon boundary, or a Shapely Polygon directly.
            pz_name (str): The name of the new processing zone.
            pz_shortname (str): Short name / abbreviation for the processing zone.
            crs (str | int | None, optional): CRS of the input coordinates as EPSG code or string.
                Defaults to the session CRS.
        Returns:
            gmc_pzone.Pzone: The newly created processing zone object.
        Raises:
            ValueError: If a pzone with the same name already exists.
        """
        if pz_name in self.pz_names:
            raise ValueError(f"Pzone '{pz_name}' already exists in the session.")

        polygon = (
            coordinates if isinstance(coordinates, Polygon) else Polygon(coordinates)
        )

        # Read current Pzones layer and build new row
        gdf = gpd.read_file(self.path_geodb, layer="Pzones", engine="pyogrio")
        row = {col: None for col in gdf.columns}
        row["pz_name"] = pz_name
        row["pz_shortname"] = pz_shortname
        row["geometry"] = polygon

        input_crs = crs if crs is not None else self.epsg
        new_row = gpd.GeoDataFrame(
            [row], crs=f"EPSG:{input_crs}" if isinstance(input_crs, int) else input_crs
        )

        if gdf.crs is not None and new_row.crs != gdf.crs:
            new_row = new_row.to_crs(gdf.crs)

        gdf = pd.concat([gdf, new_row], ignore_index=True)
        gdf.to_file(self.path_geodb, layer="Pzones", engine="pyogrio")

        # Update session state
        self._pzones = gdf.to_crs(epsg=self.epsg)
        self.pz_names = list(self._pzones.pz_name.unique())

        # Create pzone folder structure per layout
        if self._legacy_layout:
            pz_raster_dir = pathlib.Path(self.path_raster_data) / pz_name
            pz_raster_dir.mkdir(parents=True, exist_ok=True)
            logger.success(f"Pzone '{pz_name}' inserted (legacy layout).")
        else:
            pz_root = self.path_root / pz_name
            pz_root.mkdir(parents=True, exist_ok=True)
            for subdir_kind in NEW_LAYOUT_PZ_SUBDIRS:
                (pz_root / subdir_kind).mkdir(parents=True, exist_ok=True)
            logger.success(
                f"Pzone '{pz_name}' inserted with full "
                f"{len(NEW_LAYOUT_PZ_SUBDIRS)}-subfolder structure."
            )

        return gmc_pzone.Pzone(pz_name, self)

    def insert_pzone_from_file(
        self,
        source: str | pathlib.Path | gu.Vector | gu.Raster,
        pz_name: str,
        pz_shortname: str,
        layer: str | None = None,
        index: int = 0,
        dissolve: bool = False,
    ) -> gmc_pzone.Pzone:
        """Insert a processing zone from an existing AOI instead of drawing it by hand.

        The *source* is normalized through geoutils so the AOI stays consistent with the
        rest of the package: a raster contributes its footprint, a vector its geometry,
        and everything is reprojected to the session CRS via ``gu.Vector.reproject`` before
        being handed to :meth:`insert_pzone`.

        Args:
            source (str | pathlib.Path | gu.Vector | gu.Raster): AOI to insert. May be a
                vector file path (``.gpkg``/``.shp``/``.geojson`` ...), a raster file path
                (``.tif``/``.tiff`` — its footprint is used), a ``gu.Vector`` object, or a
                ``gu.Raster`` object.
            pz_name (str): The name of the new processing zone.
            pz_shortname (str): Short name / abbreviation for the processing zone.
            layer (str | None, optional): Layer name to read when *source* is a multi-layer
                vector file. Defaults to None (first/only layer).
            index (int, optional): Feature index to use when the AOI has several features and
                *dissolve* is False. Defaults to 0.
            dissolve (bool, optional): Merge all features into a single polygon
                (``union_all``) instead of picking one. Defaults to False.
        Returns:
            gmc_pzone.Pzone: The newly created processing zone object.
        Raises:
            TypeError: If *source* is not a supported type.
            ValueError: If the AOI is empty, or resolves to a multi-part ``MultiPolygon``
                without *dissolve*.
            RuntimeError: If the pzone name / shortname were not persisted in the geodatabase.
        """
        # --- Normalize the input into a session-CRS gu.Vector ---
        pz_vec = self._normalize_vector_source(source, layer=layer)

        # --- Resolve to a single Polygon ---
        if dissolve or len(pz_vec.ds) > 1:
            geom = pz_vec.ds.geometry.union_all()
        else:
            geom = pz_vec.ds.geometry.iloc[index]

        if isinstance(geom, MultiPolygon):
            parts = list(geom.geoms)
            if len(parts) == 1:
                geom = parts[0]
            else:
                raise ValueError(
                    f"AOI resolves to a MultiPolygon with {len(parts)} parts. "
                    "Pass dissolve=True to merge them, or index= to pick one feature."
                )

        # --- Delegate to insert_pzone (already in session CRS -> no extra reprojection) ---
        pz = self.insert_pzone(geom, pz_name, pz_shortname, crs=self.epsg)

        # --- Assure pz_name / pz_shortname were persisted in the geodatabase ---
        saved = gpd.read_file(self.path_geodb, layer="Pzones", engine="pyogrio")
        hit = saved[saved.pz_name == pz_name]
        if hit.empty or hit.iloc[0].pz_shortname != pz_shortname:
            raise RuntimeError(
                f"Pzone '{pz_name}' was not persisted with its name/shortname in the geodatabase."
            )
        logger.success(
            f"Pzone '{pz_name}' ({pz_shortname}) stored in geodatabase from {source!r}."
        )
        return pz

    # * Works properly
    def update_vector_data_session(self) -> None:
        """
        Update all vector layers in the geodatabase.

        Returns:
            None. Reload the current session to update files.
        # TODO: Implement unittests.
        """
        self._spines = gpd.read_file(self.path_geodb, layer="Spines", engine="pyogrio")
        self._pzones = gpd.read_file(self.path_geodb, layer="Pzones", engine="pyogrio")
        self._thumbs = gpd.read_file(self.path_geodb, layer="Thumbs", engine="pyogrio")
        self._geomorphs = gpd.read_file(
            self.path_geodb, layer="Geomorphs", engine="pyogrio"
        )
        self._xzones = gpd.read_file(self.path_geodb, layer="Xzones", engine="pyogrio")
        logger.success("Vector data updated in the session.")
        return None

    # * Works properly
    def update_thumbs(self) -> gmc_thumb.Thumb:
        """Update the Thumbs layer in the geodatabase.

        Returns:
            gmc_thumb.Thumb: Updated Thumbs layer.

        # * Works properly
        # ! TODO: Introduce pairing strategy. To do in pzone.
        # TODO: Implement unittests.
        """
        # Copy the geodatabase before the transaction
        # assert self.copy_geodb()

        # Get 2 version of the Thumbs layer
        old = self._thumbs
        if self._legacy_layout:
            opt_root = pathlib.Path(self.path_raster_data)
            glob_patterns = ["**/opticals/*.tif"]
        else:
            opt_root = self.path_root
            # Optical thumbs, then every DEM-derived type (elevation/<type>/).
            glob_patterns = ["*/optical/*.tif"] + [
                f"*/{ELEVATION_DIRNAME}/{t}/*.tif" for t in ELEVATION_IMAGE_TYPES
            ]

        candidates = [
            target_path
            for pattern in glob_patterns
            for target_path in opt_root.glob(pattern=pattern)
            if gmc_thumb.THUMBNAME_PATTERN.match(target_path.name)
        ]
        new = gpd.GeoDataFrame(
            [gmc_thumb.Thumb(target_path).to_pdserie() for target_path in candidates]
        )

        # Rows written before th_type existed are optical by construction.
        old = old.copy()
        old["th_type"] = type_column(old, "th_type")

        # Comparison.  An empty scan (no thumb on disk yet, or none matching
        # THUMBNAME_PATTERN) has no th_path column to merge on — guard it
        # rather than dying on an opaque KeyError('th_path').
        if new.empty:
            logger.warning(
                f"update_thumbs: no thumb matching <pzone>_YYYY-MM-DD_<sensor>[_<res>m].tif "
                f"found under {opt_root} ({', '.join(glob_patterns)}); pzone names "
                f"may not contain '_'"
            )
            new = gpd.GeoDataFrame(
                columns=list(old.columns), geometry="geometry", crs=old.crs
            )
        common = new.merge(old, on=["th_path"])
        stables = old[old.th_path.isin(common.th_path)]
        addeds = new[~new.th_path.isin(common.th_path)]

        # Push it into the geodatabase
        updated = pd.concat([stables, addeds])
        updated.to_file(self.path_geodb, layer="Thumbs")

        # Update instance
        self._thumbs = updated

        # # Update pairs
        # self.update_pairs()
        return updated

    def register_existing_thumbs(
        self,
        sources: str | pathlib.Path | list[str | pathlib.Path],
        pz_name: str,
        glob_pattern: str = "*.tif",
        move: bool = False,
        overwrite: bool = False,
        register: bool = True,
        print_summary: bool = True,
    ) -> list[pathlib.Path]:
        """Register already-cropped images as thumbs without re-cropping or resampling.

        Use this when your images are already cropped to the pzone AOI and share the
        same grid/resolution. Each source image is copied (or moved) into the pzone's
        optical folder and renamed to the GMC thumb convention ``<pz_name>_YYYY-MM-DD_<sensor>.tif``.
        The acquisition date and sensor are auto-detected from the filename using the
        same detectors as :meth:`map_georasters_bank` (``re_searcher`` + ``sensor_normalize``
        for the sensor, :func:`supported_sensors.extract_acquisition_date` for the date).

        Unlike :meth:`sieve_bulk`, no windowed crop or reprojection is performed —
        pixels are passed through byte-for-byte.

        Args:
            sources: A directory, a single file path, or a list of file/directory
                paths. Directories are expanded with *glob_pattern*.
            pz_name: Target processing zone. Must already be registered and must not
                contain ``_`` (it would break the thumb filename pattern).
            glob_pattern: Pattern used to expand directory sources. Defaults to ``"*.tif"``.
            move: If True, move files instead of copying. Defaults to False (copy).
            overwrite: If True, overwrite an existing thumb of the same name.
                Defaults to False (skip existing).
            register: If True, refresh the Thumbs layer in the geodatabase via
                :meth:`update_thumbs` after copying. Defaults to True.
            print_summary: If True, print a Rich summary table. Defaults to True.

        Returns:
            list[pathlib.Path]: Paths of the thumbs written into the optical folder.

        Raises:
            ValueError: If *pz_name* is unknown / contains ``_``, no images are found,
                or the sensor / date cannot be detected for a source image.
            FileNotFoundError: If a source path does not exist.
        """
        if pz_name not in self.pz_names:
            raise ValueError(
                f"Pzone '{pz_name}' is not registered. Known pzones: {self.pz_names}"
            )
        if "_" in pz_name:
            raise ValueError(
                f"pz_name '{pz_name}' must not contain '_' — it breaks the thumb "
                "filename pattern '<pzone>_YYYY-MM-DD_<sensor>.tif'."
            )

        # Resolve sources into a flat list of files
        raw = sources if isinstance(sources, (list, tuple)) else [sources]
        files: list[pathlib.Path] = []
        for item in raw:
            p = pathlib.Path(item)
            if p.is_dir():
                files.extend(sorted(p.glob(glob_pattern)))
            elif p.is_file():
                files.append(p)
            else:
                raise FileNotFoundError(f"Source path not found: {p}")
        if not files:
            raise ValueError(
                f"No images found in sources (glob_pattern={glob_pattern!r})."
            )

        dest_dir = self.pz_dir(pz_name, PZ_KIND_OPTICAL)
        dest_dir.mkdir(parents=True, exist_ok=True)

        sensor_pattern = gmc_sensors.sensors(gmc_sensors.supported_sensors)
        written: list[pathlib.Path] = []
        summary_rows: list[dict] = []

        for src in files:
            # --- Detect sensor (filename first, then parent directory) ---
            sensor = re_searcher(src.name.lower(), sensor_pattern)
            if sensor == "unknown":
                sensor = re_searcher(src.parent.name.lower(), sensor_pattern)
            sensor_norm = gmc_sensors.sensor_normalize(sensor).get("sensor", "unknown")
            if not sensor_norm or sensor_norm == "unknown":
                raise ValueError(
                    f"Could not detect a supported sensor for '{src.name}'. "
                    "Rename the file/folder to include the sensor name, or register it manually."
                )

            # --- Detect acquisition date ---
            acq_dt = gmc_sensors.extract_acquisition_date(src, sensor=sensor)
            if acq_dt is None:
                raise ValueError(
                    f"Could not detect an acquisition date for '{src.name}'. "
                    "Ensure the date is encoded in the filename or a sidecar XML/DIMAP file."
                )
            date_str = acq_dt.strftime("%Y-%m-%d")

            thumb_name = f"{pz_name}_{date_str}_{sensor_norm}.tif"
            if not gmc_thumb.THUMBNAME_PATTERN.match(thumb_name):
                raise ValueError(
                    f"Generated thumb name '{thumb_name}' does not match the GMC pattern."
                )

            dest_path = dest_dir / thumb_name
            if dest_path.exists() and not overwrite:
                logger.warning(f"Thumb already exists, skipping: {thumb_name}")
                summary_rows.append(
                    {"src": src.name, "thumb": thumb_name, "status": "↷ skipped"}
                )
                written.append(dest_path)
                continue

            if move:
                shutil.move(str(src), str(dest_path))
            else:
                shutil.copy2(str(src), str(dest_path))
            written.append(dest_path)
            summary_rows.append(
                {"src": src.name, "thumb": thumb_name, "status": "✓ copied"}
            )

        # Register the new thumbs in the geodatabase
        if register and written:
            self.update_thumbs()

        if print_summary and summary_rows:
            print_register_thumbs_summary(
                summary_rows, pz_name, print_summary=print_summary
            )

        logger.success(
            f"Registered {len(written)} thumb(s) into pzone '{pz_name}'"
            + (" (geodatabase updated)." if register and written else ".")
        )
        return written

    def validate_all_thumbs(
        self, pzone_filter: list[str] | None = None
    ) -> gpd.GeoDataFrame:
        """Mark all thumbs as valid for correlation (th_valid = '1').

        Args:
            pzone_filter: Optional list of pzone names to restrict validation.
                If None, all thumbs in the project are marked valid.

        Returns:
            Updated Thumbs GeoDataFrame.
        """
        updated = self._thumbs.copy()
        if pzone_filter is not None:
            mask = updated["pz_name"].isin(pzone_filter)
            updated.loc[mask, "th_valid"] = "1"
        else:
            updated["th_valid"] = "1"
        updated.to_file(self.path_geodb, layer="Thumbs")
        self._thumbs = updated
        return updated

    def _sync_pairs_status(self, pairs: list) -> None:
        """Sync pa_status in the geodatabase for the given pairs from filesystem state.

        Called internally after local correlation runs complete.
        For cluster (oarsub) workflows, call this manually after jobs finish.
        """
        if not pairs:
            return
        for pair in pairs:
            fresh_status = pair.get_status()
            mask = self._pairs["pa_path"] == str(pair.pa_path)
            self._pairs.loc[mask, "pa_status"] = fresh_status
        self._pairs.to_file(self.path_geodb, layer="Pairs")
        logger.save(f"Geodatabase updated: pa_status synced for {len(pairs)} pair(s).")

    def backfill_thumb_resolution(
        self,
        pz_name: str = "",
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """Fill the ``th_res`` column for thumbs already in the Thumbs layer.

        ``Thumb.to_pdserie`` records ``th_res`` for every thumb registered from
        now on, and :meth:`update_thumbs` preserves existing rows in full — which
        is why rows written before the column existed keep a missing value
        forever unless they are backfilled once. This is that one-shot pass.

        Resolution is read from each file's header via the memoised
        :meth:`_thumb_resolution`, so a project's handful of unique thumbs costs
        a few milliseconds.

        :param pz_name: Restrict to one pzone; ``""`` for all.
        :param overwrite: Recompute rows that already carry a value. Use this
            after re-sieving thumbs to a different resolution under the same
            filenames, which is the one case a present value can be stale.
        :returns: The updated Thumbs layer.
        """
        thumbs = self._thumbs
        if thumbs is None or len(thumbs) == 0:
            logger.warning("No thumbs registered. Run update_thumbs() first.")
            return thumbs

        if "th_res" not in thumbs.columns:
            thumbs["th_res"] = float("nan")

        mask = (
            thumbs["th_pz_name"] == pz_name
            if pz_name
            else pd.Series(True, index=thumbs.index)
        )
        if not overwrite:
            mask &= ~pd.to_numeric(thumbs["th_res"], errors="coerce").notna()

        targets = thumbs.index[mask]
        if len(targets) == 0:
            logger.info("th_res already set for every selected thumb — nothing to do.")
            return thumbs

        n_failed = 0
        for idx in targets:
            value = self._thumb_resolution(thumbs.at[idx, "th_path"])
            thumbs.at[idx, "th_res"] = value
            if not np.isfinite(value):
                n_failed += 1

        # Force a real float column: the rows read back from the GPKG and the
        # rows written by to_pdserie are concatenated on every update_thumbs, and
        # a mixed dtype would quietly make this an object column.
        thumbs["th_res"] = pd.to_numeric(thumbs["th_res"], errors="coerce").astype(
            "float64"
        )

        self._thumbs = thumbs
        thumbs.to_file(self.path_geodb, layer="Thumbs")
        logger.success(
            f"Backfilled th_res for {len(targets) - n_failed} / {len(targets)} thumb(s)."
        )
        if n_failed:
            logger.warning(f"{n_failed} thumb(s) had an unreadable resolution.")
        return thumbs

    def sync_pairs_stats(
        self,
        metric_map: dict,
        section,
        pairs: list | None = None,
        criterias: str | list[str] = "",
        extra_cols: dict | None = None,
    ) -> gpd.GeoDataFrame:
        """Sync selected per-pair stats (from the stats JSON) into the Pairs layer.

        Reads each pair's stats JSON (``pair.pa_stats_path``), flattens the
        requested values via
        :func:`geomulticorr.stats.stats.resolve_stat_columns`, and writes them as
        columns on the matching Pairs-layer row (matched by ``pa_path``), then
        persists the layer once. Mirrors :meth:`_sync_pairs_status`.

        These columns are a **derived cache**: they are not part of
        :meth:`~geomulticorr.core.pair.Pair.to_pdserie` and are dropped when the
        Pairs layer is rebuilt (:meth:`update_pairs`) — re-run the
        sync afterwards.

        Args:
            metric_map: ``{pairs_column: (stats_layer, stat_key)}`` mapping.
            section: Section selector passed through to ``resolve_stat_columns``
                (a section name like ``"raw_corr_stats"`` or
                ``"final_corrected_stats"``, a ``("correction_stats", "<StepName>")``
                tuple, or a callable).
            pairs: Pairs to sync. Defaults to ``get_pairs(criterias)``.
            criterias: Filter used when *pairs* is not given.
            extra_cols: Optional derived columns ``{column: value_or_callable}``.

        Returns:
            The updated in-memory Pairs GeoDataFrame.
        """
        if pairs is None:
            pairs = self.get_pairs(criterias)

        n = 0
        for pair in pairs:
            try:
                js = gmc_stats.load_pair_stats(pair)
            except FileNotFoundError:
                continue
            mask = self._pairs["pa_path"] == str(pair.pa_path)
            if not mask.any():
                continue
            cols = gmc_stats.resolve_stat_columns(js, metric_map, section, extra_cols)
            for col, val in cols.items():
                self._pairs.loc[mask, col] = val
            n += 1

        if n:
            self._pairs.to_file(self.path_geodb, layer="Pairs")
            synced = list(metric_map) + list(extra_cols or {})
            logger.save(f"Geodatabase updated: synced {synced} for {n} pair(s).")
        return self._pairs

    def sync_raw_stats(
        self,
        pairs: list | None = None,
        criterias: str | list[str] = "",
    ) -> gpd.GeoDataFrame:
        """Sync raw correlation stats (:data:`RAW_STATS_SYNC`) into the Pairs layer.

        Populates ``pa_ew_nmad_raw``, ``pa_ns_nmad_raw`` and ``pa_cc_quality050``
        from each pair's ``raw_corr_stats``. Call after
        :meth:`extract_pairs_raw_displacements` (done automatically unless
        ``sync_geodb=False``), or standalone to (re)populate the columns.
        """
        return self.sync_pairs_stats(
            RAW_STATS_SYNC, "raw_corr_stats", pairs=pairs, criterias=criterias
        )

    def sync_corrected_stats(
        self,
        pairs: list | None = None,
        criterias: str | list[str] = "",
        n_corrections: int | None = None,
    ) -> gpd.GeoDataFrame:
        """Sync final-correction stats (:data:`CORR_STATS_SYNC`) into the Pairs layer.

        Populates ``pa_ew_nmad_corr``/``pa_ns_nmad_corr`` from the canonical
        ``final_corrected_stats`` section (the fully-corrected field, written by
        :func:`~geomulticorr.stats.stats.save_final_corrected_stats`) plus
        ``pa_n_corrections``.

        Args:
            pairs: Pairs to sync. Defaults to ``get_pairs(criterias)``.
            criterias: Filter used when *pairs* is not given.
            n_corrections: Number of corrections applied. When given (the
                auto-wired call from :meth:`apply_pairs_corrections`, where the
                pipeline is fixed for all pairs) it is used directly; otherwise it
                is derived per pair from ``len(correction_stats)``.
        """
        extra_cols = {
            "pa_n_corrections": (
                n_corrections
                if n_corrections is not None
                else (lambda js: len(js.get("correction_stats", {})))
            )
        }
        return self.sync_pairs_stats(
            CORR_STATS_SYNC,
            "final_corrected_stats",
            pairs=pairs,
            criterias=criterias,
            extra_cols=extra_cols,
        )

    def sync_pairs_weights(
        self,
        pairs: list | None = None,
        criterias: str | list[str] = "",
    ) -> gpd.GeoDataFrame:
        """Sync the per-direction inversion weights (``weight`` section) into the Pairs layer.

        Populates ``pa_ew_weight`` / ``pa_ns_weight`` (the EW/NS weights),
        ``pa_weight`` (their per-pair maximum, a single-number summary), and
        ``pa_weight_mode`` (the ``mode``) from each pair's stats JSON ``weight``
        section, written by
        :func:`~geomulticorr.stats.stats.save_pair_weight`. Typically called by
        :meth:`~geomulticorr.inversion.tio_inversion.TIOInversion.write_liste_couple`
        after weights are computed, or standalone to (re)populate the columns.

        Args:
            pairs: Pairs to sync. Defaults to ``get_pairs(criterias)``.
            criterias: Filter used when *pairs* is not given.
        """

        def _ew(js):
            return js.get("weight", {}).get("ew", float("nan"))

        def _ns(js):
            return js.get("weight", {}).get("ns", float("nan"))

        def _max(js):
            w = js.get("weight", {})
            vals = [
                v for v in (w.get("ew"), w.get("ns")) if isinstance(v, (int, float))
            ]
            return max(vals) if vals else float("nan")

        extra_cols = {
            "pa_ew_weight": _ew,
            "pa_ns_weight": _ns,
            "pa_weight": _max,
            "pa_weight_mode": (lambda js: js.get("weight", {}).get("mode")),
        }
        return self.sync_pairs_stats(
            {},
            "weight",
            pairs=pairs,
            criterias=criterias,
            extra_cols=extra_cols,
        )

    #: Strategies :meth:`update_pairs` handles itself; the four classic ones are
    #: delegated to :meth:`Pzone.get_valid_pairs_with_strategy_overview`.
    SESSION_PAIR_STRATEGIES = ("similarity", "manual")

    @gmc_profile("gmc.session.update_pairs")
    def update_pairs(
        self,
        strategy: str = "consecutive",
        max_step: int | None = None,
        max_dt_days: int | str | None = None,
        min_dt_days: int | str | None = None,
        sensor_filter: str | None = None,
        pz_name: str = "",
        append: bool = False,
        image_type: str = IMAGE_TYPE_OPTICAL,
        min_similarity: float | None = None,
        similarity_metric: str = "score",
        per_image: int | None = None,
        cross_sensor_only: bool = False,
        both_directions: bool = True,
        couples=None,
    ) -> gpd.GeoDataFrame:
        """Build pairs and persist them to the geodatabase — the one pairing entry point.

        Three families of strategy:

        - **Date-based** — ``"consecutive"``, ``"step"``, ``"redundancy"``,
          ``"forward-backward"`` (unchanged). With ``min_similarity`` set, pairs
          whose thumb couple scores below it in the similarity trace are dropped
          (couples absent from the trace are dropped too, with a warning).
        - ``"similarity"`` — pairs chosen from the similarity trace of
          :meth:`compute_thumbs_similarity`: every measured couple passing the
          Δt filters, ``sensor_filter``, ``cross_sensor_only``,
          ``similarity_metric >= min_similarity`` and the ``per_image`` cap
          (see :func:`geomulticorr.similarity.selection.select_couples`). This
          is how cross-sensor bridges are formed; the interactive
          :meth:`explore_similarity_pairs` stashes exactly these arguments.
        - ``"manual"`` — the explicit ``couples`` you give.

        A typical cross-sensor study::

            session.update_pairs("redundancy", max_step=2, sensor_filter="planetscope")
            session.update_pairs("redundancy", max_step=1, sensor_filter="spot", append=True)
            session.update_pairs("similarity", cross_sensor_only=True, min_similarity=0.8,
                                 per_image=2, max_dt_days="2Y", append=True)

        Args:
            strategy: One of the above.
            max_step: Maximum index offset between paired thumbs (required for
                ``"step"`` and ``"redundancy"``).
            max_dt_days: Optional upper bound on the date gap. Accepts an integer
                number of days or a duration string: ``'30D'``, ``'6M'``, ``'1Y'``,
                ``'2W'``. Months and years use fixed approximations (30 d/month,
                365 d/year), which is appropriate given the 1-day image resolution.
            min_dt_days: Optional lower bound on the date gap. Same formats as
                ``max_dt_days``. Pairs shorter than this are dropped.
            sensor_filter: Optional sensor name substring to restrict thumbs
                (e.g. ``"spot"``, ``"planetscope"``). Pairs are built only from
                thumbs whose path contains this string.
            pz_name: Restrict to a single pzone (default ``""`` = all pzones).
            append: If ``True``, new pairs are concatenated with the existing
                ``Pairs`` layer (duplicates resolved by ``pa_path``). If ``False``
                (default), the layer is fully replaced.
            image_type: Thumb image type to pair — ``"opt"`` (default),
                ``"dem"``, ``"hillshade"``, ``"mdhillshade"`` or ``"slope"``.
                Filters on the Thumbs layer's ``th_type`` column; pairs never
                cross types. Pass ``append=True`` when adding a second type, or
                the first type's pairs are replaced.
            min_similarity: Threshold on ``similarity_metric`` (``>=``, or
                ``<=`` for a lower-is-better metric such as ``trial_sigma_px``).
                For ``score`` it is a percentile: ``0.8`` keeps couples better
                than 80 % of the run's couples.
            similarity_metric: ``"score"`` or a raw index (see
                ``similarity.selection.METRIC_HIGHER_IS_BETTER``).
            per_image: ``"similarity"`` only — at most this many couples per
                image (per image of the minority sensor with
                ``cross_sensor_only``), best first.
            cross_sensor_only: ``"similarity"`` only — keep couples whose sensor
                families differ.
            both_directions: ``"similarity"`` and ``"manual"`` — also commit
                each reverse pair, as ``"redundancy"`` does. Ignored by the
                date-based strategies, which set directions themselves.
            couples: ``"manual"`` only — ``(left_key, right_key)`` tuples of
                thumb file stems, or a frame with ``left_key`` / ``right_key``
                (a similarity frame or a selection works as is).

        Returns:
            GeoDataFrame of all pairs written to the ``Pairs`` layer.

        Raises:
            ValueError: An argument used with a strategy it does not apply to.
            FileNotFoundError: ``"similarity"`` or ``min_similarity`` without a
                similarity trace; ``"manual"`` with a missing thumb.
        """
        image_type = normalise_image_type(image_type)
        max_dt_days = _parse_dt_days(max_dt_days)
        min_dt_days = _parse_dt_days(min_dt_days)
        if couples is not None and strategy != "manual":
            raise ValueError("couples= is only used with strategy='manual'")
        if strategy not in self.SESSION_PAIR_STRATEGIES and (
            per_image is not None or cross_sensor_only
        ):
            raise ValueError(
                "per_image and cross_sensor_only apply to strategy='similarity'; "
                f"for {strategy!r} use min_similarity to filter by similarity"
            )

        if strategy == "manual":
            if couples is None:
                raise ValueError("strategy='manual' needs couples=[(left, right), ...]")
            new_pairs = self._pairs_from_couples(couples, image_type, both_directions)
            return self._write_pairs_layer(new_pairs, append)

        updated_frames = []
        for pz in self.get_pzones(pz_name):
            logger.info(f"Updating pairs for PZone: '{pz.pz_name}' ({strategy})")
            if strategy == "similarity":
                pairs_gdf = self._similarity_strategy_pairs(
                    pz.pz_name,
                    image_type,
                    metric=similarity_metric,
                    threshold=min_similarity,
                    per_image=per_image,
                    cross_sensor_only=cross_sensor_only,
                    max_dt_days=max_dt_days,
                    min_dt_days=min_dt_days,
                    sensor_filter=sensor_filter,
                    both_directions=both_directions,
                )
            else:
                pairs_gdf = pz.get_valid_pairs_with_strategy_overview(
                    strategy,
                    max_step,
                    max_dt_days,
                    min_dt_days=min_dt_days,
                    sensor_filter=sensor_filter,
                    image_type=image_type,
                )
                if min_similarity is not None and not pairs_gdf.empty:
                    pairs_gdf = self._filter_pairs_by_similarity(
                        pairs_gdf,
                        pz.pz_name,
                        image_type,
                        similarity_metric,
                        min_similarity,
                    )
            if not pairs_gdf.empty:
                updated_frames.append(pairs_gdf)

        if updated_frames:
            new_pairs = pd.concat(updated_frames, ignore_index=True)
            if "geometry" not in new_pairs.columns:
                raise ValueError("No 'geometry' column found in pairs DataFrame.")
            new_pairs = gpd.GeoDataFrame(new_pairs, geometry="geometry", crs=self.epsg)
        else:
            new_pairs = gpd.GeoDataFrame(geometry=[], crs=self.epsg)

        return self._write_pairs_layer(new_pairs, append)

    def _similarity_strategy_pairs(
        self,
        pz_name: str,
        image_type: str,
        *,
        metric: str,
        threshold: float | None,
        per_image: int | None,
        cross_sensor_only: bool,
        max_dt_days: int | None,
        min_dt_days: int | None,
        sensor_filter: str | None,
        both_directions: bool,
    ) -> gpd.GeoDataFrame:
        """Pairs for ``update_pairs(strategy="similarity")`` in one pzone."""
        from geomulticorr.similarity.selection import select_couples

        sim = self._similarity_sensor_filter(
            self.load_thumbs_similarity(pz_name, image_type), sensor_filter
        )
        # Same rule as the date strategies: only validated thumbs are paired.
        th = self._thumbs
        th = th[
            (th["th_pz_name"] == pz_name)
            & (type_column(th, "th_type") == image_type)
            & (th["th_valid"].astype(str) == "1")
        ]
        valid = {pathlib.Path(str(p)).stem for p in th["th_path"]}
        ok = sim["left_key"].isin(valid) & sim["right_key"].isin(valid)
        if (~ok).any():
            logger.warning(
                f"{int((~ok).sum())} couple(s) in '{pz_name}' involve thumbs not "
                f"validated (th_valid != 1) and are skipped; validate them in QGIS "
                f"or with session.validate_all_thumbs()"
            )
        sim = sim[ok]
        chosen = select_couples(
            sim,
            metric=metric,
            threshold=threshold,
            cross_sensor_only=cross_sensor_only,
            per_image=per_image,
            max_dt_days=max_dt_days,
            min_dt_days=min_dt_days,
        )
        logger.info(
            f"Similarity selection in '{pz_name}': {len(chosen)}/{len(sim)} couple(s) "
            f"({metric} {'>=' if threshold is not None else ''}"
            f"{threshold if threshold is not None else 'any'})"
        )
        if chosen.empty:
            return gpd.GeoDataFrame(geometry=[], crs=self.epsg)
        return self._pairs_from_couples(chosen, image_type, both_directions)

    def _filter_pairs_by_similarity(
        self,
        pairs_gdf: pd.DataFrame,
        pz_name: str,
        image_type: str,
        metric: str,
        threshold: float,
    ) -> pd.DataFrame:
        """Drop date-strategy pairs whose couple fails *metric* vs *threshold*."""
        from geomulticorr.similarity.selection import METRIC_HIGHER_IS_BETTER
        from geomulticorr.similarity.validation import couple_key_from_pair_name

        if metric not in METRIC_HIGHER_IS_BETTER:
            raise ValueError(
                f"unknown similarity metric {metric!r}; "
                f"choose from {sorted(METRIC_HIGHER_IS_BETTER)}"
            )
        sim = self.load_thumbs_similarity(pz_name, image_type)
        value = {
            tuple(sorted(k)): v
            for k, v in zip(zip(sim["left_key"], sim["right_key"]), sim[metric])
        }
        up = METRIC_HIGHER_IS_BETTER[metric]
        keep, unmeasured = [], 0
        for p in pairs_gdf["pa_path"]:
            v = value.get(couple_key_from_pair_name(pathlib.Path(str(p)).name))
            if v is None or not np.isfinite(v):
                unmeasured += 1
                keep.append(False)
            else:
                keep.append(v >= threshold if up else v <= threshold)
        if unmeasured:
            logger.warning(
                f"{unmeasured} pair(s) in '{pz_name}' have no similarity measured and "
                f"were dropped; run compute_thumbs_similarity on the same thumbs"
            )
        out = pairs_gdf[np.array(keep, dtype=bool)]
        logger.info(
            f"Similarity filter ({metric} {'>=' if up else '<='} {threshold}): "
            f"{len(out)}/{len(pairs_gdf)} pair(s) kept in '{pz_name}'"
        )
        return out

    def _pairs_from_couples(
        self, couples, image_type: str, both_directions: bool
    ) -> gpd.GeoDataFrame:
        """Build pairs from thumb-stem couples, like a strategy does (``Thumb + Thumb``).

        Thumbs are located through :meth:`thumb_dir` from the pzone that
        prefixes each stem (pzone names hold no ``_``), not via stored paths.
        """
        t = normalise_image_type(image_type)
        if isinstance(couples, pd.DataFrame):
            couples = list(zip(couples["left_key"], couples["right_key"]))
        couples = list(dict.fromkeys(tuple(c) for c in couples))
        if not couples:
            raise ValueError("at least one (left_key, right_key) couple is needed")
        thumbs: dict[str, gmc_thumb.Thumb] = {}

        def _thumb(key: str) -> gmc_thumb.Thumb:
            if key not in thumbs:
                stem = pathlib.Path(str(key)).stem
                folder = self.thumb_dir(stem.split("_", 1)[0], t)
                path = folder / f"{stem}.tif"
                if not path.exists():
                    raise FileNotFoundError(f"No thumb {path.name} in {folder}")
                thumbs[key] = gmc_thumb.Thumb(str(path))
            return thumbs[key]

        rows = []
        for left, right in couples:
            if left == right:
                raise ValueError(f"Couple pairs {left!r} with itself")
            ordered = (
                [(left, right), (right, left)] if both_directions else [(left, right)]
            )
            for a, b in ordered:
                rows.append((_thumb(a) + _thumb(b)).to_pdserie())
        logger.info(
            f"{len(rows)} pair(s) from {len(couples)} couple(s)"
            f"{' (both directions)' if both_directions else ''}"
        )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=self.epsg)

    def _write_pairs_layer(
        self, new_pairs: gpd.GeoDataFrame, append: bool
    ) -> gpd.GeoDataFrame:
        """Write *new_pairs* to the Pairs layer, replacing it or appending to it.

        Appending dedupes on ``pa_path`` (unique across types), so committing a
        pair twice keeps one row. Every strategy of :meth:`update_pairs` goes
        through here, so all commit paths behave identically.
        """
        if append and not self._pairs.empty:
            existing = self._pairs.copy()
            existing["pa_type"] = type_column(existing, "pa_type")
            combined = pd.concat([existing, new_pairs], ignore_index=True)
            combined = combined.drop_duplicates(subset=["pa_path"]).reset_index(
                drop=True
            )
            updated = gpd.GeoDataFrame(combined, geometry="geometry", crs=self.epsg)
        else:
            updated = new_pairs

        updated.to_file(self.path_geodb, layer="Pairs")
        self._pairs = updated
        return updated

    def copy_geodb(self) -> bool:
        """Quickly create a copy of the project geopackage named GMC_geodatabase_backup.gpkg"""
        backup_path = pathlib.Path(self.path_root, "GMC_geodatabase_backup.gpkg")
        shutil.copy2(self.path_geodb, backup_path)
        return pathlib.Path(self.path_root, "GMC_geodatabase_backup.gpkg").exists()

    def migrate_to_new_structure(self, dry_run: bool = False) -> dict:
        """Migrate a legacy-layout project to the new Pzone-centric structure.

        Converts folder hierarchy from ``raster_data_<project>/pz_name/{opticals,…}``
        to ``pz_name/{optical,image_correlation,reference_dem,…}`` and updates
        geodatabase path columns (th_path, pa_path, pa_disparity_f_path).

        Call this only on legacy-layout sessions (raises RuntimeError otherwise).
        It is a one-time, opt-in operation — once migrated, the session's
        _legacy_layout flag is recomputed and may differ on next open.

        Args:
            dry_run: If True, return the move manifest without touching the
                filesystem. Useful for preview before committing.

        Returns:
            Dict with keys:
            - 'moved': list of (src, dest) tuples for files/folders moved
            - 'skipped': list of paths that were not moved (unknown format)
            - 'warnings': list of warning messages

        Raises:
            RuntimeError: If called on a new-layout session (self._legacy_layout is False).
        """
        if not self._legacy_layout:
            raise RuntimeError(
                "migrate_to_new_structure() is only for legacy-layout sessions. "
                "This session is already in the new layout."
            )

        moved: list[tuple[pathlib.Path, pathlib.Path]] = []
        skipped: list[pathlib.Path] = []
        warnings: list[str] = []

        logger.launch(
            "Dry run: analyzing legacy structure..."
            if dry_run
            else "Migrating to new structure..."
        )

        # Step 1: Back up the geodatabase
        if not dry_run:
            if not self.copy_geodb():
                warnings.append("Failed to back up geodatabase; proceeding anyway.")
            else:
                logger.success("Geodatabase backed up to GMC_geodatabase_backup.gpkg")

        # Step 2: Build move manifest for each pzone
        for pz_name in self.pz_names:
            legacy_pz_root = self.path_raster_data / pz_name
            new_pz_root = self.path_root / pz_name

            if not legacy_pz_root.exists():
                warnings.append(f"Legacy pzone folder not found: {legacy_pz_root}")
                continue

            # Create new pzone root (if not already there)
            if not dry_run and not new_pz_root.exists():
                new_pz_root.mkdir(parents=True, exist_ok=True)

            # Map legacy subfolders → new subdirectories
            moves = [
                (legacy_pz_root / "opticals", new_pz_root / PZ_KIND_OPTICAL),
                (
                    legacy_pz_root / "image_correlation",
                    new_pz_root / PZ_KIND_IMAGE_CORRELATION,
                ),
            ]

            # DEM file (if exists)
            dem_file = legacy_pz_root / f"{pz_name}_dem.tif"
            if dem_file.exists():
                moves.append(
                    (dem_file, new_pz_root / PZ_KIND_REFERENCE_DEM / dem_file.name)
                )

            # Moving-areas files (raster and gpkg)
            for suffix in ["_moving-areas_round-0.tif", "_moving-areas_round-0.gpkg"]:
                ma_file = legacy_pz_root / f"{pz_name}{suffix}"
                if ma_file.exists():
                    moves.append((ma_file, new_pz_root / PZ_KIND_VECTOR / ma_file.name))

            # Similarity trace folder (already a folder in legacy)
            moves.append(
                (legacy_pz_root / "similarity", new_pz_root / PZ_KIND_SIMILARITY)
            )

            # Inversion directories (inversion_* → inversion/*)
            for legacy_inv_dir in legacy_pz_root.glob("inversion_*"):
                if legacy_inv_dir.is_dir():
                    inv_name = legacy_inv_dir.name.replace("inversion_", "")
                    new_inv_dir = new_pz_root / PZ_KIND_INVERSION / inv_name
                    moves.append((legacy_inv_dir, new_inv_dir))

            # Execute moves (or just record them if dry_run)
            for src, dst in moves:
                if not src.exists():
                    skipped.append(src)
                    continue

                if dry_run:
                    moved.append((src, dst))
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    moved.append((src, dst))
                    logger.info(
                        f"Moved: {src.relative_to(self.path_root)} → {dst.relative_to(self.path_root)}"
                    )

            # Warn about leftover files in legacy pzone root
            remaining = list(legacy_pz_root.glob("*"))
            if remaining:
                leftover_names = [f.name for f in remaining]
                warnings.append(
                    f"Pzone '{pz_name}': {len(remaining)} item(s) left in legacy root after moves: "
                    f"{', '.join(leftover_names)}. Review and remove manually if needed."
                )

        # Step 3: Migrate figures folder (project-wide)
        legacy_figures = self.path_root / f"figures_{self.project_name}"
        new_figures = self.path_root / "figures"

        if legacy_figures.exists() and not new_figures.exists():
            if not dry_run:
                shutil.move(str(legacy_figures), str(new_figures))
                moved.append((legacy_figures, new_figures))
                logger.success(
                    f"Moved figures folder: {legacy_figures.name} → {new_figures.name}"
                )
            else:
                moved.append((legacy_figures, new_figures))

        # Step 4: Rewrite geodatabase path columns (if not dry_run)
        if not dry_run:
            self._rewrite_geodatabase_paths(self.pz_names)
            logger.success("Geodatabase paths updated.")

        # Step 5: Clean up legacy raster_data folder and dead folders
        if not dry_run:
            legacy_raster_data = self.path_raster_data
            if legacy_raster_data.exists() and not list(legacy_raster_data.iterdir()):
                legacy_raster_data.rmdir()
                logger.success(f"Removed empty: {legacy_raster_data.name}")

            # Remove empty dead folders if they exist
            for dead_folder in [
                self.path_img_correlation_outs,
                self.path_tio_inversion_outs,
            ]:
                if dead_folder.exists() and not list(dead_folder.iterdir()):
                    dead_folder.rmdir()
                    logger.info(f"Removed empty: {dead_folder.name}")

            # Recompute layout detection and compat paths
            self._legacy_layout = (
                self.path_root / f"raster_data_{self.project_name}"
            ).is_dir()
            self.path_figures = self.path_root / (
                f"figures_{self.project_name}" if self._legacy_layout else "figures"
            )

            # Refresh in-memory geodatabase layers
            self._thumbs = gpd.read_file(
                self.path_geodb, engine="pyogrio", layer="Thumbs"
            ).to_crs(epsg=self.epsg)
            self._pairs = gpd.read_file(
                self.path_geodb, engine="pyogrio", layer="Pairs"
            ).to_crs(epsg=self.epsg)

            logger.success(
                "Migration complete. Session re-initialized with new structure."
            )

        return {"moved": moved, "skipped": skipped, "warnings": warnings}

    def _rewrite_geodatabase_paths(self, pz_names: list[str]) -> None:
        """Rewrite path columns in Thumbs and Pairs layers after migration.

        Updates th_path (Thumbs), pa_path and pa_disparity_f_path (Pairs) to reflect
        the new folder structure. Substitutes legacy base segment with new one per pzone.
        """
        # Rewrite Thumbs layer
        thumbs_gdf = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Thumbs")
        for pz_name in pz_names:
            # Old pattern: raster_data_<project>/<pz>/opticals/<thumb>
            # New pattern: <pz>/optical/<thumb>
            old_segment = f"raster_data_{self.project_name}{pathlib.os.sep}{pz_name}{pathlib.os.sep}opticals"
            new_segment = f"{pz_name}{pathlib.os.sep}optical"

            thumbs_gdf["th_path"] = thumbs_gdf["th_path"].str.replace(
                old_segment, new_segment, regex=False
            )

        thumbs_gdf.to_file(self.path_geodb, layer="Thumbs", engine="pyogrio")
        logger.info(f"Updated {len(thumbs_gdf)} thumb paths in geodatabase.")

        # Rewrite Pairs layer
        pairs_gdf = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Pairs")
        for pz_name in pz_names:
            # Path pattern change: raster_data_<project>/<pz>/image_correlation/... →
            # <pz>/image_correlation/...
            old_segment = f"raster_data_{self.project_name}{pathlib.os.sep}{pz_name}{pathlib.os.sep}"
            new_segment = f"{pz_name}{pathlib.os.sep}"

            for col in ["pa_path", "pa_disparity_f_path"]:
                if col in pairs_gdf.columns:
                    pairs_gdf[col] = pairs_gdf[col].str.replace(
                        old_segment, new_segment, regex=False
                    )

        pairs_gdf.to_file(self.path_geodb, layer="Pairs", engine="pyogrio")
        logger.info(f"Updated {len(pairs_gdf)} pair paths in geodatabase.")

    ###############################
    # * Works properly
    def make_georaster_metadata(self) -> dict:
        """Creates an empty geopandas dictionary to store georaster metadata.

        Returns:
            dict: An empty dictionary with keys for georaster metadata.
        """
        geo_meta = {
            "filename": None,
            "sensor": None,
            "xRes": None,
            "yRes": None,
            "bands": None,
            "rows": None,
            "cols": None,
            "acq_date": None,
            "file_path": None,
            "sensor_family": None,
            "sensor_platform": None,
            "acq_datetime": None,
            "src_crs": None,
            "geometry": None,
        }
        return geo_meta

    # * Works properly
    def map_single_georaster(
        self,
        ta: pathlib.Path | list[str] | str | None,
        target_crs: str,
        image_types: set[str],
    ) -> dict | None:
        """Map a single georaster to the target CRS and image types.

        Args:
            ta (pathlib.Path | list[str] | str | None): The input raster file(s) to process.
            target_crs (str): The target coordinate reference system (CRS) to which the raster should be transformed.
            image_types (set[str]): A set of image types to consider during processing.

        Returns:
            dict | None: A dictionary containing the processed raster information, or None if processing fails.

        Returns:
            dict | None: A dictionary containing the processed raster information, or None if processing fails.

        # * Works properly
        # ! Possible conflicts when sensor name do not match exactly
        # ? Needs further testing with different image types
        # TODO: Implement more robust date detection for datetime and metadatasearch.
        # TODO: Implement unittests.
        """
        # Creates an empty dictionary to store the processed raster information
        georaster_metadata = self.make_georaster_metadata()
        try:
            # Filling metadata with georaster information
            r = gu.Raster(ta, load_data=False)
            georaster_metadata["filename"] = ta.name
            georaster_metadata["file_path"] = str(ta)
            georaster_metadata["xRes"] = r.res[0]
            georaster_metadata["yRes"] = r.res[1]
            georaster_metadata["bands"] = r.count
            georaster_metadata["rows"] = r.shape[0]
            georaster_metadata["cols"] = r.shape[1]
            if "opt" in image_types:
                sensor = re_searcher(
                    ta.name.lower(), gmc_sensors.sensors(gmc_sensors.supported_sensors)
                )
                if sensor == "unknown":
                    sensor = re_searcher(
                        ta.parent.name.lower(),
                        gmc_sensors.sensors(gmc_sensors.supported_sensors),
                    )
                sensor_normalized = gmc_sensors.sensor_normalize(sensor)
            elif "dem" in image_types:
                # A DEM carries no readable sensor metadata: the source
                # (pleiades, spot6, lidar, uav, aerial, … or "dem") is inferred
                # from the file then folder name.  Kept verbatim — the token
                # becomes the thumb's filename field, so it must stay free of
                # "_" and "-", which sensor_normalize would not guarantee.
                sensor = gmc_sensors.infer_dem_sensor(ta)
                sensor_normalized = {
                    "sensor": sensor,
                    "platform": sensor,
                    "family": "dem",
                }
            else:
                sensor = "unknown"
                sensor_normalized = gmc_sensors.sensor_normalize(sensor)
            georaster_metadata["image_type"] = (
                "dem" if "dem" in image_types and "opt" not in image_types else "opt"
            )
            georaster_metadata["sensor"] = sensor_normalized["sensor"]
            georaster_metadata["sensor_platform"] = sensor_normalized["platform"]
            georaster_metadata["sensor_family"] = sensor_normalized["family"]

            # Struggling to extract acquisition date, may be due to sensor-specific formats or missing metadata
            acq_datetime = gmc_sensors.extract_acquisition_date(ta, sensor=sensor)
            if acq_datetime is None:
                # Raising here would surface as "'NoneType' object has no attribute
                # 'date'" from the generic handler below, which points nowhere near
                # the cause. Name the file and what was searched instead.
                hint = (
                    "a multi-temporal DEM needs its acquisition date in the file "
                    "name (e.g. 'site_2019-08-14_pleiades_dem.tif') — undated "
                    "mosaics such as Copernicus or SRTM cannot be placed in time."
                    if georaster_metadata["image_type"] == "dem"
                    else "for a band-scoped filename (e.g. Sentinel-2 'B08.tif') the date "
                    "can only come from the metadata, so check it was downloaded."
                )
                logger.warning(
                    f"Skip {ta.name}: no acquisition date found. Neither the filename nor "
                    f"any .xml/.dim sidecar in {ta.parent.name}/ yielded one — {hint}"
                )
                return None
            georaster_metadata["acq_datetime"] = acq_datetime
            georaster_metadata["acq_date"] = acq_datetime.date()

            # Handle footprint and CRS extraction
            raster_footprint = r.footprint
            del r  # free memory
            if hasattr(raster_footprint, "ds"):
                gdf_raster_footprint = raster_footprint.ds
                src_crs = gdf_raster_footprint.crs
                shp = gdf_raster_footprint.geometry.union_all()
            else:
                shp = raster_footprint
                src_crs = getattr(raster_footprint, "crs", None)
            georaster_metadata["src_crs"] = (
                (src_crs.to_string() if hasattr(src_crs, "to_string") else str(src_crs))
                if src_crs
                else None
            )
            geom_proj = shp
            try:
                if target_crs and src_crs and str(src_crs) != target_crs:
                    geom_proj = (
                        gpd.GeoSeries([shp], crs=src_crs).to_crs(target_crs).iloc[0]
                    )
            except Exception:
                pass
            # Assign the projected geometry to the metadata dictionary
            georaster_metadata["geometry"] = geom_proj
            return georaster_metadata
        except Exception as e:
            logger.warning(f"Skip {ta.name}: {e}")
            return None

    # * Works properly
    @gmc_profile("gmc.session.map_georasters_bank", memprof=True)
    def map_georasters_bank(
        self,
        georasters_bank_path: str | pathlib.Path,
        extensions: str | Sequence[str] = None,
        image_type: str = "opt",
        epsg_code: str | int | None = None,
        suffix: str = "",
        max_workers: int = None,
        save_georaster_bank: bool = False,
        exclude_patterns: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Map georasters in the specified bank directory using parallel processing.

        Args:
            georasters_bank_path (str | pathlib.Path): Path to the georaster bank directory.
            extensions (str | Sequence[str], optional): File extensions to include. Defaults to None.
            image_type (str, optional): ``"opt"`` (default) or ``"dem"``. With
                ``"dem"`` the sensor is inferred from the file/folder name by
                :func:`~geomulticorr.correlation.supported_sensors.infer_dem_sensor`
                (``pleiades``, ``spot6``, ``lidar``, ``uav``, ``aerial``… or
                ``dem``), and the map is written to ``Protomap_dem`` instead of
                ``Protomap``. A DEM still needs a date in its name or sidecar.
            epsg_code (str | int | None, optional): EPSG code for the target CRS. Defaults to None.
            suffix (str, optional): Suffix to append to output files. Defaults to "".
            max_workers (int, optional): Maximum number of workers for parallel processing. Defaults to None.
            save_georaster_bank (bool, optional): Whether to save the georaster bank. Defaults to False.
            exclude_patterns (list[str] | None, optional): Substrings to match against filenames for exclusion.
                Defaults to ``AUXILIARY_FILE_EXCLUDE_PATTERNS`` from ``supported_sensors.py``.
                Pass ``[]`` to disable filtering.

        Raises:
            FileNotFoundError: If the georaster bank path does not exist.
            RuntimeError: If no features are collected.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the mapped georasters.

        # TODO: Implement unittests.
        """
        # Validate path
        georasters_bank_path = pathlib.Path(georasters_bank_path)
        if not georasters_bank_path.exists():
            raise FileNotFoundError(f"{georasters_bank_path} is not an existing path")

        if extensions is None:
            extensions = ["tif", "jp2"]

        if exclude_patterns is None:
            exclude_patterns = gmc_sensors.AUXILIARY_FILE_EXCLUDE_PATTERNS

        # Normalize extensions (case-insensitive)
        targets = []
        for x in (extensions if isinstance(extensions, tuple) else extensions):
            targets += georasters_bank_path.glob(f"**/*.{x.lower()}")
            targets += georasters_bank_path.glob(f"**/*.{x.upper()}")
        # targets = list({p for x in extensions
        #         for p in [*georasters_bank_path.glob(f"**/*.{x.lower()}"), *georasters_bank_path.glob(f"**/*.{x.upper()}")]})

        if exclude_patterns:
            targets = [
                t for t in targets if not any(pat in t.name for pat in exclude_patterns)
            ]

        if not targets:
            raise FileNotFoundError(
                f"No raster files with extensions {extensions} found in {georasters_bank_path}"
            )
        _image_types = (
            {image_type.lower()}
            if isinstance(image_type, str)
            else {str(t).lower() for t in image_type}
        )
        protomap_layer = self._protomap_layer(image_type)  # validates early
        # Choose a project/target CRS (prefer your session's EPSG)
        target_crs = (
            f"EPSG:{epsg_code}" if epsg_code is not None else f"EPSG:{self.epsg}"
        )

        file_name = _RichText("Mapping georasters: ", style="cyan")
        progress = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=_rich_console,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(self.map_single_georaster, ta, target_crs, _image_types)
                for ta in targets
            ]
            rows = []
            errors = []
            task = progress.add_task("", total=len(futures))
            with _RichLive(
                _RichGroup(file_name, progress),
                console=_rich_console,
                transient=False,
                refresh_per_second=10,
            ) as live:
                for future in as_completed(futures):
                    try:
                        rec = future.result()
                        if rec:
                            file_name.plain = (
                                f"Mapping georasters: {rec.get('filename', '')}"
                            )
                            rows.append(rec)
                    except Exception as e:
                        errors.append(str(e))
                    finally:
                        progress.advance(task)
                        live.refresh()

        if not rows:
            error_msg = "No features collected."
            if errors:
                error_msg += f"\nErrors encountered:\n" + "\n".join(
                    errors[:5]
                )  # Show first 5 errors
            raise RuntimeError(error_msg)

        # Rebuild geometries and GeoDataFrame
        df = pd.DataFrame(rows)
        layer_crs = target_crs or (
            df["src_crs"].dropna().iloc[0] if not df["src_crs"].dropna().empty else None
        )
        features_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=layer_crs)
        features_gdf.to_file(self.path_geodb, layer=protomap_layer)
        if save_georaster_bank:
            self.save_georaster_bank(features_gdf)
        return features_gdf

    # * Works properly
    def insert_pzone_to_gdb(
        self, polygon: gpd.GeoDataFrame, pz_name: str, pz_shortname: str, **attributes
    ):
        """Insert a new polygon into the Pzones layer of the session geodatabase.
        Args:
            polygon (shapely.geometry.Polygon): The polygon geometry to insert.
            pz_name (str): The name of the processing zone.
            pz_shortname (str): The short name of the processing zone.
            **attributes: Additional attributes matching the Pzones schema.
        Returns:
            gpd.GeoDataFrame: The updated Pzones layer as a GeoDataFrame.

        # * Works properly
        # TODO: Implement unittests.
        # TODO: Test on other machines.
        """
        # Read the current Pzones layer
        gdf = gpd.read_file(self.path_geodb, layer="Pzones", engine="pyogrio")
        # Prepare new row with all required columns
        row = {"pz_name": pz_name, "pz_shortname": pz_shortname, "geometry": polygon}
        # Add any extra attributes provided
        for k, v in attributes.items():
            row[k] = v
        # Ensure all columns are present
        for col in gdf.columns:
            if col not in row:
                row[col] = None
        new_row = gpd.GeoDataFrame([row], crs=gdf.crs)
        # Append and save
        gdf = pd.concat([gdf, new_row], ignore_index=True)
        gdf.to_file(self.path_geodb, layer="Pzones", engine="pyogrio")
        # Optionally update the session's _pzones attribute
        self._pzones = gdf
        return gdf

    # * Works properly
    def draw_polygon_manually(
        self,
        center: tuple = (46.8, 8.2),
        zoom: int = 6,
        default_crs: str = "EPSG:4326",
        raster_bank: gpd.GeoDataFrame | None = None,
    ) -> callable:
        """Launch an interactive map for drawing a pzone polygon.

        Returns a callable; call it after drawing to get a GeoDataFrame of the
        drawn polygons reprojected to ``self.epsg``.

        When *raster_bank* is provided the image footprints are overlaid on the
        map (one layer per ``sensor_family``), the view is auto-fitted to their
        extent, and a layers control lets you toggle each sensor on/off.

        Usage::

            raster_bank = session.map_georasters_bank(...)
            get_gdf = session.draw_polygon_manually(raster_bank=raster_bank)
            # draw on the map, then:
            gdf = get_gdf()
            session.insert_pzone(gdf.geometry.iloc[0], ...)

        Args:
            center: Initial map center as (lat, lon). Used only when
                *raster_bank* is None. Defaults to (46.8, 8.2).
            zoom: Initial map zoom level. Used only when *raster_bank* is None.
            default_crs: CRS of coordinates returned by ipyleaflet (always EPSG:4326).
            raster_bank: Optional GeoDataFrame returned by
                :meth:`map_georasters_bank`.  When supplied, footprints are
                shown on the map, coloured by ``sensor_family``, and the view
                is fitted to their total extent.

        Returns:
            A callable ``get_gdf()`` that returns a GeoDataFrame in ``self.epsg``,
            or None if no polygon was drawn.
        """
        from ipyleaflet import GeoData, LayersControl

        _SENSOR_COLORS = {
            "spot": "#e6194b",
            "sentinel2": "#3cb44b",
            "landsat": "#4363d8",
            "pleiades": "#f58231",
            "planet": "#911eb4",
            "worldview": "#42d4f4",
        }

        def _json_safe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            """Stringify columns whose values are not JSON-serialisable (e.g. date, Path)."""
            _ok = (str, int, float, bool, type(None))
            out = gdf.copy()
            for col in out.columns:
                if col == "geometry":
                    continue
                sample = out[col].dropna()
                if not sample.empty and not isinstance(sample.iloc[0], _ok):
                    out[col] = out[col].astype(str)
            return out

        m = Map(center=center, zoom=zoom)

        if raster_bank is not None:
            rb_4326 = raster_bank.to_crs("EPSG:4326")

            if "sensor_family" in rb_4326.columns:
                for sensor, group in rb_4326.groupby("sensor_family"):
                    color = _SENSOR_COLORS.get(str(sensor).lower(), "#aaaaaa")
                    m.add_layer(
                        GeoData(
                            geo_dataframe=_json_safe(group.reset_index(drop=True)),
                            style={
                                "color": color,
                                "weight": 1.5,
                                "fillColor": color,
                                "fillOpacity": 0.1,
                            },
                            hover_style={"fillOpacity": 0.4},
                            name=str(sensor),
                        )
                    )
            else:
                m.add_layer(
                    GeoData(
                        geo_dataframe=_json_safe(rb_4326),
                        style={"color": "#3388ff", "weight": 1.5, "fillOpacity": 0.1},
                        hover_style={"fillOpacity": 0.4},
                        name="Raster Bank",
                    )
                )

            m.add_control(LayersControl())

            bounds = (
                rb_4326.total_bounds
            )  # [minx, miny, maxx, maxy] = [lon_min, lat_min, lon_max, lat_max]
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        draw_control = DrawControl(
            polyline={},
            circlemarker={},
            marker={},
            circle={},
            rectangle={"shapeOptions": {"color": "#3388ff", "weight": 3}},
        )
        m.add_control(draw_control)
        drawn_polygons = []

        def handle_draw(target, action, geo_json):
            if action == "created" and geo_json["geometry"]["type"] == "Polygon":
                poly = shape(geo_json["geometry"])
                drawn_polygons.append(poly)
                print("Polygon drawn! Coordinates:", poly.exterior.coords[:])

        draw_control.on_draw(handle_draw)
        display(m)
        print(
            "Draw your polygon(s) on the map above. When done, run the next cell to get the GeoDataFrame."
        )

        def get_gdf():
            if drawn_polygons:
                gdf = gpd.GeoDataFrame(geometry=drawn_polygons, crs=default_crs)
                target_crs = getattr(self, "epsg", None)
                if target_crs and str(gdf.crs) != f"EPSG:{target_crs}":
                    try:
                        gdf = gdf.to_crs(epsg=int(target_crs))
                    except Exception as e:
                        print(f"CRS conversion failed: {e}")
                return gdf
            else:
                print("No polygons drawn.")
                return None

        return get_gdf

    def _pairs_frame(self, criterias: str | list[str] = "") -> pd.DataFrame:
        """Read committed pairs from the geodatabase as a canonical *pairs frame*.

        Shared entry point for every pair figure — see
        :mod:`geomulticorr.utils._pairs_frame` for the column contract.

        :param criterias: Passed to :meth:`get_pairs_overview` to filter pairs.
        :returns: A pairs frame.
        :raises ValueError: If no pair matches *criterias*.
        """
        from geomulticorr.utils._pairs_frame import pairs_frame_from_overview

        pairs_gdf = self.get_pairs_overview(criterias)
        if pairs_gdf.empty:
            raise ValueError("No pairs found. Build pairs first with update_pairs().")

        # Sensor fallback for geodatabases written before pa_left_sensor/pa_right_sensor
        # existed; ignored (and not read) when the columns are present.
        date_sensor: dict[str, str] = {}
        if not {"pa_left_sensor", "pa_right_sensor"} <= set(pairs_gdf.columns):
            thumbs_gdf = self.get_thumbs_overview()
            if not thumbs_gdf.empty:
                date_sensor = dict(zip(thumbs_gdf["th_date"], thumbs_gdf["th_sensor"]))

        return pairs_frame_from_overview(pairs_gdf, date_sensor=date_sensor or None)

    def plot_pairs_network(
        self,
        criterias: str | list[str] = "",
        figsize: tuple[int, int] = (14, 5),
        color_by: str = "sensor",
        ax=None,
        **style,
    ) -> tuple:
        """Plot built image pairs as a temporal network diagram.

        Each acquisition date is a node on a horizontal timeline; each pair is
        a Bézier arc connecting its two dates.  Arc height is proportional to
        the time-delta, so short pairs sit low and long pairs arch higher.

        Args:
            criterias: passed to get_pairs_overview() to filter pairs.
            figsize: matplotlib figure size (ignored when *ax* is provided).
            color_by: "sensor" | "pzone" | "direction" | "dt" — arc/node colour.
            ax: existing matplotlib Axes to draw into; a new figure is created
                when None.
            style: extra styling forwarded to
                :func:`~geomulticorr.utils.gmc_functions.plot_pairs_network`
                (``linewidth``, ``alpha``, ``n_bezier_points``, ``node_size``,
                ``max_xticks``, ``cmap_dt``, ``dedupe``, ``fig_name``).

        Returns:
            (fig, ax) tuple.
        """
        from geomulticorr.utils import gmc_functions as gmc_fn

        return gmc_fn.plot_pairs_network(
            self._pairs_frame(criterias),
            figsize=figsize,
            color_by=color_by,
            ax=ax,
            **style,
        )

    def plot_pairs_chord(
        self,
        criterias: str | list[str] = "",
        figsize: tuple[int, int] = (10, 10),
        **style,
    ) -> tuple:
        """Plot built image pairs as a time-proportional chord diagram.

        Dates sit on a circle **proportionally to time** — Jan 1 at the top,
        running clockwise — so seasonal clustering and gaps in the archive are
        immediately visible.  The colour ring encodes position within the covered
        period, ``|`` ticks mark each Jan 1, and every pair is a Bézier chord
        whose bow depth grows with its temporal baseline.

        Args:
            criterias: passed to get_pairs_overview() to filter pairs.
            figsize: matplotlib figure size.
            style: geometry and styling options forwarded to
                :func:`~geomulticorr.utils.gmc_functions.plot_pairs_chord` —
                ``cmap``, ``n_color_segments``, ``outer_radius``, ``ring_width``,
                ``link_radius``, ``label_radius``, ``show_year_ticks``,
                ``year_tick_length`` / ``_linewidth`` / ``_color``,
                ``show_year_labels``, ``year_label_fontsize`` / ``_color``,
                ``link_color`` / ``_linewidth`` / ``_alpha`` / ``_curvature``,
                ``border_color`` / ``_linewidth``, ``show_date_nodes``,
                ``node_size``, ``xylim``, ``n_bezier_points``, ``dedupe``,
                ``fig_name``, ``ax``.

        Returns:
            (fig, ax) tuple.

        Note:
            Duplicate unordered couples are dropped by default (``dedupe=True``):
            the ``redundancy`` and ``forward-backward`` strategies emit each
            couple twice, which would otherwise overplot every chord and double
            its apparent opacity.
        """
        from geomulticorr.utils import gmc_functions as gmc_fn

        return gmc_fn.plot_pairs_chord(
            self._pairs_frame(criterias),
            figsize=figsize,
            **style,
        )

    def save_pairs_figure(
        self,
        view: str = "baseline",
        *,
        criterias: str | list[str] = "",
        frame: pd.DataFrame | None = None,
        formats: str | list[str] | tuple[str, ...] = ("html", "png"),
        stem: str | None = None,
        subdir: str = "pairs",
        title: str = "",
        dpi: int = 300,
        figsize: tuple[float, float] | None = None,
        plotlyjs: bool | str = True,
        overwrite: bool = True,
        pz_name: str | None = None,
        **view_kwargs,
    ) -> dict:
        """Save a pair figure.

        For new-layout sessions: saves to ``<pz_name>/figures/<subdir>/`` if
        the frame contains exactly one distinct pzone and ``pz_name`` is not
        explicitly overridden. For legacy-layout sessions or multi-pzone frames:
        falls back to the project-wide ``figures_<project>/<subdir>/``.

        Writes the chosen view twice: a self-contained interactive ``.html``
        (plotly — opens with no network access) and a publication-quality
        ``.png`` / ``.jpg`` / ``.pdf`` / ``.svg`` (matplotlib).  No ``kaleido``
        needed.

        Args:
            view: "baseline" | "chord" | "network" | "dt_hist".
            criterias: passed to get_pairs_overview() when *frame* is None.
            frame: pairs to draw. None reads the **committed** pairs from the
                geodatabase; :meth:`explore_pairs_strategy` passes its in-memory
                candidate frame, which is never written to disk and so could not
                be re-read.
            formats: any of "html", "png", "jpg", "pdf", "svg".
            stem: file stem; when omitted a deterministic one is built from the
                pzone, view, strategy, max_step and Δt window — so re-running
                refreshes the same files instead of piling up copies.
            overwrite: replace a file of the same name (default). False appends
                _01, _02… instead.
            subdir: subdirectory under the figures folder.
            title: figure title, applied to both backends.
            dpi: raster resolution for the matplotlib output.
            figsize: matplotlib figure size; a per-view default is used otherwise.
            plotlyjs: True inlines plotly.js (offline-capable, ~3 MB); "cdn"
                gives a small file that needs a network connection to render.
            pz_name: For new-layout sessions, explicitly route to this pzone's
                figures folder. If None and frame has exactly one distinct pzone,
                that pzone is used; otherwise falls back to project-wide folder.
            view_kwargs: view options (``dedupe``, ``show_arrows``, ``color_by``,
                ``mirror_direction``, ``nbins``…), filtered per backend.

        Returns:
            ``{format: Path}`` for every file written.
        """
        from geomulticorr.utils._pairs_export import (
            pairs_figure_stem,
            save_pairs_figure,
        )

        if frame is None:
            frame = self._pairs_frame(criterias)

        # Determine output directory per layout and pzone resolution
        if pz_name is not None:
            # Explicit pz_name: always route to that pzone's figures folder (new layout only)
            if self._legacy_layout:
                logger.warning(
                    f"pz_name={pz_name!r} ignored for legacy-layout session; using project-wide figures folder"
                )
                out_dir = self.path_figures / subdir
            else:
                out_dir = self.pz_dir(pz_name, PZ_KIND_FIGURES) / subdir
        elif not self._legacy_layout and "pz" in frame.columns:
            # New layout: check if frame has exactly one distinct pzone
            distinct_pzones = frame["pz"].unique()
            if len(distinct_pzones) == 1:
                out_dir = self.pz_dir(distinct_pzones[0], PZ_KIND_FIGURES) / subdir
            else:
                # Multi-pzone or no pz column: fall back to project-wide
                out_dir = self.path_figures / subdir
        else:
            # Legacy layout or frame has no pz column: use project-wide figures folder
            out_dir = self.path_figures / subdir

        if stem is None:
            # _last_pairs_params holds exactly the seven keys pairs_figure_stem takes
            # (and that update_pairs accepts) — see the note in _stash().
            params = getattr(self, "_last_pairs_params", None) or {
                "strategy": "committed"
            }
            stem = pairs_figure_stem(view, **params)

        paths = save_pairs_figure(
            frame,
            out_dir,
            view=view,
            formats=formats,
            stem=stem,
            title=title,
            dpi=dpi,
            figsize=figsize,
            plotlyjs=plotlyjs,
            view_kwargs=view_kwargs,
            overwrite=overwrite,
        )
        logger.success(f"Saved {view} figure to {out_dir} ({', '.join(paths)}).")
        return paths

    # ── Pairing-strategy exploration ────────────────────────────────────────────

    #: Controls that actually affect each pairing strategy (drives the explorer's
    #: show/hide logic). ``strategy`` and the Δt/sensor/pzone filters are always shown.
    _STRATEGY_CONTROLS: dict[str, set[str]] = {
        "consecutive": set(),
        "step": {"max_step"},
        "redundancy": {"max_step"},
        "forward-backward": set(),
    }

    #: Controls each explorer view uses.  Independent of ``_STRATEGY_CONTROLS`` —
    #: the two sets are unioned to decide what is visible.
    _VIEW_CONTROLS: dict[str, set[str]] = {
        "baseline": set(),
        "chord": {"dedupe", "arrows"},
        "network": {"color_by", "dedupe", "mirror"},
        "dt_hist": {"bins"},
    }

    #: Sensible ``dedupe`` per view, applied when the view changes.  Chord wants
    #: couples merged (overplotting doubles a chord's apparent opacity); network
    #: wants them kept, otherwise mirroring has no backward arcs to show.
    _VIEW_DEDUPE_DEFAULT: dict[str, bool] = {"chord": True, "network": False}

    #: What one drawn element means in each view, for the summary line.
    _DRAWN_NOUN: dict[str, str] = {
        "baseline": "points drawn",
        "chord": "chords drawn",
        "network": "arcs drawn",
        "dt_hist": "pairs binned",
    }

    #: Keyword arguments ``explore_pairs_strategy`` accepts through ``**defaults``.
    #: Anything else is a typo and raises, rather than being silently ignored.
    _EXPLORER_DEFAULTS: set[str] = {
        "max_step",
        "max_dt_days",
        "min_dt_days",
        "sensor_filter",
        "image_type",
    }

    # ------------------------------------------------------------------ #
    # Correlation-parameter explorer (see docs/correlation_parameters.md)
    # ------------------------------------------------------------------ #

    #: Controls each correlation view uses.  Everything not listed is always
    #: visible.
    #:
    #: Only genuinely **view-local** options belong here.  ``strain_rate_per_yr``
    #: and ``tau_days`` do not: they change the derived kernels and the warnings
    #: written into every ``*_CorrParameters.txt``, so hiding them off their
    #: "own" view left invisible values shaping the committed parameters — a
    #: control the user cannot see but is still being governed by.  They stay
    #: visible on every view; only ``color_by``, which touches nothing but the
    #: marker colours, is conditional.
    _CORR_VIEW_CONTROLS: dict[str, set[str]] = {
        "design_map": {"color_by"},
        "kernel_search": set(),
        "snr": set(),
        "cost": set(),
    }

    #: What one drawn element means in each correlation view, for the summary.
    _CORR_DRAWN_NOUN: dict[str, str] = {
        "design_map": "pairs plotted",
        "kernel_search": "pairs sized",
        "snr": "pairs scored",
        "cost": "groups costed",
    }

    #: Keyword arguments ``explore_correlation_params`` accepts through
    #: ``**defaults``.  Anything else is a typo and raises, rather than being
    #: silently ignored — ``velocity=5`` used to be a no-op in the pairs
    #: explorer's equivalent.
    _CORRPARAMS_EXPLORER_DEFAULTS: set[str] = {
        "velocity_m_yr",
        "footprint_m",
        "c_px",
        "k",
        "coreg_px",
        "margin_px",
        "flow_azimuth_deg",
        "strain_rate_per_yr",
        "corr_algorithm",
        "subpixel_mode",
        "tau_days",
        "dt_bin_ratio",
        "explicit_search",
        "explicit_search_above_days",
        "uniform_kernel",
        "uniform_kernel_px",
        "uniform_search",
        "uniform_search_box",
        "sensor_filter",
        "max_dt_days",
        "min_dt_days",
        "color_by",
    }

    #: The **fixed** key set of ``_last_corr_params``.  A strict subset of the
    #: derivation inputs plus the plan, so
    #: ``prepare_pairs_correlation(**params)`` is legal.  Never pruned per mode:
    #: dropping a key would make the receiving method silently substitute its
    #: own default for whatever went missing.  View-only state (``view``,
    #: ``color_by``) stays out, exactly as ``direction`` is excluded from
    #: ``_WEIGHT_PARAM_KEYS``.
    _CORR_PARAM_KEYS: tuple[str, ...] = ("plan", "params_by_pair")

    #: Derivation inputs the explorer stashes for replay, written into the plan
    #: JSON rather than splatted into ``prepare_pairs_correlation`` (which knows
    #: nothing about velocities).
    _CORR_ASSUMPTION_KEYS: tuple[str, ...] = (
        "velocity_m_yr",
        "footprint_m",
        "c_px",
        "k",
        "coreg_px",
        "margin_px",
        "flow_azimuth_deg",
        "strain_rate_per_yr",
        "corr_algorithm",
        "subpixel_mode",
        "tau_days",
        "dt_bin_ratio",
        "explicit_search",
        "explicit_search_above_days",
    )

    #: Uniform overrides. Kept apart from :attr:`_CORR_ASSUMPTION_KEYS` because
    #: they do not feed ``suggest_parameters`` — the derivation still runs in
    #: full, producing the warnings that say where the uniform value is wrong —
    #: they replace its result afterwards, as the group parameters.
    _CORR_UNIFORM_KEYS: tuple[str, ...] = (
        "uniform_kernel",
        "uniform_kernel_px",
        "uniform_search",
        "uniform_search_box",
    )

    def _thumb_resolution(self, path: str | pathlib.Path) -> float:
        """Ground sample distance of the raster at *path*, in metres, memoised.

        Nothing in the object model stores resolution — not ``Thumb``, not
        ``Pair``, not ``SENSOR_CATALOG`` — and a nominal per-sensor table would
        be wrong the moment ``sieve_bulk``/``target_resolution`` resampled the
        thumbs.  So read it from the file header, which is authoritative.

        ``load_data=False`` keeps this a metadata read: the array is never
        pulled, the same discipline ``_grid.py`` enforces for grid comparison.
        Memoised because the explorer derives parameters on every slider tick
        and would otherwise reopen the same handful of files hundreds of times.

        :param path: Raster path.
        :returns: Pixel size in metres, or ``nan`` when unreadable.
        """
        key = str(path)
        cache = getattr(self, "_thumb_res_cache", None)
        if cache is None:
            cache = self._thumb_res_cache = {}
        if key in cache:
            return cache[key]

        value = float("nan")
        try:
            import geoutils as gu

            value = abs(float(gu.Raster(key, load_data=False).res[0]))
        except Exception as exc:  # noqa: BLE001 — any read failure is non-fatal
            logger.warning(f"Could not read resolution from {key}: {exc}")
        cache[key] = value
        return value

    def explore_pairs_strategy(
        self,
        strategy: str = "consecutive",
        pz_name: str = "",
        view: str = "baseline",
        interactive: bool = True,
        savefig: bool = True,
        formats: str | list[str] | tuple[str, ...] = ("html", "png"),
        **defaults,
    ):
        """Interactive Plotly + ipywidgets tool to preview pairing strategies.

        Recomputes the candidate pairs on every control change *without* touching
        the geodatabase, so strategies can be compared before committing them with
        :meth:`update_pairs`.  Four views of the same candidate set are available
        from the ``plot`` dropdown:

        ==============  ==============================================================
        ``baseline``    Δt vs pair midpoint date — coverage and baseline spread.
        ``chord``       Time-proportional circle — seasonal clustering, redundancy.
        ``network``     Timeline with one arc per pair — who connects to whom.
        ``dt_hist``     Distribution of temporal baselines.
        ==============  ==============================================================

        Follows the same fast pattern as
        :meth:`~geomulticorr.inversion.tio_inversion.TIOInversion.explore_weights`:
        the (expensive) ``Thumb`` objects are read **once** at setup, then every
        control change only re-runs pure integer/date arithmetic — no ``Thumb``
        or ``Pair`` object is reconstructed inside the interactive loop.  Switching
        view does not even recompute the pairs; the frame is cached.

        The tuned parameters are stored on ``self._last_pairs_params`` so they can
        be committed directly::

            session.explore_pairs_strategy()                 # tune interactively
            session.update_pairs(**session._last_pairs_params)

        The **Save figure** button writes whatever is on screen — including the
        candidate pairs, which never reach the geodatabase — into
        ``figures_<project>/pairs/`` as an offline-capable ``.html`` plus the
        publication-quality formats ticked in the list.  See
        :meth:`save_pairs_figure` for the scripted equivalent.

        Requires plotly and ipywidgets (Jupyter). Returns the assembled
        ``ipywidgets.VBox`` (also displays inline in a notebook).

        **Non-interactive mode** — pass ``interactive=False`` to skip the widgets
        entirely and just get the results for the arguments given::

            frame, fig = session.explore_pairs_strategy(
                strategy="redundancy", max_step=17, view="chord", interactive=False)
            session.update_pairs(**session._last_pairs_params)   # same commit step

        ``frame`` is the candidate pairs table, ``fig`` the plotly figure for
        *view* — built but never displayed, so it can be shown, saved with
        ``fig.write_html(…)`` or ignored.  This path imports no ipywidgets and
        needs no display, which makes it the one to use from a script or a batch
        job.  ``self._last_pairs_params`` is stashed exactly as in the interactive
        mode.

        By default it also **writes the figure** to ``figures_<project>/pairs/``
        under a deterministic name built from the pzone, view, strategy,
        ``max_step`` and Δt window — so a sweep produces one predictable file set
        and re-running refreshes it rather than piling up copies.  Pass
        ``savefig=False`` to skip.

        :param strategy: Pairing strategy (initial dropdown value when interactive).
        :param pz_name: Restrict to a single pzone (default: all pzones).
        :param view: Plot kind — ``"baseline"``, ``"chord"``, ``"network"`` or
            ``"dt_hist"``.
        :param interactive: ``True`` (default) builds the widget; ``False`` returns
            ``(frame, fig)`` for the arguments given, with no UI.
        :param savefig: Non-interactive mode only — write the figure to disk.
            Ignored when *interactive* (the widget has a **Save figure** button).
        :param formats: Formats written when *savefig*; any of
            ``"html"``, ``"png"``, ``"jpg"``, ``"pdf"``, ``"svg"``.
        :param defaults: ``max_step``, ``max_dt_days``, ``min_dt_days``,
            ``sensor_filter``, ``image_type`` — initial control values when
            interactive, the actual parameters when not.  ``image_type``
            (default ``"opt"``) picks which thumbs are paired; the dropdown only
            appears when the Thumbs layer holds more than one type.  ``max_step`` defaults to 2 in both modes and is
            clamped to the largest useful value for the data.
        :returns: An ``ipywidgets.VBox`` when *interactive*, else
            ``(frame, fig)``.
        """
        from geomulticorr.utils import _pairs_plotly as gmc_plotly
        from geomulticorr.utils._pairs_frame import (
            empty_pairs_frame,
            format_pairs_summary,
            pairs_frame_from_indices,
            pairs_stats,
            unique_couples,
        )

        if view not in gmc_plotly.VIEW_BUILDERS:
            raise ValueError(
                f"Unknown view '{view}'. Valid options: {sorted(gmc_plotly.VIEW_BUILDERS)}."
            )
        unknown = set(defaults) - self._EXPLORER_DEFAULTS
        if unknown:
            raise TypeError(
                f"Unexpected argument(s) {sorted(unknown)}. "
                f"Valid options: {sorted(self._EXPLORER_DEFAULTS)}."
            )

        # --- setup: read Thumb objects ONCE, cache sorted-by-date per pzone -----
        pzones = self.get_pzones(pz_name)
        thumbs_by_pz: dict[str, list] = {}
        for pz in pzones:
            # Every image type is read once here; the type is then a pure
            # in-memory filter, like the sensor substring.
            ths = sorted(
                pz.get_valid_thumbs(image_type=None), key=lambda t: t.th_date_datetime
            )
            thumbs_by_pz[pz.pz_name] = ths
        pz_options = ["<all>"] + list(thumbs_by_pz.keys())
        init_type = normalise_image_type(defaults.get("image_type"))
        type_options = sorted(
            {
                getattr(t, "th_type", IMAGE_TYPE_OPTICAL)
                for ths in thumbs_by_pz.values()
                for t in ths
            }
            | {init_type},
            key=IMAGE_TYPES.index,
        )

        def _selected_thumbs(
            sel_pz: str, sfilter: str, itype: str = IMAGE_TYPE_OPTICAL
        ) -> list[tuple[str, list]]:
            """``(pzone name, thumbs)`` per pzone under the current filters.

            The sensor filter is a substring match on the thumb path — the same
            semantics as ``get_valid_thumbs(sensor_filter=…)``, applied here to the
            list read once at setup so no raster is reopened.  *itype* keeps one
            image type, since pairs never cross types.
            """
            out = []
            for pzn, ths in thumbs_by_pz.items():
                if sel_pz != "<all>" and pzn != sel_pz:
                    continue
                out.append(
                    (
                        pzn,
                        [
                            t
                            for t in ths
                            if getattr(t, "th_type", IMAGE_TYPE_OPTICAL) == itype
                            and (not sfilter or sfilter.lower() in t.th_path.lower())
                        ],
                    )
                )
            return out

        def _max_useful_step(
            sel_pz: str | None = None,
            sfilter: str | None = None,
            itype: str | None = None,
        ) -> int:
            """Largest ``max_step`` that still adds pairs for the current selection.

            Pairing runs per pzone, so with ``"<all>"`` the bound comes from the
            largest one.  Beyond ``n - 1`` every further step is a no-op — the
            strategies clamp to the array bounds — so this is the honest ceiling
            of the data, not an arbitrary guardrail.

            :param sel_pz: Pzone selection; ``None`` reads the widget.
            :param sfilter: Sensor substring; ``None`` reads the widget.
            """
            if sel_pz is None:
                sel_pz = c_pz.value
            if sfilter is None:
                sfilter = (c_sensor.value or "").strip()
            if itype is None:
                itype = c_type.value
            counts = [len(sel) for _, sel in _selected_thumbs(sel_pz, sfilter, itype)]
            return max(1, max(counts, default=1) - 1)

        def _step_tooltip(ceiling: int) -> str:
            return (
                "Maximum index offset between paired acquisitions "
                f"(≤ {ceiling} — pairs every acquisition with every other)"
            )

        def _build_frame(
            strat: str,
            max_step: int | None,
            max_dt: int | None,
            min_dt: int | None,
            sfilter: str,
            sel_pz: str,
            itype: str = IMAGE_TYPE_OPTICAL,
        ) -> pd.DataFrame:
            """Candidate pairs for an explicit parameter set.

            Pure — no I/O, no ``Thumb``/``Pair`` construction — so it is equally
            usable from the interactive loop and from a headless call.
            """
            frames: list[pd.DataFrame] = []
            for pzn, sel in _selected_thumbs(sel_pz, sfilter, itype):
                idx_pairs = gmc_pzone._strategy_pair_indices(len(sel), strat, max_step)
                frames.append(
                    pairs_frame_from_indices(
                        [t.th_date_datetime for t in sel],
                        idx_pairs,
                        sensors=[t.th_sensor for t in sel],
                        pz=pzn,
                    )
                )

            frame = (
                pd.concat(frames, ignore_index=True) if frames else empty_pairs_frame()
            )
            if len(frame):  # Δt filters, vectorised
                keep = np.ones(len(frame), dtype=bool)
                if max_dt is not None:
                    keep &= frame["dt_days"].to_numpy() <= max_dt
                if min_dt is not None:
                    keep &= frame["dt_days"].to_numpy() >= min_dt
                frame = frame.loc[keep].reset_index(drop=True)
            return frame

        def _view_defaults(view_key: str) -> dict:
            """Figure options for a view — the values the widgets also start from,
            so both modes render the same figure for the same pairs."""
            if view_key == "network":
                return dict(
                    color_by="dt",
                    mirror_direction=True,
                    dedupe=self._VIEW_DEDUPE_DEFAULT.get("network", False),
                )
            if view_key == "chord":
                return dict(
                    show_arrows=False,
                    dedupe=self._VIEW_DEDUPE_DEFAULT.get("chord", True),
                )
            if view_key == "dt_hist":
                return dict(nbins=40)
            return {}

        def _drawn_len(frame: pd.DataFrame, view_key: str, dedupe: bool) -> int:
            """How many curves/points a view actually renders for *frame*."""
            if view_key in ("chord", "network") and dedupe:
                return len(unique_couples(frame))
            return len(frame)

        def _title_for(
            frame: pd.DataFrame, view_key: str, strat: str, dedupe: bool
        ) -> str:
            n_drawn, n_pairs = _drawn_len(frame, view_key, dedupe), len(frame)
            noun = self._DRAWN_NOUN[view_key]
            suffix = "" if n_drawn == n_pairs else f" of {n_pairs} pairs"
            return f"Pairing preview — {n_drawn} {noun}{suffix} [{strat}]"

        def _stash_params(
            strat: str,
            max_step: int | None,
            max_dt: int | None,
            min_dt: int | None,
            sfilter: str | None,
            pzn: str,
            itype: str = IMAGE_TYPE_OPTICAL,
        ):
            # NOTE: exactly the seven keys update_pairs(**params) accepts — adding
            # any view/style key here would raise TypeError at that call.
            self._last_pairs_params = dict(
                strategy=strat,
                max_step=int(max_step) if strat in ("step", "redundancy") else None,
                max_dt_days=max_dt or None,
                min_dt_days=min_dt or None,
                sensor_filter=(sfilter or "").strip() or None,
                pz_name=pzn,
                image_type=itype,
            )

        # ── headless path: results only, no widgets, no display ──
        if not interactive:
            sfilter = str(defaults.get("sensor_filter", "") or "").strip()
            ceiling = _max_useful_step("<all>", sfilter, init_type)
            max_step = min(max(int(defaults.get("max_step", 2) or 1), 1), ceiling)
            # parse_dt_days, not int(): "1Y"/"6M"/"30D" are the forms
            # update_pairs already documents, so the same value works in both.
            max_dt = parse_dt_days(defaults.get("max_dt_days")) or None
            min_dt = parse_dt_days(defaults.get("min_dt_days")) or None

            frame = _build_frame(
                strategy,
                max_step if strategy in ("step", "redundancy") else None,
                max_dt,
                min_dt,
                sfilter,
                "<all>",
                init_type,
            )
            _stash_params(
                strategy, max_step, max_dt, min_dt, sfilter, pz_name, init_type
            )

            view_kwargs = _view_defaults(view)
            title = _title_for(frame, view, strategy, view_kwargs.get("dedupe", False))
            fig = gmc_plotly.VIEW_BUILDERS[view](frame, title=title, **view_kwargs)

            if savefig and len(frame):
                self.save_pairs_figure(
                    view=view,
                    frame=frame,
                    formats=formats,
                    title=title,
                    **view_kwargs,
                )
            elif savefig:
                logger.warning("No pairs to plot — nothing saved.")
            return frame, fig

        import ipywidgets as widgets
        from IPython.display import display as _ipy_display

        from geomulticorr.utils._pairs_export import FIGURE_FORMATS

        # ── controls ──
        c_view = widgets.Dropdown(
            options=[(label, key) for key, label in gmc_plotly.VIEW_LABELS.items()],
            value=view,
            description="plot",
        )
        c_strategy = widgets.Dropdown(
            options=["consecutive", "step", "redundancy", "forward-backward"],
            value=strategy,
            description="strategy",
        )
        # The ceiling is the data's, not an arbitrary constant: with n acquisitions
        # every couple is already paired at max_step = n - 1, and anything beyond
        # is a no-op.  It is re-fitted by _sync_step_range() when the pzone or
        # sensor filter changes the number of acquisitions in play.
        _step_ceiling = _max_useful_step(
            "<all>", str(defaults.get("sensor_filter", "") or "").strip(), init_type
        )
        c_step = widgets.IntSlider(
            value=min(max(int(defaults.get("max_step", 2)), 1), _step_ceiling),
            min=1,
            max=_step_ceiling,
            step=1,
            description="max_step",
            tooltip=_step_tooltip(_step_ceiling),
        )
        # Text, not IntText: "1Y"/"6M"/"30D" as well as a bare number of days.
        c_maxdt = widgets.Text(
            value=format_dt_days(parse_dt_days(defaults.get("max_dt_days"))),
            description="max_dt",
            placeholder="e.g. 1Y, 6M, 200",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="180px"),
        )
        c_mindt = widgets.Text(
            value=format_dt_days(parse_dt_days(defaults.get("min_dt_days"))),
            description="min_dt",
            placeholder="e.g. 1Y, 6M, 60",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="180px"),
        )
        c_sensor = widgets.Text(
            value=str(defaults.get("sensor_filter", "") or ""), description="sensor"
        )
        c_pz = widgets.Dropdown(options=pz_options, value="<all>", description="pzone")
        # Image type: only worth showing when there is a choice to make.
        c_type = widgets.Dropdown(
            options=type_options, value=init_type, description="type"
        )
        c_type.layout.display = None if len(type_options) > 1 else "none"
        # view-specific controls (hidden unless the active view uses them)
        c_colorby = widgets.Dropdown(
            options=["dt", "direction", "sensor", "pzone"],
            value="dt",
            description="colour",
        )
        c_bins = widgets.IntSlider(
            value=40, min=10, max=100, step=5, description="bins"
        )
        # Same per-view default the dropdown applies on switch — without it,
        # opening straight onto the network view would dedupe away every backward
        # pair and make "mirror directions" look broken.
        c_dedupe = widgets.Checkbox(
            value=self._VIEW_DEDUPE_DEFAULT.get(view, True),
            description="dedupe couples",
            indent=False,
        )
        c_arrows = widgets.Checkbox(value=False, description="arrowheads", indent=False)
        c_mirror = widgets.Checkbox(
            value=True, description="mirror directions", indent=False
        )
        # save row — the file name is derived from the pairing parameters, so the
        # only choices left are which formats to write.
        c_formats = widgets.SelectMultiple(
            options=list(FIGURE_FORMATS),
            value=("html", "png"),
            rows=len(FIGURE_FORMATS),
            description="formats",
        )
        b_save = widgets.Button(description="Save figure", icon="save")

        def _apply_visibility():
            """Show only the controls the current strategy *and* view actually use."""
            relevant = self._STRATEGY_CONTROLS.get(
                c_strategy.value, set()
            ) | self._VIEW_CONTROLS.get(c_view.value, set())
            for widget, key in (
                (c_step, "max_step"),
                (c_colorby, "color_by"),
                (c_bins, "bins"),
                (c_dedupe, "dedupe"),
                (c_arrows, "arrows"),
                (c_mirror, "mirror"),
            ):
                widget.layout.display = None if key in relevant else "none"

        def _read_dt(widget, label: str) -> int | None:
            """Parse one Δt field, degrading rather than raising.

            An exception inside a widget callback reaches only the kernel log,
            which VSCode hides — the control would simply stop responding with no
            visible cause. So a bad string writes a red status line and the last
            good value is reused.

            :returns: Days, or None when the field is empty.
            """
            text = (widget.value or "").strip()
            if not text:
                state.pop(f"dt_{label}", None)
                return None
            try:
                days = parse_dt_days(text if not text.isdigit() else int(text))
            except ValueError as exc:
                status.value = f"<span style='color:#c33'>✘ {label}: {exc}</span>"
                return state.get(f"dt_{label}")
            state[f"dt_{label}"] = days
            return days

        def _compute() -> pd.DataFrame:
            """Recompute from the current control values."""
            strat = c_strategy.value
            return _build_frame(
                strat,
                int(c_step.value) if strat in ("step", "redundancy") else None,
                _read_dt(c_maxdt, "max_dt"),
                _read_dt(c_mindt, "min_dt"),
                (c_sensor.value or "").strip(),
                c_pz.value,
                c_type.value,
            )

        def _stash():
            # Resolved integer days, never the typed string: the stash records
            # what was actually applied, and update_pairs(**params) is pinned by
            # test to accept exactly these seven keys.
            _stash_params(
                c_strategy.value,
                c_step.value,
                _read_dt(c_maxdt, "max_dt"),
                _read_dt(c_mindt, "min_dt"),
                c_sensor.value,
                "" if c_pz.value == "<all>" else c_pz.value,
                c_type.value,
            )

        # A plain ``go.Figure`` redrawn inside an ``Output`` is used instead of a
        # ``go.FigureWidget``: since plotly 6 the latter is an *anywidget*, whose
        # front-end JS is not bundled with the VSCode Jupyter extension and often
        # fails to load ("No version of module anywidget is registered"). Core
        # ipywidgets + the built-in plotly renderer work everywhere, offline.
        plot_out = widgets.Output()
        summary = widgets.HTML()
        status = widgets.HTML()
        state: dict = {}

        def _view_kwargs() -> dict:
            """View options, shared by the figure on screen and the saved one."""
            view_key = c_view.value
            if view_key == "network":
                return dict(
                    color_by=c_colorby.value,
                    dedupe=c_dedupe.value,
                    mirror_direction=c_mirror.value,
                )
            if view_key == "chord":
                return dict(dedupe=c_dedupe.value, show_arrows=c_arrows.value)
            if view_key == "dt_hist":
                return dict(nbins=int(c_bins.value))
            return {}

        def _drawn_frame() -> pd.DataFrame:
            """The subset the current view actually renders (dedupe applied)."""
            frame = state["frame"]
            if c_view.value in ("chord", "network") and c_dedupe.value:
                return unique_couples(frame)
            return frame

        def _n_drawn() -> int:
            """How many curves/points the current view actually shows."""
            return _drawn_len(state["frame"], c_view.value, c_dedupe.value)

        def _sync_direction_state():
            """Grey out "mirror directions" when there is nothing to mirror.

            Two ways to end up with no backward pairs, both easy to mistake for a
            broken checkbox:

            * ``consecutive`` and ``step`` are **forward-only** strategies — they
              emit only ``(i, j)`` with ``i < j``, so no arc can go below the axis;
            * ``dedupe`` keeps the first of each couple, which is the forward one,
              so ticking it on a bidirectional strategy removes every backward arc.
            """
            drawn = _drawn_frame()
            has_backward = bool(len(drawn)) and (drawn["direction"] == "backward").any()
            c_mirror.disabled = not has_backward
            if has_backward:
                c_mirror.tooltip = "Draw backward (j → i) pairs below the timeline"
            elif (
                c_view.value in ("chord", "network")
                and c_dedupe.value
                and len(state["frame"])
                and (state["frame"]["direction"] == "backward").any()
            ):
                c_mirror.tooltip = (
                    "Nothing to mirror: 'dedupe couples' merged each couple down to "
                    "its forward pair. Untick it to see both directions."
                )
            else:
                c_mirror.tooltip = (
                    f"Nothing to mirror: '{c_strategy.value}' is forward-only, so "
                    "there are no backward pairs. Use 'redundancy' or "
                    "'forward-backward' to build them."
                )

        def _fig_title() -> str:
            return _title_for(
                state["frame"], c_view.value, c_strategy.value, c_dedupe.value
            )

        def _draw():
            """Render the cached frame with the active view builder."""
            fig = gmc_plotly.VIEW_BUILDERS[c_view.value](
                state["frame"], title=_fig_title(), **_view_kwargs()
            )
            with plot_out:
                plot_out.clear_output(wait=True)
                _ipy_display(fig)

        def _render_summary():
            summary.value = format_pairs_summary(
                pairs_stats(state["frame"], strategy=c_strategy.value),
                drawn=_n_drawn(),
                drawn_label=self._DRAWN_NOUN[c_view.value],
            )

        def _redraw(change=None):
            """View-only change — reuse the cached frame, never re-pair."""
            if state.get("syncing"):
                return
            # switching view re-applies that view's sensible dedupe default
            if change is not None and change.get("owner") is c_view:
                want = self._VIEW_DEDUPE_DEFAULT.get(c_view.value)
                if want is not None and c_dedupe.value != want:
                    state["syncing"] = True
                    try:
                        c_dedupe.value = want
                    finally:
                        state["syncing"] = False
            _apply_visibility()
            _sync_direction_state()
            _draw()
            _render_summary()

        def _sync_step_range():
            """Re-fit the max_step ceiling to the pzone/sensor selection.

            Lowering ``max`` below ``value`` clamps the value *and* fires a change
            event, so this runs behind the syncing guard to keep the clamp from
            re-entering :func:`_update`.
            """
            ceiling = _max_useful_step()
            if ceiling == c_step.max:
                return
            state["syncing"] = True
            try:
                # order matters: never let traitlets see value > max
                if ceiling < c_step.max:
                    c_step.value = min(c_step.value, ceiling)
                    c_step.max = ceiling
                else:
                    c_step.max = ceiling
                c_step.tooltip = _step_tooltip(ceiling)
            finally:
                state["syncing"] = False

        def _update(_change=None):
            """Pairing change — recompute the candidate pairs, then redraw."""
            if state.get("syncing"):
                return
            _sync_step_range()
            _apply_visibility()
            state["frame"] = _compute()
            _stash()
            _sync_direction_state()
            _draw()
            _render_summary()

        def _on_save(_button):
            # Never let this raise: an uncaught exception in a widget callback
            # only reaches the kernel log, which VSCode usually hides.
            b_save.disabled = True
            status.value = "<i>saving…</i>"
            try:
                paths = self.save_pairs_figure(
                    view=c_view.value,
                    frame=state["frame"],
                    formats=tuple(c_formats.value) or ("html",),
                    title=_fig_title(),
                    **_view_kwargs(),
                )
                names = " ".join(f"<code>{p.name}</code>" for p in paths.values())
                folder = next(iter(paths.values())).parent
                status.value = (
                    f"<span style='color:#2a7'>✔ saved to</span> "
                    f"<code>{folder}</code><br>{names}"
                )
            except Exception as exc:
                logger.error(f"save_pairs_figure failed: {exc!r}")
                status.value = (
                    f"<span style='color:#c33'>✘ save failed — "
                    f"{type(exc).__name__}: {exc}</span>"
                )
            finally:
                b_save.disabled = False

        for c in (c_strategy, c_step, c_maxdt, c_mindt, c_sensor, c_pz, c_type):
            c.observe(_update, names="value")
        for c in (c_view, c_colorby, c_bins, c_dedupe, c_arrows, c_mirror):
            c.observe(_redraw, names="value")
        b_save.on_click(_on_save)

        _update()

        row1 = widgets.HBox([c_view, c_strategy, c_step, c_pz, c_type])
        row2 = widgets.HBox([c_maxdt, c_mindt, c_sensor])
        row3 = widgets.HBox([c_colorby, c_bins, c_dedupe, c_arrows, c_mirror])
        row4 = widgets.HBox([c_formats, b_save])
        return widgets.VBox([row1, row2, row3, plot_out, summary, status, row4])

    @staticmethod
    def _compute_canonical_grid(
        pz_vec: gu.Vector,
        target_epsg: int,
        target_resolution: float,
    ) -> tuple:
        """Return (bounds, (width, height)) snapped to the target pixel grid.

        Ensures all rasters cropped to the same pzone AOI have identical shapes.
        """
        import math
        import rasterio.coords as rio_coords

        xmin, ymin, xmax, ymax = pz_vec.get_bounds_projected(
            out_crs=f"EPSG:{target_epsg}"
        )
        xmin = math.floor(xmin / target_resolution) * target_resolution
        ymin = math.floor(ymin / target_resolution) * target_resolution
        xmax = math.ceil(xmax / target_resolution) * target_resolution
        ymax = math.ceil(ymax / target_resolution) * target_resolution
        w = round((xmax - xmin) / target_resolution)
        h = round((ymax - ymin) / target_resolution)
        return rio_coords.BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax), (
            w,
            h,
        )

    def sieve_single_mosaic(
        self,
        pzone_vector: gu.Vector | gpd.GeoDataFrame | gpd.GeoSeries,
        mosaic_path: pathlib.Path | str,
        target_resolution: float,
        target_epsg: int | None = None,
        resampling_method: str = "bilinear",
        selected_band: int | list[int] = 1,
        source_nodata: int | float | None = None,
        out_dtype: str = "uint16",
        n_threads: int = 4,
        memory_limit_mb: int = 1024,
        use_multiproc_threshold_mb: int = 2048,
        canonical_bounds: "object | None" = None,
        canonical_grid_size: "tuple[int, int] | None" = None,
    ) -> gu.Raster | None:
        """Crop a single mosaic to a pzone AOI using geoutils, memory-safe.

        Opens the raster lazily (metadata only), does a windowed read of the
        pzone overlap in the source CRS, then reprojects the small cropped
        tile to the target grid. Never loads the full source raster.

        Args:
            pzone_vector: AOI geometry (gu.Vector, GeoDataFrame, or GeoSeries).
            mosaic_path: Path to the source raster.
            target_resolution: Destination pixel size in target CRS units.
            target_epsg: Destination EPSG code. Defaults to ``self.epsg``.
            resampling_method: Rasterio resampling name (e.g. 'bilinear', 'nearest').
            selected_band: Band index or list of band indices to read.
            source_nodata: Override nodata for the source raster (metadata only,
                no array load). Use only when the source file lacks nodata metadata.
            out_dtype: Output dtype for the returned raster.
            n_threads: Threads for the reprojection warp.
            memory_limit_mb: Memory limit for the reprojection warp (MB).
            use_multiproc_threshold_mb: If the estimated output grid size exceeds
                this, switch to out-of-core reprojection via MultiprocConfig.
            canonical_bounds: Exact output bounds (rasterio BoundingBox). If provided,
                overrides any internal bounds computation and ensures identical shapes.
            canonical_grid_size: Exact output grid size (width, height). If provided,
                forces the output to have exactly this shape with no internal rounding.

        Returns:
            A cropped (and reprojected, if needed) gu.Raster, or None if the
            pzone does not intersect the source raster.
        """
        # Resolve target CRS
        if target_epsg is None:
            target_epsg = self.epsg

        # Normalize the AOI to a gu.Vector without copying pixels
        if isinstance(pzone_vector, gu.Vector):
            pz_vec = pzone_vector
        elif isinstance(pzone_vector, gpd.GeoDataFrame):
            pz_vec = gu.Vector(pzone_vector)
        elif isinstance(pzone_vector, gpd.GeoSeries):
            pz_vec = gu.Vector(
                gpd.GeoDataFrame(geometry=pzone_vector, crs=pzone_vector.crs)
            )
        else:
            raise TypeError(
                f"pzone_vector must be gu.Vector, GeoDataFrame, or GeoSeries (got {type(pzone_vector)})."
            )

        # Open the source raster lazily. Do NOT call set_nodata — it triggers a full load.
        rst = gu.Raster(mosaic_path, bands=selected_band, nodata=source_nodata)

        # Early-out: disjoint bounds → no work to do.
        src_xmin, src_ymin, src_xmax, src_ymax = pz_vec.get_bounds_projected(
            out_crs=rst.crs
        )
        rxmin, rymin, rxmax, rymax = (
            rst.bounds.left,
            rst.bounds.bottom,
            rst.bounds.right,
            rst.bounds.top,
        )
        if (
            (src_xmax <= rxmin)
            or (src_xmin >= rxmax)
            or (src_ymax <= rymin)
            or (src_ymin >= rymax)
        ):
            return None

        # Windowed crop in the source CRS (reads only the overlap from disk)
        try:
            m_crop = rst.crop(pz_vec, mode="match_pixel")
        except Exception as e:
            logger.error(f"Crop failed on {mosaic_path}: {e}")
            return None

        # Empty window after intersection rounding
        if m_crop is None or 0 in m_crop.shape[-2:]:
            return None

        shape_mismatch = canonical_grid_size is not None and m_crop.shape[-2:] != (
            canonical_grid_size[1],
            canonical_grid_size[0],
        )
        needs_reproject = (
            (m_crop.crs.to_epsg() != target_epsg)
            or (m_crop.res[0] != target_resolution)
            or shape_mismatch
        )
        if not needs_reproject:
            return m_crop

        # Estimate output array size to decide between in-memory and multiproc warp
        tgt_xmin, tgt_ymin, tgt_xmax, tgt_ymax = pz_vec.get_bounds_projected(
            out_crs=f"EPSG:{target_epsg}"
        )
        est_w = max(1, int((tgt_xmax - tgt_xmin) / target_resolution))
        est_h = max(1, int((tgt_ymax - tgt_ymin) / target_resolution))
        bytes_per_px = np.dtype(out_dtype).itemsize
        n_bands = 1 if isinstance(selected_band, int) else len(selected_band)
        est_mb = (est_w * est_h * n_bands * bytes_per_px) / (1024 * 1024)

        reproject_kwargs = dict(
            crs=f"EPSG:{target_epsg}",
            resampling=resampling_method,
            n_threads=n_threads,
            memory_limit=memory_limit_mb,
            dtype=out_dtype,
        )
        if canonical_grid_size is not None:
            # When grid_size is provided, bounds + grid_size define the output exactly
            # Do not specify res; geoutils will compute it from bounds/grid_size
            reproject_kwargs["bounds"] = canonical_bounds
            reproject_kwargs["grid_size"] = canonical_grid_size
        else:
            # Otherwise use resolution-based reprojection
            reproject_kwargs["res"] = target_resolution

        if est_mb > use_multiproc_threshold_mb:
            from geoutils.raster.distributed_computing import MultiprocConfig
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp.close()
            try:
                chunk = max(
                    512,
                    int(
                        (use_multiproc_threshold_mb * 1024 * 1024 / bytes_per_px) ** 0.5
                    ),
                )
                cfg = MultiprocConfig(chunk_size=chunk, outfile=tmp.name)
                m_crop.reproject(**reproject_kwargs, multiproc_config=cfg)
                m_final = gu.Raster(tmp.name)
                m_final.load()
            finally:
                try:
                    pathlib.Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            m_final = m_crop.reproject(**reproject_kwargs)

        del m_crop
        return m_final

    @gmc_profile("gmc.session.sieve_bulk", memprof=True)
    def sieve_bulk(
        self,
        georasters_map: gpd.GeoDataFrame | None = None,
        image_type: str = "opt",
        suffix: str = "",
        target_resolution: float = 1.5,
        resampling_method: str = "bilinear",
        selected_band: int | list[int] = 1,
        out_dtype: str = "uint16",
        skip_existing: bool = True,
        merge_algorithm: Callable = np.nanmean,
        n_threads: int = 4,
        memory_limit_mb: int = 1024,
        use_multiproc_threshold_mb: int = 2048,
        pzone_filter: list[str] | None = None,
        canonical_bounds: "object | None" = None,
        canonical_grid_size: "tuple[int, int] | None" = None,
        print_summary: bool = True,
        terrain_products: Sequence[str] = ("mdhillshade", "slope"),
        keep_dem: bool = True,
        hillshade_azimuth: float = 315.0,
        hillshade_altitude: float = 45.0,
        md_azimuths: Sequence[float] = DEFAULT_MD_AZIMUTHS,
        z_factor: float = 1.0,
        resolution_in_name: bool = True,
        mask_vector: "gu.Vector | gpd.GeoDataFrame | str | pathlib.Path | None" = None,
        mask_buffer_value: float = 0.0,
        mask_layer: str | None = None,
    ) -> dict[str, list[pathlib.Path]]:
        """Bulk sieve: crop all mosaics in ``georasters_map`` to each pzone AOI.

        For each pzone and each (acq_date, sensor) group, sequentially crops
        every intersecting mosaic via ``sieve_single_mosaic``, merges
        the cropped tiles if needed, and writes a single thumb to the pzone's
        raster_data directory. Sequential processing keeps peak memory bounded.

        Args:
            georasters_map: Georasters map (as returned by ``get_georasters_map``).
                If None, loaded from the session using ``image_type``/``suffix``.
            image_type: What the map holds and what is written:

                - ``"opt"`` — optical thumbs, ``<pz>/optical/<pz>_<date>_<sensor>.tif``.
                - ``"dem"`` — **multi-temporal DEMs**. Each ``(date, sensor)``
                  group is cropped, reprojected and merged onto the pzone's
                  canonical grid as float32, *then* each of *terrain_products*
                  is derived from that merged DEM and written to
                  ``<pz>/elevation/<product>/<pz>_<date>_<sensor>.tif`` (the DEM
                  itself to ``elevation/dem/`` when *keep_dem*). Deriving after
                  the merge, never per tile, keeps gradient artefacts off mosaic
                  seams and computes slope/shading at the resolution that is
                  correlated. Reads the ``Protomap_dem`` layer.
                - ``"reference_dem"`` — the single static correction DEM,
                  ``reference_dem/<pz>_dem.tif`` (what ``"dem"`` meant before
                  multi-temporal DEMs; :meth:`build_reference_dem` is the
                  usual route). Reads the ``Protomap_dem`` layer.
            suffix: Suffix on the georasters map name.
            target_resolution: Output pixel size in session CRS units.
            resampling_method: Rasterio resampling name.
            selected_band: Band index (or list) to read from each source raster.
            out_dtype: Output dtype for the written thumbs.
            skip_existing: If True, skip groups whose thumb file already exists.
            merge_algorithm: Reductor passed to ``merge_rasters`` when a pzone
                intersects multiple tiles for the same date/sensor.
            n_threads: Warp threads per reprojection.
            memory_limit_mb: Warp memory limit (MB).
            use_multiproc_threshold_mb: Threshold for switching to out-of-core
                reprojection.
            pzone_filter: Optional list of pzone names to restrict processing.
            canonical_bounds: Optional exact output bounds (rasterio BoundingBox).
                If provided, used for all pzones instead of computing per-pzone.
            canonical_grid_size: Optional exact output grid size (width, height).
                If provided, used for all pzones instead of computing per-pzone.
            terrain_products: ``"dem"`` only — products derived from each DEM,
                any of ``"hillshade"`` (single light), ``"mdhillshade"``
                (multidirectional) and ``"slope"`` (degrees). Hillshades are
                written as uint8 ``1..255`` with nodata 0, slope as float32 with
                nodata -9999.
            keep_dem: ``"dem"`` only — also write the cropped float32 DEM thumb.
            hillshade_azimuth: Light azimuth of ``"hillshade"`` (compass °).
            hillshade_altitude: Light altitude of both hillshades (°).
            md_azimuths: Light azimuths averaged by ``"mdhillshade"``.
            z_factor: Vertical exaggeration applied to both hillshades.
            resolution_in_name: Append the output resolution to the thumb name,
                ``<pz>_<date>_<sensor>_<res>m.tif`` (``3m``, ``10m``, ``1p5m``;
                the point is spelled ``p``). ``False`` keeps the old
                three-field names. Not applied to ``"reference_dem"``.
            mask_vector: Polygons (path, GeoDataFrame or ``gu.Vector``) of regions
                useless for correlation (forest, very high/low contrast...).
                Pixels inside are set to ``0`` in optical and hillshade thumbs
                (ASP ignores 0); float DEM/slope thumbs get their nodata
                (``-9999``), since 0 is a valid elevation/slope there. The mask
                is rasterised once per pzone on the output grid. A large file
                (e.g. a continental inventory) is cropped to the pzone grid plus
                the buffer *before* it is reprojected, per pzone, so only nearby
                polygons are read, reprojected and buffered. Existing
                thumbs are not rewritten: use ``skip_existing=False``.
            mask_buffer_value: Buffer (CRS units, metres) applied to the polygons
                before masking, as in ``vec.buffer(100).create_mask(raster)``.
            mask_layer: Layer name when *mask_vector* is a multi-layer file.

        Returns:
            Dict mapping pzone name to the list of thumb paths written (for
            ``"dem"``: every product file, skipped ones included).

        After sieving, run :meth:`update_thumbs` to register the thumbs, then
        :meth:`update_pairs` with ``image_type=<product>``.
        """
        assert (
            len(self.get_pzones_overview()) > 0
        ), "no pzone registered for this project"

        image_type = str(image_type).lower()
        if image_type not in ("opt", "dem", "reference_dem"):
            raise ValueError(
                f"Unsupported image_type: {image_type!r}; use 'opt', 'dem' or 'reference_dem'"
            )
        if image_type == "dem":
            terrain_products = tuple(terrain_products or ())
            bad = set(terrain_products) - set(TERRAIN_PRODUCTS)
            if bad:
                raise ValueError(
                    f"Unknown terrain product(s) {sorted(bad)}; expected any of {TERRAIN_PRODUCTS}"
                )
            if not terrain_products and not keep_dem:
                raise ValueError(
                    "Nothing to write: terrain_products is empty and keep_dem=False"
                )
            logger.info(
                "image_type='dem' writes dated DEM thumbs and their terrain products "
                "under elevation/; use 'reference_dem' (or build_reference_dem) for "
                "the correction DEM"
            )
            # Elevations must survive the crop/reproject/merge: a uint16 cast
            # would truncate decimals and wrap negative values.
            if not np.issubdtype(np.dtype(out_dtype), np.floating):
                logger.settings(f"out_dtype {out_dtype!r} → 'float32' for DEM sieving")
                out_dtype = "float32"
            terrain_params = dict(
                azimuth=hillshade_azimuth,
                altitude=hillshade_altitude,
                md_azimuths=tuple(md_azimuths),
                z_factor=z_factor,
            )

        if georasters_map is None:
            georasters_map = self.get_georasters_map(
                "dem" if image_type == "reference_dem" else image_type, suffix
            )
            if georasters_map is None:
                raise ValueError(
                    f"No raster-bank map for image_type={image_type!r}; run "
                    f"map_georasters_bank(image_type="
                    f"{'dem' if image_type != 'opt' else 'opt'!r}) first"
                )

        logger.info(f"Map epsg : {georasters_map.crs}")
        logger.info(f"Target resolution: {target_resolution} (EPSG:{self.epsg})")

        written: dict[str, list[pathlib.Path]] = {}

        if mask_vector is not None and image_type == "reference_dem":
            logger.warning("mask_vector is ignored for image_type='reference_dem'")
            mask_vector = None

        res_tag = (
            f"_{target_resolution:g}".replace(".", "p") + "m"
            if resolution_in_name
            else ""
        )

        for pz in self.get_pzones_overview().iloc:
            if pzone_filter is not None and pz["pz_name"] not in pzone_filter:
                continue

            logger.info(f"Pzone: {pz['pz_name']}")

            pz_vec = gu.Vector(
                gpd.GeoDataFrame(
                    [pz], crs=self.get_pzones_overview().crs, geometry="geometry"
                )
            )

            # Use provided canonical grid if given, otherwise compute per pzone
            if canonical_bounds is not None and canonical_grid_size is not None:
                canon_bounds, canon_size = canonical_bounds, canonical_grid_size
            else:
                canon_bounds, canon_size = self._compute_canonical_grid(
                    pz_vec, self.epsg, target_resolution
                )

            # Polygon mask on the output grid, built once per pzone (not per date).
            pz_mask = None
            if mask_vector is not None:
                # Crop first, reproject second: only the polygons near this pzone
                # (grid + buffer margin) are read and reprojected, so a continental
                # inventory costs what the pzone costs.  The buffer is applied after
                # reprojection, in metres, so the crop margin is only a pre-filter.
                mask_vec = self._normalize_vector_source(
                    mask_vector,
                    layer=mask_layer,
                    bounds=(
                        canon_bounds.left - mask_buffer_value,
                        canon_bounds.bottom - mask_buffer_value,
                        canon_bounds.right + mask_buffer_value,
                        canon_bounds.top + mask_buffer_value,
                    ),
                )
                if len(mask_vec.ds) and mask_buffer_value:
                    mask_vec = mask_vec.buffer(mask_buffer_value)
                _tmpl = gu.Raster.from_array(
                    np.zeros((canon_size[1], canon_size[0]), dtype="uint8"),
                    transform=rio_transform.from_bounds(
                        canon_bounds.left,
                        canon_bounds.bottom,
                        canon_bounds.right,
                        canon_bounds.top,
                        canon_size[0],
                        canon_size[1],
                    ),
                    crs=rio_crs.CRS.from_epsg(self.epsg),
                    nodata=None,
                )
                if len(mask_vec.ds) == 0:
                    pz_mask = np.zeros((canon_size[1], canon_size[0]), dtype=bool)
                else:
                    pz_mask = np.asarray(
                        np.ma.getdata(mask_vec.create_mask(_tmpl).data), dtype=bool
                    ).reshape(canon_size[1], canon_size[0])
                del mask_vec, _tmpl  # free the cropped polygons before the next pzone
                frac = float(pz_mask.mean())
                logger.info(f"Polygon mask on {pz['pz_name']}: {frac:.1%} of the grid")
                if frac == 0:
                    logger.info(
                        f"No mask polygon intersects {pz['pz_name']}: nothing masked"
                    )
                    pz_mask = None
                elif frac > 0.8:
                    logger.warning(f"mask_vector covers {frac:.0%} of the pzone grid")
                if skip_existing:
                    logger.info(
                        "mask_vector: existing thumbs are skipped, not re-masked "
                        "(use skip_existing=False)"
                    )

            pz_name = pz["pz_name"]
            if image_type == "opt":
                pz_opticals = self.pz_dir(pz_name, PZ_KIND_OPTICAL)
                pz_disps = self.pz_dir(pz_name, PZ_KIND_IMAGE_CORRELATION)
                for p in [pz_opticals, pz_disps]:
                    if not p.exists():
                        p.mkdir(parents=True)
            elif image_type == "dem":
                # (image type, folder) of every file one DEM group produces.
                dem_outputs = (
                    [("dem", self.thumb_dir(pz_name, "dem"))] if keep_dem else []
                ) + [(prod, self.thumb_dir(pz_name, prod)) for prod in terrain_products]
                for t, folder in dem_outputs:
                    folder.mkdir(parents=True, exist_ok=True)
                    self.pair_dir(pz_name, t).mkdir(parents=True, exist_ok=True)
            else:  # reference_dem
                pz_dem_dir = self.pz_dir(pz_name, PZ_KIND_REFERENCE_DEM)
                if not pz_dem_dir.exists():
                    pz_dem_dir.mkdir(parents=True)

            selection = georasters_map[
                georasters_map["geometry"].intersects(pz["geometry"])
            ]
            logger.info(f"Intersecting rasters: {len(selection)}")
            if len(selection) == 0:
                written[pz["pz_name"]] = []
                continue

            groups = selection.groupby(["acq_date", "sensor"])["file_path"].apply(list)
            pz_written: list[pathlib.Path] = []
            _summary_rows: list[dict] = []

            thumb_name_text = _RichText("", style="cyan")
            progress = Progress(
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=_rich_console,
            )
            task = progress.add_task("", total=len(groups))

            with _RichLive(
                _RichGroup(thumb_name_text, progress),
                console=_rich_console,
                transient=False,
                refresh_per_second=10,
            ) as live:
                for group_id, group in enumerate(groups):
                    acq_date = groups.index[group_id][0]
                    sensor = groups.index[group_id][1]

                    if image_type == "opt":
                        thumb_name = f"{pz['pz_name']}_{acq_date}_{sensor}{res_tag}.tif"
                        outputs = [
                            (IMAGE_TYPE_OPTICAL, pathlib.Path(pz_opticals, thumb_name))
                        ]
                    elif image_type == "dem":
                        # Same <pz>_<date>_<sensor> name in every product folder:
                        # the folder carries the type.
                        thumb_name = f"{pz['pz_name']}_{acq_date}_{sensor}{res_tag}.tif"
                        outputs = [
                            (t, pathlib.Path(folder, thumb_name))
                            for t, folder in dem_outputs
                        ]
                    else:  # reference_dem
                        thumb_name = f"{pz['pz_name']}_dem.tif"
                        outputs = [
                            ("reference_dem", pathlib.Path(pz_dem_dir, thumb_name))
                        ]

                    thumb_name_text.plain = thumb_name
                    live.refresh()

                    # skip_existing is judged per output file: an existing DEM
                    # whose slope is missing only computes the slope.
                    todo = [
                        (t, path)
                        for t, path in outputs
                        if not (skip_existing and path.exists())
                    ]
                    for t, path in outputs:
                        if (t, path) not in todo:
                            pz_written.append(path)
                    if not todo:
                        _summary_rows.append(
                            {
                                "name": thumb_name,
                                "sensor": sensor,
                                "status": "↷ skipped",
                            }
                        )
                        progress.advance(task)
                        live.refresh()
                        continue

                    # Cropping phase
                    fully: list[gu.Raster] = []
                    for mosaic_path in group:
                        try:
                            cropped = self.sieve_single_mosaic(
                                pzone_vector=pz_vec,
                                mosaic_path=mosaic_path,
                                target_resolution=target_resolution,
                                target_epsg=self.epsg,
                                resampling_method=resampling_method,
                                selected_band=selected_band,
                                out_dtype=out_dtype,
                                n_threads=n_threads,
                                memory_limit_mb=memory_limit_mb,
                                use_multiproc_threshold_mb=use_multiproc_threshold_mb,
                                canonical_bounds=canon_bounds,
                                canonical_grid_size=canon_size,
                            )
                            if cropped is not None:
                                fully.append(cropped)
                        except Exception as e:
                            logger.error(f"Error on {mosaic_path}: {e}")

                    if len(fully) == 0:
                        logger.warning(f"No usable data for {thumb_name}")
                        _summary_rows.append(
                            {
                                "name": thumb_name,
                                "sensor": sensor,
                                "status": "✗ no data",
                            }
                        )
                        progress.advance(task)
                        live.refresh()
                        continue
                    elif len(fully) == 1:
                        thumb = fully[0]
                    else:
                        thumb = gu.raster.multiraster.merge_rasters(
                            fully,
                            merge_algorithm=merge_algorithm,
                            resampling_method=resampling_method,
                            use_ref_bounds=False,
                        )

                    if image_type == "dem":
                        pz_written.extend(
                            self._write_dem_products(
                                thumb, todo, terrain_params, mask=pz_mask
                            )
                        )
                        done = ", ".join(t for t, _ in todo)
                        _summary_rows.append(
                            {
                                "name": thumb_name,
                                "sensor": sensor,
                                "status": f"✓ {done}",
                            }
                        )
                    else:
                        thumb_path = todo[0][1]
                        if pz_mask is not None:
                            # 0 is ASP's default ignored value; make it the nodata too.
                            data = np.ma.getdata(thumb.data).copy()
                            data[..., pz_mask] = 0
                            thumb = gu.Raster.from_array(
                                np.ma.masked_array(
                                    data, mask=np.ma.getmaskarray(thumb.data) | pz_mask
                                ),
                                transform=thumb.transform,
                                crs=thumb.crs,
                                nodata=0,
                            )
                        thumb.to_file(thumb_path, dtype=out_dtype)
                        pz_written.append(thumb_path)
                        _summary_rows.append(
                            {"name": thumb_name, "sensor": sensor, "status": "✓ done"}
                        )
                    del thumb, fully
                    progress.advance(task)
                    live.refresh()

            written[pz["pz_name"]] = pz_written

        if print_summary and _summary_rows:
            print_sieve_bulk_summary(
                _summary_rows, target_resolution, print_summary=print_summary
            )

        return written

    #: Nodata written into DEM-derived float thumbs (DEM and slope).  Hillshades
    #: are uint8 and use 0, like the optical thumbs.
    DEM_FLOAT_NODATA: float = -9999.0

    @classmethod
    def _write_dem_products(
        cls,
        dem: gu.Raster,
        todo: list[tuple[str, pathlib.Path]],
        terrain_params: dict,
        mask: "np.ndarray | None" = None,
    ) -> list[pathlib.Path]:
        """Write the DEM and/or its terrain products for one sieved DEM group.

        *dem* is already on the pzone's canonical grid; every product is derived
        from it with the true pixel size, so all dates share one lattice.

        Each file is built with ``gu.Raster.from_array(..., nodata=…)`` rather
        than ``dem.copy()``, so it carries its own dtype and nodata instead of
        inheriting the DEM's.

        *mask* (True = blank) is applied to each **derived** product, so slope
        and shading are computed on the full DEM and the polygon edge is not
        contaminated: hillshades get 0, DEM/slope their float nodata.
        """
        arr = np.ma.masked_invalid(np.ma.asarray(dem.data, dtype=np.float64))
        if arr.ndim == 3:
            arr = arr[0]
        z = arr.filled(np.nan)
        res_x, res_y = abs(float(dem.res[0])), abs(float(dem.res[1]))
        nd = cls.DEM_FLOAT_NODATA

        written = []
        for t, path in todo:
            if t == "dem":
                data = np.where(np.isfinite(z), z, nd).astype(np.float32)
                nodata = nd
            else:
                prod = terrain_product(t, z, res_x, res_y, **terrain_params)
                if t in ("hillshade", "mdhillshade"):
                    data, nodata = shade_to_uint8(prod), 0
                else:  # slope, degrees
                    data = np.where(np.isfinite(prod), prod, nd).astype(np.float32)
                    nodata = nd
            if mask is not None:
                data = np.where(mask, nodata, data).astype(data.dtype)
            # Masked explicitly: handing geoutils a plain array that contains
            # the nodata value makes it warn and mask it anyway.
            out = gu.Raster.from_array(
                np.ma.masked_equal(data, nodata),
                transform=dem.transform,
                crs=dem.crs,
                nodata=nodata,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_file(path)
            written.append(path)
        return written

    def save_correlation_figure(
        self,
        view: str = "design_map",
        *,
        frame: pd.DataFrame | None = None,
        criterias: str | list[str] = "",
        formats: str | list[str] | tuple[str, ...] = ("html", "png"),
        stem: str | None = None,
        subdir: str = "correlation",
        title: str = "",
        dpi: int = 300,
        figsize: tuple[float, float] | None = None,
        plotlyjs: bool | str = True,
        overwrite: bool = True,
        pz_name: str | None = None,
        **view_kwargs,
    ) -> dict:
        """Save a correlation-parameters figure.

        Same routing as :meth:`save_pairs_figure`: for new-layout sessions,
        ``<pz_name>/figures/<subdir>/`` when the frame holds exactly one pzone
        or *pz_name* is given; otherwise the project-wide figures folder.
        Legacy-layout sessions always get the project-wide folder, with a
        warning when *pz_name* was supplied — ``pz_dir`` discards it there, so
        pretending to route per pzone would be a lie.

        :param view: ``"design_map"`` | ``"kernel_search"`` | ``"snr"`` | ``"cost"``.
        :param frame: The frame to draw. None derives one from the committed
            pairs using the last explorer settings.
        :param criterias: Passed to :meth:`get_pairs` when *frame* is None.
        :param formats: Any of ``html``, ``png``, ``jpg``, ``pdf``, ``svg``.
        :param stem: File stem; a deterministic one is built when omitted, so
            re-running refreshes the same files instead of piling up copies.
        :param subdir: Subdirectory under the figures folder.
        :param title: Figure title, applied to both backends.
        :param dpi: Raster resolution for the matplotlib output.
        :param figsize: Matplotlib figure size; a per-view default otherwise.
        :param plotlyjs: True inlines plotly.js (offline-capable).
        :param overwrite: Replace a file of the same name (the default).
        :param pz_name: Explicitly route to this pzone's figures folder.
        :param view_kwargs: View options, filtered per backend.
        :returns: ``{format: Path}`` for every file written.
        """
        from geomulticorr.utils._corrparams_export import (
            corrparams_figure_stem,
            save_corrparams_figure,
        )

        if frame is None:
            frame = self._corrparams_frame(criterias=criterias)

        if pz_name is not None:
            if self._legacy_layout:
                logger.warning(
                    f"pz_name={pz_name!r} ignored for legacy-layout session; "
                    f"using project-wide figures folder"
                )
                out_dir = self.path_figures / subdir
            else:
                out_dir = self.pz_dir(pz_name, PZ_KIND_FIGURES) / subdir
        elif not self._legacy_layout and "pz" in frame.columns and len(frame):
            distinct = [p for p in frame["pz"].unique() if p]
            out_dir = (
                self.pz_dir(distinct[0], PZ_KIND_FIGURES) / subdir
                if len(distinct) == 1
                else self.path_figures / subdir
            )
        else:
            out_dir = self.path_figures / subdir

        if stem is None:
            assumptions = getattr(self, "_last_corr_assumptions", None) or {}
            pz_for_stem = pz_name or ""
            if not pz_for_stem and len(frame) and "pz" in frame.columns:
                distinct = [p for p in frame["pz"].unique() if p]
                pz_for_stem = distinct[0] if len(distinct) == 1 else ""
            stem = corrparams_figure_stem(view, pz_name=pz_for_stem, **assumptions)

        paths = save_corrparams_figure(
            frame,
            out_dir,
            view=view,
            formats=formats,
            stem=stem,
            title=title,
            dpi=dpi,
            figsize=figsize,
            plotlyjs=plotlyjs,
            view_kwargs=view_kwargs,
            overwrite=overwrite,
        )
        logger.success(f"Saved {view} figure to {out_dir} ({', '.join(paths)}).")
        return paths

    def _corrparams_pair_facts(
        self,
        criterias: str | list[str] = "",
        pz_name: str = "",
        image_type: str | None = None,
    ) -> list[dict]:
        """Per-pair facts the derivation needs, read once — **from tables only**.

        Reads :meth:`get_pairs_overview` and the Thumbs layer, and constructs no
        ``Pair`` and no ``Thumb``.  That is the whole point: ``Thumb.__init__``
        computes ``ras.footprint``, a densified ~20 000-vertex polygon costing
        ~150 ms, and ``get_pairs()`` builds two Thumbs per pair — about 140 s on
        a 460-pair archive, which is what made this explorer slow to open.  The
        Pairs layer already carries every fact needed except resolution, and the
        ``th_res`` column supplies that.

        Called once at setup, so every subsequent control change is pure
        arithmetic — the same discipline ``explore_pairs_strategy`` applies to
        ``get_valid_thumbs()``.

        :param criterias: Passed to :meth:`get_pairs_overview`.
        :param pz_name: Restrict to one pzone; ``""`` for all.
        :param image_type: Restrict to one image type; ``None`` for all.
            Groups are ``(sensor, Δt bin)``, so they only mean something
            within one type — and the plan is keyed by ``pa_key``, which is
            unique only within one type.
        :returns: One dict per pair, ready for
            :func:`~geomulticorr.utils._corrparams_frame.corrparams_frame_from_pairs`.
        """
        overview = self.get_pairs_overview(criterias)
        if overview is None or len(overview) == 0:
            return []
        if pz_name:
            overview = overview[overview["pa_pz_name"] == pz_name]
        if image_type is not None:
            overview = overview[
                type_column(overview, "pa_type") == normalise_image_type(image_type)
            ]

        resolutions = self._thumb_resolution_lookup()

        facts: list[dict] = []
        for _, row in overview.iterrows():
            pz = row["pa_pz_name"]
            left_res = self._resolution_for(
                resolutions, pz, row["pa_left_date"], row["pa_left_sensor"]
            )
            right_res = self._resolution_for(
                resolutions, pz, row["pa_right_date"], row["pa_right_sensor"]
            )
            # pa_path is built as ".../image_correlation/{pa_key}", so the
            # basename IS the key by construction.  Re-formatting it from the
            # thumb components would be a second source of truth that
            # drifts the moment Pair's key format changes.
            pa_key = pathlib.Path(str(row["pa_path"])).name
            if (
                np.isfinite(left_res)
                and np.isfinite(right_res)
                and abs(left_res - right_res) > 0.01 * max(left_res, right_res)
            ):
                # ASP reports disparity in LEFT-image pixels, so the left GSD is
                # the one every formula uses; say so rather than silently
                # averaging two different grids.
                logger.warning(
                    f"{pa_key}: left and right resolutions differ "
                    f"({left_res:g} vs {right_res:g} m) — using the left image's "
                    f"GSD, since ASP disparity is in left-image pixels"
                )
            facts.append(
                {
                    "pa_key": pa_key,
                    "pz": pz,
                    "dt_days": float(row["pa_dt_days"]),
                    "resolution_m": left_res,
                    "sensor_i": row["pa_left_sensor"],
                    "sensor_j": row["pa_right_sensor"],
                }
            )
        return facts

    def _thumb_resolution_lookup(self) -> dict[tuple[str, str, str], float]:
        """``{(pzone, date, sensor): resolution}`` from the Thumbs layer.

        Built once per explorer setup. Rows whose ``th_res`` is missing — an
        older geodatabase, or one where :meth:`backfill_thumb_resolution` has not
        been run — fall back to reading the file header, which is bounded by the
        number of *unique thumbs* rather than the number of pairs.

        :returns: The lookup; missing thumbs are simply absent.
        """
        thumbs = self._thumbs
        if thumbs is None or len(thumbs) == 0:
            return {}

        has_column = "th_res" in thumbs.columns
        lookup: dict[tuple[str, str, str], float] = {}
        n_from_header = 0

        for _, row in thumbs.iterrows():
            value = float("nan")
            if has_column:
                try:
                    value = float(row["th_res"])
                except (TypeError, ValueError):
                    value = float("nan")
            if not np.isfinite(value):
                value = self._thumb_resolution(row["th_path"])
                n_from_header += 1
            lookup[(row["th_pz_name"], row["th_date"], row["th_sensor"])] = value

        if n_from_header:
            # Once, naming the fix — not once per pair.
            logger.info(
                f"{n_from_header} thumb(s) have no stored resolution; read from the "
                f"file headers. Run session.backfill_thumb_resolution() to persist it."
            )
        return lookup

    @staticmethod
    def _resolution_for(
        lookup: dict[tuple[str, str, str], float],
        pz_name: str,
        date: str,
        sensor: str,
    ) -> float:
        """Look one thumb's resolution up, or ``nan`` when it is not registered."""
        return lookup.get((pz_name, str(date), sensor), float("nan"))

    def _corrparams_frame(
        self,
        criterias: str | list[str] = "",
        pz_name: str = "",
        image_type: str | None = None,
        **assumptions,
    ) -> pd.DataFrame:
        """Derive a correlation-parameters frame from the committed pairs.

        The headless counterpart of the explorer's cached path: convenient for
        scripts, but it re-reads the thumb headers on every call, so a parameter
        sweep should go through :meth:`explore_correlation_params` instead.

        :param criterias: Passed to :meth:`get_pairs`.
        :param pz_name: Restrict to one pzone.
        :param assumptions: Derivation inputs; the last explorer settings are
            used for anything omitted.
        :returns: A correlation-parameters frame.
        """
        from geomulticorr.utils._corrparams_frame import (
            corrparams_frame_from_pairs,
            empty_corrparams_frame,
        )

        facts = self._corrparams_pair_facts(criterias, pz_name, image_type)
        if not facts:
            return empty_corrparams_frame()

        merged = dict(getattr(self, "_last_corr_assumptions", None) or {})
        merged.update(assumptions)
        return corrparams_frame_from_pairs(
            [f["pa_key"] for f in facts],
            [f["dt_days"] for f in facts],
            [f["resolution_m"] for f in facts],
            pz=[f["pz"] for f in facts],
            sensor_i=[f["sensor_i"] for f in facts],
            sensor_j=[f["sensor_j"] for f in facts],
            **merged,
        )

    def explore_correlation_params(
        self,
        criterias: str | list[str] = "",
        pz_name: str = "",
        view: str = "design_map",
        interactive: bool = True,
        savefig: bool = True,
        formats: str | list[str] | tuple[str, ...] = ("html", "png"),
        plan_name: str = "default",
        write_plan: bool = True,
        image_type: str | None = None,
        **defaults,
    ):
        """Choose ASP correlation parameters for a whole archive, interactively.

        A pair measures ``v * Δt``, and one ASP parameter set spans only
        ``S / (k * c)`` — roughly 30-100x — so a single setting cannot serve a
        wide archive.  This derives parameters per **group**
        ``(sensor, Δt band)``, shows the feasibility band the choice implies, and
        writes a replayable plan.  See ``docs/correlation_parameters.md``.

        Commit the result with::

            session.explore_correlation_params(velocity_m_yr=2.0)
            session.prepare_pairs_correlation(**session._last_corr_params,
                                              overwrite_scripts=True)

        :param criterias: Passed to :meth:`get_pairs` to filter pairs.
        :param pz_name: Restrict to one pzone; ``""`` for all.
        :param view: Initial view — a key of
            :data:`~geomulticorr.utils._corrparams_plotly.VIEW_BUILDERS`.
        :param interactive: False returns ``(frame, fig)`` and imports no
            ipywidgets, for scripts, batch jobs and CLI entry points.
        :param savefig: In headless mode, also write the figure. Ignored when
            interactive — there is a **Save figure** button instead.
        :param formats: Any of ``html``, ``png``, ``jpg``, ``pdf``, ``svg``.
        :param plan_name: Name of the correlation plan written.
        :param write_plan: Write the plan JSON. False derives and plots only.
        :param image_type: Restrict to one image type (``"opt"``,
            ``"mdhillshade"``, …); ``None`` keeps every pair. Pass it whenever
            the Pairs layer holds more than one type: the plan is keyed by
            ``pa_key``, and ``prepare_pairs_correlation`` refuses a plan whose
            keys occur in several types.
        :param defaults: Initial values; only
            :attr:`_CORRPARAMS_EXPLORER_DEFAULTS` keys are accepted, anything
            else raises rather than being silently ignored.
        :returns: A widget, or ``(frame, fig)`` when *interactive* is False.
        :raises ValueError: On an unknown *view*.
        :raises TypeError: On an unknown key in *defaults*.
        """
        from geomulticorr.correlation.corr_params import (
            TEXTURE_FLOOR_PX,
            geometric_dt_bins,
            parse_search_box,
            to_odd,
        )
        from geomulticorr.utils import _corrparams_plotly as corr_plotly
        from geomulticorr.utils._corrparams_frame import (
            corrparams_frame_from_pairs,
            corrparams_stats,
            empty_corrparams_frame,
            format_corrparams_summary,
            group_table,
        )

        if view not in corr_plotly.VIEW_BUILDERS:
            raise ValueError(
                f"Unknown view '{view}'. Valid options: "
                f"{list(corr_plotly.VIEW_BUILDERS)}."
            )
        unknown = set(defaults) - self._CORRPARAMS_EXPLORER_DEFAULTS
        if unknown:
            raise TypeError(
                f"Unknown keyword(s) {sorted(unknown)}. Accepted: "
                f"{sorted(self._CORRPARAMS_EXPLORER_DEFAULTS)}."
            )

        # ── expensive setup, ONCE ─────────────────────────────────────────── #
        facts = self._corrparams_pair_facts(criterias, pz_name, image_type)

        initial = {
            "velocity_m_yr": 1.0,
            "footprint_m": 60.0,
            "c_px": 0.15,
            "k": 3.0,
            "coreg_px": 2.0,
            "margin_px": 2.0,
            "flow_azimuth_deg": None,
            "strain_rate_per_yr": None,
            "corr_algorithm": "asp_bm",
            "subpixel_mode": 2,
            "tau_days": None,
            "dt_bin_ratio": 3.0,
            "explicit_search": True,
            "explicit_search_above_days": 365,
            "uniform_kernel": False,
            "uniform_kernel_px": 21,
            "uniform_search": False,
            "uniform_search_box": "20",
            "sensor_filter": "",
            "max_dt_days": None,
            "min_dt_days": None,
            "color_by": "group",
        }
        initial.update(defaults)
        # "1Y"/"6M"/"30D" from a script, not only from the widget.
        for _key in ("max_dt_days", "min_dt_days", "explicit_search_above_days"):
            initial[_key] = parse_dt_days(initial[_key])

        # ── pure closures, shared by both modes ───────────────────────────── #
        def _selected(sfilter: str | None, max_dt, min_dt) -> list[dict]:
            """Filter the cached facts. Pure — no I/O, no widget reads."""
            rows = facts
            if sfilter:
                needle = str(sfilter).lower()
                rows = [
                    f
                    for f in rows
                    if needle in str(f["sensor_i"]).lower()
                    or needle in str(f["sensor_j"]).lower()
                ]
            if max_dt is not None:
                rows = [f for f in rows if f["dt_days"] <= float(max_dt)]
            if min_dt is not None:
                rows = [f for f in rows if f["dt_days"] >= float(min_dt)]
            return rows

        def _derive(rows, group_params, **assumptions) -> pd.DataFrame:
            """One derivation pass over *rows*. Pure."""
            return corrparams_frame_from_pairs(
                [f["pa_key"] for f in rows],
                [f["dt_days"] for f in rows],
                [f["resolution_m"] for f in rows],
                pz=[f["pz"] for f in rows],
                sensor_i=[f["sensor_i"] for f in rows],
                sensor_j=[f["sensor_j"] for f in rows],
                group_params=group_params,
                **assumptions,
            )

        def _build_frame(sfilter, max_dt, min_dt, uniform=None, **assumptions):
            """Derive the frame **and** the group parameters it commits to. Pure.

            Two passes, and the second one matters: the first derives each
            pair's ideal parameters, then those are reduced to one set per
            group, and the frame is rebuilt with the reduced values.

            Without the second pass the figures would show the per-pair ideal
            while the files on disk carried the group's reduction — the design
            map and the group table would disagree with every
            ``*_CorrParameters.txt``, and replaying the plan would not
            reproduce the frame it came from.

            :param uniform: Uniform overrides, or None for the pure derivation.
            :returns: ``(frame, groups)``; the frame reflects what will run.
            """
            rows = _selected(sfilter, max_dt, min_dt)
            if not rows:
                return empty_corrparams_frame(), {}

            ideal = _derive(rows, None, **assumptions)
            groups = _group_params(ideal, assumptions, uniform or {})
            return _derive(rows, groups, **assumptions), groups

        def _group_params(
            frame: pd.DataFrame, assumptions: dict, uniform: dict
        ) -> dict:
            """One ASP parameter set per group, from the per-pair derivation.

            Uniform overrides replace the derived value here rather than inside
            ``suggest_parameters``: the derivation still runs in full, so its
            warnings — and the per-pair conflicts the override creates — survive
            into the frame's ``flags``.
            """
            if len(frame) == 0:
                return {}

            fixed_kernel = (
                to_odd(
                    int(uniform.get("uniform_kernel_px", 21)), minimum=TEXTURE_FLOOR_PX
                )
                if uniform.get("uniform_kernel")
                else None
            )
            fixed_box = (
                uniform.get("_parsed_search_box")
                if uniform.get("uniform_search")
                else None
            )

            def _kernel_for(row) -> int:
                return fixed_kernel if fixed_kernel is not None else int(row.kernel_px)

            def _box_for(row):
                # The Δt threshold still gates WHETHER a box is used; the
                # uniform value only decides WHAT it is.  A group entirely below
                # the threshold stays on ASP auto even when uniform is on.
                derived_box = _search_box(frame, row.group, assumptions)
                if derived_box is None:
                    return None
                return fixed_box if fixed_box is not None else derived_box

            return {
                row.group: {
                    "params": {
                        "corr_kernel": (_kernel_for(row), _kernel_for(row)),
                        "corr_search": _box_for(row),
                        "corr_algorithm": assumptions["corr_algorithm"],
                        "subpixel_mode": int(assumptions["subpixel_mode"]),
                        "subpixel_kernel": (_kernel_for(row), _kernel_for(row)),
                        # Seed mode moves with the box: 0 when an explicit range
                        # is given, 1 when ASP is left to estimate it — that
                        # stage is what produces the estimate (guide 6.1).
                        "corr_seed_mode": 0 if _box_for(row) is not None else 1,
                    },
                    "n_pairs": int(row.n_pairs),
                    "dt_days": [int(row.dt_min_days), int(row.dt_max_days)],
                    "resolution_m": float(row.resolution_m),
                }
                for row in group_table(frame).itertuples()
            }

        def _view_defaults(view_key: str, assumptions: dict, color_by: str) -> dict:
            """Figure options for *view_key*. The non-interactive twin of
            ``_view_kwargs``; both must return the same shape or the two modes
            would render differently."""
            if view_key == "design_map":
                return {
                    "k": assumptions["k"],
                    "c_px": assumptions["c_px"],
                    "coreg_px": assumptions["coreg_px"],
                    "margin_px": assumptions["margin_px"],
                    "tau_days": assumptions["tau_days"],
                    "color_by": color_by,
                }
            if view_key == "kernel_search":
                return {"strain_rate_per_yr": assumptions["strain_rate_per_yr"]}
            if view_key == "snr":
                return {"k": assumptions["k"]}
            return {}

        def _title_for(frame: pd.DataFrame, view_key: str, assumptions: dict) -> str:
            noun = self._CORR_DRAWN_NOUN.get(view_key, "pairs")
            n = frame["group"].nunique() if view_key == "cost" else len(frame)
            return (
                f"Correlation parameters — {n} {noun} "
                f"[v={assumptions['velocity_m_yr']:g} m/yr, "
                f"L={assumptions['footprint_m']:g} m]"
            )

        def _assumptions_from(source: dict) -> dict:
            return {key: source[key] for key in self._CORR_ASSUMPTION_KEYS}

        def _uniform_from(source: dict) -> dict:
            """The uniform overrides from a defaults dict, box already parsed.

            Headless callers get the ``ValueError`` — only the widget degrades,
            because only there would an exception be swallowed by a hidden log.
            """
            out = {key: source[key] for key in self._CORR_UNIFORM_KEYS}
            out["_parsed_search_box"] = (
                parse_search_box(source["uniform_search_box"])
                if source["uniform_search"]
                else None
            )
            return out

        def _stash_params(
            frame: pd.DataFrame,
            groups: dict,
            assumptions: dict,
            sfilter,
            max_dt,
            min_dt,
            uniform: dict | None = None,
        ) -> dict:
            """Write the stash and the plan. Returns the plan document.

            NOTE: ``_last_corr_params`` carries exactly ``_CORR_PARAM_KEYS`` —
            the keys ``prepare_pairs_correlation(**params)`` accepts.  The
            derivation inputs live in ``_last_corr_assumptions`` and in the plan
            instead: ``prepare_pairs_correlation`` knows nothing about
            velocities, and adding one here would make the documented commit
            workflow raise ``TypeError``.
            """
            from geomulticorr.correlation._correlation_plan import (
                build_correlation_plan,
            )
            from geomulticorr.utils._corrparams_frame import relevant_corr_keys

            edges, _labels = (
                geometric_dt_bins(
                    frame["dt_days"].tolist(), ratio=assumptions["dt_bin_ratio"]
                )
                if len(frame)
                else ([], [])
            )

            # Only claim a pzone when the frame really holds exactly one, the
            # same test save_correlation_figure applies.  Taking .iloc[0] filed
            # a multi-pzone plan under whichever pzone happened to sort first,
            # and made the "no single pzone" guard in _maybe_write_plan
            # unreachable for any non-empty frame.
            distinct_pz = [p for p in frame["pz"].unique() if p] if len(frame) else []
            # The uniform flags belong in the trace: a plan that recorded only
            # the resolved numbers would not say WHY every group shares them.
            recorded = dict(assumptions)
            recorded.update(
                {
                    key: value
                    for key, value in (uniform or {}).items()
                    if not key.startswith("_")
                }
            )
            plan = build_correlation_plan(
                name=plan_name,
                pzone=pz_name or (distinct_pz[0] if len(distinct_pz) == 1 else ""),
                assumptions=recorded,
                groups=groups,
                pairs=[
                    {
                        "pa_key": row.pa_key,
                        "pz": row.pz,
                        "dt_days": int(row.dt_days),
                        "resolution_m": float(row.resolution_m),
                        "sensor_i": row.sensor_i,
                        "sensor_j": row.sensor_j,
                        "group": row.group,
                    }
                    for row in frame.itertuples()
                ],
                relevant_params=sorted(
                    relevant_corr_keys(
                        assumptions["corr_algorithm"],
                        flow_azimuth_deg=assumptions["flow_azimuth_deg"],
                        strain_rate_per_yr=assumptions["strain_rate_per_yr"],
                        explicit_search=bool(assumptions["explicit_search"]),
                        uniform_kernel=bool((uniform or {}).get("uniform_kernel")),
                        uniform_search=bool((uniform or {}).get("uniform_search")),
                    )
                ),
                dt_bin_edges=edges,
            )

            self._last_corr_assumptions = dict(recorded)
            self._last_corr_filters = {
                "sensor_filter": sfilter,
                "max_dt_days": max_dt,
                "min_dt_days": min_dt,
            }
            self._last_corr_plan = plan
            self._last_corr_params = {"plan": plan, "params_by_pair": None}
            return plan

        def _search_box(
            frame: pd.DataFrame, group: str, assumptions: dict
        ) -> tuple[int, int, int, int]:
            """The group's search box: sized for its widest-reaching pair.

            A box sized for the median pair in the bin would rail on the longest
            baseline sharing that bin.

            Returns ``None`` when **no** pair in the group reached the explicit
            threshold — the whole group stays on ASP's automatic range, and
            fabricating a box for it would substitute a guessed velocity for
            ASP's own estimate.

            Takes *assumptions* explicitly rather than closing over ``initial``:
            reading the initial values here would silently ignore every
            allowance the user changed after opening the explorer, and the
            plan's boxes would then disagree with the frame's ``search_px``.
            """
            from geomulticorr.correlation.corr_params import required_search_px

            sub = frame[frame["group"] == group]
            if len(sub) == 0:
                return None
            # search_px is nan for pairs left on auto; a group with none left is
            # entirely below the threshold.
            explicit = sub[np.isfinite(sub["search_px"].to_numpy(dtype="float64"))]
            if len(explicit) == 0:
                return None
            worst = explicit.loc[explicit["dt_days"].idxmax()]
            return required_search_px(
                float(worst["velocity_m_yr"]),
                float(worst["dt_days"]),
                float(worst["resolution_m"]),
                coreg_px=assumptions["coreg_px"],
                margin_px=assumptions["margin_px"],
                flow_azimuth_deg=assumptions["flow_azimuth_deg"],
            )

        def _maybe_write_plan(plan: dict) -> None:
            """Write the plan JSON when asked and a destination is resolvable."""
            if not write_plan:
                return
            from geomulticorr.correlation._correlation_plan import (
                plan_path,
                write_correlation_plan,
            )

            target_pz = plan.get("pzone") or ""
            if not target_pz:
                logger.info(
                    "No single pzone in the frame — correlation plan not written."
                )
                return
            try:
                directory = self.pz_dir(target_pz, PZ_KIND_IMAGE_CORRELATION)
            except (ValueError, KeyError) as exc:
                logger.warning(f"Could not resolve a plan directory: {exc}")
                return
            write_correlation_plan(plan_path(directory, plan_name), plan)

        # ── headless path — imports no ipywidgets, needs no display ───────── #
        if not interactive:
            assumptions = _assumptions_from(initial)
            uniform = _uniform_from(initial)
            frame, groups = _build_frame(
                initial["sensor_filter"],
                initial["max_dt_days"],
                initial["min_dt_days"],
                uniform=uniform,
                **assumptions,
            )
            plan = _stash_params(
                frame,
                groups,
                assumptions,
                initial["sensor_filter"],
                initial["max_dt_days"],
                initial["min_dt_days"],
                uniform,
            )
            _maybe_write_plan(plan)

            view_kwargs = _view_defaults(view, assumptions, initial["color_by"])
            title = _title_for(frame, view, assumptions)
            fig = corr_plotly.VIEW_BUILDERS[view](frame, title=title, **view_kwargs)
            if savefig and len(frame):
                self.save_correlation_figure(
                    view=view,
                    frame=frame,
                    formats=formats,
                    title=title,
                    **view_kwargs,
                )
            logger.info(
                format_corrparams_summary(
                    corrparams_stats(frame, k=assumptions["k"]), html=False
                )
            )
            return frame, fig

        # ── widgets (imported only past the headless return) ──────────────── #
        import ipywidgets as widgets

        from IPython.display import clear_output, display as _ipy_display

        from geomulticorr.utils._pairs_export import FIGURE_FORMATS

        c_view = widgets.Dropdown(
            options=[(label, key) for key, label in corr_plotly.VIEW_LABELS.items()],
            value=view,
            description="View:",
            style={"description_width": "initial"},
        )
        c_velocity = widgets.FloatText(
            value=initial["velocity_m_yr"],
            description="velocity (m/yr):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        c_footprint = widgets.FloatText(
            value=initial["footprint_m"],
            description="template L (m):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        c_c = widgets.FloatText(
            value=initial["c_px"],
            step=0.01,
            description="precision c (px):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        c_k = widgets.FloatSlider(
            value=initial["k"],
            min=1.0,
            max=5.0,
            step=0.5,
            description="k:",
            style={"description_width": "initial"},
            continuous_update=False,
        )
        c_coreg = widgets.FloatText(
            value=initial["coreg_px"],
            description="coreg (px):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="170px"),
        )
        c_margin = widgets.FloatText(
            value=initial["margin_px"],
            description="margin (px):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="170px"),
        )
        c_azimuth = widgets.FloatText(
            value=(
                initial["flow_azimuth_deg"]
                if initial["flow_azimuth_deg"] is not None
                else -1.0
            ),
            description="flow az (° from N, <0 = off):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="290px"),
        )
        c_strain = widgets.FloatText(
            value=initial["strain_rate_per_yr"] or 0.0,
            step=0.005,
            description="strain (1/yr, 0 = off):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="260px"),
        )
        c_tau = widgets.FloatText(
            value=initial["tau_days"] or 0.0,
            description="τ (days, 0 = off):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="230px"),
        )
        c_explicit = widgets.Checkbox(
            value=bool(initial["explicit_search"]),
            description="explicit --corr-search",
            indent=False,
            layout=widgets.Layout(width="220px"),
        )
        c_explicit_above = widgets.Text(
            value=format_dt_days(initial["explicit_search_above_days"]),
            description="above Δt:",
            placeholder="e.g. 1Y (empty = all pairs)",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="250px"),
        )
        c_uniform_kernel = widgets.Checkbox(
            value=bool(initial["uniform_kernel"]),
            description="uniform kernel",
            indent=False,
            layout=widgets.Layout(width="170px"),
        )
        c_uniform_kernel_px = widgets.IntText(
            value=int(initial["uniform_kernel_px"]),
            description="px:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="120px"),
        )
        c_uniform_search = widgets.Checkbox(
            value=bool(initial["uniform_search"]),
            description="uniform search box",
            indent=False,
            layout=widgets.Layout(width="200px"),
        )
        c_uniform_search_box = widgets.Text(
            value=str(initial["uniform_search_box"]),
            description="box:",
            placeholder="20  or  -80 -2 20 2",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="230px"),
        )
        c_ratio = widgets.FloatSlider(
            value=initial["dt_bin_ratio"],
            min=1.5,
            max=10.0,
            step=0.5,
            description="Δt bin ratio:",
            style={"description_width": "initial"},
            continuous_update=False,
        )
        c_algorithm = widgets.Dropdown(
            options=["asp_bm", "asp_sgm", "asp_mgm", "asp_final_mgm"],
            value=initial["corr_algorithm"],
            description="algorithm:",
            style={"description_width": "initial"},
        )
        c_subpixel = widgets.Dropdown(
            options=[
                ("2 — Bayes EM (no pixel-locking)", 2),
                ("1 — parabola", 1),
                ("3 — affine", 3),
                ("0 — none", 0),
            ],
            value=initial["subpixel_mode"],
            description="subpixel:",
            style={"description_width": "initial"},
        )
        c_sensor = widgets.Text(
            value=initial["sensor_filter"] or "",
            description="sensor:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        # Text, not IntText: "1Y"/"6M"/"30D" as well as a bare number of days.
        c_maxdt = widgets.Text(
            value=format_dt_days(initial["max_dt_days"]),
            description="max Δt:",
            placeholder="e.g. 1Y, 6M, 200 (empty = off)",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="260px"),
        )
        c_mindt = widgets.Text(
            value=format_dt_days(initial["min_dt_days"]),
            description="min Δt:",
            placeholder="e.g. 6M, 60 (empty = off)",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="260px"),
        )
        c_colorby = widgets.Dropdown(
            options=["group", "sensor", "none"],
            value=initial["color_by"],
            description="colour by:",
            style={"description_width": "initial"},
        )
        c_formats = widgets.SelectMultiple(
            options=list(FIGURE_FORMATS),
            value=("html", "png"),
            rows=len(FIGURE_FORMATS),
            description="formats:",
            style={"description_width": "initial"},
        )
        b_save = widgets.Button(description="Save figure", icon="save")
        b_plan = widgets.Button(description="Write plan", icon="file-code")

        # A plain go.Figure inside an Output, never go.FigureWidget: since
        # plotly 6 the latter is an anywidget whose front-end JS is not bundled
        # with the VSCode Jupyter extension ("No version of module anywidget is
        # registered").
        plot_out = widgets.Output()
        summary = widgets.HTML()
        status = widgets.HTML()
        group_out = widgets.Output()
        state: dict = {"syncing": False}

        #: Control name → widget, for the per-view visibility pass.  Only
        #: view-local options: a derivation input must never be hidden, or it
        #: would keep shaping the committed parameters unseen.
        tunables = {"color_by": c_colorby}

        def _read_assumptions() -> dict:
            return {
                "velocity_m_yr": float(c_velocity.value),
                "footprint_m": float(c_footprint.value),
                "c_px": float(c_c.value),
                "k": float(c_k.value),
                "coreg_px": float(c_coreg.value),
                "margin_px": float(c_margin.value),
                # Sentinels rather than None-able widgets: ipywidgets has no
                # nullable float input, and a checkbox per option would double
                # the control count for three rarely-used refinements.
                "flow_azimuth_deg": (
                    float(c_azimuth.value) if c_azimuth.value >= 0 else None
                ),
                "strain_rate_per_yr": (
                    float(c_strain.value) if c_strain.value > 0 else None
                ),
                "corr_algorithm": c_algorithm.value,
                "subpixel_mode": int(c_subpixel.value),
                "tau_days": float(c_tau.value) if c_tau.value > 0 else None,
                "dt_bin_ratio": float(c_ratio.value),
                "explicit_search": bool(c_explicit.value),
                "explicit_search_above_days": float(
                    _read_dt(c_explicit_above, "explicit above") or 0
                ),
            }

        def _read_uniform() -> dict:
            """The uniform overrides, with the box already parsed.

            Parsing here rather than in ``_group_params`` keeps the failure
            visible: a bad box writes a red status line and the last good value
            is reused, instead of raising inside a widget callback where the
            traceback would reach only the hidden kernel log.
            """
            box = None
            if c_uniform_search.value:
                try:
                    box = parse_search_box(c_uniform_search_box.value)
                    state["uniform_box"] = box
                except ValueError as exc:
                    status.value = (
                        f"<span style='color:#c33'>✘ search box: {exc}</span>"
                    )
                    box = state.get("uniform_box")
            return {
                "uniform_kernel": bool(c_uniform_kernel.value),
                "uniform_kernel_px": int(c_uniform_kernel_px.value),
                "uniform_search": bool(c_uniform_search.value) and box is not None,
                "uniform_search_box": c_uniform_search_box.value,
                "_parsed_search_box": box,
            }

        def _read_dt(widget, label: str) -> int | None:
            """Parse one Δt field, degrading rather than raising.

            An exception inside a widget callback reaches only the kernel log,
            which VSCode hides — the control would simply stop responding with no
            visible cause. So a bad string writes a red status line and the last
            good value is reused.

            :returns: Days, or None when the field is empty.
            """
            text = (widget.value or "").strip()
            if not text:
                state.pop(f"dt_{label}", None)
                return None
            try:
                days = parse_dt_days(int(text) if text.isdigit() else text)
            except ValueError as exc:
                status.value = f"<span style='color:#c33'>✘ {label}: {exc}</span>"
                return state.get(f"dt_{label}")
            state[f"dt_{label}"] = days
            return days

        def _read_filters() -> tuple:
            return (
                c_sensor.value.strip() or None,
                _read_dt(c_maxdt, "max Δt"),
                _read_dt(c_mindt, "min Δt"),
            )

        def _apply_visibility() -> None:
            wanted = self._CORR_VIEW_CONTROLS.get(c_view.value, set())
            for name, widget in tunables.items():
                widget.layout.display = None if name in wanted else "none"

        def _compute() -> tuple[pd.DataFrame, dict]:
            sfilter, max_dt, min_dt = _read_filters()
            return _build_frame(
                sfilter, max_dt, min_dt, uniform=_read_uniform(), **_read_assumptions()
            )

        def _stash() -> dict:
            sfilter, max_dt, min_dt = _read_filters()
            return _stash_params(
                state["frame"],
                state["groups"],
                _read_assumptions(),
                sfilter,
                max_dt,
                min_dt,
                _read_uniform(),
            )

        def _view_kwargs() -> dict:
            return _view_defaults(c_view.value, _read_assumptions(), c_colorby.value)

        def _fig_title() -> str:
            return _title_for(state["frame"], c_view.value, _read_assumptions())

        def _draw() -> None:
            fig = corr_plotly.VIEW_BUILDERS[c_view.value](
                state["frame"], title=_fig_title(), **_view_kwargs()
            )
            with plot_out:
                clear_output(wait=True)
                _ipy_display(fig)

        def _render_summary() -> None:
            summary.value = format_corrparams_summary(
                corrparams_stats(state["frame"], k=float(c_k.value))
            )
            with group_out:
                clear_output(wait=True)
                table = group_table(state["frame"])
                if len(table):
                    _ipy_display(table.style.hide(axis="index"))

        def _redraw(change=None) -> None:
            """View-only change: reuse the cached frame, never re-derive."""
            if state.get("syncing"):
                return
            _apply_visibility()
            _draw()

        def _update(_change=None) -> None:
            """A derivation input changed: recompute, restash, redraw."""
            if state.get("syncing"):
                return
            _apply_visibility()
            state["frame"], state["groups"] = _compute()
            _stash()
            _draw()
            _render_summary()

        def _on_save(_button) -> None:
            # Never raises: an uncaught exception in a widget callback reaches
            # only the kernel log, which VSCode hides.
            b_save.disabled = True
            status.value = "<i>saving…</i>"
            try:
                paths = self.save_correlation_figure(
                    view=c_view.value,
                    frame=state["frame"],
                    formats=tuple(c_formats.value) or ("html",),
                    title=_fig_title(),
                    **_view_kwargs(),
                )
                names = " ".join(f"<code>{p.name}</code>" for p in paths.values())
                folder = next(iter(paths.values())).parent
                status.value = (
                    f"<span style='color:#2a7'>✔ saved to</span> "
                    f"<code>{folder}</code>: {names}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Saving the correlation figure failed: {exc}")
                status.value = f"<span style='color:#c33'>✘ save failed — {exc}</span>"
            finally:
                b_save.disabled = False

        def _on_write_plan(_button) -> None:
            b_plan.disabled = True
            status.value = "<i>writing plan…</i>"
            try:
                _maybe_write_plan(_stash())
                status.value = (
                    "<span style='color:#2a7'>✔ plan written</span> — commit with "
                    "<code>prepare_pairs_correlation(**session._last_corr_params, "
                    "overwrite_scripts=True)</code>"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Writing the correlation plan failed: {exc}")
                status.value = f"<span style='color:#c33'>✘ plan failed — {exc}</span>"
            finally:
                b_plan.disabled = False

        for widget in (
            c_velocity,
            c_footprint,
            c_c,
            c_k,
            c_coreg,
            c_margin,
            c_azimuth,
            c_strain,
            c_tau,
            c_ratio,
            c_algorithm,
            c_subpixel,
            c_sensor,
            c_maxdt,
            c_mindt,
            c_explicit,
            c_explicit_above,
            c_uniform_kernel,
            c_uniform_kernel_px,
            c_uniform_search,
            c_uniform_search_box,
        ):
            widget.observe(_update, "value")
        for widget in (c_view, c_colorby):
            widget.observe(_redraw, "value")
        b_save.on_click(_on_save)
        b_plan.on_click(_on_write_plan)

        _update()

        row1 = widgets.HBox([c_view, c_colorby, c_algorithm, c_subpixel])
        row2 = widgets.HBox([c_velocity, c_footprint, c_c, c_k])
        row3 = widgets.HBox([c_coreg, c_margin, c_azimuth, c_strain, c_tau])
        row4 = widgets.HBox([c_ratio, c_sensor, c_mindt, c_maxdt])
        row5 = widgets.HBox(
            [
                c_explicit,
                c_explicit_above,
                c_uniform_kernel,
                c_uniform_kernel_px,
                c_uniform_search,
                c_uniform_search_box,
            ]
        )
        row6 = widgets.HBox([c_formats, widgets.VBox([b_save, b_plan])])
        return widgets.VBox(
            [row1, row2, row3, row4, row5, plot_out, summary, group_out, status, row6]
        )

    def _corr_params_changed(
        self,
        params_file: pathlib.Path,
        asp,
        effective: Mapping[str, Any],
        *,
        save_disparity_difference: bool = True,
    ) -> bool:
        """Whether *effective* differs from the parameters already on disk.

        Thin wrapper over
        :func:`~geomulticorr.correlation._correlation_plan.params_differ_from_file`,
        kept as a method so the reuse guard in
        :meth:`prepare_pairs_correlation` reads as one decision rather than an
        import plus a call.

        :param params_file: The pair's ``<pa_key>_CorrParameters.txt``.
        :param asp: The :class:`~geomulticorr.correlation.correlation.ASP`
            instance (unused today; accepted so a future format change can be
            resolved through it rather than reparsed here).
        :param effective: Resolved per-pair parameters.
        :param save_disparity_difference: Whether the bare flag is expected.
        :returns: True when the script and settings file must be regenerated.
        """
        from geomulticorr.correlation._correlation_plan import params_differ_from_file

        return params_differ_from_file(
            params_file,
            effective,
            save_disparity_difference=save_disparity_difference,
        )

    @gmc_profile("gmc.session.prepare_pairs_correlation", memprof=True)
    def prepare_pairs_correlation(
        self,
        criterias: str | list[str] = "",
        cluster: str | None = None,
        overwrite: bool = False,
        asp_bin_dir: str | pathlib.Path | None = None,
        nodes: int = 1,
        cores: int = 8,
        walltime: str = "12:00:00",
        besteffort: bool = False,
        conda_env: str = "gmc_env",  # TO delete (deprecated, use cluster-specific envs instead)
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        corr_tile_size: int = 2048,
        corr_algorithm: str = "asp_bm",
        corr_kernel: tuple[int, int] = (21, 21),
        corr_search: tuple[int, int, int, int] | None = None,
        corr_xthreshold: int = 10,
        corr_seed_mode: int = 1,
        subpixel_mode: int = 2,
        subpixel_kernel: tuple[int, int] = (21, 21),
        prefilter_mode: int = 2,
        cost_mode: int = 2,
        save_disparity_difference: bool = True,
        correval: bool = True,
        metric: str = "ncc",
        overwrite_scripts: bool = False,
        print_summary: bool = True,
        plan: str | pathlib.Path | dict | None = None,
        params_by_pair: Mapping[str, dict] | None = None,
        image_type: str | None = None,
        nodata_value: float | None = None,
        individual_normalization: bool | None = None,
        pre_normalization: bool | None = None,
    ) -> list[pathlib.Path]:
        """Prepare the output directory and bash script for each pair.

        For every pair matching *criterias*:

        1. Skip if the pair is already ``'complete'`` (unless *overwrite=True*).
        2. Create the pair output directory (``pair.pa_path``).
        3. Clip left and right images to their spatial intersection if not done yet.
        4. Write ``correlation_job.sh`` with the full ASP command.

        Parameters are resolved **per pair**, with the precedence
        ``scalar arguments -> plan group -> per-pair override``.  Each pair
        already receives its own ``<pa_key>_CorrParameters.txt``; the only thing
        that used to force a uniform setting was that the scalar arguments were
        constant across the loop.  With neither *plan* nor *params_by_pair*,
        behaviour is identical to passing the scalars alone.

        Why per-pair parameters matter: a pair measures ``v * dt``, and one ASP
        parameter set spans only ``S / (k * c)`` — roughly 30-100x — so a single
        setting cannot serve both a 12-day and a 6-year baseline.  See
        ``docs/correlation_parameters.md``.

        Args:
            criterias: Passed to :meth:`get_pairs` to filter pairs.
            cluster: ``None`` / ``"local"`` / ``"gricad"`` / ``"isterre"``.
            overwrite: If *True*, redo pairs that are already complete.
            asp_bin_dir: Optional path to the ASP bin directory.
            nodes / cores / walltime / besteffort / conda_env: OAR cluster options.
            processes / threads_* / corr_* / etc: ``parallel_stereo`` parameters.
            plan: A correlation plan — a path to a ``correlation_plan_*.json``,
                or the already-loaded mapping.  Supplies per-group parameters
                and the pair-to-group assignment.
            params_by_pair: ``{pa_key: {...}}`` pinning individual pairs, which
                wins over both the scalars and the plan.
            image_type: Keep only pairs of this image type (``"opt"``,
                ``"hillshade"``, ``"mdhillshade"``, ``"slope"``, ``"dem"``);
                ANDed with *criterias*. ``None`` keeps every type.
            nodata_value / individual_normalization / pre_normalization: ASP
                input encoding (``nodata-value``, ``individually-normalize``,
                ``force-use-entire-range``). ``None`` takes the per-pair
                default of its image type (:data:`CORR_INPUT_DEFAULTS`): nothing
                for optical, which keeps its settings file unchanged; nodata
                ``0`` for hillshades and ``-9999`` for slope/DEM, with
                individual normalisation, for elevation types. Explicit values
                win for every pair.

        Returns:
            List of paths to the generated bash scripts.
        """
        from geomulticorr.correlation import ASP
        from geomulticorr.correlation._correlation_plan import (
            input_params_differ_from_file,
            read_correlation_plan,
            resolve_pair_params,
        )

        asp = ASP(session=self, asp_bin_dir=asp_bin_dir)
        pairs = self.get_pairs(criterias, image_type=image_type)

        # The plan and params_by_pair are keyed by pa_key, which is unique only
        # within one image type: an optical and a hillshade pair of the same
        # dates and sensor share it.  Applying a keyed setting to both would be
        # a guess, so refuse and ask for image_type.
        if plan is not None or params_by_pair:
            key_counts = collections.Counter(p.pa_key for p in pairs)
            dupes = sorted(k for k, n in key_counts.items() if n > 1)
            if dupes:
                raise ValueError(
                    f"{len(dupes)} pa_key(s) occur in more than one image type "
                    f"(e.g. {dupes[0]!r}), so plan/params_by_pair would apply "
                    f"ambiguously; pass image_type= to select one type"
                )
        explicit_input = {
            "nodata_value": nodata_value,
            "individual_normalization": individual_normalization,
            "pre_normalization": pre_normalization,
        }

        if not pairs:
            logger.warning("No pairs found for the given criteria. Nothing to prepare.")
            return []

        cluster_label = cluster if cluster else "local"
        logger.info(
            f"Preparing {len(pairs)} pair(s) for correlation "
            f"[cluster={cluster_label}]"
        )

        # Resolve the plan once: reading it per pair would re-parse the same
        # JSON for every row.
        plan_document: dict = {}
        if plan is not None:
            plan_document = (
                dict(plan) if isinstance(plan, dict) else read_correlation_plan(plan)
            )
        plan_groups = plan_document.get("groups") or {}
        pair_group = {
            str(row.get("pa_key")): str(row.get("group", ""))
            for row in (plan_document.get("pairs") or [])
        }
        plan_overrides = dict(plan_document.get("overrides") or {})
        if params_by_pair:
            plan_overrides.update({str(k): dict(v) for k, v in params_by_pair.items()})

        #: The scalar arguments, as the floor of the precedence chain.
        scalar_params = {
            "corr_algorithm": corr_algorithm,
            "corr_kernel": corr_kernel,
            "corr_search": corr_search,
            "corr_xthreshold": corr_xthreshold,
            "corr_seed_mode": corr_seed_mode,
            "subpixel_mode": subpixel_mode,
            "subpixel_kernel": subpixel_kernel,
            "prefilter_mode": prefilter_mode,
            "cost_mode": cost_mode,
        }

        summary_rows: list[dict] = []
        bash_scripts: list[pathlib.Path] = []

        pair_name = _RichText("", style="cyan")
        progress = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=_rich_console,
        )
        task = progress.add_task("", total=len(pairs))

        with _RichLive(
            _RichGroup(pair_name, progress),
            console=_rich_console,
            transient=False,
            refresh_per_second=10,
        ) as live:
            for pair in pairs:
                pair_name.plain = pair.pa_key
                rec = {
                    "pair": pair.pa_key,
                    "directory": "—",
                    "symlink": "—",
                    "group": "—",
                    "params": "—",
                    "script": "—",
                }
                status = pair.get_status()

                # Resolve this pair's effective parameters up front: the reuse
                # decision below depends on whether they differ from what is
                # already on disk.
                group = pair_group.get(pair.pa_key, "")
                effective, source = resolve_pair_params(
                    pair.pa_key,
                    group,
                    scalar_params,
                    groups=plan_groups,
                    overrides=plan_overrides,
                )
                rec["group"] = group if source != "scalar" else "scalar"
                if source == "override":
                    rec["group"] = f"{group or '—'} (override)"
                input_params = _corr_input_params(
                    getattr(pair, "pa_type", IMAGE_TYPE_OPTICAL), explicit_input
                )

                if all(pair.check_corr_outputs().values()) and not overwrite:
                    rec["script"] = "skipped (complete)"
                    summary_rows.append(rec)
                    progress.advance(task)
                    live.refresh()
                    continue

                # 1. Create output directory
                pair.pa_path.mkdir(parents=True, exist_ok=True)
                rec["directory"] = "created"

                # 2. Clip inputs if not already done
                if status not in ("clipped", "complete"):
                    pair.clip()
                    rec["symlink"] = "created"
                else:
                    rec["symlink"] = "exists"

                # 3. Reuse the existing script only when the parameters it was
                #    built from still match.  Skipping on mere existence made a
                #    parameter change a SILENT no-op: the `continue` below
                #    returned before step 4 could regenerate the params file, so
                #    a freshly tuned setting was discarded and the old one re-run
                #    with nothing in the log to say so.
                bash_path = pair.pa_path / f"{pair.pa_key}_CorrelationJob.sh"
                params_file = pair.pa_path / f"{pair.pa_key}_CorrParameters.txt"
                if bash_path.exists() and not overwrite_scripts:
                    changed = self._corr_params_changed(
                        params_file,
                        asp,
                        effective,
                        save_disparity_difference=save_disparity_difference,
                    ) or input_params_differ_from_file(params_file, input_params)
                    if not changed:
                        bash_scripts.append(bash_path)
                        rec["script"] = "reused"
                        summary_rows.append(rec)
                        progress.advance(task)
                        live.refresh()
                        continue
                    logger.info(
                        f"{pair.pa_key}: correlation parameters changed — "
                        f"regenerating the script and parameters file"
                    )

                # 4. Write correlation parameters settings file
                asp.build_correlation_params(
                    save_disparity_difference=save_disparity_difference,
                    params_file_path=params_file,
                    **_asp_param_kwargs(effective),
                    **input_params,
                )
                rec["params"] = params_file.name

                # 5. Write bash script referencing that params file
                bash_path = asp.write_bash_script(
                    pair,
                    corr_params_file=params_file,
                    cluster=cluster,
                    nodes=nodes,
                    cores=cores,
                    walltime=walltime,
                    besteffort=besteffort,
                    conda_env=conda_env,
                    processes=processes,
                    threads_multiprocess=threads_multiprocess,
                    threads_singleprocess=threads_singleprocess,
                    corr_memory_limit_mb=corr_memory_limit_mb,
                    corr_tile_size=corr_tile_size,
                    correval=correval,
                    corr_algorithm=effective["corr_algorithm"],
                    corr_kernel=effective["corr_kernel"],
                    metric=metric,
                )
                bash_scripts.append(bash_path)
                rec["script"] = bash_path.name

                summary_rows.append(rec)
                progress.advance(task)
                live.refresh()

        print_prepare_correlation_summary(summary_rows, print_summary=print_summary)

        return bash_scripts

    @gmc_profile("gmc.session.launch_pairs_correlation")
    def launch_pairs_correlation(
        self,
        criterias: str | list[str] = "",
        cluster: str | None = None,
        dry_run: bool = False,
        overwrite: bool = False,
        asp_bin_dir: str | pathlib.Path | None = None,
        nodes: int = 1,
        cores: int = 8,
        walltime: str = "12:00:00",
        besteffort: bool = False,
        conda_env: str = "gmc_env",
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        corr_tile_size: int = 2048,
        clean: bool = True,
        correval: bool = True,
        overwrite_scripts: bool = False,
        print_summary: bool = True,
        image_type: str | None = None,
    ) -> list[int]:
        """Prepare (if needed) and launch correlation for all pairs matching *criterias*.

        Calls :meth:`prepare_pairs_correlation` to generate bash scripts (skipping pairs
        whose script already exists unless *overwrite_scripts=True*), then submits or
        executes each script:

        - **local** (``cluster=None`` or ``"local"``): ``bash <pa_key>_CorrelationJob.sh``
        - **cluster** (``"gricad"`` or ``"isterre"``): ``oarsub -S ./<pa_key>_CorrelationJob.sh``

        ``corr_eval`` is embedded in the bash script itself (written by
        :meth:`prepare_pairs_correlation`) — it is not run as a separate step here.
        Correlation tuning parameters (kernel, algorithm, etc.) are set via
        :meth:`prepare_pairs_correlation` directly.

        Args:
            criterias: Passed to :meth:`get_pairs` to filter pairs.
            cluster: ``None`` / ``"local"`` / ``"gricad"`` / ``"isterre"``.
            dry_run: Print the launch command for each pair without executing.
            overwrite: Redo pairs whose ASP outputs are already complete.
            overwrite_scripts: Rewrite the bash script and params file even if they exist.
            asp_bin_dir: Optional path to the ASP bin directory.
            nodes / cores / walltime / besteffort / conda_env: OAR cluster options.
            processes / threads_* / corr_memory_limit_mb / corr_tile_size:
                Forwarded to ``parallel_stereo`` inside the script.
            clean: Delete intermediate ASP files after a successful run.
            correval: Embed a ``corr_eval`` call in the generated bash script.
            image_type: Keep only pairs of this image type; ANDed with
                *criterias*. ``None`` keeps every type.

        Returns:
            List of return codes (one per pair; 0 = success / submitted).
        """
        pairs = self.get_pairs(criterias, image_type=image_type)
        if not pairs:
            logger.warning("No pairs found matching criterias.")
            return []

        use_cluster = cluster in ("gricad", "isterre")

        if clean:
            from geomulticorr.correlation import ASP

            asp = ASP(session=self, asp_bin_dir=asp_bin_dir)
        else:
            asp = None

        # Resolve bash script paths and warn for any that are missing.
        ready: list[tuple] = []  # (pair, script_path)
        for pair in pairs:
            script = pair.pa_path / f"{pair.pa_key}_CorrelationJob.sh"
            if script.exists():
                ready.append((pair, script))
            else:
                logger.warning(
                    f"Script not found for {pair.pa_key} — run prepare_pairs_correlation() first."
                )

        if not ready:
            logger.warning("No launch scripts found. Aborting.")
            return []

        logger.info(
            f"Launching {len(ready)}/{len(pairs)} pair(s) "
            f"[cluster={'cluster' if use_cluster else 'local'}, dry_run={dry_run}]"
        )

        return_codes: list[int] = []
        summary_rows: list[dict] = []

        pair_name = _RichText("", style="cyan")
        progress = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=_rich_console,
        )
        task = progress.add_task("", total=len(ready))

        with _RichLive(
            _RichGroup(pair_name, progress),
            console=_rich_console,
            transient=False,
            refresh_per_second=10,
        ) as live:
            for pair, script in ready:
                pair_name.plain = pair.pa_key
                live.refresh()

                if use_cluster:
                    cmd = ["oarsub", "-S", str(script.resolve())]
                else:
                    cmd = ["bash", str(script)]

                if dry_run:
                    status = f"DRY RUN: {' '.join(cmd)}"
                    return_codes.append(0)
                    summary_rows.append({"pair": pair.pa_key, "status": "dry run"})
                    progress.advance(task)
                    live.refresh()
                    continue

                result = subprocess.run(
                    cmd,
                    cwd=script.parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return_codes.append(result.returncode)

                if result.returncode != 0:
                    summary_rows.append(
                        {
                            "pair": pair.pa_key,
                            "status": f"[red]failed (rc={result.returncode})[/red]",
                        }
                    )
                    logger.error(f"Failed [{pair.pa_key}]: {result.stderr.strip()}")
                else:
                    if clean and not use_cluster:
                        asp.clean_pair_directory(pair)
                    summary_rows.append(
                        {"pair": pair.pa_key, "status": "[green]launched[/green]"}
                    )

                progress.advance(task)
                live.refresh()

        # Summary table
        print_launch_correlation_summary(summary_rows, print_summary=print_summary)

        # Sync pa_status back to the geodatabase for local runs.
        # For cluster (oarsub) jobs the processes are async — call
        # _sync_pairs_status(get_pairs(criterias)) manually after they finish.
        if not use_cluster and not dry_run:
            self._sync_pairs_status([p for p, _ in ready])

        return return_codes

    def sync_pairs_after_cluster(
        self,
        criterias: str | list[str] = "",
        clean: bool = True,
        asp_bin_dir: str | pathlib.Path | None = None,
        print_summary: bool = True,
        image_type: str | None = None,
    ) -> None:
        """Sync geodatabase status and optionally clean after OAR jobs complete.

        Call this once all ``oarsub`` jobs have finished on the cluster
        (check with ``oarstat -u $USER``).  It reads the filesystem state of
        each pair to determine whether the correlation completed successfully,
        updates ``pa_status`` in the geodatabase, and — if *clean* is ``True``
        — removes intermediate ASP files from pairs whose status is
        ``"complete"``.

        Args:
            criterias: Same filter syntax as :meth:`get_pairs`.
            image_type: Keep only pairs of this image type (``"opt"``, ``"hillshade"``, ``"mdhillshade"``, ``"slope"``, ``"dem"``); ANDed with *criterias*. Prefer it over a criterias string: ``"hillshade"`` is also a substring of ``"mdhillshade"``.
            clean: Delete intermediate ASP files for completed pairs (default ``True``).
            asp_bin_dir: Optional path to the ASP bin directory (only used when *clean* is ``True``).
        """
        from geomulticorr.correlation import ASP

        pairs = self.get_pairs(criterias, image_type=image_type)
        if not pairs:
            logger.warning("sync_pairs_after_cluster: no pairs found.")
            return

        asp = ASP(session=self, asp_bin_dir=asp_bin_dir) if clean else None

        summary_rows: list[dict] = []

        pair_name = _RichText("", style="cyan")
        progress = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=_rich_console,
        )
        task = progress.add_task("", total=len(pairs))

        with _RichLive(
            _RichGroup(pair_name, progress),
            console=_rich_console,
            transient=False,
            refresh_per_second=10,
        ) as live:
            for pair in pairs:
                pair_name.plain = pair.pa_key
                live.refresh()

                status = pair.get_status()
                n_deleted = 0

                if clean and asp is not None and status == "complete":
                    removed = asp.clean_pair_directory(pair)
                    n_deleted = len(removed)

                summary_rows.append(
                    {
                        "pair": pair.pa_key,
                        "correlated": (
                            "[green]yes[/green]"
                            if status == "complete"
                            else f"[red]no ({status})[/red]"
                        ),
                        "deleted": str(n_deleted) if n_deleted else "—",
                        "db_status": status,
                    }
                )

                progress.advance(task)
                live.refresh()

        self._sync_pairs_status(pairs)

        print_sync_after_cluster_summary(summary_rows, print_summary=print_summary)

    def reset_pairs_outputs(
        self,
        criterias: str | list[str] = "",
        dry_run: bool = False,
        print_summary: bool = True,
        image_type: str | None = None,
    ) -> list[pathlib.Path]:
        """Delete all correlation outputs for matching pairs, resetting them to 'clipped' status.

        Removes every non-symlink file inside each pair's directory, preserving only the
        two input symlinks (left / right image links).  After deletion ``get_status()``
        returns ``"clipped"``, so the pair can be re-correlated with new parameters via
        :meth:`prepare_pairs_correlation` and :meth:`launch_pairs_correlation`.

        Args:
            criterias: Filter pairs — same syntax as :meth:`get_pairs`.
            image_type: Keep only pairs of this image type (``"opt"``, ``"hillshade"``, ``"mdhillshade"``, ``"slope"``, ``"dem"``); ANDed with *criterias*. Prefer it over a criterias string: ``"hillshade"`` is also a substring of ``"mdhillshade"``.
            dry_run: List files that would be deleted without removing them.
            print_summary: Print a summary table after completion.

        Returns:
            List of deleted (or would-be-deleted) file paths.
        """
        pairs = self.get_pairs(criterias, image_type=image_type)
        if not pairs:
            logger.warning("reset_pairs_outputs: no pairs found.")
            return []

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Resetting outputs for {len(pairs)} pair(s)."
        )

        deleted_all: list[pathlib.Path] = []
        summary_rows: list[dict] = []

        pair_name = _RichText("", style="cyan")
        progress = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=_rich_console,
        )
        task = progress.add_task("", total=len(pairs))

        with _RichLive(
            _RichGroup(pair_name, progress),
            console=_rich_console,
            transient=False,
            refresh_per_second=10,
        ) as live:
            for pair in pairs:
                pair_name.plain = pair.pa_key
                live.refresh()

                status_before = pair.get_status()
                deleted: list[pathlib.Path] = []

                if pair.pa_path.exists():
                    for f in sorted(pair.pa_path.iterdir()):
                        if f.is_symlink() or not f.is_file():
                            continue
                        if not dry_run:
                            f.unlink()
                        deleted.append(f)

                deleted_all.extend(deleted)
                status_after = pair.get_status() if not dry_run else "dry-run"
                summary_rows.append(
                    {
                        "pair": pair.pa_key,
                        "status_before": status_before,
                        "deleted": str(len(deleted)),
                        "status_after": (
                            f"[dim]{status_after}[/dim]" if dry_run else status_after
                        ),
                    }
                )
                progress.advance(task)
                live.refresh()

        if not dry_run:
            self._sync_pairs_status(pairs)

        print_reset_outputs_summary(summary_rows, print_summary=print_summary)

        return deleted_all

    @gmc_profile("gmc.session.extract_pairs_raw_displacements", memprof=True)
    def extract_pairs_raw_displacements(
        self,
        criterias: str | list[str] = "",
        save_plot: bool = True,
        overwrite: bool = False,
        target_resolution: float | None = None,
        resampling: str = "bilinear",
        canonical_bounds: "object | None" = None,
        canonical_grid_size: "tuple[int, int] | None" = None,
        print_summary: bool = True,
        sync_geodb: bool = True,
        verbose: bool = False,
        image_type: str | None = None,
    ) -> dict:
        """Extract EW/NS displacements and compute raw stats for all complete pairs.

        Iterates over all pairs matching *criterias*, calls
        :meth:`~geomulticorr.core.pair.Pair.extract_raw_displacements` on each
        complete pair, and returns a summary dict keyed by ``pa_key``.

        Args:
            criterias: Same filter syntax as :meth:`get_pairs`.
            image_type: Keep only pairs of this image type (``"opt"``, ``"hillshade"``, ``"mdhillshade"``, ``"slope"``, ``"dem"``); ANDed with *criterias*. Prefer it over a criterias string: ``"hillshade"`` is also a substring of ``"mdhillshade"``.
            save_plot: Save the 9-panel control figure for each pair.
            overwrite: Re-process pairs that already have EW/NS files on disk.
            target_resolution: If provided, resample EW, NS, and NCC outputs to this
                pixel size (in the session CRS units, typically metres) after extraction.
                Useful for homogenising multi-sensor archives — e.g. downsample SPOT
                1.5 m outputs to 3.0 m to match PlanetScope before TIO inversion.
                ``None`` keeps the native resolution.
            resampling: Rasterio resampling algorithm used when *target_resolution* is
                set. ``"bilinear"`` (default) gives smooth interpolation; ``"average"``
                is physically appropriate for aggregating displacement fields.
            canonical_bounds: Exact output bounds (rasterio BoundingBox) to use for the
                resampling step. When provided together with *canonical_grid_size*,
                overrides the auto-computed per-pzone canonical grid. Use this to pin
                all pairs to a specific known grid.
            canonical_grid_size: Exact output grid size ``(width, height)`` in pixels.
                Must be provided together with *canonical_bounds*.
            sync_geodb: If ``True`` (default), sync raw stats
                (:data:`RAW_STATS_SYNC`) into the Pairs geodatabase layer for the
                pairs processed in this call. See :meth:`sync_raw_stats`.
            verbose: If ``True``, keep the per-pair messages the extraction
                helpers emit (``Saved EW displacement``, ``Pair stats file
                initialised``, …). The default ``False`` prints one header
                describing the whole run instead, so the progress bar stays
                readable across hundreds of pairs. Warnings and errors are
                unaffected either way.

        Returns:
            Dict mapping ``pa_key`` → full stats dict (or ``None`` if skipped).
        """
        import geopandas as gpd

        pairs = self.get_pairs(criterias, image_type=image_type)
        results: dict = {}

        # Pre-compute one canonical grid per pzone so all pairs within the same pzone
        # are reprojected to identical bounds and dimensions when target_resolution is set.
        pzone_canonical: dict = {}
        if target_resolution is not None and canonical_bounds is None:
            pzones_df = self.get_pzones_overview()
            for _, row in pzones_df.iterrows():
                pz_vec = gu.Vector(
                    gpd.GeoDataFrame([row], crs=pzones_df.crs, geometry="geometry")
                )
                cb, cs = self._compute_canonical_grid(
                    pz_vec, self.epsg, target_resolution
                )
                pzone_canonical[row["pz_name"]] = (cb, cs)

        summary_rows: list[dict] = []

        # One header for the whole run, in place of 4-7 log lines per pair.
        # pa_path is <pzone>/image_correlation/<pa_key>/, so the parent differs
        # per pzone — report the pattern, not one pair's path.
        stats_dirs = sorted({str(p.pa_path.parent) for p in pairs})
        where = (
            stats_dirs[0] if len(stats_dirs) == 1 else f"{len(stats_dirs)} pzone dir(s)"
        )

        logger.info(f"Extracting raw displacements — {len(pairs)} pair(s)")
        logger.file(f"Extracting EW | NS | CC files → {where}/<pa_key>/")
        logger.file(f"Pair stats file initialised: {where}/<pa_key>/<pa_key>.json")
        if save_plot:
            logger.save("Saving control plot: <pa_key>_raw_disp.jpg (one per pair)")
        if target_resolution is not None:
            grid = (
                "explicit canonical grid"
                if canonical_bounds is not None
                else "per-pzone canonical grid"
            )
            logger.settings(
                f"Resampling EW | NS | CC to {target_resolution} m "
                f"({resampling}, {grid})"
            )
        if not verbose:
            logger.settings(
                "Per-pair messages suppressed — pass verbose=True to show them"
            )

        with BatchProgress(
            len(pairs),
            description="extract_pairs_raw_displacements",
            verbose=verbose,
        ) as bar:
            for pair in bar.track(pairs, key=lambda p: p.pa_key):
                rec = {
                    "pair": pair.pa_key,
                    "ew": "—",
                    "ns": "—",
                    "cc": "—",
                    "plot": "—",
                    "stats": "—",
                }

                if not pair.pa_disparity_f_path.exists():
                    logger.warning(
                        f"Skipping '{pair.pa_key}' — disparity raster not found"
                    )
                    rec["ew"] = rec["ns"] = rec["cc"] = "[red]no disparity[/red]"
                    results[pair.pa_key] = None
                    summary_rows.append(rec)
                    continue  # bar.track() advances on continue

                if (
                    not overwrite
                    and pair.pa_ew_path.exists()
                    and pair.pa_ns_path.exists()
                ):
                    rec["ew"] = "[dim]exists[/dim]"
                    rec["ns"] = "[dim]exists[/dim]"
                    rec["cc"] = "[dim]exists[/dim]" if pair.pa_cc_path.exists() else "—"
                    rec["plot"] = (
                        "[dim]exists[/dim]" if pair.pa_plot_raw_path.exists() else "—"
                    )
                    rec["stats"] = (
                        "[dim]exists[/dim]" if pair.pa_stats_path.exists() else "—"
                    )
                    results[pair.pa_key] = None
                    summary_rows.append(rec)
                    continue  # bar.track() advances on continue

                # Resolve canonical grid for this pair.
                if canonical_bounds is not None and canonical_grid_size is not None:
                    pair_cb, pair_cs = canonical_bounds, canonical_grid_size
                else:
                    pair_cb, pair_cs = pzone_canonical.get(
                        pair.pa_pz_name, (None, None)
                    )

                try:
                    stats = pair.extract_raw_displacements(
                        save_plot=save_plot,
                        target_resolution=target_resolution,
                        resampling=resampling,
                        canonical_bounds=pair_cb,
                        canonical_grid_size=pair_cs,
                    )
                    results[pair.pa_key] = stats
                    rec["ew"] = (
                        "[green]ok[/green]"
                        if pair.pa_ew_path.exists()
                        else "[red]missing[/red]"
                    )
                    rec["ns"] = (
                        "[green]ok[/green]"
                        if pair.pa_ns_path.exists()
                        else "[red]missing[/red]"
                    )
                    rec["cc"] = (
                        "[green]ok[/green]"
                        if pair.pa_cc_path.exists()
                        else "[red]missing[/red]"
                    )
                    rec["plot"] = (
                        "[green]ok[/green]"
                        if pair.pa_plot_raw_path.exists()
                        else ("—" if not save_plot else "[red]missing[/red]")
                    )
                    rec["stats"] = (
                        "[green]ok[/green]"
                        if pair.pa_stats_path.exists()
                        else "[red]missing[/red]"
                    )
                except Exception as exc:
                    logger.error(f"Failed for '{pair.pa_key}': {exc}")
                    results[pair.pa_key] = None
                    rec["ew"] = rec["ns"] = rec["cc"] = f"[red]error[/red]"
                    rec["plot"] = rec["stats"] = "[red]error[/red]"

                summary_rows.append(rec)

        print_extract_displacements_summary(
            summary_rows, target_resolution, resampling, print_summary=print_summary
        )

        if sync_geodb:
            done = [p for p in pairs if results.get(p.pa_key) is not None]
            if done:
                self.sync_raw_stats(pairs=done)

        return results

    @gmc_profile("gmc.session.apply_pairs_corrections", memprof=True)
    def apply_pairs_corrections(
        self,
        correction_pipeline=None,
        filter_pipeline=None,
        criterias: str | list[str] = "",
        overwrite: bool = False,
        save_plot: bool = True,
        plot_level: str = "light",
        print_summary: bool = True,
        sync_geodb: bool = True,
        verbose: bool = False,
        image_type: str | None = None,
        **kwargs,
    ) -> dict:
        """Apply a filter and/or correction pipeline to all processed pairs.

        Follows the two-pipeline architecture:

        1. **filter_pipeline** (:class:`~geomulticorr.corrections.FilterPipeline`):
           Masks bad pixels (CC threshold, outlier rejection, slope, …).
           Also produces per-component stable-area masks used when fitting
           corrections.

        2. **correction_pipeline** (:class:`~geomulticorr.corrections.CorrectionPipeline`):
           Fits and subtracts systematic trends (median shift, ramp, topo, …)
           estimated on stable pixels only.

        At least one of the two must be provided.  Typical call::

            session.apply_pairs_corrections(
                filter_pipeline=CCFilter(0.5) + OutlierFilter((-15, 15)),
                correction_pipeline=MedianCentering() + RampCorrection() + TopoCorrection(),
                dem=dem,
            )

        Args:
            correction_pipeline: A :class:`~geomulticorr.corrections.CorrectionPipeline`
                instance (or any :class:`~geomulticorr.corrections.BaseCorrection`).
                ``None`` skips the correction step.
            filter_pipeline: A :class:`~geomulticorr.corrections.FilterPipeline`
                instance (or any :class:`~geomulticorr.corrections.BaseMask`).
                ``None`` skips filtering (all valid pixels used as stable ground).
            criterias: Same filter syntax as :meth:`get_pairs`.
            image_type: Keep only pairs of this image type (``"opt"``, ``"hillshade"``, ``"mdhillshade"``, ``"slope"``, ``"dem"``); ANDed with *criterias*. Prefer it over a criterias string: ``"hillshade"`` is also a substring of ``"mdhillshade"``.
            overwrite: Re-process pairs whose corrected files already exist.
                Pairs corrected before the stable-ground masks were persisted
                are re-processed regardless when a *filter_pipeline* is given,
                so their ``*EWmask``/``*NSmask`` files get written.
            save_plot: Save a before/after control figure for each pair.
            plot_level: Rendering cost of those figures — ``"light"`` (default)
                decimates rasters to 1400 px on the long axis, uses a 200-bin
                hexbin and writes at 110 dpi; ``"full"`` renders at native
                resolution, 1000-bin hexbin and 150 dpi.  Both produce the same
                figures — ``"light"`` is roughly 9× faster per pair on 10+ Mpx
                correlation output and is what makes large batches practical.
                Statistics written to the pair stats JSON are unaffected by this
                setting; only the numbers annotated *inside* the figures are
                computed on the decimated sample.
            sync_geodb: If ``True`` (default) and a *correction_pipeline* is given,
                sync last-correction stats (:data:`CORR_STATS_SYNC`) and
                ``pa_n_corrections`` into the Pairs geodatabase layer for the pairs
                corrected in this call. See :meth:`sync_corrected_stats`.
            **kwargs: Forwarded to both pipelines
                (e.g. ``dem=`` for :class:`~geomulticorr.corrections.TopoCorrection`).
                ``cc=`` is auto-injected from the pair folder when available.

        Returns:
            Dict mapping ``pa_key`` → ``{"ew": Path, "ns": Path}`` or ``None`` if
            skipped.  When a *filter_pipeline* is given, the per-component
            stable-ground keep-masks are also written into the pair folder as
            1-bit GeoTIFFs and reported as ``"ew_mask"`` / ``"ns_mask"``
            (see :meth:`~geomulticorr.core.pair.Pair.save_correction_masks`).

        Raises:
            ValueError: If neither *correction_pipeline* nor *filter_pipeline* is provided,
                if *plot_level* is not a known level, or if a required kwarg
                (e.g. ``dem=``) is missing.

        Note:
            *verbose* (default ``False``) keeps the per-pair, per-step messages
            the helpers emit — ``save_corrected_stats`` alone logs once per
            correction step per pair, so a 3-step pipeline over 200 pairs is
            ~600 lines. The default prints one header describing the run
            instead. Warnings and errors surface either way, including the
            misaligned-DEM warning from the topographic corrections.
        """
        import geoutils as gu
        import matplotlib.pyplot as plt
        from geomulticorr.corrections.corrections import (
            CorrectionPipeline,
            make_corrections,
        )
        from geomulticorr.corrections.masks import FilterPipeline, BaseMask

        if correction_pipeline is None and filter_pipeline is None:
            raise ValueError(
                "Provide at least one of correction_pipeline= or filter_pipeline=."
            )

        if plot_level not in PLOT_LEVELS:
            raise ValueError(
                f"Unknown plot_level={plot_level!r} — expected one of "
                f"{sorted(PLOT_LEVELS)}."
            )
        plot_opts = PLOT_LEVELS[plot_level]

        pairs = self.get_pairs(criterias, image_type=image_type)
        results: dict = {}

        _RAMP_TOPO_CLASSES = {
            "RampCorrection",
            "TopoCorrection",
            "TopoRampCorrection",
            "SlopeRampCorrection",
        }

        summary_rows: list[dict] = []

        # One header for the whole run, in place of per-pair *and* per-step logs.
        filt_steps = (
            []
            if filter_pipeline is None
            else (
                filter_pipeline.masks
                if isinstance(filter_pipeline, FilterPipeline)
                else [filter_pipeline]
            )
        )
        corr_steps = (
            []
            if correction_pipeline is None
            else (
                correction_pipeline.steps
                if isinstance(correction_pipeline, CorrectionPipeline)
                else [correction_pipeline]
            )
        )

        logger.info(f"Applying corrections — {len(pairs)} pair(s)")
        logger.settings(
            "Filter pipeline     : "
            + (" + ".join(type(s).__name__ for s in filt_steps) or "none")
        )
        logger.settings(
            "Correction pipeline : "
            + (" → ".join(type(s).__name__ for s in corr_steps) or "none")
        )
        logger.file(
            "Writing per pair: <pa_key>-F_EW_corr.tif | <pa_key>-F_NS_corr.tif"
            + (" | -F_EWmask.tif | -F_NSmask.tif" if filt_steps else "")
        )
        logger.file(
            "Corrected stats appended to <pa_key>.json "
            "(sections 'correction_stats' → 'final_corrected_stats')"
        )
        if save_plot:
            logger.save(
                f"Saving control plots: one per correction step "
                f"(<pa_key>_NN_<Step>_disp.jpg) + a corrections dashboard "
                f"(plot_level={plot_level!r}, preview_px={plot_opts['preview_px']}, "
                f"dpi={plot_opts['dpi']})"
            )
        if not verbose:
            logger.settings(
                "Per-pair / per-step messages suppressed — pass verbose=True to show them"
            )

        with BatchProgress(
            len(pairs),
            description="apply_pairs_corrections",
            verbose=verbose,
        ) as bar:
            for pair in bar.track(pairs, key=lambda p: p.pa_key):
                ew_corr = pair.pa_path / f"{pair.pa_key}-F_EW_corr.tif"
                ns_corr = pair.pa_path / f"{pair.pa_key}-F_NS_corr.tif"

                rec = {
                    "pair": pair.pa_key,
                    "median": "—",
                    "ramp_topo": "—",
                    "stats": "—",
                    "ew_corr": "—",
                    "ns_corr": "—",
                }

                if not pair.pa_ew_path.exists() or not pair.pa_ns_path.exists():
                    logger.warning(
                        f"Skipping '{pair.pa_key}' — EW/NS rasters not found"
                    )
                    rec["ew_corr"] = rec["ns_corr"] = "[red]no input[/red]"
                    results[pair.pa_key] = None
                    summary_rows.append(rec)
                    continue  # bar.track() advances on continue

                # A pair corrected before masks were persisted has no mask files;
                # regenerate it so the stable-ground record catches up.
                masks_ok = filter_pipeline is None or (
                    pair.pa_ew_mask_path.exists() and pair.pa_ns_mask_path.exists()
                )
                if not overwrite and ew_corr.exists() and ns_corr.exists() and masks_ok:
                    rec["median"] = rec["ramp_topo"] = rec["stats"] = (
                        "[dim]exists[/dim]"
                    )
                    rec["ew_corr"] = rec["ns_corr"] = "[dim]exists[/dim]"
                    results[pair.pa_key] = None
                    summary_rows.append(rec)
                    continue  # bar.track() advances on continue

                try:
                    x_raw = gu.Raster(str(pair.pa_ew_path), nodata=0)
                    y_raw = gu.Raster(str(pair.pa_ns_path), nodata=0)

                    # Auto-inject cc= from the pair folder if not already provided
                    pair_kwargs = dict(kwargs)
                    if "cc" not in pair_kwargs and pair.pa_cc_path.exists():
                        pair_kwargs["cc"] = gu.Raster(str(pair.pa_cc_path))

                    # Validate required kwargs upfront
                    all_steps = []
                    if filter_pipeline is not None:
                        fp = filter_pipeline
                        all_steps += (
                            fp.masks if isinstance(fp, FilterPipeline) else [fp]
                        )
                    if correction_pipeline is not None:
                        cp = correction_pipeline
                        all_steps += (
                            cp.steps if isinstance(cp, CorrectionPipeline) else [cp]
                        )
                    missing = [
                        f"{type(s).__name__} requires '{k}'"
                        for s in all_steps
                        for k in getattr(s, "_REQUIRED_KWARGS", ())
                        if k not in pair_kwargs
                    ]
                    if missing:
                        raise ValueError(
                            "Missing required kwargs — pass them to apply_pairs_corrections():\n  "
                            + "\n  ".join(missing)
                        )

                    # Step 1: filter pipeline → masked rasters + stable-area masks
                    x_stable = y_stable = None
                    if filter_pipeline is not None:
                        x_stable = filter_pipeline.compute(x_raw, **pair_kwargs)
                        y_stable = filter_pipeline.compute(y_raw, **pair_kwargs)
                        x_filt, y_filt = filter_pipeline.apply(
                            x_raw, y_raw, **pair_kwargs
                        )
                    else:
                        x_filt, y_filt = x_raw, y_raw

                    # Step 2: correction pipeline — fit + apply each step, track names
                    import copy
                    import geomulticorr.stats.stats as gmc_stats

                    x_corr, y_corr = x_filt, y_filt
                    ramp_topo_names: list[str] = []
                    if correction_pipeline is not None:
                        cp_x = copy.deepcopy(correction_pipeline)
                        cp_y = copy.deepcopy(correction_pipeline)
                        steps_x = (
                            cp_x.steps
                            if isinstance(cp_x, CorrectionPipeline)
                            else [cp_x]
                        )
                        steps_y = (
                            cp_y.steps
                            if isinstance(cp_y, CorrectionPipeline)
                            else [cp_y]
                        )
                        for idx, (step_x, step_y) in enumerate(zip(steps_x, steps_y)):
                            step_name = type(step_x).__name__
                            before_x, before_y = x_corr, y_corr
                            step_x.fit(x_corr, stable_mask=x_stable, **pair_kwargs)
                            x_corr = step_x.apply(x_corr)
                            step_y.fit(y_corr, stable_mask=y_stable, **pair_kwargs)
                            y_corr = step_y.apply(y_corr)
                            gmc_stats.save_corrected_stats(
                                pair, x_corr, y_corr, step_name
                            )

                            # Per-step before/after control figure (correct template
                            # per correction type; both EW/NS steps feed the panels)
                            if save_plot:
                                try:
                                    from geomulticorr.utils.gmc_functions import (
                                        plot_correction_result,
                                    )

                                    fig = plot_correction_result(
                                        step_x,
                                        step_y,
                                        before_x,
                                        x_corr,
                                        before_y,
                                        y_corr,
                                        fig_name=pair.pa_key,
                                        preview_px=plot_opts["preview_px"],
                                        **pair_kwargs,
                                    )
                                    plot_path = (
                                        pair.pa_path
                                        / f"{pair.pa_key}_{idx:02d}_{step_name}_disp.jpg"
                                    )
                                    fig.savefig(
                                        str(plot_path),
                                        dpi=plot_opts["dpi"],
                                        format="JPEG",
                                    )
                                    plt.close(fig)
                                except Exception as plot_exc:
                                    logger.warning(
                                        f"Could not save control plot for {step_name}: {plot_exc}"
                                    )

                            if step_name == "MedianCentering":
                                rec["median"] = "[green]ok[/green]"
                            elif step_name in _RAMP_TOPO_CLASSES:
                                ramp_topo_names.append(step_name)

                        # After all corrections: copy the last step's block into
                        # the canonical 'final_corrected_stats' section.
                        if steps_x:
                            gmc_stats.save_final_corrected_stats(
                                pair, last_correction_name=type(steps_x[-1]).__name__
                            )

                    if ramp_topo_names:
                        rec["ramp_topo"] = (
                            "[green]" + "+".join(ramp_topo_names) + "[/green]"
                        )
                    rec["stats"] = (
                        "[green]ok[/green]"
                        if pair.pa_stats_path.exists()
                        else "[red]missing[/red]"
                    )

                    # Optional corrected-map dashboard (final state + stable areas)
                    if save_plot:
                        try:
                            from geomulticorr.utils.gmc_functions import (
                                plot_show_corrected_results,
                            )

                            fig = plot_show_corrected_results(
                                x_corr,
                                y_corr,
                                pair_kwargs.get("cc"),
                                x_stable=x_stable,
                                y_stable=y_stable,
                                fig_name=pair.pa_key,
                                preview_px=plot_opts["preview_px"],
                                hexbin_gridsize=plot_opts["hexbin_gridsize"],
                            )
                            plot_path = (
                                pair.pa_path / f"{pair.pa_key}_corrections_disp.jpg"
                            )
                            fig.savefig(
                                str(plot_path), dpi=plot_opts["dpi"], format="JPEG"
                            )
                            plt.close(fig)
                        except Exception as plot_exc:
                            logger.warning(
                                f"Could not save corrected dashboard: {plot_exc}"
                            )

                    x_corr.save(str(ew_corr))
                    y_corr.save(str(ns_corr))
                    rec["ew_corr"] = (
                        "[green]ok[/green]"
                        if ew_corr.exists()
                        else "[red]missing[/red]"
                    )
                    rec["ns_corr"] = (
                        "[green]ok[/green]"
                        if ns_corr.exists()
                        else "[red]missing[/red]"
                    )
                    pair_result = {"ew": ew_corr, "ns": ns_corr}

                    # Persist the stable-ground masks the corrections were fitted
                    # on, as 1-bit GeoTIFFs alongside the rasters they describe.
                    # A failure here must not lose the corrected output.
                    if x_stable is not None or y_stable is not None:
                        try:
                            masks = pair.save_correction_masks(
                                x_stable, y_stable, reference=x_raw
                            )
                            pair_result.update(
                                {f"{k}_mask": v for k, v in masks.items()}
                            )
                        except Exception as mask_exc:
                            logger.warning(
                                f"Could not save correction masks for "
                                f"'{pair.pa_key}': {mask_exc}"
                            )

                    results[pair.pa_key] = pair_result

                except Exception as exc:
                    logger.error(f"Failed for '{pair.pa_key}': {exc}")
                    rec["ew_corr"] = rec["ns_corr"] = "[red]error[/red]"
                    results[pair.pa_key] = None

                summary_rows.append(rec)

        print_apply_corrections_summary(summary_rows, print_summary=print_summary)

        if sync_geodb and correction_pipeline is not None:
            steps = (
                correction_pipeline.steps
                if isinstance(correction_pipeline, CorrectionPipeline)
                else [correction_pipeline]
            )
            done = [p for p in pairs if results.get(p.pa_key)]
            if done and steps:
                self.sync_corrected_stats(
                    pairs=done,
                    n_corrections=len(steps),
                )

        return results

    # TODO: to be completed with the other parameters
    def pr_full(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10):
        """Launch all the possible correlations accross all the valid images among the project pzones"""
        logs = []
        for pzone in self.get_pzones():
            logs.append(
                pzone.pz_full(
                    self.epsg, corr_algorithm, corr_kernel_size, corr_xthreshold
                )
            )
        return pd.DataFrame(logs)

    def _search_engine_thibaut(
        self, layername: str, criterias: str | list[str] = ""
    ) -> gpd.GeoDataFrame:
        """A search engine among project layers

        Args:
            layername (str): layer name.
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            gpd.GeoDataFrame: return a dataframe with the filtered data.
        """

        # if there is only one criteria we store it in a list
        # if type(criterias) in (str, int):
        #     criterias = [criterias]
        criterias = (
            []
            if criterias in ("", [])
            else ([criterias] if isinstance(criterias, (str, int)) else list(criterias))
        )

        # # add an "and" statement between each criteria
        # pattern = ""
        # for c in criterias:
        #     if type(c) == int:
        #         c = str(c)
        #     pattern += f"(?=.*{c.lower()})"
        # pattern = re.compile(pattern)
        pattern = (
            None
            if not criterias
            else "^" + "".join(f"(?=.*{re.escape(str(c))})" for c in criterias)
        )

        match layername:
            case "Thumbs":
                normal_th = self._thumbs
                lower_th = normal_th.apply(lambda x: x.str.lower(), 1)
                return normal_th[lower_th.th_path.str.contains(pattern)]

            case "Pairs":
                normal_pa = self._pairs
                lower_pa = normal_pa.apply(lambda x: x.str.lower(), 1)
                return normal_pa[lower_pa.pa_path.str.contains(pattern)]

            case "Pzones":
                pz_layer = self._pzones
                if criterias != [""]:
                    requested_pz_name = criterias[0].lower()
                    return pz_layer[pz_layer.pz_name == requested_pz_name]
                else:
                    return pz_layer

            case "Geomorphs":
                # We suppose than the criteria can only be a Geomorph ID or a pz_name
                selection = self._geomorphs
                if criterias != [""]:
                    for criteria in criterias:
                        if criteria in self.pz_names:
                            selection = selection[selection.ge_pz_name == criteria]
                        else:
                            selection = selection[selection.ge_frogi_id == criteria]
                    return selection
                else:
                    return selection
        # if layername == "Thumbs":
        #     s = self._thumbs.th_path
        #     return self._thumbs if pattern is None else self._thumbs[s.str.contains(pattern, case=False, regex=True, na=False)]
        # if layername == "Pairs":
        #     s = self._pairs.pa_path
        #     return self._pairs if pattern is None else self._pairs[s.str.contains(pattern, case=False, regex=True, na=False)]
        # if layername == "Pzones":
        #     return self._pzones if pattern is None else self._pzones[self._pzones.pz_name.str.contains(pattern, case=False, regex=True, na=False)]
        # if layername == "Geomorphs":
        #     df = self._geomorphs
        #     if pattern is None: return df
        #     m1 = df.ge_pz_name.str.contains(pattern, case=False, regex=True, na=False)
        #     m2 = df.ge_frogi_id.astype(str).str.contains(pattern, case=False, regex=True, na=False)
        #     return df[m1 | m2]
        # raise ValueError(f"Unknown layer: {layername}")

    def _search_engine(
        self, layername: str, criterias: str | list[str] = ""
    ) -> gpd.GeoDataFrame:
        """
        A search engine among project layers.

        Args:
            layername (str): layer name.
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            gpd.GeoDataFrame: DataFrame with the filtered data.
        """
        # Normalize criterias to a list of strings
        if criterias in ("", [], None):
            criterias = []
        elif isinstance(criterias, (str, int)):
            criterias = [str(criterias)]
        else:
            criterias = [str(c) for c in criterias]

        # Build AND regex pattern for all criterias
        pattern = (
            None
            if not criterias
            else "".join(f"(?=.*{re.escape(c)})" for c in criterias)
        )

        # Helper for case-insensitive contains
        def contains_ci(series, pat):
            if pat is None:
                return pd.Series([True] * len(series), index=series.index)
            return series.str.contains(pat, case=False, regex=True, na=False)

        if layername == "Thumbs":
            df = self._thumbs
            if "th_path" not in df.columns:
                raise KeyError("Column 'th_path' not found in Thumbs layer.")
            mask = contains_ci(df["th_path"], pattern)
            return df[mask]

        elif layername == "Pairs":
            df = self._pairs
            if "pa_path" not in df.columns:
                raise KeyError("Column 'pa_path' not found in Pairs layer.")
            mask = contains_ci(df["pa_path"], pattern)
            return df[mask]

        elif layername == "Pzones":
            df = self._pzones
            if not criterias:
                return df
            if "pz_name" not in df.columns:
                raise KeyError("Column 'pz_name' not found in Pzones layer.")
            mask = contains_ci(df["pz_name"], pattern)
            return df[mask]

        elif layername == "Geomorphs":
            df = self._geomorphs
            if not criterias:
                return df
            # Try to match either ge_pz_name or ge_frogi_id
            m1 = contains_ci(
                df.get("ge_pz_name", pd.Series("", index=df.index)), pattern
            )
            m2 = contains_ci(
                df.get("ge_frogi_id", pd.Series("", index=df.index)).astype(str),
                pattern,
            )
            return df[m1 | m2]

        else:
            raise ValueError(f"Unknown layer: {layername}")

    def __repr__(self):
        return f"""
            ------------
            This is a GeoMultiCorr Session open on the project named {self.project_name}
            Their processing zones are : {self.pz_names}
            ------------
            """
