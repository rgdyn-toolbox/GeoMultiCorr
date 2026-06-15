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

import copy
import pathlib
import warnings
from collections import abc
from contextlib import ExitStack
from typing import IO, TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, overload

import re
from shapely.geometry import Polygon
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
import shapely.wkt as _wkt
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from alive_progress import alive_bar

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
import geomulticorr.correlation.supported_sensors as gmc_sensors

from geomulticorr._logging import logger
from geomulticorr._typing import (
    DTypeLike,
    MArrayNum,
    NDArrayBool,
    NDArrayNum,
    Number
    )

file_location = pathlib.Path(__file__)
package_root = file_location.parent.parent  # geomulticorr/core/ -> geomulticorr/
resources_location = package_root / "resources"

project_template_location = resources_location / "project_template"

_DT_UNIT_TO_DAYS: dict[str, int] = {"D": 1, "W": 7, "M": 30, "Y": 365}

def _parse_dt_days(value: int | str | None) -> int | None:
    """Convert a duration string to integer days.

    Accepts an integer (passed through unchanged), ``None``, or a string of the
    form ``'<n><unit>'`` where unit is one of ``D`` (days), ``W`` (weeks),
    ``M`` (months ≈ 30 d), ``Y`` (years ≈ 365 d).  Case-insensitive.

    Examples::

        _parse_dt_days(30)     -> 30
        _parse_dt_days("30D")  -> 30
        _parse_dt_days("6M")   -> 180
        _parse_dt_days("1Y")   -> 365
        _parse_dt_days("2W")   -> 14
    """
    if value is None or isinstance(value, int):
        return value
    match = re.fullmatch(r"(\d+)([DdWwMmYy])", str(value).strip())
    if not match:
        raise ValueError(
            f"Cannot parse duration '{value}'. "
            "Use an integer (days) or a string like '30D', '6M', '1Y', '2W'."
        )
    n, unit = int(match.group(1)), match.group(2).upper()
    return n * _DT_UNIT_TO_DAYS[unit]

def is_conform_to_gmc_template(target_root_path: str | pathlib.Path) -> bool:
    #! I dont know if this functions is useful. Check with Thibaut
    """Check GeoMultiCorr structure.
    Checks the 3 conditions to ensure that the 'target_root_path' leads to a folder conforming to the GeoMultiCorr project data architecture.

    Args:
        target_root_path (str | pathlib.Path): Path where GeoMultiCorr project data will be stored.

    Returns:
        bool: True if the folder conforms to the GeoMultiCorr project data architecture. False otherwise.
    """
    target_root_path = pathlib.Path(target_root_path)
    target_name = target_root_path.name

    # Firstly, there is a folder named raster_data_parent-folder-name
    target_raster_data_expected_path = pathlib.Path(target_root_path, f"raster_data_{target_name}")
    if not target_raster_data_expected_path.is_dir():
        return False

    # Secondly, there is a file "GMC_mapset_{target_name}.qgz"
    target_map_expected_path = pathlib.Path(target_root_path, f"GMC_mapset_{target_name}.qgz")
    if not target_map_expected_path.exists():
        return False

    # Thirdly, there is a file GMC_geodatabase_{target_name}.gpkg
    target_geodb_expected_path = pathlib.Path(target_root_path, f"GMC_geodatabase_{target_name}.gpkg")
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
    #END try
#END def

# * Works properly
def open_gmc_session(target_directory_path: str | pathlib.Path,
                     epsg_code: str | int | None = None,
                     empty_geodatabase: str | pathlib.Path = None) -> "Session":
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
    #END if

    if not isinstance(target_directory_path, pathlib.Path):
        target_directory_path = pathlib.Path(target_directory_path)
    #END if
    return Session(target_directory_path, epsg_code, empty_geodatabase)

class Session:
    """
    Manipulate data specific to a sample of sites related to an earth surface displacement study.
    """
    # * It works properly
    def __init__(self,
                 target_root_path: str | pathlib.Path,
                 epsg_code: str | int | None = None,
                 empty_geodatabase: bool | str | pathlib.Path = None):
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
                    raw_gdb_path = project_template_location / "GMC_geodatabase_empty.gpkg"
                    shutil.copy(raw_gdb_path, target_root_path)
                    new_gdb = target_root_path / "GMC_geodatabase_empty.gpkg"
                    new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")
                else:
                    logger.file(f"Default GMC-geodatabase template copied into {target_root_path}")
                    raw_gdb_path = project_template_location / "GMC_geodatabase_project.gpkg"
                    shutil.copy(raw_gdb_path, target_root_path)
                    new_gdb = target_root_path / "GMC_geodatabase_project.gpkg"
                    new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")

                logger.info(f"Setting GMC folder structure")
                # And we create en empty folder "raster_data_{project_name}"
                if not (target_root_path / f"raster_data_{project_name}").exists():
                    logger.folder(f"Creating 'raster_data_{project_name}' folder")
                    (target_root_path / f"raster_data_{project_name}").mkdir(parents=True)
            elif is_conform_to_gmc_template(target_root_path):
                logger.success("GMC project structure found")
                pass
            else:
                msg = (f"Something else exists at {target_root_path}. Please clean it or choose another path.")
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
                new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")
            else:
                logger.file(f"Default GMC-geodatabase template copied into {target_root_path}")
                raw_gdb_path = project_template_location / "GMC_geodatabase_project.gpkg"
                shutil.copy(raw_gdb_path, target_root_path)
                new_gdb = target_root_path / "GMC_geodatabase_project.gpkg"
                new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")
        
            # Copy GMC-Mapset template
            logger.file(f"Making a copy of 'GMC-QGIS-Mapset' template into {target_root_path}")
            raw_mapset_path = project_template_location / "GMC_mapset_project.qgz"
            shutil.copy(raw_mapset_path, target_root_path)
            new_mapset = target_root_path / "GMC_mapset_project.qgz"
            new_mapset.rename(target_root_path / f"GMC_mapset_{project_name}.qgz")

            logger.success(f"Setting GMC folder structure")
            # And we create en empty folder "raster_data_{project_name}"
            if not (target_root_path / f"raster_data_{project_name}").exists():
                logger.folder(f"Creating 'raster_data_{project_name}' folder")
                (target_root_path / f"raster_data_{project_name}").mkdir(parents=True)
        # END if

        # Load project data into the current session — runs for all code paths
        self.path_root = target_root_path
        self.project_name = target_root_path.name
        self.path_raster_data = target_root_path / f"raster_data_{self.project_name}"
        self.path_img_correlation_outs = target_root_path / f"img_correlation_outs_{self.project_name}"
        self.path_tio_inversion_outs = target_root_path / f"tio_inversion_outs_{self.project_name}"
        self.path_geodb = target_root_path / f"GMC_geodatabase_{self.project_name}.gpkg"
        self.path_qgis_project = target_root_path / f"GMC_mapset_{self.project_name}.qgz"
        self.epsg = epsg_code

        # There is underscore before the attribute name because user have to access to this data from the getters to ensure that the data is up to date
        self._pzones = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Pzones").to_crs(epsg=epsg_code)
        self._thumbs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Thumbs").to_crs(epsg=epsg_code)
        self._geomorphs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Geomorphs").to_crs(epsg=epsg_code)
        self._pairs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Pairs").to_crs(epsg=epsg_code)
        self._xzones = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Xzones").to_crs(epsg=epsg_code)
        self._spines = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Spines").to_crs(epsg=epsg_code)

        # List the processing zones names
        self.pz_names = list(self._pzones.pz_name.unique())
        logger.launch(f"Session '{self.project_name}' initialized.")
    #END def
    # * Works properly
    def find_qgis_binary(self,
                        qgis_binary_path: str | None = None) -> str:
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
        raise FileNotFoundError("QGIS binary not found. Please install QGIS or specify its path.")
    
    # * Works properly
    # ! This function may overload the system if the project as QGIs is invoked from terminal. We can add a warning in the docstring to inform the user about this point. We can also add an optional argument to choose if the project should be opened or not.
    def open_qgis_project(self,
                          project_path: str | pathlib.Path | None = None,
                          qgis_binary_path: str | None = None) -> None:
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
            raise FileNotFoundError("QGIS binary not found. Please install QGIS or specify its path.")
        else:
            call = []
            call.extend([f"{qgis_binary_path}"])
            call.extend(["--nologo", "--noplugins", '--noversioncheck', '--hide-browser'])
            if project_path is not None:
                call.extend(["--project", f"{project_path}"])
            else:
                call.extend(["--project", f"{self.path_qgis_project}"])
            call.extend([f"{self.path_geodb}"])
        subprocess.run(call, check=True)
        logger.info(f"Opening QGIS project {self.path_qgis_project}.")

    ###################################################
    #-------------------- GETTERS --------------------#
    # * Works properly
    def get_thumbs_overview(self,
                            criterias: str | list[str] = "") -> gpd.GeoDataFrame:
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
    def get_pairs_overview(self, criterias: str | list[str] = "") -> gpd.GeoDataFrame:
        """Get pairs overview informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            gpd.GeoDataFrame: Send a dataframe with each possible Pair according to the Thumbs
        # TODO: Implement unittests.
        """
        return self._search_engine("Pairs", criterias)
    
    # * Works properly
    def get_pairs(self, criterias: str | list[str] = "") -> list[gmc_pair.Pair]:
        """Get pairs informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a list of pair.Pairs objects meeting the criterias
        # TODO: solve issue regarding order of return pairs. When session.get_pairs(['ribon', '2014', '2018']), pair 2018-2014 is returned first. 

        """
        # selected_pairs = self.get_pairs_overview(criterias)
        # return [gmc_pair.Pair(self, target_path=x.pa_path) for x in selected_pairs.iloc]
        selected_pairs = self.get_pairs_overview(criterias)
        # return [gmc_pair.Pair(self, target_path=x.pa_path) for x in selected_pairs]
        return [gmc_pair.Pair(self, target_path=row['pa_path']) for _, row in selected_pairs.iterrows()]

    def get_failed_pairs(self, criterias: str | list[str] = "") -> list[gmc_pair.Pair]:
        """Return pairs where one or more ASP output files are missing.

        Checks all files listed in ``ASP._KEEP_SUFFIXES`` inside each pair's
        ``pa_path``. Logs the missing files for each failed pair.

        Args:
            criterias: Same filter syntax as :meth:`get_pairs`.

        Returns:
            List of ``Pair`` objects with at least one missing output file.
        """
        pairs = self.get_pairs(criterias)
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
        return [gmc_pzone.Pzone(x.pz_name, self) for x in selected_pzones.itertuples(index=False)]

    def get_geomorphs_overview(self, criterias: str | list[str] = "") -> pd.DataFrame:
        """Get geomorphos overview informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a dataframe with each geomorph.
        """
        return self._search_engine("Geomorphs", criterias)

    def get_geomorphs(self, criterias: str | list[str] = "") -> list[gmc_geomorph.Geomorph]:
        """Get geomorphs informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list: Send a list of geomorph.Geomorphs objects.
        """
        selected_geomorphs = self.get_geomorphs_overview(criterias)
        return [gmc_geomorph.Geomorph(self, x.ge_frogi_id) for x in selected_geomorphs.itertuples(index=False)]

    def get_xzone(self, xz_id: str | int | float) -> gmc_xzone.Xzone:
        """Get a xzone

        Args:
            xz_id (str | int | float): xzone id.

        Returns:
            gmc_xzone.Xzone: Send a xzones.Xzones object
        """
        return gmc_xzone.Xzone(self, xz_id)

    def get_pairs_overview_on_period(self,
                                     ymin: str | pd.Timestamp,
                                     ymax: str | pd.Timestamp,
                                     criterias: str | list[str] = "") -> pd.DataFrame:
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
            lambda row: min(int(row.pa_left_date.split("-")[0]), int(row.pa_right_date.split("-")[0])), axis=1)
        pairs["chrono_max"] = pairs.apply(
            lambda row: max(int(row.pa_left_date.split("-")[0]), int(row.pa_right_date.split("-")[0])), axis=1)
        # Possible error on text and number
        pairs = pairs[(pairs.chrono_min >= ymin) & (pairs.chrono_max <= ymax)]
        return pairs

    def get_pairs_on_period(self,
                            ymin: str | pd.Timestamp,
                            ymax: str | pd.Timestamp,
                            criterias: str | list[str] = "") -> list[gmc_pair.Pair]:
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
        return [pz.get_dem() for pz in self.get_pzones(criterias) if pz.get_dem() != False]

    def get_georasters_map(self,
                           image_type: str | list[str] = "opt",
                           suffix: str = "") -> gpd.GeoDataFrame:
        layername = f"Protomap"
        try:
            return gpd.read_file(self.path_geodb, layer=layername, engine="pyogrio")
        except ValueError:
            logger.error(f"No georasters map named {layername} in the session geodatabase")
            return None
    #------------------ END GETTERS ------------------#
    ###################################################

    ###################################################
    #-------------------- SETTERS --------------------#
    def insert_pzone(self,
                     coordinates: list[tuple[float, float]] | Polygon,
                     pz_name: str,
                     pz_shortname: str,
                     crs: str | int | None = None) -> gmc_pzone.Pzone:
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

        polygon = coordinates if isinstance(coordinates, Polygon) else Polygon(coordinates)

        # Read current Pzones layer and build new row
        gdf = gpd.read_file(self.path_geodb, layer="Pzones", engine="pyogrio")
        row = {col: None for col in gdf.columns}
        row["pz_name"] = pz_name
        row["pz_shortname"] = pz_shortname
        row["geometry"] = polygon

        input_crs = crs if crs is not None else self.epsg
        new_row = gpd.GeoDataFrame(
            [row],
            crs=f"EPSG:{input_crs}" if isinstance(input_crs, int) else input_crs
        )

        if gdf.crs is not None and new_row.crs != gdf.crs:
            new_row = new_row.to_crs(gdf.crs)

        gdf = pd.concat([gdf, new_row], ignore_index=True)
        gdf.to_file(self.path_geodb, layer="Pzones", engine="pyogrio")

        # Update session state
        self._pzones = gdf.to_crs(epsg=self.epsg)
        self.pz_names = list(self._pzones.pz_name.unique())

        # Create raster data directory required by Pzone.__init__
        pz_raster_dir = pathlib.Path(self.path_raster_data) / pz_name
        pz_raster_dir.mkdir(parents=True, exist_ok=True)
        logger.success(f"Pzone '{pz_name}' inserted.")

        return gmc_pzone.Pzone(pz_name, self)

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
        self._geomorphs = gpd.read_file(self.path_geodb, layer="Geomorphs", engine="pyogrio")
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
        opt_root = pathlib.Path(self.path_raster_data)
        old = self._thumbs
        new = gpd.GeoDataFrame([gmc_thumb.Thumb(target_path).to_pdserie()
                                for target_path in filter(lambda x:
                                                          gmc_thumb.THUMBNAME_PATTERN.match(x.name),
                                                          list(opt_root.glob(pattern="**/opticals/*.tif")),)])

        # Comparison
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

    def validate_all_thumbs(self,
                            pzone_filter: list[str] | None = None
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

    # * Works properly
    def update_pairs(self) -> gpd.GeoDataFrame:
        """Update pairs in the geodatabase.

        Raises:
            ValueError: If no 'geometry' column is found in pairs DataFrame.

        Returns:
            gpd.GeoDataFrame: pz.pairs with updated pairs.
        
        # ! TODO: Introduce pairing strategy. To do in pzone.
        # TODO: Implement unittests.
        """
        # Comment code coresponds to Thibaut code.
        # Copy the geodatabase before the transaction
        # assert self.copy_geodb() why?

        # For each processing zone we
        # updated = []
        # for pz in self.get_pzones():
        #     pairs = pz.get_pairs_overview()
        #     [updated.append(pa) for pa in pairs.iloc()]

        # # Push it into the geodatabase
        # updated = gpd.GeoDataFrame(updated).set_crs(epsg=self.epsg)
        # updated.to_file(self.path_geodb, layer="Pairs")

        # # Update instance
        # self._pairs = updated
        updated_frames = []
        for pz in self.get_pzones():
            logger.info(f"Updating pairs for PZone: '{pz.pz_name}'")
            pairs = pz.get_valid_pairs_overview()
            if not pairs.empty:
                updated_frames.append(pairs)
        
        if updated_frames:
            updated = pd.concat(updated_frames, ignore_index=True)
            if "geometry" not in updated.columns:
                raise ValueError("No 'geometry' column found in pairs DataFrame.")
            updated = gpd.GeoDataFrame(updated, geometry="geometry", crs=self.epsg)
        else:
            updated = gpd.GeoDataFrame(geometry=[], crs=self.epsg)
        
        # updated = (gpd.GeoDataFrame(pd.concat(updated_frames, ignore_index=True))
        #    .set_crs(epsg=self.epsg)) if updated_frames else gpd.GeoDataFrame(crs=self.epsg)
        updated.to_file(self.path_geodb, layer="Pairs")
        self._pairs = updated
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

    def update_pairs_with_strategy(
        self,
        strategy: str = "consecutive",
        max_step: int | None = None,
        max_dt_days: int | str | None = None,
        sensor_filter: str | None = None,
        append: bool = False,
    ) -> gpd.GeoDataFrame:
        """Build pairs for all pzones using the specified strategy and persist to geodatabase.

        Args:
            strategy: ``"consecutive"``, ``"step"``, or ``"redundancy"``.
            max_step: Maximum index offset between paired thumbs (required for
                ``"step"`` and ``"redundancy"``).
            max_dt_days: Optional upper bound on the date gap. Accepts an integer
                number of days or a duration string: ``'30D'``, ``'6M'``, ``'1Y'``,
                ``'2W'``. Months and years use fixed approximations (30 d/month,
                365 d/year), which is appropriate given the 1-day image resolution.
            sensor_filter: Optional sensor name substring to restrict thumbs
                (e.g. ``"spot"``, ``"planetscope"``). Pairs are built only from
                thumbs whose path contains this string.
            append: If ``True``, new pairs are concatenated with the existing
                ``Pairs`` layer (duplicates resolved by ``pa_path``). If ``False``
                (default), the layer is fully replaced.

        Returns:
            GeoDataFrame of all pairs written to the ``Pairs`` layer.
        """
        max_dt_days = _parse_dt_days(max_dt_days)
        updated_frames = []
        for pz in self.get_pzones():
            logger.info(f"Updating pairs for PZone: '{pz.pz_name}'")
            pairs_gdf = pz.get_valid_pairs_with_strategy_overview(strategy, max_step, max_dt_days, sensor_filter=sensor_filter)
            if not pairs_gdf.empty:
                updated_frames.append(pairs_gdf)

        if updated_frames:
            new_pairs = pd.concat(updated_frames, ignore_index=True)
            if "geometry" not in new_pairs.columns:
                raise ValueError("No 'geometry' column found in pairs DataFrame.")
            new_pairs = gpd.GeoDataFrame(new_pairs, geometry="geometry", crs=self.epsg)
        else:
            new_pairs = gpd.GeoDataFrame(geometry=[], crs=self.epsg)

        if append and not self._pairs.empty:
            combined = pd.concat([self._pairs, new_pairs], ignore_index=True)
            combined = combined.drop_duplicates(subset=["pa_path"]).reset_index(drop=True)
            updated = gpd.GeoDataFrame(combined, geometry="geometry", crs=self.epsg)
        else:
            updated = new_pairs

        updated.to_file(self.path_geodb, layer="Pairs")
        self._pairs = updated
        return updated

    def copy_geodb(self) -> bool:
        """Quickly create a copy of the project geopackage named backupath_geodb.gpkg
        """
        backup_path = pathlib.Path(self.path_root, "GMC_geodatabase_backup.gpkg")
        shutil.copytree(self.path_geodb, backup_path)
        # os.system(f"cp -r {self.path_geodb} {backup_path}")
        return pathlib.Path(self.path_root, "GMC_geodatabase_backup.gpkg").exists()

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
            image_types: set[str]
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
            georaster_metadata['filename'] = ta.name
            georaster_metadata['file_path'] = str(ta)
            georaster_metadata['xRes'] = r.res[0]
            georaster_metadata['yRes'] = r.res[1]
            georaster_metadata['bands'] = r.count
            georaster_metadata['rows'] = r.shape[0]
            georaster_metadata['cols'] = r.shape[1]
            sensor = re_searcher(ta.name.lower(), gmc_sensors.sensors(gmc_sensors.supported_sensors)) if "opt" in image_types else ("dem" if "dem" in image_types else "unknown")
            sensor_normalized = gmc_sensors.sensor_normalize(sensor)
            georaster_metadata['sensor'] = sensor_normalized['sensor']
            georaster_metadata['sensor_platform'] = sensor_normalized['platform']
            georaster_metadata['sensor_family'] = sensor_normalized['family']

            # Struggling to extract acquisition date, may be due to sensor-specific formats or missing metadata
            georaster_metadata["acq_datetime"] = gmc_sensors.extract_acquisition_date(ta, sensor=sensor)
            georaster_metadata["acq_date"] = georaster_metadata["acq_datetime"].date()

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
            georaster_metadata["src_crs"] = (src_crs.to_string() if hasattr(src_crs, "to_string") else str(src_crs)) if src_crs else None
            geom_proj = shp
            try:
                if target_crs and src_crs and str(src_crs) != target_crs:
                    geom_proj = gpd.GeoSeries([shp], crs=src_crs).to_crs(target_crs).iloc[0]
            except Exception:
                pass
            # Assign the projected geometry to the metadata dictionary
            georaster_metadata["geometry"] = geom_proj
            return georaster_metadata
        except Exception as e:
            print(f"Skip {ta}: {e}")
            return None
    
    # * Works properly
    def map_georasters_bank(self,
                            georasters_bank_path: str | pathlib.Path,
                            extensions: str | Sequence[str] = None,
                            image_type: str = "opt",
                            epsg_code: str | int | None = None,
                            suffix: str = "",
                            max_workers: int = None,
                            save_georaster_bank: bool = False) -> gpd.GeoDataFrame:
        """Map georasters in the specified bank directory using parallel processing.

        Args:
            georasters_bank_path (str | pathlib.Path): Path to the georaster bank directory.
            extensions (str | Sequence[str], optional): File extensions to include. Defaults to None.
            image_type (str, optional): Type of image to process. Defaults to "opt".
            epsg_code (str | int | None, optional): EPSG code for the target CRS. Defaults to None.
            suffix (str, optional): Suffix to append to output files. Defaults to "".
            max_workers (int, optional): Maximum number of workers for parallel processing. Defaults to None.
            save_georaster_bank (bool, optional): Whether to save the georaster bank. Defaults to False.

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

        # Normalize extensions (case-insensitive)
        targets = []
        for x in (extensions if isinstance(extensions, tuple) else extensions):
                targets += georasters_bank_path.glob(f"**/*.{x.lower()}")
                targets += georasters_bank_path.glob(f"**/*.{x.upper()}")
        # targets = list({p for x in extensions
        #         for p in [*georasters_bank_path.glob(f"**/*.{x.lower()}"), *georasters_bank_path.glob(f"**/*.{x.upper()}")]})

        if not targets:
            raise FileNotFoundError(f"No raster files with extensions {extensions} found in {georasters_bank_path}")
        _image_types = {image_type.lower()} if isinstance(image_type, str) else {str(t).lower() for t in image_type}
        # Choose a project/target CRS (prefer your session's EPSG)
        target_crs = f"EPSG:{epsg_code}" if epsg_code is not None else f"EPSG:{self.epsg}"
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.map_single_georaster, ta, target_crs, _image_types) for ta in targets]
            rows = []
            errors = []
            with alive_bar(len(futures), title=logger.search("Mapping georasters"), force_tty=True) as bar:
                for future in as_completed(futures):
                    try:
                        rec = future.result()
                        if rec:
                            rows.append(rec)
                    except Exception as e:
                        errors.append(str(e))
                    bar()

        if not rows:
            error_msg = "No features collected."
            if errors:
                error_msg += f"\nErrors encountered:\n" + "\n".join(errors[:5])  # Show first 5 errors
            raise RuntimeError(error_msg)

        # Rebuild geometries and GeoDataFrame
        df = pd.DataFrame(rows)
        layer_crs = target_crs or (df["src_crs"].dropna().iloc[0] if not df["src_crs"].dropna().empty else None)
        features_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=layer_crs)
        features_gdf.to_file(self.path_geodb, layer=f"Protomap")
        if save_georaster_bank:
            self.save_georaster_bank(features_gdf)
        return features_gdf
    
    # * Works properly
    def insert_pzone_to_gdb(self,
                            polygon: gpd.GeoDataFrame,
                            pz_name: str,
                            pz_shortname: str,
                            **attributes):
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
        row = {"pz_name": pz_name, "pz_shortname": pz_shortname,"geometry": polygon}
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
            "spot":      "#e6194b",
            "sentinel2": "#3cb44b",
            "landsat":   "#4363d8",
            "pleiades":  "#f58231",
            "planet":    "#911eb4",
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
                    m.add_layer(GeoData(
                        geo_dataframe=_json_safe(group.reset_index(drop=True)),
                        style={
                            "color": color, "weight": 1.5,
                            "fillColor": color, "fillOpacity": 0.1,
                        },
                        hover_style={"fillOpacity": 0.4},
                        name=str(sensor),
                    ))
            else:
                m.add_layer(GeoData(
                    geo_dataframe=_json_safe(rb_4326),
                    style={"color": "#3388ff", "weight": 1.5, "fillOpacity": 0.1},
                    hover_style={"fillOpacity": 0.4},
                    name="Raster Bank",
                ))

            m.add_control(LayersControl())

            bounds = rb_4326.total_bounds  # [minx, miny, maxx, maxy] = [lon_min, lat_min, lon_max, lat_max]
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        draw_control = DrawControl(
            polyline={}, circlemarker={}, marker={}, circle={},
            rectangle={"shapeOptions": {"color": "#3388ff", "weight": 3}},
        )
        m.add_control(draw_control)
        drawn_polygons = []

        def handle_draw(target, action, geo_json):
            if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
                poly = shape(geo_json['geometry'])
                drawn_polygons.append(poly)
                print('Polygon drawn! Coordinates:', poly.exterior.coords[:])

        draw_control.on_draw(handle_draw)
        display(m)
        print("Draw your polygon(s) on the map above. When done, run the next cell to get the GeoDataFrame.")

        def get_gdf():
            if drawn_polygons:
                gdf = gpd.GeoDataFrame(geometry=drawn_polygons, crs=default_crs)
                target_crs = getattr(self, 'epsg', None)
                if target_crs and str(gdf.crs) != f"EPSG:{target_crs}":
                    try:
                        gdf = gdf.to_crs(epsg=int(target_crs))
                    except Exception as e:
                        print(f"CRS conversion failed: {e}")
                return gdf
            else:
                print('No polygons drawn.')
                return None
        return get_gdf

    def plot_pairs_network(
        self,
        criterias: str | list[str] = "",
        figsize: tuple[int, int] = (14, 5),
        color_by: str = "sensor",
        ax=None,
    ) -> tuple:
        """Plot built image pairs as a temporal network diagram.

        Each acquisition date is a node on a horizontal timeline; each pair is
        a Bézier arc connecting its two dates.  Arc height is proportional to
        the time-delta (pa_dt_years), so short pairs sit low and long pairs
        arch higher.

        Args:
            criterias: passed to get_pairs_overview() to filter pairs.
            figsize: matplotlib figure size (ignored when *ax* is provided).
            color_by: "sensor" | "pzone" | "dt" — controls arc/node colour.
            ax: existing matplotlib Axes to draw into; a new figure is created
                when None.

        Returns:
            (fig, ax) tuple.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.path import Path as MplPath

        pairs_gdf = self.get_pairs_overview(criterias)
        if pairs_gdf.empty:
            raise ValueError("No pairs found. Build pairs first with update_pairs().")

        thumbs_gdf = self.get_thumbs_overview()

        # --- decimal-year helper ------------------------------------------------
        def _to_dec(date_str: str) -> float:
            dt = pd.Timestamp(date_str)
            yr = dt.year
            return yr + (dt - pd.Timestamp(yr, 1, 1)) / pd.Timedelta(days=365.25)

        # --- date → sensor mapping (from thumbs) --------------------------------
        date_sensor: dict[str, str] = {}
        if not thumbs_gdf.empty:
            for row in thumbs_gdf.itertuples(index=False):
                date_sensor[row.th_date] = row.th_sensor

        # --- unique sorted dates ------------------------------------------------
        left_dates = pairs_gdf["pa_left_date"].tolist()
        right_dates = pairs_gdf["pa_right_date"].tolist()
        all_date_strs = sorted(set(left_dates + right_dates))
        date_x: dict[str, float] = {d: _to_dec(d) for d in all_date_strs}

        # --- colour palette for the chosen dimension ----------------------------
        if color_by == "sensor":
            sensors = sorted({date_sensor.get(d, "unknown") for d in all_date_strs})
            palette = cm.get_cmap("tab10", max(len(sensors), 1))
            sensor_color = {s: palette(i) for i, s in enumerate(sensors)}
            def arc_color(row):
                return sensor_color.get(date_sensor.get(row.pa_left_date, "unknown"), "grey")
            def node_color(date_str):
                return sensor_color.get(date_sensor.get(date_str, "unknown"), "grey")
            legend_handles = [
                mpatches.Patch(color=sensor_color[s], label=s) for s in sensors
            ]

        elif color_by == "pzone":
            pzones = sorted(pairs_gdf["pa_pz_name"].unique())
            palette = cm.get_cmap("Set2", max(len(pzones), 1))
            pzone_color = {pz: palette(i) for i, pz in enumerate(pzones)}
            def arc_color(row):
                return pzone_color.get(row.pa_pz_name, "grey")
            def node_color(date_str):
                return "black"
            legend_handles = [
                mpatches.Patch(color=pzone_color[pz], label=pz) for pz in pzones
            ]

        else:  # "dt"
            dt_vals = pairs_gdf["pa_dt_years"].astype(float)
            dt_min, dt_max = dt_vals.min(), dt_vals.max()
            norm = mcolors.Normalize(vmin=dt_min, vmax=dt_max)
            cmap = cm.get_cmap("plasma_r")
            def arc_color(row):
                return cmap(norm(float(row.pa_dt_years)))
            def node_color(date_str):
                return "black"
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            legend_handles = [sm]

        # --- figure setup -------------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # --- Bézier arc helper --------------------------------------------------
        def _draw_arc(x1: float, x2: float, height: float, color, lw: float = 1.5, alpha: float = 0.75):
            verts = [(x1, 0.0), (x1, height), (x2, height), (x2, 0.0)]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
            patch = mpatches.PathPatch(
                MplPath(verts, codes),
                facecolor="none", edgecolor=color, lw=lw, alpha=alpha,
            )
            ax.add_patch(patch)

        # --- draw each pair as an arc -------------------------------------------
        for row in pairs_gdf.itertuples(index=False):
            x1 = date_x[row.pa_left_date]
            x2 = date_x[row.pa_right_date]
            if x1 == x2:
                continue
            height = max(float(row.pa_dt_years) * 0.5, 0.05)
            _draw_arc(x1, x2, height, arc_color(row))

        # --- timeline and date nodes --------------------------------------------
        x_vals = list(date_x.values())
        ax.axhline(0, color="black", lw=0.8, zorder=1)
        for date_str, x in date_x.items():
            ax.scatter(x, 0, color=node_color(date_str), s=60, zorder=3,
                       edgecolors="black", linewidths=0.5)

        # --- axes formatting ----------------------------------------------------
        x_pad = (max(x_vals) - min(x_vals)) * 0.03 if len(x_vals) > 1 else 0.1
        ax.set_xlim(min(x_vals) - x_pad, max(x_vals) + x_pad)
        ax.set_ylim(-0.05, None)

        ax.set_xticks(list(date_x.values()))
        ax.set_xticklabels(list(date_x.keys()), rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_ylabel("")
        for spine in ("left", "right", "top", "bottom"):
            ax.spines[spine].set_visible(False)

        n_pairs = len(pairs_gdf)
        ax.set_title(
            f"Pair network — {n_pairs} pair{'s' if n_pairs != 1 else ''} "
            f"({len(all_date_strs)} acquisition dates)",
            fontsize=10,
        )

        if color_by == "dt":
            fig.colorbar(legend_handles[0], ax=ax, orientation="vertical",
                         label="Time delta (years)", fraction=0.03, pad=0.02)
        else:
            ax.legend(handles=legend_handles, title=color_by.capitalize(),
                      loc="upper left", fontsize=8, title_fontsize=9)

        fig.tight_layout()
        return fig, ax

    def plot_pairs_chord(
        self,
        criterias: str | list[str] = "",
        figsize: tuple[int, int] = (10, 10),
    ) -> tuple:
        """Plot built image pairs as a chord diagram.

        Dates are arranged in a circle, each pair is a chord (line) connecting
        its two dates. Date nodes are coloured by viridis (temporal progression),
        and chord lines are black.

        Args:
            criterias: passed to get_pairs_overview() to filter pairs.
            figsize: matplotlib figure size.

        Returns:
            (fig, ax) tuple.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import numpy as np

        pairs_gdf = self.get_pairs_overview(criterias)
        if pairs_gdf.empty:
            raise ValueError("No pairs found. Build pairs first with update_pairs().")

        # --- decimal-year helper ------------------------------------------------
        def _to_dec(date_str: str) -> float:
            dt = pd.Timestamp(date_str)
            yr = dt.year
            return yr + (dt - pd.Timestamp(yr, 1, 1)) / pd.Timedelta(days=365.25)

        # --- unique sorted dates ------------------------------------------------
        left_dates = pairs_gdf["pa_left_date"].tolist()
        right_dates = pairs_gdf["pa_right_date"].tolist()
        all_date_strs = sorted(set(left_dates + right_dates))
        n_dates = len(all_date_strs)

        if n_dates < 2:
            raise ValueError("Need at least 2 dates to draw a chord diagram.")

        # --- place dates on circle (full 360°) ----------------------------------
        angles = np.linspace(0, 2 * np.pi, n_dates, endpoint=False)
        date_angle: dict[str, float] = {d: angles[i] for i, d in enumerate(all_date_strs)}
        date_xy: dict[str, tuple] = {
            d: (np.cos(date_angle[d]), np.sin(date_angle[d]))
            for d in all_date_strs
        }

        # --- colour dates by viridis (decimal year) ----------------------------
        dec_years = np.array([_to_dec(d) for d in all_date_strs])
        norm = mcolors.Normalize(vmin=dec_years.min(), vmax=dec_years.max())
        cmap = cm.get_cmap("viridis")

        # --- figure setup -------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))

        # --- draw chords (pair lines) in black ----------------------------------
        for row in pairs_gdf.itertuples(index=False):
            x1, y1 = date_xy[row.pa_left_date]
            x2, y2 = date_xy[row.pa_right_date]
            ax.plot([x1, x2], [y1, y2], color="black", lw=0.8, alpha=0.6, zorder=1)

        # --- draw date nodes on circle, coloured by viridis --------------------
        for date_str, (x, y) in date_xy.items():
            dec_yr = _to_dec(date_str)
            color = cmap(norm(dec_yr))
            ax.scatter(x, y, s=100, color=color, edgecolors="black", linewidths=0.8,
                       zorder=3, alpha=0.9)

        # --- labels around the circle (rotated) ---------------------------------
        for date_str, (x, y) in date_xy.items():
            angle_deg = date_angle[date_str] * 180 / np.pi
            # Adjust label position to be slightly outside the circle
            offset = 1.15
            label_x = offset * x
            label_y = offset * y
            # Rotate text to align radially
            ha = "center"
            if 90 < angle_deg < 270:
                angle_deg += 180  # flip for readability
                ha = "center"
            ax.text(label_x, label_y, date_str, fontsize=7, ha=ha, va="center",
                    rotation_mode="anchor", rotation=angle_deg)

        # --- clean up axes ------------------------------------------------------
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        n_pairs = len(pairs_gdf)
        ax.set_title(
            f"Pair network (chord diagram) — {n_pairs} pair{'s' if n_pairs != 1 else ''} "
            f"({n_dates} acquisition dates)",
            fontsize=11, pad=20,
        )

        # --- colorbar for viridis (time progression) ---------------------------
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
        cbar.set_label("Acquisition year", fontsize=9)

        fig.tight_layout()
        return fig, ax

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
        xmin, ymin, xmax, ymax = pz_vec.get_bounds_projected(out_crs=f"EPSG:{target_epsg}")
        xmin = math.floor(xmin / target_resolution) * target_resolution
        ymin = math.floor(ymin / target_resolution) * target_resolution
        xmax = math.ceil(xmax  / target_resolution) * target_resolution
        ymax = math.ceil(ymax  / target_resolution) * target_resolution
        w = round((xmax - xmin) / target_resolution)
        h = round((ymax - ymin) / target_resolution)
        return rio_coords.BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax), (w, h)

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
            pz_vec = gu.Vector(gpd.GeoDataFrame(geometry=pzone_vector, crs=pzone_vector.crs))
        else:
            raise TypeError(f"pzone_vector must be gu.Vector, GeoDataFrame, or GeoSeries (got {type(pzone_vector)}).")

        # Open the source raster lazily. Do NOT call set_nodata — it triggers a full load.
        rst = gu.Raster(mosaic_path, bands=selected_band, nodata=source_nodata)

        # Early-out: disjoint bounds → no work to do.
        src_xmin, src_ymin, src_xmax, src_ymax = pz_vec.get_bounds_projected(out_crs=rst.crs)
        rxmin, rymin, rxmax, rymax = rst.bounds.left, rst.bounds.bottom, rst.bounds.right, rst.bounds.top
        if (src_xmax <= rxmin) or (src_xmin >= rxmax) or (src_ymax <= rymin) or (src_ymin >= rymax):
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

        shape_mismatch = (
            canonical_grid_size is not None
            and m_crop.shape[-2:] != (canonical_grid_size[1], canonical_grid_size[0])
        )
        needs_reproject = (
            (m_crop.crs.to_epsg() != target_epsg)
            or (m_crop.res[0] != target_resolution)
            or shape_mismatch
        )
        if not needs_reproject:
            return m_crop

        # Estimate output array size to decide between in-memory and multiproc warp
        tgt_xmin, tgt_ymin, tgt_xmax, tgt_ymax = pz_vec.get_bounds_projected(out_crs=f"EPSG:{target_epsg}")
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
                chunk = max(512, int((use_multiproc_threshold_mb * 1024 * 1024 / bytes_per_px) ** 0.5))
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
    ) -> dict[str, list[pathlib.Path]]:
        """Bulk sieve: crop all mosaics in ``georasters_map`` to each pzone AOI.

        For each pzone and each (acq_date, sensor) group, sequentially crops
        every intersecting mosaic via ``sieve_single_mosaic``, merges
        the cropped tiles if needed, and writes a single thumb to the pzone's
        raster_data directory. Sequential processing keeps peak memory bounded.

        Args:
            georasters_map: Georasters map (as returned by ``get_georasters_map``).
                If None, loaded from the session using ``image_type``/``suffix``.
            image_type: 'opt' or 'dem'.
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

        Returns:
            Dict mapping pzone name to the list of thumb paths written.
        """
        assert len(self.get_pzones_overview()) > 0, "no pzone registered for this project"

        if georasters_map is None:
            georasters_map = self.get_georasters_map(image_type, suffix)

        logger.info(f"Map epsg : {georasters_map.crs}\nPzone epsg : {self.get_pzones_overview().crs}")
        logger.info(f"Target resolution: {target_resolution} (EPSG:{self.epsg})")

        written: dict[str, list[pathlib.Path]] = {}

        for pz in self.get_pzones_overview().iloc:
            if pzone_filter is not None and pz["pz_name"] not in pzone_filter:
                continue

            logger.info(f"Pzone: {pz['pz_name']}")

            pz_vec = gu.Vector(
                gpd.GeoDataFrame([pz], crs=self.get_pzones_overview().crs, geometry="geometry")
            )

            # Use provided canonical grid if given, otherwise compute per pzone
            if canonical_bounds is not None and canonical_grid_size is not None:
                canon_bounds, canon_size = canonical_bounds, canonical_grid_size
            else:
                canon_bounds, canon_size = self._compute_canonical_grid(
                    pz_vec, self.epsg, target_resolution
                )

            pz_rasterdata = pathlib.Path(self.path_root, f"raster_data_{self.project_name}", pz["pz_name"])
            if image_type == "opt":
                pz_opticals = pathlib.Path(pz_rasterdata, "opticals")
                pz_disps = pathlib.Path(pz_rasterdata, "image_correlation")
                for p in [pz_rasterdata, pz_opticals, pz_disps]:
                    if not p.exists():
                        p.mkdir(parents=True)
            else:
                if not pz_rasterdata.exists():
                    pz_rasterdata.mkdir(parents=True)

            selection = georasters_map[georasters_map["geometry"].intersects(pz["geometry"])]
            logger.info(f"Intersecting rasters: {len(selection)}")
            if len(selection) == 0:
                written[pz["pz_name"]] = []
                continue

            groups = selection.groupby(["acq_date", "sensor"])["file_path"].apply(list)
            pz_written: list[pathlib.Path] = []

            for group_id, group in enumerate(groups):
                acq_date = groups.index[group_id][0]
                sensor = groups.index[group_id][1]

                if image_type == "opt":
                    thumb_name = f"{pz['pz_name']}_{acq_date}_{sensor}.tif"
                    thumb_path = pathlib.Path(pz_opticals, thumb_name)
                elif image_type == "dem":
                    thumb_name = f"{pz['pz_name']}_dem.tif"
                    thumb_path = pathlib.Path(pz_rasterdata, thumb_name)
                else:
                    raise ValueError(f"Unsupported image_type: {image_type}")

                if skip_existing and thumb_path.exists():
                    logger.info(f"Skipping existing: {thumb_name}")
                    pz_written.append(thumb_path)
                    continue

                fully: list[gu.Raster] = []
                with alive_bar(len(group), title=f"Cropping {thumb_name}", force_tty=True) as bar:
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
                        finally:
                            bar()

                if len(fully) == 0:
                    logger.warning(f"No usable data for {thumb_name}")
                    continue
                elif len(fully) == 1:
                    thumb = fully[0]
                else:
                    logger.info(f"Merging {len(fully)} tiles for {thumb_name}")
                    thumb = gu.raster.multiraster.merge_rasters(
                        fully,
                        merge_algorithm=merge_algorithm,
                        resampling_method=resampling_method,
                        use_ref_bounds=False,
                    )

                thumb.save(thumb_path, dtype=out_dtype)
                logger.save(f"Saved: {thumb_path}")
                pz_written.append(thumb_path)

                del thumb
                del fully

            written[pz["pz_name"]] = pz_written

        return written
    
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
        conda_env: str = "gmc_env", # TO delete (deprecated, use cluster-specific envs instead)
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        corr_tile_size: int = 2048,
        corr_algorithm: str = "asp_bm",
        corr_kernel: tuple[int, int] = (21, 21),
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
    ) -> list[pathlib.Path]:
        """Prepare the output directory and bash script for each pair.

        For every pair matching *criterias*:

        1. Skip if the pair is already ``'complete'`` (unless *overwrite=True*).
        2. Create the pair output directory (``pair.pa_path``).
        3. Clip left and right images to their spatial intersection if not done yet.
        4. Write ``correlation_job.sh`` with the full ASP command.

        Args:
            criterias: Passed to :meth:`get_pairs` to filter pairs.
            cluster: ``None`` / ``"local"`` / ``"gricad"`` / ``"isterre"``.
            overwrite: If *True*, redo pairs that are already complete.
            asp_bin_dir: Optional path to the ASP bin directory.
            nodes / cores / walltime / besteffort / conda_env: OAR cluster options.
            processes / threads_* / corr_* / etc: ``parallel_stereo`` parameters.

        Returns:
            List of paths to the generated bash scripts.
        """
        from geomulticorr.correlation import ASP

        asp = ASP(session=self, asp_bin_dir=asp_bin_dir)
        pairs = self.get_pairs(criterias)

        if not pairs:
            logger.warning("No pairs found for the given criteria. Nothing to prepare.")
            return []

        cluster_label = cluster if cluster else "local"
        logger.info(
            f"Preparing {len(pairs)} pair(s) for correlation "
            f"[cluster={cluster_label}]"
        )

        bash_scripts: list[pathlib.Path] = []

        for pair in pairs:
            status = pair.get_status()

            if all(pair.check_corr_outputs().values()) and not overwrite:
                logger.info(f"Skipping '{pair.pa_key}' (already complete)")
                continue

            # 1. Create output directory
            pair.pa_path.mkdir(parents=True, exist_ok=True)

            # 2. Clip inputs if not already done
            if status not in ("clipped", "complete"):
                logger.info(f"Clipping '{pair.pa_key}'")
                pair.clip()

            # 3. Reuse existing script if allowed
            bash_path = pair.pa_path / f"{pair.pa_key}_CorrelationJob.sh"
            if bash_path.exists() and not overwrite_scripts:
                logger.info(f"Reusing existing script: {bash_path.name}")
                bash_scripts.append(bash_path)
                continue

            # 4. Write correlation parameters settings file
            params_file = pair.pa_path / f"{pair.pa_key}_CorrParameters.txt"
            asp.build_correlation_params(
                corr_algorithm=corr_algorithm,
                corr_kernel=corr_kernel,
                corr_xthreshold=corr_xthreshold,
                corr_seed_mode=corr_seed_mode,
                subpixel_refinement_mode=subpixel_mode,
                subpixel_kernel=subpixel_kernel,
                prefilter_mode=prefilter_mode,
                cost_mode=cost_mode,
                save_disparity_difference=save_disparity_difference,
                params_file_path=params_file,
            )

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
                corr_algorithm=corr_algorithm,
                corr_kernel=corr_kernel,
                metric=metric,
            )
            bash_scripts.append(bash_path)
            logger.info(f"Ready: '{pair.pa_key}' → {bash_path.name}")

        logger.success(
            f"Prepared {len(bash_scripts)} bash script(s) in their respective pair directories."
        )
        return bash_scripts

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

        Returns:
            List of return codes (one per pair; 0 = success / submitted).
        """
        bash_scripts = self.prepare_pairs_correlation(
            criterias=criterias,
            cluster=cluster,
            overwrite=overwrite,
            overwrite_scripts=overwrite_scripts,
            asp_bin_dir=asp_bin_dir,
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
        )

        if not bash_scripts:
            return []

        if clean:
            from geomulticorr.correlation import ASP
            asp = ASP(session=self, asp_bin_dir=asp_bin_dir)
            pair_map = {p.pa_key: p for p in self.get_pairs(criterias)}
            active_pairs = [pair_map.get(s.parent.name) for s in bash_scripts]
        else:
            asp = None
            active_pairs = [None] * len(bash_scripts)

        return_codes: list[int] = []
        use_cluster = cluster in ("gricad", "isterre")

        for pair, script in zip(active_pairs, bash_scripts):
            if use_cluster:
                cmd = ["oarsub", "-S", f"./{script.name}"]
            else:
                cmd = ["bash", str(script)]

            if dry_run:
                print(f"DRY RUN [{script.parent.name}]: {' '.join(cmd)}")
                return_codes.append(0)
                continue

            logger.info(f"Launching: {script.name}")
            result = subprocess.run(
                cmd,
                cwd=script.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                logger.error(
                    f"Failed [{script.parent.name}]: {result.stderr.strip()}"
                )
            else:
                if clean and pair is not None:
                    asp.clean_pair_directory(pair)
            return_codes.append(result.returncode)

        n_ok = sum(r == 0 for r in return_codes)
        logger.success(
            f"Launched {n_ok}/{len(return_codes)} pair(s) successfully."
        )

        # Sync pa_status back to the geodatabase for local runs.
        # For cluster (oarsub) jobs the processes are async — call
        # _sync_pairs_status(get_pairs(criterias)) manually after they finish.
        if not use_cluster and not dry_run:
            run_pairs = [p for p in active_pairs if p is not None]
            self._sync_pairs_status(run_pairs)

        return return_codes

    def extract_pairs_raw_displacements(
        self,
        criterias: str | list[str] = "",
        save_plot: bool = True,
        overwrite: bool = False,
    ) -> dict:
        """Extract EW/NS displacements and compute raw stats for all complete pairs.

        Iterates over all pairs matching *criterias*, calls
        :meth:`~geomulticorr.core.pair.Pair.extract_raw_displacements` on each
        complete pair, and returns a summary dict keyed by ``pa_key``.

        Args:
            criterias: Same filter syntax as :meth:`get_pairs`.
            save_plot: Save the 9-panel control figure for each pair.
            overwrite: Re-process pairs that already have EW/NS files on disk.

        Returns:
            Dict mapping ``pa_key`` → full stats dict (or ``None`` if skipped).
        """
        pairs = self.get_pairs(criterias)
        results: dict = {}
        n_done = 0
        n_skip = 0

        for pair in pairs:
            if not pair.pa_disparity_f_path.exists():
                logger.warning(f"Skipping '{pair.pa_key}' — disparity raster not found")
                results[pair.pa_key] = None
                n_skip += 1
                continue

            if not overwrite and pair.pa_ew_path.exists() and pair.pa_ns_path.exists():
                logger.info(f"Already processed '{pair.pa_key}', skipping (use overwrite=True to redo)")
                results[pair.pa_key] = None
                n_skip += 1
                continue

            logger.launch(f"Extracting raw displacements for '{pair.pa_key}'")
            try:
                stats = pair.extract_raw_displacements(save_plot=save_plot)
                results[pair.pa_key] = stats
                n_done += 1
            except Exception as exc:
                logger.error(f"Failed for '{pair.pa_key}': {exc}")
                results[pair.pa_key] = None

        logger.success(
            f"Extracted raw displacements for {n_done}/{len(pairs)} pair(s) "
            f"({n_skip} skipped)."
        )
        return results

    def apply_pairs_corrections(
        self,
        correction_pipeline=None,
        filter_pipeline=None,
        criterias: str | list[str] = "",
        overwrite: bool = False,
        save_plot: bool = True,
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
            overwrite: Re-process pairs whose corrected files already exist.
            save_plot: Save a before/after control figure for each pair.
            **kwargs: Forwarded to both pipelines
                (e.g. ``dem=`` for :class:`~geomulticorr.corrections.TopoCorrection`).
                ``cc=`` is auto-injected from the pair folder when available.

        Returns:
            Dict mapping ``pa_key`` → ``{"ew": Path, "ns": Path}`` or ``None`` if skipped.

        Raises:
            ValueError: If neither *correction_pipeline* nor *filter_pipeline* is provided,
                or if a required kwarg (e.g. ``dem=``) is missing.
        """
        import geoutils as gu
        import matplotlib.pyplot as plt
        from geomulticorr.corrections.corrections import make_corrections
        from geomulticorr.corrections.masks import FilterPipeline, BaseMask

        if correction_pipeline is None and filter_pipeline is None:
            raise ValueError(
                "Provide at least one of correction_pipeline= or filter_pipeline=."
            )

        pairs = self.get_pairs(criterias)
        results: dict = {}
        n_done = 0
        n_skip = 0

        for pair in pairs:
            ew_corr = pair.pa_path / f"{pair.pa_key}-F_EW_corr.tif"
            ns_corr = pair.pa_path / f"{pair.pa_key}-F_NS_corr.tif"

            if not pair.pa_ew_path.exists() or not pair.pa_ns_path.exists():
                logger.warning(f"Skipping '{pair.pa_key}' — EW/NS rasters not found")
                results[pair.pa_key] = None
                n_skip += 1
                continue

            if not overwrite and ew_corr.exists() and ns_corr.exists():
                logger.info(f"Already corrected '{pair.pa_key}', skipping (use overwrite=True to redo)")
                results[pair.pa_key] = None
                n_skip += 1
                continue

            logger.launch(f"Applying corrections for '{pair.pa_key}'")
            try:
                x_raw = gu.Raster(str(pair.pa_ew_path))
                y_raw = gu.Raster(str(pair.pa_ns_path))

                # Auto-inject cc= from the pair folder if not already provided
                pair_kwargs = dict(kwargs)
                if "cc" not in pair_kwargs and pair.pa_cc_path.exists():
                    pair_kwargs["cc"] = gu.Raster(str(pair.pa_cc_path))

                # Validate required kwargs upfront
                all_steps = []
                if filter_pipeline is not None:
                    fp = filter_pipeline
                    all_steps += fp.masks if isinstance(fp, FilterPipeline) else [fp]
                if correction_pipeline is not None:
                    from geomulticorr.corrections.corrections import CorrectionPipeline
                    cp = correction_pipeline
                    all_steps += cp.steps if isinstance(cp, CorrectionPipeline) else [cp]
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
                    x_filt, y_filt = filter_pipeline.apply(x_raw, y_raw, **pair_kwargs)
                else:
                    x_filt, y_filt = x_raw, y_raw

                # Step 2: correction pipeline — fit + apply each step individually
                # so stats can be recorded after every correction.
                import copy
                import geomulticorr.stats.stats as gmc_stats

                x_corr, y_corr = x_filt, y_filt
                if correction_pipeline is not None:
                    cp_x = copy.deepcopy(correction_pipeline)
                    cp_y = copy.deepcopy(correction_pipeline)
                    steps_x = cp_x.steps if isinstance(cp_x, CorrectionPipeline) else [cp_x]
                    steps_y = cp_y.steps if isinstance(cp_y, CorrectionPipeline) else [cp_y]
                    for step_x, step_y in zip(steps_x, steps_y):
                        step_name = type(step_x).__name__
                        step_x.fit(x_corr, stable_mask=x_stable, **pair_kwargs)
                        x_corr = step_x.apply(x_corr)
                        step_y.fit(y_corr, stable_mask=y_stable, **pair_kwargs)
                        y_corr = step_y.apply(y_corr)
                        gmc_stats.save_corrected_stats(pair, x_corr, y_corr, step_name)

                # Optional before/after control plot
                if save_plot:
                    try:
                        from geomulticorr.utils.gmc_functions import plot_correction_result
                        fig = plot_correction_result(
                            None, x_raw, x_corr, y_raw, y_corr,
                            fig_name=pair.pa_key,
                            **pair_kwargs,
                        )
                        plot_path = pair.pa_path / f"{pair.pa_key}_corrections_disp.jpg"
                        fig.savefig(str(plot_path), dpi=150, format="JPEG", bbox_inches="tight")
                        plt.close(fig)
                        logger.save(f"Correction plot saved: {plot_path.name}")
                    except Exception as plot_exc:
                        logger.warning(f"Could not save control plot: {plot_exc}")

                x_corr.save(str(ew_corr))
                y_corr.save(str(ns_corr))
                logger.file(f"Saved corrected EW: {ew_corr.name}")
                logger.file(f"Saved corrected NS: {ns_corr.name}")
                results[pair.pa_key] = {"ew": ew_corr, "ns": ns_corr}
                n_done += 1
            except Exception as exc:
                logger.error(f"Failed for '{pair.pa_key}': {exc}")
                results[pair.pa_key] = None

        logger.success(
            f"Applied corrections for {n_done}/{len(pairs)} pair(s) "
            f"({n_skip} skipped)."
        )
        return results

    # TODO: to be completed with the other parameters
    def pr_full(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10):
        """Launch all the possible correlations accross all the valid images among the project pzones"""
        logs = []
        for pzone in self.get_pzones():
            logs.append(
                pzone.pz_full(self.epsg, corr_algorithm, corr_kernel_size, corr_xthreshold))
        return pd.DataFrame(logs)

    def _search_engine_thibaut(self, layername: str, criterias: str | list[str] = "") -> gpd.GeoDataFrame:
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
        criterias = [] if criterias in ("", []) else ([criterias] if isinstance(criterias, (str, int)) else list(criterias))

        # # add an "and" statement between each criteria
        # pattern = ""
        # for c in criterias:
        #     if type(c) == int:
        #         c = str(c)
        #     pattern += f"(?=.*{c.lower()})"
        # pattern = re.compile(pattern)
        pattern = None if not criterias else "^" + "".join(f"(?=.*{re.escape(str(c))})" for c in criterias)

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

    def _search_engine(self, layername: str, criterias: str | list[str] = "") -> gpd.GeoDataFrame:
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
        pattern = None if not criterias else "".join(f"(?=.*{re.escape(c)})" for c in criterias)

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
            m1 = contains_ci(df.get("ge_pz_name", pd.Series("", index=df.index)), pattern)
            m2 = contains_ci(df.get("ge_frogi_id", pd.Series("", index=df.index)).astype(str), pattern)
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
