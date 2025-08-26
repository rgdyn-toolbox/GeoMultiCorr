#!/usr/bin/env python
# coding=utf-8
# ----------------------------------------------------------------------------- #
# GeoMultiCorr (GMC) project
# Copyright (C) GeoMultiCorr developer team, 2024.
# session.py
# creation date: 2024-06-10.
#
# Author(s) metadata
# -> author: Diego CUSICANQUI
#   -> affiliation: CNES | ISTerre | Univ. Grenoble Alpes
#   -> email(s): diego.cusicanqui@univ-grenoble-alpes.fr | diego.cusicanqui.vg@gmail.com
# -> author: Thibault DUVANEL
#   -> affiliation: UNIL | Univ. Lausanne
#   -> email(s): thibaut.duvanel@unil.ch
# ----------------------------------------------------------------------------- #

import re
import tqdm
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional, Sequence

# Importing third-party libraries
import itertools
from osgeo import gdal
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
import geomulticorr.geomorph as gmc_geomorph
import geomulticorr.pzone as gmc_pzone
import geomulticorr.pair as gmc_pair
import geomulticorr.spine as gmc_spine
import geomulticorr.thumb as gmc_thumb
import geomulticorr.xzone as gmc_xzone

file_location = Path(__file__)

project_template_location = Path(file_location.parent, "resources", "project_template")

# 1. Catalog of known sensors and their patterns
SENSOR_CATALOG = [
    # Landsat
    {"patterns": [r"landsat\s*([4-9])", r"\bl([4-9])\b"], "sensor": "LANDSAT{num}", "family": "Landsat", "vendor": "USGS/NASA"},
    # Sentinel-2
    {"patterns": [r"sentinel[-\s]?2[ab]?", r"\bs2[ab]?\b"], "sensor": "SENTINEL2", "family": "Sentinel-2", "vendor": "ESA/Copernicus"},
    # SPOT
    {"patterns": [r"spot\s*([4-7])", r"\bsp([4-7])\b"], "sensor": "SPOT{num}", "family": "SPOT", "vendor": "Airbus"},
    # Pléiades
    {"patterns": [r"pl[eé]iades\s*(1a|1b)?", r"pleiades-neo", r"pneo"], "sensor": "PLEIADES", "family": "Pléiades", "vendor": "Airbus"},
    # PlanetScope
    {"patterns": [r"planetscope", r"psscene", r"superdove", r"ps2(sd)?", r"pscope"], "sensor": "PLANETSCOPE", "family": "PlanetScope", "vendor": "Planet"},
    # WorldView
    {"patterns": [r"worldview[-\s]?([1-4])", r"\bwv([1-4])\b"], "sensor": "WORLDVIEW{num}", "family": "WorldView", "vendor": "Maxar"},
    # GeoEye
    {"patterns": [r"geoeye[-\s]?1", r"\bge1\b"], "sensor": "GEOEYE1", "family": "GeoEye", "vendor": "Maxar"},
    # QuickBird
    {"patterns": [r"quickbird", r"\bqb[12]?\b"], "sensor": "QUICKBIRD", "family": "QuickBird", "vendor": "Maxar"},
    # IKONOS
    {"patterns": [r"ikonos"], "sensor": "IKONOS", "family": "IKONOS", "vendor": "Maxar"},
    # RapidEye
    {"patterns": [r"rapideye", r"\bre([1-5])\b"], "sensor": "RAPIDEYE{num}", "family": "RapidEye", "vendor": "Planet"},
    # Aerial/UAV
    {"patterns": [r"swissimage", r"aerial", r"uav", r"drone", r"orthophoto"], "sensor": "AERIAL", "family": "Aerial", "vendor": "Various"},
    # DEM
    {"patterns": [r"dem", r"hillshade"], "sensor": "DEM", "family": "DEM", "vendor": "Various"},
]

def print_infoBM(text: str,
                bold: bool = False) -> None:
    """Prints a formatted message to the console.

    Args:
        text (str): The string to be printed.
        bold (bool, optional): Whether to print the text in bold style. Defaults to False.
    """
    GMC_TEXT = "[ GMC-info ] :"
    if bold:
        print(f"\033[1m{GMC_TEXT} {text}\033[1m")
    else:
        print(f"{GMC_TEXT} {text}")
#END def

def is_conform_to_gmc_template(target_root_path: str | Path) -> bool:
    #! I dont know if this functions is useful. Check with Thibaut
    """Check GeoMultiCorr structure.
    Checks the 3 conditions to ensure that the 'target_root_path' leads to a folder conforming to the GeoMultiCorr project data architecture.

    Args:
        target_root_path (str | Path): Path where GeoMultiCorr project data will be stored.

    Returns:
        bool: True if the folder conforms to the GeoMultiCorr project data architecture. False otherwise.
    """
    target_root_path = Path(target_root_path)
    target_name = target_root_path.name

    # Firstly, there is a folder named raster_data_parent-folder-name
    target_raster_data_expected_path = Path(target_root_path, f"raster_data_{target_name}")
    if not target_raster_data_expected_path.is_dir():
        return False

    # Secondly, there is a file "GMC_mapset_{target_name}.qgz"
    target_map_expected_path = Path(target_root_path, f"GMC_mapset_{target_name}.qgz")
    if not target_map_expected_path.exists():
        return False

    # Thirdly, there is a file GMC_geodatabase_{target_name}.gpkg
    target_geodb_expected_path = Path(target_root_path, f"GMC_geodatabase_{target_name}.gpkg")
    if not target_geodb_expected_path.exists():
        return False

    return True

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

def re_searcherB(string: str, pattern: str) -> str:
    """Search for a pattern in a string and return the first match.

    Args:
        string (str): The string to search in.
        pattern (str): The pattern to search for.

    Returns:
        str: The first match within the string.
    """
    regex = re.compile(r"(?i)(" + "|".join(map(re.escape, string)) + r")")
    match = regex.search(string)
    return match.group(0) if match else "unknown"
    #END try

def search_date_in_filename(filename: str | Path) -> str | None:
    """Find the first date in a filename and normalize to YYYY-MM-dd.

    Supports the following inputs:
    - YYYYmmdd
    - YYYY-mm-dd
    - YYYY/mm/dd
    - DD-MM-YYYY
    - DD/MM/YYYY
    Returns None if no date is found.
    """
    # Ordered list of (regex, strptime_format) pairs
    patterns = [
        # YYYYmmdd
        (r"(19\d{2}|20\d{2}|21\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", "%Y%m%d"),
        # YYYY-mm-dd
        (r"(19\d{2}|20\d{2}|21\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", "%Y-%m-%d"),
        # YYYY/mm/dd
        (r"(19\d{2}|20\d{2}|21\d{2})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])", "%Y/%m/%d"),
        # DD-MM-YYYY
        (r"(0[1-9]|[12]\d|3[01])-(0[1-9]|1[0-2])-(19\d{2}|20\d{2}|21\d{2})", "%d-%m-%Y"),
        # DD/MM/YYYY
        (r"(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(19\d{2}|20\d{2}|21\d{2})", "%d/%m/%Y"),
        # YYYYmmddhhmmss (14 digits)
        (r"(19\d{2}|20\d{2}|21\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])([01]\d|2[0-3])([0-5]\d){2}", "%Y%m%d%H%M%S"),
        # YYYYmmddThhmmss (with 'T' separator)
        (r"(19\d{2}|20\d{2}|21\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])T([01]\d|2[0-3])([0-5]\d){2}", "%Y%m%dT%H%M%S"),
    ]

    s = str(filename)
    for pat, fmt in patterns:
            m = re.search(pat, s)
            if not m:
                continue
            raw = m.group(0)
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                # If 14-digit or T pattern fails, try first 8 digits as date
                if fmt in ("%Y%m%d%H%M%S", "%Y%m%dT%H%M%S") and len(raw) >= 8:
                    try:
                        dt = datetime.strptime(raw[:8], "%Y%m%d")
                        return dt.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
                continue
    return None

def sensors(sensors_names: Optional[Iterable[str]] = None) -> str:
    base = [
            # Landsat
            "landsat4", "landsat5", "landsat7", "landsat8", "landsat9",
            "l4", "l5", "l7", "l8", "l9", "lt04", "lt05", "lc08", "lc09", "le07",
            # Sentinel-2 (MSI) # May be some problems with "T32TLR", "T31TGL", targets.
            "sentinel2", "sentinel-2", "s2", "s2a", "s2b", "msi", "T32TLR", "T31TGL",
            # SPOT / Airbus VHR
            "spot4", "spot5", "spot6", "spot7", "sp4", "sp5", "sp6", "sp7", "s4p", "s5p", "s6p", "s7p",
            "pleiades", "pléiades", "pleiades-neo", "pneo", "ple",
            # Planet
            "planetscope", "planet", "psscene", "superdove", "ps2", "ps2sd", "pscope", "planet"
            # Maxar family
            "worldview", "wv1", "wv2", "wv3", "wv4", "geoeye1", "ge1", "quickbird", "ikonos",
            # RapidEye
            "rapideye", "re1", "re2", "re3", "re4", "re5",
            # Aerial / DEM
            "aerial", "uav", "drone", "swissimage", "dem", "hillshade",
        ]
    # s = ""
    # for string in sensors_names:
    #     s += string.lower() + "|"
    #     s += string.upper() + "|"
    # s = s[:-1]
    # return s
    # Merge user-supplied tokens (if any) with defaults
    if sensors_names is None:
        names = list(base)
    elif isinstance(sensors_names, str):
        names = list(base) + [sensors_names]
    else:
        names = list(base) + [str(x) for x in sensors_names if x]
    # Escape regex specials, de-duplicate while preserving order
    seen = set()
    tokens = []
    for tok in names:
        esc = re.escape(tok)
        if esc not in seen:
            tokens.append(esc)
            seen.add(esc)

    # (?i) turns on case-insensitive matching, so no need to duplicate upper/lower
    return "(?i)(?:" + "|".join(tokens) + ")"

def open_gmc_session(target_directory_path: str | Path,
                     epsg: str | int | None = None) -> "Session":
    """Create a new GMC session.

    Args:
        location (str | Path): Path to the directory where the project data will be stored.
        epsg (str | int): EPSG code of the project. If the EPSG code is not provided, it defaults to 4326.

    Returns:
        Session: Returns a Session object.
    """
    if epsg is None:
        epsg = 4326
    #END if

    if not isinstance(target_directory_path, Path):
        target_directory_path = Path(target_directory_path)
    #END if
    return Session(target_directory_path, epsg)

def _gdal_info_json(path: Path) -> dict:
    """Open with GDAL and return the JSON info dict; {} on failure."""
    try:
        ds = gdal.Open(str(path))
        return gdal.Info(ds, format="json") if ds is not None else {}
    except Exception:
        return {}
        
def _first_date_from_metadata(md_flat: dict[str, str]) -> str | None:
    """Return YYYY-MM-DD if any acquisition key exists in metadata (case-insensitive)."""
    # Keys commonly seen across SPOT/Pleiades, PlanetScope, Sentinel-2, Landsat, generic JP2/TIFF
    keys = [
        "imaging_date", "imagedate",
        "acquisition_date", "acquisitiondate", "acqdate",
        "eop:acquisitiondate",
        "product_start_time", "start_time", "sensing_time",
        "date_acquired", "scene_center_time",
    ]
    for k in keys:
        v = md_flat.get(k)
        if not v:
            continue
        s = str(v)
        # Extract a calendar date from strings like '2020-07-14T10:11:12.3Z', '20200714', '2020/07/14'
        m = re.search(r"(19|20|21)\d{2}[-/]?(0[1-9]|1[0-2])[-/]?(0[1-9]|[12]\d|3[01])", s)
        if m:
            y, mth, d = m.group(0), m.group(2), m.group(3)
            # normalize possible 'YYYYmmdd' or 'YYYY/mm/dd'
            if "-" not in m.group(0) and "/" not in m.group(0):
                return f"{y[:4]}-{y[4:6]}-{y[6:8]}"
            return m.group(0).replace("/", "-")
    return None
# * Works properly
def sensor_normalize(text: Optional[str]) -> dict:
    """Normalize sensor names to a canonical form.

    Args:
        text (Optional[str]): The input sensor name.

    Returns:
        dict: A dictionary with normalized sensor information.
    # * Works properly
    """
    if not text:
        return {"sensor": "unknown", "platform": None}

    t = text.lower()

    def out(sensor, *, platform=None, family=None, conf=0.95, matched=None):
        return {"sensor": sensor, "platform": platform, "family": family}
    # --- Sentinel-2 (S2A/S2B or generic)
    m = re.search(r"\b(sentinel[-\s]?2|s2)\s*([ab])?\b|\bsen2*([ab])?\b|\bsent2*([ab])?\b|\b[A-Za-z]{1}\d{2}[A-Za-z]{3}\b|\b[A-Za-z]{1}\d{2}[A-Za-z]{3}\b", t)
    if m:
        var = m.group(2).upper() if m.group(2) else None
        plat = f"S2{var}" if var else None
        return out("Sentinel-2", platform=plat, family="Sentinel-2")

    # --- Landsat (L5/L7/L8/L9 codes; classic names)
    # m = re.search(r"\b(?:landsat\s*(5|7|8|9)|l(5|7|8|9))\b", t)
    m = re.search(r"\b(?:landsat\s*(4|5|7|8|9)|l(4|5|7|8|9)|lc0?(4|5|7|8|9)|le0?(4|5|7|8|9)|lt0?(4|5|7|8|9))\b", t)
    if m:
        num = next(g for g in m.groups() if g)
        return out(f"Landsat-{num}", platform=f"Landsat-{num}", family="Landsat")

    # --- SPOT (4/5/6/7)
    # m = re.search(r"\bspot\s*(4|5|6|7)\b", t)
    m = re.search(r"\bspot\s*(4|5|6|7)\b|\bsp\s*(4|5|6|7)\b|\bs\s*(4|5|6|7)p\b", t)
    if m:
        # num = m.group(1)
        num = next(g for g in m.groups() if g)
        return out(f"SPOT-{num}", platform=f"SPOT-{num}", family="SPOT")

    # --- Pléiades 1A/1B
    m = re.search(r"\b(pl[eé]iades)\s*(1a|1b)\b|\bple\b|\bpl1[ab]\b|\bphr[1-2]a\b|\bphr[1-2]b\b", t)
    if m:
        var = m.group(2).upper()
        return out("Pléiades", platform=f"Pléiades-{var}", family="Pléiades")

    # --- Pléiades Neo (Neo1/Neo2)
    m = re.search(r"\b(pl[eé]iades[-\s]?neo|pneo)\s*(\d+)?\b", t)
    if m:
        var = m.group(2)
        plat = f"Pléiades Neo-{var}" if var else None
        return out("Pléiades Neo", platform=plat, family="Pléiades Neo")

    # --- PlanetScope (PS2, PS2.SD / SuperDove, PSScene)
    m = re.search(r"\b(superdove|ps2\.?sd|psscene|planetscope|ps2|psb|planet)\b", t)
    if m:
        token = m.group(1)
        var = "SuperDove" if token in {"superdove", "ps2.sd", "ps2sd"} else None
        return out("PlanetScope", platform=None, family="PlanetScope")

    # --- RapidEye (RE1..RE5)
    m = re.search(r"\brapideye\b|\bre([1-5])\b", t)
    if m:
        var = m.group(1).upper() if m.group(1) else None
        plat = f"RapidEye-{var}" if var else None
        return out("RapidEye", platform=plat, family="RapidEye")

    # --- WorldView (WV1..WV4)
    m = re.search(r"\b(?:worldview[-\s]?(1|2|3|4)|wv(1|2|3|4))\b", t)
    if m:
        num = next(g for g in m.groups() if g)
        return out(f"WorldView-{num}", platform=f"WorldView-{num}", family="WorldView")

    # --- GeoEye-1
    m = re.search(r"\bgeoeye[-\s]?1\b|\bge1\b", t)
    if m:
        return out("GeoEye-1", platform="GeoEye-1", family="GeoEye")

    # --- QuickBird
    m = re.search(r"\bquickbird\b|\bqb[12]?\b", t)
    if m:
        return out("QuickBird", platform="QuickBird-2", family="QuickBird")

    # --- IKONOS
    m = re.search(r"\bikonos\b", t)
    if m:
        return out("IKONOS", platform="IKONOS", family="IKONOS")

    # --- Aerial / UAV / SwissImage
    if re.search(r"\bswissimage\b", t):
        return out("Aerial", platform="SWISSIMAGE", family="Aerial")#, matched="swissimage")
    if re.search(r"\b(aerial|orthophoto|uav|drone)\b", t):
        return out("Aerial", platform=None, family="Aerial")#, matched="aerial/uav/drone")

    # --- DEM / hillshade
    if re.search(r"\bdem\b|\bhillshade\b", t):
        return out("DEM", platform=None, family="DEM", variant=None)#, matched="dem/hillshade")

    # --- Fallback: try to infer a family token (e.g., “sentinel”, “landsat” without number)
    if "sentinel" in t:
        return out("Sentinel-2", vendor="ESA/Copernicus", family="Sentinel-2",
                conf=0.6, matched="sentinel")
    if "landsat" in t:
        return out("Landsat", vendor="USGS/NASA", family="Landsat",
                conf=0.6, matched="landsat")

    # Unknown
    return out("unknown", conf=0.0, matched=None)

class Session:
    """
    Manipulate data specific to a sample of sites related to an earth surface displacement study.
    """
    # * It works properly
    def __init__(self,
                 target_root_path: str | Path,
                 empty_geodatabase: str | Path = None,
                 epsg: str | int | None = None):
        """Initialize a new Session.

        Args:
            target_root_path (str | Path): The root path for the target data.
            epsg (str | int | None, optional): The EPSG code for the target data. Defaults to None.

        Raises:
            FileNotFoundError: If the target root path does not exist.
            ValueError: If the target root path is not a valid GeoMultiCorr project.
        """

        # pathlib.Path conversion of the string target_root_path
        target_root_path = Path(target_root_path).resolve()

        # If EPSG code is not provided, EPSG:4326 is set by default.
        if epsg is None:
            epsg = 4326

        # Check the address validity
        if not target_root_path.parent.exists():
            msg = f"Invalid path: {target_root_path}"
            print_infoBM(msg)
            raise FileNotFoundError(msg)

        if target_root_path.exists():
            if not any(target_root_path.iterdir()):
                # Folder exists and is empty: create GMC structure
                project_name = target_root_path.name
                print_infoBM(f"New GMC project: '{project_name}'")
                print_infoBM(f"Creating new GMC structure in: {target_root_path}")

                if empty_geodatabase is not None:
                    # Copy GMC-geodatabase template
                    print_infoBM(f"Using empty geodatabase template")
                    raw_empty_gdb_path = project_template_location / "GMC_geodatabase_empty.gpkg"
                    shutil.copy(raw_empty_gdb_path, target_root_path)
                    new_gdb = target_root_path / "GMC_geodatabase_empty.gpkg"
                    new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")
                else:
                    print_infoBM(f"Copying GMC-geodatabase template into {target_root_path}")
                    raw_gdb_path = project_template_location / "GMC_geodatabase_project.gpkg"
                    shutil.copy(raw_gdb_path, target_root_path)
                    new_gdb = target_root_path / "GMC_geodatabase_project.gpkg"
                    new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")

                print_infoBM(f"Setting GMC folder structure")
                # And we create en empty folder "raster_data_{project_name}"
                if not (target_root_path / f"raster_data_{project_name}").exists():
                    print_infoBM(f"Creating raster_data_{project_name} folder")
                    (target_root_path / f"raster_data_{project_name}").mkdir(parents=True)
            elif is_conform_to_gmc_template(target_root_path):
                print_infoBM("GMC project structure found")
                pass
            else:
                msg = (f"Something else exists at {target_root_path}. Please clean it or choose another path.")
                raise ValueError(msg)
        else:
            # Folder does not exist: create and initialize GMC structure
            project_name = target_root_path.name
            print_infoBM(f"New GMC project: '{project_name}'")
            target_root_path.mkdir(parents=True)
            print_infoBM(f"Creating new GMC structure in: {target_root_path}")

            if empty_geodatabase is not None:
                # Copy GMC-geodatabase template
                print_infoBM(f"Using empty geodatabase template")
                raw_empty_gdb_path = project_template_location / "GMC_geodatabase_empty.gpkg"
                shutil.copy(raw_empty_gdb_path, target_root_path)
                new_gdb = target_root_path / "GMC_geodatabase_empty.gpkg"
                new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")
            else:
                print_infoBM(f"Copying GMC-geodatabase template into {target_root_path}")
                raw_gdb_path = project_template_location / "GMC_geodatabase_project.gpkg"
                shutil.copy(raw_gdb_path, target_root_path)
                new_gdb = target_root_path / "GMC_geodatabase_project.gpkg"
                new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")

            # Copy GMC-geodatabase template
            print_infoBM(f"Copying GMC-geodatabase template into {target_root_path}")
            raw_gdb_path = project_template_location / "GMC_geodatabase_project.gpkg"
            shutil.copy(raw_gdb_path, target_root_path)
            new_gdb = target_root_path / "GMC_geodatabase_project.gpkg"
            new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")

            # Copy GMC-Mapset template
            print_infoBM(f"Making a copy of GMC-Mapset template into {target_root_path}")
            raw_mapset_path = project_template_location / "GMC_mapset_project.qgz"
            shutil.copy(raw_mapset_path, target_root_path)
            new_mapset = target_root_path / "GMC_mapset_project.qgz"
            new_mapset.rename(target_root_path / f"GMC_mapset_{project_name}.qgz")

            print_infoBM(f"Setting GMC folder structure")
            # And we create en empty folder "raster_data_{project_name}"
            if not (target_root_path / f"raster_data_{project_name}").exists():
                print_infoBM(f"Creating raster_data_{project_name} folder")
                (target_root_path / f"raster_data_{project_name}").mkdir(parents=True)
        #END if
        # Load project data into the current session
        self.path_root = target_root_path
        self.project_name = target_root_path.name
        self.path_raster_data = target_root_path / f"raster_data_{self.project_name}"
        self.path_img_correlation_outs = target_root_path / f"img_correlation_outs_{self.project_name}"
        self.path_tio_inversion_outs = target_root_path / f"tio_inversion_outs_{self.project_name}"
        self.path_geodb = target_root_path / f"GMC_geodatabase_{self.project_name}.gpkg"
        # os.path.join(target_root_path, f"geodatabase_{self.project_name}.gpkg")
        self.path_qgis_project = target_root_path / f"GMC_mapset_{self.project_name}.qgz"
        self.epsg = epsg

        # There is underscore before the attribute name because user have to access to this data from the getters to ensure that the data is up to date
        self._pzones = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Pzones").to_crs(epsg=epsg)
        self._thumbs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Thumbs").to_crs(epsg=epsg)
        self._geomorphs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Geomorphs").to_crs(epsg=epsg)
        self._pairs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Pairs").to_crs(epsg=epsg)
        self._xzones = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Xzones").to_crs(epsg=epsg)
        self._spines = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Spines").to_crs(epsg=epsg)

        # List the processing zones names
        self.pz_names = list(self._pzones.pz_name.unique())
        print_infoBM(f"Session {self.project_name} initialized.")
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
        # * Works properly
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
            if Path(path).exists():
                return path
        raise FileNotFoundError("QGIS binary not found. Please install QGIS or specify its path.")
    # * Works properly
    def open_qgis_project(self,
                          project_path: str | Path | None = None,
                          qgis_binary_path: str | None = None) -> None:
        """Open the QGIS project

        Args:
            project_path (str | Path | None, optional): Path to the QGIS project file. Defaults to None.
            qgis_binary_path (str | None, optional): Path to the QGIS binary. Defaults to None.

        Raises:
            FileNotFoundError: If the QGIS binary or project file is not found.
        # * Works properly
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
        print_infoBM(f"Opening QGIS project {self.path_qgis_project}.")

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
        # * Works properly
        # TODO: Implement unittests.
        """
        return self._search_engine_b("Thumbs", criterias)
    
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
        # * Works properly
        # TODO: Implement unittests.
        """
        return self._search_engine_b("Pairs", criterias)
    # * Works properly
    def get_pairs(self, criterias: str | list[str] = "") -> list[gmc_pair.Pair]:
        """Get pairs informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a list of pair.Pairs objects meeting the criterias
        # * Works properly
        # TODO: Implement unittests.
        # TODO: solve issue regarding order of return pairs. When session.get_pairs(['ribon', '2014', '2018']), pair 2018-2014 is returned first. 

        """
        # selected_pairs = self.get_pairs_overview(criterias)
        # return [gmc_pair.Pair(self, target_path=x.pa_path) for x in selected_pairs.iloc]
        selected_pairs = self.get_pairs_overview(criterias)
        # return [gmc_pair.Pair(self, target_path=x.pa_path) for x in selected_pairs]
        return [gmc_pair.Pair(self, target_path=row['pa_path']) for _, row in selected_pairs.iterrows()]
    # * Works properly
    def get_pzones_overview(self, pz_name: str | list[str] = "") -> pd.DataFrame:
        """Get pzones overview informations

        Args:
            pz_name (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a dataframe with each pzone
        # * Works properly
        # TODO: Implement unittests.
        """
        return self._search_engine_b("Pzones", pz_name)

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
            print(f"No georasters map named {layername} in the session geodatabase")
            return None
    #------------------ END GETTERS ------------------#
    ###################################################

    ###################################################
    #-------------------- SETTERS --------------------#
    # * Works properly
    def update_vector_data_session(self) -> None:
        """
        Update all vector layers in the geodatabase.

        Returns:
            None. Reload the current session to update files.

        # * Works properly
        # TODO: Implement unittests.
        """
        self._spines = gpd.read_file(self.path_geodb, layer="Spines", engine="pyogrio")
        self._pzones = gpd.read_file(self.path_geodb, layer="Pzones", engine="pyogrio")
        self._thumbs = gpd.read_file(self.path_geodb, layer="Thumbs", engine="pyogrio")
        self._geomorphs = gpd.read_file(self.path_geodb, layer="Geomorphs", engine="pyogrio")
        self._xzones = gpd.read_file(self.path_geodb, layer="Xzones", engine="pyogrio")
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
        opt_root = Path(self.path_raster_data)
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

        # Update pairs
        self.update_pairs()

        return updated
    # * Works properly
    def update_pairs(self) -> gpd.GeoDataFrame:
        """Update pairs in the geodatabase.

        Raises:
            ValueError: If no 'geometry' column is found in pairs DataFrame.

        Returns:
            gpd.GeoDataFrame: pz.pairs with updated pairs.
        
        # * Works properly
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
            print_infoBM(f"Updating pairs for PZone: '{pz.pz_name}'")
            # GeoDataFrame
            pairs = pz.get_pairs_overview() #! add pair strategy
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
    # ! To be implemented in pzone.
    def update_pairs_with_strategy(self,
                                   strategy: str = "consecutive",
                                   fixed_step: int = 3) -> gpd.GeoDataFrame:
        updated_frames = []
        for pz in self.get_pzones():
            print_infoBM(f"Updating pairs for PZone: '{pz.pz_name}'")
            pairs = pz.get_pairs_with_strategy(strategy, fixed_step)         # GeoDataFrame
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

    def copy_geodb(self) -> bool:
        """Quickly create a copy of the project geopackage named backupath_geodb.gpkg
        """
        backup_path = Path(self.path_root, "GMC_geodatabase_backup.gpkg")
        shutil.copytree(self.path_geodb, backup_path)
        # os.system(f"cp -r {self.path_geodb} {backup_path}")
        return Path(self.path_root, "GMC_geodatabase_backup.gpkg").exists()

    ###############################
    # * Works properly
    def map_single_georaster(
            self,
            ta: Path | list[str] | str | None,
            target_crs: str,
            image_types: set[str]
            ) -> dict | None:
        """Map a single georaster to the target CRS and image types.

        Args:
            ta (Path | list[str] | str | None): The input raster file(s) to process.
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
        try:
            r = gu.Raster(ta)
            # --- core attributes
            sensor = re_searcher(str(ta.name), sensors()) if "opt" in image_types else ("dem" if "dem" in image_types else "unknown")
            sensor_normalized = sensor_normalize(sensor)
            ft = {
                "filename": ta.name,
                "sensor": sensor_normalized['sensor'],
                "xRes": r.res[0],
                "yRes": r.res[1],
                "bands": r.count,
                "rows": r.shape[0],
                "cols": r.shape[1],
                "filepath": r.filename,
            }
            name = ta.name
            if "aerial" in sensor_normalized['sensor'].lower():
                ft["acq_date"] = search_date_in_filename(name)
            if "spot" in sensor_normalized['sensor'].lower():
                try:
                    md = gdal.Info(gdal.Open(str(ta))).split("\n")[2].strip()
                    with open(md, "r") as mdf:
                        mds = mdf.read()
                    ft["acq_date"] = re_searcher(re_searcher(mds, "<IMAGING_DATE>.+</IMAGING_DATE>"), "[0-9]+-[0-9]+-[0-9]+")
                except FileNotFoundError:
                    ft["acq_date"] = search_date_in_filename(name)
            if "planet" in sensor_normalized['sensor'].lower():
                try:
                    md = gdal.Info(gdal.Open(str(ta))).split("\n")[2].strip()
                    with open(md, "r") as mdf:
                        mds = mdf.read()
                    ft["acq_date"] = re_searcher(re_searcher(mds, "<eop:acquisitionDate>.+</eop:acquisitionDate>"), "[0-9]+-[0-9]+-[0-9]+")
                except FileNotFoundError:
                    ft["acq_date"] = search_date_in_filename(name)
            if "landsat" in sensor_normalized['sensor'].lower():
                try:
                    md = gdal.Info(gdal.Open(str(ta))).split("\n")[2].strip()
                    with open(md, "r") as mdf:
                        mds = mdf.read()
                    ft["acq_date"] = re_searcher(re_searcher(mds, "<IMAGING_DATE>.+</IMAGING_DATE>"), "[0-9]+-[0-9]+-[0-9]+")
                except FileNotFoundError:
                    ft["acq_date"] = search_date_in_filename(name)
            if "sentinel" in sensor_normalized['sensor'].lower():
                try:
                    md = gdal.Info(gdal.Open(str(ta))).split("\n")[2].strip()
                    with open(md, "r") as mdf:
                        mds = mdf.read()
                    ft["acq_date"] = re_searcher(re_searcher(mds, "<SENSING_TIME>.+</SENSING_TIME>"), "[0-9]+-[0-9]+-[0-9]+")
                except FileNotFoundError:
                    ft["acq_date"] = search_date_in_filename(name)
            if sensor_normalized['sensor'] == "dem" or "swissimage" in name.lower():
                ft["acq_date"] = ft.get("acq_date") or search_date_in_filename(name)
                if "swissimage" in name.lower():
                    ft["sensor"] = "aerial"
            
            # footprint + CRS
            fp = r.footprint
            if hasattr(fp, "ds"):
                gdf_fp = fp.ds
                src_crs = gdf_fp.crs
                shp = gdf_fp.geometry.union_all()
            else:
                shp = fp
                src_crs = getattr(r, "crs", None)

            ft["src_crs"] = (src_crs.to_string() if hasattr(src_crs, "to_string") else str(src_crs)) if src_crs else None

            geom_proj = shp
            try:
                if target_crs and src_crs and str(src_crs) != target_crs:
                    geom_proj = gpd.GeoSeries([shp], crs=src_crs).to_crs(target_crs).iloc[0]
            except Exception:
                pass

            ft["geometry"] = geom_proj  # serialize for cross-process return
            return ft
        except Exception as e:
            print(f"Skip {ta}: {e}")
            return None
    # * Works properly
    def map_georasters_bank(self,
                            georasters_bank_path: str | Path,
                            extensions: str | Sequence[str] = None,
                            image_type: str = "opt",
                            epsg: str | int | None = None,
                            suffix: str = "",
                            max_workers: int = None) -> gpd.GeoDataFrame:
        """Map georasters in the specified bank directory using parallel processing.

        Args:
            georasters_bank_path (str | Path): Path to the georaster bank directory.
            extensions (str | Sequence[str], optional): File extensions to include. Defaults to None.
            image_type (str, optional): Type of image to process. Defaults to "opt".
            epsg (str | int | None, optional): EPSG code for the target CRS. Defaults to None.
            suffix (str, optional): Suffix to append to output files. Defaults to "".
            max_workers (int, optional): Maximum number of workers for parallel processing. Defaults to None.

        Raises:
            FileNotFoundError: If the georaster bank path does not exist.
            RuntimeError: If no features are collected.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the mapped georasters.

        # * Works properly
        # TODO: Implement unittests.
        """
        # Validate path
        georasters_bank_path = Path(georasters_bank_path)
        if not georasters_bank_path.exists():
            raise FileNotFoundError(f"{georasters_bank_path} is not an existing path")
        
        if extensions is None:
            extensions = ["tif", "jp2"]

        # Normalize extensions (case-insensitive)
        targets = []
        for x in (extensions[0] if isinstance(extensions, tuple) else extensions):
                targets += georasters_bank_path.glob(f"**/*.{x.lower()}")
                targets += georasters_bank_path.glob(f"**/*.{x.upper()}")

        # Normalize image_type(s) to a lowercase set for robust checks
        _image_types = {image_type.lower()} if isinstance(image_type, str) else {str(t).lower() for t in image_type}
        # Choose a project/target CRS (prefer your session's EPSG)
        target_crs = f"EPSG:{epsg}" if hasattr(epsg, "epsg") else "EPSG:2154"
        
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.map_single_georaster, ta, target_crs, _image_types) for ta in targets]
            rows = []
            with alive_bar(len(futures), title="Mapping georasters", force_tty=True) as bar:
                for _ in as_completed(futures):
                    rec = _.result()
                    if rec:
                        rows.append(rec)
                    bar()
        
        if not rows:
            raise RuntimeError("No features collected.")

        # Rebuild geometries and GeoDataFrame
        df = pd.DataFrame(rows)
        layer_crs = target_crs or (df["src_crs"].dropna().iloc[0] if not df["src_crs"].dropna().empty else None)
        features_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=layer_crs)
        features_gdf.to_file(self.path_geodb, layer=f"Protomap")
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
    def draw_polygon_manually(self,
                              center: tuple = (46.8, 8.2),
                              zoom: int = 6,
                              default_crs: str = 'EPSG:4326'):
        """
        Launch an interactive map for drawing polygons. Returns a GeoDataFrame of drawn polygons.
        The drawn polygons are always returned in the main session CRS (self.epsg) if set.
        Usage:
            gdf = session.draw_polygon_manually()
        Then use: session.insert_pzone(gdf.geometry.iloc[0], ...)

        # * Works properly
        # TODO: Implement unittests.
        # TODO: Test on other machines.
        # TODO: Current coordinates fits for France. Solve it after depending on the country.
        """
        m = Map(center=center, zoom=zoom)
        draw_control = DrawControl(polyline={}, circlemarker={}, marker={}, circle={})
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
                # Normalize to session CRS if needed
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
    # * Works properly
    def sieve_single_mosaic(
                            self,
                            pzone_vector: gu.Vector | gpd.GeoDataFrame | gpd.GeoSeries | gu.Raster | None,
                            mosaic_path: Path | str | None,
                            raster_resolution: int | float | str,
                            resampling_method: str | list[str],
                            selected_band: int | list[int],
                            crop_mode: str | list[str] | None,
                            n_threads: int,
                            memory_limit: int
                        ) -> gu.Raster:
        """Process a single mosaic using geoutils

        Args:
            pzone_vector (gu.Vector | gpd.GeoDataFrame | gpd.GeoSeries | gu.Raster | None): The processing zone vector.
            mosaic_path (Path | str | None): The path to the mosaic file.
            raster_resolution (int | float | str): The desired raster resolution.
            resampling_method (str | list[str]): The resampling method to use. Options include gdal methods 'nearest', 'bilinear', 'cubic', etc.
            selected_band (int | list[int]): The band(s) to process.
            crop_mode (str | list[str] | None): The crop mode to use. Options include 'match_pixel', 'match_extent'
            n_threads (int): The number of threads to use for processing.
            memory_limit (int): The memory limit for processing.

        Returns:
            gu.Raster: The processed mosaic.
        
        # * Works properly
        # TODO: Implement unittests.
        # ! TODO: Problems with crop method. Ask geoutils team.
        """
        mosaic = gu.Raster(mosaic_path, bands=selected_band)
        mosaic.set_nodata(0)
        if (mosaic.crs.to_epsg() == self.epsg) and (mosaic.res[0] == raster_resolution):
            m_crop = mosaic.crop(pzone_vector, mode=crop_mode)
        elif (mosaic.crs.to_epsg() != self.epsg) and (mosaic.res[0] == raster_resolution):
            m_proj = mosaic.reproject(crs=f"EPSG:{self.epsg}", resampling=resampling_method, n_threads=n_threads, memory_limit=memory_limit)
            m_crop = m_proj.crop(pzone_vector, mode=crop_mode)
        elif (mosaic.crs.to_epsg() == self.epsg) and (mosaic.res[0] != raster_resolution):
            m_proj = mosaic.reproject(res=raster_resolution, resampling=resampling_method, n_threads=n_threads, memory_limit=memory_limit)
            m_crop = m_proj.crop(pzone_vector, mode=crop_mode)
        else:
            m_proj = mosaic.reproject(res=raster_resolution, crs=f"EPSG:{self.epsg}", resampling=resampling_method, n_threads=n_threads, memory_limit=memory_limit)
            m_crop = m_proj.crop(pzone_vector, mode=crop_mode)
        del mosaic
        if m_proj is not None:
            del m_proj
        return m_crop
    # * Works properly
    def sieve(self,
            georasters_map: gpd.GeoDataFrame | None = None,
            image_type: str | list[str] = "opt",
            suffix: str | list[str] = "",
            raster_resolution: int | float | str = 1.5,
            crop_mode: str | list[str] | None = "match_pixel",
            resampling_method: str | list[str] = "nearest",
            selected_band: int | list[int] = [1],
            max_workers: int = 12,
            n_threads: int = 10,
            memory_limit: int = 1024) -> None:
        """
        Intersect a georasters map layer and the project pzones layer to create cropped and merged raster "thumbs" for each processing zone.

        For each processing zone (pzone), this function:
        - Finds all rasters in the georasters map that intersect the pzone geometry.
        - Groups rasters by acquisition date and sensor.
        - Crops and merges (if needed) the rasters for each group to the pzone extent, using the specified resolution, crop mode, and resampling method.
        - Saves the resulting "thumb" rasters in the appropriate output directory for each pzone.

        Args:
            georasters_map (gpd.GeoDataFrame | None, optional): The georasters map to use. If None, it is loaded from the session.
            image_type (str | list[str], optional): The type(s) of images to process (e.g., "opt" or "dem"). Defaults to "opt".
            suffix (str | list[str], optional): Suffix(es) to append to output filenames. Defaults to "".
            raster_resolution (int | float | str, optional): Desired output raster resolution. Defaults to 1.5.
            crop_mode (str | list[str] | None, optional): Cropping mode to use (e.g., "match_pixel" or "match_extent"). Defaults to "match_pixel".
            resampling_method (str | list[str], optional): Resampling method(s) to use (e.g., "nearest", "bilinear"). Defaults to "nearest".
            selected_band (int | list[int], optional): Band(s) to process. Defaults to [1].
            max_workers (int, optional): Maximum number of parallel workers for cropping/merging. Defaults to 12.
            n_threads (int, optional): Number of threads for raster processing. Defaults to 10.
            memory_limit (int, optional): Memory limit (in MB) for raster processing. Defaults to 1024.

        Returns:
            None
        # * Works properly
        # TODO: Implement unittests.
        # ! TODO: Problems with crop method. Ask geoutils team.
        # ! TODO: merge raster not tested.
        """
        # Check if user have drawn some pzones
        assert (len(self.get_pzones_overview()) > 0), "no pzone registered for this project"

        # Open the corresponding georasters map
        if georasters_map is None:
            georasters_map = self.get_georasters_map(image_type, suffix)

        # Inform user about CRS differences between georasters map and pzones layer
        print_infoBM(f"Map epsg : {georasters_map.crs}\nPzone  epsg : {self.get_pzones_overview().crs}")

        # For each processing zone
        for pz in self.get_pzones_overview().iloc:
            print_infoBM(f"Creating data for Pzone: {pz['pz_name']}", bold=True)
            print_infoBM(f"Geometry extent: {pz['geometry']}")
            print_infoBM(f"Raster resolution: {raster_resolution}")

            pz_vec = gu.Vector(gpd.GeoDataFrame([pz], crs=self.get_pzones_overview().crs, geometry="geometry"))

            # Create a directory to store the raster_data of the pzone
            pz_rasterdata = Path(self.path_root, f"raster_data_{self.project_name}", pz['pz_name'])

            # Create subdirectories for the thumbs and the displacements measurements associated
            if image_type == "opt":
                pz_opticals = Path(pz_rasterdata, "opticals")
                pz_disps = Path(pz_rasterdata, "image_correlation")
                for p in [pz_rasterdata, pz_opticals, pz_disps]:
                    if not p.exists():
                        print_infoBM(f"Creating directory {p}")
                        p.mkdir(parents=True)

            # Get the images intersecting the zone
            selection = georasters_map[georasters_map["geometry"].intersects(pz["geometry"])]

            """
            Here we have to regroup the lines of 'selection' where the acquisition date and the sensor are identicals. Then we loop on thoses groups and we merge the part of each group
            """

            # Group by acquisition date and sensor
            selection_merged_by_date_and_sensor = selection.groupby(["acq_date", "sensor"])["filepath"].apply(list)
            # For each group
            for group_id, group in enumerate(selection_merged_by_date_and_sensor):

                acq_date = selection_merged_by_date_and_sensor.index[group_id][0]
                sensor = selection_merged_by_date_and_sensor.index[group_id][1]

                # Define output filename and full path (ex: raster-data_vanoise/sachette/opticals/sachette_2021-03-08_AERIAL.tif)
                if image_type == "opt":
                    thumb_name = f"{pz['pz_name']}_{acq_date}_{sensor}_{raster_resolution}m.tif"
                    thumb_path = str(Path(pz_opticals, thumb_name))

                elif image_type == "dem":
                    thumb_name = f"{pz['pz_name']}_{raster_resolution}m_dem.tif"
                    thumb_path = str(Path(pz_rasterdata, thumb_name))

                if Path(thumb_path).exists():
                    pass

                fully = []
                with alive_bar(len(group), title="Cropping raster data", force_tty=True) as bar:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(
                            self.sieve_single_mosaic,
                            pz_vec,
                            mosaic_path,
                            raster_resolution,
                            resampling_method,
                            selected_band,
                            crop_mode,
                            n_threads,
                            memory_limit
                            )
                            for mosaic_path in group
                        ]
                        results = [f.result() for f in as_completed(futures)]
                        fully.extend(results)
                        bar()
                    #END with
                #END with
                if len(fully) > 1:
                    # Merge the fully cropped rasters
                    thumb = gu.raster.multiraster.merge_rasters(fully)
                else:
                    thumb = fully[0]
                #END if
                thumb.save(thumb_path, dtype="uint16")
            #END for
        #END for
        return None

    def sieve_thibaut(self,
              georasters_map: gpd.GeoDataFrame | None = None,
              image_type: str | list[str] = "opt",
              suffix: str | list[str] = "",
              image_resolution: int | float | str = 1,
              crop_mode: str | list[str] | None = "match_extent",
              resampling_method: str | list[str] = "nearest",
              selected_band: int | list[int] = 1):
        """Intersect a georasters map layer and the project pzones layer to create thumbs"""
        #! TODO: remove this function after ask Thibaut.

        # Check if user have drawn some pzones
        assert (len(self.get_pzones_overview()) > 0), "no pzone registered for this project"

        # Open the corresponding georasters map
        if georasters_map is None:
            georasters_map = self.get_georasters_map(image_type, suffix)

        # Inform user about CRS differences between georasters map and pzones layer
        print_infoBM(f"Map epsg : {georasters_map.crs}\nPzone  epsg : {self.get_pzones_overview().crs}")

        # For each processing zone
        for pz in self.get_pzones_overview().iloc:
            print_infoBM(f"Working on {pz['pz_name']}")

            # Create a directory to store the raster_data of the pzone
            pz_rasterdata = Path(self.path_root, f"raster_data_{self.project_name}", pz["pz_name"])

            # Create subdirectories for the thumbs and the displacements measurements associated
            if image_type == "opt":
                pz_opticals = Path(pz_rasterdata, "opticals")
                pz_disps = Path(pz_rasterdata, "image_correlation")
                for p in [pz_rasterdata, pz_opticals, pz_disps]:
                    if not p.exists():
                        p.mkdir(parents=True)

            # Get the images intersecting the zone
            selection = georasters_map[georasters_map["geometry"].intersects(pz["geometry"])]

            """
            Here we have to regroup the lines of 'selection' where the acquisition date and the sensor are identicals
            Then we loop on thoses groups
            And we merge the part of each group
            """

            # Group by acquisition date and sensor
            selection_merged_by_date_and_sensor = selection.groupby(["acq_date", "sensor"])["filepath"].apply(list)

            # For each group
            for group_id, group in enumerate(selection_merged_by_date_and_sensor):

                acq_date = selection_merged_by_date_and_sensor.index[group_id][0]
                sensor = selection_merged_by_date_and_sensor.index[group_id][1]

                # Define output filename and full path (ex: raster-data_vanoise/sachette/opticals/sachette_2021-03-08_AERIAL.tif)
                if image_type == "opt":
                    thumb_name = f"{pz['pz_name']}_{acq_date}_{sensor}.tif"
                    thumb_path = str(Path(pz_opticals, thumb_name))

                elif image_type == "dem":
                    thumb_name = f"{pz['pz_name']}_dem.tif"
                    thumb_path = str(Path(pz_rasterdata, thumb_name))

                if Path(thumb_path).exists():
                    continue

                # We check if the file is already existing
                # if not os.path.exists(thumb_path):

                # We crop each image of the group on the p_zone
                # Here, each image receive a number considered as nodata value by gdal
                # where the pixels are inside the p_zone but outside the image
                fully = []
                print_infoBM("Cropping raster data")
                for mosaic_path in tqdm.tqdm(group):
                    if image_type == "opt":
                        mosaic = gu.Raster(mosaic_path, bands=selected_band)
                        # If CRS do not match, reproject already to target CRS and desired pixel size
                        if mosaic.crs.to_epsg() != self.epsg:
                            mosaic = mosaic.reproject(res=1.0, dtype="uint16", crs=f"EPSG:{self.epsg}", resampling=resampling_method)
                            mosaic_crop = mosaic.crop(pz, nodata=0, mode=crop_mode)
                        else:
                            mosaic = mosaic.reproject(res=1.0, dtype="uint16", crs=f"EPSG:{self.epsg}", resampling=resampling_method)
                            mosaic_crop = mosaic.crop(pz, nodata=0, mode=crop_mode)

                    # elif image_type == "dem":
                    #     mosaic = gu.Raster(mosaic_path)
                    #     if mosaic.res
                    #     if rt.getPixelSize(mosaic_path)[0] == image_resolution:
                    #         mosaic = rt.Open(
                    #             target=mosaic_path, geoExtent=pz.geometry.bounds
                    #         )

                    #     else:
                    #         mosaic = rt.Open(
                    #             target=mosaic_path,
                    #             geoExtent=pz.geometry.bounds,
                    #             nRes=res,
                    #             resMethod=alg,
                    #         )

                    fully.append(mosaic)

                # If there is more than one image acquired at the same date and from the same sensor
                # we accept that it is the initial same image and we merged them together
                # because gdal considered as nodata the places outside each part, the merged work easily
                """
                TELENVI ISSUE 1
                ############### but it's not working if we don't extract a band with rt.Open()
                """

                # if len(fully) > 1:
                #     print("assemblage")
                #     thumb = rt.merge(fully)
                # else:
                #     thumb = fully[0]

                # # Write the thumb
                # rt.write(thumb, thumb_path)
    
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
    # Thibauld function
    # def sieve(self,
    #           georasters_map: gpd.GeoDataFrame | None = None,
    #           image_type: str | list[str] = "opt",
    #           suffix: str | list[str] = "",
    #           res: int | float | str = 1,
    #           alg: str | list[str] = "nearest",
    #           band: int = 1):
    #     """Intersect a georasters map layer and the project pzones layer to create thumbs"""

    #     # Check if user have drawn some pzones
    #     assert (len(self.get_pzones_overview()) > 0), "no pzone registered for this project"

    #     # Open the corresponding georasters map
    #     if georasters_map is None:
    #         georasters_map = self.get_georasters_map(image_type, suffix)

    #     # Inform user about CRS differences between georasters map and pzones layer
    #     print_infoBM(f"Map epsg : {georasters_map.crs}\nPzone  epsg : {self.get_pzones_overview().crs}")

    #     # For each processing zone
    #     for pz in self.get_pzones_overview().iloc:
    #         print_infoBM(f"Working on {pz['pz_name']}")

    #         # Create a directory to store the raster_data of the pzone
    #         pz_rasterdata = Path(self.path_root, f"raster_data_{self.project_name}", pz["pz_name"])

    #         # Create subdirectories for the thumbs and the displacements measurements associated
    #         if image_type == "opt":
    #             pz_opticals = Path(pz_rasterdata, "opticals")
    #             pz_disps = Path(pz_rasterdata, "image_correlation")
    #             for p in [pz_rasterdata, pz_opticals, pz_disps]:
    #                 if not p.exists():
    #                     p.mkdir(parents=True)

    #         # Get the images intersecting the zone
    #         selection = georasters_map[georasters_map["geometry"].intersects(pz["geometry"])]

    #         """
    #         Here we have to regroup the lines of 'selection' where the acquisition date and the sensor are identicals
    #         Then we loop on thoses groups
    #         And we merge the part of each group
    #         """

    #         # Group by acquisition date and sensor
    #         selection_merged_by_date_and_sensor = selection.groupby(["acq_date", "sensor"])["filepath"].apply(list)

    #         # For each group
    #         for group_id, group in enumerate(selection_merged_by_date_and_sensor):

    #             acq_date = selection_merged_by_date_and_sensor.index[group_id][0]
    #             sensor = selection_merged_by_date_and_sensor.index[group_id][1]

    #             # Define output filename and full path (ex: raster-data_vanoise/sachette/opticals/sachette_2021-03-08_AERIAL.tif)
    #             if image_type == "opt":
    #                 thumb_name = f"{pz['pz_name']}_{acq_date}_{sensor}.tif"
    #                 thumb_path = str(Path(pz_opticals, thumb_name))

    #             elif image_type == "dem":
    #                 thumb_name = f"{pz['pz_name']}_dem.tif"
    #                 thumb_path = str(Path(pz_rasterdata, thumb_name))

    #             if Path(thumb_path).exists():
    #                 continue

    #             # We check if the file is already existing
    #             # if not os.path.exists(thumb_path):

    #             # We crop each image of the group on the p_zone
    #             # Here, each image receive a number considered as nodata value by gdal
    #             # where the pixels are inside the p_zone but outside the image
    #             fully = []
    #             print_infoBM("Cropping raster data")
    #             for mosaic_path in tqdm.tqdm(group):

    #                 # Difference between opt and dem because of bugs with merge, when the input raster have 1 or many bands.
    #                 # But if you work with an optical raster containing only one band, bugs can happened... ?
    #                 if image_type == "opt":
    #                     # If we don't have to make a resampling
    #                     if rt.getPixelSize(mosaic_path)[0] == res:
    #                         mosaic = rt.Open(
    #                             target=mosaic_path,
    #                             geoExtent=pz.geometry.bounds,
    #                             nBands=1,)

    #                     else:
    #                         mosaic = rt.Open(
    #                             target=mosaic_path,
    #                             geoExtent=pz.geometry.bounds,
    #                             nRes=res,
    #                             resMethod=alg,
    #                             nBands=1,
    #                         )

    #                 elif image_type == "dem":
    #                     if rt.getPixelSize(mosaic_path)[0] == res:
    #                         mosaic = rt.Open(
    #                             target=mosaic_path, geoExtent=pz.geometry.bounds
    #                         )

    #                     else:
    #                         mosaic = rt.Open(
    #                             target=mosaic_path,
    #                             geoExtent=pz.geometry.bounds,
    #                             nRes=res,
    #                             resMethod=alg,
    #                         )

    #                 fully.append(mosaic)

    #             # If there is more than one image acquired at the same date and from the same sensor
    #             # we accept that it is the initial same image and we merged them together
    #             # because gdal considered as nodata the places outside each part, the merged work easily
    #             """
    #             TELENVI ISSUE 1
    #             ############### but it's not working if we don't extract a band with rt.Open()
    #             """

    #             if len(fully) > 1:
    #                 print("assemblage")
    #                 thumb = rt.merge(fully)
    #             else:
    #                 thumb = fully[0]

    #             # Write the thumb
    #             rt.write(thumb, thumb_path)

# def map_georasters_bank(self,
#                         georasters_bank_path: str | Path,
#                         extensions: str | list[str] = ["tif", "jp2"],
#                         image_type: str = "opt",
#                         epsg: str | int | None = None,
#                         suffix: str = ""):
#     #! this function has become useless.
#     # check the validity of the georasters_bank_path
#     georasters_bank_path = Path(georasters_bank_path)
#     assert (georasters_bank_path.exists()), f"{georasters_bank_path} is not an existing path"

#     # Get a list of all the .tif or .jp2 under georasters_bank_path
#     targets = []
#     for x in extensions:
#         targets += georasters_bank_path.glob(f"**/*.{x.lower()}")
#         targets += georasters_bank_path.glob(f"**/*.{x.upper()}")

#     # Build a vector layer, with metadata and extent of each image
#     features = []
#     print_infoBM(f"Mapping the images in {georasters_bank_path}...")
#     for ta in tqdm.tqdm(targets):
#         r = gu.Raster(ta)
#         # Create empty serie
#         ft = gpd.GeoSeries()
#         # Write all kind of metadata
#         ft["filename"] = (ta.name).split('/')[-1]
#         if image_type == "opt":
#             ft["sensor"] = re_searcher(str(ta), sensors())
#         elif "dem" in image_type.lower():
#             ft["sensor"] = "dem"
#         #END if
#         # ft["xRes"], ft["yRes"] = rt.getPixelSize(str(ta)) #! to delete after testing
#         ft["xRes"], ft["yRes"] = r.res
#         # ft["bands"], ft["rows"], ft["cols"] = rt.getShape(str(ta)) #! to delete after testing
#         ft["bands"] = r.count
#         ft["rows"], ft["cols"] = r.shape

#         ###### ###### ###### ######
#         ###### Modulable code - you have to adapt this block according to the way of naming of your rasters in the rasters_bank
#         ###### ###### ###### ######

#         # Get acquisition date : Diego PhD thesis harddisk naming way
#         if "aerial" in ft["sensor"].lower():
#             # We split the filename to get the date -- according to UNILgis naming way
#             # acq_date = ft["filename"].split("-")[1]
#             # acq_date = f"{acq_date[:4]}-{acq_date[4:6]}-{acq_date[6:]}"
#             ft["acq_date"] = search_date_in_filename(ft["filename"])
#         #END if

#         # Get acquisition date : SPOT naming way
#         if "spot" in ft["sensor"].lower():
#             # Here we get the path to the metadata.xml file associated to the image
#             md_file_path = gdal.Info(gdal.Open(str(ta))).split("\n")[2]

#             # Oppening the metadata file
#             try:
#                 with open(md_file_path.strip(), "r") as mdf:
#                     mds = mdf.read()

#                 # Here we track the 'imaging_date' tag and extract his value
#                 imaging_date_line = re_searcher(mds, "<IMAGING_DATE>.+</IMAGING_DATE>")
#                 ft["acq_date"] = re_searcher(imaging_date_line, "[0-9]+-[0-9]+-[0-9]+")

#             except FileNotFoundError:
#                 print_infoBM("Metadata file no found. Getting metadata from the image name by default")
#                 ft['acq_date'] = search_date_in_filename(ft["filename"])
#                 continue
#             #END try
#         #END if

#         # # Get acquisition date : PlanetScope naming way
#         #TODO: check if th metadata path is correct & check if the date is correct.
#         if 'planet' in ft['sensor'].lower():
#             # Here we get the path to the metadata.xml file associated to the image
#             md_planet_file_path = gdal.Info(gdal.Open(str(ta))).split("\n")[2]

#             # Oppening the metadata file
#             try:
#                 with open(md_planet_file_path.strip(), "r") as mdf:
#                     mds = mdf.read()
                
#                 # Here we track the 'imaging_date' tag and extract his value
#                 imaging_date_line = re_searcher(mds, "<eop:acquisitionDate>.+</eop:acquisitionDate>")
#                 ft['acq_date'] = re_searcher(imaging_date_line, "[0-9]+-[0-9]+-[0-9]+")
            
#             except FileNotFoundError:
#                 print_infoBM("Metadata file no found. Getting metadata from the image name by default")
#                 ft['acq_date'] = search_date_in_filename(ft["filename"])
#                 continue

#         # Get acquisition date : swiss alti 3D DEM
#         if ft["sensor"] == "dem":
#             ft["acq_date"] = ft["filename"].split("_")[1]

#         # Get acquisition date : swissimage format
#         if "swissimage" in ta.name:
#             ft["acq_date"] = f"{ft['filename'].split('_')[1]}-01-01"
#             ft["sensor"] = "aerial"

#         ###### ###### ###### #######
#         ###### End of modulable code ------------------------------
#         ###### ###### ###### #######

#         # Get the image geographic extent
#         # ft["geometry"] = rt.drawGeomExtent(str(ta), geomType="shly") #! to delete after testing
#         ft["geometry"] = r.footprint
#         ft["filepath"] = str(ta)

#         # Add the feature to the future layer (just a list for now)
#         features.append(ft)

#     # Transform the list of GeoSeries features into a GeoDataFrame, and set his CRS
#     layer = gpd.GeoDataFrame(features, geometry="geometry", crs=epsg)

#     # Save the map in the geodatabase with dynamic name
#     # self.copy_geodb()
#     # layer.to_file(self.path_geodb, layer=f"extents-map_{image_type}_{suffix}")
#     layer.to_file(self.path_geodb, layer=f"Georaster_bank_{image_type}_{suffix}")
#     return layer