#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# supported_sensors.py
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
Utilities for normalizing and extracting information from satellite sensor names.
"""
import os
import re

import pathlib

from osgeo import gdal
from datetime import datetime
from typing import Iterable, Optional, Sequence

# 1. Catalog of known sensors and their patterns
SENSOR_CATALOG = [
    # Landsat
    {"patterns": [r"landsat\s*([4-9])", r"\bl([4-9])\b"], "sensor": "LANDSAT{num}", "family": "landsat"},
    # Sentinel-2
    {"patterns": [r"sentinel[-\s]?2[ab]?", r"\bs2[ab]?\b"], "sensor": "SENTINEL2", "family": "sentinel"},
    # SPOT
    {"patterns": [r"spot\s*([4-7])", r"\bsp([4-7])\b"], "sensor": "SPOT{num}", "family": "spot"},
    # Pléiades
    {"patterns": [r"pl[eé]iades\s*(1a|1b)?", r"pleiades-neo", r"pneo"], "sensor": "PLEIADES", "family": "pleiades"},
    # PlanetScope
    {"patterns": [r"planetscope", r"psscene", r"superdove", r"ps2(sd)?", r"pscope", r"analyticms"], "sensor": "PLANETSCOPE", "family": "planetscope"},
    # WorldView
    {"patterns": [r"worldview[-\s]?([1-4])", r"\bwv([1-4])\b"], "sensor": "WORLDVIEW{num}", "family": "worldview"},
    # GeoEye
    {"patterns": [r"geoeye[-\s]?1", r"\bge1\b"], "sensor": "GEOEYE1", "family": "geoeye"},
    # QuickBird
    {"patterns": [r"quickbird", r"\bqb[12]?\b"], "sensor": "QUICKBIRD", "family": "quickbird"},
    # IKONOS
    {"patterns": [r"ikonos"], "sensor": "IKONOS", "family": "ikonos"},
    # RapidEye
    {"patterns": [r"rapideye", r"\bre([1-5])\b"], "sensor": "RAPIDEYE{num}", "family": "rapideye"},
    # Aerial/UAV
    {"patterns": [r"swissimage", r"aerial", r"uav", r"drone", r"orthophoto"], "sensor": "AERIAL", "family": "aerial"},
    # DEM
    {"patterns": [r"dem", r"hillshade"], "sensor": "DEM", "family": "dem"},
]

AUXILIARY_FILE_PATTERNS: dict[str, list[str]] = {
    "planetscope": ["_udm2_", "_udm_", "_confidence_"],
    "sentinel":    ["_TCI_", "_SCL_", "_WVP_", "_AOT_"],
    "landsat":     ["_QA_PIXEL_", "_QA_RADSAT_", "_ST_QA_"],
    "worldview":   [],
    "spot":        [],
    "pleiades":    [],
}

AUXILIARY_FILE_EXCLUDE_PATTERNS: list[str] = [
    pat for pats in AUXILIARY_FILE_PATTERNS.values() for pat in pats
]

supported_sensors = [
        # Landsat
        "landsat4", "landsat5", "landsat7", "landsat8", "landsat9",
        "l4", "l5", "l7", "l8", "l9", "lt04", "lt05", "lc08", "lc09", "le07",
        # Sentinel-2 (MSI) # May be some problems with "T32TLR", "T31TGL", targets.
        "sentinel2", "sentinel-2", "s2", "s2a", "s2b", "msi", "T32TLR", "T31TGL",
        # SPOT / Airbus VHR
        "spot4", "spot5", "spot6", "spot7", "sp4", "sp5", "sp6", "sp7", "s4p", "s5p", "s6p", "s7p",
        "pleiades", "pléiades", "pleiades-neo", "pneo", "ple",
        # Planet
        "planetscope", "planet", "psscene", "superdove", "ps2", "ps2sd", "pscope", "analyticms",
        # Maxar family
        "worldview", "wv1", "wv2", "wv3", "wv4", "geoeye1", "ge1", "quickbird", "ikonos",
        # RapidEye
        "rapideye", "re1", "re2", "re3", "re4", "re5",
        # Aerial / DEM
        "aerial", "uav", "drone", "swissimage", "dem", "hillshade",
    ]

# Sensor-family → XML tag pattern mapping for acquisition date extraction
_XML_DATE_TAGS: dict[str, str] = {
    "spot":        r"<IMAGING_DATE>\s*([^<]+)\s*</IMAGING_DATE>",
    "pleiades":    r"<IMAGING_DATE>\s*([^<]+)\s*</IMAGING_DATE>",
    "planetscope": r"<eop:acquisitionDate>\s*([^<]+)\s*</eop:acquisitionDate>",
    "planet":      r"<eop:acquisitionDate>\s*([^<]+)\s*</eop:acquisitionDate>",
    "landsat":     r"<IMAGING_DATE>\s*([^<]+)\s*</IMAGING_DATE>",
    "sentinel":    r"<SENSING_TIME>\s*([^<]+)\s*</SENSING_TIME>",
    "worldview":   r"<FIRSTLINETIME>\s*([^<]+)\s*</FIRSTLINETIME>",
    "geoeye":      r"<FIRSTLINETIME>\s*([^<]+)\s*</FIRSTLINETIME>",
}

# Applied when sensor family is unknown or the sensor-specific tag is not found
_GENERIC_XML_DATE_PATTERNS: list[str] = [
    r"<(?:IMAGING_DATE|imaging_date)>\s*([^<]+)\s*</",
    r"<(?:SENSING_TIME|sensing_time)>\s*([^<]+)\s*</",
    r"<(?:eop:acquisitionDate)>\s*([^<]+)\s*</",
    r"<(?:FIRSTLINETIME|firstlinetime)>\s*([^<]+)\s*</",
    r"<(?:DATE_ACQUIRED|date_acquired)>\s*([^<]+)\s*</",
    r"<(?:ACQUISITION_DATE|acquisition_date)>\s*([^<]+)\s*</",
]

# Sensor-family → XML tag pattern mapping for acquisition *time* extraction.
# Only needed for sensors whose date and time are stored in separate XML tags.
# Sensors whose date tag already embeds the time (e.g. ISO "2021-07-14T10:30:45Z")
# are handled automatically without a separate time tag.
_XML_TIME_TAGS: dict[str, str] = {
    "spot":     r"<IMAGING_TIME>\s*([^<]+)\s*</IMAGING_TIME>",
    "pleiades": r"<IMAGING_TIME>\s*([^<]+)\s*</IMAGING_TIME>",
    "landsat":  r"<SCENE_CENTER_TIME>\s*([^<]+)\s*</SCENE_CENTER_TIME>",
}

# Generic time-tag fallbacks tried when sensor family is unknown or has no specific entry
_GENERIC_XML_TIME_PATTERNS: list[str] = [
    r"<(?:IMAGING_TIME|imaging_time)>\s*([^<]+)\s*</",
    r"<(?:SCENE_CENTER_TIME|scene_center_time)>\s*([^<]+)\s*</",
    r"<(?:TIME|time)>\s*([^<]+)\s*</",
]

def _gdal_info_json(path: pathlib.Path) -> dict:
    """Open with GDAL and return the JSON info dict; {} on failure."""
    try:
        ds = gdal.Open(str(path))
        return gdal.Info(ds, format="json") if ds is not None else {}
    except Exception:
        return {}
    
def _extract_time_components(s: str) -> tuple[int, int, int] | None:
    """Extract (hour, minute, second) from any string containing time information.

    Handles the most common encodings found in satellite metadata and filenames:

    - ISO T-separator with or without colons: ``T10:30:45``, ``T103045``
    - Plain ``HH:MM:SS`` (with optional fractional seconds / timezone suffix)
    - Bare ``HHMMSS`` immediately after a ``YYYYMMDD`` date (e.g. PlanetScope
      filenames: ``PSScene_20170801_094239_…``)

    Args:
        s: Any string that may contain a time (tag value, filename, …).

    Returns:
        ``(hour, minute, second)`` as integers, or ``None`` if no time found.
    """
    # ISO T-separator: T10:30:45 | T10:30:45.123Z | T103045
    m = re.search(r"T(\d{2}):?(\d{2}):?(\d{2})", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Plain HH:MM:SS (with optional fractional seconds / timezone)
    m = re.search(r"\b(\d{2}):(\d{2}):(\d{2})", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Bare HHMMSS immediately after YYYYMMDD (e.g. PlanetScope: _094239_)
    m = re.search(r"(?:19|20|21)\d{6}[_\-T](\d{2})(\d{2})(\d{2})[_\-.]", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None

def _parse_time_tag_from_xml(
    xml_text: str, sensor_family: str | None
) -> tuple[int, int, int] | None:
    """Look for a dedicated acquisition-time XML tag and return ``(h, m, s)``.

    Tries the sensor-specific tag from *_XML_TIME_TAGS* first, then the
    generic fallbacks in *_GENERIC_XML_TIME_PATTERNS*.

    Args:
        xml_text: Full text content of a metadata / sidecar file.
        sensor_family: Lowercase sensor family name, or ``None``.

    Returns:
        ``(hour, minute, second)`` as integers, or ``None`` if not found.
    """
    patterns_to_try: list[str] = []
    if sensor_family and sensor_family in _XML_TIME_TAGS:
        patterns_to_try.append(_XML_TIME_TAGS[sensor_family])
    patterns_to_try.extend(_GENERIC_XML_TIME_PATTERNS)

    for pat in patterns_to_try:
        m = re.search(pat, xml_text)
        if m:
            tc = _extract_time_components(m.group(1).strip())
            if tc:
                return tc
    return None

def search_date_in_filename(filename: str | pathlib.Path) -> str | None:
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

def _parse_date_from_xml(
    xml_text: str, sensor_family: str | None
) -> tuple[str | None, tuple[int, int, int] | None]:
    """Search *xml_text* for an acquisition date (and time) and return both.

    Strategy:

    1. Tries the sensor-specific date tag (via *_XML_DATE_TAGS*), then the
       generic fallback patterns (*_GENERIC_XML_DATE_PATTERNS*).
    2. After finding the raw tag value, tries to extract time directly from it
       (handles sensors like PlanetScope / Sentinel whose date tag contains a
       full ISO datetime, e.g. ``2021-07-14T10:30:45Z``).
    3. If no time was embedded in the date value, looks for a dedicated time
       XML tag via :func:`_parse_time_tag_from_xml`.

    Args:
        xml_text: Full text content of a metadata / sidecar file.
        sensor_family: Lowercase sensor family name (e.g. "spot", "sentinel").
                       May be ``None`` — only generic patterns are tried.

    Returns:
        ``(date_str, time_components)`` where *date_str* is ``"YYYY-MM-DD"``
        and *time_components* is ``(hour, minute, second)`` or ``None``.
        Both are ``None`` if no date was found.
    """
    patterns_to_try: list[str] = []
    if sensor_family and sensor_family in _XML_DATE_TAGS:
        patterns_to_try.append(_XML_DATE_TAGS[sensor_family])
    patterns_to_try.extend(_GENERIC_XML_DATE_PATTERNS)

    for pat in patterns_to_try:
        m = re.search(pat, xml_text)
        if m:
            raw = m.group(1).strip()
            date = search_date_in_filename(raw)
            if date:
                # Try time embedded in the same tag value first
                time_c = _extract_time_components(raw)
                # Fall back to a dedicated time tag if the date tag had no time
                if time_c is None:
                    time_c = _parse_time_tag_from_xml(xml_text, sensor_family)
                return date, time_c
    return None, None

# * Works properly
def sensors(sensors_names: Optional[Iterable[str]] = None) -> str:
    s = ""
    for string in sensors_names:
        s += string.lower() + "|"
        s += string.upper() + "|"
    s = s[:-1]
    return s

# * Works properly
def sensor_normalize(text: Optional[str]) -> dict:
    """Normalize sensor names to a canonical form.

    Args:
        text (Optional[str]): The input sensor name.

    Returns:
        dict: A dictionary with normalized sensor information.
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
        plat = f"sentinel-{var}" if var else None
        return out("sentinel-2", platform=plat, family="sentinel")

    # --- Landsat (L5/L7/L8/L9 codes; classic names)
    # m = re.search(r"\b(?:landsat\s*(5|7|8|9)|l(5|7|8|9))\b", t)
    m = re.search(r"\b(?:landsat\s*(4|5|7|8|9)|l(4|5|7|8|9)|lc0?(4|5|7|8|9)|le0?(4|5|7|8|9)|lt0?(4|5|7|8|9))\b", t)
    if m:
        num = next(g for g in m.groups() if g)
        return out(f"landsat{num}", platform=f"landsat{num}", family="landsat")

    # --- SPOT (4/5/6/7)
    # m = re.search(r"\bspot\s*(4|5|6|7)\b", t)
    m = re.search(r"\bspot\s*(4|5|6|7)\b|\bsp\s*(4|5|6|7)\b|\bs\s*(4|5|6|7)p\b", t)
    if m:
        # num = m.group(1)
        num = next(g for g in m.groups() if g)
        return out(f"spot{num}", platform=f"spot{num}", family="spot")

    # --- Pléiades 1A/1B
    m = re.search(r"\b(pl[eé]iades)\s*(1a|1b)\b|\bple\b|\bpl1[ab]\b|\bphr[1-2]a\b|\bphr[1-2]b\b", t)
    if m:
        var = m.group(2).lower()
        return out("pleiades", platform=f"pleiades-{var}", family="pleiades")

    # --- Pléiades Neo (Neo1/Neo2)
    m = re.search(r"\b(pl[eé]iades[-\s]?neo|pneo)\s*(\d+)?\b", t)
    if m:
        var = m.group(2)
        plat = f"pleiadesneo{var}" if var else None
        return out("pleiadesneo", platform=plat, family="pleiades")

    # --- PlanetScope (PS2, PS2.SD / SuperDove, PSScene)
    m = re.search(r"\b(superdove|ps2\.?sd|psscene|planetscope|ps2|psb|planet|analyticms)\b", t)
    if m:
        token = m.group(1)
        var = "SuperDove" if token in {"superdove", "ps2.sd", "ps2sd"} else None
        return out("planetscope", platform="planetscope", family="planetscope")

    # --- RapidEye (RE1..RE5)
    m = re.search(r"\brapideye\b|\bre([1-5])\b", t)
    if m:
        var = m.group(1).lower() if m.group(1) else None
        plat = f"rapideye-{var}" if var else None
        return out("rapideye", platform=plat, family="rapideye")

    # --- WorldView (WV1..WV4)
    m = re.search(r"\b(?:worldview[-\s]?(1|2|3|4)|wv(1|2|3|4))\b", t)
    if m:
        num = next(g for g in m.groups() if g)
        return out(f"worldview{num}", platform=f"worldview{num}", family="worldview")

    # --- GeoEye-1
    m = re.search(r"\bgeoeye[-\s]?1\b|\bge1\b", t)
    if m:
        return out("geoeye1", platform="geoeye1", family="geoeye")

    # --- QuickBird
    m = re.search(r"\bquickbird\b|\bqb[12]?\b", t)
    if m:
        return out("quickbird", platform="quickbird", family="quickbird")

    # --- IKONOS
    m = re.search(r"\bikonos\b", t)
    if m:
        return out("ikonos", platform="ikonos", family="ikonos")

    # --- SwissImage
    if re.search(r"\bswissimage\b", t):
        return out("aerial", platform="swissimage", family="aerial")#, matched="swissimage")
    
    # --- Aerial / UAV / Drone / Orthophoto (generic)
    if re.search(r"\b(aerial|orthophoto|uav|drone)\b", t):
        return out("aerial", platform="aerial", family="aerial")#, matched="aerial/uav/drone")

    # --- DEM / hillshade
    if re.search(r"\bdem\b|\bhillshade\b", t):
        return out("dem", platform=None, family="dem")#, matched="dem/hillshade")

    # Unknown
    return out("unknown", conf=0.0, matched=None)

def extract_acquisition_date(
    raster_path: str | pathlib.Path,
    sensor: str | None = None,
) -> datetime | None:
    """Extract the acquisition date (and time when available) from a satellite raster.

    Tries the following strategies in order, stopping at the first success:

    1. **Metadata files** — searches the raster's parent directory for sidecar
       XML / DIMAP files and parses date and time via :func:`_parse_date_from_xml`.
    2. **Filename** — calls :func:`search_date_in_filename`; for sensors that
       encode acquisition time in the filename the time is extracted too.
    3. **Default time** — if no time was found by either strategy, defaults to
       ``00:00:00``.

    Args:
        raster_path: Path to the raster file (.tif, .jp2, etc.)
        sensor: Optional sensor name hint (e.g. ``"spot6"``, ``"sentinel-2"``).
                If ``None``, the sensor family is inferred from the filename.

    Returns:
        A :class:`datetime` with the acquisition date and time, or ``None`` if
        no date could be found.
    """
    raster_path = pathlib.Path(raster_path)

    sensor_hint = sensor or raster_path.name
    sensor_family: str | None = sensor_normalize(sensor_hint).get("family")

    date: str | None = None
    time_c: tuple[int, int, int] | None = None

    # --- Step 1: Metadata files (XML / DIMAP) ---
    parent = raster_path.parent
    stem = raster_path.stem
    candidates: list[pathlib.Path] = []
    for name_pattern in [f"{stem}.xml", f"{stem}_metadata.xml", f"{stem}.dim"]:
        p = parent / name_pattern
        if p.exists():
            candidates.append(p)
    candidates.extend(sorted(parent.glob("*MTL.xml")))

    # --- Step 1b: Date+time prefix matching (sensor-agnostic) ---
    # Filters candidate XMLs to those sharing the same date+time token as the
    # image file, so directories with multiple scenes don't bleed into each other.
    _date_raw = search_date_in_filename(raster_path.name)
    if _date_raw:
        _date_compact = _date_raw.replace("-", "")  # "20170801"
        _tm = re.search(
            rf"{_date_compact}[_\-T](\d{{6}})[_\-.]", raster_path.name
        )
        _time_token: str | None = _tm.group(1) if _tm else None  # "094239"
        for ext in ("*.xml", "*.dim"):
            for p in sorted(parent.glob(ext)):
                if p in candidates:
                    continue
                if _date_compact not in p.name:
                    continue
                if _time_token is not None and _time_token not in p.name:
                    continue
                candidates.append(p)

    # --- Step 1c: Generic glob fallback (last resort) ---
    for ext in ("*.xml", "*.dim"):
        for p in sorted(parent.glob(ext)):
            if p not in candidates:
                candidates.append(p)

    for candidate in candidates:
        try:
            xml_text = candidate.read_text(errors="replace")
        except OSError:
            continue
        date, time_c = _parse_date_from_xml(xml_text, sensor_family)
        if date:
            break

    # --- Step 2: Filename fallback ---
    if not date:
        date = search_date_in_filename(raster_path.name)
        if date:
            time_c = _extract_time_components(raster_path.name)

    if not date:
        return None

    # --- Step 3: Build datetime; default time to 00:00:00 if not found ---
    h, m, s = time_c if time_c is not None else (0, 0, 0)
    try:
        return datetime.strptime(date, "%Y-%m-%d").replace(hour=h, minute=m, second=s)
    except ValueError:
        return None