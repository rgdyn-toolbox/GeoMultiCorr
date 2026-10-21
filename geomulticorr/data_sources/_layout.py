#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _layout.py
# creation date: 2026-08-24.
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
"""Archive layout: where a downloaded scene goes, and what it is called.

Pure functions — no network, no GDAL, no GMC imports — shared by the USGS M2M
and Planetary Computer finders so the sensor mapping is defined exactly once.

The layout these build is a *regional archive*, not a GMC project: files keep
the names the catalog gave them and are grouped by sensor and by acquisition
grid (WRS-2 path/row for Landsat, MGRS tile for Sentinel-2)::

    <root>/LANDSAT8/232_083/LC08_L1TP_232083_20230301_20230307_02_T1_B8.TIF
    <root>/SENTINEL2/T19HCC/S2A_MSIL2A_20230301T143721_R096_T19HCC_.../B08.tif

Renaming into the GMC thumb convention (``<pzone>_<date>_<sensor>.tif``) happens
later and per-pzone, via ``Session.map_georasters_bank`` + ``Session.sieve_bulk``.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional, Tuple

__all__ = [
    "ORGANIZE_SENSOR_PATHROW",
    "ORGANIZE_SCHEMES",
    "LANDSAT_SENSOR_DIRS",
    "landsat_dirs",
    "sentinel2_dirs",
    "safe_filename",
    "filename_from_content_disposition",
]

#: The only layout scheme implemented so far. Kept as a string rather than a
#: boolean flag so further schemes can be added without another parameter.
ORGANIZE_SENSOR_PATHROW = "sensor_pathrow"

ORGANIZE_SCHEMES = (ORGANIZE_SENSOR_PATHROW,)

#: Landsat platform directory, keyed on the first field of a Collection-2
#: ``display_id``. This prefix is the *only* thing that separates Landsat 8 from
#: Landsat 9: both live in the single M2M dataset ``landsat_ot_c2_l1``, so the
#: collection key a scene was searched under cannot be trusted for this.
LANDSAT_SENSOR_DIRS = {
    "LC09": "LANDSAT9",
    "LC08": "LANDSAT8",
    "LE07": "LANDSAT7",
    "LT05": "LANDSAT5",
    "LT04": "LANDSAT4",
    "LM05": "LANDSAT5",
    "LM04": "LANDSAT4",
}

_UNKNOWN_GRID = "unknown"

# WRS-2 path/row as Landsat packs it into field 2 of a display_id: PPPRRR.
_PATHROW_RE = re.compile(r"^(\d{3})(\d{3})$")

# "T19HCC" or "19HCC"; MPC also reports it as grid:code = "MGRS-19HCC".
_MGRS_RE = re.compile(r"(\d{2}[A-Z]{3})\b")

# RFC 5987 extended form: filename*=UTF-8''LC08%5F...%5FB8.TIF
_CD_EXTENDED_RE = re.compile(
    r"filename\*\s*=\s*(?:[\w.-]+)?'[^']*'([^;]+)", re.IGNORECASE
)
# Plain form: filename="LC08_..._B8.TIF" or filename=LC08_..._B8.TIF
_CD_PLAIN_RE = re.compile(r'filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;]+)', re.IGNORECASE)


def landsat_dirs(display_id: str) -> Tuple[str, str]:
    """Return ``(sensor_dir, grid_dir)`` for a Landsat Collection-2 product id.

    Parameters
    ----------
    display_id : str
        Landsat product identifier, e.g. ``"LC08_L1TP_232083_20230301_20230307_02_T1"``.

    Returns
    -------
    tuple[str, str]
        Platform directory and zero-padded WRS-2 ``path_row``, e.g.
        ``("LANDSAT8", "232_083")``. Unrecognised input degrades to
        ``("LANDSAT", "unknown")`` rather than raising: a single odd product id
        should land somewhere inspectable, not abort a multi-hour bulk download.

    Examples
    --------
    >>> landsat_dirs("LC09_L1TP_232083_20230301_20230307_02_T1")
    ('LANDSAT9', '232_083')
    >>> landsat_dirs("nonsense")
    ('LANDSAT', 'unknown')
    """
    parts = str(display_id).split("_")
    sensor = LANDSAT_SENSOR_DIRS.get(parts[0].upper(), "LANDSAT") if parts else "LANDSAT"

    grid = _UNKNOWN_GRID
    if len(parts) >= 3:
        match = _PATHROW_RE.match(parts[2])
        if match:
            grid = f"{match.group(1)}_{match.group(2)}"
    return sensor, grid


def sentinel2_dirs(
    item_id: str, properties: Optional[Mapping[str, object]] = None
) -> Tuple[str, str]:
    """Return ``(sensor_dir, grid_dir)`` for a Sentinel-2 STAC item.

    The MGRS tile is read from the item's STAC properties when available —
    ``s2:mgrs_tile`` or ``grid:code`` — and only falls back to splitting the
    item id when neither is present.

    Parameters
    ----------
    item_id : str
        STAC item id, e.g.
        ``"S2A_MSIL2A_20230301T143721_R096_T19HCC_20230302T190112"``.
    properties : Mapping, optional
        The item's STAC ``properties`` dict.

    Returns
    -------
    tuple[str, str]
        ``("SENTINEL2", "T19HCC")``, or ``("SENTINEL2", "unknown")`` when the
        tile cannot be determined.

    Examples
    --------
    >>> sentinel2_dirs("S2A_MSIL2A_20230301T143721_R096_T19HCC_20230302T190112")
    ('SENTINEL2', 'T19HCC')
    >>> sentinel2_dirs("whatever", {"grid:code": "MGRS-19HCC"})
    ('SENTINEL2', 'T19HCC')
    """
    candidates = []
    if properties:
        for key in ("s2:mgrs_tile", "grid:code", "sentinel:utm_zone"):
            value = properties.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
    # Field 4 of the item id is the tile token (T19HCC).
    parts = str(item_id).split("_")
    if len(parts) >= 5:
        candidates.append(parts[4])
    candidates.append(str(item_id))

    for candidate in candidates:
        match = _MGRS_RE.search(candidate.upper())
        if match:
            return "SENTINEL2", f"T{match.group(1)}"
    return "SENTINEL2", _UNKNOWN_GRID


def safe_filename(name: str, fallback: str = "download") -> str:
    """Reduce a server-supplied name to a single safe path component.

    Download filenames arrive from HTTP headers, i.e. from outside the process.
    Anything that could climb out of the destination directory — separators,
    ``..``, a leading dot, NUL bytes — is stripped, so the name can only ever
    resolve inside the folder it was meant for.

    Parameters
    ----------
    name : str
        Candidate filename, possibly containing path separators.
    fallback : str
        Returned when nothing usable survives sanitisation.

    Returns
    -------
    str
        A safe single path component.

    Examples
    --------
    >>> safe_filename("../../etc/passwd")
    'passwd'
    >>> safe_filename("  LC08_B8.TIF  ")
    'LC08_B8.TIF'
    >>> safe_filename("///")
    'download'
    """
    cleaned = str(name).replace("\x00", "").strip().strip('"').strip()
    # Take the last component under either separator, so "a/b\\c.tif" -> "c.tif".
    cleaned = re.split(r"[\\/]", cleaned)[-1]
    cleaned = cleaned.replace("..", "").strip().strip(".").strip()
    return cleaned or fallback


def filename_from_content_disposition(header: Optional[str]) -> Optional[str]:
    """Extract the filename from a ``Content-Disposition`` header.

    Tries the RFC 5987 ``filename*`` form first (it carries the encoding and
    wins over the plain form when both are present), then plain ``filename``.

    Parameters
    ----------
    header : str or None
        Raw header value, e.g. ``'attachment; filename="LC08_..._B8.TIF"'``.

    Returns
    -------
    str or None
        The sanitised filename, or ``None`` when the header is absent or
        carries nothing usable.

    Examples
    --------
    >>> filename_from_content_disposition('attachment; filename="LC08_B8.TIF"')
    'LC08_B8.TIF'
    >>> filename_from_content_disposition(None) is None
    True
    """
    if not header:
        return None

    match = _CD_EXTENDED_RE.search(header)
    if match:
        from urllib.parse import unquote

        name = safe_filename(unquote(match.group(1)), fallback="")
        if name:
            return name

    match = _CD_PLAIN_RE.search(header)
    if match:
        raw = match.group(1) or match.group(2) or ""
        name = safe_filename(raw, fallback="")
        if name:
            return name
    return None
