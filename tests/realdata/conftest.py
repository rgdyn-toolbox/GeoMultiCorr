#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# conftest.py
# creation date: 2026-11-20.
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
"""Real study sites as GMC projects, installed through :mod:`geomulticorr.examples`.

Each site has, per sensor, an image bank and a project GMC built from it. A test gets its own
copy of the project in ``tmp_path`` (stages write into their project), with the geodatabase
paths rewritten to that copy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio

from geomulticorr import examples
from geomulticorr.core.session import open_gmc_session

SITE = "PierreBrune"
#: sensor key (bank) → project name
PROJECTS = {
    "planetscope": "MS_PierreBrune_Planet",
    "spot": "MS_PierreBrune_SPOT",
    "sentinel2": "MS_PierreBrune_Sentinel2",
    "landsat": "MS_PierreBrune_Landsat",
}


#: EPSG:2154 sample points, from ``scripts/plot_site.py`` in the data repository: the fastest
#: place of the SPOT series inside the moving areas, and the place within 1 km of the site
#: centre farthest from any of them.
SAMPLE_POINTS = {
    "rock_glacier": (1001120.0, 6483170.0),
    "stable": (1001932.0, 6482162.0),
}


def pytest_collection_modifyitems(items):
    here = Path(__file__).parent
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.realdata)


def load_asset(name: str) -> Path:
    """The cached asset, downloaded if needed.

    A registered asset that cannot be fetched skips locally but fails under
    ``GMC_REQUIRE_DATA=1`` (CI), so real-data tests cannot silently vanish. One outside the
    registry is not published yet, and always skips.
    """
    try:
        return examples.get_path(name)
    except (ValueError, OSError) as exc:
        if name in examples.REGISTRY and os.environ.get("GMC_REQUIRE_DATA") == "1":
            raise
        pytest.skip(f"asset {name!r} unavailable: {exc}")


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def read_band(path: Path, band: int = 1) -> np.ndarray:
    """One band as float, with nodata as NaN."""
    with rasterio.open(path) as src:
        data = src.read(band).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data


def nmad(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


@pytest.fixture
def project_from(tmp_path):
    """Factory: ``(session, cached_project_root, manifest)`` for a sensor key."""

    def _open(sensor: str):
        cached = load_asset(PROJECTS[sensor])
        root = examples.get_project(PROJECTS[sensor], tmp_path)
        return open_gmc_session(root), cached, read_manifest(cached)

    return _open
