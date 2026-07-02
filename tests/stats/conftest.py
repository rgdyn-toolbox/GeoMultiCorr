#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# tests/stats/conftest.py
# creation date: 2026-07-02.
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
"""Shared fixtures and lightweight mocks for the stats test module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest
import rasterio
from rasterio.transform import from_bounds


# ── lightweight mocks ──────────────────────────────────────────────────────────


@dataclass
class MockThumb:
    """Minimal mock of geomulticorr.core.thumb.Thumb."""

    th_date: str
    th_sensor: str


@dataclass
class MockPair:
    """Minimal mock of geomulticorr.core.pair.Pair for stats tests.

    Provides only the attributes accessed by stats.py functions.
    """

    pa_key: str
    pa_pz_name: str
    pa_left: MockThumb
    pa_right: MockThumb
    pa_dt_days: float
    pa_dt_months: float
    pa_dt_years: float
    pa_direction: str
    pa_disparity_f_path: Path
    pa_stats_path: Path
    pa_ew_path: Path
    pa_ns_path: Path
    pa_cc_raw_path: Path
    pa_magn_path: Path


@dataclass
class MockRasterData:
    """Minimal mock for gu.Raster — only the .data attribute is needed.

    Used in save_corrected_stats() tests, which call _compute_array_stats()
    on xDisp_corr.data / yDisp_corr.data (both must be MaskedArrays).
    """

    data: ma.MaskedArray


# ── GeoTIFF helper ─────────────────────────────────────────────────────────────


def write_geotiff(
    path: Path,
    data: np.ndarray,
    nodata: float | None = None,
    crs: str = "EPSG:32632",
) -> Path:
    """Write a single-band float32 GeoTIFF from a 2-D numpy array.

    :param path: Destination file path.
    :param data: 2-D array (rows × cols).
    :param nodata: Optional nodata value embedded in the file metadata.
    :param crs: Coordinate reference system (default: UTM zone 32N).
    :returns: The same *path* for convenient chaining.
    """
    transform = from_bounds(0.0, 0.0, 100.0, 100.0, data.shape[1], data.shape[0])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def make_tif(tmp_path):
    """Factory fixture: write a named GeoTIFF and return its Path.

    :param tmp_path: Built-in pytest temporary directory.
    :returns: Callable ``(data, name, nodata) → Path``.
    """

    def _write(
        data: np.ndarray,
        name: str = "raster.tif",
        nodata: float | None = None,
    ) -> Path:
        p = tmp_path / name
        write_geotiff(p, data, nodata=nodata)
        return p

    return _write


@pytest.fixture
def mock_pair(tmp_path) -> MockPair:
    """MockPair with temporary file paths; no rasters written to disk.

    Raster paths (ew, ns, cc, magn, disparity) are set but the files do not
    exist unless individual tests create them via ``write_geotiff()``.

    :param tmp_path: Built-in pytest temporary directory.
    :returns: Configured MockPair instance.
    """
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    return MockPair(
        pa_key="testpz_20220601_20220901",
        pa_pz_name="testpz",
        pa_left=MockThumb(th_date="2022-06-01", th_sensor="planetscope"),
        pa_right=MockThumb(th_date="2022-09-01", th_sensor="planetscope"),
        pa_dt_days=92.0,
        pa_dt_months=3.04,
        pa_dt_years=0.252,
        pa_direction="forward",
        pa_disparity_f_path=tmp_path / "disparity.tif",
        pa_stats_path=stats_dir / "testpz_20220601_20220901_stats.json",
        pa_ew_path=tmp_path / "ew.tif",
        pa_ns_path=tmp_path / "ns.tif",
        pa_cc_raw_path=tmp_path / "cc.tif",
        pa_magn_path=tmp_path / "magn.tif",
    )
