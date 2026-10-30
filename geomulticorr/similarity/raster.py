#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# raster.py
# creation date: 2026-10-30.
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
Similarity between two GeoTIFFs: put them on one grid, mask stable ground, measure.

The georeferenced layer over :mod:`~geomulticorr.similarity.metrics`. It knows
rasterio and geopandas but **nothing of a GMC session**, so it compares any two
images — two Landsat scenes, a Sentinel-2 tile and a PlanetScope strip, two
hillshades — given only their paths.

Grid rules
----------
- ``grid="coarser"`` (default): the pair is compared on the grid of its
  coarser image. That is the grid a cross-sensor pair must be correlated on,
  since the finer image has to be degraded to match.
- ``grid=Grid(...)``: an explicit target grid (e.g. a GMC pzone reference grid).
- An image already on the target grid is used as read, pixel for pixel.
  Otherwise it is **anti-aliased before resampling** (Gaussian, σ = (f − 1)/2
  input pixels) and warped bilinearly: skipping the low-pass aliases the finer
  image's high frequencies into the coarse grid, which reads as dissimilarity.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

from geomulticorr.similarity.metrics import SIMILARITY_KEYS, similarity_report
from geomulticorr.similarity.preprocess import antialias_sigma, as_float, nan_gaussian

#: Columns of :func:`compare_rasters`' result: the grid resolution, then the report.
REPORT_KEYS: tuple[str, ...] = ("res",) + SIMILARITY_KEYS

#: Parameters that change a report's numbers, with their defaults.
REPORT_DEFAULTS: Mapping[str, Any] = {
    "kernel": 21,
    "search": 8,
    "prefilter_mode": 2,
    "prefilter_width": 1.4,
    "bins": 64,
    "tile": 128,
    "n_tiles": 200,
    "min_peak": 0.3,
    "rng": 0,
}


class Grid(NamedTuple):
    """A raster grid: where every pixel is, nothing about its values."""

    transform: Any
    crs: Any
    shape: tuple[int, int]

    @property
    def res(self) -> float:
        return float(abs(self.transform.a))

    @classmethod
    def of(cls, path) -> "Grid":
        """The grid of an existing raster file (header only)."""
        with rasterio.open(path) as src:
            return cls(src.transform, src.crs, (src.height, src.width))

    def key(self) -> tuple:
        """Hashable identity, for caching arrays already warped onto this grid."""
        return (tuple(self.transform)[:6], str(self.crs), tuple(self.shape))


def read_image(path, band: int = 1) -> tuple[np.ndarray, Grid]:
    """Read one band as float64 with no-data as NaN, plus its grid."""
    with rasterio.open(path) as src:
        arr = as_float(src.read(band), src.nodata)
        return arr, Grid(src.transform, src.crs, (src.height, src.width))


def warp_to_grid(
    path, grid: Grid, band: int = 1, resampling: str = "bilinear"
) -> np.ndarray:
    """Read *path* onto *grid*, anti-aliasing first when it is finer than the grid."""
    arr, own = read_image(path, band)
    if own.key() == grid.key():
        return arr
    arr = nan_gaussian(arr, antialias_sigma(own.res, grid.res))
    out = np.full(grid.shape, np.nan)
    reproject(
        arr,
        out,
        src_transform=own.transform,
        src_crs=own.crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling[resampling],
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return out


def coarser_grid(path_a, path_b) -> Grid:
    """Grid of whichever image has the larger pixel (*path_a* on a tie)."""
    ga, gb = Grid.of(path_a), Grid.of(path_b)
    return ga if ga.res >= gb.res else gb


def _as_geodataframe(source, layer: str | None = None) -> gpd.GeoDataFrame | None:
    if source is None:
        return None
    if isinstance(source, gpd.GeoDataFrame):
        return source
    if isinstance(getattr(source, "ds", None), gpd.GeoDataFrame):
        return source.ds  # a geoutils.Vector, without importing geoutils
    return gpd.read_file(str(source), layer=layer)


def stable_mask(
    moving_areas, grid: Grid, buffer: float = 0.0, layer: str | None = None
) -> np.ndarray:
    """Boolean *keep* mask on *grid*: ``True`` outside the moving areas.

    *moving_areas* is a path, a GeoDataFrame, a ``geoutils.Vector`` or None
    (everything kept). The polygons are reprojected to the grid CRS: a CRS
    mismatch would otherwise put every polygon off-grid and silently keep the
    moving terrain. A CRS-less frame raises rather than guessing.
    """
    gdf = _as_geodataframe(moving_areas, layer)
    if gdf is None or gdf.empty:
        return np.ones(grid.shape, dtype=bool)
    if gdf.crs is None:
        raise ValueError("moving-area polygons have no CRS; set one before use")
    geoms = gdf.to_crs(grid.crs).geometry
    if buffer:
        geoms = geoms.buffer(buffer)
    return geometry_mask(geoms, out_shape=grid.shape, transform=grid.transform)


def report_params(**report_kwargs) -> dict:
    """:data:`REPORT_DEFAULTS` updated with *report_kwargs*; unknown keys raise.

    The result fingerprints a stored row, so a re-run with other settings
    recomputes instead of silently reusing it.
    """
    unknown = set(report_kwargs) - set(REPORT_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown similarity parameter(s): {sorted(unknown)}")
    return {**REPORT_DEFAULTS, **report_kwargs}


def compare_rasters(
    path_a,
    path_b,
    *,
    moving_areas=None,
    buffer: float = 0.0,
    grid: str | Grid = "coarser",
    band: int = 1,
    cache: dict | None = None,
    **report_kwargs,
) -> dict:
    """All similarity metrics for two GeoTIFFs, on a common grid, on stable ground.

    :param path_a: Reference (left) image.
    :param path_b: Secondary (right) image.
    :param moving_areas: Polygons to exclude (see :func:`stable_mask`).
    :param buffer: Margin around the polygons, in grid CRS units.
    :param grid: ``"coarser"`` or an explicit :class:`Grid`.
    :param band: Band to read from both files.
    :param cache: Optional dict reused across calls: each image is read and
        warped once per grid, so an all-couples run costs one read per image
        rather than two per couple.
    :param report_kwargs: Any of :data:`REPORT_DEFAULTS`, forwarded to
        :func:`~geomulticorr.similarity.metrics.similarity_report`.
    :returns: A dict in :data:`REPORT_KEYS` order.
    """
    params = report_params(**report_kwargs)
    if isinstance(grid, str):
        if grid != "coarser":
            raise ValueError(f"grid must be 'coarser' or a Grid, got {grid!r}")
        grid = coarser_grid(path_a, path_b)
    cache = {} if cache is None else cache

    def _get(path):
        k = (str(path), band, grid.key())
        if k not in cache:
            cache[k] = warp_to_grid(path, grid, band)
        return cache[k]

    a, b = _get(path_a), _get(path_b)
    keep = stable_mask(moving_areas, grid, buffer)
    return {
        "res": grid.res,
        **similarity_report(a, b, mask=keep, res=grid.res, **params),
    }
