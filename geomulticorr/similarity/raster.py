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

After correlation
-----------------
:func:`compare_after_correlation` measures the same indices once the right
image has been aligned on the left one by a displacement field: ASP's raw
disparity (:func:`read_disparity`, ``<pair>-F.tif``) or the corrected maps
(:func:`displacement_to_disparity`, ``<pair>-F_{EW,NS}_corr.tif``). Both are
expressed as ``(dx, dy)`` in left-image pixels: the left pixel ``(row, col)``
matches the right pixel ``(row + dy, col + dx)``.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from geomulticorr.similarity.metrics import SIMILARITY_KEYS, nmad, similarity_report
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


# --------------------------------------------------------------------------- #
# after correlation
# --------------------------------------------------------------------------- #
def read_disparity(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Grid]:
    """ASP's filtered disparity ``<pair>-F.tif``: ``(dx, dy, valid, grid)``.

    Band 1 is dx, band 2 dy (left-image pixels), band 3 the validity flag.
    A pixel is valid when that flag is set and both offsets are finite.
    """
    with rasterio.open(path) as src:
        dx = as_float(src.read(1), src.nodata)
        dy = as_float(src.read(2), src.nodata)
        valid = src.read(3) > 0 if src.count >= 3 else np.ones(dx.shape, bool)
        grid = Grid(src.transform, src.crs, (src.height, src.width))
    return dx, dy, valid & np.isfinite(dx) & np.isfinite(dy), grid


def displacement_to_disparity(
    ew_path, ns_path, grid: Grid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corrected EW/NS maps (metres) as ``(dx, dy, valid)`` on *grid* (pixels).

    The maps usually sit on the extraction grid, not the thumb grid, so both
    are warped onto *grid* first; their declared no-data (``0`` for GMC's
    corrected maps) becomes NaN. NS is **north-positive** while rows grow
    southwards, hence ``dy = −NS / res``.
    """
    ew = warp_to_grid(ew_path, grid)
    ns = warp_to_grid(ns_path, grid)
    dx = ew / grid.res
    dy = -ns / float(abs(grid.transform.e))
    return dx, dy, np.isfinite(dx) & np.isfinite(dy)


def apply_disparity(b, dx, dy, valid) -> np.ndarray:
    """Resample *b* at ``(row + dy, col + dx)``: the right image aligned on the left.

    Bilinear. NaN where the disparity is invalid, where the sample falls
    outside *b*, and wherever a NaN of *b* contributes to the interpolation,
    so a hole never leaks a fabricated value.
    """
    b = np.asarray(b, dtype=np.float64)
    if np.shape(dx) != b.shape or np.shape(dy) != b.shape:
        raise ValueError(f"disparity {np.shape(dx)} does not match image {b.shape}")
    ok = np.asarray(valid, bool) & np.isfinite(dx) & np.isfinite(dy)
    rows, cols = np.indices(b.shape, dtype=np.float64)
    coords = [rows + np.where(ok, dy, 0.0), cols + np.where(ok, dx, 0.0)]
    holes = ~np.isfinite(b)
    out = ndimage.map_coordinates(
        np.where(holes, 0.0, b), coords, order=1, mode="constant", cval=np.nan
    )
    touched = ndimage.map_coordinates(
        holes.astype(np.float64), coords, order=1, mode="constant", cval=1.0
    )
    out[~ok | (touched > 0) | ~np.isfinite(out)] = np.nan
    return out


def masked_stats(arr, keep=None) -> dict:
    """Median, fraction ≥ 0.5, NMAD and count of the finite pixels of *arr* in *keep*."""
    arr = np.asarray(arr, dtype=np.float64)
    sel = (
        np.isfinite(arr) if keep is None else np.asarray(keep, bool) & np.isfinite(arr)
    )
    v = arr[sel]
    if v.size == 0:
        return {"median": np.nan, "frac_gte_050": np.nan, "nmad": np.nan, "n": 0}
    return {
        "median": float(np.median(v)),
        "frac_gte_050": float(np.mean(v >= 0.5)),
        "nmad": nmad(v),
        "n": int(v.size),
    }


def compare_after_correlation(
    path_left,
    path_right,
    dx,
    dy,
    valid,
    *,
    grid: Grid,
    keep=None,
    band: int = 1,
    **report_kwargs,
) -> dict:
    """The similarity indices once the right image is aligned by ``(dx, dy)``.

    :param path_left: Left image of the pair (the disparity's reference).
    :param path_right: Right image; put on *grid*, then resampled by the disparity.
    :param dx, dy, valid: Disparity on *grid*, in left-image pixels.
    :param grid: The disparity's grid (the left image's for ASP output).
    :param keep: Stable-ground *keep* mask on *grid* (None → everything).
    :param report_kwargs: Any of :data:`REPORT_DEFAULTS`.
    :returns: ``{"indices": {...REPORT_KEYS}, "disp_valid_frac": float}``; the
        valid fraction is taken over the kept pixels where the left image has data.
    :raises ValueError: The disparity is not on *grid*.
    """
    params = report_params(**report_kwargs)
    if np.shape(dx) != tuple(grid.shape):
        raise ValueError(
            f"disparity shape {np.shape(dx)} is not the grid's {tuple(grid.shape)}"
        )
    keep = np.ones(grid.shape, bool) if keep is None else np.asarray(keep, bool)
    a = warp_to_grid(path_left, grid, band)
    b = apply_disparity(warp_to_grid(path_right, grid, band), dx, dy, valid)
    base = keep & np.isfinite(a)
    frac = float(np.asarray(valid, bool)[base].mean()) if base.any() else np.nan
    return {
        "indices": {
            "res": grid.res,
            **similarity_report(a, b, mask=keep, res=grid.res, **params),
        },
        "disp_valid_frac": frac,
    }
