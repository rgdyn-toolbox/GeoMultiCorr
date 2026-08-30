#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _grid.py
# creation date: 2026-08-17.
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
"""Raster-grid identity, comparison and materialisation.

A raster's *grid* is the triple ``(shape, transform, crs)``.  Two rasters can
share a shape and still be on completely different ground — a different origin,
a different pixel size, or a different CRS.  Comparing shapes alone therefore
lets a misaligned raster through silently, which is exactly the defect this
module exists to make impossible.

Nothing here imports from :mod:`geomulticorr`, so both
:mod:`geomulticorr.corrections` and :mod:`geomulticorr.core` can use it without
an import cycle — the same rule that governs
:mod:`geomulticorr.utils._pairs_geometry`.
"""

from __future__ import annotations

import pathlib

import numpy as np
import geoutils as gu

# Decimal places used when comparing affine coefficients.  Transforms that
# survive a GeoTIFF write/read round-trip differ in the last bits of the
# float64 mantissa; without rounding, a grid would fail to match itself.
_TRANSFORM_NDIGITS = 9

__all__ = [
    "grid_key",
    "grids_match",
    "describe_grid",
    "regrid_to_ref",
    "write_binary_raster",
]


def _shape_of(raster) -> tuple[int, ...]:
    """Spatial shape ``(rows, cols)`` of *raster*, band dimension dropped.

    Prefers the ``.shape`` metadata attribute, which a :class:`geoutils.Raster`
    populates from the file header.  Reading ``.data`` instead would force a
    raster opened with ``load_data=False`` to pull its whole array into memory
    — so a grid *comparison* would cost a full read.  Falls back to
    ``.data.shape`` for array-backed stand-ins such as the test-suite's
    ``MockRaster``.
    """
    shape = getattr(raster, "shape", None)
    if shape is None:
        shape = raster.data.shape

    shape = tuple(shape)
    if len(shape) == 3 and shape[0] == 1:
        shape = shape[1:]
    return shape


def grid_key(raster) -> tuple:
    """Return the hashable grid identity of *raster*.

    The identity is ``(shape, rounded transform coefficients, crs)`` — the
    three things that must agree for two rasters to describe the same pixels
    on the same ground.

    ``transform`` and ``crs`` are read with :func:`getattr` defaults because
    the corrections test-suite drives these helpers through ``MockRaster``
    (``tests/corrections/conftest.py``), which carries a ``transform`` and a
    ``res`` but no ``crs``.  A missing attribute compares equal to another
    missing attribute, so mock-vs-mock comparisons still work.

    :param raster: Any object exposing ``.data`` and, ideally, ``.transform``
        and ``.crs`` — a :class:`geoutils.Raster` or a compatible stand-in.
    :returns: Tuple usable for equality comparison and as a dict key.
    :rtype: tuple
    """
    transform = getattr(raster, "transform", None)
    crs = getattr(raster, "crs", None)

    if transform is None:
        transform_key: tuple | None = None
    else:
        transform_key = tuple(
            round(float(coeff), _TRANSFORM_NDIGITS) for coeff in tuple(transform)[:6]
        )

    return (_shape_of(raster), transform_key, crs)


def grids_match(a, b) -> bool:
    """Whether *a* and *b* occupy the same pixels on the same ground.

    This is the check that replaces ``a.data.shape != b.data.shape``.  A
    half-pixel origin shift or a differing CRS returns ``False`` here where a
    shape comparison would have returned "same".

    :param a: First raster.
    :param b: Second raster.
    :returns: ``True`` when shape, transform and CRS all agree.
    :rtype: bool
    """
    return grid_key(a) == grid_key(b)


def describe_grid(raster) -> str:
    """One-line human-readable grid description, for log messages.

    Example: ``3708x3158 @ (500000.0, 9900000.0) res 3.0 EPSG:32717``.

    :param raster: Raster to describe.
    :returns: Compact description; degrades gracefully when the transform or
        CRS is absent.
    :rtype: str
    """
    shape = _shape_of(raster)
    parts = ["x".join(str(n) for n in reversed(shape))]

    transform = getattr(raster, "transform", None)
    if transform is not None:
        parts.append(f"@ ({transform.c:g}, {transform.f:g})")
        parts.append(f"res {abs(transform.a):g}")

    crs = getattr(raster, "crs", None)
    if crs is not None:
        epsg = None
        try:
            epsg = crs.to_epsg()
        except AttributeError:
            pass
        parts.append(f"EPSG:{epsg}" if epsg else str(crs))

    return " ".join(parts)


def regrid_to_ref(src, ref, resampling: str = "bilinear"):
    """Return *src* on *ref*'s grid, reprojecting only when it differs.

    When the grids already match, *src* is returned **unchanged and by
    identity** — no copy, no warp.  That mirrors
    :func:`geomulticorr.utils.gmc_functions._preview`, which likewise hands
    back the same object when there is nothing to do.

    :param src: Raster to place on the reference grid.
    :param ref: Raster defining the target grid.
    :param resampling: Resampling method passed to
        :meth:`geoutils.Raster.reproject`.  Defaults to ``"bilinear"``, which
        is also geoutils' own default.
    :returns: ``(raster, regridded)`` — the raster on *ref*'s grid, and whether
        a reprojection actually happened.
    :rtype: tuple
    """
    if grids_match(src, ref):
        return src, False
    return src.reproject(ref=ref, resampling=resampling), True


def write_binary_raster(
    array: np.ndarray,
    transform,
    crs,
    path: str | pathlib.Path,
) -> pathlib.Path:
    """Write a boolean/0-1 *array* as a 1-bit GeoTIFF.

    Binary rasters compress dramatically when packed one pixel per bit: a
    1500x1500 mask lands around 14 kB rather than the ~87 kB the same data
    costs as float32.  GDAL, rasterio and QGIS expand ``NBITS=1``
    transparently on read, so consumers just see a uint8 0/1 raster.

    Three geoutils 0.2.5 behaviours make this fiddlier than it looks, and all
    three are load-bearing:

    - **The raster is built from scratch, not copied from a reference.**
      ``reference.copy(new_array=…)`` inherits the reference's nodata — for
      GMC's displacement rasters that is ``0.0``, which would flag every zero
      pixel as nodata.  Here ``0`` is a real value.
    - **The array is cast to uint8 explicitly.**  Handing ``to_file`` a
      ``bool`` array makes geoutils cast it *and* force ``nodata=255``, which
      cannot be represented in one bit and silently defeats ``NBITS=1``.
    - **``dtype=`` is not used.**  It is accepted by ``to_file`` in 0.2.5 but
      never read; the on-disk dtype follows the array, so the cast above is
      the only thing that controls it.

    :param array: Boolean or 0/1 array.  A leading singleton band axis is
        squeezed away.
    :type array: numpy.ndarray
    :param transform: Affine geotransform for the output.
    :param crs: Coordinate reference system for the output.
    :param path: Destination ``.tif`` path.
    :returns: The path written.
    :rtype: pathlib.Path
    """
    path = pathlib.Path(path)
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    raster = gu.Raster.from_array(
        arr.astype("uint8"),
        transform=transform,
        crs=crs,
        nodata=None,
    )
    raster.to_file(
        str(path),
        nodata=None,
        co_opts={"NBITS": "1", "COMPRESS": "DEFLATE", "TILED": "YES"},
    )
    return path
