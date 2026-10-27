#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _terrain.py
# creation date: 2026-09-30.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
# ---------------------------------------------------------------------------- #
"""
Terrain derivatives of a DEM: slope, single-light and multidirectional hillshade.

Pure NumPy, no GMC imports — same layering as ``corrections/fit.py``. These are
the products ``Session.sieve_bulk(image_type="dem")`` correlates in place of the
raw DEM, whose artefacts (voids, stripes, stereo blunders) correlate badly.

Conventions
-----------
- Arrays are row-major, north-up: row index increases southwards, column index
  eastwards. ``res_x`` / ``res_y`` are **positive** pixel sizes in metres.
- No-data is ``NaN`` in and ``NaN`` out. The Horn operator reads a 3×3
  neighbourhood, so any pixel touching a hole — and the one-pixel border of the
  grid — comes out ``NaN`` rather than with a fabricated gradient.
- Azimuths are compass degrees (0 = north, clockwise); altitude is degrees
  above the horizon.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Products derivable from a DEM, in the order ``sieve_bulk`` writes them.
TERRAIN_PRODUCTS = ("hillshade", "mdhillshade", "slope")

#: Default azimuths of the multidirectional hillshade (compass degrees).
DEFAULT_MD_AZIMUTHS = (225.0, 270.0, 315.0, 360.0)


def _horn_gradients(dem: np.ndarray, res_x: float, res_y: float):
    """Horn (1981) 3×3 gradients ``(dz/dx, dz/dy)``.

    ``dz/dx`` is positive when elevation rises eastwards; ``dz/dy`` is positive
    when it rises **southwards** (along increasing row). Border pixels are NaN.
    """
    z = np.asarray(dem, dtype=np.float64)
    p = np.pad(z, 1, mode="constant", constant_values=np.nan)
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * res_x)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * res_y)
    # Horn's kernel never reads the centre pixel, so a one-pixel hole would
    # otherwise come out with a finite gradient at the hole itself.
    hole = np.isnan(z)
    dzdx[hole] = np.nan
    dzdy[hole] = np.nan
    return dzdx, dzdy


def slope_degrees(
    dem: np.ndarray, res_x: float, res_y: float, method: str = "horn"
) -> np.ndarray:
    """Slope magnitude in degrees.

    :param method: ``"horn"`` (3×3, NaN-propagating, used for correlation
        products) or ``"gradient"`` (``np.gradient`` central differences with
        one-sided edges — the historical formula of
        :func:`~geomulticorr.corrections.fit.compute_slope`, kept so the slope
        predictor of ``SlopeRampCorrection`` does not move).
    """
    if method == "horn":
        dzdx, dzdy = _horn_gradients(dem, res_x, res_y)
    elif method == "gradient":
        dzdy, dzdx = np.gradient(dem, abs(res_y), abs(res_x))
    else:
        raise ValueError(f"Unknown slope method {method!r}; use 'horn' or 'gradient'")
    return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))


def hillshade(
    dem: np.ndarray,
    res_x: float,
    res_y: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Single-light hillshade in ``[0, 1]`` (Burrough & McDonnell / ESRI formula).

    A flat surface returns ``sin(altitude)``. Unlike
    ``matplotlib.colors.LightSource.hillshade`` called without ``dx``/``dy``,
    this uses the real pixel spacing, so the shading does not depend on the
    grid resolution's numeric value.
    """
    dzdx, dzdy = _horn_gradients(dem, res_x, res_y)
    dzdx = dzdx * z_factor
    dzdy = dzdy * z_factor
    zenith = np.radians(90.0 - altitude)
    az_math = np.radians((360.0 - azimuth + 90.0) % 360.0)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(dzdy, -dzdx)
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)
    shade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
        az_math - aspect
    )
    return np.clip(shade, 0.0, 1.0)


def multidirectional_hillshade(
    dem: np.ndarray,
    res_x: float,
    res_y: float,
    azimuths: Sequence[float] = DEFAULT_MD_AZIMUTHS,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Mean of single-light hillshades over *azimuths*, in ``[0, 1]``.

    Averaging several light directions reduces the aspect dependence of a
    single light: slopes facing away from one source are lit by another, so
    they keep texture instead of going flat and dark.

    The default set (225°–360°, the cartographic north-western lights) spans a
    135° arc, so some aspect contrast remains — deliberately: a full circle of
    lights makes the shade a function of slope alone and erases the aspect
    texture the correlator matches on. Pass evenly spaced azimuths to get that.
    """
    azimuths = tuple(azimuths)
    if not azimuths:
        raise ValueError("azimuths must contain at least one direction")
    acc = np.zeros(np.shape(dem), dtype=np.float64)
    for az in azimuths:
        acc += hillshade(
            dem, res_x, res_y, azimuth=az, altitude=altitude, z_factor=z_factor
        )
    return acc / len(azimuths)


def terrain_product(
    name: str,
    dem: np.ndarray,
    res_x: float,
    res_y: float,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    md_azimuths: Sequence[float] = DEFAULT_MD_AZIMUTHS,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Dispatch to the product named *name* (one of :data:`TERRAIN_PRODUCTS`)."""
    if name == "hillshade":
        return hillshade(
            dem, res_x, res_y, azimuth=azimuth, altitude=altitude, z_factor=z_factor
        )
    if name == "mdhillshade":
        return multidirectional_hillshade(
            dem,
            res_x,
            res_y,
            azimuths=md_azimuths,
            altitude=altitude,
            z_factor=z_factor,
        )
    if name == "slope":
        return slope_degrees(dem, res_x, res_y, method="horn")
    raise ValueError(
        f"Unknown terrain product {name!r}; expected one of {TERRAIN_PRODUCTS}"
    )


def shade_to_uint8(shade: np.ndarray) -> np.ndarray:
    """Encode a ``[0, 1]`` shade as uint8 ``1..255``, with ``0`` reserved for NaN.

    Same convention as the optical thumbs (nodata 0), so ASP needs no special
    handling beyond ``nodata-value 0``.
    """
    out = np.zeros(np.shape(shade), dtype=np.uint8)
    valid = np.isfinite(shade)
    out[valid] = (1 + np.round(np.clip(shade[valid], 0.0, 1.0) * 254)).astype(np.uint8)
    return out
