#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# conftest.py
# creation date: 2026-06-02.
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
"""Shared fixtures and MockRaster for corrections module tests."""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.ma as ma
import pytest
from affine import Affine


@dataclass
class MockRaster:
    """Lightweight mock of geoutils.Raster for testing corrections.

    Provides the minimal interface needed by BaseCorrection and masks:
    - .data: np.ma.MaskedArray
    - .transform: Affine
    - .res: tuple (res_x, res_y)
    - .copy(new_array): returns new MockRaster
    """
    data: np.ma.MaskedArray
    transform: Affine = field(
        default_factory=lambda: Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    )
    res: Tuple[float, float] = (10.0, 10.0)

    def copy(self, new_array: np.ma.MaskedArray) -> "MockRaster":
        """Return a new MockRaster with updated data, same georeference."""
        return MockRaster(data=new_array, transform=self.transform, res=self.res)
    #END def

    def coords(self, grid: bool = True, force_offset: str = "ll", **_):
        """Map-coordinate meshgrids (x, y), mimicking geoutils.Raster.coords.

        Lower-left ("ll") pixel-corner offset, derived from the affine
        transform — enough for DirectionalBiasCorrection's rotated projection.
        """
        n_rows, n_cols = self.data.shape
        cols = np.arange(n_cols)
        rows = np.arange(n_rows)
        x = self.transform.c + (cols + 0.0) * self.transform.a
        y = self.transform.f + (rows + 1.0) * self.transform.e
        if not grid:
            return x, y
        return np.meshgrid(x, y)
    #END def

# ---------------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------------- #

@pytest.fixture
def flat_raster() -> MockRaster:
    """10×10 raster of all zeros, no mask."""
    data = ma.array(np.zeros((10, 10)), mask=False)
    return MockRaster(data=data)
#END def

@pytest.fixture
def offset_raster() -> MockRaster:
    """10×10 raster of all 5.0, no mask. Used for MedianCentering tests."""
    data = ma.array(np.full((10, 10), 5.0), mask=False)
    return MockRaster(data=data)
#END def

@pytest.fixture
def ramp_raster() -> MockRaster:
    """10×10 raster with linear ramp: data[i,j] = 0.1·i + 0.2·j + 1.0.

    Key values:
    - data[0,0] = 1.0
    - data[9,9] = 0.1·9 + 0.2·9 + 1.0 = 3.7
    - median ≈ 2.35
    """
    i, j = np.mgrid[:10, :10]
    data = 0.1 * i + 0.2 * j + 1.0
    return MockRaster(data=ma.array(data, mask=False))
#END def

@pytest.fixture
def partial_mask_raster() -> MockRaster:
    """10×10 ramp raster with corners masked: mask[0,0] = mask[9,9] = True.

    98 valid pixels (100 - 2 masked).
    """
    i, j = np.mgrid[:10, :10]
    data = 0.1 * i + 0.2 * j + 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    mask[9, 9] = True
    return MockRaster(data=ma.array(data, mask=mask))
#END def

@pytest.fixture
def dem_raster() -> MockRaster:
    """10×10 DEM with random elevation: uniform(100, 500).

    Uses rng(0) for reproducibility. Independent of row/col to avoid collinearity
    in predictor tests.
    """
    dem_data = np.random.default_rng(0).uniform(100, 500, (10, 10))
    return MockRaster(data=ma.array(dem_data, mask=False))
#END def

@pytest.fixture
def mixed_raster(dem_raster) -> MockRaster:
    """10×10 raster combining spatial ramp and DEM predictor.

    data = 0.1·R + 0.2·C + 3.0·dem + 2.0

    Recoverable coefficients (within 1e-5 for linear fit):
    - a_row ≈ 0.1
    - b_col ≈ 0.2
    - c_pred ≈ 3.0
    - d ≈ 2.0
    - r2 ≈ 1.0
    """
    i, j = np.mgrid[:10, :10]
    dem_vals = ma.filled(dem_raster.data, np.nan)
    data = 0.1 * i + 0.2 * j + 3.0 * dem_vals + 2.0
    return MockRaster(data=ma.array(data, mask=False))


@pytest.fixture
def slope_dem_raster() -> MockRaster:
    """10×10 DEM with pure row-gradient: dem[i,j] = 10.0·i.

    With 10m pixel spacing:
    - dz/drow = 10.0 / (2·10.0) = 0.5 at interior pixels
    - Interior slope = arctan(0.5) ≈ 26.57° (not 45°)

    Actually, for a simple ramp, gradient returns dz/drow = 1.0 per pixel.
    So for dem[i] = 10.0·i, dz/drow = 10.0 / (2·10.0) = 0.5...
    Wait, let me reconsider: np.gradient on [0, 10, 20, 30, ...]
    with spacing 10 gives [1.0, 1.0, ...] (each difference is 10, spacing is 10).
    So interior slope = arctan(1.0) = 45.0°.
    """
    i, j = np.mgrid[:10, :10]
    dem_data = 10.0 * i
    return MockRaster(data=ma.array(dem_data, mask=False))


@pytest.fixture
def cc_raster() -> MockRaster:
    """10×10 raster of all 0.8 (high correlation coefficient)."""
    data = ma.array(np.full((10, 10), 0.8), mask=False)
    return MockRaster(data=data)


@pytest.fixture
def all_stable() -> np.ndarray:
    """10×10 boolean array of all True (all pixels stable)."""
    return np.ones((10, 10), dtype=bool)


@pytest.fixture
def no_stable() -> np.ndarray:
    """10×10 boolean array of all False (no pixels stable)."""
    return np.zeros((10, 10), dtype=bool)
