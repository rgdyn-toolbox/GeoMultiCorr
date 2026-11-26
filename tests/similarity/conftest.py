#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# conftest.py
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
"""Synthetic GeoTIFFs for the similarity raster layer and Session wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from scipy import ndimage

EPSG = 2154
ORIGIN = (900_000.0, 6_450_000.0)  # upper-left corner, Lambert-93


def texture(n, sigma=3.0, seed=0):
    g = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(n, n)), sigma)
    return (g - g.mean()) / g.std()


def write_tif(path, arr, res, origin=ORIGIN, nodata=0, epsg=EPSG, dtype="uint16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=f"EPSG:{epsg}",
        transform=Affine(res, 0, origin[0], 0, -res, origin[1]),
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def scene(n_fine=256, seed=0):
    """One landscape seen by two sensors: 1.5 m 'SPOT' and 3 m 'PlanetScope'.

    The coarse image is a 2×2 block mean of the fine one with a different gain,
    offset and noise — what a second sensor sees of the same ground.
    """
    fine = texture(n_fine, seed=seed)
    coarse = fine.reshape(n_fine // 2, 2, n_fine // 2, 2).mean(axis=(1, 3))
    rng = np.random.default_rng(seed + 100)
    spot = 400 + 80 * fine
    ps = 1500 + 300 * coarse + 20 * rng.normal(size=coarse.shape)
    return spot, ps


@pytest.fixture
def scene_files(tmp_path):
    spot, ps = scene()
    other = 1500 + 300 * texture(128, seed=9)
    return {
        "spot": write_tif(tmp_path / "spot.tif", spot, 1.5),
        "ps": write_tif(tmp_path / "ps.tif", ps, 3.0),
        "other": write_tif(tmp_path / "other.tif", other, 3.0),
    }


#: Thumbs of the project built by :func:`make_similarity_project`.
SPOT = "TestPz_2020-07-01_spot6"
PS_SAME_1 = "TestPz_2020-07-10_planetscope"
PS_SAME_2 = "TestPz_2021-07-10_planetscope"
PS_OTHER = "TestPz_2022-07-10_planetscope"


def make_similarity_project(tmp_path):
    """A GMC project, pzone 'TestPz': one 1.5 m SPOT thumb, two 3 m
    PlanetScope thumbs of the same ground, one PlanetScope thumb of unrelated
    ground."""
    from geomulticorr import open_gmc_session

    side = 384.0  # metres: 256 px at 1.5 m, 128 px at 3 m
    aoi = [
        (ORIGIN[0], ORIGIN[1] - side),
        (ORIGIN[0] + side, ORIGIN[1] - side),
        (ORIGIN[0] + side, ORIGIN[1]),
        (ORIGIN[0], ORIGIN[1]),
    ]
    s = open_gmc_session(tmp_path / "simproj", epsg_code=EPSG)
    s.insert_pzone(aoi, "TestPz", "TP")
    opt = s.thumb_dir("TestPz")
    spot, ps = scene(seed=0)
    write_tif(opt / f"{SPOT}.tif", spot, 1.5)
    write_tif(opt / f"{PS_SAME_1}.tif", ps, 3.0)
    write_tif(opt / f"{PS_SAME_2}.tif", ps + 5, 3.0)
    write_tif(opt / f"{PS_OTHER}.tif", 1500 + 300 * texture(128, seed=7), 3.0)
    s.update_thumbs()
    s.validate_all_thumbs()  # date strategies and "similarity" pair valid thumbs only
    return s
