#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_filter_roles.py
# creation date: 2026-11-25.
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
"""The inversion applies the ``quality`` part of its filter pipeline only (D3)."""

import types

import numpy as np
import pytest
import rasterio

from geomulticorr.corrections import CCFilter, OutlierFilter, SlopeMask, StableAreaMask
from geomulticorr.inversion.tio_inversion import TIOInversion


def _write(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=rasterio.transform.from_origin(0, 100, 10, 10),
    ) as dst:
        dst.write(data.astype("float32"), 1)


@pytest.fixture
def pair(tmp_path):
    ew = np.full((10, 10), 1.0)
    ew[:5] = 2.0  # the moving half
    ew[0, 0] = 50.0  # one outlier
    paths = {}
    for name, data in (("EW", ew), ("NS", ew * 0.5), ("cc", np.full((10, 10), 0.9))):
        paths[name] = tmp_path / f"{name}.tif"
        _write(paths[name], data)
    return types.SimpleNamespace(
        pa_ew_corr_path=tmp_path / "no_corr_EW.tif",
        pa_ns_corr_path=tmp_path / "no_corr_NS.tif",
        pa_ew_path=paths["EW"],
        pa_ns_path=paths["NS"],
        pa_cc_path=paths["cc"],
    )


def _load(pair, pipeline):
    inv = TIOInversion.__new__(TIOInversion)
    inv.filter_pipeline = pipeline
    x, y = inv._load_and_filter(pair)
    return np.ma.getmaskarray(x.data), np.ma.getmaskarray(y.data)


def test_stable_area_mask_does_not_remove_the_moving_ground(pair):
    stable = np.ones((10, 10), bool)
    stable[:5] = False
    mx, my = _load(pair, OutlierFilter((0.0, 10.0)) + StableAreaMask(stable))
    assert mx.sum() == 1 and mx[0, 0]  # only the outlier goes
    assert my.sum() == 1 and my[0, 0]
    assert not mx[1:5].any()  # moving ground kept


def test_quality_masks_still_remove_pixels(pair):
    mx, _ = _load(pair, CCFilter(0.95))
    assert mx.all()


def test_single_mask_is_accepted(pair):
    mx, _ = _load(pair, OutlierFilter((0.0, 10.0)))
    assert mx.sum() == 1 and mx[0, 0]


def test_no_pipeline_keeps_everything(pair):
    mx, my = _load(pair, None)
    assert not mx.any() and not my.any()


def test_slope_mask_is_fit_only():
    assert SlopeMask.ROLE == "fit"
