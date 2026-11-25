#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_raster.py
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
"""Similarity between GeoTIFFs (similarity/raster.py): grids, masks, cache."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from scipy import ndimage
from shapely.geometry import box

from geomulticorr.similarity import raster as sr
from tests.similarity.conftest import EPSG, ORIGIN, texture, write_tif

KW = dict(kernel=21, search=4, tile=32)


class TestGrid:
    def test_of_and_res(self, scene_files):
        g = sr.Grid.of(scene_files["ps"])
        assert g.shape == (128, 128) and g.res == 3.0

    def test_coarser(self, scene_files):
        assert sr.coarser_grid(scene_files["spot"], scene_files["ps"]).res == 3.0
        assert sr.coarser_grid(scene_files["ps"], scene_files["spot"]).res == 3.0

    def test_read_nodata_to_nan(self, tmp_path):
        from tests.similarity.conftest import write_tif

        arr = np.full((4, 4), 7.0)
        arr[0, 0] = 0
        a, _ = sr.read_image(write_tif(tmp_path / "x.tif", arr, 3.0))
        assert np.isnan(a[0, 0]) and a[1, 1] == 7


class TestWarp:
    def test_same_grid_is_exact(self, scene_files):
        g = sr.Grid.of(scene_files["ps"])
        a, _ = sr.read_image(scene_files["ps"])
        np.testing.assert_array_equal(sr.warp_to_grid(scene_files["ps"], g), a)

    def test_fine_to_coarse_matches_block_mean(self, scene_files):
        g = sr.Grid.of(scene_files["ps"])
        warped = sr.warp_to_grid(scene_files["spot"], g)
        a, _ = sr.read_image(scene_files["spot"])
        block = a.reshape(128, 2, 128, 2).mean(axis=(1, 3))
        inner = np.s_[5:-5, 5:-5]
        assert np.corrcoef(warped[inner].ravel(), block[inner].ravel())[0, 1] > 0.99


class TestStableMask:
    def test_none_keeps_all(self, scene_files):
        g = sr.Grid.of(scene_files["ps"])
        assert sr.stable_mask(None, g).all()

    def test_polygon_reprojected(self, scene_files):
        """Polygons in another CRS must still land on the grid."""
        g = sr.Grid.of(scene_files["ps"])
        poly = box(ORIGIN[0] + 30, ORIGIN[1] - 90, ORIGIN[0] + 90, ORIGIN[1] - 30)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs=EPSG).to_crs(4326)
        keep = sr.stable_mask(gdf, g)
        assert not keep[10:30, 10:30].any() and keep[60:, 60:].all()

    def test_buffer_grows_exclusion(self, scene_files):
        g = sr.Grid.of(scene_files["ps"])
        poly = box(ORIGIN[0] + 150, ORIGIN[1] - 210, ORIGIN[0] + 210, ORIGIN[1] - 150)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs=EPSG)
        assert (~sr.stable_mask(gdf, g, buffer=30)).sum() > (
            ~sr.stable_mask(gdf, g)
        ).sum()

    def test_from_file(self, scene_files, tmp_path):
        g = sr.Grid.of(scene_files["ps"])
        poly = box(ORIGIN[0], ORIGIN[1] - 60, ORIGIN[0] + 60, ORIGIN[1])
        p = tmp_path / "ma.gpkg"
        gpd.GeoDataFrame(geometry=[poly], crs=EPSG).to_file(p)
        assert not sr.stable_mask(p, g)[:20, :20].any()

    def test_no_crs_raises(self, scene_files):
        g = sr.Grid.of(scene_files["ps"])
        with pytest.raises(ValueError, match="no CRS"):
            sr.stable_mask(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]), g)


class TestCompare:
    def test_keys(self, scene_files):
        r = sr.compare_rasters(scene_files["spot"], scene_files["ps"], **KW)
        assert tuple(r) == sr.REPORT_KEYS and r["res"] == 3.0

    def test_cross_sensor_same_ground_beats_other_ground(self, scene_files):
        same = sr.compare_rasters(scene_files["spot"], scene_files["ps"], **KW)
        diff = sr.compare_rasters(scene_files["ps"], scene_files["other"], **KW)
        assert same["trial_valid_frac"] > 0.9 > diff["trial_valid_frac"]
        assert same["trial_peak_median"] > diff["trial_peak_median"] + 0.3
        assert abs(same["trial_dx_median_px"]) < 0.2

    def test_mask_reduces_pixels(self, scene_files):
        poly = box(ORIGIN[0], ORIGIN[1] - 192, ORIGIN[0] + 192, ORIGIN[1])
        gdf = gpd.GeoDataFrame(geometry=[poly], crs=EPSG)
        full = sr.compare_rasters(scene_files["spot"], scene_files["ps"], **KW)
        part = sr.compare_rasters(
            scene_files["spot"], scene_files["ps"], moving_areas=gdf, **KW
        )
        assert part["n_pixels"] < full["n_pixels"]

    def test_cache_reads_each_image_once(self, scene_files, monkeypatch):
        calls = []
        real = sr.warp_to_grid
        monkeypatch.setattr(
            sr, "warp_to_grid", lambda p, *a, **k: calls.append(p) or real(p, *a, **k)
        )
        cache = {}
        for other in ("ps", "other"):
            sr.compare_rasters(
                scene_files["spot"], scene_files[other], cache=cache, **KW
            )
        assert sorted(map(str, calls)) == sorted(
            str(scene_files[k]) for k in ("spot", "ps", "other")
        )

    def test_explicit_grid(self, scene_files):
        g = sr.Grid.of(scene_files["spot"])
        r = sr.compare_rasters(scene_files["spot"], scene_files["ps"], grid=g, **KW)
        assert r["res"] == 1.5

    def test_bad_grid(self, scene_files):
        with pytest.raises(ValueError):
            sr.compare_rasters(scene_files["spot"], scene_files["ps"], grid="finer")

    def test_unknown_parameter(self, scene_files):
        with pytest.raises(TypeError, match="unknown"):
            sr.compare_rasters(scene_files["spot"], scene_files["ps"], kernal=21)


def test_report_params_defaults_and_override():
    p = sr.report_params(kernel=53)
    assert p["kernel"] == 53 and p["search"] == sr.REPORT_DEFAULTS["search"]
    assert set(p) == set(sr.REPORT_DEFAULTS)


# --------------------------------------------------------------------------- #
# after correlation
# --------------------------------------------------------------------------- #
DX, DY = 2.3, -1.7  # px: left (row, col) matches right (row + DY, col + DX)


def _write_disparity(path, dx, dy, valid, res=3.0, origin=ORIGIN):
    import rasterio
    from affine import Affine

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=dx.shape[0],
        width=dx.shape[1],
        count=3,
        dtype="float32",
        crs=f"EPSG:{EPSG}",
        transform=Affine(res, 0, origin[0], 0, -res, origin[1]),
    ) as dst:
        for i, band in enumerate((dx, dy, valid.astype("float32")), start=1):
            dst.write(np.asarray(band, dtype="float32"), i)
    return path


@pytest.fixture
def shifted_pair(tmp_path):
    """A 3 m left image and a right image whose content is moved by (DX, DY)."""
    left = 1000 + 200 * texture(128, seed=3)
    right = ndimage.shift(left, (DY, DX), order=3, mode="nearest")
    paths = {
        "left": write_tif(tmp_path / "left.tif", left, 3.0, dtype="float32"),
        "right": write_tif(tmp_path / "right.tif", right, 3.0, dtype="float32"),
    }
    shape = left.shape
    paths["disp"] = _write_disparity(
        tmp_path / "pair-F.tif",
        np.full(shape, DX),
        np.full(shape, DY),
        np.ones(shape, bool),
    )
    return paths


class TestAfterCorrelation:
    def test_read_disparity(self, shifted_pair):
        dx, dy, valid, grid = sr.read_disparity(shifted_pair["disp"])
        assert grid.key() == sr.Grid.of(shifted_pair["left"]).key()
        assert np.allclose(dx, DX) and np.allclose(dy, DY) and valid.all()

    def test_alignment_recovers_the_left_image(self, shifted_pair):
        dx, dy, valid, grid = sr.read_disparity(shifted_pair["disp"])
        before = sr.compare_rasters(shifted_pair["left"], shifted_pair["right"], **KW)
        after = sr.compare_after_correlation(
            shifted_pair["left"], shifted_pair["right"], dx, dy, valid, grid=grid, **KW
        )
        idx = after["indices"]
        assert tuple(idx) == sr.REPORT_KEYS
        assert idx["zncc_local_median"] > 0.95 > before["zncc_local_median"]
        assert abs(idx["trial_dx_median_px"]) < 0.1
        assert abs(idx["trial_dy_median_px"]) < 0.1
        assert abs(before["trial_dx_median_px"] - DX) < 0.2  # the shift was there
        assert after["disp_valid_frac"] == pytest.approx(1.0)

    def test_apply_disparity_holes(self):
        b = np.arange(25, dtype=float).reshape(5, 5)
        b[2, 2] = np.nan
        zero = np.zeros_like(b)
        valid = np.ones(b.shape, bool)
        valid[0, 0] = False
        out = sr.apply_disparity(b, zero + 0.5, zero, valid)
        assert np.isnan(out[0, 0])  # invalid disparity
        assert np.isnan(out[2, 1]) and np.isnan(out[2, 2])  # touches the hole
        assert np.isnan(out[1, 4])  # samples past the edge
        assert out[1, 1] == pytest.approx(6.5)

    def test_shape_mismatch_raises(self, shifted_pair):
        grid = sr.Grid.of(shifted_pair["left"])
        small = np.zeros((10, 10))
        with pytest.raises(ValueError, match="disparity"):
            sr.compare_after_correlation(
                shifted_pair["left"],
                shifted_pair["right"],
                small,
                small,
                small > -1,
                grid=grid,
            )

    def test_displacement_to_disparity_sign_and_grid(self, tmp_path):
        """EW/NS in metres, NS north-positive, on a 2x coarser grid shifted by
        half a coarse pixel: back to (DX, DY) left-image pixels on the fine grid."""
        fine = sr.Grid.of(write_tif(tmp_path / "fine.tif", np.ones((64, 64)), 1.5))
        origin = (ORIGIN[0] - 1.5, ORIGIN[1] + 1.5)
        ew = np.full((34, 34), DX * 1.5)
        ns = np.full((34, 34), -DY * 1.5)  # north-positive
        ew[:6, :6] = 0.0  # the corrected maps' no-data
        kw = dict(res=3.0, origin=origin, dtype="float32")
        write_tif(tmp_path / "ew.tif", ew, **kw)
        write_tif(tmp_path / "ns.tif", ns, **kw)
        dx, dy, valid = sr.displacement_to_disparity(
            tmp_path / "ew.tif", tmp_path / "ns.tif", fine
        )
        assert dx.shape == fine.shape
        assert np.allclose(dx[valid], DX) and np.allclose(dy[valid], DY)
        assert not valid[0, 0] and np.isnan(dx[0, 0])  # 0 is no-data, not 0 m
        assert valid[30:, 30:].all()

    def test_masked_stats(self):
        arr = np.array([[0.2, 0.6], [0.9, np.nan]])
        keep = np.array([[True, True], [False, True]])
        st = sr.masked_stats(arr, keep)
        assert st["n"] == 2 and st["median"] == pytest.approx(0.4)
        assert st["frac_gte_050"] == pytest.approx(0.5)
        empty = sr.masked_stats(arr, np.zeros_like(keep))
        assert empty["n"] == 0 and np.isnan(empty["median"])
