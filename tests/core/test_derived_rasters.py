#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_derived_rasters.py
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
"""Derived rasters of a corrected pair: magnitude and velocities."""

from __future__ import annotations

import types
import typing

import numpy as np
import numpy.ma as ma
import pytest
import rasterio

from geomulticorr.cli.config import CorrectConfig
from geomulticorr.core.pair import Pair
from geomulticorr.core.session import Session
from geomulticorr.utils import _derived

DT = 2.0  # years
EW = np.array([[3.0, 0.0, -6.0], [1.0, 2.0, 0.0]])  # 0 = nodata
NS = np.array([[4.0, 5.0, 8.0], [0.0, 2.0, 0.0]])


def _write(path, data, nodata=0.0, dtype="float32"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:32632",
        transform=rasterio.transform.from_origin(0, 20, 10, 10),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)


def _pair(tmp_path, with_inputs=True):
    pair = Pair.__new__(Pair)
    pair.pa_key = "Z_2020-01-01_s__2022-01-01_s"
    pair.pa_path = tmp_path
    pair.pa_dt_years = DT
    pair.pa_ew_corr_path = tmp_path / f"{pair.pa_key}-F_EW_corr.tif"
    pair.pa_ns_corr_path = tmp_path / f"{pair.pa_key}-F_NS_corr.tif"
    if with_inputs:
        _write(pair.pa_ew_corr_path, EW)
        _write(pair.pa_ns_corr_path, NS)
    return pair


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.nodata, src.dtypes[0], src.transform, src.crs


class TestFormulas:
    def test_magnitude_and_velocity(self):
        assert _derived.magnitude(3.0, 4.0) == 5.0
        assert _derived.velocity(10.0, 2.0) == 5.0

    def test_check_products(self):
        assert _derived.check_products("velocity") == ("velocity",)
        assert _derived.check_products(["velocity", "magnitude", "velocity"]) == (
            "velocity",
            "magnitude",
        )
        with pytest.raises(ValueError, match="Unknown derived product"):
            _derived.check_products(["velocity", "speed"])
        with pytest.raises(ValueError):
            _derived.check_products([])

    def test_cli_literal_matches_the_products(self):
        field = CorrectConfig.model_fields["derived"].annotation
        (literal,) = typing.get_args(field)
        assert set(typing.get_args(literal)) == set(_derived.DERIVED_PRODUCTS)

    def test_get_interesting_geoim_uses_the_same_formula(self):
        """m / vm on raw maps are the shared functions: same arithmetic, same numbers."""
        x = ma.array([[3.0, 6.0]])
        y = ma.array([[4.0, 8.0]])
        assert np.array_equal(
            _derived.velocity(_derived.magnitude(x, y), 2.0),
            ((x * x + y * y) ** 0.5) / 2.0,
        )


class TestPaths:
    def test_names(self, tmp_path):
        p = _pair(tmp_path, with_inputs=False)
        assert p.derived_path("magnitude").name.endswith("-F_MAG_corr.tif")
        assert p.derived_path("velocity").name.endswith("-F_VEL_corr.tif")
        assert p.derived_path("ew_velocity").name.endswith("-F_EW_VEL_corr.tif")
        assert p.derived_path("ns_velocity").name.endswith("-F_NS_VEL_corr.tif")
        assert p.derived_path("magnitude").parent == tmp_path

    def test_unknown_product(self, tmp_path):
        with pytest.raises(ValueError):
            _pair(tmp_path, with_inputs=False).derived_path("nope")


class TestWriteDerived:
    PRODUCTS = ("magnitude", "velocity", "ew_velocity", "ns_velocity")

    def test_values(self, tmp_path):
        pair = _pair(tmp_path)
        written = pair.write_derived(self.PRODUCTS)
        assert set(written) == set(self.PRODUCTS)
        mag, nodata, dtype, _, _ = _read(written["magnitude"])
        assert nodata == _derived.DERIVED_NODATA and dtype == "float32"
        assert mag[0, 0] == 5.0  # sqrt(3² + 4²)
        assert mag[1, 1] == pytest.approx(np.hypot(2, 2))
        vel = _read(written["velocity"])[0]
        assert vel[0, 0] == 2.5 and vel[1, 1] == pytest.approx(np.hypot(2, 2) / DT)
        assert _read(written["ew_velocity"])[0][0, 0] == 1.5
        assert _read(written["ns_velocity"])[0][0, 0] == 2.0

    def test_nodata_propagation(self, tmp_path):
        """A pixel is nodata when an input it needs is nodata; a velocity of 0 is kept."""
        pair = _pair(tmp_path)
        pair.write_derived(self.PRODUCTS)
        nd = _derived.DERIVED_NODATA
        # (0, 1): EW nodata, NS = 5. (1, 0): EW = 1, NS nodata. (1, 2): both nodata
        mag = _read(pair.pa_mag_corr_path)[0]
        assert mag[0, 1] == nd and mag[1, 0] == nd and mag[1, 2] == nd
        ew_vel = _read(pair.pa_ew_vel_corr_path)[0]
        ns_vel = _read(pair.pa_ns_vel_corr_path)[0]
        assert ew_vel[0, 1] == nd and ew_vel[1, 0] == 0.5  # NS nodata does not hit EW
        assert ns_vel[1, 0] == nd and ns_vel[0, 1] == 2.5  # EW nodata does not hit NS

    def test_geo_reference_is_kept(self, tmp_path):
        pair = _pair(tmp_path)
        pair.write_derived("velocity")
        _, _, _, transform, crs = _read(pair.pa_vel_corr_path)
        assert transform == rasterio.transform.from_origin(0, 20, 10, 10)
        assert crs.to_epsg() == 32632

    def test_existing_files_are_skipped_unless_overwrite(self, tmp_path):
        pair = _pair(tmp_path)
        assert set(pair.write_derived(["magnitude"])) == {"magnitude"}
        stamp = pair.pa_mag_corr_path.stat().st_mtime_ns
        assert pair.write_derived(["magnitude"]) == {}  # nothing to do
        assert pair.pa_mag_corr_path.stat().st_mtime_ns == stamp
        # a new product is still written, the existing one is left alone
        assert set(pair.write_derived(["magnitude", "velocity"])) == {"velocity"}
        assert set(pair.write_derived(["magnitude"], overwrite=True)) == {"magnitude"}

    def test_missing_corrected_maps(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _pair(tmp_path, with_inputs=False).write_derived("magnitude")

    def test_non_positive_dt(self, tmp_path):
        pair = _pair(tmp_path)
        pair.pa_dt_years = 0.0
        with pytest.raises(ValueError, match="time span"):
            pair.write_derived("velocity")

    def test_grid_mismatch(self, tmp_path):
        pair = _pair(tmp_path)
        _write(pair.pa_ns_corr_path, np.ones((3, 3)))
        with pytest.raises(ValueError, match="same grid"):
            pair.write_derived("magnitude")

    def test_nan_in_the_input_is_nodata(self, tmp_path):
        pair = _pair(tmp_path, with_inputs=False)
        ew = EW.copy()
        ew[0, 0] = np.nan
        _write(pair.pa_ew_corr_path, ew, nodata=None)
        _write(pair.pa_ns_corr_path, NS, nodata=None)
        pair.write_derived("magnitude")
        mag = _read(pair.pa_mag_corr_path)[0]
        assert mag[0, 0] == _derived.DERIVED_NODATA
        assert np.isfinite(mag).all()


class TestSessionMethod:
    @staticmethod
    def _stub(pairs):
        return types.SimpleNamespace(get_pairs=lambda *a, **k: pairs)

    def test_writes_for_every_pair_and_reports(self, tmp_path):
        dirs = [tmp_path / n for n in "abc"]
        for d in dirs:
            d.mkdir()
        pairs = [_pair(dirs[0]), _pair(dirs[1]), _pair(dirs[2], with_inputs=False)]
        for pair, key in zip(pairs, ("a", "b", "c")):
            pair.pa_key = key
            pair.pa_ew_corr_path = dirs[ord(key) - 97] / f"{key}-F_EW_corr.tif"
            pair.pa_ns_corr_path = dirs[ord(key) - 97] / f"{key}-F_NS_corr.tif"
        _write(pairs[0].pa_ew_corr_path, EW)
        _write(pairs[0].pa_ns_corr_path, NS)
        _write(pairs[1].pa_ew_corr_path, EW)
        _write(pairs[1].pa_ns_corr_path, NS)
        out = Session.write_derived_rasters(
            self._stub(pairs),
            products=("magnitude", "ew_velocity"),
            print_summary=False,
        )
        assert set(out["a"]) == {"magnitude", "ew_velocity"}
        assert set(out["b"]) == {"magnitude", "ew_velocity"}
        assert out["c"] is None
        assert pairs[0].pa_mag_corr_path.exists() and pairs[1].pa_mag_corr_path.exists()
        assert not pairs[2].pa_mag_corr_path.exists()

    def test_pair_without_corrected_maps_is_none(self, tmp_path):
        pair = _pair(tmp_path, with_inputs=False)
        out = Session.write_derived_rasters(self._stub([pair]), print_summary=False)
        assert out == {pair.pa_key: None}

    def test_a_failing_pair_does_not_stop_the_others(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        bad, good = _pair(a), _pair(b)
        bad.pa_key = "bad"
        good.pa_key = "good"
        bad.pa_dt_years = 0.0
        out = Session.write_derived_rasters(
            self._stub([bad, good]), products="velocity", print_summary=False
        )
        assert out["bad"] is None and set(out["good"]) == {"velocity"}

    def test_overwrite_flag(self, tmp_path):
        pair = _pair(tmp_path)
        Session.write_derived_rasters(self._stub([pair]), print_summary=False)
        again = Session.write_derived_rasters(self._stub([pair]), print_summary=False)
        assert again[pair.pa_key] == {}
        redo = Session.write_derived_rasters(
            self._stub([pair]), overwrite=True, print_summary=False
        )
        assert set(redo[pair.pa_key]) == {"magnitude", "velocity"}

    def test_unknown_product_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown derived product"):
            Session.write_derived_rasters(
                self._stub([]), products=["speed"], print_summary=False
            )
