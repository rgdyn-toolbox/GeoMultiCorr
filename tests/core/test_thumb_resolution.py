#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The ``th_res`` column on the Thumbs layer.

Resolution is read from each file's **header**, not from ``sieve_bulk``'s
``target_resolution`` argument: that argument is never persisted, and
``register_existing_thumbs`` bypasses the sieve entirely, so a value recorded
from it would be absent or wrong for those thumbs. The header is what ASP will
actually see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio

from rasterio.transform import from_origin

from geomulticorr.core.thumb import Thumb


def _write_raster(path, res: float = 3.0, size: int = 8):
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="uint16", crs="EPSG:32717",
        transform=from_origin(500000, 9800000, res, res),
    ) as dst:
        dst.write(np.ones((size, size), dtype="uint16"), 1)
    return path


@pytest.fixture
def thumb_file(tmp_path):
    return _write_raster(tmp_path / "Chimborazo_2020-01-01_planetscope.tif", res=3.0)


class TestThumbResolution:
    def test_reads_the_header(self, thumb_file):
        assert Thumb(thumb_file).th_res == pytest.approx(3.0)

    def test_is_a_real_float(self, thumb_file):
        """A mixed dtype would quietly turn the GPKG column to object."""
        assert isinstance(Thumb(thumb_file).th_res, float)

    def test_reflects_the_actual_file_not_a_nominal_sensor_value(self, tmp_path):
        """A resampled thumb must report what it really is."""
        path = _write_raster(tmp_path / "Chimborazo_2020-02-01_planetscope.tif", res=1.5)
        assert Thumb(path).th_res == pytest.approx(1.5)

    def test_positive_even_though_north_up_rasters_have_negative_y_scale(self, thumb_file):
        assert Thumb(thumb_file).th_res > 0

    def test_unreadable_resolution_degrades_to_nan(self, thumb_file):
        """One odd header must not block registering a whole project."""
        class _NoRes:
            res = ()

        assert np.isnan(Thumb._read_resolution(_NoRes()))

    @pytest.mark.parametrize("broken", [object(), None])
    def test_read_resolution_never_raises(self, broken):
        assert np.isnan(Thumb._read_resolution(broken))


class TestToPdserie:
    def test_carries_th_res(self, thumb_file):
        assert Thumb(thumb_file).to_pdserie()["th_res"] == pytest.approx(3.0)

    def test_th_res_is_float_in_the_serie(self, thumb_file):
        assert isinstance(Thumb(thumb_file).to_pdserie()["th_res"], float)

    def test_schema_is_otherwise_unchanged(self, thumb_file):
        """Adding a column must not disturb the existing nine."""
        keys = set(Thumb(thumb_file).to_pdserie().index)
        assert keys == {
            "th_pz_name", "th_path", "th_sensor", "th_date", "th_year",
            "th_date_dec", "th_date_datetime", "th_valid", "th_res", "geometry",
        }


class TestLayerRoundTrip:
    """``update_thumbs`` keeps existing rows in full and appends new ones, so a
    concatenation of both must not produce an object column."""

    def test_concat_of_old_and_new_rows_stays_float(self, thumb_file):
        new = pd.DataFrame([Thumb(thumb_file).to_pdserie()])
        # a row as it comes back from the GPKG: float64 already
        old = pd.DataFrame({"th_path": ["/other.tif"], "th_res": [1.5]})
        merged = pd.concat([old, new], ignore_index=True)
        assert pd.api.types.is_float_dtype(merged["th_res"])

    def test_missing_column_on_old_rows_becomes_nan_not_string(self, thumb_file):
        new = pd.DataFrame([Thumb(thumb_file).to_pdserie()])
        old = pd.DataFrame({"th_path": ["/other.tif"]})  # pre-th_res layer
        merged = pd.concat([old, new], ignore_index=True)
        assert pd.api.types.is_float_dtype(merged["th_res"])
        assert np.isnan(merged["th_res"].iloc[0])
