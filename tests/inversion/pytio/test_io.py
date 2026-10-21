#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Readers/writers of the TIO flat-binary and list-file formats.

The ``depl_cumule`` layout is a contract with the Fortran post-processor
(float32 LE, pixel-interleaved, ``9999`` no-data), so the round trip is
asserted at the byte level, not only through the reader.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geomulticorr.inversion.pytio import NODATA_OUT, io


class TestListFiles:
    def test_read_pair_list_with_weights(self, tmp_path):
        p = tmp_path / "liste_couple"
        p.write_text("20200101 20200301 0.750000\n20200301 20200101 1.000000\n\n")
        d1, d2, w = io.read_pair_list(p)
        assert list(d1) == ["20200101", "20200301"]
        assert list(d2) == ["20200301", "20200101"]
        np.testing.assert_allclose(w, [0.75, 1.0])

    def test_read_pair_list_without_weights_defaults_to_one(self, tmp_path):
        p = tmp_path / "liste_couple"
        p.write_text("20200101 20200301\n")
        _, _, w = io.read_pair_list(p)
        np.testing.assert_allclose(w, [1.0])

    def test_read_image_list_four_columns(self, tmp_path):
        p = tmp_path / "liste_image_inv"
        p.write_text("20200101 2020.0000 0.0000 0\n20210101 2021.0000 1.0000 0\n")
        imgs = io.read_image_list(p)
        assert list(imgs["dates"]) == ["20200101", "20210101"]
        np.testing.assert_allclose(imgs["t"], [2020.0, 2021.0])
        np.testing.assert_allclose(imgs["elapsed"], [0.0, 1.0])
        np.testing.assert_allclose(imgs["quality"], [1.0, 1.0])


class TestDeplCumule:
    def test_round_trip_and_byte_layout(self, tmp_path):
        rng = np.random.default_rng(0)
        cube = rng.normal(size=(3, 4, 5)).astype(np.float32)     # (date, y, x)
        cube[1, 2, 3] = np.nan
        da = xr.DataArray(cube, dims=("date", "y", "x"),
                          coords={"date": ["20200101", "20210101", "20220101"]})
        path = tmp_path / "depl_cumule"
        io.write_depl_cumule(da, str(path))

        raw = np.fromfile(path, dtype="<f4")
        assert raw.size == 3 * 4 * 5
        # pixel-interleaved: the three dates of pixel (0, 0) come first
        np.testing.assert_allclose(raw[:3], cube[:, 0, 0])
        assert raw[(2 * 5 + 3) * 3 + 1] == NODATA_OUT
        assert (tmp_path / "depl_cumule.hdr").exists()

        back = io.read_depl_cumule(str(path), width=5, length=4, n_dates=3,
                                   dates=da["date"].values)
        assert back.dims == ("date", "y", "x")
        assert np.isnan(back.values[1, 2, 3])
        mask = np.isfinite(cube)
        np.testing.assert_allclose(back.values[mask], cube[mask])

    def test_reader_tolerates_a_missing_last_row(self, tmp_path):
        cube = np.ones((2, 4, 3), np.float32)
        path = tmp_path / "depl_cumule"
        io.write_depl_cumule(cube, str(path))
        back = io.read_depl_cumule(str(path), width=3, length=5, n_dates=2)
        assert back.shape == (2, 4, 3)


class TestPairStack:
    def _write_pairs(self, tmp_path, shape=(4, 3)):
        d1 = ["20200101", "20200101"]
        d2 = ["20210101", "20220101"]
        arrays = []
        for a, b in zip(d1, d2):
            arr = np.full(shape, float(b[:4]) - 2020.0, np.float32)
            io.write_r4(arr, str(tmp_path / f"{a}_{b}_EW"))
            arrays.append(arr)
        return d1, d2, np.stack(arrays)

    def test_load_with_explicit_shape(self, tmp_path):
        d1, d2, expected = self._write_pairs(tmp_path)
        stack = io.load_pair_stack(str(tmp_path), d1, d2, template="{d1}_{d2}_EW", shape=(4, 3))
        assert stack.dims == ("pair", "y", "x")
        assert list(stack["date2"].values) == d2
        np.testing.assert_allclose(np.asarray(stack), expected)

    def test_shape_is_read_from_the_envi_header(self, tmp_path):
        d1, d2, expected = self._write_pairs(tmp_path)
        stack = io.load_pair_stack(str(tmp_path), d1, d2, template="{d1}_{d2}_EW")
        assert stack.shape == (2, 4, 3)

    def test_lazy_when_dask_is_available(self, tmp_path):
        da = pytest.importorskip("dask.array")
        d1, d2, _ = self._write_pairs(tmp_path)
        stack = io.load_pair_stack(str(tmp_path), d1, d2, template="{d1}_{d2}_EW", shape=(4, 3))
        assert isinstance(stack.data, da.Array)

    def test_guess_shape_without_sidecar_raises(self, tmp_path):
        (tmp_path / "orphan").write_bytes(b"\0" * 16)
        with pytest.raises(FileNotFoundError):
            io.guess_shape(str(tmp_path / "orphan"))
