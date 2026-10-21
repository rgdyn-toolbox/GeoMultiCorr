#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``_stack``: TOT rasters ↔ one ``(time, y, x)`` dataset, and the shared discovery.

The writer must leave files a solver could have written — same names, same
profile — so ``InversionExtractor`` reads a fused product like any other. The
loader must refuse rasters that are not on one grid rather than resample.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import xarray as xr

from geomulticorr.inversion import load_cumulative_stack, write_cumulative_stack
from geomulticorr.inversion._stack import COMPONENTS, magnitude, tot_tif_name
from geomulticorr.inversion.tio_inversion import TIOInversion
from geomulticorr.stats.inversion_extractor import (InversionExtractor,
                                                    discover_tot_rasters,
                                                    resolve_inversion_dir)
from tests.inversion.conftest import write_tot_stack, years_since

DATES = ["20200101", "20201201", "20220201", "20221201", "20240101"]


def _linear(v):
    return lambda t, yy, xx: v * (1 + 0.1 * (yy * 5 + xx)) * t


@pytest.fixture
def inv_dir(tmp_path):
    write_tot_stack(tmp_path, DATES, "EW", _linear(0.5))
    write_tot_stack(tmp_path, DATES, "NS", _linear(-0.3))
    return tmp_path


class TestLoad:
    def test_dims_coords_and_values(self, inv_dir):
        ds = load_cumulative_stack(inv_dir)
        assert set(ds.data_vars) == set(COMPONENTS)
        assert ds["EW"].dims == ("time", "y", "x")
        assert ds["EW"].shape == (5, 6, 5)
        assert ds["EW"].dtype == np.float32
        assert str(ds["time"].dtype).startswith("datetime64")
        assert list(ds["date"].values) == DATES
        assert ds.rio.crs.to_epsg() == 32632
        t = years_since(DATES)
        yy, xx = np.meshgrid(np.arange(6), np.arange(5), indexing="ij")
        expected = np.stack([_linear(0.5)(tk, yy, xx) for tk in t])
        np.testing.assert_allclose(ds["EW"].values, expected, rtol=1e-6)
        assert ds.attrs["inversion_dir"] == str(inv_dir)

    def test_accepts_an_object_with_inversion_dir(self, inv_dir):
        ds = load_cumulative_stack(SimpleNamespace(inversion_dir=inv_dir))
        assert ds["NS"].shape == (5, 6, 5)

    def test_lazy_with_dask(self, inv_dir):
        da = pytest.importorskip("dask.array")
        ds = load_cumulative_stack(inv_dir)
        assert isinstance(ds["EW"].data, da.Array)

    def test_eager_when_asked(self, inv_dir):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        assert isinstance(ds["EW"].data, np.ndarray)

    def test_nan_nodata_round_trips(self, tmp_path):
        def fn(t, yy, xx):
            data = np.full(yy.shape, 1.0 + t, np.float32)
            data[2, 3] = np.nan
            return data
        write_tot_stack(tmp_path, DATES[:3], "EW", fn)
        ds = load_cumulative_stack(tmp_path, components=("EW",), chunks=None)
        assert np.isnan(ds["EW"].values[:, 2, 3]).all()
        assert np.isfinite(ds["EW"].values[:, 0, 0]).all()

    def test_missing_component_raises_with_the_hint(self, tmp_path):
        write_tot_stack(tmp_path, DATES, "EW", _linear(1.0))
        with pytest.raises(FileNotFoundError, match="launch\\(mode='python'\\)"):
            load_cumulative_stack(tmp_path)

    def test_dates_are_intersected_across_components(self, tmp_path, caplog_gmc):
        write_tot_stack(tmp_path, DATES, "EW", _linear(1.0))
        write_tot_stack(tmp_path, DATES[:4], "NS", _linear(1.0))
        ds = load_cumulative_stack(tmp_path)
        assert list(ds["date"].values) == DATES[:4]
        assert "dropped" in caplog_gmc.text and DATES[4] in caplog_gmc.text

    def test_grid_mismatch_raises(self, tmp_path):
        write_tot_stack(tmp_path, DATES[:2], "EW", _linear(1.0))
        write_tot_stack(tmp_path, DATES[:2], "NS", _linear(1.0), origin=(500003.0, 4649000.0))
        with pytest.raises(ValueError, match="grid"):
            load_cumulative_stack(tmp_path)

    def test_fortran_style_short_raster_does_not_align_with_full_height(self, tmp_path):
        write_tot_stack(tmp_path, DATES[:2], "EW", _linear(1.0), shape=(6, 5))
        write_tot_stack(tmp_path, DATES[:2], "NS", _linear(1.0), shape=(5, 5))
        with pytest.raises(ValueError):
            load_cumulative_stack(tmp_path)


class TestMagnitude:
    def test_hypot_and_nan_propagation(self, inv_dir):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        m = magnitude(ds)
        np.testing.assert_allclose(m.values, np.hypot(ds["EW"].values, ds["NS"].values))
        ds["EW"].values[1, 0, 0] = np.nan
        assert np.isnan(magnitude(ds).values[1, 0, 0])

    def test_needs_both(self, inv_dir):
        ds = load_cumulative_stack(inv_dir, components=("EW",), chunks=None)
        with pytest.raises(KeyError):
            magnitude(ds)


class TestWrite:
    def test_round_trip_and_layout(self, inv_dir, tmp_path):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        out = tmp_path / "fused"
        written = write_cumulative_stack(ds, out)
        assert set(written) == {"EW", "NS", "magn"}
        assert written == discover_tot_rasters(out)
        assert written["EW"][DATES[0]] == tot_tif_name(out / "inverse_EW", DATES[0], "EW")
        back = load_cumulative_stack(out, chunks=None)
        xr.testing.assert_allclose(back["EW"], ds["EW"])
        xr.testing.assert_allclose(back["NS"], ds["NS"])
        assert back.rio.crs == ds.rio.crs
        assert back["EW"].rio.transform() == ds["EW"].rio.transform()

    def test_files_have_the_solver_profile(self, inv_dir, tmp_path):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        written = write_cumulative_stack(ds, tmp_path / "out")
        with rasterio.open(written["EW"][DATES[1]]) as src:
            assert src.dtypes == ("float32",)
            assert np.isnan(src.nodata)
            assert src.profile["compress"].lower() == "lzw"
            assert src.crs.to_epsg() == 32632

    def test_magnitude_is_derived_from_the_written_components(self, inv_dir, tmp_path):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        written = write_cumulative_stack(ds, tmp_path / "out")
        with rasterio.open(written["magn"][DATES[2]]) as src:
            data = src.read(1)
        np.testing.assert_allclose(
            data, np.hypot(ds["EW"].values[2], ds["NS"].values[2]), rtol=1e-6)

    def test_a_magn_variable_is_never_written_as_given(self, inv_dir, tmp_path, caplog_gmc):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        ds["magn"] = ds["EW"] * 0 + 42.0
        written = write_cumulative_stack(ds, tmp_path / "out", components=("EW", "NS", "magn"))
        with rasterio.open(written["magn"][DATES[1]]) as src:
            assert not np.allclose(src.read(1), 42.0)
        assert "derived from EW/NS" in caplog_gmc.text

    def test_no_magnitude_when_only_one_component(self, inv_dir, tmp_path):
        ds = load_cumulative_stack(inv_dir, components=("EW",), chunks=None)
        written = write_cumulative_stack(ds, tmp_path / "out")
        assert set(written) == {"EW"}
        assert not (tmp_path / "out" / "inverse_magn").exists()

    def test_overwrite_false_keeps_existing(self, inv_dir, tmp_path):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        out = tmp_path / "out"
        written = write_cumulative_stack(ds, out)
        path = written["EW"][DATES[0]]
        stamp = path.stat().st_mtime_ns
        write_cumulative_stack(ds, out, overwrite=False)
        assert path.stat().st_mtime_ns == stamp

    def test_fit_and_parameters_land_beside_the_rasters(self, inv_dir, tmp_path):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        fit = xr.Dataset({
            v: (("component", "y", "x"), np.ones((2, 6, 5), np.float32) * k)
            for k, v in enumerate(("offset", "rate", "rms", "n_valid"))
        }, coords={"component": ["EW", "NS"], "y": ds["y"], "x": ds["x"]})
        fit = fit.rio.write_crs(ds.rio.crs)
        out = tmp_path / "out"
        write_cumulative_stack(ds, out, fit=fit, parameters={"kind": "fusion", "n": 1})
        for comp in ("EW", "NS"):
            for var in ("offset", "rate", "rms", "n_valid"):
                assert (out / "calibration" / f"{comp}_{var}.tif").exists()
        with rasterio.open(out / "calibration" / "NS_rate.tif") as src:
            np.testing.assert_allclose(src.read(1), 1.0)
        import json
        assert json.loads((out / "fusion_parameters.json").read_text())["kind"] == "fusion"

    def test_trace_failure_is_logged_not_raised(self, inv_dir, tmp_path, monkeypatch, caplog_gmc):
        monkeypatch.setattr(
            "geomulticorr.inversion._run_parameters.write_run_parameters",
            lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
        ds = load_cumulative_stack(inv_dir, chunks=None)
        written = write_cumulative_stack(ds, tmp_path / "out", parameters={"kind": "fusion"})
        assert written["EW"][DATES[0]].exists()
        assert "could not write the trace" in caplog_gmc.text

    def test_extractor_reads_the_product(self, inv_dir, tmp_path):
        ds = load_cumulative_stack(inv_dir, chunks=None)
        out = tmp_path / "out"
        write_cumulative_stack(ds, out)
        ex = InversionExtractor(out)
        assert ex.available_dates == DATES
        assert set(ex.available_components) == {"EW", "NS", "magn"}


class TestSharedDiscovery:
    def test_resolve_inversion_dir(self, tmp_path):
        assert resolve_inversion_dir(tmp_path) == tmp_path
        assert resolve_inversion_dir(str(tmp_path)) == tmp_path
        assert resolve_inversion_dir(SimpleNamespace(inversion_dir=tmp_path)) == tmp_path
        with pytest.raises(TypeError):
            resolve_inversion_dir(42)

    def test_discover_matches_the_extractor(self, inv_dir):
        ex = InversionExtractor(inv_dir)
        assert ex.tif_paths == discover_tot_rasters(inv_dir)
        assert ex.inversion_dir == inv_dir

    def test_tot_tif_name_is_the_method(self, tmp_path):
        assert TIOInversion._tot_tif_name(tmp_path, "20200101", "magn") == \
            tot_tif_name(tmp_path, "20200101", "magn")

    def test_to_xarray_on_a_bare_instance(self, inv_dir):
        inv = TIOInversion.__new__(TIOInversion)
        inv.inversion_dir = inv_dir
        inv.inversion_name = "t"
        inv.pzone_name = "PZ"
        ds = inv.to_xarray(chunks=None)
        assert ds.attrs["inversion_name"] == "t"
        assert ds.attrs["pzone"] == "PZ"
        assert ds["EW"].shape == (5, 6, 5)
