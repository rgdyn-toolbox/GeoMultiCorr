#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``fusion``: calibrating one sensor's series to another, fusing, and the closure check.

Everything is synthetic: a reference series with a known per-pixel velocity,
a secondary series with a known per-pixel offset and drift on top of it. The
calibration must recover both to numerical precision, the fusion must let
the reference win on shared dates, and the closure diagnostic must read a
pair's rate bias straight off a residual map.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from geomulticorr.inversion import (calibrate_to_reference, closure_summary,
                                    fuse_inversions, fuse_series,
                                    load_cumulative_stack,
                                    pair_closure_against_series)
from geomulticorr.inversion.fusion import build_fusion_parameters
from geomulticorr.stats.inversion_extractor import InversionExtractor
from tests.inversion.conftest import write_tot_stack

H, W = 6, 5
REF_DATES = ["20160101", "20180101", "20200101", "20220101", "20240101", "20250101"]
SEC_DATES = ["20180701", "20200701", "20210701", "20230701"]
CRS = "EPSG:32632"


def _grid():
    return np.meshgrid(np.arange(H), np.arange(W), indexing="ij")


def _v(comp):
    yy, xx = _grid()
    return (0.5 if comp == "EW" else -0.3) * (1 + 0.1 * (yy * W + xx))


def _a(comp):
    yy, xx = _grid()
    return (0.3 if comp == "EW" else -0.2) + 0.01 * (yy - xx)


def _b(comp):
    yy, xx = _grid()
    return (-0.15 if comp == "EW" else 0.08) * (1 + 0.05 * xx)


def _years(dates, t0):
    return ((pd.to_datetime([str(d) for d in dates], format="%Y%m%d") - pd.Timestamp(t0))
            / pd.Timedelta(days=365.25)).to_numpy()


def _series(dates, fn, *, t0="20160101", crs=CRS, origin=(500000.0, 4649000.0)):
    """A cumulative-stack dataset in memory: EW/NS on (time, y, x), georeferenced."""
    t = _years(dates, t0)
    data = {}
    for comp in ("EW", "NS"):
        arr = np.stack([np.broadcast_to(fn(comp, tk), (H, W)) for tk in t]).astype("float32")
        data[comp] = (("time", "y", "x"), arr)
    x = origin[0] + 3.0 * (np.arange(W) + 0.5)
    y = origin[1] - 3.0 * (np.arange(H) + 0.5)
    ds = xr.Dataset(data, coords={
        "time": pd.to_datetime(list(dates), format="%Y%m%d").values,
        "date": ("time", list(dates)), "y": y, "x": x})
    return ds.rio.write_crs(crs)


def reference():
    return _series(REF_DATES, lambda c, t: _v(c) * t)


def secondary_biased(t0_sec="20180701"):
    """reference + a + b·(t − t0_sec) on the secondary dates (t0_sec is the first overlap epoch)."""
    def fn(c, t):
        t_sec = t - _years([t0_sec], "20160101")[0]
        return _v(c) * t + _a(c) + _b(c) * t_sec
    return _series(SEC_DATES, fn)


class TestCalibration:
    def test_recovers_the_bias_to_precision(self):
        ref, sec = reference(), secondary_biased()
        corrected, fit = calibrate_to_reference(sec, ref)
        for comp in ("EW", "NS"):
            np.testing.assert_allclose(fit["rate"].sel(component=comp).values, _b(comp), atol=1e-4)
            np.testing.assert_allclose(fit["offset"].sel(component=comp).values, _a(comp), atol=1e-4)
            assert float(fit["rms"].sel(component=comp).max()) < 1e-4
            assert (fit["n_valid"].sel(component=comp) == len(SEC_DATES)).all()
            expected = ref[comp].interp(time=sec["time"])
            np.testing.assert_allclose(corrected[comp].values, expected.values, atol=1e-4)
        assert fit.attrs["model"] == "offset+rate"
        assert fit.attrs["overlap_dates"] == SEC_DATES
        assert fit.attrs["n_overlap"] == 4
        assert corrected.attrs["calibration"]["extrapolated"] is True
        assert fit["offset"].dims == ("component", "y", "x")

    def test_correction_is_applied_on_the_full_axis(self):
        """Secondary epochs outside the reference span get the extrapolated model."""
        ref = reference().sel(time=slice("2019-01-01", "2023-12-31"))   # 2020, 2022 only
        sec = secondary_biased()
        corrected, fit = calibrate_to_reference(sec, ref)
        assert fit.attrs["overlap_dates"] == ["20200701", "20210701"]
        # 2018-07 and 2023-07 lie outside the (2020, 2022) span but are corrected
        truth = reference()
        for comp in ("EW", "NS"):
            np.testing.assert_allclose(corrected[comp].values,
                                       truth[comp].interp(time=sec["time"]).values, atol=1e-3)

    def test_offset_only_leaves_the_drift(self):
        ref, sec = reference(), secondary_biased()
        corrected, fit = calibrate_to_reference(sec, ref, model="offset")
        assert (fit["rate"] == 0).all()
        assert float(fit["rms"].sel(component="EW").max()) > 0.05
        residual = corrected["EW"].values - ref["EW"].interp(time=sec["time"]).values
        assert np.abs(residual).max() > 0.05

    def test_nan_handling_and_fallbacks(self, caplog_gmc):
        ref, sec = reference(), secondary_biased()
        ref["EW"].values[:, 0, 0] = np.nan                  # nothing to calibrate against
        sec["EW"].values[1:, 1, 1] = np.nan                 # a single overlap epoch
        with warnings.catch_warnings():
            warnings.simplefilter("error")                  # no RankWarning may escape
            corrected, fit = calibrate_to_reference(sec, ref)
        ew = fit.sel(component="EW")
        assert int(ew["n_valid"][0, 0]) == 0
        assert np.isnan(float(ew["offset"][0, 0])) and np.isnan(float(ew["rate"][0, 0]))
        assert np.isnan(corrected["EW"].values[:, 0, 0]).all()
        assert int(ew["n_valid"][1, 1]) == 1
        assert float(ew["rate"][1, 1]) == 0.0
        d = float(sec["EW"].values[0, 1, 1] - ref["EW"].interp(time=sec["time"][0]).values[1, 1])
        assert float(ew["offset"][1, 1]) == pytest.approx(d, abs=1e-5)
        assert fit.attrs["n_fallback_pixels"] == 1
        assert "single overlap epoch" in caplog_gmc.text
        # the NS component and the other pixels are untouched
        np.testing.assert_allclose(fit["rate"].sel(component="NS").values, _b("NS"), atol=1e-4)
        np.testing.assert_allclose(float(ew["rate"][2, 2]), _b("EW")[2, 2], atol=1e-4)

    def test_too_few_epochs_for_the_rate_model_raises(self):
        ref = reference().sel(time=slice("2021-06-01", "2024-06-30"))    # 2022, 2024 → only 2023-07 inside
        with pytest.raises(ValueError, match="model='offset'"):
            calibrate_to_reference(secondary_biased(), ref)
        corrected, fit = calibrate_to_reference(secondary_biased(), ref, model="offset")
        assert fit.attrs["n_overlap"] == 1

    def test_no_overlap_raises(self):
        ref = reference().sel(time=slice("2016-01-01", "2016-12-31"))
        with pytest.raises(ValueError, match="no overlap"):
            calibrate_to_reference(secondary_biased(), ref)

    def test_overlap_argument_restricts_the_fit(self):
        ref, sec = reference(), secondary_biased()
        _, fit = calibrate_to_reference(sec, ref, overlap=("20200101", "20220101"))
        assert fit.attrs["overlap_dates"] == ["20200701", "20210701"]
        np.testing.assert_allclose(fit["rate"].sel(component="EW").values, _b("EW"), atol=1e-4)

    def test_grid_mismatch_raises(self):
        ref = reference()
        sec = _series(SEC_DATES, lambda c, t: _v(c) * t, origin=(500003.0, 4649000.0))
        with pytest.raises(ValueError, match="same grid"):
            calibrate_to_reference(sec, ref)

    def test_bad_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            calibrate_to_reference(secondary_biased(), reference(), model="quadratic")

    def test_dask_backed_input(self):
        pytest.importorskip("dask")
        ref, sec = reference().chunk({"y": 3}), secondary_biased().chunk({"y": 3})
        corrected, fit = calibrate_to_reference(sec, ref)
        np.testing.assert_allclose(fit["rate"].sel(component="EW").values, _b("EW"), atol=1e-4)


class TestFuseSeries:
    def test_union_with_reference_winning(self):
        ref = _series(["20200101", "20210101", "20220101"], lambda c, t: 1.0 + t)
        sec = _series(["20210101", "20230101"], lambda c, t: 99.0 + t)
        fused = fuse_series(ref, sec, reference_label="SPOT", secondary_label="PS")
        assert list(fused["date"].values) == ["20200101", "20210101", "20220101", "20230101"]
        assert list(fused["source"].values) == ["SPOT", "SPOT", "SPOT", "PS"]
        np.testing.assert_allclose(fused["EW"].sel(time="2021-01-01").values,
                                   ref["EW"].sel(time="2021-01-01").values)
        assert float(fused["EW"].sel(time="2023-01-01").mean()) > 90
        assert "magn" not in fused.data_vars
        assert fused.attrs["fusion"] == {"reference": "SPOT", "secondary": "PS",
                                         "n_reference": 3, "n_secondary_added": 1,
                                         "n_shared_dropped": 1}

    def test_magn_is_dropped_with_a_warning(self, caplog_gmc):
        ref = _series(["20200101", "20210101", "20220101"], lambda c, t: 1.0 + t)
        ref["magn"] = ref["EW"] * 2
        sec = _series(["20230101"], lambda c, t: 1.0 + t)
        fused = fuse_series(ref, sec)
        assert "magn" not in fused.data_vars
        assert "re-derived" in caplog_gmc.text

    def test_grid_mismatch_raises(self):
        ref = _series(["20200101", "20210101"], lambda c, t: 1.0 + t)
        sec = _series(["20230101"], lambda c, t: 1.0 + t, origin=(500003.0, 4649000.0))
        with pytest.raises(ValueError, match="same grid"):
            fuse_series(ref, sec)


def _write_pair_raster(path: Path, data: np.ndarray, *, rows=H, origin=(500000.0, 4649000.0)) -> Path:
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": W, "height": rows,
               "crs": CRS, "transform": from_origin(origin[0], origin[1], 3.0, 3.0), "nodata": np.nan}
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(np.ascontiguousarray(data[:rows].astype("float32")), 1)
    return path


def _pair(tmp_path, d1, d2, fn, sensor="spot6", *, rows=H):
    """A stand-in Pair whose rasters hold fn(comp, t1, t2) — shape (rows, W)."""
    t1, t2 = _years([d1, d2], "20160101")
    ew = _write_pair_raster(tmp_path / f"{d1}_{d2}_EW.tif", fn("EW", t1, t2), rows=rows)
    ns = _write_pair_raster(tmp_path / f"{d1}_{d2}_NS.tif", fn("NS", t1, t2), rows=rows)
    fmt = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return SimpleNamespace(
        pa_key=f"PZ_{fmt(d1)}-{sensor}_{fmt(d2)}-{sensor}",
        pa_left=SimpleNamespace(th_date=fmt(d1), th_sensor=sensor),
        pa_right=SimpleNamespace(th_date=fmt(d2), th_sensor=sensor),
        pa_ew_path=ew, pa_ns_path=ns,
    )


class TestClosure:
    def test_consistent_pair_has_zero_residual(self, tmp_path):
        series = reference()
        pair = _pair(tmp_path, "20180101", "20220101", lambda c, t1, t2: _v(c) * (t2 - t1))
        closure = pair_closure_against_series([pair], series)
        assert closure["EW"].dims == ("pair", "y", "x")
        assert list(closure["pair"].values) == [pair.pa_key]
        assert list(closure["sensor"].values) == ["spot6"]
        assert float(closure["dt_days"].values[0]) == 1461.0
        np.testing.assert_allclose(closure["EW"].values, 0.0, atol=1e-4)
        np.testing.assert_allclose(closure["NS"].values, 0.0, atol=1e-4)

    def test_biased_pair_reads_its_rate_bias(self, tmp_path):
        series = reference()
        b = 0.25
        pair = _pair(tmp_path, "20180101", "20220101",
                     lambda c, t1, t2: _v(c) * (t2 - t1) + (b * (t2 - t1) if c == "EW" else 0.0))
        summary = closure_summary(pair_closure_against_series([pair], series))
        ew = summary[summary["component"] == "EW"].iloc[0]
        ns = summary[summary["component"] == "NS"].iloc[0]
        assert ew["rate_bias_m_yr"] == pytest.approx(b, abs=1e-3)
        assert ew["median"] == pytest.approx(b * 4.0, abs=1e-3)
        assert ns["rate_bias_m_yr"] == pytest.approx(0.0, abs=1e-3)
        assert ew["n"] == H * W
        assert set(summary.columns) >= {"pair", "sensor", "date1", "date2", "dt_days",
                                        "component", "median", "nmad", "rms", "n", "rate_bias_m_yr"}

    def test_interpolates_between_series_epochs(self, tmp_path):
        series = reference()
        pair = _pair(tmp_path, "20180701", "20210701", lambda c, t1, t2: _v(c) * (t2 - t1))
        closure = pair_closure_against_series([pair], series)
        np.testing.assert_allclose(closure["EW"].values, 0.0, atol=1e-4)

    def test_fortran_style_short_series_takes_an_exact_row_drop(self, tmp_path):
        series = reference().isel(y=slice(0, H - 1))            # H−1 rows, same origin
        pair = _pair(tmp_path, "20180101", "20220101", lambda c, t1, t2: _v(c) * (t2 - t1))
        closure = pair_closure_against_series([pair], series)
        assert closure["EW"].shape == (1, H - 1, W)
        np.testing.assert_allclose(closure["EW"].values, 0.0, atol=1e-4)

    def test_pairs_outside_the_span_are_skipped_and_listed(self, tmp_path, caplog_gmc):
        series = reference().sel(time=slice("2018-01-01", "2022-12-31"))
        inside = _pair(tmp_path, "20180101", "20220101", lambda c, t1, t2: _v(c) * (t2 - t1))
        outside = _pair(tmp_path, "20160101", "20180101", lambda c, t1, t2: _v(c) * (t2 - t1))
        closure = pair_closure_against_series([inside, outside], series)
        assert list(closure["pair"].values) == [inside.pa_key]
        assert closure.attrs["skipped"] == [outside.pa_key]
        assert "skipped" in caplog_gmc.text

    def test_no_pair_inside_raises(self, tmp_path):
        series = reference().sel(time=slice("2024-01-01", "2025-12-31"))
        pair = _pair(tmp_path, "20160101", "20180101", lambda c, t1, t2: _v(c) * (t2 - t1))
        with pytest.raises(ValueError, match="time span"):
            pair_closure_against_series([pair], series)

    def test_mask_restricts_the_summary(self, tmp_path):
        series = reference()
        pair = _pair(tmp_path, "20180101", "20220101", lambda c, t1, t2: _v(c) * (t2 - t1))
        closure = pair_closure_against_series([pair], series)
        mask = xr.DataArray(np.zeros((H, W), bool), dims=("y", "x"))
        mask.values[:2, :] = True
        summary = closure_summary(closure, mask=mask)
        assert (summary["n"] == 2 * W).all()

    def test_paths_override(self, tmp_path):
        series = reference()
        pair = _pair(tmp_path, "20180101", "20220101", lambda c, t1, t2: _v(c) * (t2 - t1))
        alt = tmp_path / "alt"
        alt.mkdir()
        biased = _pair(alt, "20180101", "20220101", lambda c, t1, t2: _v(c) * (t2 - t1) + 1.0)
        closure = pair_closure_against_series(
            [pair], series, paths=lambda p: (biased.pa_ew_path, biased.pa_ns_path))
        np.testing.assert_allclose(closure["EW"].values, 1.0, atol=1e-4)


class TestOrchestrator:
    @pytest.fixture
    def dirs(self, tmp_path):
        ref_dir, sec_dir = tmp_path / "ref", tmp_path / "sec"
        for comp in ("EW", "NS"):
            write_tot_stack(ref_dir, REF_DATES, comp,
                            lambda t, yy, xx, c=comp: _v(c) * t, t0="20160101")
            write_tot_stack(sec_dir, SEC_DATES, comp,
                            lambda t, yy, xx, c=comp: _v(c) * t + _a(c)
                            + _b(c) * (t - _years(["20180701"], "20160101")[0]),
                            t0="20160101")
        return ref_dir, sec_dir, tmp_path / "fused"

    def test_end_to_end(self, dirs, caplog_gmc):
        ref_dir, sec_dir, out = dirs
        fused, fit, written = fuse_inversions(
            ref_dir, sec_dir, out, reference_label="SPOT", secondary_label="PlanetScope")
        assert set(written) == {"EW", "NS", "magn"}
        assert list(fused["date"].values) == sorted(REF_DATES + SEC_DATES)
        assert fused["source"].values.tolist().count("PlanetScope") == len(SEC_DATES)
        np.testing.assert_allclose(fit["rate"].sel(component="EW").values, _b("EW"), atol=1e-3)
        for comp in ("EW", "NS"):
            for var in ("offset", "rate", "rms", "n_valid"):
                assert (out / "calibration" / f"{comp}_{var}.tif").exists()
        doc = json.loads((out / "fusion_parameters.json").read_text())
        assert doc["kind"] == "fusion"
        assert doc["calibration"]["model"] == "offset+rate"
        assert doc["calibration"]["extrapolated_outside_overlap"] is True
        assert doc["fused"]["source_per_date"]["20180701"] == "PlanetScope"
        assert doc["fused"]["source_per_date"]["20180101"] == "SPOT"
        assert doc["reference"]["label"] == "SPOT"
        assert abs(doc["calibration"]["per_component"]["EW"]["rate_m_yr"]["median"] - np.median(_b("EW"))) < 1e-3
        assert "median rate bias" in caplog_gmc.text

        # the product reads like any other inversion, and the fused series is
        # continuous: PlanetScope dates now sit on the SPOT line
        ex = InversionExtractor(out)
        assert ex.available_dates == sorted(REF_DATES + SEC_DATES)
        back = load_cumulative_stack(out, chunks=None)
        t = _years(list(back["date"].values), "20160101")
        for comp in ("EW", "NS"):
            expected = np.stack([_v(comp) * tk for tk in t])
            np.testing.assert_allclose(back[comp].values, expected, atol=1e-3)

    def test_accepts_inversion_objects(self, dirs):
        ref_dir, sec_dir, out = dirs
        fused, _, _ = fuse_inversions(SimpleNamespace(inversion_dir=ref_dir),
                                      SimpleNamespace(inversion_dir=sec_dir), out)
        assert fused.sizes["time"] == len(REF_DATES) + len(SEC_DATES)

    def test_trace_failure_does_not_cost_the_product(self, dirs, monkeypatch, caplog_gmc):
        ref_dir, sec_dir, out = dirs
        monkeypatch.setattr(
            "geomulticorr.inversion._run_parameters.write_run_parameters",
            lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
        _, _, written = fuse_inversions(ref_dir, sec_dir, out)
        assert written["EW"][REF_DATES[0]].exists()
        assert not (out / "fusion_parameters.json").exists()
        assert "could not write the trace" in caplog_gmc.text

    def test_build_fusion_parameters_is_json_able(self):
        ref, sec = reference(), secondary_biased()
        corrected, fit = calibrate_to_reference(sec, ref)
        fused = fuse_series(ref, corrected)
        doc = build_fusion_parameters(
            reference_dir="/r", secondary_dir="/s", reference_label="a", secondary_label="b",
            fit=fit, fused=fused, components=("EW", "NS"))
        json.dumps(doc)
        assert doc["calibration"]["n_overlap"] == 4
        assert doc["fused"]["n_dates"] == 10
        assert "written_utc" in doc
