#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for InversionExtractor._discover_tif_paths().

Locks down that the component-suffixed TOT_<date>_{EW,NS,magn}.tif naming
(see tests/inversion/test_post_process.py) still yields clean, matching
YYYYMMDD date keys across EW/NS/magn — a stray "_EW"/"_NS" left in the key
would silently break every date alignment downstream (available_dates,
compute_stats, extract's wide-format pivot).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from geomulticorr.stats.inversion_extractor import InversionExtractor

_DATES = ["20210901", "20211001"]


def _touch(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@pytest.fixture
def extractor(tmp_path):
    for date in _DATES:
        _touch(tmp_path / "inverse_EW" / f"TOT_{date}_EW.tif")
        _touch(tmp_path / "inverse_NS" / f"TOT_{date}_NS.tif")
        _touch(tmp_path / "inverse_magn" / f"TOT_{date}_magn.tif")

    obj = InversionExtractor.__new__(InversionExtractor)
    obj.inversion = SimpleNamespace(inversion_dir=tmp_path)
    return obj


class TestDiscoverTifPaths:
    def test_date_keys_have_no_component_suffix(self, extractor):
        result = extractor._discover_tif_paths(components=None)

        for comp, suffix in (("EW", "_EW"), ("NS", "_NS"), ("magn", "_magn")):
            assert set(result[comp].keys()) == set(_DATES)
            for date, path in result[comp].items():
                assert path.name == f"TOT_{date}{suffix}.tif"

    def test_dates_align_across_components(self, extractor):
        result = extractor._discover_tif_paths(components=None)
        assert set(result["EW"]) == set(result["NS"]) == set(result["magn"])

    def test_restricts_to_requested_components(self, extractor):
        result = extractor._discover_tif_paths(components=["EW"])
        assert list(result.keys()) == ["EW"]

    def test_backward_compatible_with_unsuffixed_files(self, tmp_path):
        # A pre-change post_process() run left plain TOT_<date>.tif behind —
        # removesuffix("_EW") on an already-bare date string is a no-op.
        _touch(tmp_path / "inverse_EW" / f"TOT_{_DATES[0]}.tif")
        obj = InversionExtractor.__new__(InversionExtractor)
        obj.inversion = SimpleNamespace(inversion_dir=tmp_path)

        result = obj._discover_tif_paths(components=["EW"])

        assert set(result["EW"].keys()) == {_DATES[0]}

    def test_available_dates_and_components_properties(self, extractor):
        extractor.tif_paths = extractor._discover_tif_paths(components=None)
        assert extractor.available_dates == _DATES
        assert sorted(extractor.available_components) == ["EW", "NS", "magn"]

    def test_init_raises_when_nothing_found(self, tmp_path):
        empty = SimpleNamespace(inversion_dir=tmp_path / "no_such_run")
        with pytest.raises(FileNotFoundError):
            InversionExtractor(empty)


class TestDirectoryInput:
    """A fused product has no TIOInversion; the folder alone must do."""

    def test_accepts_a_directory(self, tmp_path):
        for date in _DATES:
            _touch(tmp_path / "inverse_EW" / f"TOT_{date}_EW.tif")
        ex = InversionExtractor(tmp_path)
        assert ex.inversion_dir == tmp_path
        assert ex.available_dates == _DATES
        assert ex.available_components == ["EW"]

    def test_accepts_a_string_path(self, tmp_path):
        _touch(tmp_path / "inverse_NS" / f"TOT_{_DATES[0]}_NS.tif")
        assert InversionExtractor(str(tmp_path)).available_components == ["NS"]

    def test_module_helper_equals_the_method(self, extractor):
        from geomulticorr.stats.inversion_extractor import discover_tot_rasters
        assert extractor._discover_tif_paths(None) == discover_tot_rasters(
            extractor.inversion.inversion_dir)

    def test_empty_directory_names_both_ways_to_produce_rasters(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="launch\\(mode='python'\\)"):
            InversionExtractor(tmp_path)


# ── pixel-reading tests (real synthetic TOT rasters) ──────────────────────────

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import LineString, Point, box  # noqa: E402

from geomulticorr.stats import inversion_extractor as ie  # noqa: E402
from tests.inversion.conftest import DEFAULT_CRS, DEFAULT_ORIGIN, write_tot_stack  # noqa: E402

_X0, _Y0 = DEFAULT_ORIGIN


def _pix_value(t, yy, xx):
    return yy * 10 + xx


def _center(row, col, res=3.0):
    return Point(_X0 + col * res + res / 2, _Y0 - row * res - res / 2)


@pytest.fixture
def real_extractor(tmp_path):
    root = tmp_path / "inv"
    write_tot_stack(root, _DATES, "EW", _pix_value)
    write_tot_stack(root, _DATES, "NS", lambda t, yy, xx: -(yy * 10 + xx))
    return InversionExtractor(root)


def _write_gpkg(tmp_path, geoms, crs=DEFAULT_CRS, **cols):
    gdf = gpd.GeoDataFrame(cols, geometry=geoms, crs=crs)
    path = tmp_path / "geoms.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


class TestComputeStats:
    def test_columns_dates_and_values(self, real_extractor):
        df = real_extractor.compute_stats()
        assert set(df["component"]) == {"EW", "NS"}
        assert len(df) == 4
        assert np.issubdtype(df["date"].dtype, np.datetime64)
        ew = df[df["component"] == "EW"]
        assert ew["mean"].iloc[0] == pytest.approx(27.0)
        assert ew["count_valid"].iloc[0] == 30

    def test_component_subset(self, real_extractor):
        df = real_extractor.compute_stats(components=["NS"])
        assert set(df["component"]) == {"NS"}

    def test_unavailable_component_warns_and_is_skipped(self, real_extractor, caplog_gmc):
        df = real_extractor.compute_stats(components=["EW", "magn"])
        assert set(df["component"]) == {"EW"}
        assert "magn" in caplog_gmc.text

    def test_unreadable_raster_gives_empty_frame(self, real_extractor, monkeypatch, caplog_gmc):
        monkeypatch.setattr(ie, "compute_raster_stats", lambda *a, **k: None)
        df = real_extractor.compute_stats()
        assert df.empty
        assert "Could not read stats" in caplog_gmc.text


class TestExtractPoints:
    def test_nearest_pixel_value(self, real_extractor, tmp_path):
        gpkg = _write_gpkg(tmp_path, [_center(1, 2)], point_id=["a"])
        out = real_extractor.extract(gpkg, components=["EW"])
        assert len(out) == 2  # two dates
        assert (out["mean"] == 12.0).all()
        assert (out["std"] == 0.0).all()
        assert (out["count"] == 1).all()
        assert set(out["point_id"]) == {"a"}

    def test_point_outside_raster_is_nan(self, real_extractor, tmp_path):
        gpkg = _write_gpkg(tmp_path, [Point(_X0 + 5000, _Y0 - 5000)])
        out = real_extractor.extract(gpkg, components=["EW"])
        assert out["mean"].isna().all()
        assert (out["count"] == 0).all()
        assert out["std"].isna().all()

    def test_geographic_input_is_reprojected(self, real_extractor, tmp_path):
        pt = gpd.GeoSeries([_center(1, 2)], crs=DEFAULT_CRS).to_crs("EPSG:4326")
        gpkg = _write_gpkg(tmp_path, list(pt), crs="EPSG:4326")
        out = real_extractor.extract(gpkg, components=["EW"])
        assert (out["mean"] == 12.0).all()

    def test_crs_less_input_warns(self, real_extractor, tmp_path, caplog_gmc):
        gdf = gpd.GeoDataFrame(geometry=[_center(1, 2)], crs=None)
        path = tmp_path / "nocrs.gpkg"
        gdf.to_file(path, driver="GPKG")
        real_extractor.extract(path, components=["EW"])
        assert "no CRS" in caplog_gmc.text

    def test_linestring_rejected(self, real_extractor, tmp_path):
        gpkg = _write_gpkg(tmp_path, [LineString([_center(0, 0), _center(1, 1)])])
        with pytest.raises(ValueError, match="Unsupported geometry"):
            real_extractor.extract(gpkg)


class TestExtractZonal:
    def test_buffered_point_is_zonal(self, real_extractor, tmp_path):
        gpkg = _write_gpkg(tmp_path, [_center(2, 2)])
        out = real_extractor.extract(gpkg, buffer_m=4.0, components=["EW"])
        assert (out["count"] >= 1).all()
        assert out["mean"].notna().all()

    def test_polygon_stats(self, real_extractor, tmp_path):
        poly = box(_X0, _Y0 - 6, _X0 + 6, _Y0)  # pixels (0,0),(0,1),(1,0),(1,1)
        gpkg = _write_gpkg(tmp_path, [poly])
        out = real_extractor.extract(gpkg, components=["EW"])
        assert (out["count"] == 4).all()
        assert out["mean"].tolist() == pytest.approx([5.5, 5.5])
        assert (out["min"] == 0).all() and (out["max"] == 11).all()


class TestWideFormat:
    def test_wide_columns_and_values(self, real_extractor, tmp_path):
        gpkg = _write_gpkg(tmp_path, [_center(1, 2), _center(0, 0)], point_id=["a", "b"])
        wide = real_extractor.extract(gpkg, output_format="wide")
        assert len(wide) == 2
        assert "EW_2021-09-01_mean" in wide.columns
        assert "NS_2021-10-01_count" in wide.columns
        row_a = wide[wide["point_id"] == "a"].iloc[0]
        assert row_a["EW_2021-09-01_mean"] == 12.0
        assert row_a["NS_2021-09-01_mean"] == -12.0
        assert wide.geometry.notna().all()

    def test_to_wide_pure(self):
        long = gpd.GeoDataFrame(
            {
                "point_id": ["p", "p"],
                "date": ["2020-01-01", "2020-02-01"],
                "component": ["EW", "EW"],
                "mean": [1.0, 2.0], "median": [1.0, 2.0], "min": [1.0, 2.0],
                "max": [1.0, 2.0], "std": [0.0, 0.0], "count": [1, 1],
            },
            geometry=[Point(0, 0), Point(0, 0)], crs="EPSG:32632",
        )
        wide = ie._to_wide(long)
        assert len(wide) == 1
        assert wide["EW_2020-02-01_mean"].iloc[0] == 2.0


class TestHelpers:
    def test_fmt_date(self):
        assert ie._fmt_date("20210901") == "2021-09-01"

    @pytest.mark.parametrize("cols,expected", [
        ({"point_id": ["a"], "id": ["z"]}, ["a"]),
        ({"id": ["z"], "name": ["n"]}, ["z"]),
        ({"Name": ["N"]}, ["N"]),
        ({}, [0]),
    ])
    def test_extract_ids_priority(self, cols, expected):
        gdf = gpd.GeoDataFrame(cols, geometry=[Point(0, 0)])
        assert ie._extract_ids(gdf) == expected

    def test_resolve_inversion_dir_rejects_bad_type(self):
        with pytest.raises(TypeError):
            ie.resolve_inversion_dir(42)

    def test_raster_crs(self, real_extractor):
        assert real_extractor._raster_crs().to_string() == DEFAULT_CRS
