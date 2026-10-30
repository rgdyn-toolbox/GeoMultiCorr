"""Similarity between GeoTIFFs (similarity/raster.py): grids, masks, cache."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from geomulticorr.similarity import raster as sr
from tests.similarity.conftest import EPSG, ORIGIN

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
