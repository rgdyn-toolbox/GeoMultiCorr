#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_planet_georeference.py
# creation date: 2026-11-18.
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
"""Georeferencing Planet thumbnails from their footprints (synthetic PNGs, offline).

The synthetic thumbnails are drawn the way the tiles API draws them: the
footprint's EPSG:3857 bounding box, letterboxed into a square frame, with the
footprint opaque and the padding transparent. The letterbox is computed here
independently of :func:`thumbnail_transform`, so the tests are not circular.
"""

import logging

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from PIL import Image
from affine import Affine
from rasterio.features import rasterize
from shapely import affinity
from shapely.geometry import Polygon, box

from geomulticorr.data_sources.planet_image_search_download import (
    GEOREF_COLUMNS,
    georeference_thumbnail,
    georeference_thumbnails,
    thumbnail_transform,
)

# A real PSScene footprint over Rasuwa (Nepal), EPSG:4326: rotated, 4 vertices.
FOOTPRINT = Polygon(
    [
        (85.24272249, 28.36680017),
        (85.20210894, 28.19093991),
        (85.53471615, 28.12893431),
        (85.57685059, 28.30532242),
    ]
)
AOI = box(85.484593, 28.270047, 85.545291, 28.308649)  # inside FOOTPRINT
FAR_AOI = box(10.0, 45.0, 10.1, 45.1)

# pytest's own threads make the parent multi-threaded, so Python 3.12+ warns
# that fork() may deadlock; that is the documented reason "thread" is the default.
_FORK_OK = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)


def _merc(geom):
    return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(3857).iloc[0]


def _letterbox(geom, size):
    """Expected transform, written out independently of the library."""
    minx, miny, maxx, maxy = _merc(geom).bounds
    w, h = maxx - minx, maxy - miny
    side = max(w, h)
    s = side / size
    return Affine(s, 0, minx - (side - w) / 2, 0, -s, maxy + (side - h) / 2)


def _write_thumb(path, geom=FOOTPRINT, size=128, shift_px=0.0, seed=0):
    tr = _letterbox(geom, size) * Affine.translation(shift_px, 0)
    alpha = rasterize(
        [_merc(geom)],
        out_shape=(size, size),
        transform=tr,
        fill=0,
        default_value=255,
        dtype="uint8",
    )
    rgb = np.random.default_rng(seed).integers(1, 255, (size, size, 3), dtype=np.uint8)
    rgb[alpha == 0] = 0
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path)
    return path


def _frame(ids, geoms=None):
    geoms = geoms or [FOOTPRINT] * len(ids)
    return gpd.GeoDataFrame({"image_id": list(ids)}, geometry=geoms, crs="EPSG:4326")


class TestThumbnailTransform:
    def test_wide_box_fills_width_and_centres_height(self):
        tr = thumbnail_transform((0, 0, 200, 100), 512, 512)
        s = 200 / 512
        assert tr.a == pytest.approx(s) and tr.e == pytest.approx(-s)
        assert tr.c == pytest.approx(0) and tr.f == pytest.approx(150)  # 50 m pad

    def test_tall_box_fills_height_and_centres_width(self):
        tr = thumbnail_transform((0, 0, 100, 200), 512, 512)
        assert tr.c == pytest.approx(-50) and tr.f == pytest.approx(200)
        assert tr.a == pytest.approx(-tr.e)  # square pixels

    def test_non_square_frame(self):
        tr = thumbnail_transform((0, 0, 100, 100), 200, 100)
        assert tr.a == pytest.approx(1.0) and tr.c == pytest.approx(-50)

    @pytest.mark.parametrize(
        "bounds, size", [((0, 0, 0, 10), (10, 10)), ((0, 0, 10, 10), (0, 10))]
    )
    def test_degenerate_raises(self, bounds, size):
        with pytest.raises(ValueError, match="Degenerate"):
            thumbnail_transform(bounds, *size)


class TestGeoreferenceThumbnail:
    def test_native_grid_is_exact(self, tmp_path):
        png = _write_thumb(tmp_path / "id1.png")
        rec = georeference_thumbnail(png, FOOTPRINT)
        assert rec["path"] == tmp_path / "id1.tif" and rec["image_id"] == "id1"
        assert rec["iou"] > 0.95
        with rasterio.open(rec["path"]) as ds:
            assert ds.crs.to_epsg() == 3857 and ds.count == 4
            assert ds.transform.almost_equals(_letterbox(FOOTPRINT, 128))
            data = ds.read()
            assert ds.colorinterp[3].name == "alpha"
        src = np.moveaxis(np.asarray(Image.open(png)), -1, 0)
        np.testing.assert_array_equal(data, src)  # no resampling
        assert not list(tmp_path.glob("*.part"))

    def test_geoseries_footprint_in_another_crs(self, tmp_path):
        png = _write_thumb(tmp_path / "id1.png")
        fp = gpd.GeoSeries([FOOTPRINT], crs="EPSG:4326").to_crs(32645)
        rec = georeference_thumbnail(png, fp, out_path=tmp_path / "o" / "x.tif")
        assert rec["path"].exists() and rec["iou"] > 0.95

    def test_utm_reprojection_keeps_transparency(self, tmp_path):
        png = _write_thumb(tmp_path / "id1.png")
        rec = georeference_thumbnail(png, FOOTPRINT, dst_crs="utm")
        with rasterio.open(rec["path"]) as ds:
            assert ds.crs.to_epsg() == 32645  # UTM 45N
            alpha = ds.read(4)
        assert alpha[0, 0] == 0 and alpha[-1, -1] == 0  # padding stays transparent
        assert (alpha == 255).mean() > 0.3

    def test_explicit_epsg(self, tmp_path):
        png = _write_thumb(tmp_path / "id1.png")
        rec = georeference_thumbnail(png, FOOTPRINT, dst_crs=4326)
        assert rec["crs"] == "EPSG:4326"

    def test_crop_to_aoi_bounds(self, tmp_path):
        png = _write_thumb(tmp_path / "id1.png", size=512)
        aoi = gpd.GeoDataFrame(geometry=[AOI], crs="EPSG:4326")
        rec = georeference_thumbnail(png, FOOTPRINT, dst_crs="utm", crop_to=aoi)
        aoi_b = aoi.to_crs(rec["crs"]).total_bounds
        with rasterio.open(rec["path"]) as ds:
            b, res = ds.bounds, ds.res[0]
            assert ds.width < 512 and ds.height < 512
        # whole pixels covering the AOI box: at most one pixel beyond it
        assert b.left <= aoi_b[0] and b.left > aoi_b[0] - res - 1e-6
        assert b.bottom <= aoi_b[1] and b.bottom > aoi_b[1] - res - 1e-6
        assert b.right >= aoi_b[2] and b.right < aoi_b[2] + res + 1e-6
        assert b.top >= aoi_b[3] and b.top < aoi_b[3] + res + 1e-6

    def test_crop_from_a_vector_file(self, tmp_path):
        png = _write_thumb(tmp_path / "id1.png")
        gpkg = tmp_path / "aoi.gpkg"
        gpd.GeoDataFrame(geometry=[AOI], crs="EPSG:4326").to_file(gpkg)
        rec = georeference_thumbnail(png, FOOTPRINT, crop_to=gpkg)
        assert rec["path"].exists()

    def test_crop_outside_writes_nothing(self, tmp_path, caplog):
        png = _write_thumb(tmp_path / "id1.png")
        with caplog.at_level(logging.WARNING):
            rec = georeference_thumbnail(png, FOOTPRINT, crop_to=FAR_AOI)
        assert rec["path"] is None and not (tmp_path / "id1.tif").exists()
        assert "does not overlap" in caplog.text

    def test_crop_in_the_letterbox_corner_is_outside(self, tmp_path, caplog):
        # inside the PNG frame (bbox of the rotated footprint) but off the scene
        minx, miny, maxx, maxy = FOOTPRINT.bounds
        corner = box(minx, maxy - 0.01, minx + 0.01, maxy)
        assert not corner.intersects(FOOTPRINT)
        png = _write_thumb(tmp_path / "id1.png")
        with caplog.at_level(logging.WARNING):
            rec = georeference_thumbnail(png, FOOTPRINT, crop_to=corner)
        assert rec["path"] is None and "footprint" in caplog.text

    def test_common_grid_without_crop_is_snapped(self, tmp_path):
        png = _write_thumb(tmp_path / "id1.png")
        rec = georeference_thumbnail(png, FOOTPRINT, dst_crs="utm", resolution=30)
        assert rec["res"] == 30
        with rasterio.open(rec["path"]) as ds:
            t = ds.transform
            assert ds.res == (30.0, 30.0)
            assert t.c % 30 == pytest.approx(0) and t.f % 30 == pytest.approx(0)
            assert ds.read(4)[0, 0] == 0  # still transparent outside the scene

    def test_common_grid_with_crop_gives_identical_grids(self, tmp_path):
        # two scenes of different size: their native pixel sizes differ
        big = affinity.scale(FOOTPRINT, 1.3, 1.3)
        a = _write_thumb(tmp_path / "a.png", seed=1)
        b = _write_thumb(tmp_path / "b.png", geom=big, seed=2)
        ra = georeference_thumbnail(a, FOOTPRINT, dst_crs="utm", crop_to=AOI)
        rb = georeference_thumbnail(b, big, dst_crs="utm", crop_to=AOI)
        assert ra["res"] != pytest.approx(rb["res"])  # what the option fixes
        grids = []
        for png, fp in ((a, FOOTPRINT), (b, big)):
            rec = georeference_thumbnail(
                png,
                fp,
                out_path=tmp_path / f"g_{png.stem}.tif",
                dst_crs="utm",
                crop_to=AOI,
                resolution=50,
            )
            with rasterio.open(rec["path"]) as ds:
                grids.append((ds.crs, ds.transform, ds.shape))
        assert grids[0] == grids[1]
        crs, t, (h, w) = grids[0]
        aoi_b = gpd.GeoSeries([AOI], crs="EPSG:4326").to_crs(crs).total_bounds
        assert t.c <= aoi_b[0] < t.c + 50 and t.f - 50 < aoi_b[3] <= t.f
        assert t.c + w * 50 >= aoi_b[2] and t.f - h * 50 <= aoi_b[1]

    def test_resampling_is_applied(self, tmp_path):
        png = _write_thumb(tmp_path / "id1.png")
        kw = dict(dst_crs="utm", crop_to=AOI, resolution=20)
        near = georeference_thumbnail(png, FOOTPRINT, out_path=tmp_path / "n.tif", **kw)
        bil = georeference_thumbnail(
            png, FOOTPRINT, out_path=tmp_path / "b.tif", resampling="bilinear", **kw
        )
        with rasterio.open(near["path"]) as n, rasterio.open(bil["path"]) as b:
            src_values = set(np.unique(np.asarray(Image.open(png))[..., 0]))
            assert set(np.unique(n.read(1))) <= src_values  # nearest copies values
            assert not np.array_equal(n.read(1), b.read(1))

    @pytest.mark.parametrize(
        "kw, match",
        [
            (dict(resolution=0), "resolution"),
            (dict(resolution=-5), "resolution"),
            (dict(resampling="sinc"), "resampling"),
        ],
    )
    def test_bad_grid_arguments_raise(self, tmp_path, kw, match):
        png = _write_thumb(tmp_path / "id1.png")
        with pytest.raises(ValueError, match=match):
            georeference_thumbnail(png, FOOTPRINT, **kw)
        with pytest.raises(ValueError, match=match):
            georeference_thumbnails(tmp_path, _frame(["id1"]), **kw)

    def test_misplaced_thumbnail_warns(self, tmp_path, caplog):
        png = _write_thumb(tmp_path / "id1.png", shift_px=40)
        with caplog.at_level(logging.WARNING):
            rec = georeference_thumbnail(png, FOOTPRINT)
        assert rec["iou"] < 0.9 and "IoU" in caplog.text
        assert rec["path"].exists()  # a warning, not a failure


class TestGeoreferenceThumbnails:
    def _folder(self, tmp_path, ids=("a", "b", "c")):
        d = tmp_path / "thumbs"
        d.mkdir()
        for k, i in enumerate(ids):
            _write_thumb(d / f"{i}.png", seed=k)
        return d

    def test_folder_statuses_skip_and_overwrite(self, tmp_path, caplog):
        d = self._folder(tmp_path)
        gdf = _frame(["a", "b"])  # "c" has no footprint
        with caplog.at_level(logging.WARNING):
            out = georeference_thumbnails(d, gdf, n_jobs=1, show_progress=False)
        assert list(out.columns) == GEOREF_COLUMNS
        assert list(out["image_id"]) == ["a", "b", "c"]
        assert list(out["status"]) == ["written", "written", "no_footprint"]
        assert "no footprint" in caplog.text and "c" in caplog.text

        again = georeference_thumbnails(d, gdf, n_jobs=1, show_progress=False)
        assert list(again["status"][:2]) == ["skipped", "skipped"]
        forced = georeference_thumbnails(
            d, gdf, n_jobs=1, overwrite=True, show_progress=False
        )
        assert list(forced["status"][:2]) == ["written", "written"]

    def test_gpkg_path_list_input_and_output_dir(self, tmp_path):
        d = self._folder(tmp_path, ids=("a",))
        gpkg = tmp_path / "aois.gpkg"
        _frame(["a"]).to_file(gpkg)
        out = georeference_thumbnails(
            [d / "a.png"], gpkg, output_dir=tmp_path / "tif", show_progress=True
        )
        assert out.loc[0, "path"] == tmp_path / "tif" / "a.tif"
        assert out.loc[0, "status"] == "written"

    def test_utm_is_resolved_once_for_the_batch(self, tmp_path):
        d = self._folder(tmp_path, ids=("a", "b"))
        # second footprint straddles into the next UTM zone (UTM 46 starts at 90°E)
        east = affinity.translate(FOOTPRINT, xoff=4.6)
        gdf = _frame(["a", "b"], [FOOTPRINT, east])
        out = georeference_thumbnails(
            d, gdf, dst_crs="utm", crop_to=AOI, n_jobs=1, show_progress=False
        )
        assert out["crs"].nunique() == 1 and out.loc[0, "crs"] == "EPSG:32645"
        # "b"'s image does not cover the AOI once its footprint moved
        assert list(out["status"]) == ["written", "outside"]

    @_FORK_OK
    @pytest.mark.parametrize(
        "n_jobs, executor", [(2, "thread"), (None, "thread"), (2, "process")]
    )
    def test_parallel_equals_serial(self, tmp_path, n_jobs, executor):
        d = self._folder(tmp_path)
        gdf = _frame(["a", "b", "c"])
        # dst_crs="utm" exercises the reprojection, the GDAL-heavy part
        ref = georeference_thumbnails(
            d,
            gdf,
            output_dir=tmp_path / "serial",
            dst_crs="utm",
            resolution=40,
            n_jobs=1,
            show_progress=False,
        )
        par = georeference_thumbnails(
            d,
            gdf,
            output_dir=tmp_path / "par",
            dst_crs="utm",
            resolution=40,
            n_jobs=n_jobs,
            executor=executor,
            show_progress=False,
        )
        assert list(par["image_id"]) == ["a", "b", "c"]  # input order kept
        assert (par["status"] == "written").all()
        for p_ref, p_par in zip(ref["path"], par["path"]):
            with rasterio.open(p_ref) as a, rasterio.open(p_par) as b:
                assert a.transform == b.transform and a.crs == b.crs
                np.testing.assert_array_equal(a.read(), b.read())

    @_FORK_OK
    @pytest.mark.parametrize("executor", ["thread", "process"])
    def test_a_corrupt_png_fails_alone_and_warnings_reach_the_parent(
        self, tmp_path, caplog, executor
    ):
        d = self._folder(tmp_path, ids=("a", "b"))
        (d / "bad.png").write_bytes(b"not a png")
        _write_thumb(d / "shifted.png", shift_px=40)
        gdf = _frame(["a", "b", "bad", "shifted"])
        with caplog.at_level(logging.WARNING):
            out = georeference_thumbnails(
                d, gdf, n_jobs=2, executor=executor, show_progress=False
            )
        status = dict(zip(out["image_id"], out["status"]))
        assert status == {
            "a": "written",
            "b": "written",
            "bad": "failed",
            "shifted": "written",
        }
        bad = out.set_index("image_id").loc["bad", "message"]
        assert "UnidentifiedImageError" in bad
        assert "bad [failed]" in caplog.text and "shifted [written]" in caplog.text

    def test_argument_validation(self, tmp_path):
        with pytest.raises(ValueError, match="executor"):
            georeference_thumbnails(tmp_path, _frame(["a"]), executor="mpi")
        with pytest.raises(ValueError, match="n_jobs"):
            georeference_thumbnails(tmp_path, _frame(["a"]), n_jobs=0)
        with pytest.raises(FileNotFoundError):
            georeference_thumbnails(tmp_path / "nope", _frame(["a"]))
        with pytest.raises(ValueError, match="image_id"):
            georeference_thumbnails(tmp_path, gpd.GeoDataFrame(geometry=[AOI]))

    def test_default_executor_is_thread(self, tmp_path, monkeypatch):
        from geomulticorr.data_sources import planet_image_search_download as pmod

        used = []
        real = pmod._futures.ThreadPoolExecutor

        def spy(*a, **k):
            used.append("thread")
            return real(*a, **k)

        monkeypatch.setattr(pmod._futures, "ThreadPoolExecutor", spy)
        d = self._folder(tmp_path, ids=("a", "b"))
        out = georeference_thumbnails(
            d, _frame(["a", "b"]), n_jobs=2, show_progress=False
        )
        assert used == ["thread"] and (out["status"] == "written").all()

    def test_empty_folder(self, tmp_path):
        out = georeference_thumbnails(tmp_path, _frame(["a"]), show_progress=False)
        assert out.empty and list(out.columns) == GEOREF_COLUMNS
