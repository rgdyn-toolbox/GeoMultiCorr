#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_dinamis_spot_image_search_download.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""DINAMIS SPOT 6/7: credentials, filename grammar and templates, URL signing."""
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from shapely.geometry import box

from geomulticorr.data_sources import dinamis_spot_image_search_download as mod
from geomulticorr.data_sources.dinamis_spot_image_search_download import (
    DEFAULT_FILENAME_TEMPLATE,
    LEGACY_FILENAME_TEMPLATE,
    DinamisClient,
    DinamisImageFinder,
)

PAN_HREF = (
    "https://minio.example/spot/COG_SPOT7_P_201806231000132_ORT_SPOT7_20180705_"
    "0813381atzn5pa9ztdm_1.tif?X-Amz-Signature=abc"
)
ITEM = {
    "id": "scene-1",
    "properties": {"datetime": "2018-06-23T10:00:13Z", "platform": "SPOT-7"},
    "assets": {
        "src_pan": {"href": PAN_HREF},
        "dimap_pan": {"href": "https://minio.example/DIM_SPOT7_P_201806231000132_ORT_SPOT7_20180705_job_1.xml"},
        "cld_msk_vec_pan": {"href": "https://minio.example/MSK_SPOT7_P_201806231000132_ORT_SPOT7_20180705_job_1_MSK.gpkg"},
        "roi_msk_vec_pan": {"href": "https://minio.example/ROI_SPOT7_P_201806231000132_ORT_SPOT7_20180705_job_1_MSK.gpkg"},
    },
}


@pytest.fixture
def stub_stac(monkeypatch):
    monkeypatch.setattr(mod.pystac_client.Client, "open", lambda *a, **k: SimpleNamespace(opened=True))


def _finder(stub_stac, **kw):
    client = DinamisClient(access_key="ak", secret_key="sk")
    return DinamisImageFinder(client, box(6.0, 45.0, 6.1, 45.1), **kw)


class TestClient:
    def test_missing_keys(self):
        with pytest.raises(ValueError, match="MDT_SPOT67_STAC_ACCESS_KEY"):
            DinamisClient()

    def test_one_key_is_not_enough(self):
        with pytest.raises(ValueError):
            DinamisClient(access_key="ak")

    def test_keys_from_environment_and_headers(self, monkeypatch, stub_stac):
        monkeypatch.setenv("MDT_SPOT67_STAC_ACCESS_KEY", "ea")
        monkeypatch.setenv("MDT_SPOT67_STAC_SECRET_KEY", "es")
        client = DinamisClient()
        assert client.headers == {"access-key": "ea", "secret-key": "es"}

    def test_exports_tld_names_without_overriding_existing(self, monkeypatch, stub_stac):
        monkeypatch.setenv("TLD_ACCESS_KEY", "already-set")
        monkeypatch.setenv("TLD_SECRET_KEY", "")  # registers cleanup for the setdefault below
        monkeypatch.delenv("TLD_SECRET_KEY")
        import os

        DinamisClient(access_key="ak", secret_key="sk")
        assert os.environ["TLD_ACCESS_KEY"] == "already-set"
        assert os.environ["TLD_SECRET_KEY"] == "sk"
        monkeypatch.delenv("TLD_SECRET_KEY")  # do not leak into other tests


class TestSigning:
    def test_sign_delegates_to_the_sdk(self, monkeypatch):
        import teledetection

        monkeypatch.setattr(teledetection, "sign", lambda href: href + "?signed=1")
        assert mod._sign_href("https://x/a.tif") == "https://x/a.tif?signed=1"
        assert DinamisClient.sign("https://x/a.tif") == "https://x/a.tif?signed=1"

    def test_missing_sdk_gives_an_actionable_error(self, monkeypatch):
        import builtins

        real = builtins.__import__

        def fake(name, *a, **k):
            if name == "teledetection":
                raise ImportError("no sdk")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake)
        with pytest.raises(ImportError, match="pip install teledetection"):
            mod._sign_href("https://x/a.tif")


class TestNameFields:
    def test_parses_the_airbus_asset_name(self):
        f = mod._name_fields(ITEM, "src_pan")
        assert f["mission"] == "SPOT7" and f["mission_short"] == "S7" and f["platform_num"] == "7"
        assert f["sensor"] == "P" and f["sensor_tag"] == "S7P"
        assert f["stamp"] == "201806231000132" and f["time"] == "1000132"
        assert f["date"] == "20180623" and (f["year"], f["month"], f["day"]) == ("2018", "06", "23")
        assert f["geom_level"] == "ORT" and f["ext"] == "tif" and f["suffix"] == ""

    @pytest.mark.parametrize(
        "asset, suffix, ext",
        [("dimap_pan", "_metadata", "xml"), ("cld_msk_vec_pan", "_cloudmask", "gpkg"),
         ("roi_msk_vec_pan", "_roimask", "gpkg")],
    )
    def test_sidecar_assets_get_distinct_suffixes(self, asset, suffix, ext):
        f = mod._name_fields(ITEM, asset)
        assert (f["suffix"], f["ext"]) == (suffix, ext)

    def test_cloud_and_roi_masks_do_not_collide(self):
        names = {
            DEFAULT_FILENAME_TEMPLATE.format(**mod._name_fields(ITEM, k))
            for k in ("cld_msk_vec_pan", "roi_msk_vec_pan")
        }
        assert len(names) == 2

    def test_unparseable_name_falls_back_to_stac_datetime_with_a_warning(self, caplog):
        item = {**ITEM, "assets": {"src_pan": {"href": "https://x/weird.tif"}}}
        with caplog.at_level(logging.WARNING, logger="dinamis_spot_image_search_download"):
            f = mod._name_fields(item, "src_pan")
        assert f["stamp"] == "20180623100013"
        assert any("could not parse an acquisition stamp" in r.message for r in caplog.records)

    def test_unknown_asset_does_not_raise(self):
        f = mod._name_fields(ITEM, "mystery")
        assert f["ext"] == "bin" and f["sensor"] == ""


class TestFilenames:
    def test_default_template(self, stub_stac):
        assert _finder(stub_stac)._clean_filename(ITEM, "src_pan") == "IMG_S7P_201806231000132.tif"

    def test_legacy_template(self, stub_stac):
        f = _finder(stub_stac, filename_template=LEGACY_FILENAME_TEMPLATE)
        assert f._clean_filename(ITEM, "src_pan") == "IMG_SPOT7_P_20180623.tif"
        assert f._clean_filename(ITEM, "dimap_pan") == "IMG_SPOT7_P_20180623_metadata.xml"

    def test_overrides_replace_fields(self, stub_stac):
        f = _finder(stub_stac, filename_template="{sensor_tag}_{stamp}{suffix}.{ext}")
        assert f._clean_filename(ITEM, "src_pan", suffix="_clip") == "S7P_201806231000132_clip.tif"

    def test_unknown_placeholder_lists_the_available_ones(self, stub_stac):
        f = _finder(stub_stac, filename_template="{nope}.tif")
        with pytest.raises(ValueError, match="Available placeholders"):
            f._clean_filename(ITEM, "src_pan")

    def test_preview_filenames_uses_cache_and_never_downloads(self, stub_stac):
        f = _finder(stub_stac)
        f._item_cache["scene-1"] = ITEM
        frame = pd.DataFrame({"image_id": ["scene-1"]})
        out = f.preview_filenames(frame, assets=("src_pan", "dimap_pan"))
        assert list(out["filename"]) == [
            "IMG_S7P_201806231000132.tif",
            "IMG_S7P_201806231000132_metadata.xml",
        ]

    def test_preview_respects_max_images(self, stub_stac):
        f = _finder(stub_stac)
        f._item_cache.update({"a": ITEM, "b": ITEM})
        out = f.preview_filenames(pd.DataFrame({"image_id": ["a", "b"]}), max_images=1)
        assert len(out) == 1


class TestFinder:
    def test_aoi_forms(self, stub_stac):
        assert _finder(stub_stac).bbox == (6.0, 45.0, 6.1, 45.1)
        client = DinamisClient(access_key="ak", secret_key="sk")
        assert DinamisImageFinder(client, (6.0, 45.0, 6.1, 45.1)).bbox == (6.0, 45.0, 6.1, 45.1)

    def test_bad_aoi(self, stub_stac):
        client = DinamisClient(access_key="ak", secret_key="sk")
        with pytest.raises(TypeError):
            DinamisImageFinder(client, "Chamonix")

    def test_aoi_area(self, stub_stac):
        assert 50 < _finder(stub_stac).aoi_area_km2 < 100  # 0.1 deg x 0.1 deg at 45N ~ 78 km2

    def test_resolve_item_hits_the_cache_first(self, stub_stac):
        f = _finder(stub_stac)
        f._item_cache["scene-1"] = ITEM
        assert f._resolve_item("scene-1") is ITEM

    def test_resolve_item_unknown_id(self, stub_stac):
        f = _finder(stub_stac)
        f.client._stac = SimpleNamespace(
            search=lambda **k: SimpleNamespace(items_as_dicts=lambda: iter([]))
        )
        with pytest.raises(RuntimeError, match="not found"):
            f._resolve_item("ghost")


class TestStreamDownload:
    class _Resp:
        def __init__(self, chunks, fail=False):
            self.chunks, self.fail = chunks, fail

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            if self.fail:
                raise RuntimeError("403")

        def iter_content(self, chunk_size):
            return iter(self.chunks)

    def test_signs_each_url_right_before_the_request(self, stub_stac, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(mod, "_sign_href", lambda u: seen.setdefault("signed", u + "?sig"))
        monkeypatch.setattr(mod.requests, "get", lambda url, **k: seen.update(url=url) or self._Resp([b"a", b"b"]))
        out = tmp_path / "sub" / "x.tif"
        _finder(stub_stac)._stream_download("https://x/a.tif", out, timeout=5)
        assert seen["url"] == "https://x/a.tif?sig" and out.read_bytes() == b"ab"

    def test_failure_leaves_neither_part_file_nor_target(self, stub_stac, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "_sign_href", lambda u: u)
        monkeypatch.setattr(mod.requests, "get", lambda *a, **k: self._Resp([b"a"], fail=True))
        out = tmp_path / "x.tif"
        with pytest.raises(RuntimeError):
            _finder(stub_stac)._stream_download("https://x/a.tif", out, timeout=5)
        assert not out.exists() and not list(tmp_path.glob("*.part"))


# ---------------------------------------------------------------------------
# AOI coercion, search, downloads, COG clips
# ---------------------------------------------------------------------------
import copy

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import mapping

AOI = box(6.0, 45.0, 6.1, 45.1)


@pytest.fixture
def mk(monkeypatch, stub_stac):
    """Build a finder with the TLD_* variables pre-registered for cleanup."""
    monkeypatch.setenv("TLD_ACCESS_KEY", "ak")
    monkeypatch.setenv("TLD_SECRET_KEY", "sk")

    def _mk(aoi=AOI, **kw):
        return DinamisImageFinder(DinamisClient(access_key="ak", secret_key="sk"), aoi, **kw)

    return _mk


def _stac_item(item_id, dt, cloud=None, platform="SPOT-7"):
    props = {"datetime": dt, "platform": platform}
    if cloud is not None:
        props["eo:cloud_cover"] = cloud
    return {
        "id": item_id,
        "properties": props,
        "geometry": mapping(AOI),
        "assets": ITEM["assets"],
    }


def _stub_search(finder, items):
    seen = {}

    def search(**kw):
        seen.update(kw)
        return SimpleNamespace(items_as_dicts=lambda: iter(items))

    finder.client._stac = SimpleNamespace(search=search)
    return seen


class TestAoiCoercion:
    def test_geodataframe_in_another_crs_is_reprojected(self, mk):
        gdf = gpd.GeoDataFrame(geometry=[AOI], crs="EPSG:4326").to_crs(2154)
        b = mk(gdf).bbox
        assert b == pytest.approx((6.0, 45.0, 6.1, 45.1), abs=1e-6)

    def test_geodataframe_already_wgs84_is_kept(self, mk):
        gdf = gpd.GeoDataFrame(geometry=[AOI], crs="EPSG:4326")
        assert mk(gdf).bbox == (6.0, 45.0, 6.1, 45.1)

    def test_shapely_and_geojson_mapping(self, mk):
        assert mk(AOI).bbox == (6.0, 45.0, 6.1, 45.1)
        assert mk(mapping(AOI)).bbox == (6.0, 45.0, 6.1, 45.1)


class TestSearchMetadata:
    def test_default_range_sort_and_cache(self, mk):
        f = mk(year_range=(2018, 2019))
        seen = _stub_search(f, [_stac_item("b", "2018-08-01T10:00:00Z", cloud=5), _stac_item("a", "2018-06-01T10:00:00Z")])
        gdf = f.search_metadata(limit=7)
        assert seen["datetime"] == "2018-01-01/2019-12-31" and seen["limit"] == 7
        assert seen["bbox"] == f.bbox and seen["collections"] == [f.collection]
        assert list(gdf["image_id"]) == ["a", "b"]  # ascending date
        assert gdf.crs.to_epsg() == 4326 and set(f._item_cache) == {"a", "b"}
        assert gdf.loc[1, "cloud_percent"] == 5

    def test_explicit_dates_override_and_open_end(self, mk):
        f = mk()
        seen = _stub_search(f, [])
        f.search_metadata(date_start="2019-01-01")
        assert seen["datetime"] == "2019-01-01/.."
        f.search_metadata(date_end="2020-01-01")
        assert seen["datetime"] == "../2020-01-01"

    def test_month_and_cloud_filters(self, mk):
        f = mk(month_range=(6, 8), cloud_max=20)
        _stub_search(f, [
            _stac_item("winter", "2018-01-10T10:00:00Z"),
            _stac_item("cloudy", "2018-07-10T10:00:00Z", cloud=80),
            _stac_item("ok", "2018-07-11T10:00:00Z", cloud=10),
            _stac_item("unknown_cloud", "2018-07-12T10:00:00Z"),
        ])
        assert set(f.search_metadata()["image_id"]) == {"ok", "unknown_cloud"}

    def test_no_rows_gives_typed_empty_frame(self, mk):
        f = mk()
        _stub_search(f, [])
        gdf = f.search_metadata()
        assert gdf.empty and gdf.crs.to_epsg() == 4326
        assert {"image_id", "acquisition_date", "assets"} <= set(gdf.columns)


class TestDownloadImages:
    def _patch_stream(self, monkeypatch):
        calls = []

        def fake(self, href, out_path, timeout=300.0):
            calls.append((href, Path(out_path), timeout))
            Path(out_path).write_bytes(b"x")

        monkeypatch.setattr(DinamisImageFinder, "_stream_download", fake)
        return calls

    def test_missing_image_id_column(self, mk, tmp_path):
        with pytest.raises(ValueError, match="image_id"):
            mk().download_images(pd.DataFrame({"x": [1]}), output_dir=tmp_path)

    def test_metadata_sibling_is_appended_by_default(self, mk, tmp_path, monkeypatch):
        calls = self._patch_stream(monkeypatch)
        f = mk()
        f._item_cache["scene-1"] = ITEM
        out = f.download_images(pd.DataFrame({"image_id": ["scene-1"]}), output_dir=tmp_path / "o", show_progress=False)
        assert [p.name for p in out] == ["IMG_S7P_201806231000132.tif", "IMG_S7P_201806231000132_metadata.xml"]
        assert len(calls) == 2 and all(c[2] == 300.0 for c in calls)

    def test_metadata_can_be_disabled(self, mk, tmp_path, monkeypatch):
        self._patch_stream(monkeypatch)
        f = mk()
        f._item_cache["scene-1"] = ITEM
        out = f.download_images(
            pd.DataFrame({"image_id": ["scene-1"]}), output_dir=tmp_path, download_metadata=False, show_progress=False
        )
        assert len(out) == 1

    def test_unknown_asset_warns_and_is_skipped(self, mk, tmp_path, monkeypatch, caplog):
        calls = self._patch_stream(monkeypatch)
        f = mk()
        f._item_cache["scene-1"] = ITEM
        with caplog.at_level(logging.WARNING, logger="dinamis_spot_image_search_download"):
            out = f.download_images(
                pd.DataFrame({"image_id": ["scene-1"]}), output_dir=tmp_path, assets=("src_xs",),
                download_metadata=False, show_progress=False,
            )
        assert out == [] and calls == []
        assert any("not found on item" in r.message for r in caplog.records)

    def test_existing_file_is_skipped_unless_overwrite(self, mk, tmp_path, monkeypatch):
        calls = self._patch_stream(monkeypatch)
        f = mk()
        f._item_cache["scene-1"] = ITEM
        frame = pd.DataFrame({"image_id": ["scene-1"]})
        kw = dict(output_dir=tmp_path, download_metadata=False, show_progress=False)
        f.download_images(frame, **kw)
        assert len(calls) == 1
        out = f.download_images(frame, **kw)
        assert len(calls) == 1 and len(out) == 1  # returned, not re-downloaded
        f.download_images(frame, overwrite=True, **kw)
        assert len(calls) == 2

    def test_max_images_and_progress_bar(self, mk, tmp_path, monkeypatch):
        calls = self._patch_stream(monkeypatch)
        f = mk()
        f._item_cache.update({"a": ITEM, "b": ITEM})
        f.download_images(
            pd.DataFrame({"image_id": ["a", "b"]}), output_dir=tmp_path, download_metadata=False, max_images=1,
            show_progress=True,
        )
        assert len(calls) == 1


class TestDownloadCogClip:
    HREF_NAME = PAN_HREF.split("?")[0].rsplit("/", 1)[1]

    def _raster(self, tmp_path, bounds=(5.9, 44.9, 6.2, 45.2)):
        path = tmp_path / self.HREF_NAME
        w = 60
        res = (bounds[2] - bounds[0]) / w
        data = np.arange(w * w, dtype="uint16").reshape(1, w, w)
        with rasterio.open(
            path, "w", driver="GTiff", height=w, width=w, count=1, dtype="uint16", crs="EPSG:4326",
            transform=from_origin(bounds[0], bounds[3], res, res),
        ) as dst:
            dst.write(data)
        return path

    def _item(self, path):
        item = copy.deepcopy(ITEM)
        item["assets"]["src_pan"]["href"] = str(path)
        return item

    def test_clips_to_the_aoi(self, mk, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "_sign_href", lambda h: h)
        f = mk()
        f._item_cache["scene-1"] = self._item(self._raster(tmp_path))
        out = f.download_cog_clip(
            pd.DataFrame({"image_id": ["scene-1"]}), output_dir=tmp_path / "clips", assets=("src_pan",),
            show_progress=False,
        )
        assert len(out) == 1 and out[0].name == "IMG_S7P_201806231000132_clip.tif"
        with rasterio.open(out[0]) as src:
            assert 0 < src.width < 60 and 0 < src.height < 60
            assert src.bounds.left == pytest.approx(6.0, abs=0.01) and src.bounds.top == pytest.approx(45.1, abs=0.01)

    def test_existing_clip_is_skipped_unless_overwrite(self, mk, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "_sign_href", lambda h: h)
        f = mk()
        f._item_cache["scene-1"] = self._item(self._raster(tmp_path))
        frame = pd.DataFrame({"image_id": ["scene-1"]})
        kw = dict(output_dir=tmp_path / "c", assets=("src_pan",), show_progress=False)
        (out,) = f.download_cog_clip(frame, **kw)
        out.write_bytes(b"sentinel")
        f.download_cog_clip(frame, **kw)
        assert out.read_bytes() == b"sentinel"
        f.download_cog_clip(frame, overwrite=True, **kw)
        assert out.read_bytes() != b"sentinel"

    def test_unknown_asset_warns(self, mk, tmp_path, monkeypatch, caplog):
        f = mk()
        f._item_cache["scene-1"] = ITEM
        with caplog.at_level(logging.WARNING, logger="dinamis_spot_image_search_download"):
            out = f.download_cog_clip(pd.DataFrame({"image_id": ["scene-1"]}), output_dir=tmp_path, show_progress=False)
        assert out == [] and any("not found on item" in r.message for r in caplog.records)

    def test_scene_missing_the_aoi_is_skipped_not_fatal(self, mk, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(mod, "_sign_href", lambda h: h)
        far = self._raster(tmp_path, bounds=(10.0, 10.0, 10.3, 10.3))
        f = mk()
        f._item_cache["far"] = self._item(far)
        with caplog.at_level(logging.WARNING, logger="dinamis_spot_image_search_download"):
            out = f.download_cog_clip(
                pd.DataFrame({"image_id": ["far"]}), output_dir=tmp_path / "c", assets=("src_pan",), show_progress=False
            )
        assert out == [] and any("does not overlap the AOI" in r.message for r in caplog.records)


class TestResolveAndNames:
    def test_search_path_caches_the_item(self, mk):
        f = mk()
        calls = []

        def search(**kw):
            calls.append(kw)
            return SimpleNamespace(items_as_dicts=lambda: iter([ITEM]))

        f.client._stac = SimpleNamespace(search=search)
        assert f._resolve_item("scene-1") is ITEM
        assert f._resolve_item("scene-1") is ITEM
        assert len(calls) == 1 and calls[0]["ids"] == ["scene-1"]

    def test_name_fields_delegates(self):
        assert DinamisImageFinder.name_fields(ITEM, "src_pan") == mod._name_fields(ITEM, "src_pan")
