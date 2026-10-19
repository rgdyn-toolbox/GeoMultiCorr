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
