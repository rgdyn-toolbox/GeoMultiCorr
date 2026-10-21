#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_mpc_image_search_download.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""Microsoft Planetary Computer: catalog, archive routing, search and asset download."""
from datetime import datetime, timezone
from types import SimpleNamespace

import geopandas as gpd
import pystac
import pytest
from shapely.geometry import box, mapping

from geomulticorr.data_sources import mpc_image_search_download as mod
from geomulticorr.data_sources.mpc_image_search_download import (
    CollectionSpec,
    ImageCollectionCatalog,
    ImageFinder,
)

AOI = box(6.0, 45.0, 6.1, 45.1)


def _item(item_id="LC08_L2SP_232083_20230301", month=3, cloud=5.0, assets=None, props=None):
    return pystac.Item(
        id=item_id,
        geometry=mapping(box(5.9, 44.9, 6.2, 45.2)),
        bbox=[5.9, 44.9, 6.2, 45.2],
        datetime=datetime(2023, month, 1, tzinfo=timezone.utc),
        properties={"eo:cloud_cover": cloud, **(props or {})},
        assets=assets or {},
    )


class TestCatalog:
    def test_default_keys(self):
        assert ImageCollectionCatalog.default().keys() == [
            "L9_SR", "L8_SR", "L7_SR", "L5_SR", "S2_SR"
        ]

    def test_landsat_entries_share_a_collection_split_by_platform(self):
        cat = ImageCollectionCatalog.default()
        assert {cat.get(k).stac_collection for k in ("L9_SR", "L8_SR")} == {"landsat-c2-l2"}
        assert cat.get("L9_SR").platform != cat.get("L8_SR").platform
        assert cat.get("S2_SR").platform is None

    def test_get_unknown_key(self):
        with pytest.raises(KeyError):
            ImageCollectionCatalog.default().get("nope")

    def test_custom_catalog(self):
        spec = CollectionSpec("X", "my-collection", 10, "X")
        cat = ImageCollectionCatalog([spec])
        assert cat.specs() == [spec] and cat.get("X") is spec

    def test_spec_is_immutable(self):
        with pytest.raises(Exception):
            ImageCollectionCatalog.default().get("S2_SR").key = "other"


class TestSceneDirs:
    def test_landsat_routes_by_collection(self):
        spec = ImageCollectionCatalog.default().get("L8_SR")
        item = _item("LC08_L2SP_232083_20230301_20230307_02_T1")
        assert mod._scene_dirs(spec, item) == ("LANDSAT8", "232_083")

    def test_sentinel2_reads_the_property(self):
        spec = ImageCollectionCatalog.default().get("S2_SR")
        item = _item("S2A_whatever", props={"s2:mgrs_tile": "T32TLR"})
        assert mod._scene_dirs(spec, item) == ("SENTINEL2", "T32TLR")

    def test_user_spec_is_routed_by_collection_not_by_key(self):
        spec = CollectionSpec("my_s2", "sentinel-2-l2a", 10, "mine")
        item = _item("x", props={"s2:mgrs_tile": "T32TLR"})
        assert mod._scene_dirs(spec, item)[0] == "SENTINEL2"

    def test_unknown_collection_is_kept_inspectable(self):
        spec = CollectionSpec("odd", "naip", 1, "NAIP")
        assert mod._scene_dirs(spec, _item("x")) == ("odd", "unknown")


class _FakeStac:
    def __init__(self, items_by_collection):
        self.items_by_collection = items_by_collection
        self.queries = []

    def search(self, collections, intersects, datetime, query):
        self.queries.append({"collections": collections, "datetime": datetime, "query": query})
        return SimpleNamespace(items=lambda: iter(self.items_by_collection[collections[0]]))


@pytest.fixture
def finder(monkeypatch):
    stac = _FakeStac(
        {
            "landsat-c2-l2": [_item("a", month=3), _item("b", month=7), _item("c", month=7, cloud=1.0)],
            "sentinel-2-l2a": [_item("far", month=7)],
        }
    )
    monkeypatch.setattr(mod.pystac_client.Client, "open", lambda *a, **k: stac)
    f = ImageFinder(ImageCollectionCatalog.default(), aoi=AOI, year_range=(2023, 2023), month_range=(6, 8))
    f._fake = stac
    return f


class TestImageFinder:
    def test_construction_does_not_need_credentials(self, finder):
        assert finder.aoi_geom.equals(AOI)

    def test_search_filters_months_and_sorts(self, finder):
        gdf = finder.search_metadata(["L8_SR"])
        # month 3 dropped; same date -> ascending cloud
        assert list(gdf["image_id"]) == ["c", "b"]
        assert set(gdf["collection_key"]) == {"L8_SR"}
        assert str(gdf.crs) == "EPSG:4326"
        assert (gdf["aoi_coverage_percent"] > 99).all()  # footprint covers the AOI

    def test_query_carries_cloud_limit_platform_and_dates(self, finder):
        finder.search_metadata(["L8_SR"])
        q = finder._fake.queries[0]
        assert q["datetime"] == "2023-01-01/2023-12-31"
        assert q["query"] == {"eo:cloud_cover": {"lt": 30}, "platform": {"eq": "landsat-8"}}

    def test_items_are_cached_for_download(self, finder):
        finder.search_metadata(["L8_SR"])
        assert "L8_SR/b" in finder._item_cache

    def test_no_hits_gives_empty_typed_frame(self, finder):
        finder._fake.items_by_collection["landsat-c2-l2"] = []
        gdf = finder.search_metadata(["L8_SR"])
        assert gdf.empty and "image_id" in gdf.columns and str(gdf.crs) == "EPSG:4326"

    def test_unknown_collection_key(self, finder):
        with pytest.raises(KeyError):
            finder.search_metadata(["nope"])


class TestAssetDownload:
    class _Resp:
        def __init__(self, chunks):
            self._chunks = chunks

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            return iter(self._chunks)

    def _item_with_asset(self, href):
        return _item(assets={"nir08": pystac.Asset(href=href)})

    def test_keeps_original_name_and_strips_sas_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod.requests, "get", lambda *a, **k: self._Resp([b"ab", b"cd"]))
        item = self._item_with_asset("https://x/y/B8.TIF?sv=2020&sig=SECRET")
        out = ImageFinder._download_asset_file(item, "nir08", tmp_path, overwrite=False)
        assert out.name == "B8.TIF" and out.read_bytes() == b"abcd"
        assert not list(tmp_path.glob("*.part"))

    def test_rename_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod.requests, "get", lambda *a, **k: self._Resp([b"x"]))
        item = self._item_with_asset("https://x/B8.TIF?sig=s")
        out = ImageFinder._download_asset_file(
            item, "nir08", tmp_path, overwrite=False, rename_prefix="pz_2023-03-01_landsat8"
        )
        assert out.name == "pz_2023-03-01_landsat8_B8.TIF"

    def test_existing_file_is_skipped_unless_overwrite(self, tmp_path, monkeypatch):
        (tmp_path / "B8.TIF").write_bytes(b"old")
        monkeypatch.setattr(mod.requests, "get", lambda *a, **k: pytest.fail("re-downloaded"))
        item = self._item_with_asset("https://x/B8.TIF?sig=s")
        assert ImageFinder._download_asset_file(item, "nir08", tmp_path, overwrite=False) is None
        assert (tmp_path / "B8.TIF").read_bytes() == b"old"

    def test_missing_asset_key(self, tmp_path):
        with pytest.raises(KeyError, match="nir08"):
            ImageFinder._download_asset_file(_item(), "nir08", tmp_path, overwrite=False)


class TestAoiCoercion:
    def test_forms(self):
        c = ImageFinder._coerce_to_wgs84_geometry
        assert c(AOI) == AOI
        assert c(mapping(AOI)).equals(AOI)
        gdf = gpd.GeoDataFrame(geometry=[AOI], crs="EPSG:4326").to_crs(2154)
        assert c(gdf).symmetric_difference(AOI).area < 1e-9

    def test_unsupported(self):
        with pytest.raises(TypeError):
            ImageFinder._coerce_to_wgs84_geometry(42)
