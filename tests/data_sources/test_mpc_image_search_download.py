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

    def search(self, collections, intersects=None, datetime=None, query=None, ids=None):
        self.queries.append(
            {"collections": collections, "datetime": datetime, "query": query, "ids": ids}
        )
        found = self.items_by_collection[collections[0]]
        if ids is not None:
            found = [i for i in found if i.id in ids]
        return SimpleNamespace(items=lambda: iter(found))


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


# ---------------------------------------------------------------------------
# search: footprints that miss the AOI, id resolution, bulk download, crop
# ---------------------------------------------------------------------------
import logging  # noqa: E402

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

L8_ID_B = "LC08_L2SP_232083_20230701_20230707_02_T1"
L8_ID_C = "LC08_L2SP_232084_20230701_20230707_02_T1"


def _make_finder(monkeypatch, items_by_collection):
    stac = _FakeStac(items_by_collection)
    monkeypatch.setattr(mod.pystac_client.Client, "open", lambda *a, **k: stac)
    f = ImageFinder(ImageCollectionCatalog.default(), aoi=AOI, year_range=(2023, 2023), month_range=(6, 8))
    f._fake = stac
    return f


def _l8_item(item_id, month=7):
    assets = {
        "nir08": pystac.Asset(href=f"https://x/{item_id}_B5.TIF?sig=s"),
        "red": pystac.Asset(href=f"https://x/{item_id}_B4.TIF?sig=s"),
        "green": pystac.Asset(href=f"https://x/{item_id}_B3.TIF?sig=s"),
        "mtl.xml": pystac.Asset(href=f"https://x/{item_id}_MTL.xml?sig=s"),
    }
    return _item(item_id, month=month, assets=assets)


@pytest.fixture
def bulk(monkeypatch, tmp_path):
    items = [_l8_item(L8_ID_B), _l8_item(L8_ID_C)]
    f = _make_finder(monkeypatch, {"landsat-c2-l2": items, "sentinel-2-l2a": []})
    gdf = f.search_metadata(["L8_SR"])
    sleeps = []
    monkeypatch.setattr(mod._time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(mod.planetary_computer, "sign", lambda item: item)
    f._sleeps = sleeps
    f._gdf = gdf
    f._out = tmp_path / "out"
    return f


class TestSearchEdges:
    def test_item_outside_aoi_is_dropped(self, monkeypatch):
        far = pystac.Item(
            id="far",
            geometry=mapping(box(10.0, 10.0, 10.1, 10.1)),
            bbox=[10.0, 10.0, 10.1, 10.1],
            datetime=datetime(2023, 7, 1, tzinfo=timezone.utc),
            properties={"eo:cloud_cover": 1.0},
        )
        f = _make_finder(monkeypatch, {"landsat-c2-l2": [far, _item("near", month=7)]})
        assert list(f.search_metadata(["L8_SR"])["image_id"]) == ["near"]


class TestResolveItem:
    def test_cache_hit_is_re_signed(self, finder, monkeypatch):
        signed = []
        monkeypatch.setattr(mod.planetary_computer, "sign", lambda it: signed.append(it.id) or it)
        finder.search_metadata(["L8_SR"])
        assert finder._resolve_item("L8_SR", "b").id == "b"
        assert signed == ["b"]

    def test_miss_searches_by_id_and_caches(self, finder):
        got = finder._resolve_item("L8_SR", "a")
        assert got.id == "a"
        assert finder._fake.queries[-1]["ids"] == ["a"]
        assert finder._item_cache["L8_SR/a"] is got

    def test_not_found(self, finder):
        with pytest.raises(RuntimeError, match="not found"):
            finder._resolve_item("L8_SR", "zzz")


class TestBulkValidation:
    def test_bad_mode(self, bulk):
        with pytest.raises(ValueError, match="mode"):
            bulk.bulk_download(bulk._gdf, output_dir=bulk._out, mode="zip")

    def test_bad_organize_by(self, bulk):
        with pytest.raises(ValueError, match="organize_by"):
            bulk.bulk_download(bulk._gdf, output_dir=bulk._out, mode="full", organize_by="nope")

    def test_missing_columns(self, bulk):
        with pytest.raises(ValueError, match="missing required columns"):
            bulk.bulk_download(bulk._gdf.drop(columns=["image_id"]), output_dir=bulk._out)


class TestBulkDownload:
    @pytest.fixture(autouse=True)
    def _record(self, monkeypatch):
        self.crops, self.assets = [], []

        def crop(self_, item, bands, out_path, *, overwrite):
            self.crops.append((item.id, list(bands), out_path))
            return out_path

        def asset(item, key, out_dir, *, overwrite, rename_prefix=None, chunk_bytes=0):
            self.assets.append((item.id, key, out_dir, rename_prefix))
            return out_dir / f"{rename_prefix or item.id}_{key}"

        monkeypatch.setattr(ImageFinder, "_download_crop", crop)
        monkeypatch.setattr(ImageFinder, "_download_asset_file", staticmethod(asset))

    @pytest.mark.parametrize("tqdm_on", [True, False])
    def test_crop_mode_names_and_metadata_prefix(self, bulk, monkeypatch, tqdm_on):
        monkeypatch.setattr(mod, "_HAS_TQDM", tqdm_on and mod._HAS_TQDM)
        out = bulk.bulk_download(bulk._gdf, output_dir=bulk._out, file_basename="pz")
        assert len(self.crops) == 2
        _id, bands, path = self.crops[0]
        assert bands == ["nir08", "red", "green"]
        assert path.parent == bulk._out / "L8_SR"
        assert path.name == f"pz_2023-07-01_{_id}.tif"
        prefixes = [a[3] for a in self.assets]
        assert all(p.startswith("pz_2023-07-01_") for p in prefixes)
        assert len(out) == 4  # two rasters + two metadata files

    def test_no_metadata_and_max_images_and_bands_override(self, bulk):
        out = bulk.bulk_download(
            bulk._gdf, output_dir=bulk._out, download_metadata=False,
            max_images=1, bands_override=["red"],
        )
        assert len(out) == 1 and self.assets == []
        assert self.crops[0][1] == ["red"]

    def test_full_mode_default_layout(self, bulk):
        bulk.bulk_download(bulk._gdf, output_dir=bulk._out, mode="full", download_metadata=False)
        dirs = {a[2] for a in self.assets}
        assert dirs == {bulk._out / "L8_SR" / L8_ID_B, bulk._out / "L8_SR" / L8_ID_C}
        assert all(d.is_dir() for d in dirs)

    def test_full_mode_sensor_pathrow_layout_and_metadata(self, bulk):
        bulk.bulk_download(
            bulk._gdf, output_dir=bulk._out, mode="full", organize_by="sensor_pathrow"
        )
        assert bulk._out / "LANDSAT8" / "232_083" / L8_ID_B in {a[2] for a in self.assets}
        assert any(a[1] == "mtl.xml" and a[3] is None for a in self.assets)

    def test_organize_by_ignored_in_crop_mode(self, bulk, caplog):
        with caplog.at_level(logging.WARNING, logger="mpc_image_search_download"):
            bulk.bulk_download(
                bulk._gdf, output_dir=bulk._out, organize_by="sensor_pathrow",
                download_metadata=False,
            )
        assert "ignored in crop mode" in caplog.text
        assert self.crops[0][2].parent == bulk._out / "L8_SR"

    def test_missing_metadata_asset_warns(self, bulk, caplog):
        for it in bulk._item_cache.values():
            del it.assets["mtl.xml"]
        with caplog.at_level(logging.WARNING, logger="mpc_image_search_download"):
            out = bulk.bulk_download(bulk._gdf, output_dir=bulk._out)
        assert "not found on item" in caplog.text and len(out) == 2

    def test_throttle_sleeps_between_scenes(self, bulk):
        bulk.bulk_download(bulk._gdf, output_dir=bulk._out, throttle_s=0.25, download_metadata=False)
        assert bulk._sleeps == [0.25, 0.25]

    def test_retry_then_success_uses_exponential_backoff(self, bulk, monkeypatch):
        calls = {"n": 0}

        def flaky(self_, item, bands, out_path, *, overwrite):
            calls["n"] += 1
            if calls["n"] < 3:
                raise OSError("transient")
            return out_path

        monkeypatch.setattr(ImageFinder, "_download_crop", flaky)
        bulk.bulk_download(
            bulk._gdf, output_dir=bulk._out, max_images=1, download_metadata=False,
            retry_backoff_s=0.5,
        )
        assert bulk._sleeps == [0.5, 1.0]

    def test_retries_exhausted_raises(self, bulk, monkeypatch):
        def boom(self_, item, bands, out_path, *, overwrite):
            raise OSError("down")

        monkeypatch.setattr(ImageFinder, "_download_crop", boom)
        with pytest.raises(RuntimeError, match="Last error: down"):
            bulk.bulk_download(
                bulk._gdf, output_dir=bulk._out, max_retries=1, retry_backoff_s=0.5,
                download_metadata=False,
            )
        assert bulk._sleeps == [0.5]


class TestDownloadCrop:
    @staticmethod
    def _write(path, west, north, res, size, value=1):
        data = np.full((size, size), value, dtype="uint16")
        with rasterio.open(
            path, "w", driver="GTiff", width=size, height=size, count=1, dtype="uint16",
            crs="EPSG:4326", transform=from_origin(west, north, res, res), nodata=0,
        ) as dst:
            dst.write(data, 1)
        return str(path)

    def _item_for(self, **hrefs):
        return _item(assets={k: pystac.Asset(href=v) for k, v in hrefs.items()})

    @pytest.fixture
    def finder_(self, monkeypatch):
        return _make_finder(monkeypatch, {"landsat-c2-l2": []})

    def test_multiband_crop_has_descriptions_and_aoi_extent(self, finder_, tmp_path):
        a = self._write(tmp_path / "a.tif", 5.95, 45.15, 0.002, 100, value=7)
        b = self._write(tmp_path / "b.tif", 5.95, 45.15, 0.002, 100, value=9)
        out = tmp_path / "crop.tif"
        got = finder_._download_crop(self._item_for(red=a, green=b), ["red", "green"], out, overwrite=False)
        assert got == out
        with rasterio.open(out) as src:
            assert src.count == 2 and src.descriptions == ("red", "green")
            assert (src.width, src.height) == (50, 50)
            assert src.read(1).max() == 7 and src.read(2).max() == 9

    def test_no_bands_raises(self, finder_, tmp_path):
        with pytest.raises(ValueError, match="at least one band"):
            finder_._download_crop(_item(), [], tmp_path / "x.tif", overwrite=False)

    def test_existing_output_skipped_unless_overwrite(self, finder_, tmp_path):
        a = self._write(tmp_path / "a.tif", 5.95, 45.15, 0.002, 100)
        out = tmp_path / "crop.tif"
        out.write_bytes(b"old")
        item = self._item_for(red=a)
        assert finder_._download_crop(item, ["red"], out, overwrite=False) is None
        assert out.read_bytes() == b"old"
        assert finder_._download_crop(item, ["red"], out, overwrite=True) == out
        with rasterio.open(out) as src:
            assert src.count == 1

    def test_missing_asset_key(self, finder_, tmp_path):
        with pytest.raises(KeyError, match="nir08"):
            finder_._download_crop(_item(), ["nir08"], tmp_path / "x.tif", overwrite=False)

    def test_aoi_outside_raster(self, finder_, tmp_path):
        a = self._write(tmp_path / "far.tif", 20.0, 30.1, 0.002, 100)
        # rasterio raises WindowError before the module's own "does not intersect" guard
        with pytest.raises(rasterio.errors.WindowError):
            finder_._download_crop(self._item_for(red=a), ["red"], tmp_path / "x.tif", overwrite=False)

    def test_bands_on_different_grids_raise(self, finder_, tmp_path):
        a = self._write(tmp_path / "a.tif", 5.95, 45.15, 0.002, 100)
        b = self._write(tmp_path / "b.tif", 5.95, 45.15, 0.004, 50)
        with pytest.raises(RuntimeError, match="grid differs"):
            finder_._download_crop(
                self._item_for(red=a, green=b), ["red", "green"], tmp_path / "x.tif", overwrite=False
            )
