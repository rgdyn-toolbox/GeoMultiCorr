#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_gee_image_search_download.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""Google Earth Engine: connection logic (mocked), catalog and AOI handling.

No test initialises Earth Engine or authenticates: ``ee.Initialize`` and
``ee.Authenticate`` are replaced wherever ``connect`` could reach them.
"""

import ee
import geopandas as gpd
import pytest
from shapely.geometry import box, mapping

from geomulticorr.data_sources import gee_image_search_download as mod
from geomulticorr.data_sources.gee_image_search_download import (
    CollectionSpec,
    GEEClient,
    ImageCollectionCatalog,
    ImageFinder,
)

AOI = box(6.0, 45.0, 6.1, 45.1)


@pytest.fixture
def ee_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(ee.data, "_initialized", False, raising=False)
    monkeypatch.setattr(ee, "Initialize", lambda **k: calls.append(("init", k)))
    monkeypatch.setattr(ee, "Authenticate", lambda **k: calls.append(("auth", k)))
    monkeypatch.setattr(
        ee, "ServiceAccountCredentials", lambda sa, key: ("creds", sa, key)
    )
    return calls


class _FakeGeometry(ee.geometry.Geometry):
    """Offline stand-in: a real ``ee.Geometry`` needs an initialised EE client."""

    def __init__(
        self, geo=None, *a, **k
    ):  # noqa: D401 - deliberately skips super().__init__
        self.geo = geo


@pytest.fixture(autouse=True)
def offline_geometry(monkeypatch):
    monkeypatch.setattr(ee, "Geometry", _FakeGeometry)


class TestClient:
    def test_connect_is_idempotent(self, monkeypatch, ee_calls):
        monkeypatch.setattr(ee.data, "_initialized", True, raising=False)
        c = GEEClient(project="p")
        assert c.connect() is c
        assert ee_calls == []

    def test_plain_initialise(self, ee_calls):
        GEEClient().connect()
        assert ee_calls == [("init", {})]

    def test_project_is_forwarded(self, ee_calls):
        GEEClient(project="my-proj").connect()
        assert ee_calls == [("init", {"project": "my-proj"})]

    def test_service_account_needs_a_key_file(self, ee_calls):
        with pytest.raises(ValueError, match="key_file"):
            GEEClient(service_account="sa@x.iam").connect()

    def test_service_account_credentials(self, ee_calls):
        GEEClient(service_account="sa@x.iam", key_file="k.json", project="p").connect()
        assert ee_calls == [
            ("init", {"credentials": ("creds", "sa@x.iam", "k.json"), "project": "p"})
        ]

    def test_interactive_auth_only_as_a_fallback(self, monkeypatch, ee_calls):
        n = {"init": 0}

        def flaky(**k):
            n["init"] += 1
            if n["init"] == 1:
                raise RuntimeError("not authenticated")
            ee_calls.append(("init", k))

        monkeypatch.setattr(ee, "Initialize", flaky)
        GEEClient().connect()
        assert [c[0] for c in ee_calls] == ["auth", "init"]

    def test_no_interactive_auth_when_disabled(self, monkeypatch, ee_calls):
        monkeypatch.setattr(
            ee, "Initialize", lambda **k: (_ for _ in ()).throw(RuntimeError("nope"))
        )
        with pytest.raises(RuntimeError, match="nope"):
            GEEClient(authenticate_if_needed=False).connect()
        assert ee_calls == []

    def test_service_account_failure_never_opens_a_browser(self, monkeypatch, ee_calls):
        monkeypatch.setattr(
            ee, "Initialize", lambda **k: (_ for _ in ()).throw(RuntimeError("denied"))
        )
        with pytest.raises(RuntimeError, match="denied"):
            GEEClient(service_account="sa", key_file="k").connect()
        assert ee_calls == []


class TestCatalog:
    def test_default_keys(self):
        assert ImageCollectionCatalog.default().keys() == [
            "L9_SR",
            "L8_SR",
            "L7_SR",
            "L5_SR",
            "S2_SR",
            "L9_TOA",
            "L8_TOA",
            "L7_TOA",
        ]

    @pytest.mark.parametrize(
        "key, cloud", [("L8_SR", "CLOUD_COVER"), ("S2_SR", "CLOUDY_PIXEL_PERCENTAGE")]
    )
    def test_cloud_property_per_family(self, key, cloud):
        assert ImageCollectionCatalog.default().get(key).cloud_property == cloud

    def test_toa_entries_select_the_panchromatic_band(self):
        cat = ImageCollectionCatalog.default()
        assert all(
            cat.get(k).bands == ["B8"] and cat.get(k).nominal_scale_m == 15
            for k in ("L9_TOA", "L8_TOA", "L7_TOA")
        )

    def test_unknown_key(self):
        with pytest.raises(KeyError):
            ImageCollectionCatalog.default().get("nope")

    def test_spec_is_immutable(self):
        with pytest.raises(Exception):
            CollectionSpec("k", "id", "CLOUD", 30, "n").key = "z"


class TestAoi:
    def test_shapely(self):
        g = ImageFinder._coerce_to_ee_geometry(AOI)
        assert isinstance(g, _FakeGeometry) and g.geo == mapping(AOI)

    def test_geojson_mapping(self):
        g = ImageFinder._coerce_to_ee_geometry(mapping(AOI))
        assert g.geo == mapping(AOI)

    def test_ee_geometry_passes_through(self):
        g = _FakeGeometry({"type": "Point"})
        assert ImageFinder._coerce_to_ee_geometry(g) is g

    def test_geodataframe_is_reprojected_to_wgs84(self):
        gdf = gpd.GeoDataFrame(geometry=[AOI], crs="EPSG:4326").to_crs(2154)
        from shapely.geometry import shape

        got = shape(ImageFinder._coerce_to_ee_geometry(gdf).geo)
        assert got.symmetric_difference(AOI).area < 1e-9

    def test_unsupported(self):
        with pytest.raises(TypeError):
            ImageFinder._coerce_to_ee_geometry(42)


class TestFinderExportCrs:
    def _finder(self, aoi, **kw):
        return ImageFinder(GEEClient(), ImageCollectionCatalog.default(), aoi=aoi, **kw)

    def test_geographic_aoi_exports_in_wgs84(self):
        assert self._finder(AOI)._aoi_export_crs == "EPSG:4326"

    def test_projected_geodataframe_keeps_its_crs(self):
        gdf = gpd.GeoDataFrame(geometry=[AOI], crs="EPSG:4326").to_crs(2154)
        assert self._finder(gdf)._aoi_export_crs == "EPSG:2154"

    def test_explicit_projected_aoi_crs(self):
        assert self._finder(AOI, aoi_crs=32632)._aoi_export_crs == "EPSG:32632"

    def test_defaults(self):
        f = self._finder(AOI)
        assert (f.year_range, f.month_range, f.cloud_max) == ((2020, 2025), (1, 12), 30)


# ---------------------------------------------------------------------------
# connect fallback with a project, specs(), bad aoi_crs, search_metadata, bulk_download
# ---------------------------------------------------------------------------
import datetime as _datetime  # noqa: E402
import logging  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pandas as pd  # noqa: E402
from shapely.geometry import box as _box  # noqa: E402


class TestSmallBranches:
    def test_auth_fallback_reinitialises_with_project(self, monkeypatch, ee_calls):
        n = {"init": 0}

        def flaky(**k):
            n["init"] += 1
            if n["init"] == 1:
                raise RuntimeError("not authenticated")
            ee_calls.append(("init", k))

        monkeypatch.setattr(ee, "Initialize", flaky)
        GEEClient(project="p").connect()
        assert ee_calls == [("auth", {}), ("init", {"project": "p"})]

    def test_specs_lists_every_collection(self):
        cat = ImageCollectionCatalog.default()
        assert [s.key for s in cat.specs()] == cat.keys() and len(cat.specs()) == 8

    def test_unparseable_aoi_crs_falls_back_to_wgs84(self):
        f = ImageFinder(
            GEEClient(), ImageCollectionCatalog.default(), aoi=AOI, aoi_crs="not-a-crs"
        )
        assert f._aoi_export_crs == "EPSG:4326"


def _feature(image_id, date, cloud, key="L8_SR"):
    return {
        "geometry": mapping(_box(5.9, 44.9, 6.2, 45.2)),
        "properties": {
            "image_id": image_id,
            "collection_key": key,
            "collection_name": "name",
            "acquisition_date": date,
            "cloud_percent": cloud,
            "nominal_scale_m": 30,
            "footprint_area_m2": 2.0e8,
            "footprint_area_km2": 200.0,
            "aoi_coverage_percent": 100.0,
            "overlap_ratio": 0.04,
        },
    }


class _FakeFC:
    def __init__(self, features):
        self.features = features

    def filter(self, *a, **k):
        return self

    def merge(self, other):
        return _FakeFC(self.features + other.features)

    def getInfo(self):
        return {"features": self.features}


class _FakeIC:
    def __init__(self, features, on_map):
        self._features, self._on_map = features, on_map

    def filterBounds(self, *a):
        return self

    def filter(self, *a):
        return self

    def map(self, fn):
        self._on_map(fn(MagicMock()))
        return _FakeFC(self._features)


@pytest.fixture
def search_env(monkeypatch, ee_calls):
    seen = {"ids": [], "props": []}
    features = {
        "LANDSAT/LC08/C02/T1_L2": [
            _feature("LC08_a", "2023-03-01", 10.0),
            _feature("LC08_b", "2023-07-01", 20.0),
            _feature("LC08_c", "2023-07-01", 5.0),
            _feature("LC08_d", "2023-01-01", None),
        ],
        "COPERNICUS/S2_SR_HARMONIZED": [
            _feature("S2_a", "2023-07-01", 1.0, key="S2_SR")
        ],
    }

    def image_collection(ee_id):
        seen["ids"].append(ee_id)
        return _FakeIC(features.get(ee_id, []), lambda feat: None)

    monkeypatch.setattr(ee, "ImageCollection", image_collection)
    monkeypatch.setattr(ee, "Number", lambda x: MagicMock())
    monkeypatch.setattr(ee, "Date", lambda x: MagicMock())
    monkeypatch.setattr(ee, "Filter", MagicMock())
    monkeypatch.setattr(
        ee, "Feature", lambda geom, props: seen["props"].append(props) or props
    )
    monkeypatch.setattr(ee, "FeatureCollection", lambda fc: fc)
    f = ImageFinder(GEEClient(), ImageCollectionCatalog.default(), aoi=AOI)
    f.aoi_ee = MagicMock()
    f._seen = seen
    return f


class TestSearchMetadata:
    def test_sorted_by_date_desc_then_cloud_asc(self, search_env):
        gdf = search_env.search_metadata(["L8_SR"])
        assert list(gdf["image_id"]) == ["LC08_c", "LC08_b", "LC08_a", "LC08_d"]
        assert str(gdf.crs) == "EPSG:4326"
        assert pd.isna(gdf["cloud_percent"].iloc[-1])
        assert gdf["acquisition_date"].dtype.kind == "M"

    def test_only_selected_collection_is_queried(self, search_env):
        search_env.search_metadata(["L8_SR"])
        assert search_env._seen["ids"] == ["LANDSAT/LC08/C02/T1_L2"]

    def test_server_side_properties_are_built_per_image(self, search_env):
        search_env.search_metadata(["L8_SR"])
        props = search_env._seen["props"][0]
        assert props["collection_key"] == "L8_SR" and props["nominal_scale_m"] == 30
        assert {
            "overlap_ratio",
            "aoi_coverage_percent",
            "aoi_intersection_area_m2",
        } <= set(props)

    def test_two_collections_are_merged(self, search_env):
        gdf = search_env.search_metadata(["L8_SR", "S2_SR"])
        assert set(gdf["collection_key"]) == {"L8_SR", "S2_SR"} and len(gdf) == 5

    def test_empty_catalog_gives_empty_typed_frame(self, search_env):
        search_env._catalog = ImageCollectionCatalog([])
        gdf = search_env.search_metadata()
        assert gdf.empty and "image_id" in gdf.columns and str(gdf.crs) == "EPSG:4326"


class _FakeImage:
    def __init__(self, image_id):
        self.image_id, self.bands = image_id, None

    def select(self, bands):
        self.bands = bands
        return self


@pytest.fixture
def export_env(monkeypatch, tmp_path, ee_calls):
    rec = {"exports": [], "images": [], "sleeps": [], "fail": 0}

    def image(image_id):
        img = _FakeImage(image_id)
        rec["images"].append(img)
        return img

    def export(img, filename, scale, crs, region):
        if rec["fail"] > 0:
            rec["fail"] -= 1
            raise OSError("quota")
        rec["exports"].append(
            {
                "img": img,
                "filename": filename,
                "scale": scale,
                "crs": crs,
                "region": region,
            }
        )

    monkeypatch.setattr(ee, "Image", image)
    monkeypatch.setattr(mod.geemap, "ee_export_image", export)
    monkeypatch.setattr(mod._time, "sleep", lambda s: rec["sleeps"].append(s))
    f = ImageFinder(GEEClient(), ImageCollectionCatalog.default(), aoi=AOI)
    f._rec, f._out = rec, tmp_path / "dl"
    return f


def _rows(**cols):
    base = {
        "image_id": ["LC08_a", "LC08_b"],
        "collection_key": ["L8_SR", "L8_SR"],
        "nominal_scale_m": [30, 30],
        "acquisition_date": pd.to_datetime(["2023-07-01", "2023-08-02"]),
        "geometry": [_box(5.9, 44.9, 6.2, 45.2)] * 2,
    }
    base.update(cols)
    return gpd.GeoDataFrame(base, geometry="geometry", crs="EPSG:4326")


L8 = ImageCollectionCatalog.default().get("L8_SR")


class TestBulkDownload:
    def test_missing_columns(self, export_env):
        with pytest.raises(ValueError, match="missing required columns"):
            export_env.bulk_download(
                _rows().drop(columns=["nominal_scale_m"]), output_dir=export_env._out
            )

    @pytest.mark.parametrize("tqdm_on", [True, False])
    def test_paths_ids_bands_and_crs(self, export_env, monkeypatch, tqdm_on):
        monkeypatch.setattr(mod, "_HAS_TQDM", tqdm_on and mod._HAS_TQDM)
        out = export_env.bulk_download(_rows(), output_dir=export_env._out)
        assert [p.name for p in out] == [
            "L8_SR_20230701_30m.tif",
            "L8_SR_20230802_30m.tif",
        ]
        assert all(p.parent == export_env._out / "L8_SR" for p in out)
        imgs = export_env._rec["images"]
        assert imgs[0].image_id == f"{L8.ee_id}/LC08_a" and imgs[0].bands == L8.bands
        ex = export_env._rec["exports"][0]
        assert (
            ex["crs"] == "EPSG:4326"
            and ex["scale"] == 30
            and ex["region"] is export_env.aoi_ee
        )

    def test_full_image_id_is_kept_and_unknown_key_has_no_bands(self, export_env):
        gdf = _rows(image_id=["X/Y/img1", "X/Y/img2"], collection_key=["NOPE", "NOPE"])
        export_env.bulk_download(gdf, output_dir=export_env._out, max_images=1)
        img = export_env._rec["images"][0]
        assert img.image_id == "X/Y/img1" and img.bands is None
        assert len(export_env._rec["exports"]) == 1

    def test_unknown_key_with_short_id_cannot_be_rebuilt(self, export_env):
        gdf = _rows(collection_key=["NOPE", "NOPE"])
        with pytest.raises(RuntimeError, match="Cannot reconstruct"):
            export_env.bulk_download(gdf, output_dir=export_env._out)

    def test_overrides_and_footprint_region(self, export_env):
        export_env.bulk_download(
            _rows(),
            output_dir=export_env._out,
            region_mode="footprint",
            crs="EPSG:2154",
            scale_override_m=60,
            bands_override=["SR_B2"],
            max_images=1,
        )
        ex = export_env._rec["exports"][0]
        assert ex["scale"] == 60 and ex["crs"] == "EPSG:2154"
        assert (
            isinstance(ex["region"], _FakeGeometry)
            and ex["region"] is not export_env.aoi_ee
        )
        assert ex["img"].bands == ["SR_B2"]
        assert ex["filename"].endswith("L8_SR_20230701_60m.tif")

    def test_unusable_acquisition_date_gives_nodate(self, export_env):
        gdf = _rows(acquisition_date=["2023-07-01", "2023-08-02"])
        out = export_env.bulk_download(gdf, output_dir=export_env._out, max_images=1)
        assert out[0].name == "L8_SR_nodate_30m.tif"

    def test_no_date_column_gives_nodate(self, export_env):
        out = export_env.bulk_download(
            _rows().drop(columns=["acquisition_date"]),
            output_dir=export_env._out,
            max_images=1,
        )
        assert out[0].name == "L8_SR_nodate_30m.tif"

    def test_file_basename_is_currently_ignored(self, export_env):
        # pins today's behaviour: the basename-based path is built and then overwritten
        out = export_env.bulk_download(
            _rows(), output_dir=export_env._out, file_basename="custom", max_images=1
        )
        assert out[0].name == "L8_SR_20230701_30m.tif"

    def test_throttle_sleeps_after_each_export(self, export_env):
        export_env.bulk_download(_rows(), output_dir=export_env._out, throttle_s=0.3)
        assert export_env._rec["sleeps"] == [0.3, 0.3]

    def test_retry_then_success_uses_exponential_backoff(self, export_env):
        export_env._rec["fail"] = 2
        out = export_env.bulk_download(
            _rows(), output_dir=export_env._out, max_images=1, retry_backoff_s=0.5
        )
        assert len(out) == 1 and export_env._rec["sleeps"] == [0.5, 1.0]

    def test_retries_exhausted(self, export_env, caplog):
        export_env._rec["fail"] = 99
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="Last error: quota"):
                export_env.bulk_download(
                    _rows(),
                    output_dir=export_env._out,
                    max_retries=1,
                    retry_backoff_s=0.5,
                )
        assert export_env._rec["sleeps"] == [0.5]

    def test_nat_acquisition_date_gives_nodate(self, export_env):
        gdf = _rows(acquisition_date=pd.to_datetime([None, "2023-08-02"]))
        out = export_env.bulk_download(gdf, output_dir=export_env._out, max_images=1)
        assert out[0].name == "L8_SR_nodate_30m.tif"
