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
    monkeypatch.setattr(ee, "ServiceAccountCredentials", lambda sa, key: ("creds", sa, key))
    return calls


class _FakeGeometry(ee.geometry.Geometry):
    """Offline stand-in: a real ``ee.Geometry`` needs an initialised EE client."""

    def __init__(self, geo=None, *a, **k):  # noqa: D401 - deliberately skips super().__init__
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
        assert ee_calls == [("init", {"credentials": ("creds", "sa@x.iam", "k.json"), "project": "p"})]

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
        monkeypatch.setattr(ee, "Initialize", lambda **k: (_ for _ in ()).throw(RuntimeError("nope")))
        with pytest.raises(RuntimeError, match="nope"):
            GEEClient(authenticate_if_needed=False).connect()
        assert ee_calls == []

    def test_service_account_failure_never_opens_a_browser(self, monkeypatch, ee_calls):
        monkeypatch.setattr(ee, "Initialize", lambda **k: (_ for _ in ()).throw(RuntimeError("denied")))
        with pytest.raises(RuntimeError, match="denied"):
            GEEClient(service_account="sa", key_file="k").connect()
        assert ee_calls == []


class TestCatalog:
    def test_default_keys(self):
        assert ImageCollectionCatalog.default().keys() == [
            "L9_SR", "L8_SR", "L7_SR", "L5_SR", "S2_SR", "L9_TOA", "L8_TOA", "L7_TOA"
        ]

    @pytest.mark.parametrize("key, cloud", [("L8_SR", "CLOUD_COVER"), ("S2_SR", "CLOUDY_PIXEL_PERCENTAGE")])
    def test_cloud_property_per_family(self, key, cloud):
        assert ImageCollectionCatalog.default().get(key).cloud_property == cloud

    def test_toa_entries_select_the_panchromatic_band(self):
        cat = ImageCollectionCatalog.default()
        assert all(cat.get(k).bands == ["B8"] and cat.get(k).nominal_scale_m == 15
                   for k in ("L9_TOA", "L8_TOA", "L7_TOA"))

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
