#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_planet_image_search_download.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""Planet client and finder: auth, filters, perspective groups, quota-guarded orders."""
import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box, mapping

from geomulticorr.data_sources.planet_image_search_download import (
    PlanetClient,
    PlanetImageFinder,
)

AOI = box(6.0, 45.0, 6.1, 45.1)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _finder(**kw):
    return PlanetImageFinder(PlanetClient(api_key="k"), aoi=AOI, **kw)


def _scenes(n=2, inter=10.0, full=50.0):
    return gpd.GeoDataFrame(
        {
            "image_id": [f"s{i}" for i in range(n)],
            "aoi_intersection_km2": [inter] * n,
            "footprint_area_km2": [full] * n,
        },
        geometry=[AOI] * n,
        crs="EPSG:4326",
    )


class TestClient:
    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="PL_API_KEY"):
            PlanetClient()

    def test_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("PL_API_KEY", "from-env")
        assert PlanetClient().session.auth == ("from-env", "")

    def test_argument_wins_over_environment(self, monkeypatch):
        monkeypatch.setenv("PL_API_KEY", "from-env")
        assert PlanetClient(api_key="arg").api_key == "arg"

    def test_get_quota_sums_active_subscriptions_only(self, monkeypatch):
        client = PlanetClient(api_key="k")
        subs = [
            {"quota_sqkm": 100, "quota_used": 30, "state": "active"},
            {"quota_sqkm": 50, "quota_used": None},  # state defaults to active
            {"quota_sqkm": 999, "quota_used": 0, "state": "expired"},
            {"quota_sqkm": None, "quota_used": 5},
        ]
        monkeypatch.setattr(client.session, "get", lambda *a, **k: _Resp(subs))
        q = client.get_quota()
        assert (q["total_km2"], q["used_km2"], q["remaining_km2"]) == (150.0, 30.0, 120.0)
        assert q["subscriptions"] == subs


class TestSearchFilter:
    def test_base_clauses(self):
        f = _finder(year_range=(2021, 2022), cloud_max=20).build_search_filter()
        assert f["type"] == "AndFilter"
        by_field = {c["field_name"]: c for c in f["config"]}
        assert by_field["geometry"]["config"] == mapping(AOI)
        assert by_field["acquired"]["config"] == {
            "gte": "2021-01-01T00:00:00Z",
            "lte": "2022-12-31T23:59:59Z",
        }
        assert by_field["cloud_cover"]["config"] == {"lte": 0.2}

    def test_optional_clauses_only_when_requested(self):
        base = {c["field_name"] for c in _finder().build_search_filter()["config"]}
        assert not {"clear_percent", "instrument", "view_angle"} & base
        full = {
            c["field_name"]: c["config"]
            for c in _finder(
                clear_min=80, instruments=["PSB.SD"], max_view_angle=5
            ).build_search_filter()["config"]
        }
        assert full["clear_percent"] == {"gte": 80.0}
        assert full["instrument"] == ["PSB.SD"]
        assert full["view_angle"] == {"gte": -5.0, "lte": 5.0}


class TestPerspective:
    @pytest.mark.parametrize(
        "az, sign", [(10, 1), (100, 1), (189.9, 1), (190, -1), (9.9, -1), (350, -1), (370, 1)]
    )
    def test_signed_view_angle(self, az, sign):
        assert PlanetImageFinder._signed_view_angle(3.0, az) == sign * 3.0

    def test_missing_inputs_give_none(self):
        assert PlanetImageFinder._signed_view_angle(None, 100) is None
        assert PlanetImageFinder._signed_view_angle(3.0, None) is None

    def test_groups_split_on_gaps_per_instrument(self):
        gdf = gpd.GeoDataFrame(
            {
                "instrument": ["PSB.SD"] * 4 + ["PS2"] * 2 + ["PS2"],
                "signed_view_angle": [0.0, 0.2, 2.0, 2.3, 0.0, 0.1, np.nan],
            },
            geometry=[AOI] * 7,
            crs="EPSG:4326",
        )
        out = PlanetImageFinder.assign_perspective_groups(gdf, va_diff_thresh=0.5)
        assert list(out["perspective_group"][:4]) == ["PSB.SD#g1"] * 2 + ["PSB.SD#g2"] * 2
        assert list(out["perspective_group"][4:6]) == ["PS2#g1"] * 2
        assert out["perspective_group"].iloc[6] is None
        assert list(out["group_size"]) == [2, 2, 2, 2, 2, 2, 0]
        assert "perspective_group" not in gdf.columns  # input not mutated


class TestOrders:
    def test_cost_clipped_vs_full(self):
        f = _finder()
        assert f.estimate_order_cost_km2(_scenes(), clip=True) == 20.0
        assert f.estimate_order_cost_km2(_scenes(), clip=False) == 100.0

    def test_min_charge_floors_each_scene(self):
        assert _finder().estimate_order_cost_km2(_scenes(inter=1.0), min_charge_km2=4.0) == 8.0

    def test_request_body(self):
        req = _finder().build_order_request(_scenes(), order_name="o", harmonize_to_sentinel2=True)
        assert req["products"][0]["item_ids"] == ["s0", "s1"]
        assert req["products"][0]["product_bundle"] == "analytic_sr_udm2"
        assert [list(t)[0] for t in req["tools"]] == ["clip", "harmonize"]
        assert req["delivery"] == {"archive_type": "zip", "single_archive": True}

    def test_request_without_tools_or_archive(self):
        req = _finder().build_order_request(_scenes(), order_name="o", clip=False, archive_zip=False)
        assert "tools" not in req and "delivery" not in req

    def test_empty_selection_is_refused(self):
        with pytest.raises(ValueError, match="no scenes"):
            _finder().build_order_request(_scenes(0), order_name="o")


class TestSubmitOrderGuards:
    """The quota guards are what stand between a typo and a spent budget."""

    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        finder = _finder()
        gdf = _scenes()  # 2 scenes x 10 km2 = 20 km2
        req = finder.build_order_request(gdf, order_name="o")
        posts = []

        def post(url, json=None, timeout=None):
            posts.append(json)
            return _Resp({"id": "order-1", "state": "queued"})

        monkeypatch.setattr(finder._client.session, "post", post)

        def set_quota(remaining):
            monkeypatch.setattr(
                finder._client,
                "get_quota",
                lambda: {"remaining_km2": remaining, "used_km2": 0.0, "total_km2": remaining},
            )

        set_quota(1000.0)
        return finder, gdf, req, posts, tmp_path / "ledger.json", set_quota

    def test_refused_when_over_quota_and_nothing_is_sent(self, setup):
        finder, gdf, req, posts, ledger, set_quota = setup
        set_quota(5.0)
        with pytest.raises(RuntimeError, match="remaining quota"):
            finder.submit_order(req, ledger_path=ledger, images_gdf=gdf, require_confirmation=False)
        assert posts == []

    def test_refused_when_over_budget(self, setup):
        finder, gdf, req, posts, ledger, _ = setup
        with pytest.raises(RuntimeError, match="budget"):
            finder.submit_order(
                req, ledger_path=ledger, images_gdf=gdf, quota_budget_km2=10, require_confirmation=False
            )
        assert posts == []

    def test_dry_run_sends_nothing(self, setup):
        finder, gdf, req, posts, ledger, _ = setup
        assert finder.submit_order(req, ledger_path=ledger, images_gdf=gdf, dry_run=True) is None
        assert posts == [] and not ledger.exists()

    def test_declined_confirmation_sends_nothing(self, setup, monkeypatch):
        finder, gdf, req, posts, ledger, _ = setup
        monkeypatch.setattr("builtins.input", lambda *_: "no")
        assert finder.submit_order(req, ledger_path=ledger, images_gdf=gdf) is None
        assert posts == []

    def test_success_writes_the_ledger_and_flags_double_orders(self, setup, capsys):
        finder, gdf, req, posts, ledger, _ = setup
        oid = finder.submit_order(req, ledger_path=ledger, images_gdf=gdf, require_confirmation=False)
        assert oid == "order-1" and len(posts) == 1
        entry = json.loads(ledger.read_text())[0]
        assert entry["item_ids"] == ["s0", "s1"] and entry["estimated_km2"] == 20.0

        finder.submit_order(req, ledger_path=ledger, images_gdf=gdf, require_confirmation=False)
        assert "possible double charge" in capsys.readouterr().out
        assert len(json.loads(ledger.read_text())) == 2

    def test_unreadable_quota_does_not_block_the_order(self, setup, monkeypatch):
        finder, gdf, req, posts, ledger, _ = setup

        def boom():
            raise RuntimeError("endpoint unavailable")

        monkeypatch.setattr(finder._client, "get_quota", boom)
        assert finder.submit_order(req, ledger_path=ledger, images_gdf=gdf, require_confirmation=False)


class TestAoiCoercion:
    def test_forms(self):
        c = PlanetImageFinder._coerce_to_wgs84_geometry
        assert c(AOI) == AOI
        assert c(mapping(AOI)).equals(AOI)
        gdf = gpd.GeoDataFrame(geometry=[AOI], crs="EPSG:4326").to_crs(2154)
        assert c(gdf).symmetric_difference(AOI).area < 1e-9

    def test_unsupported(self):
        with pytest.raises(TypeError):
            PlanetImageFinder._coerce_to_wgs84_geometry("Chamonix")


class TestSearchMetadata:
    def _feature(self, fid, acquired, **props):
        return {
            "id": fid,
            "geometry": mapping(box(5.95, 44.95, 6.05, 45.05)),
            "properties": {"acquired": acquired, "instrument": "PSB.SD", "cloud_cover": 0.1,
                           "view_angle": 2.0, "satellite_azimuth": 100.0, **props},
        }

    def test_filters_months_and_builds_columns(self, monkeypatch):
        finder = _finder(month_range=(6, 8))
        feats = [
            self._feature("summer", "2022-07-01T10:00:00Z"),
            self._feature("winter", "2022-01-01T10:00:00Z"),
        ]
        monkeypatch.setattr(finder._client.session, "post", lambda *a, **k: _Resp({"features": feats}))
        gdf = finder.search_metadata()
        assert list(gdf["image_id"]) == ["summer"]
        row = gdf.iloc[0]
        assert row["cloud_percent"] == pytest.approx(10.0)
        assert row["signed_view_angle"] == 2.0
        assert 0 < row["aoi_coverage_percent"] <= 100

    def test_follows_pagination(self, monkeypatch):
        finder = _finder()
        first = {"features": [self._feature("a", "2022-07-01T10:00:00Z")], "_links": {"_next": "next-url"}}
        second = {"features": [self._feature("b", "2022-07-02T10:00:00Z")]}
        monkeypatch.setattr(finder._client.session, "post", lambda *a, **k: _Resp(first))
        monkeypatch.setattr(finder._client.session, "get", lambda url, **k: _Resp(second))
        assert set(finder.search_metadata()["image_id"]) == {"a", "b"}

    def test_no_result_gives_empty_frame_with_columns(self, monkeypatch):
        finder = _finder()
        monkeypatch.setattr(finder._client.session, "post", lambda *a, **k: _Resp({"features": []}))
        gdf = finder.search_metadata()
        assert gdf.empty and "image_id" in gdf.columns
