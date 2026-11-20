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
import requests
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
        assert (q["total_km2"], q["used_km2"], q["remaining_km2"]) == (
            150.0,
            30.0,
            120.0,
        )
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
        "az, sign",
        [(10, 1), (100, 1), (189.9, 1), (190, -1), (9.9, -1), (350, -1), (370, 1)],
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
        assert (
            list(out["perspective_group"][:4]) == ["PSB.SD#g1"] * 2 + ["PSB.SD#g2"] * 2
        )
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
        assert (
            _finder().estimate_order_cost_km2(_scenes(inter=1.0), min_charge_km2=4.0)
            == 8.0
        )

    def test_request_body(self):
        req = _finder().build_order_request(
            _scenes(), order_name="o", harmonize_to_sentinel2=True
        )
        assert req["products"][0]["item_ids"] == ["s0", "s1"]
        assert req["products"][0]["product_bundle"] == "analytic_sr_udm2"
        assert [list(t)[0] for t in req["tools"]] == ["clip", "harmonize"]
        assert req["delivery"] == {"archive_type": "zip", "single_archive": True}

    def test_request_without_tools_or_archive(self):
        req = _finder().build_order_request(
            _scenes(), order_name="o", clip=False, archive_zip=False
        )
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
                lambda: {
                    "remaining_km2": remaining,
                    "used_km2": 0.0,
                    "total_km2": remaining,
                },
            )

        set_quota(1000.0)
        return finder, gdf, req, posts, tmp_path / "ledger.json", set_quota

    def test_refused_when_over_quota_and_nothing_is_sent(self, setup):
        finder, gdf, req, posts, ledger, set_quota = setup
        set_quota(5.0)
        with pytest.raises(RuntimeError, match="remaining quota"):
            finder.submit_order(
                req, ledger_path=ledger, images_gdf=gdf, require_confirmation=False
            )
        assert posts == []

    def test_refused_when_over_budget(self, setup):
        finder, gdf, req, posts, ledger, _ = setup
        with pytest.raises(RuntimeError, match="budget"):
            finder.submit_order(
                req,
                ledger_path=ledger,
                images_gdf=gdf,
                quota_budget_km2=10,
                require_confirmation=False,
            )
        assert posts == []

    def test_dry_run_sends_nothing(self, setup):
        finder, gdf, req, posts, ledger, _ = setup
        assert (
            finder.submit_order(req, ledger_path=ledger, images_gdf=gdf, dry_run=True)
            is None
        )
        assert posts == [] and not ledger.exists()

    def test_declined_confirmation_sends_nothing(self, setup, monkeypatch):
        finder, gdf, req, posts, ledger, _ = setup
        monkeypatch.setattr("builtins.input", lambda *_: "no")
        assert finder.submit_order(req, ledger_path=ledger, images_gdf=gdf) is None
        assert posts == []

    def test_success_writes_the_ledger_and_flags_double_orders(self, setup, capsys):
        finder, gdf, req, posts, ledger, _ = setup
        oid = finder.submit_order(
            req, ledger_path=ledger, images_gdf=gdf, require_confirmation=False
        )
        assert oid == "order-1" and len(posts) == 1
        entry = json.loads(ledger.read_text())[0]
        assert entry["item_ids"] == ["s0", "s1"] and entry["estimated_km2"] == 20.0

        finder.submit_order(
            req, ledger_path=ledger, images_gdf=gdf, require_confirmation=False
        )
        assert "possible double charge" in capsys.readouterr().out
        assert len(json.loads(ledger.read_text())) == 2

    def test_unreadable_quota_does_not_block_the_order(self, setup, monkeypatch):
        finder, gdf, req, posts, ledger, _ = setup

        def boom():
            raise RuntimeError("endpoint unavailable")

        monkeypatch.setattr(finder._client, "get_quota", boom)
        assert finder.submit_order(
            req, ledger_path=ledger, images_gdf=gdf, require_confirmation=False
        )


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
            "properties": {
                "acquired": acquired,
                "instrument": "PSB.SD",
                "cloud_cover": 0.1,
                "view_angle": 2.0,
                "satellite_azimuth": 100.0,
                **props,
            },
        }

    def test_filters_months_and_builds_columns(self, monkeypatch):
        finder = _finder(month_range=(6, 8))
        feats = [
            self._feature("summer", "2022-07-01T10:00:00Z"),
            self._feature("winter", "2022-01-01T10:00:00Z"),
        ]
        monkeypatch.setattr(
            finder._client.session, "post", lambda *a, **k: _Resp({"features": feats})
        )
        gdf = finder.search_metadata()
        assert list(gdf["image_id"]) == ["summer"]
        row = gdf.iloc[0]
        assert row["cloud_percent"] == pytest.approx(10.0)
        assert row["signed_view_angle"] == 2.0
        assert 0 < row["aoi_coverage_percent"] <= 100

    def test_follows_pagination(self, monkeypatch):
        finder = _finder()
        first = {
            "features": [self._feature("a", "2022-07-01T10:00:00Z")],
            "_links": {"_next": "next-url"},
        }
        second = {"features": [self._feature("b", "2022-07-02T10:00:00Z")]}
        monkeypatch.setattr(
            finder._client.session, "post", lambda *a, **k: _Resp(first)
        )
        monkeypatch.setattr(
            finder._client.session, "get", lambda url, **k: _Resp(second)
        )
        assert set(finder.search_metadata()["image_id"]) == {"a", "b"}

    def test_no_result_gives_empty_frame_with_columns(self, monkeypatch):
        finder = _finder()
        monkeypatch.setattr(
            finder._client.session, "post", lambda *a, **k: _Resp({"features": []})
        )
        gdf = finder.search_metadata()
        assert gdf.empty and "image_id" in gdf.columns


# ---------------------------------------------------------------------------
# Thumbnails, order polling/download, AOI-less estimate, contact sheet
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomulticorr.data_sources import planet_image_search_download as pmod
from geomulticorr.data_sources.planet_image_search_download import plot_thumbnail_grid


def test_feature_outside_the_aoi_is_dropped(monkeypatch):
    finder = _finder()
    far = {
        "id": "far",
        "geometry": mapping(box(20, 20, 21, 21)),
        "properties": {
            "acquired": "2022-07-01T10:00:00Z",
            "instrument": "PSB.SD",
            "cloud_cover": 0.1,
        },
    }
    monkeypatch.setattr(
        finder._client.session, "post", lambda *a, **k: _Resp({"features": [far]})
    )
    assert finder.search_metadata().empty


class _ThumbResp:
    def __init__(self, content=b"png", status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def _thumb_frame(ids=("a", "b")):
    return gpd.GeoDataFrame(
        {
            "image_id": list(ids),
            "acquisition_date": pd.to_datetime(
                ["2022-07-01", "2022-07-02"][: len(ids)]
            ),
        },
        geometry=[AOI] * len(ids),
        crs="EPSG:4326",
    )


class TestDownloadThumbnails:
    def test_downloads_names_and_skips_existing(self, tmp_path, monkeypatch):
        finder = _finder()
        calls = []

        def get(url, params=None, timeout=None):
            calls.append((url, params))
            return _ThumbResp(b"PNGDATA")

        monkeypatch.setattr(finder._client.session, "get", get)
        paths = finder.download_thumbnails(
            _thumb_frame(), output_dir=tmp_path / "t", width=256, show_progress=False
        )
        assert [p.name for p in paths] == ["a.png", "b.png"]
        assert all(p.read_bytes() == b"PNGDATA" for p in paths)
        assert calls[0][1] == {"width": 256} and "/items/a/thumb" in calls[0][0]

        again = finder.download_thumbnails(
            _thumb_frame(), output_dir=tmp_path / "t", show_progress=False
        )
        assert len(calls) == 2 and again == paths  # existing ones are not re-fetched

        finder.download_thumbnails(
            _thumb_frame(("a",)),
            output_dir=tmp_path / "t",
            overwrite=True,
            show_progress=False,
        )
        assert len(calls) == 3

    def test_progress_bar_and_string_dates(self, tmp_path, monkeypatch):
        finder = _finder()
        monkeypatch.setattr(finder._client.session, "get", lambda *a, **k: _ThumbResp())
        gdf = _thumb_frame(("a",))
        gdf["acquisition_date"] = ["2022-07-01T10:00:00"]  # plain strings, no strftime
        (p,) = finder.download_thumbnails(gdf, output_dir=tmp_path, show_progress=True)
        assert p.name == "a.png"  # id only, no duplicated date


class TestOrderWithoutScenesFrame:
    def test_estimate_falls_back_to_n_scenes_times_aoi_area(self, capsys):
        finder = _finder()
        req = finder.build_order_request(_scenes(3), order_name="o")
        assert (
            finder.submit_order(
                req, ledger_path="unused.json", dry_run=True, min_charge_km2=0.0
            )
            is None
        )
        out = capsys.readouterr().out
        assert f"{3 * finder.aoi_area_km2:.1f} km2" in out and "Dry run" in out


class _StreamResp:
    def __init__(self, chunks=(b"ab", b"cd")):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        return iter(self.chunks)


class TestWaitAndDownloadOrder:
    def _router(self, finder, states, results=None):
        seq = list(states)
        urls = []

        def get(url, stream=False, timeout=None, **k):
            urls.append(url)
            if url.startswith(pmod.PLANET_ORDERS_URL):
                state = seq.pop(0) if len(seq) > 1 else seq[0]
                return _Resp({"state": state, "_links": {"results": results or []}})
            return _StreamResp()

        monkeypatch_target = finder._client.session
        return get, urls, monkeypatch_target

    def test_polls_until_success_then_downloads(self, tmp_path, monkeypatch):
        finder = _finder()
        sleeps = []
        monkeypatch.setattr(pmod._time, "sleep", sleeps.append)
        get, urls, sess = self._router(
            finder,
            ["queued", "running", "success"],
            results=[
                {"name": "sub/dir/a.zip", "location": "https://dl/a"},
                {"name": "b.tif", "location": "https://dl/b"},
            ],
        )
        monkeypatch.setattr(sess, "get", get)
        out = finder.wait_and_download_order(
            "o1", output_dir=tmp_path / "o", poll_interval_s=7
        )
        assert sleeps == [7, 7]
        assert [p.name for p in out] == [
            "a.zip",
            "b.tif",
        ]  # server path components are dropped
        assert (tmp_path / "o" / "a.zip").read_bytes() == b"abcd"
        assert not list((tmp_path / "o").glob("*.part"))

    def test_partial_counts_as_success_and_existing_files_are_kept(
        self, tmp_path, monkeypatch
    ):
        finder = _finder()
        get, urls, sess = self._router(
            finder, ["partial"], results=[{"name": "a.zip", "location": "https://dl/a"}]
        )
        monkeypatch.setattr(sess, "get", get)
        (tmp_path / "a.zip").write_bytes(b"old")
        out = finder.wait_and_download_order("o1", output_dir=tmp_path)
        assert (
            out == [tmp_path / "a.zip"] and (tmp_path / "a.zip").read_bytes() == b"old"
        )
        assert "https://dl/a" not in urls
        finder.wait_and_download_order("o1", output_dir=tmp_path, overwrite=True)
        assert (tmp_path / "a.zip").read_bytes() == b"abcd"

    @pytest.mark.parametrize("state", ["failed", "cancelled"])
    def test_failed_orders_raise(self, tmp_path, monkeypatch, state):
        finder = _finder()
        get, _, sess = self._router(finder, [state])
        monkeypatch.setattr(sess, "get", get)
        with pytest.raises(RuntimeError, match=state):
            finder.wait_and_download_order("o1", output_dir=tmp_path)

    def test_timeout(self, tmp_path, monkeypatch):
        finder = _finder()
        get, _, sess = self._router(finder, ["running"])
        monkeypatch.setattr(sess, "get", get)
        monkeypatch.setattr(pmod._time, "sleep", lambda s: None)
        with pytest.raises(TimeoutError, match="still 'running'"):
            finder.wait_and_download_order("o1", output_dir=tmp_path, timeout_s=-1)


class TestPlotThumbnailGrid:
    def _thumbs(self, tmp_path, n):
        paths = []
        for i in range(n):
            p = tmp_path / f"t{i}.png"
            plt.imsave(p, np.random.default_rng(i).random((8, 8, 3)))
            paths.append(p)
        return paths

    def test_without_metadata(self, tmp_path):
        fig = plot_thumbnail_grid(self._thumbs(tmp_path, 3), ncols=2)
        try:
            assert len(fig.axes) == 4  # 2x2 grid, the spare cell is switched off
            assert [ax.get_title() for ax in fig.axes[:3]] == [""] * 3
        finally:
            plt.close(fig)

    def test_captions_with_and_without_optional_fields(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {
                "acquisition_date": pd.to_datetime(["2022-07-01", "2022-07-02"]),
                "instrument": ["PSB.SD", "PS2"],
                "clear_percent": [95.0, np.nan],
                "signed_view_angle": [2.5, np.nan],
            },
            geometry=[AOI] * 2,
            crs="EPSG:4326",
        )
        fig = plot_thumbnail_grid(self._thumbs(tmp_path, 2), gdf, ncols=2)
        try:
            t0, t1 = fig.axes[0].get_title(), fig.axes[1].get_title()
            assert "2022-07-01 PSB.SD" in t0 and "clear 95%" in t0 and "va +2.5" in t0
            assert t1 == "2022-07-02 PS2"
        finally:
            plt.close(fig)

    def test_string_dates_in_captions(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {"acquisition_date": ["2022-07-01T10:00:00"], "instrument": ["PS2"]},
            geometry=[AOI],
            crs="EPSG:4326",
        )
        fig = plot_thumbnail_grid(self._thumbs(tmp_path, 1), gdf, ncols=1)
        try:
            assert fig.axes[0].get_title() == "2022-07-01 PS2"
        finally:
            plt.close(fig)


class TestWorkdirSaving:
    """search_metadata() saves footprints.gpkg; thumbnails only when asked."""

    _feature = TestSearchMetadata._feature

    def _setup(self, monkeypatch, tmp_path, ids=("a", "b"), fail_on=None, **kw):
        finder = _finder(workdir=tmp_path / "rasuwa", **kw)
        feats = [
            self._feature(i, f"2022-07-0{k + 1}T10:00:00Z") for k, i in enumerate(ids)
        ]
        calls = []

        def get(url, params=None, timeout=None):
            calls.append(url)
            if fail_on and f"/items/{fail_on}/" in url:
                raise requests.HTTPError("503")
            return _ThumbResp(b"PNG")

        monkeypatch.setattr(
            finder._client.session, "post", lambda *a, **k: _Resp({"features": feats})
        )
        monkeypatch.setattr(finder._client.session, "get", get)
        return finder, calls

    def test_saves_only_the_footprints_by_default(self, monkeypatch, tmp_path):
        finder, calls = self._setup(monkeypatch, tmp_path)
        gdf = finder.search_metadata()
        wd = tmp_path / "rasuwa"
        assert finder.footprints_path == wd / "footprints.gpkg"
        saved = gpd.read_file(finder.footprints_path, layer="footprints")
        assert sorted(saved["image_id"]) == sorted(gdf["image_id"])
        assert saved.crs.to_epsg() == 4326
        assert calls == [] and not (wd / "thumbnails").exists()  # nothing fetched
        assert not list(wd.glob("*.part*"))

    def test_thumbnails_only_when_asked(self, monkeypatch, tmp_path):
        finder, calls = self._setup(monkeypatch, tmp_path)
        finder.search_metadata(download_thumbnails=True, show_progress=False)
        wd = tmp_path / "rasuwa"
        assert sorted(p.name for p in (wd / "thumbnails").iterdir()) == [
            "a.png",
            "b.png",
        ]
        finder.search_metadata(download_thumbnails=True, show_progress=False)
        assert len(calls) == 2  # existing PNGs were not fetched again

    def test_thumbnails_without_saving_the_table(self, monkeypatch, tmp_path):
        finder, calls = self._setup(monkeypatch, tmp_path)
        finder.search_metadata(
            save=False, download_thumbnails=True, show_progress=False
        )
        assert len(calls) == 2 and not finder.footprints_path.exists()

    def test_later_download_of_the_cleaned_table(self, monkeypatch, tmp_path):
        finder, calls = self._setup(monkeypatch, tmp_path)
        gdf = finder.search_metadata()
        kept = gdf[gdf["image_id"] == "a"]
        (p,) = finder.download_thumbnails(kept, show_progress=False)
        assert p == tmp_path / "rasuwa" / "thumbnails" / "a.png" and len(calls) == 1

    def test_save_false_and_no_workdir_write_nothing(self, monkeypatch, tmp_path):
        finder, calls = self._setup(monkeypatch, tmp_path)
        finder.search_metadata(save=False)
        assert not (tmp_path / "rasuwa").exists() and calls == []
        bare = _finder()
        monkeypatch.setattr(
            bare._client.session, "post", lambda *a, **k: _Resp({"features": []})
        )
        bare.search_metadata()  # no workdir → no save, no error
        assert bare.footprints_path is None and bare.thumbnail_dir is None

    def test_save_true_without_workdir_raises(self):
        with pytest.raises(ValueError, match="workdir"):
            _finder().search_metadata(save=True)
        with pytest.raises(ValueError, match="workdir"):
            _finder().search_metadata(download_thumbnails=True)
        with pytest.raises(ValueError, match="workdir"):
            _finder().download_thumbnails(_thumb_frame())

    def test_empty_search_writes_nothing(self, monkeypatch, tmp_path):
        finder, _ = self._setup(monkeypatch, tmp_path, ids=())
        assert finder.search_metadata().empty
        assert not (tmp_path / "rasuwa" / "footprints.gpkg").exists()

    def test_failed_thumbnail_keeps_the_search(self, monkeypatch, tmp_path, caplog):
        finder, _ = self._setup(monkeypatch, tmp_path, fail_on="a")
        with caplog.at_level("WARNING"):
            gdf = finder.search_metadata(download_thumbnails=True, show_progress=False)
        assert len(gdf) == 2 and finder.footprints_path.exists()
        assert "download_thumbnails" in caplog.text

    def test_finder_georeference_uses_workdir_defaults(self, monkeypatch, tmp_path):
        finder, _ = self._setup(monkeypatch, tmp_path)
        seen = {}

        def fake(thumbs, images_gdf, **kw):
            seen.update(thumbs=thumbs, images_gdf=images_gdf, **kw)
            return pd.DataFrame()

        monkeypatch.setattr(pmod, "georeference_thumbnails", fake)
        finder.georeference_thumbnails(resolution=50)
        assert seen["thumbs"] == tmp_path / "rasuwa" / "thumbnails"
        assert seen["images_gdf"] == tmp_path / "rasuwa" / "footprints.gpkg"
        assert seen["crop_to"].equals(finder.aoi_geom) and seen["resolution"] == 50
        finder.georeference_thumbnails(crop_to=None)
        assert seen["crop_to"] is None


class TestWideThumbnails:
    """Thumbnails > 512 px need visual-asset download rights, per scene."""

    def test_search_records_the_permission(self, monkeypatch):
        finder = _finder()
        f_ok = TestSearchMetadata()._feature("ok", "2022-07-01T10:00:00Z")
        f_ok["_permissions"] = ["assets.ortho_visual:download", "assets.udm2:download"]
        f_no = TestSearchMetadata()._feature("no", "2022-07-02T10:00:00Z")
        f_no["_permissions"] = ["assets.basic_udm2:download"]
        f_none = TestSearchMetadata()._feature("none", "2022-07-03T10:00:00Z")
        monkeypatch.setattr(
            finder._client.session,
            "post",
            lambda *a, **k: _Resp({"features": [f_ok, f_no, f_none]}),
        )
        gdf = finder.search_metadata().set_index("image_id")
        assert gdf["visual_download"].to_dict() == {
            "ok": True,
            "no": False,
            "none": False,
        }

    def test_empty_search_has_the_column(self, monkeypatch):
        finder = _finder()
        monkeypatch.setattr(
            finder._client.session, "post", lambda *a, **k: _Resp({"features": []})
        )
        assert "visual_download" in finder.search_metadata().columns

    def _get(self, refused=(), status=400):
        calls = []

        def get(url, params=None, timeout=None):
            iid = url.split("/items/")[1].split("/")[0]
            calls.append((iid, params["width"]))
            if iid in refused and params["width"] > 512:
                return _ThumbResp(status_code=status)
            return _ThumbResp(f"{iid}@{params['width']}".encode())

        return get, calls

    def test_fallback_to_512(self, tmp_path, monkeypatch, caplog):
        finder = _finder()
        get, calls = self._get(refused=("old",))
        monkeypatch.setattr(finder._client.session, "get", get)
        gdf = _thumb_frame(("ok", "no"))
        gdf["visual_download"] = [True, False]
        with caplog.at_level("WARNING"):
            finder.download_thumbnails(
                gdf, output_dir=tmp_path, width=1024, show_progress=False
            )
            # an older frame without the column: try 1024, refused → 512
            finder.download_thumbnails(
                _thumb_frame(("old",)),
                output_dir=tmp_path,
                width=1024,
                show_progress=False,
            )
        assert calls == [("ok", 1024), ("no", 512), ("old", 1024), ("old", 512)]
        assert (tmp_path / "ok.png").read_bytes() == b"ok@1024"
        assert (tmp_path / "no.png").read_bytes() == b"no@512"
        assert (tmp_path / "old.png").read_bytes() == b"old@512"
        assert finder.last_thumbnail_fallbacks == ["old"]
        assert caplog.text.count("instead of 1024") == 2

    def test_403_also_falls_back(self, tmp_path, monkeypatch):
        finder = _finder()
        get, _ = self._get(refused=("a",), status=403)
        monkeypatch.setattr(finder._client.session, "get", get)
        finder.download_thumbnails(
            _thumb_frame(("a",)), output_dir=tmp_path, width=2048, show_progress=False
        )
        assert finder.last_thumbnail_fallbacks == ["a"]

    def test_other_errors_still_raise(self, tmp_path, monkeypatch):
        finder = _finder()
        get, _ = self._get(refused=("a",), status=500)
        monkeypatch.setattr(finder._client.session, "get", get)
        with pytest.raises(requests.HTTPError, match="500"):
            finder.download_thumbnails(
                _thumb_frame(("a",)), output_dir=tmp_path, width=1024
            )

    def test_512_never_falls_back(self, tmp_path, monkeypatch):
        finder = _finder()
        get, calls = self._get()
        monkeypatch.setattr(finder._client.session, "get", get)
        gdf = _thumb_frame(("a",))
        gdf["visual_download"] = [False]
        finder.download_thumbnails(gdf, output_dir=tmp_path, show_progress=False)
        assert calls == [("a", 512)] and finder.last_thumbnail_fallbacks == []


class TestClientInfo:
    """client.info() / item_permissions(): payloads shaped like the live API, values made up."""

    SUBS = [
        {
            "id": 11,
            "state": "active",
            "active_from": "2026-01-01T00:00:00+00:00",
            "active_to": "2027-01-01T00:00:00+00:00",
            "organization": {"id": 7, "name": "lab"},
            "plan": {
                "id": 99,
                "name": "Education and Research Basic",
                "meta": {"item_types": ["PSScene"], "union_aoi": 511207893.4},
            },
            "plan_id": 99,
            "quota_sqkm": 3000,
            "quota_used": 1200.0,
            "quota_interval": "monthly",
            "quota_anchor_date": "2026-01-01T00:00:00+00:00",
            "quota_reset_at": None,
            "quota_enable_overage": False,
            "scene_tile_quota": {"current": 10, "limit": 100000, "reset": "2026-12-01"},
            "basemap_tile_quota": None,
        }
    ]
    PLAN = {
        "id": 99,
        "policies": [
            {
                "name": "PlanetScope",
                "constraints": [
                    {
                        "key": "type_permissions",
                        "value": {"map": {"PSScene": ["ortho_visual", "basic_udm2"]}},
                    },
                    {"key": "toi", "value": {"delay": 30, "ranges": [{}]}},
                    {"key": "flags", "value": {"geotiff_download": False}},
                ],
            },
            {
                "name": "Basemaps",
                "constraints": [
                    {"key": "type_permissions", "value": {"map": {"series": ["quad"]}}},
                    {"key": "toi", "value": {"delay": 0, "ranges": [{}]}},
                ],
            },
        ],
    }

    def _client(self, monkeypatch, fail=()):
        client = PlanetClient(api_key="secret-key")
        routes = {
            pmod.PLANET_QUOTA_URL: self.SUBS,
            f"{pmod.PLANET_PLANS_URL}/99": self.PLAN,
            pmod.PLANET_PRODUCTS_URL: {"results": []},
        }

        def get(url, timeout=None, **kw):
            if any(f in url for f in fail):
                return _ThumbResp(status_code=500)
            resp = _Resp(routes[url])
            resp.raise_for_status = lambda: None
            return resp

        monkeypatch.setattr(client.session, "get", get)
        return client

    def test_summary(self, monkeypatch, capsys):
        info = self._client(monkeypatch).info()
        (sub,) = info["subscriptions"]
        assert sub["plan_name"] == "Education and Research Basic"
        assert sub["global_aoi"] is True
        assert sub["quota_km2"]["remaining"] == pytest.approx(1800)
        assert sub["tile_quota"] == {
            "scene": {"current": 10, "limit": 100000, "reset": "2026-12-01"}
        }
        ps, bm = sub["policies"]
        assert (ps["delay_days"], ps["visual_download"], ps["geotiff_download"]) == (
            30,
            True,
            False,
        )
        assert bm["delay_days"] == 0 and bm["visual_download"] is False
        assert info["organization"]["name"] == "lab" and info["errors"] == {}
        out = capsys.readouterr().out
        assert "Education and Research Basic" in out and "30 days" in out
        assert "global" in out and "secret-key" not in out

    def test_quiet(self, monkeypatch, capsys):
        self._client(monkeypatch).info(verbose=False)
        assert capsys.readouterr().out == ""

    def test_a_failing_plan_does_not_lose_the_rest(self, monkeypatch, capsys):
        info = self._client(monkeypatch, fail=("/plans/",)).info()
        assert "plan:99" in info["errors"]
        assert info["subscriptions"][0]["quota_km2"]["total"] == 3000
        assert info["subscriptions"][0]["policies"] == []
        assert "! plan:99" in capsys.readouterr().out

    def test_failing_subscriptions_still_return(self, monkeypatch):
        info = self._client(monkeypatch, fail=("subscriptions", "products")).info(
            verbose=False
        )
        assert set(info["errors"]) == {"subscriptions", "products"}
        assert info["subscriptions"] == []

    @pytest.mark.parametrize(
        "perms, visual",
        [
            (["assets.ortho_visual:download", "assets.basic_udm2:download"], True),
            ([], False),
        ],
    )
    def test_item_permissions(self, monkeypatch, perms, visual):
        client = PlanetClient(api_key="k")
        seen = []

        def get(url, timeout=None, **kw):
            seen.append(url)
            return _Resp(
                {
                    "id": "x",
                    "_permissions": perms,
                    "properties": {
                        "acquired": "2026-11-15",
                        "publishing_stage": "finalized",
                    },
                }
            )

        monkeypatch.setattr(client.session, "get", get)
        rec = client.item_permissions("x")
        assert seen == [f"{pmod.PLANET_DATA_URL}/item-types/PSScene/items/x"]
        assert rec["visual_download"] is visual and rec["permissions"] == perms
        assert rec["publishing_stage"] == "finalized"

    def test_fallback_warning_points_to_info(self, tmp_path, monkeypatch, caplog):
        finder = _finder()
        monkeypatch.setattr(finder._client.session, "get", lambda *a, **k: _ThumbResp())
        gdf = _thumb_frame(("a",))
        gdf["visual_download"] = [False]
        with caplog.at_level("WARNING"):
            finder.download_thumbnails(
                gdf, output_dir=tmp_path, width=1024, show_progress=False
            )
        assert "client.info()" in caplog.text and "30 days" in caplog.text
