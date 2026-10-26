#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_usgs_m2m_image_search_download.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""USGS M2M: token login, rate-limit retry, error diagnosis and download matching."""
import io
from pathlib import Path
import json
from urllib.error import HTTPError

import pytest
from shapely.geometry import box

from geomulticorr.data_sources import usgs_m2m_image_search_download as mod
from geomulticorr.data_sources.usgs_m2m_image_search_download import (
    M2M_LOGIN_ENDPOINT,
    M2MClient,
    M2MImageFinder,
)


class TestLogin:
    def test_username_required(self):
        with pytest.raises(ValueError, match="USGS_M2M_USERNAME"):
            M2MClient(api_token="t").login()

    def test_token_required(self):
        with pytest.raises(ValueError, match="USGS_M2M_TOKEN"):
            M2MClient(username="u").login()

    def test_password_only_explains_the_retirement(self):
        with pytest.raises(RuntimeError, match="retired"):
            M2MClient(username="u", password="pw").login()

    def test_credentials_from_environment(self, monkeypatch):
        monkeypatch.setenv("USGS_M2M_USERNAME", "eu")
        monkeypatch.setenv("USGS_M2M_TOKEN", "et")
        c = M2MClient()
        assert (c.username, c.api_token) == ("eu", "et")

    def test_login_posts_username_and_token_unauthenticated(self, monkeypatch):
        calls = []

        def fake_post(self, url, payload, authenticated=True, **k):
            calls.append((url, payload, authenticated))
            return {"data": "SESSION"}

        monkeypatch.setattr(M2MClient, "_post", fake_post)
        c = M2MClient(username="u", api_token="t")
        assert c.login() == "SESSION" and c.session_token == "SESSION"
        assert calls == [(M2M_LOGIN_ENDPOINT, {"username": "u", "token": "t"}, False)]

    def test_login_without_a_returned_token_fails(self, monkeypatch):
        monkeypatch.setattr(M2MClient, "_post", lambda *a, **k: {"data": None})
        with pytest.raises(RuntimeError, match="login failed"):
            M2MClient(username="u", api_token="t").login()

    def test_logout_clears_the_session(self, monkeypatch):
        monkeypatch.setattr(M2MClient, "_post", lambda *a, **k: {})
        c = M2MClient(username="u", api_token="t")
        c.session_token = "S"
        c.logout()
        assert c.session_token is None


class _Resp(io.BytesIO):
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, body):
    return HTTPError("http://x/api/" + body.get("endpoint", "scene-search"), code, "m", {},
                     io.BytesIO(json.dumps(body).encode()))


class TestPost:
    def test_session_token_travels_in_the_header_not_the_body(self, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout):
            seen["headers"] = dict(req.header_items())
            seen["body"] = req.data
            return _Resp(b'{"data": 1}')

        monkeypatch.setattr(mod, "urlopen", fake_urlopen)
        c = M2MClient(username="u", api_token="t")
        c.session_token = "SESSION"
        assert c._post("http://x/scene-search", {"a": 1}) == {"data": 1}
        assert seen["headers"]["X-auth-token"] == "SESSION"
        assert b"SESSION" not in seen["body"]

    def test_unauthenticated_call_sends_no_token(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(mod, "urlopen", lambda req, timeout: seen.update(h=dict(req.header_items())) or _Resp(b"{}"))
        c = M2MClient(username="u", api_token="t")
        c.session_token = "SESSION"
        c._post("http://x/login-token", {}, authenticated=False)
        assert "X-auth-token" not in seen["h"]

    def test_error_code_in_a_200_body_raises(self, monkeypatch):
        monkeypatch.setattr(
            mod, "urlopen", lambda req, timeout: _Resp(b'{"errorCode": "AUTH", "errorMessage": "bad"}')
        )
        with pytest.raises(RuntimeError, match="AUTH"):
            M2MClient(username="u", api_token="t")._post("http://x/y", {})

    def test_403_on_download_endpoint_points_at_the_role(self, monkeypatch):
        def fake(req, timeout):
            raise _http_error(403, {"errorCode": "FORBIDDEN", "errorMessage": "no", "endpoint": "download-request"})

        monkeypatch.setattr(mod, "urlopen", fake)
        c = M2MClient(username="u", api_token="t")
        with pytest.raises(RuntimeError) as exc:
            c._post("http://x/api/download-request", {})
        assert "download role" in str(exc.value) and "FORBIDDEN" in str(exc.value)

    def test_403_on_search_points_at_the_token(self, monkeypatch):
        monkeypatch.setattr(mod, "urlopen", lambda req, timeout: (_ for _ in ()).throw(_http_error(401, {})))
        with pytest.raises(RuntimeError, match="regenerate the application token"):
            M2MClient(username="u", api_token="t")._post("http://x/api/scene-search", {})

    def test_rate_limit_is_retried_with_backoff(self, monkeypatch):
        attempts, sleeps = [], []

        def once(self, url, payload, authenticated, timeout):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("M2M API error RATE_LIMIT: busy")
            return {"data": "ok"}

        monkeypatch.setattr(M2MClient, "_post_once", once)
        monkeypatch.setattr(mod._time, "sleep", sleeps.append)
        assert M2MClient(username="u", api_token="t")._post("http://x/y", {}) == {"data": "ok"}
        assert len(attempts) == 3
        assert sleeps == [mod.RATE_LIMIT_BACKOFF_S, 2 * mod.RATE_LIMIT_BACKOFF_S]

    def test_rate_limit_gives_up_after_the_retry_budget(self, monkeypatch):
        n = []
        monkeypatch.setattr(
            M2MClient, "_post_once",
            lambda *a, **k: n.append(1) or (_ for _ in ()).throw(RuntimeError("RATE_LIMIT")),
        )
        monkeypatch.setattr(mod._time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError, match="RATE_LIMIT"):
            M2MClient(username="u", api_token="t")._post("http://x/y", {}, rate_limit_retries=2)
        assert len(n) == 3

    def test_other_errors_are_not_retried(self, monkeypatch):
        n = []
        monkeypatch.setattr(
            M2MClient, "_post_once",
            lambda *a, **k: n.append(1) or (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError, match="boom"):
            M2MClient(username="u", api_token="t")._post("http://x/y", {})
        assert len(n) == 1


def _sec(name, available=True, ident="shared"):
    return {"displayId": name, "entityId": "E_" + name.replace(".", "_"), "id": ident, "available": available}


def _options():
    scene = "LC09_L1TP_232083_20231221_20231223_02_T1"
    secs = [_sec(f"{scene}_B1.TIF"), _sec(f"{scene}_B10.TIF"), _sec(f"{scene}_B8.TIF"),
            _sec(f"{scene}_MTL.txt"), _sec(f"{scene}_ANG.txt"), _sec(f"{scene}_B5.TIF", available=False)]
    return [
        {"entityId": "ent", "available": True, "productName": "Band File", "secondaryDownloads": secs},
        {"entityId": "ent", "available": True, "productName": "Band File", "secondaryDownloads": secs},
        {"entityId": "ent", "available": True, "productName": "Landsat Collection 2 Level-1 Product Bundle"},
        {"entityId": "other", "available": True, "productName": "Product Bundle"},
    ], scene


class TestMatching:
    def test_scene_files_dedupe_and_skip_unavailable(self):
        options, scene = _options()
        files = M2MImageFinder._scene_files(options, "ent")
        assert f"{scene}_B8.TIF" in files and f"{scene}_B5.TIF" not in files
        assert len(files) == 5  # two identical parents are not double counted

    def test_scene_files_ignore_other_entities(self):
        options, _ = _options()
        assert M2MImageFinder._scene_files(options, "nobody") == {}

    def test_no_bands_picks_the_bundle(self):
        options, _ = _options()
        (opt,) = M2MImageFinder._match_products(options, "ent", None)
        assert "bundle" in opt["productName"].lower()

    def test_b1_does_not_match_b10(self):
        options, scene = _options()
        (hit,) = M2MImageFinder._match_products(options, "ent", ["B1"])
        assert hit["displayId"] == f"{scene}_B1.TIF"

    def test_band_with_or_without_extension_and_case(self):
        options, scene = _options()
        for band in ("B8", "b8", "B8.TIF"):
            (hit,) = M2MImageFinder._match_products(options, "ent", [band])
            assert hit["displayId"] == f"{scene}_B8.TIF"

    def test_missing_band_is_skipped_with_a_warning(self, caplog):
        options, _ = _options()
        got = M2MImageFinder._match_products(options, "ent", ["B8", "B5"])  # B5 unavailable
        assert len(got) == 1
        assert any("B5" in r.message for r in caplog.records)

    def test_unknown_entity_returns_nothing(self):
        options, _ = _options()
        assert M2MImageFinder._match_products(options, "ghost", None) == []

    def test_metadata_sidecars(self):
        options, scene = _options()
        names = {m["displayId"] for m in M2MImageFinder._match_metadata(options, "ent")}
        assert names == {f"{scene}_MTL.txt", f"{scene}_ANG.txt"}


class TestUrlBasename:
    @pytest.mark.parametrize(
        "url, expected",
        [("https://x/y/LC08_B8.TIF?token=1", "LC08_B8.TIF"), ("https://x/download", None),
         ("https://x/", None), ("https://x/a/..%2F..%2Fx.tif", "x.tif")],
    )
    def test_cases(self, url, expected):
        got = mod._url_basename(url)
        assert got == expected or (expected is not None and got is not None and "/" not in got)


class TestDownloadFile:
    def _patch(self, monkeypatch, body, cd=None):
        resp = _Resp(body)
        resp.headers = {"Content-Disposition": cd} if cd else {}
        monkeypatch.setattr(mod, "urlopen", lambda url, timeout: resp)

    def test_uses_fallback_name_by_default(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, b"data", cd='attachment; filename="server.tif"')
        out = M2MImageFinder._download_file("https://x/dl", tmp_path, "mine.tif")
        assert out.name == "mine.tif" and out.read_bytes() == b"data"
        assert not list(tmp_path.glob("*.part"))

    def test_keeps_the_server_name_when_asked(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, b"d", cd='attachment; filename="server.tif"')
        out = M2MImageFinder._download_file("https://x/dl", tmp_path, "mine.tif", keep_original_names=True)
        assert out.name == "server.tif"

    def test_server_name_cannot_escape_the_directory(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, b"d", cd='attachment; filename="../../evil.sh"')
        out = M2MImageFinder._download_file("https://x/dl", tmp_path / "out", "mine.tif", keep_original_names=True)
        assert out.parent == tmp_path / "out" and out.name == "evil.sh"

    def test_url_basename_then_fallback(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, b"d")
        out = M2MImageFinder._download_file("https://x/dir/f.tif?t=1", tmp_path, "mine.tif", keep_original_names=True)
        assert out.name == "f.tif"
        out = M2MImageFinder._download_file("https://x/dl", tmp_path, "mine.tif", keep_original_names=True)
        assert out.name == "mine.tif"


def test_finder_construction_takes_a_shapely_aoi():
    f = M2MImageFinder(M2MClient(username="u", api_token="t"), aoi=box(6, 45, 6.1, 45.1), cloud_max=10)
    assert f.cloud_max == 10 and f.aoi_area_km2 > 0


def test_collection_tables_are_consistent():
    assert set(mod.LANDSAT_COLLECTIONS) >= {"L8_L1", "L9_L1", "L8_L2", "L9_L2"}
    for spec in mod.LANDSAT_COLLECTIONS.values():
        assert {"api_id", "name", "scale_m"} <= set(spec)
    assert mod.BAND_NAMES["B8"].startswith("Panchromatic")


# ---------------------------------------------------------------------------
# Endpoints, search, download orchestration, queue recovery
# ---------------------------------------------------------------------------
import logging

import geopandas as gpd
import pandas as pd


@pytest.fixture
def routed(monkeypatch):
    """A logged-in client whose ``_post`` is a recorder routed by endpoint URL."""
    calls, responses = [], {}

    def fake_post(self, url, payload, authenticated=True, **k):
        calls.append((url, payload, authenticated))
        r = responses.get(url, {})
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(M2MClient, "_post", fake_post)
    client = M2MClient(username="u", api_token="t")
    client.session_token = "S"
    return client, calls, responses


class TestClientEndpoints:
    def test_logout_without_session_posts_nothing(self, routed):
        client, calls, _ = routed
        client.session_token = None
        client.logout()
        assert calls == []

    def test_logout_failure_only_warns(self, routed, caplog):
        client, _, responses = routed
        responses[mod.M2M_LOGOUT_ENDPOINT] = RuntimeError("down")
        caplog.set_level(logging.WARNING)
        client.logout()
        assert client.session_token == "S"
        assert any("logout failed" in r.message for r in caplog.records)

    def test_permissions_logs_in_and_returns_strings(self, routed, monkeypatch):
        client, calls, responses = routed
        client.session_token = None
        monkeypatch.setattr(client, "login", lambda: setattr(client, "session_token", "S"))
        responses[mod.M2M_PERMISSIONS_ENDPOINT] = {"data": ["user", "download"]}
        assert client.permissions() == ["user", "download"]
        assert client.session_token == "S"

    def test_permissions_empty_data(self, routed):
        client, _, responses = routed
        responses[mod.M2M_PERMISSIONS_ENDPOINT] = {"data": None}
        assert client.permissions() == []

    def test_search_scenes_builds_mbr_payload(self, routed):
        client, calls, responses = routed
        responses[mod.M2M_SEARCH_ENDPOINT] = {"data": {"results": [{"entityId": "e"}]}}
        out = client.search_scenes("landsat_ot_c2_l1", box(6, 45, 7, 46), ("2020-01-01", "2020-12-28"), cloud_max=15)
        assert out == [{"entityId": "e"}]
        _, payload, _ = calls[-1]
        flt = payload["sceneFilter"]
        assert flt["spatialFilter"]["lowerLeft"] == {"latitude": 45.0, "longitude": 6.0}
        assert flt["spatialFilter"]["upperRight"] == {"latitude": 46.0, "longitude": 7.0}
        assert flt["cloudCoverFilter"]["max"] == 15
        assert flt["acquisitionFilter"] == {"start": "2020-01-01", "end": "2020-12-28"}

    def test_search_scenes_reraises(self, routed):
        client, _, responses = routed
        responses[mod.M2M_SEARCH_ENDPOINT] = RuntimeError("nope")
        with pytest.raises(RuntimeError, match="nope"):
            client.search_scenes("d", box(0, 0, 1, 1), ("a", "b"))

    def test_download_options_request_retrieve(self, routed):
        client, calls, responses = routed
        responses[mod.M2M_DOWNLOAD_OPTIONS_ENDPOINT] = {"data": [{"id": 1}]}
        responses[mod.M2M_DOWNLOAD_REQUEST_ENDPOINT] = {"data": {"availableDownloads": []}}
        responses[mod.M2M_DOWNLOAD_RETRIEVE_ENDPOINT] = {"data": {"available": [{"url": "u"}]}}
        assert client.download_options("ds", ["a", "b"]) == [{"id": 1}]
        assert calls[-1][1] == {"datasetName": "ds", "entityIds": ["a", "b"]}
        assert client.download_request([{"entityId": "a", "productId": "p"}], label="L") == {"availableDownloads": []}
        assert calls[-1][1]["label"] == "L"
        assert client.download_retrieve(label="L") == {"available": [{"url": "u"}]}
        assert calls[-1][1] == {"label": "L"}

    def test_download_search_with_and_without_label(self, routed):
        client, calls, responses = routed
        responses[mod.M2M_DOWNLOAD_SEARCH_ENDPOINT] = {"data": None}
        assert client.download_search() == []
        assert calls[-1][1] == {}
        responses[mod.M2M_DOWNLOAD_SEARCH_ENDPOINT] = {"data": [{"downloadId": 1}]}
        assert client.download_search(label="L") == [{"downloadId": 1}]
        assert calls[-1][1] == {"label": "L"}

    def test_rate_limit_summary(self, routed):
        client, calls, responses = routed
        responses[mod.M2M_RATE_LIMIT_ENDPOINT] = {"data": {"quota": 5}}
        assert client.rate_limit_summary() == {"quota": 5}
        responses[mod.M2M_RATE_LIMIT_ENDPOINT] = {"data": None}
        assert client.rate_limit_summary() == {}

    def test_endpoints_log_in_when_there_is_no_session(self, routed, monkeypatch):
        client, _, responses = routed
        logins = []
        monkeypatch.setattr(client, "login", lambda: logins.append(1))
        client.session_token = None
        responses[mod.M2M_DOWNLOAD_OPTIONS_ENDPOINT] = {"data": []}
        responses[mod.M2M_DOWNLOAD_REQUEST_ENDPOINT] = {"data": {}}
        responses[mod.M2M_DOWNLOAD_RETRIEVE_ENDPOINT] = {"data": {}}
        responses[mod.M2M_DOWNLOAD_SEARCH_ENDPOINT] = {"data": []}
        responses[mod.M2M_RATE_LIMIT_ENDPOINT] = {"data": {}}
        responses[mod.M2M_SEARCH_ENDPOINT] = {"data": {"results": []}}
        client.download_options("d", [])
        client.download_request([])
        client.download_retrieve()
        client.download_search()
        client.rate_limit_summary()
        client.search_scenes("d", box(0, 0, 1, 1), ("a", "b"))
        assert len(logins) == 6


def test_non_json_error_body_still_reports_the_status(monkeypatch):
    def fake(req, timeout):
        raise HTTPError("http://x/api/scene-search", 500, "m", {}, io.BytesIO(b"<html>oops</html>"))

    monkeypatch.setattr(mod, "urlopen", fake)
    with pytest.raises(RuntimeError, match="HTTP 500 on scene-search"):
        M2MClient(username="u", api_token="t")._post("http://x/api/scene-search", {})


SCENE = "LC09_L1TP_232083_20231221_20231223_02_T1"


class _Client(M2MClient):
    """Logged-in client with scripted download endpoints (no network)."""

    def __init__(self):
        super().__init__(username="u", api_token="t")
        self.session_token = "S"
        self.options = []
        self.search_results = {}
        self.requests = []
        self.request_result = {}
        self.retrieve_results = []
        self.queue = []

    def search_scenes(self, collection_id, geometry, date_range, cloud_max=100, max_results=5000):
        r = self.search_results.get(collection_id, [])
        if isinstance(r, Exception):
            raise r
        return r

    def download_options(self, collection_id, entity_ids):
        if isinstance(self.options, Exception):
            raise self.options
        return self.options

    def download_request(self, downloads, label="geomulticorr"):
        self.requests.append((list(downloads), label))
        r = self.request_result
        if isinstance(r, Exception):
            raise r
        return r

    def download_retrieve(self, label="geomulticorr"):
        if not self.retrieve_results:
            return {}
        r = self.retrieve_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def download_search(self, label=None):
        return self.queue


def _finder(client=None, **kw):
    return M2MImageFinder(client or _Client(), aoi=box(6, 45, 6.1, 45.1), **kw)


def _images(n=1, key="L9_L1"):
    return gpd.GeoDataFrame(
        {
            "image_id": ["ent"] * n if n == 1 else [f"ent{i}" for i in range(n)],
            "collection_key": key,
            "acquisition_date": pd.to_datetime(["2023-12-21"] * n),
            "display_id": SCENE,
        },
        geometry=[box(6, 45, 6.1, 45.1)] * n,
        crs="EPSG:4326",
    )


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(mod._time, "sleep", lambda s: None)


class TestSearchMetadata:
    def test_geodataframe_aoi_is_accepted(self):
        aoi = gpd.GeoDataFrame(geometry=[box(6, 45, 6.1, 45.1)], crs="EPSG:4326")
        f = M2MImageFinder(_Client(), aoi=aoi)
        assert f.aoi.bounds == (6, 45, 6.1, 45.1)

    def test_rows_dates_and_cloud_fallbacks(self):
        c = _Client()
        c.search_results["landsat_ot_c2_l1"] = [
            {"entityId": "a", "displayId": "da", "acquisitionDate": "2023-01-01", "cloudCover": "5"},
            {"entityId": "b", "displayId": "db", "temporalCoverage": {"startDate": "2023-06-01"}, "cloudCover": "-1"},
            {"entityId": "c", "displayId": "dc", "temporalCoverage": {"endDate": "2023-03-01"}, "cloudCover": None},
        ]
        gdf = _finder(c).search_metadata(["L8_L1"])
        assert list(gdf["image_id"]) == ["b", "c", "a"]  # newest first
        assert gdf.crs.to_epsg() == 4326
        assert gdf.set_index("image_id").loc["a", "cloud_percent"] == 5.0
        assert gdf["cloud_percent"].isna().sum() == 2

    def test_unknown_key_and_search_error_are_skipped(self, caplog):
        c = _Client()
        c.search_results["landsat_ot_c2_l1"] = RuntimeError("boom")
        caplog.set_level(logging.WARNING)
        gdf = _finder(c).search_metadata(["NOPE", "L8_L1"])
        assert gdf.empty and gdf.crs.to_epsg() == 4326
        assert "cloud_percent" in gdf.columns
        assert any("Unknown collection key" in r.message for r in caplog.records)


class TestDownloadImages:
    def _ready_client(self):
        c = _Client()
        options, _ = _options()
        for i, o in enumerate(options):
            o["id"] = f"opt{i}"
        c.options = options
        return c

    def _patch_download(self, monkeypatch, record=None):
        def fake(url, out_dir, fallback_name, keep_original_names=False, **k):
            p = Path(out_dir) / fallback_name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
            if record is not None:
                record.append((url, p))
            return p

        monkeypatch.setattr(M2MImageFinder, "_download_file", staticmethod(fake))

    def test_invalid_organize_by_raises(self, tmp_path):
        with pytest.raises(ValueError, match="organize_by"):
            _finder().download_images(_images(), output_dir=tmp_path, organize_by="bogus")

    def test_logs_in_when_there_is_no_session(self, tmp_path, monkeypatch):
        c = self._ready_client()
        c.session_token = None
        logins = []
        monkeypatch.setattr(c, "login", lambda: logins.append(1))
        c.options = []
        _finder(c).download_images(_images(), output_dir=tmp_path)
        assert logins == [1]

    def test_happy_path_names_and_files_per_collection(self, tmp_path, monkeypatch, fast):
        c = self._ready_client()
        c.request_result = {"availableDownloads": [{"entityId": "ent", "url": "https://x/f.tar.gz?t=1"}]}
        self._patch_download(monkeypatch)
        out = _finder(c).download_images(_images(), output_dir=tmp_path, show_progress=False)
        assert len(out) == 1
        assert out[0].parent == tmp_path / "L9_L1"
        assert out[0].name == f"{SCENE}_2023-12-21_Landsat_Collection_2_Level_1_Product_Bundle.gz"
        assert c.requests[0][1] == "gmc_L9_L1"

    def test_bands_pull_metadata_and_sensor_pathrow_layout(self, tmp_path, monkeypatch, fast):
        c = self._ready_client()
        options, scene = _options()
        # per-file entries carry their own entityId, as M2M's secondaryDownloads do
        for i, o in enumerate(options):
            o["id"] = f"opt{i}"
        for sec in options[0]["secondaryDownloads"]:
            sec["id"] = "pid_" + sec["displayId"]
        c.options = options
        c.request_result = {
            "availableDownloads": [
                {"entityId": "E_" + f"{scene}_B8.TIF".replace(".", "_"), "url": "https://x/a.TIF"},
                {"entityId": "E_" + f"{scene}_MTL.txt".replace(".", "_"), "url": "https://x/m.txt"},
            ]
        }
        self._patch_download(monkeypatch)
        out = _finder(c).download_images(
            _images(), output_dir=tmp_path, bands=["B8"], keep_original_names=True,
            organize_by="sensor_pathrow", show_progress=False,
        )
        sent = {d["entityId"] for d in c.requests[0][0]}
        assert len(sent) == 3  # B8 + MTL + ANG, each with its own entity id
        assert {p.name for p in out} == {f"{scene}_B8.TIF", f"{scene}_MTL.txt"}
        assert all(p.parent == tmp_path / "LANDSAT9" / "232_083" for p in out)

    def test_download_options_failure_is_skipped(self, tmp_path, caplog):
        c = _Client()
        c.options = RuntimeError("no options")
        caplog.set_level(logging.ERROR)
        assert _finder(c).download_images(_images(), output_dir=tmp_path) == []
        assert any("download_options failed" in r.message for r in caplog.records)

    def test_no_match_warns_and_requests_nothing(self, tmp_path, caplog):
        c = _Client()
        caplog.set_level(logging.WARNING)
        assert _finder(c).download_images(_images(), output_dir=tmp_path) == []
        assert c.requests == []
        assert any("No downloadable products" in r.message for r in caplog.records)

    def test_chunking_by_file_count(self, tmp_path, monkeypatch, fast):
        c = _Client()
        c.options = [
            {"entityId": f"ent{i}", "available": True, "productName": "Product Bundle", "id": f"p{i}"}
            for i in range(3)
        ]
        c.request_result = {"availableDownloads": []}
        _finder(c).download_images(_images(3), output_dir=tmp_path, max_files_per_request=1)
        assert [len(r[0]) for r in c.requests] == [1, 1, 1]

    def test_preparing_downloads_are_polled_until_ready(self, tmp_path, monkeypatch, fast):
        c = self._ready_client()
        c.request_result = {"preparingDownloads": [{"downloadId": 7}]}
        c.retrieve_results = [{"available": [{"downloadId": 7, "entityId": "ent", "url": "https://x/f.tar"}]}]
        self._patch_download(monkeypatch)
        out = _finder(c).download_images(_images(), output_dir=tmp_path, show_progress=False)
        assert len(out) == 1

    def test_request_failure_recovers_through_the_label(self, tmp_path, monkeypatch, fast, caplog):
        c = self._ready_client()
        c.request_result = RuntimeError("timeout")
        c.retrieve_results = [
            RuntimeError("flaky"),
            {"available": [{"downloadId": 1, "entityId": "ent", "url": "https://x/f.tar"}]},
        ]
        self._patch_download(monkeypatch)
        caplog.set_level(logging.WARNING)
        out = _finder(c).download_images(_images(), output_dir=tmp_path, show_progress=False)
        assert len(out) == 1
        assert any("recovering via" in r.message for r in caplog.records)

    def test_unresolved_batch_names_resume_label(self, tmp_path, fast, caplog):
        c = self._ready_client()
        c.request_result = {"preparingDownloads": [{"downloadId": 7}]}
        caplog.set_level(logging.WARNING)
        _finder(c).download_images(_images(), output_dir=tmp_path, max_poll_attempts=2)
        assert any("resume_label" in r.message and "gmc_L9_L1" in r.message for r in caplog.records)


class TestWriteDownloads:
    def _write(self, tmp_path, ready, pre_meta=None, monkeypatch=None, **kw):
        names = []

        def fake(url, out_dir, fallback_name, keep_original_names=False, **k):
            names.append((Path(out_dir), fallback_name))
            return Path(out_dir) / fallback_name

        monkeypatch.setattr(M2MImageFinder, "_download_file", staticmethod(fake))
        f = _finder()
        res = f._write_downloads(
            ready, pre_meta or {}, out_root=tmp_path, default_dir=tmp_path / "d",
            keep_original_names=kw.get("keep", False), organize_by=kw.get("organize_by"),
            show_progress=False, desc="x",
        )
        return res, names

    def test_entry_without_url_is_skipped(self, tmp_path, monkeypatch):
        res, names = self._write(tmp_path, [{"entityId": "e"}], monkeypatch=monkeypatch)
        assert res == [] and names == []

    def test_unmatched_entity_uses_download_id_and_default_extension(self, tmp_path, monkeypatch):
        _, names = self._write(tmp_path, [{"downloadId": 9, "url": "https://x/blob"}], monkeypatch=monkeypatch)
        assert names == [(tmp_path / "d", "download_9.tar.gz")]

    def test_keep_original_names_uses_the_catalog_name(self, tmp_path, monkeypatch):
        meta = {"e": ("disp", "2020-01-01", "Band File", "orig_B8.TIF")}
        _, names = self._write(tmp_path, [{"entityId": "e", "url": "https://x/f.TIF"}], meta, monkeypatch, keep=True)
        assert names[0][1] == "orig_B8.TIF"

    def test_download_exception_is_logged_not_raised(self, tmp_path, monkeypatch, caplog):
        def boom(*a, **k):
            raise OSError("disk")

        monkeypatch.setattr(M2MImageFinder, "_download_file", staticmethod(boom))
        caplog.set_level(logging.ERROR)
        res = _finder()._write_downloads(
            [{"url": "https://x/f.tif"}], {}, out_root=tmp_path, default_dir=tmp_path,
            keep_original_names=False, organize_by=None, show_progress=False, desc="x",
        )
        assert res == [] and any("Failed to download" in r.message for r in caplog.records)

    def test_tqdm_progress_branch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "_HAS_TQDM", True)
        monkeypatch.setattr(mod, "_tqdm", lambda it, **k: it, raising=False)
        res, _ = self._write(tmp_path, [{"downloadId": 1, "url": "https://x/f.tif"}], monkeypatch=monkeypatch)
        assert len(res) == 1


class TestResumeLabel:
    def _setup(self, monkeypatch):
        def fake(url, out_dir, fallback_name, keep_original_names=False, **k):
            p = Path(out_dir) / fallback_name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
            return p

        monkeypatch.setattr(M2MImageFinder, "_download_file", staticmethod(fake))

    def _rec(self, n, product=True):
        r = {"entityId": f"e{n}", "displayId": f"{SCENE}_B{n}.TIF", "downloadId": n}
        if product:
            r["productId"] = f"p{n}"
        return r

    def test_bad_organize_by_raises(self, tmp_path):
        with pytest.raises(ValueError, match="organize_by"):
            _finder().resume_label("L", output_dir=tmp_path, organize_by="bogus")

    def test_empty_queue_returns_empty(self, tmp_path):
        assert _finder().resume_label("L", output_dir=tmp_path) == []

    def test_retrieves_ready_records_into_pathrow_layout(self, tmp_path, monkeypatch, fast):
        self._setup(monkeypatch)
        c = _Client()
        c.queue = [self._rec(1)]
        c.retrieve_results = [{"available": [{"downloadId": 1, "entityId": "e1", "url": "https://x/a.TIF"}]}]
        out = _finder(c).resume_label("L", output_dir=tmp_path, show_progress=False)
        assert out == [tmp_path / "LANDSAT9" / "232_083" / f"{SCENE}_B1.TIF"]
        assert c.requests == []

    def test_re_requests_records_that_are_no_longer_retrievable(self, tmp_path, monkeypatch, fast):
        self._setup(monkeypatch)
        c = _Client()
        c.queue = [self._rec(1), self._rec(2), self._rec(3, product=False)]
        c.retrieve_results = [RuntimeError("flaky")]
        c.request_result = {
            "availableDownloads": [{"downloadId": 1, "entityId": "e1", "url": "https://x/a.TIF"}],
            "preparingDownloads": [{"downloadId": 2}],
        }
        c.retrieve_results += [{}, {}, {"available": [{"downloadId": 2, "entityId": "e2", "url": "https://x/b.TIF"}]}]
        out = _finder(c).resume_label("L", output_dir=tmp_path, show_progress=False, max_files_per_request=1)
        assert {r[0][0]["entityId"] for r in c.requests} == {"e1", "e2"}  # e3 has no productId
        assert len(out) >= 1

    def test_re_request_failure_is_warned_and_skipped(self, tmp_path, fast, caplog):
        c = _Client()
        c.queue = [self._rec(1)]
        c.request_result = RuntimeError("quota")
        caplog.set_level(logging.WARNING)
        assert _finder(c).resume_label("L", output_dir=tmp_path) == []
        assert any("re-request chunk" in r.message for r in caplog.records)
        assert any("no fetchable URLs" in r.message for r in caplog.records)

    def test_no_re_request_only_warns(self, tmp_path, fast, caplog):
        c = _Client()
        c.queue = [self._rec(1)]
        caplog.set_level(logging.WARNING)
        assert _finder(c).resume_label("L", output_dir=tmp_path, re_request=False) == []
        assert c.requests == []
        assert any("re_request=True" in r.message for r in caplog.records)


def test_no_bundle_falls_back_to_the_first_available_option():
    opts = [{"entityId": "ent", "available": True, "productName": "Something Else", "id": 1},
            {"entityId": "ent", "available": True, "productName": "Other", "id": 2}]
    assert M2MImageFinder._match_products(opts, "ent", None) == opts[:1]


def test_part_file_is_removed_when_the_transfer_fails(tmp_path, monkeypatch):
    class _Boom(_Resp):
        def read(self, n=-1):
            raise OSError("reset")

    resp = _Boom(b"data")
    resp.headers = {}
    monkeypatch.setattr(mod, "urlopen", lambda url, timeout: resp)
    with pytest.raises(OSError):
        M2MImageFinder._download_file("https://x/dl", tmp_path, "mine.tif")
    assert list(tmp_path.iterdir()) == []
