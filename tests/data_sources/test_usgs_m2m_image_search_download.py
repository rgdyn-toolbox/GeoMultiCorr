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
