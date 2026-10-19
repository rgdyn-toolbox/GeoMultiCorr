#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_global_dem_download.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""Global DEM download: tile naming, coverage, caching and HTTP error handling."""
import io
import urllib.error
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from geomulticorr.data_sources import global_dem_download as mod
from geomulticorr.data_sources.global_dem_download import (
    CopernicusProvider,
    DEMClient,
    DEMDownloadError,
    DEMFinder,
    SRTMProvider,
)


class TestSRTMProvider:
    @pytest.mark.parametrize(
        "lon, lat, expected",
        [(6, 45, "N45E006"), (-70, -33, "S33W070"), (0, 0, "N00E000"), (-1, -1, "S01W001")],
    )
    def test_tile_id(self, lon, lat, expected):
        assert SRTMProvider().tile_id(lon, lat) == expected

    def test_url_and_cache_path(self, tmp_path):
        p = SRTMProvider()
        assert p.tile_url("N45E006") == f"{p.BASE_URL}/N45E006.SRTMGL1.hgt.zip"
        assert p.cached_raster(tmp_path, "N45E006") == tmp_path / "srtm" / "N45E006.hgt"

    @pytest.mark.parametrize("lat, covered", [(59, True), (60, False), (-56, True), (-57, False)])
    def test_latitude_coverage_is_half_open(self, lat, covered):
        assert SRTMProvider().covers_latitude(lat) is covered

    def test_store_extracts_hgt_and_removes_payload(self, tmp_path):
        p = SRTMProvider()
        p.HGT_BYTES = 8
        payload = tmp_path / "dl.zip"
        with zipfile.ZipFile(payload, "w") as zf:
            zf.writestr("N45E006.hgt", b"\x00" * 8)
        dest = p.store(payload, tmp_path / "cache", "N45E006")
        assert dest.read_bytes() == b"\x00" * 8
        assert not payload.exists()
        assert not list(dest.parent.glob("*.part*"))

    def test_store_rejects_wrong_size_and_leaves_no_partial(self, tmp_path):
        p = SRTMProvider()
        p.HGT_BYTES = 8
        payload = tmp_path / "dl.zip"
        with zipfile.ZipFile(payload, "w") as zf:
            zf.writestr("N45E006.hgt", b"\x00" * 3)
        with pytest.raises(DEMDownloadError, match="expected 8"):
            p.store(payload, tmp_path / "cache", "N45E006")
        cache = tmp_path / "cache" / "srtm"
        assert not (cache / "N45E006.hgt").exists()
        assert not list(cache.glob("*.part*"))

    def test_store_rejects_archive_without_hgt(self, tmp_path):
        payload = tmp_path / "dl.zip"
        with zipfile.ZipFile(payload, "w") as zf:
            zf.writestr("readme.txt", b"x")
        with pytest.raises(DEMDownloadError, match="No .hgt"):
            SRTMProvider().store(payload, tmp_path / "cache", "N45E006")


class TestCopernicusProvider:
    def test_tile_id_and_url(self):
        p = CopernicusProvider()
        tid = p.tile_id(6, 45)
        assert tid == "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"
        assert p.tile_url(tid) == f"{p.BASE_URL}/{tid}/{tid}.tif"

    def test_southern_western_hemisphere(self):
        assert CopernicusProvider().tile_id(-70, -33).endswith("S33_00_W070_00_DEM")

    def test_covers_the_poles(self):
        p = CopernicusProvider()
        assert p.covers_latitude(75) and p.covers_latitude(-89)

    def test_store_moves_the_cog(self, tmp_path):
        src = tmp_path / "payload.bin"
        src.write_bytes(b"cog")
        dest = CopernicusProvider().store(src, tmp_path / "cache", "T")
        assert dest.read_bytes() == b"cog" and not src.exists()


class TestTileIndices:
    def test_interior_bbox(self):
        got = sorted(SRTMProvider().tile_indices(5.5, 45.2, 6.5, 45.8))
        assert got == [(5, 45), (6, 45)]

    def test_bbox_on_integer_edges_does_not_add_a_tile(self):
        assert list(SRTMProvider().tile_indices(6.0, 45.0, 7.0, 46.0)) == [(6, 45)]

    def test_degenerate_point_bbox_still_yields_its_tile(self):
        assert list(SRTMProvider().tile_indices(6.3, 45.3, 6.3, 45.3)) == [(6, 45)]

    def test_srtm_drops_tiles_beyond_its_latitude_cover(self):
        assert list(SRTMProvider().tile_indices(0, 59.5, 1, 61.5)) == [(0, 59)]
        assert len(list(CopernicusProvider().tile_indices(0, 59.5, 1, 61.5))) == 3

    def test_antimeridian_wraps(self):
        lons = {lon for lon, _ in CopernicusProvider().tile_indices(179.5, 10, 181.0, 11)}
        assert lons == {179, -180}


class TestDEMClient:
    def test_default_cache_is_created(self, tmp_path):
        client = DEMClient(tmp_path / "c")
        assert (tmp_path / "c").is_dir()
        assert set(client.providers) == {"srtm", "cop30"}

    def test_cache_hit_skips_the_network(self, tmp_path, monkeypatch):
        client = DEMClient(tmp_path)
        cached = client.providers["cop30"].cached_raster(tmp_path, "T")
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"x")
        monkeypatch.setattr(
            client, "_download_url", lambda *a, **k: pytest.fail("network used on cache hit")
        )
        assert client.download_tile("cop30", "T") == cached

    def test_404_returns_none(self, tmp_path, monkeypatch):
        client = DEMClient(tmp_path)
        monkeypatch.setattr(client, "_download_url", lambda *a, **k: None)
        assert client.download_tile("cop30", "T") is None

    def test_failed_download_leaves_no_temp_file(self, tmp_path, monkeypatch):
        client = DEMClient(tmp_path)
        seen = {}

        def boom(url, dest, timeout=60.0):
            seen["dest"] = dest
            raise DEMDownloadError("boom")

        monkeypatch.setattr(client, "_download_url", boom)
        with pytest.raises(DEMDownloadError):
            client.download_tile("cop30", "T")
        assert not Path(seen["dest"]).exists()

    def test_success_stores_through_the_provider(self, tmp_path, monkeypatch):
        client = DEMClient(tmp_path)

        def fake(url, dest, timeout=60.0):
            Path(dest).write_bytes(b"cog")
            return 3

        monkeypatch.setattr(client, "_download_url", fake)
        out = client.download_tile("cop30", "T")
        assert out.read_bytes() == b"cog"

    def test_unknown_provider_is_a_keyerror(self, tmp_path):
        with pytest.raises(KeyError):
            DEMClient(tmp_path).download_tile("nope", "T")


class TestDownloadUrl:
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_streams_to_dest_and_sends_user_agent(self, tmp_path, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["ua"] = request.get_header("User-agent")
            return self._Resp(b"abcdef")

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        dest = tmp_path / "out.bin"
        assert DEMClient(tmp_path)._download_url("http://x/y", dest) == 6
        assert dest.read_bytes() == b"abcdef"
        assert captured["ua"] == mod.USER_AGENT
        assert not list(tmp_path.glob("*.part*"))

    def _http_error(self, code):
        return urllib.error.HTTPError("http://x", code, "msg", {}, None)

    def test_http_404_is_none(self, tmp_path, monkeypatch):
        def fake(request, timeout):
            raise self._http_error(404)

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
        assert DEMClient(tmp_path)._download_url("http://x", tmp_path / "o") is None

    def test_other_http_errors_raise(self, tmp_path, monkeypatch):
        def fake(request, timeout):
            raise self._http_error(500)

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
        with pytest.raises(DEMDownloadError, match="HTTP 500"):
            DEMClient(tmp_path)._download_url("http://x", tmp_path / "o")

    def test_unreachable_host_raises(self, tmp_path, monkeypatch):
        def fake(request, timeout):
            raise urllib.error.URLError("dns")

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
        with pytest.raises(DEMDownloadError, match="Cannot reach"):
            DEMClient(tmp_path)._download_url("http://x", tmp_path / "o")


class TestDEMFinder:
    def test_bbox_coercion(self, tmp_path):
        client = DEMClient(tmp_path)
        bbox = (5.5, 45.2, 6.5, 45.8)
        assert DEMFinder(client, bbox).bbox == bbox
        assert DEMFinder(client, list(bbox)).bbox == bbox
        assert DEMFinder(client, box(*bbox)).bbox == bbox
        gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
        assert DEMFinder(client, gdf).bbox == pytest.approx(bbox)

    def test_uncoercible_aoi(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot coerce"):
            DEMFinder(DEMClient(tmp_path), "Chamonix")

    @pytest.mark.parametrize("provider", ["srtm", "cop30"])
    def test_search_metadata_geometry_matches_the_tile(self, tmp_path, provider):
        finder = DEMFinder(DEMClient(tmp_path), (5.5, 45.2, 6.5, 45.8), provider=provider)
        gdf = finder.search_metadata()
        assert len(gdf) == 2
        assert str(gdf.crs) == "EPSG:4326"
        got = sorted(g.bounds for g in gdf.geometry)
        assert got == [(5.0, 45.0, 6.0, 46.0), (6.0, 45.0, 7.0, 46.0)]
        assert set(gdf["provider"]) == {provider}

    @pytest.mark.parametrize("provider", ["srtm", "cop30"])
    def test_search_metadata_southern_western_tiles(self, tmp_path, provider):
        """Tile SW corners are (-70,-34) and (-70,-33): each covers one degree north."""
        finder = DEMFinder(
            DEMClient(tmp_path), (-69.7, -33.2, -69.3, -32.9), provider=provider
        )
        got = sorted(g.bounds for g in finder.search_metadata().geometry)
        assert got == [(-70.0, -34.0, -69.0, -33.0), (-70.0, -33.0, -69.0, -32.0)]

    def test_download_with_no_tiles_raises(self, tmp_path):
        finder = DEMFinder(DEMClient(tmp_path / "c"), (0, 70, 1, 71), provider="srtm")
        with pytest.raises(ValueError, match="No tiles"):
            finder.download_dem(output_dir=tmp_path / "out", show_progress=False)
