#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# global_dem_download.py
# creation date: 2026-08-11.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------- #
"""Global DEM download: SRTM GL1, Copernicus GLO-30, etc.

Supports multiple public DEM sources via a pluggable provider architecture.
Tiles are cached locally; mosaics are built per AOI via GDAL.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import box

from geomulticorr._typing import NDArrayNum

__all__ = ["DEMClient", "DEMFinder"]

# ============================================================================
# Constants
# ============================================================================

USER_AGENT = "GeoMultiCorr (+https://github.com/rgdyn-toolbox/GeoMultiCorr)"
CHUNK_SIZE = 1 << 20  # 1 MB

# Default cache location: ~/.cache/geomulticorr/dems
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "geomulticorr" / "dems"


# ============================================================================
# DEM Provider Base Class & Implementations
# ============================================================================


class DEMDownloadError(Exception):
    """Raised when a DEM tile cannot be downloaded."""


class DEMProvider(ABC):
    """Abstract base for a public global DEM source.

    A provider knows how to name, locate, download, store, and validate
    tiles from one DEM mission (SRTM, Copernicus, etc.). New providers are
    one subclass with five abstract methods.
    """

    #: short id: cache sub-directory name and user-facing identifier
    name: str = "base"
    #: human-readable label for logs and output filenames
    label: str = ""
    #: vertical datum of delivered heights (e.g. "EGM96" / "EGM2008")
    vertical_datum: str = ""
    #: inclusive latitude coverage [min, max)
    lat_range: tuple[float, float] = (-90.0, 90.0)
    #: native ground resolution in metres
    native_res_m: float = 30.0

    @abstractmethod
    def tile_id(self, lon: int, lat: int) -> str:
        """Unique tile identifier for SW corner (lon, lat)."""

    @abstractmethod
    def tile_url(self, tile_id: str) -> str:
        """Download URL for this tile."""

    @abstractmethod
    def cached_raster(self, cache_dir: Path, tile_id: str) -> Path:
        """Path where this tile's cached raster lives."""

    @abstractmethod
    def store(self, payload: Path, cache_dir: Path, tile_id: str) -> Path:
        """Store a downloaded payload as a cached raster. Must be atomic."""

    def covers_latitude(self, lat: int) -> bool:
        """True when this provider has data at this latitude."""
        low, high = self.lat_range
        return low <= lat < high

    def tile_indices(self, west: float, south: float, east: float, north: float):
        """Yield (lon, lat) pairs of 1x1-degree tiles intersecting the bbox."""
        lat0 = math.floor(south)
        lat1 = math.ceil(north) - 1
        if lat1 < lat0:
            lat1 = lat0

        def lon_run(lo, hi):
            lon0 = math.floor(lo)
            lon1 = math.ceil(hi) - 1
            return range(lon0, max(lon0, lon1) + 1)

        if west < -180.0 or east > 180.0:
            lons = list(lon_run(((west + 180.0) % 360.0) - 180.0, 180.0))
            lons += list(lon_run(-180.0, ((east + 180.0) % 360.0) - 180.0))
        else:
            lons = list(lon_run(west, east))

        for lon in sorted(set(lons)):
            for lat in range(lat0, lat1 + 1):
                if self.covers_latitude(lat):
                    yield lon, lat

    def tile_ids(self, west: float, south: float, east: float, north: float):
        """Tile ids covering the bounding box, latitude-filtered."""
        return [self.tile_id(lon, lat) for lon, lat in self.tile_indices(west, south, east, north)]


class SRTMProvider(DEMProvider):
    """SRTM GL1 (1 arc-second, ~30 m) via ESA STEP mirror.

    Coverage: 56S to 60N, land only. Delivered as zipped .hgt files
    with EGM96 vertical reference.
    """

    name = "srtm"
    label = "SRTM GL1 (1 arc-sec, ESA STEP)"
    vertical_datum = "EGM96"
    native_res_m = 30.0
    lat_range = (-56.0, 60.0)
    BASE_URL = "https://step.esa.int/auxdata/dem/SRTMGL1"
    HGT_BYTES = 3601 * 3601 * 2  # raw binary samples

    def tile_id(self, lon: int, lat: int) -> str:
        return (
            f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
        )

    def tile_url(self, tile_id: str) -> str:
        return f"{self.BASE_URL}/{tile_id}.SRTMGL1.hgt.zip"

    def cached_raster(self, cache_dir: Path, tile_id: str) -> Path:
        return Path(cache_dir) / self.name / f"{tile_id}.hgt"

    def store(self, payload: Path, cache_dir: Path, tile_id: str) -> Path:
        """Extract .hgt from the zip archive."""
        dest = self.cached_raster(cache_dir, tile_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.part-{id(self)}")
        try:
            with zipfile.ZipFile(payload) as zf:
                members = [n for n in zf.namelist() if n.lower().endswith(".hgt")]
                if not members:
                    raise DEMDownloadError(f"No .hgt in {tile_id} archive")
                with zf.open(members[0]) as src, tmp.open("wb") as out:
                    while chunk := src.read(CHUNK_SIZE):
                        out.write(chunk)
            size = tmp.stat().st_size
            if size != self.HGT_BYTES:
                raise DEMDownloadError(
                    f"{tile_id}.hgt is {size} bytes, expected {self.HGT_BYTES}"
                )
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
            payload.unlink(missing_ok=True)
        return dest


class CopernicusProvider(DEMProvider):
    """Copernicus DEM GLO-30 (~30 m) via AWS Open Data.

    Coverage: global (including above 60N). Delivered as Cloud-Optimized
    GeoTIFFs with EGM2008 vertical reference. Tiles are decimated above
    50°N (smaller file sizes).
    """

    name = "cop30"
    label = "Copernicus DEM GLO-30 (AWS Open Data)"
    vertical_datum = "EGM2008"
    native_res_m = 30.0
    lat_range = (-90.0, 90.0)
    BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

    def tile_id(self, lon: int, lat: int) -> str:
        return (
            "Copernicus_DSM_COG_10_"
            f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
        )

    def tile_url(self, tile_id: str) -> str:
        return f"{self.BASE_URL}/{tile_id}/{tile_id}.tif"

    def cached_raster(self, cache_dir: Path, tile_id: str) -> Path:
        return Path(cache_dir) / self.name / f"{tile_id}.tif"

    def store(self, payload: Path, cache_dir: Path, tile_id: str) -> Path:
        """Move the COG into cache (already in final format)."""
        dest = self.cached_raster(cache_dir, tile_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Path(payload).replace(dest)
        return dest


# ============================================================================
# DEM Client: Authentication, Caching, Downloads
# ============================================================================


class DEMClient:
    """Handles DEM tile caching and downloads.

    One client serves all providers; tiles are cached per provider in a
    shared directory. No authentication needed — all sources are public.
    """

    def __init__(self, cache_dir: Path | str | None = None):
        """Initialize the DEM client with a cache directory.

        Parameters
        ----------
        cache_dir : Path | str | None
            Local cache directory for tiles. If None, uses ~/.cache/geomulticorr/dems.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.providers = {
            "srtm": SRTMProvider(),
            "cop30": CopernicusProvider(),
        }

    def download_tile(
        self, provider_name: str, tile_id: str, timeout: float = 60.0
    ) -> Path | None:
        """Download a single tile, caching it locally.

        Parameters
        ----------
        provider_name : str
            Provider to use ("srtm" or "cop30").
        tile_id : str
            Tile identifier from the provider.
        timeout : float
            Download timeout in seconds.

        Returns
        -------
        Path | None
            Path to cached raster, or None if tile doesn't exist (404).

        Raises
        ------
        DEMDownloadError
            If download fails for any reason other than 404.
        """
        provider = self.providers[provider_name]
        cached = provider.cached_raster(self.cache_dir, tile_id)
        if cached.is_file():
            return cached  # cache hit

        url = provider.tile_url(tile_id)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = self._download_url(url, tmp_path, timeout=timeout)
            if result is None:
                return None  # 404: tile doesn't exist
            return provider.store(tmp_path, self.cache_dir, tile_id)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _download_url(self, url: str, dest: Path, timeout: float = 60.0) -> int | None:
        """Download URL to dest; return bytes written, or None on 404."""
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return self._write_atomic(response, dest)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise DEMDownloadError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DEMDownloadError(f"Cannot reach {url}: {exc}") from exc

    @staticmethod
    def _write_atomic(stream, dest: Path) -> int:
        """Stream to a temp file, then rename atomically."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.part-{os.getpid()}")
        try:
            with tmp.open("wb") as fh:
                while chunk := stream.read(CHUNK_SIZE):
                    fh.write(chunk)
            size = tmp.stat().st_size
            tmp.replace(dest)
            return size
        finally:
            tmp.unlink(missing_ok=True)


# ============================================================================
# DEM Finder: Search & Download for an AOI
# ============================================================================


class DEMFinder:
    """Search and download DEMs for an AOI.

    Takes an AOI (bounding box or GeoDataFrame) and returns metadata about
    available tiles, then downloads them on request.
    """

    def __init__(
        self,
        client: DEMClient,
        aoi,
        provider: str = "cop30",
    ):
        """Initialize the DEM finder.

        Parameters
        ----------
        client : DEMClient
            DEM client handling caching and downloads.
        aoi
            Area of interest: GeoDataFrame, Shapely geometry, or bbox tuple
            (west, south, east, north).
        provider : str
            DEM provider to use ("srtm" or "cop30"). Default: "cop30" (global).
        """
        self.client = client
        self.provider_name = provider
        self.provider = client.providers[provider]

        # Coerce AOI to bbox
        if isinstance(aoi, gpd.GeoDataFrame):
            bounds = aoi.total_bounds
            self.bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
        elif hasattr(aoi, "bounds"):  # Shapely geometry
            b = aoi.bounds
            self.bbox = (b[0], b[1], b[2], b[3])
        elif isinstance(aoi, (tuple, list)) and len(aoi) == 4:
            self.bbox = tuple(aoi)
        else:
            raise ValueError(f"Cannot coerce {type(aoi)} to bbox")

    def search_metadata(self) -> gpd.GeoDataFrame:
        """Metadata about available tiles for the AOI.

        Returns
        -------
        GeoDataFrame
            Columns: tile_id, provider, resolution_m, vertical_datum, geometry.
        """
        west, south, east, north = self.bbox
        tile_ids = self.provider.tile_ids(west, south, east, north)

        rows = []
        for tile_id in tile_ids:
            # Tile geometry: extract lon/lat from tile_id
            # SRTM: N45E006, Copernicus: Copernicus_DSM_COG_10_N45_00_E006_00_DEM
            if self.provider_name == "srtm":
                parts = (
                    tile_id[0:3],  # N/S + lat
                    tile_id[3:7],  # E/W + lon
                )
                lat_str, lon_str = parts
            else:  # cop30
                # Extract from middle part
                parts = tile_id.split("_")
                lat_str = f"{parts[4]}{parts[5]}"
                lon_str = f"{parts[6]}{parts[7]}"

            lat = int(lat_str[1:3]) if lat_str[0] in "NS" else int(lat_str[:2])
            lon = int(lon_str[1:4]) if lon_str[0] in "EW" else int(lon_str[:3])
            if lat_str[0] == "S":
                lat = -lat
            if lon_str[0] == "W":
                lon = -lon

            tile_box = box(lon, lat, lon + 1, lat + 1)
            rows.append({
                "tile_id": tile_id,
                "provider": self.provider_name,
                "resolution_m": self.provider.native_res_m,
                "vertical_datum": self.provider.vertical_datum,
                "geometry": tile_box,
            })

        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        return gdf

    def download_dem(
        self,
        output_dir: Path | str | None = None,
        output_name: str | None = None,
        timeout: float = 60.0,
        show_progress: bool = True,
    ) -> Path:
        """Download and mosaic all tiles for the AOI.

        Parameters
        ----------
        output_dir : Path | str | None
            Output directory. If None, uses current directory.
        output_name : str | None
            Output filename. If None, uses {provider}_{bbox}.tif.
        timeout : float
            Download timeout per tile in seconds.
        show_progress : bool
            Show progress bar during download.

        Returns
        -------
        Path
            Path to the output mosaic GeoTIFF.
        """
        import subprocess

        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_name:
            west, south, east, north = self.bbox
            output_name = f"{self.provider_name}_{west:.1f}_{south:.1f}_{east:.1f}_{north:.1f}.tif"
        output_path = output_dir / output_name

        # Get tile list
        west, south, east, north = self.bbox
        tile_ids = self.provider.tile_ids(west, south, east, north)

        if not tile_ids:
            raise ValueError(f"No tiles for {self.provider_name} in bbox {self.bbox}")

        # Download tiles
        cached_tiles = []
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(tile_ids, desc=f"Downloading {self.provider_name} tiles")
            except ImportError:
                pbar = tile_ids
        else:
            pbar = tile_ids

        for tile_id in pbar:
            tile_path = self.client.download_tile(self.provider_name, tile_id, timeout=timeout)
            if tile_path:
                cached_tiles.append(str(tile_path))

        if not cached_tiles:
            raise DEMDownloadError(f"No tiles downloaded for {self.bbox}")

        # Mosaic with gdalbuildvrt + gdal_translate
        if len(cached_tiles) == 1:
            # Single tile: just copy/convert
            shutil.copy(cached_tiles[0], output_path)
        else:
            # Multiple tiles: build VRT then translate
            vrt_path = output_path.with_suffix(".vrt")
            gdalbuildvrt_cmd = ["gdalbuildvrt", "-resolution", "highest", str(vrt_path)] + cached_tiles
            subprocess.run(gdalbuildvrt_cmd, check=True, capture_output=True)

            gdal_translate_cmd = ["gdal_translate", "-co", "COMPRESS=deflate", str(vrt_path), str(output_path)]
            subprocess.run(gdal_translate_cmd, check=True, capture_output=True)

            vrt_path.unlink(missing_ok=True)

        return output_path
