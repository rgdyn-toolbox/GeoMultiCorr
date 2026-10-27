#!/usr/bin/env python
# coding=utf-8
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# mpc_image_search_download.py
# creation date: 2026-07-10.
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
"""Search and bulk-download Landsat / Sentinel-2 imagery from Microsoft Planetary Computer.

This module is the STAC-based counterpart of
:mod:`geomulticorr.gee.gee_image_search_download`. It queries the Microsoft
Planetary Computer (MPC) STAC API — **no account, no API key, no quota-limited
export tasks** — and downloads Cloud-Optimized GeoTIFF (COG) assets straight
from Azure Blob Storage.

Three download modes are supported per scene:

* ``mode="crop"``  — windowed COG read of the AOI only (HTTP range requests;
  only the needed bytes travel over the network).
* ``mode="full"``  — stream the original, unmodified full-extent asset files.
* ``download_metadata=True`` — additionally fetch the original vendor metadata
  (Landsat ``MTL.xml``, Sentinel-2 ``MTD_TL.xml`` / ``MTD_MSIL2A.xml``).

Coverage notes
--------------
* MPC ``landsat-c2-l2`` spans Landsat 4/5/7/8/9 surface reflectance at 30 m.
* The 15 m panchromatic band (Landsat 7/8/9 Level-1 ``B8``) is **not** hosted
  on MPC (its ``landsat-c2-l1`` collection is MSS-only). Use the GEE module or
  USGS M2M for panchromatic products.
* Landsat 6 never reached orbit; no archive exists anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import time as _time
import logging as _logging

import geopandas as gpd
import numpy as np
import requests
import rasterio
from rasterio.windows import Window, from_bounds as _window_from_bounds
from rasterio.warp import transform_bounds as _transform_bounds
from shapely.geometry import shape as _shape
from shapely.geometry import mapping as _mapping
from shapely.geometry.base import BaseGeometry as _BaseGeometry
from pyproj import Geod as _Geod

import planetary_computer
import pystac
import pystac_client

from geomulticorr.data_sources._layout import (
    ORGANIZE_SCHEMES,
    ORGANIZE_SENSOR_PATHROW,
    landsat_dirs,
    sentinel2_dirs,
)

# Optional progress bar
try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

LOGGER = _logging.getLogger("mpc_image_search_download")

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

#: GDAL environment tuned for remote COG reads (fewer requests, automatic retries).
_GDAL_COG_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="2",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
)

_WGS84_GEOD = _Geod(ellps="WGS84")


def _scene_dirs(spec: "CollectionSpec", item: pystac.Item) -> Tuple[str, str]:
    """Archive ``(sensor_dir, grid_dir)`` for one STAC item.

    Dispatches on the STAC collection rather than the catalog key, so a
    user-defined :class:`CollectionSpec` pointed at ``sentinel-2-l2a`` or
    ``landsat-c2-*`` is routed correctly whatever it calls itself.

    Landsat platform and path/row come from the item id (``LC08_L2SP_232083_…``);
    the Sentinel-2 MGRS tile is read from the item's STAC properties first and
    only falls back to parsing the id.
    """
    collection = (spec.stac_collection or "").lower()
    if collection.startswith("sentinel-2"):
        return sentinel2_dirs(item.id, item.properties)
    if collection.startswith("landsat"):
        return landsat_dirs(item.id)
    # Unknown collection: keep it inspectable rather than guessing.
    return spec.key, "unknown"


def _geodesic_area_m2(geom: _BaseGeometry) -> float:
    """Return the geodesic area (m²) of a WGS84 shapely geometry."""
    area, _ = _WGS84_GEOD.geometry_area_perimeter(geom)
    return abs(area)


# -------------------------
# 1) Collection catalog
# -------------------------
@dataclass(frozen=True)
class CollectionSpec:
    """Description of one queryable MPC collection (optionally platform-filtered).

    Parameters
    ----------
    key : str
        Short identifier used throughout this module (e.g. ``"L8_L2"``).
    stac_collection : str
        MPC STAC collection id (e.g. ``"landsat-c2-l2"``).
    nominal_scale_m : int
        Nominal ground pixel size of the default bands.
    pretty_name : str
        Human-readable collection name.
    platform : str, optional
        Value of the STAC ``platform`` property used to split multi-satellite
        collections (e.g. ``"landsat-8"``). ``None`` -> no platform filter.
    bands : sequence of str, optional
        Default asset keys downloaded for imagery (e.g. ``("nir08", "red", "green")``).
    metadata_assets : sequence of str
        Asset keys holding the original vendor metadata files (XML/TXT).
    """

    key: str
    stac_collection: str
    nominal_scale_m: int
    pretty_name: str
    platform: Optional[str] = None
    bands: Optional[Sequence[str]] = None
    metadata_assets: Sequence[str] = field(default_factory=tuple)


class ImageCollectionCatalog:
    """Registry of commonly used MPC optical collections.

    Mirrors the API of the GEE catalog in
    :mod:`geomulticorr.gee.gee_image_search_download`.
    """

    def __init__(self, specs: Sequence[CollectionSpec]):
        self._specs: Dict[str, CollectionSpec] = {s.key: s for s in specs}

    @classmethod
    def default(cls) -> "ImageCollectionCatalog":
        """Return a catalog with Landsat 5/7/8/9 SR and Sentinel-2 L2A.

        All Landsat entries point to the same ``landsat-c2-l2`` STAC collection
        and are split by the ``platform`` property. Default bands follow the
        GEE module's choices (NIR/red/green for Landsat, B08/B04/B03/B02 for
        Sentinel-2) — asset names are identical across TM/ETM+/OLI thanks to
        Collection-2 harmonised naming (``nir08``, ``red``, ``green``).
        """
        landsat_meta = ("mtl.xml",)
        return cls(
            [
                CollectionSpec(
                    key="L9_SR",
                    stac_collection="landsat-c2-l2",
                    platform="landsat-9",
                    nominal_scale_m=30,
                    pretty_name="Landsat 9 SR (C2 L2)",
                    bands=("nir08", "red", "green"),
                    metadata_assets=landsat_meta,
                ),
                CollectionSpec(
                    key="L8_SR",
                    stac_collection="landsat-c2-l2",
                    platform="landsat-8",
                    nominal_scale_m=30,
                    pretty_name="Landsat 8 SR (C2 L2)",
                    bands=("nir08", "red", "green"),
                    metadata_assets=landsat_meta,
                ),
                CollectionSpec(
                    key="L7_SR",
                    stac_collection="landsat-c2-l2",
                    platform="landsat-7",
                    nominal_scale_m=30,
                    pretty_name="Landsat 7 SR (C2 L2)",
                    bands=("nir08", "red", "green"),
                    metadata_assets=landsat_meta,
                ),
                CollectionSpec(
                    key="L5_SR",
                    stac_collection="landsat-c2-l2",
                    platform="landsat-5",
                    nominal_scale_m=30,
                    pretty_name="Landsat 5 SR (C2 L2)",
                    bands=("nir08", "red", "green"),
                    metadata_assets=landsat_meta,
                ),
                CollectionSpec(
                    key="S2_SR",
                    stac_collection="sentinel-2-l2a",
                    platform=None,
                    nominal_scale_m=10,
                    pretty_name="Sentinel-2 L2A",
                    bands=("B08", "B04", "B03", "B02"),
                    metadata_assets=("granule-metadata", "product-metadata"),
                ),
            ]
        )

    def specs(self) -> List[CollectionSpec]:
        """List all collection specifications."""
        return list(self._specs.values())

    def get(self, key: str) -> CollectionSpec:
        """Retrieve a collection specification by key (raises ``KeyError`` if absent)."""
        return self._specs[key]

    def keys(self) -> List[str]:
        """Return all registered collection keys."""
        return list(self._specs.keys())


# -------------------------
# 2) Image search and download
# -------------------------
class ImageFinder:
    """Search MPC collections, build GeoDataFrame metadata and download scenes.

    Parameters
    ----------
    catalog : ImageCollectionCatalog
        Registry of collections to query.
    aoi : GeoDataFrame | shapely geometry | GeoJSON-like mapping
        Area of interest. A GeoDataFrame is re-projected to WGS84 and dissolved
        into a single union.
    year_range : tuple[int, int], default (2020, 2025)
        Inclusive year range filter.
    month_range : tuple[int, int], default (1, 12)
        Inclusive month (1-12) range filter (applied client-side).
    cloud_max : int, default 30
        Maximum ``eo:cloud_cover`` percentage.
    stac_url : str
        STAC API endpoint (defaults to the public MPC endpoint).

    Notes
    -----
    No authentication is required: asset URLs are signed anonymously through
    ``planetary_computer.sign_inplace`` (short-lived read-only SAS tokens).
    """

    def __init__(
        self,
        catalog: ImageCollectionCatalog,
        *,
        aoi: Union[gpd.GeoDataFrame, _BaseGeometry, Mapping],
        year_range: Tuple[int, int] = (2020, 2025),
        month_range: Tuple[int, int] = (1, 12),
        cloud_max: int = 30,
        stac_url: str = MPC_STAC_URL,
    ) -> None:
        self._catalog = catalog
        self.year_range = year_range
        self.month_range = month_range
        self.cloud_max = cloud_max
        self.aoi_geom: _BaseGeometry = self._coerce_to_wgs84_geometry(aoi)
        self._stac = pystac_client.Client.open(
            stac_url, modifier=planetary_computer.sign_inplace
        )
        # Cache of STAC items from the last search, keyed by "<collection_key>/<item_id>"
        self._item_cache: Dict[str, pystac.Item] = {}

    # ---- public API
    def search_metadata(
        self, collection_keys: Optional[Sequence[str]] = None
    ) -> gpd.GeoDataFrame:
        """Search collections and build a GeoDataFrame of per-image metadata.

        Parameters
        ----------
        collection_keys : sequence[str], optional
            Subset of catalog keys to search. ``None`` -> all in the catalog.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per scene, columns compatible with the GEE module:
            ``image_id``, ``collection_key``, ``collection_name``,
            ``acquisition_date``, ``cloud_percent``, ``nominal_scale_m``,
            ``footprint_area_m2`` / ``footprint_area_km2``,
            ``aoi_coverage_percent``, ``overlap_ratio``, ``geometry``
            (footprint polygon, EPSG:4326). Sorted by date (desc) then cloud (asc).
        """
        selected = (
            [self._catalog.get(k) for k in collection_keys]
            if collection_keys
            else self._catalog.specs()
        )

        aoi_area_m2 = _geodesic_area_m2(self.aoi_geom)
        datetime_range = f"{self.year_range[0]}-01-01/{self.year_range[1]}-12-31"

        rows: List[dict] = []
        for spec in selected:
            query: Dict[str, dict] = {"eo:cloud_cover": {"lt": self.cloud_max}}
            if spec.platform:
                query["platform"] = {"eq": spec.platform}

            search = self._stac.search(
                collections=[spec.stac_collection],
                intersects=_mapping(self.aoi_geom),
                datetime=datetime_range,
                query=query,
            )
            for item in search.items():
                dt = item.datetime
                if dt is None or not (
                    self.month_range[0] <= dt.month <= self.month_range[1]
                ):
                    continue
                footprint = _shape(item.geometry)
                inter = footprint.intersection(self.aoi_geom)
                if inter.is_empty:
                    continue
                footprint_area_m2 = _geodesic_area_m2(footprint)
                inter_area_m2 = _geodesic_area_m2(inter)
                self._item_cache[f"{spec.key}/{item.id}"] = item
                rows.append(
                    {
                        "image_id": item.id,
                        "collection_key": spec.key,
                        "collection_name": spec.pretty_name,
                        "acquisition_date": dt.strftime("%Y-%m-%d"),
                        "cloud_percent": float(
                            item.properties.get("eo:cloud_cover", np.nan)
                        ),
                        "nominal_scale_m": int(spec.nominal_scale_m),
                        "footprint_area_m2": footprint_area_m2,
                        "footprint_area_km2": footprint_area_m2 / 1e6,
                        "aoi_coverage_percent": 100.0 * inter_area_m2 / aoi_area_m2,
                        "overlap_ratio": inter_area_m2 / footprint_area_m2,
                        "geometry": footprint,
                    }
                )

        if not rows:
            return gpd.GeoDataFrame(
                columns=[
                    "image_id",
                    "collection_key",
                    "collection_name",
                    "acquisition_date",
                    "cloud_percent",
                    "nominal_scale_m",
                    "footprint_area_m2",
                    "footprint_area_km2",
                    "aoi_coverage_percent",
                    "overlap_ratio",
                    "geometry",
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        gdf["acquisition_date"] = gdf["acquisition_date"].astype("datetime64[ns]")
        gdf = gdf.sort_values(
            ["acquisition_date", "cloud_percent"], ascending=[False, True]
        ).reset_index(drop=True)
        return gdf

    def bulk_download(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        output_dir: Union[str, Path],
        mode: str = "crop",  # "crop" or "full"
        download_metadata: bool = True,
        file_basename: Optional[str] = None,
        bands_override: Optional[Sequence[str]] = None,
        max_images: Optional[int] = None,
        throttle_s: float = 0.0,
        max_retries: int = 2,
        retry_backoff_s: float = 2.0,
        show_progress: bool = True,
        overwrite: bool = False,
        organize_by: Optional[str] = None,
    ) -> List[Path]:
        """Download the scenes referenced in a metadata GeoDataFrame.

        Parameters
        ----------
        images_gdf : geopandas.GeoDataFrame
            Output of :meth:`search_metadata` (possibly filtered by the user).
            Must contain ``image_id`` and ``collection_key`` columns.
        output_dir : str | Path
            Root directory; one subdirectory per collection key is created.
        mode : {"crop", "full"}, default "crop"
            ``"crop"``: windowed COG read of the AOI bounding box only; the
            selected bands are stacked into a single multi-band GeoTIFF in the
            asset's native CRS/grid. ``"full"``: stream each original asset
            file untouched (full scene extent, one file per band).
        download_metadata : bool, default True
            Also download the vendor metadata assets (Landsat ``MTL.xml``,
            Sentinel-2 ``MTD_TL.xml`` / ``MTD_MSIL2A.xml``).
        file_basename : str, optional
            Prefix for crop-mode filenames. Default: the collection key.
        bands_override : sequence[str], optional
            Explicit asset keys to download instead of the catalog defaults.
        max_images : int, optional
            Limit the number of scenes processed.
        throttle_s : float, default 0
            Sleep seconds between scenes (politeness pacing; MPC needs none).
        max_retries : int, default 2
            Retry attempts per scene on transient errors.
        retry_backoff_s : float, default 2.0
            Base seconds for exponential backoff (sleep = base * 2**attempt).
        show_progress : bool, default True
            Show a tqdm progress bar when available.
        overwrite : bool, default False
            Re-download files that already exist on disk.
        organize_by : {"sensor_pathrow"}, optional
            Directory layout under *output_dir*, for ``mode="full"`` only.
            ``None`` (default) keeps the historical
            ``<collection_key>/<item_id>/`` layout. ``"sensor_pathrow"`` writes
            to ``<PLATFORM>/<grid>/<item_id>/`` — ``SENTINEL2/T19HCC/…`` by MGRS
            tile, ``LANDSAT8/232_083/…`` by WRS-2 path/row — which is the layout
            a regional archive wants. Ignored with a warning in ``"crop"`` mode,
            which renames scenes by design.

            The per-scene ``<item_id>/`` level is kept because MPC asset hrefs
            are band-scoped (``B08.tif``): the scene identity lives in that
            directory name, not in the filenames.

        Returns
        -------
        list[pathlib.Path]
            Paths of all files written (rasters and metadata).

        Raises
        ------
        ValueError
            On missing columns, an unknown ``mode``, or an unknown
            ``organize_by`` scheme.
        RuntimeError
            If a scene ultimately fails after retries.
        """
        if mode not in ("crop", "full"):
            raise ValueError(f"mode must be 'crop' or 'full', got {mode!r}")
        if organize_by is not None and organize_by not in ORGANIZE_SCHEMES:
            raise ValueError(
                f"organize_by must be one of {ORGANIZE_SCHEMES} or None, got {organize_by!r}"
            )
        if organize_by is not None and mode == "crop":
            LOGGER.warning(
                "organize_by=%r ignored in crop mode, which renames scenes by design; "
                "use mode='full' to build a catalog-named archive.",
                organize_by,
            )
            organize_by = None
        required = {"image_id", "collection_key"}
        missing = required - set(images_gdf.columns)
        if missing:
            raise ValueError(
                f"images_gdf is missing required columns: {sorted(missing)}"
            )

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        rows = list(images_gdf.itertuples(index=False))
        if max_images is not None:
            rows = rows[: int(max_images)]
        iterator = (
            _tqdm(rows, desc="Downloading", unit="scene")
            if (_HAS_TQDM and show_progress)
            else rows
        )

        written: List[Path] = []
        for row in iterator:
            image_id: str = getattr(row, "image_id")
            collection_key: str = getattr(row, "collection_key")
            spec = self._catalog.get(collection_key)
            bands = (
                list(bands_override)
                if bands_override is not None
                else list(spec.bands or [])
            )

            attempt = 0
            while True:
                try:
                    item = self._resolve_item(collection_key, image_id)
                    scene_paths: List[Path] = []
                    if mode == "crop":
                        collection_dir = out_root / collection_key
                        collection_dir.mkdir(parents=True, exist_ok=True)
                        date_str = (
                            item.datetime.strftime("%Y-%m-%d")
                            if item.datetime
                            else "nodate"
                        )
                        base = file_basename or collection_key
                        out_path = collection_dir / f"{base}_{date_str}_{item.id}.tif"
                        p = self._download_crop(
                            item, bands, out_path, overwrite=overwrite
                        )
                        if p is not None:
                            scene_paths.append(p)
                        meta_dir = collection_dir
                    else:  # full
                        if organize_by == ORGANIZE_SENSOR_PATHROW:
                            sensor_dir, grid_dir = _scene_dirs(spec, item)
                            scene_dir = out_root / sensor_dir / grid_dir / item.id
                        else:
                            scene_dir = out_root / collection_key / item.id
                        scene_dir.mkdir(parents=True, exist_ok=True)
                        for band in bands:
                            p = self._download_asset_file(
                                item, band, scene_dir, overwrite=overwrite
                            )
                            if p is not None:
                                scene_paths.append(p)
                        meta_dir = scene_dir

                    if download_metadata:
                        for meta_key in spec.metadata_assets:
                            if meta_key not in item.assets:
                                LOGGER.warning(
                                    "metadata asset %r not found on item %s",
                                    meta_key,
                                    item.id,
                                )
                                continue
                            prefix = None
                            if mode == "crop":
                                # match the raster filename so scene and metadata pair up
                                date_str = (
                                    item.datetime.strftime("%Y-%m-%d")
                                    if item.datetime
                                    else "nodate"
                                )
                                prefix = f"{file_basename or collection_key}_{date_str}_{item.id}"
                            p = self._download_asset_file(
                                item,
                                meta_key,
                                meta_dir,
                                overwrite=overwrite,
                                rename_prefix=prefix,
                            )
                            if p is not None:
                                scene_paths.append(p)

                    written.extend(scene_paths)
                    if throttle_s > 0:
                        _time.sleep(throttle_s)
                    break
                except Exception as e:
                    LOGGER.warning(
                        "scene_fail id=%s attempt=%d error=%s", image_id, attempt, e
                    )
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"Failed to download {image_id} (collection {collection_key}). "
                            f"Last error: {e}"
                        ) from e
                    _time.sleep(retry_backoff_s * (2**attempt))
                    attempt += 1

        return written

    # ---- internals
    def _resolve_item(self, collection_key: str, image_id: str) -> pystac.Item:
        """Return the (signed) STAC item, from cache or by a targeted id search."""
        cached = self._item_cache.get(f"{collection_key}/{image_id}")
        if cached is not None:
            # Re-sign: SAS tokens from an earlier search may have expired.
            return planetary_computer.sign(cached)
        spec = self._catalog.get(collection_key)
        search = self._stac.search(collections=[spec.stac_collection], ids=[image_id])
        items = list(search.items())
        if not items:
            raise RuntimeError(
                f"Item {image_id!r} not found in collection {spec.stac_collection!r}."
            )
        self._item_cache[f"{collection_key}/{image_id}"] = items[0]
        return items[0]

    def _download_crop(
        self,
        item: pystac.Item,
        bands: Sequence[str],
        out_path: Path,
        *,
        overwrite: bool,
    ) -> Optional[Path]:
        """Windowed-read `bands` over the AOI bbox and write one multi-band GeoTIFF.

        All requested bands must share the same grid (true for the catalog
        defaults: Landsat SR 30 m, Sentinel-2 10 m bands).
        """
        if not bands:
            raise ValueError("crop mode requires at least one band asset key")
        if out_path.exists() and not overwrite:
            LOGGER.info("skip existing %s", out_path)
            return None

        arrays: List[np.ndarray] = []
        profile: Optional[dict] = None
        with rasterio.Env(**_GDAL_COG_ENV):
            for band in bands:
                if band not in item.assets:
                    raise KeyError(
                        f"asset {band!r} not found on item {item.id} "
                        f"(available: {sorted(item.assets)})"
                    )
                href = item.assets[band].href
                with rasterio.open(href) as src:
                    bounds = _transform_bounds(
                        "EPSG:4326", src.crs, *self.aoi_geom.bounds, densify_pts=21
                    )
                    window = (
                        _window_from_bounds(*bounds, transform=src.transform)
                        .round_offsets()
                        .round_lengths()
                    )
                    window = window.intersection(Window(0, 0, src.width, src.height))
                    if window.width <= 0 or window.height <= 0:
                        raise RuntimeError(
                            f"AOI does not intersect the valid grid of {item.id}/{band}"
                        )
                    data = src.read(1, window=window)
                    if profile is None:
                        profile = {
                            "driver": "GTiff",
                            "width": int(window.width),
                            "height": int(window.height),
                            "count": len(bands),
                            "dtype": data.dtype,
                            "crs": src.crs,
                            "transform": src.window_transform(window),
                            "nodata": src.nodata,
                            "compress": "deflate",
                            "tiled": True,
                        }
                    elif data.shape != arrays[0].shape:
                        raise RuntimeError(
                            f"band {band!r} grid differs from {bands[0]!r} on {item.id}; "
                            "crop mode requires same-resolution bands "
                            "(use bands_override to pick a consistent set)"
                        )
                    arrays.append(data)

        assert profile is not None
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, (band, data) in enumerate(zip(bands, arrays), start=1):
                dst.write(data, i)
                dst.set_band_description(i, band)
        LOGGER.info("wrote %s", out_path)
        return out_path

    @staticmethod
    def _download_asset_file(
        item: pystac.Item,
        asset_key: str,
        out_dir: Path,
        *,
        overwrite: bool,
        rename_prefix: Optional[str] = None,
        chunk_bytes: int = 4 * 1024 * 1024,
    ) -> Optional[Path]:
        """Stream one asset (raster or metadata file) to disk, unmodified.

        The original archive filename (from the URL path, SAS token stripped)
        is kept; with ``rename_prefix`` the file is saved as
        ``<prefix>_<original-suffix-name>``.
        """
        if asset_key not in item.assets:
            raise KeyError(f"asset {asset_key!r} not found on item {item.id}")
        href = item.assets[asset_key].href
        original_name = Path(href.split("?", 1)[0]).name
        filename = (
            f"{rename_prefix}_{original_name}" if rename_prefix else original_name
        )
        out_path = out_dir / filename
        if out_path.exists() and not overwrite:
            LOGGER.info("skip existing %s", out_path)
            return None

        with requests.get(href, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_bytes):
                    f.write(chunk)
            tmp_path.rename(out_path)
        LOGGER.info("wrote %s", out_path)
        return out_path

    @staticmethod
    def _coerce_to_wgs84_geometry(
        aoi: Union[gpd.GeoDataFrame, _BaseGeometry, Mapping],
    ) -> _BaseGeometry:
        """Coerce AOI input (GeoDataFrame / shapely / GeoJSON-like) to a WGS84 shapely union."""
        if isinstance(aoi, gpd.GeoDataFrame):
            gdf = aoi
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            return gdf.union_all()
        if isinstance(aoi, _BaseGeometry):
            return aoi
        if isinstance(aoi, Mapping):
            return _shape(aoi)
        raise TypeError(
            "AOI must be a GeoDataFrame, shapely geometry, or GeoJSON-like mapping"
        )


# -------------------------
# Optional: Command Line Interface (CLI)
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search and download Landsat/Sentinel-2 imagery from Microsoft Planetary Computer (no API key)."
    )
    parser.add_argument(
        "aoi_file", type=str, help="Path to AOI file (GeoPackage/GeoJSON/Shapefile)"
    )
    parser.add_argument("out_dir", type=str, help="Output directory for downloads")
    parser.add_argument(
        "--years",
        type=int,
        nargs=2,
        default=(2020, 2025),
        help="Year range inclusive, e.g. 2019 2021",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs=2,
        default=(1, 12),
        help="Month range inclusive, e.g. 6 9",
    )
    parser.add_argument("--cloud", type=int, default=30, help="Max cloud percent")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=None,
        help="Collection keys (e.g., S2_SR L8_SR). Default: all.",
    )
    parser.add_argument(
        "--mode",
        choices=["crop", "full"],
        default="crop",
        help="Download mode: AOI crop or full original scenes",
    )
    parser.add_argument(
        "--no-metadata", action="store_true", help="Skip vendor metadata (XML) download"
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Override asset keys to download (applies to all images)",
    )
    parser.add_argument(
        "--basename", type=str, default=None, help="Filename prefix for crop mode"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max scenes to download"
    )
    parser.add_argument(
        "--throttle", type=float, default=0.0, help="Seconds to sleep between scenes"
    )
    parser.add_argument("--retries", type=int, default=2, help="Max retries per scene")
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Base seconds for exponential backoff",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-download existing files"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar output"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()
    _logging.basicConfig(level=getattr(_logging, args.log_level.upper(), _logging.INFO))

    aoi_gdf = gpd.read_file(args.aoi_file)
    catalog = ImageCollectionCatalog.default()

    if args.collections is not None:
        unknown = sorted(set(args.collections) - set(catalog.keys()))
        if unknown:
            raise SystemExit(
                f"Unknown collection keys: {unknown}. Available: {catalog.keys()}"
            )

    finder = ImageFinder(
        catalog,
        aoi=aoi_gdf,
        year_range=tuple(args.years),
        month_range=tuple(args.months),
        cloud_max=int(args.cloud),
    )

    gdf = finder.search_metadata(collection_keys=args.collections)
    print("Found images:", len(gdf))
    print(gdf.head())

    paths = finder.bulk_download(
        gdf,
        output_dir=args.out_dir,
        mode=args.mode,
        download_metadata=not args.no_metadata,
        file_basename=args.basename,
        bands_override=args.bands,
        max_images=args.limit,
        throttle_s=args.throttle,
        max_retries=args.retries,
        retry_backoff_s=args.backoff,
        show_progress=not args.no_progress,
        overwrite=args.overwrite,
    )
    print(f"Downloaded {len(paths)} file(s) to {args.out_dir}")
