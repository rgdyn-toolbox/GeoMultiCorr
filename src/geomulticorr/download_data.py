#!/usr/bin/env python
# coding=utf-8
# ----------------------------------------------------------------------------- #
# GeoMultiCorr (GMC) project
# Copyright (C) GeoMultiCorr developer team, 2024.
# session.py
# creation date: 2024-08-12.
#
# Author(s) metadata
# -> author: Diego CUSICANQUI
#   -> affiliation: CNES | ISTerre | Univ. Grenoble Alpes
#   -> email(s): diego.cusicanqui@univ-grenoble-alpes.fr | diego.cusicanqui.vg@gmail.com
# -> author: Juan Cruz GHILARDI
#   -> affiliation: IANIGLA | Univ. Mendoza
#   -> email(s): jcghilardi@mendoza-conicet.gob.ar
# ----------------------------------------------------------------------------- #
from __future__ import annotations
from pathlib import Path
import ee
import geemap
import geopandas as gpd
from typing import Optional


def _ensure_ee_initialized(
    authenticate_if_needed: bool = True,
    service_account: Optional[str] = None,
    key_file: Optional[str] = None,
    project: Optional[str] = None,
):
    """Idempotent Earth Engine initialization.

    Parameters
    ----------
    authenticate_if_needed : bool
        If True, attempt interactive ee.Authenticate() when initialization fails.
    service_account : str, optional
        Service account email. If provided, uses key_file credentials path.
    key_file : str, optional
        Path to JSON key for the service account.
    project : str, optional
        GEE Cloud Project (for newer EE projects quota separation).
    """
    # Newer ee library exposes ee.data._initialized; fall back safely if absent.
    already = getattr(ee.data, "_initialized", False)
    if already:
        return
    try:
        if service_account:
            if not key_file:
                raise ValueError("key_file must be provided when using service_account")
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials=credentials, project=project)
        else:
            # Standard user credentials (assumes prior ee.Authenticate())
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
    except Exception as e:  # noqa: BLE001 - broad to optionally trigger auth
        if not authenticate_if_needed or service_account:
            raise
        # Attempt interactive auth then retry once
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()

DEFAULT_COLLECTIONS = {
    # Landsat Collection 2 Level-2 Surface Reflectance (no Panchromatic SR_B8 band provided)
    # Use SR_B5 (NIR), SR_B4 (Red), SR_B3 (Green) or adjust as needed. 30 m resolution.
    "L9": {"id": "LANDSAT/LC09/C02/T1_L2", "bands": ['SR_B5','SR_B4','SR_B3'], "cloud_field": "CLOUD_COVER", "scale": 30},
    "L8": {"id": "LANDSAT/LC08/C02/T1_L2", "bands": ['SR_B5','SR_B4','SR_B3'], "cloud_field": "CLOUD_COVER", "scale": 30},
    "L7": {"id": "LANDSAT/LE07/C02/T1_L2", "bands": ['SR_B5','SR_B4','SR_B3'], "cloud_field": "CLOUD_COVER", "scale": 30},
    "L5": {"id": "LANDSAT/LT05/C02/T1_L2", "bands": ['SR_B4','SR_B3','SR_B2'], "cloud_field": "CLOUD_COVER", "scale": 30},
    # Optional Panchromatic (15 m) from Level-1 products (uncorrected TOA). Band name is B8.
    "L9_PAN": {"id": "LANDSAT/LC09/C02/T1", "bands": ["B8"], "cloud_field": "CLOUD_COVER", "scale": 15},
    "L8_PAN": {"id": "LANDSAT/LC08/C02/T1", "bands": ["B8"], "cloud_field": "CLOUD_COVER", "scale": 15},
    "L7_PAN": {"id": "LANDSAT/LE07/C02/T1", "bands": ["B8"], "cloud_field": "CLOUD_COVER", "scale": 15},
    # Sentinel-2 (harmonized) 10 m core bands
    "S2":    {"id": "COPERNICUS/S2_SR_HARMONIZED", "bands": ["B8", "B4", "B3", "B2"], "cloud_field": "CLOUDY_PIXEL_PERCENTAGE", "scale": 10},
}

# Mission operational year ranges (inclusive) for quick pruning
MISSION_YEARS = {
    "L5_SR": (1984, 2013),
    "L7_SR": (1999, 2025),  # ETM+ still delivering (with SLC-off gaps)
    "L8_SR": (2013, 2025),
    "L9_SR": (2021, 2025),
    "L9_PAN": (2021, 2025),
    "L8_PAN": (2013, 2025),
    "L7_PAN": (1999, 2025),
    "S2": (2015, 2025),
}

class EEDownloadManager:
    def __init__(self,
                 polygon: ee.Geometry,
                 point: ee.Geometry = None,
                 collections: dict[str, dict] = None,
                 year_range: tuple[int,int] = (2020,2020),
                 month_range: tuple[int,int] = (1,12),
                 cloud_max: int = 30,
                 crs: str = "EPSG:4326"):
        ee.Initialize()
        self.polygon = polygon
        self.point = point
        self.collections_cfg = collections or DEFAULT_COLLECTIONS.copy()
        self.year_range = year_range
        self.month_range = month_range
        self.cloud_max = cloud_max
        self.crs = crs
        self._built: dict[str, ee.ImageCollection] = {}

    @staticmethod
    def to_geojson(gdf: gpd.GeoDataFrame) -> ee.Geometry:
        """Return an ee.Geometry built from a GeoDataFrame (dissolved union).

        Ensures CRS is WGS84. Supports Polygon / MultiPolygon.
        """
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        dissolved = gdf.union_all()
        geom_json = dissolved.__geo_interface__
        if dissolved.geom_type == "Polygon":
            return ee.Geometry.Polygon(geom_json["coordinates"])
        if dissolved.geom_type == "MultiPolygon":
            return ee.Geometry.MultiPolygon(geom_json["coordinates"])
        raise ValueError(f"Unsupported geometry type: {dissolved.geom_type}")

    def add_collection(self, key: str, cfg: dict) -> None:
        self.collections_cfg[key] = cfg

    def list_collections(self) -> list[str]:
        return list(self.collections_cfg.keys())

    def _base_filters(self, ic: ee.ImageCollection, cfg: dict) -> ee.ImageCollection:
        ic = (ic
              .filterBounds(self.polygon)
              .filter(ee.Filter.lt(cfg["cloud_field"], self.cloud_max))
              .filter(ee.Filter.calendarRange(self.year_range[0], self.year_range[1], 'year'))
              .filter(ee.Filter.calendarRange(self.month_range[0], self.month_range[1], 'month')))
        if self.point:
            ic = ic.filterBounds(self.point)
        return ic

    def build(self, keys: list[str] | None = None, overwrite: bool = False) -> dict[str, ee.ImageCollection]:
        target = keys or self.list_collections()
        for k in target:
            if not overwrite and k in self._built:
                continue
            cfg = self.collections_cfg[k]
            ic = ee.ImageCollection(cfg["id"])
            ic = self._base_filters(ic, cfg).select(cfg["bands"])
            self._built[k] = ic
        return {k: self._built[k] for k in target}

    def get(self, key: str) -> ee.ImageCollection:
        if key not in self._built:
            self.build([key])
        return self._built[key]

    def export(self,
               out_dir: str | Path,
               keys: list[str] | None = None,
               make_dirs: bool = True,
               region: ee.Geometry = None) -> dict:
        """Export imagery for selected collections.

        Parameters
        ----------
        out_dir : Path-like
            Base output directory.
        keys : list[str], optional
            Collection keys to export; defaults to all built or all configured.
        make_dirs : bool
            Create subdirectories per collection.
        region : ee.Geometry, optional
            Region override; defaults to manager polygon.

        Returns
        -------
        dict
            Mapping collection key -> list of (image_id, status, message)
        """
        out_dir = Path(out_dir)
        exports = keys or self._built.keys() or self.list_collections()
        self.build(list(exports))  # ensure built

        for k in exports:
            cfg = self.collections_cfg[k]
            col = self._built[k]
            subdir = out_dir / k
            if make_dirs:
                subdir.mkdir(parents=True, exist_ok=True)
            geemap.ee_export_image_collection(
                col,
                out_dir=str(subdir),
                scale=cfg["scale"],
                crs=self.crs,
                region=region or self.polygon,
            )

    def summary(self) -> dict:
        return {k: self._built[k].size().getInfo() if k in self._built else 0 for k in self.list_collections()}

    
