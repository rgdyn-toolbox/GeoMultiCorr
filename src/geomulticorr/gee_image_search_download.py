"""gee_image_search_download
=================================

High‑level, explicit helpers for:

1. Authenticating with / initializing the Google Earth Engine (GEE) Python API.
2. Searching multiple optical image collections (Surface Reflectance & TOA & Sentinel‑2) for
    images intersecting a user supplied Area of Interest (AOI) with filters on year, month,
    and cloud cover.
3. Computing a compact, analysis‑ready :class:`geopandas.GeoDataFrame` containing per‑image
    metadata plus footprint geometries suitable for plotting.
4. Bulk downloading the selected images with optional band & scale overrides, throttling,
    retries and progress indication.

Key derived metrics stored client‑side
-------------------------------------
* ``footprint_area_m2`` / ``footprint_area_km2`` – image footprint area.
* ``overlap_ratio`` – intersection_area / footprint_area (0–1) – how much of the footprint is over the AOI.
* ``aoi_coverage_percent`` – (intersection_area / AOI_area) * 100 – how much of the AOI is covered by the image.

Design principles
-----------------
* No hidden globals – everything flows through small objects.
* AOI accepted in several common forms (``GeoDataFrame`` / Shapely geometry / GeoJSON‑like mapping / ``ee.Geometry``).
* All spatial area computations happen server‑side in GEE for robustness / consistency.
* Returned GeoDataFrame geometry is always the **image footprint** (not the intersection) to keep plotting intuitive.
* Safe, explicit downloads with retry + backoff + optional progress bar (``tqdm``).

Quick example
-------------
>>> from gee_image_search_download import GEEClient, ImageCollectionCatalog, ImageFinder
>>> client = GEEClient(project=None).connect()
>>> catalog = ImageCollectionCatalog.default()
>>> finder = ImageFinder(client, catalog, aoi=my_aoi_gdf, year_range=(2019, 2021), month_range=(6, 9), cloud_max=20)
>>> images = finder.search_metadata()
>>> images.head()
>>> finder.bulk_download(images, output_dir="./downloads", region_mode="aoi", max_images=3)

References
----------
[1] Gorelick et al. (2017) Google Earth Engine. *PNAS*. https://doi.org/10.1073/pnas.1703516114
[2] Jordahl et al. (2020) GeoPandas. *JOSS*. https://doi.org/10.21105/joss.02026
[3] Wu (2020) geemap. *JOSS*. https://doi.org/10.21105/joss.02192
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import datetime as _dt
import time as _time
import logging as _logging

import ee
import geemap
import geopandas as gpd
from shapely.geometry import shape as _shape
from shapely.geometry import mapping as _mapping
from shapely.geometry.base import BaseGeometry as _BaseGeometry
from pyproj import CRS as _CRS

# Optional progress bar
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

LOGGER = _logging.getLogger("gee_image_search_download")

# -------------------------
# 1) Connection utilities
# -------------------------
class GEEClient:
    """Thin, explicit Google Earth Engine client wrapper.

    Parameters
    ----------
    project : str, optional
        GCP / GEE project identifier to bind requests (improves quota clarity).
    service_account : str, optional
        Email of a GEE‑enabled service account. When provided a ``key_file`` must also be
        supplied. If omitted, user / application default credentials are used.
    key_file : str, optional
        Path to a JSON key file for the service account.
    authenticate_if_needed : bool, default True
        When True and initialization fails for user credentials, trigger the interactive
        ``ee.Authenticate()`` flow (ignored for service accounts).

    Notes
    -----
    The object is intentionally lightweight and *idempotent*. Calling :meth:`connect` multiple
    times safely reuses an already initialized EE connection.
    """

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        service_account: Optional[str] = None,
        key_file: Optional[str] = None,
        authenticate_if_needed: bool = True,
    ) -> None:
        self.project = project
        self.service_account = service_account
        self.key_file = key_file
        self.authenticate_if_needed = authenticate_if_needed

    def connect(self) -> "GEEClient":
        """Initialize the Earth Engine API if not already initialized.

        Returns
        -------
        GEEClient
            The same instance (allows call chaining).

        Raises
        ------
        ValueError
            If a service account email is provided without a ``key_file``.
        Exception
            Any underlying authentication / initialization exceptions not handled by an
            interactive auth retry.
        """
        already = getattr(ee.data, "_initialized", False)
        if already:
            return self
        try:
            if self.service_account:
                if not self.key_file:
                    raise ValueError("When using a service account, provide 'key_file'.")
                credentials = ee.ServiceAccountCredentials(self.service_account, self.key_file)
                ee.Initialize(credentials=credentials, project=self.project)
            else:
                if self.project:
                    ee.Initialize(project=self.project)
                else:
                    ee.Initialize()
        except Exception:
            if not self.authenticate_if_needed or self.service_account:
                raise
            ee.Authenticate()
            if self.project:
                ee.Initialize(project=self.project)
            else:
                ee.Initialize()
        return self


# -------------------------
# 2) Collection catalog
# -------------------------
@dataclass(frozen=True)
class CollectionSpec:
    key: str
    ee_id: str
    cloud_property: str
    nominal_scale_m: int
    pretty_name: str
    bands: Optional[Sequence[str]] = None  # Optional selection for exports


class ImageCollectionCatalog:
    """Registry of commonly used optical image collections.

    Wraps a list of :class:`CollectionSpec` objects and offers simple accessor helpers.

    Methods
    -------
    default()
        Build a catalog pre‑populated with Landsat (SR + TOA) and Sentinel‑2 SR collections.
    specs()
        Return list of all collection specs.
    get(key)
        Retrieve a :class:`CollectionSpec` by its key.
    keys()
        List all registered keys.
    """

    def __init__(self, specs: Sequence[CollectionSpec]):
        self._specs: Dict[str, CollectionSpec] = {s.key: s for s in specs}

    @classmethod
    def default(cls) -> "ImageCollectionCatalog":
        """Return a catalog with a curated set of Landsat & Sentinel‑2 collections.

        Includes Surface Reflectance (SR) and Top‑Of‑Atmosphere (TOA) variants plus a
        panchromatic band selection (B8) for TOA collections where appropriate.
        """
        return cls(
            [
                CollectionSpec(
                    key="L9_SR",
                    ee_id="LANDSAT/LC09/C02/T1_L2",
                    cloud_property="CLOUD_COVER",
                    nominal_scale_m=30,
                    pretty_name="Landsat 9 SR (C2 L2)",
                    bands=["SR_B5", "SR_B4", "SR_B3"],
                ),
                CollectionSpec(
                    key="L8_SR",
                    ee_id="LANDSAT/LC08/C02/T1_L2",
                    cloud_property="CLOUD_COVER",
                    nominal_scale_m=30,
                    pretty_name="Landsat 8 SR (C2 L2)",
                    bands=["SR_B5", "SR_B4", "SR_B3"],
                ),
                CollectionSpec(
                    key="L7_SR",
                    ee_id="LANDSAT/LE07/C02/T1_L2",
                    cloud_property="CLOUD_COVER",
                    nominal_scale_m=30,
                    pretty_name="Landsat 7 SR (C2 L2)",
                    bands=["SR_B5", "SR_B4", "SR_B3"],
                ),
                CollectionSpec(
                    key="L5_SR",
                    ee_id="LANDSAT/LT05/C02/T1_L2",
                    cloud_property="CLOUD_COVER",
                    nominal_scale_m=30,
                    pretty_name="Landsat 5 SR (C2 L2)",
                    bands=["SR_B4", "SR_B3", "SR_B2"],
                ),
                CollectionSpec(
                    key="S2_SR",
                    ee_id="COPERNICUS/S2_SR_HARMONIZED",
                    cloud_property="CLOUDY_PIXEL_PERCENTAGE",
                    nominal_scale_m=10,
                    pretty_name="Sentinel‑2 SR (harmonized)",
                    bands=["B8", "B4", "B3", "B2"],
                ),
                # --- TOA (C2 L1) with Panchromatic defaults where available
                CollectionSpec(
                    key="L9_TOA",
                    ee_id="LANDSAT/LC09/C02/T1_TOA",
                    cloud_property="CLOUD_COVER",
                    nominal_scale_m=15,  # PAN resolution
                    pretty_name="Landsat 9 TOA (C2 L1)",
                    bands=["B8"],  # Panchromatic
                ),
                CollectionSpec(
                    key="L8_TOA",
                    ee_id="LANDSAT/LC08/C02/T1_TOA",
                    cloud_property="CLOUD_COVER",
                    nominal_scale_m=15,
                    pretty_name="Landsat 8 TOA (C2 L1)",
                    bands=["B8"],
                ),
                CollectionSpec(
                    key="L7_TOA",
                    ee_id="LANDSAT/LE07/C02/T1_TOA",
                    cloud_property="CLOUD_COVER",
                    nominal_scale_m=15,
                    pretty_name="Landsat 7 TOA (C2 L1)",
                    bands=["B8"],
                ),
            ]
        )

    def specs(self) -> List[CollectionSpec]:
        """List all collection specifications.

        Returns
        -------
        list[CollectionSpec]
        """
        return list(self._specs.values())

    def get(self, key: str) -> CollectionSpec:
        """Retrieve a collection specification by key.

        Parameters
        ----------
        key : str
            Collection key (see :meth:`keys`).

        Returns
        -------
        CollectionSpec

        Raises
        ------
        KeyError
            If the key is not present in the catalog.
        """
        return self._specs[key]

    def keys(self) -> List[str]:
        """Return all registered collection keys."""
        return list(self._specs.keys())

# -------------------------
# 3) Image search and metadata
# -------------------------
class ImageFinder:
    """Search image collections, build GeoDataFrame metadata & perform downloads.

    Parameters
    ----------
    client : GEEClient
        An initialized (or lazily connectable) Earth Engine client wrapper.
    catalog : ImageCollectionCatalog
        Registry of image collections to query.
    aoi : GeoDataFrame | shapely geometry | mapping | ee.Geometry
        Area of interest; can be provided in several formats. Re‑projected to WGS84 if a
        GeoDataFrame in another CRS. Dissolved union is used when multiple geometries exist.
    aoi_crs : str | int, optional
        Explicit CRS if *aoi* is provided as a non‑geospatial mapping that needs disambiguation.
    year_range : tuple[int, int], default (2020, 2025)
        Inclusive year range filter.
    month_range : tuple[int, int], default (1, 12)
        Inclusive month (1–12) range filter.
    cloud_max : int, default 30
        Maximum cloud percentage (using each collection's cloud property field).

    Notes
    -----
    The AOI export CRS is heuristically chosen from the supplied AOI: if projected, its CRS
    (EPSG form when available) is retained for exports; otherwise WGS84.
    """

    def __init__(
        self,
        client: GEEClient,
        catalog: ImageCollectionCatalog,
        *,
        aoi: Union[gpd.GeoDataFrame, _BaseGeometry, Mapping, ee.Geometry],
        aoi_crs: Optional[Union[str, int]] = None,
        year_range: Tuple[int, int] = (2020, 2025),
        month_range: Tuple[int, int] = (1, 12),
        cloud_max: int = 30,
    ) -> None:
        self._client = client
        self._catalog = catalog
        self.year_range = year_range
        self.month_range = month_range
        self.cloud_max = cloud_max
        self._aoi_export_crs: str = "EPSG:4326"
        if isinstance(aoi, gpd.GeoDataFrame) and (aoi.crs is not None):
            try:
                _c = _CRS.from_user_input(aoi.crs)
                if _c.is_projected:
                    epsg = _c.to_epsg()
                    self._aoi_export_crs = f"EPSG:{epsg}" if epsg else _c.to_string()
            except Exception:
                pass
        elif aoi_crs is not None:
            try:
                _c = _CRS.from_user_input(aoi_crs)
                if _c.is_projected:
                    epsg = _c.to_epsg()
                    self._aoi_export_crs = f"EPSG:{epsg}" if epsg else _c.to_string()
            except Exception:
                pass
        self.aoi_ee: ee.Geometry = self._coerce_to_ee_geometry(aoi)



    # ---- public API
    def search_metadata(self, collection_keys: Optional[Sequence[str]] = None) -> gpd.GeoDataFrame:
        """Search collections and build a GeoDataFrame of per‑image metadata.

        Parameters
        ----------
        collection_keys : sequence[str], optional
            Subset of catalog keys to search. ``None`` -> all in the catalog.

        Returns
        -------
        geopandas.GeoDataFrame
            Columns (dtypes may vary slightly):

            * ``image_id`` – Full Earth Engine image ID.
            * ``collection_key`` / ``collection_name`` – Source collection identifiers.
            * ``acquisition_date`` – Timestamp (UTC) normalized to pandas ``datetime64[ns]``.
            * ``cloud_percent`` – Cloud metric as reported by the collection (float).
            * ``nominal_scale_m`` – Nominal pixel size used for sampling / default exports.
            * ``footprint_area_m2`` / ``footprint_area_km2`` – Image footprint area.
            * ``aoi_coverage_percent`` – Percent of AOI covered by the image footprint.
            * ``overlap_ratio`` – (intersection_area / footprint_area) in [0, 1].
            * ``geometry`` – Shapely polygon (footprint) in EPSG:4326.

        Raises
        ------
        Exception
            Propagates any underlying Earth Engine API errors encountered while fetching
            metadata.
        """
        self._client.connect()  # explicit stage 1 (connect)

        selected = (
            [self._catalog.get(k) for k in collection_keys]
            if collection_keys
            else self._catalog.specs()
        )

        aoi_area_m2 = ee.Number(self.aoi_ee.area(1))
        feature_collections: List[ee.FeatureCollection] = []
        for spec in selected:
            ic = ee.ImageCollection(spec.ee_id)
            ic = (
                ic.filterBounds(self.aoi_ee)
                .filter(ee.Filter.lt(spec.cloud_property, self.cloud_max))
                .filter(ee.Filter.calendarRange(self.year_range[0], self.year_range[1], "year"))
                .filter(ee.Filter.calendarRange(self.month_range[0], self.month_range[1], "month"))
            )

            # Map to a FeatureCollection with server-side computed metrics
            def _to_feature(img: ee.Image) -> ee.Feature:
                footprint = img.geometry()
                inter = footprint.intersection(self.aoi_ee, 1)
                footprint_area_m2 = ee.Number(footprint.area(1))
                footprint_area_km2 = footprint_area_m2.divide(ee.Number(1e6))
                inter_area_m2 = ee.Number(inter.area(1))
                overlap_ratio = inter_area_m2.divide(footprint_area_m2)
                aoi_coverage_percent = inter_area_m2.divide(aoi_area_m2).multiply(ee.Number(100))
                props = {
                    "image_id": img.id(),
                    "collection_key": spec.key,
                    "collection_name": spec.pretty_name,
                    "acquisition_date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
                    "cloud_percent": ee.Number(img.get(spec.cloud_property)),
                    "nominal_scale_m": spec.nominal_scale_m,
                    "footprint_area_m2": footprint_area_m2,
                    "footprint_area_km2": footprint_area_km2,
                    # Keep intersection area only for filtering; not returned to the GeoDataFrame
                    "aoi_intersection_area_m2": inter_area_m2,
                    "aoi_coverage_percent": aoi_coverage_percent,
                    "overlap_ratio": overlap_ratio,
                }
                # Use the image footprint as the Feature geometry for plotting
                return ee.Feature(footprint, props)

            fc = ic.map(_to_feature).filter(ee.Filter.gt("aoi_intersection_area_m2", ee.Number(0)))
            feature_collections.append(fc)

        # Merge and bring client-side (single getInfo call per merged FC)
        if not feature_collections:
            return gpd.GeoDataFrame(columns=[
                "image_id","collection_key","collection_name","acquisition_date",
                "cloud_percent","nominal_scale_m","footprint_area_m2","footprint_area_km2",
                "aoi_coverage_percent","overlap_ratio","geometry"
            ], geometry="geometry", crs="EPSG:4326")

        merged_fc = ee.FeatureCollection(feature_collections[0])
        for fc in feature_collections[1:]:
            merged_fc = merged_fc.merge(fc)

        features = merged_fc.getInfo()["features"]

        # Build GeoDataFrame from GeoJSON-like features
        rows = []
        for f in features:
            geom = _shape(f["geometry"])  # shapely geometry (footprint)
            props = f["properties"]
            rows.append(
                {
                    "image_id": props["image_id"],
                    "collection_key": props["collection_key"],
                    "collection_name": props["collection_name"],
                    "acquisition_date": props["acquisition_date"],
                    "cloud_percent": float(props["cloud_percent"]) if props["cloud_percent"] is not None else None,
                    "nominal_scale_m": int(props["nominal_scale_m"]),
                    "footprint_area_m2": float(props["footprint_area_m2"]),
                    "footprint_area_km2": float(props["footprint_area_km2"]),
                    "aoi_coverage_percent": float(props["aoi_coverage_percent"]),
                    "overlap_ratio": float(props["overlap_ratio"]),
                    "geometry": geom,
                }
            )

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        # Sort by date (desc) then cloud (asc)
        gdf["acquisition_date"] = gdf["acquisition_date"].astype("datetime64[ns]")
        gdf = gdf.sort_values(["acquisition_date", "cloud_percent"], ascending=[False, True]).reset_index(drop=True)
        return gdf

    def bulk_download(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        output_dir: Union[str, Path],
        region_mode: str = "aoi",  # "aoi" or "footprint"
        file_basename: Optional[str] = None,
        crs: Optional[str] = None,
        scale_override_m: Optional[int] = None,
        bands_override: Optional[Sequence[str]] = None,
        max_images: Optional[int] = None,
        throttle_s: float = 0.0,
        max_retries: int = 2,
        retry_backoff_s: float = 2.0,
        show_progress: bool = True,
    ) -> List[Path]:
        """Download imagery referenced in a metadata GeoDataFrame.

        Parameters
        ----------
        images_gdf : geopandas.GeoDataFrame
            Output from :meth:`search_metadata` (or compatible). Must contain columns
            ``image_id``, ``collection_key``, ``nominal_scale_m`` and a geometry column.
        output_dir : str | Path
            Root directory to create (or reuse) collection subdirectories.
        region_mode : {"aoi", "footprint"}, default "aoi"
            Export region: full AOI (clip) vs each image's footprint polygon.
        file_basename : str, optional
            Base name prefix. If omitted an informative pattern including collection key,
            acquisition date and scale is used.
        crs : str, optional
            Target export CRS. Default chooses AOI CRS when projected, else EPSG:4326.
        scale_override_m : int, optional
            Force a single pixel size (meters) for all exports; otherwise per‑collection
            nominal scale is applied.
        bands_override : sequence[str], optional
            Explicit, global band selection. If omitted the default ``CollectionSpec.bands``
            (if present) are used per row.
        max_images : int, optional
            Limit number of rows exported (useful for tests / sampling).
        throttle_s : float, default 0
            Sleep seconds between successful exports (basic quota pacing).
        max_retries : int, default 2
            Maximum retry attempts for a failing export (transient errors only).
        retry_backoff_s : float, default 2.0
            Base seconds for exponential backoff (sleep = base * 2**attempt).
        show_progress : bool, default True
            Show a tqdm progress bar when ``tqdm`` is installed.

        Returns
        -------
        list[pathlib.Path]
            Paths to successfully written GeoTIFFs.

        Raises
        ------
        ValueError
            If required columns are missing from ``images_gdf``.
        RuntimeError
            If an image ultimately fails to export after retries.
        """
        self._client.connect()
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        exported_paths: List[Path] = []
        # Safeguard: ensure required columns exist
        required = {"image_id", "collection_key", "nominal_scale_m", "geometry"}
        missing = required - set(images_gdf.columns)
        if missing:
            raise ValueError(f"images_gdf is missing required columns: {sorted(missing)}")

        # Prepare AOI geometry once
        region_aoi = self.aoi_ee

        # Helper to pick bands from the catalog unless overridden
        def _bands_for(collection_key: str) -> Optional[Sequence[str]]:
            if bands_override is not None:
                return list(bands_override)
            try:
                return self._catalog.get(collection_key).bands
            except KeyError:
                return None

        # Iterate rows and export
        rows_iter = images_gdf.itertuples(index=False)
        if max_images is not None:
            rows_iter = list(rows_iter)[: int(max_images)]

        total_count = len(images_gdf) if max_images is None else min(len(images_gdf), int(max_images))
        iterator = _tqdm(rows_iter, total=total_count, desc="Downloading", unit="img") if (_HAS_TQDM and show_progress) else rows_iter

        for row in iterator:
            image_id_val: str = getattr(row, "image_id")
            collection_key: str = getattr(row, "collection_key")
            nominal_scale_m: int = int(getattr(row, "nominal_scale_m"))
            footprint_geom = getattr(row, "geometry")  # shapely

            # Reconstruct full ID if the row contains only the short scene ID
            if "/" in image_id_val:
                image_id_full = image_id_val
            else:
                try:
                    collection_ee_id = self._catalog.get(collection_key).ee_id
                except KeyError:
                    raise RuntimeError(
                        f"Cannot reconstruct full image id for collection '{collection_key}'."
                    )
                image_id_full = f"{collection_ee_id}/{image_id_val}"

            export_scale = int(scale_override_m or nominal_scale_m)
            region = region_aoi if region_mode == "aoi" else ee.Geometry(_mapping(footprint_geom))

            # Build output path: <root>/<collection_key>/<basename>_<date>_<shortid>.tif
            collection_dir = out_root / collection_key
            collection_dir.mkdir(parents=True, exist_ok=True)

            date_str = None
            if hasattr(row, "acquisition_date") and row.acquisition_date is not None:
                # row.acquisition_date may be numpy datetime64; normalize to date string
                try:
                    ts = row.acquisition_date
                    if hasattr(ts, "to_pydatetime"):
                        ts = ts.to_pydatetime()
                    if isinstance(ts, _dt.datetime):
                        date_str = ts.strftime("%Y%m%d")
                except Exception:
                    date_str = None
            short_id = image_id_full.split("/")[-1]
            base = file_basename or "img"
            out_path = collection_dir / f"{base}_{date_str or 'nodate'}_{short_id}.tif"
            out_path = collection_dir / (
                f"{collection_key}_{date_str or 'nodate'}_{int(export_scale)}m.tif"
            )

            img = ee.Image(image_id_full)
            bands = _bands_for(collection_key)
            if bands:
                img = img.select(list(bands))

            # geemap handles URL/download for us [3]
            attempt = 0
            while True:
                try:
                    LOGGER.info(
                        "export_start",
                        extra={
                            "image_id": image_id_full,
                            "collection": collection_key,
                            "scale": export_scale,
                            "region_mode": region_mode,
                            "out": str(out_path),
                        },
                    )
                    target_crs = crs or getattr(self, "_aoi_export_crs", "EPSG:4326")
                    geemap.ee_export_image(
                        img,
                        filename=str(out_path),
                        scale=export_scale,
                        crs=target_crs,
                        region=region,
                    )
                    LOGGER.info("export_done", extra={"out": str(out_path)})
                    exported_paths.append(out_path)
                    if throttle_s > 0:
                        _time.sleep(throttle_s)
                    break
                except Exception as e:
                    msg = str(e)
                    LOGGER.warning("export_fail", extra={"error": msg, "attempt": attempt})
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"Failed to export {image_id_full} (collection {collection_key}). Last error: {e}"
                        )
                    sleep_s = retry_backoff_s * (2 ** attempt)
                    _time.sleep(sleep_s)
                    attempt += 1

        return exported_paths

    # ---- helpers
    @staticmethod
    def _coerce_to_ee_geometry(
        aoi: Union[gpd.GeoDataFrame, _BaseGeometry, Mapping, ee.Geometry]
    ) -> ee.Geometry:
        """
        Coerce multiple AOI representations to an ``ee.Geometry``.

        Accepted input types:
        * An ``ee.Geometry`` (returned unchanged).
        * A :class:`geopandas.GeoDataFrame` – reprojected to EPSG:4326 if needed then
            dissolved into a single union.
        * A Shapely (Multi)Polygon / geometry.
        * A GeoJSON‑like mapping (``__geo_interface__`` style dict).

        Parameters
        ----------
        aoi : various (see above)
                AOI definition.

        Returns
        -------
        ee.Geometry

        Raises
        ------
        TypeError
                If the input type is not one of the supported forms.
        """
        if isinstance(aoi, ee.geometry.Geometry):
            return aoi
        if isinstance(aoi, gpd.GeoDataFrame):
            gdf = aoi
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            dissolved = gdf.unary_union  # robust union
            return ImageFinder._coerce_to_ee_geometry(dissolved)
        if isinstance(aoi, _BaseGeometry):
            return ee.Geometry(_mapping(aoi))
        if isinstance(aoi, Mapping):  # GeoJSON‑like
            return ee.Geometry(aoi)
        raise TypeError(
            "AOI must be a GeoDataFrame, Shapely geometry, GeoJSON‑like mapping, or ee.Geometry"
        )

# -------------------------
# Optional: Command Line Interface (CLI)
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search and download GEE imagery.")
    parser.add_argument("aoi_file", type=str, help="Path to AOI file (GeoPackage/GeoJSON/Shapefile)")
    parser.add_argument("out_dir", type=str, help="Output directory for downloads")
    parser.add_argument("--years", type=int, nargs=2, default=(2020, 2025), help="Year range inclusive, e.g. 2019 2021")
    parser.add_argument("--months", type=int, nargs=2, default=(1, 12), help="Month range inclusive, e.g. 6 9")
    parser.add_argument("--cloud", type=int, default=30, help="Max cloud percent")
    parser.add_argument("--collections", nargs="+", default=None, help="Collection keys to use (e.g., S2_SR L7_SR L8_TOA). If omitted, uses catalog defaults.")
    parser.add_argument("--region", choices=["aoi", "footprint"], default="aoi", help="Export region mode")
    parser.add_argument("--scale", type=int, default=None, help="Override export scale (meters)")
    parser.add_argument("--bands", nargs="+", default=None, help="Override bands for export (applies to all images)")
    parser.add_argument("--limit", type=int, default=None, help="Max images to download")
    parser.add_argument("--throttle", type=float, default=0.0, help="Seconds to sleep between downloads")
    parser.add_argument("--retries", type=int, default=2, help="Max retries per download")
    parser.add_argument("--backoff", type=float, default=2.0, help="Base seconds for exponential backoff")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    _logging.basicConfig(level=getattr(_logging, args.log_level.upper(), _logging.INFO))

    aoi_gdf = gpd.read_file(args.aoi_file)
    client = GEEClient().connect()
    catalog = ImageCollectionCatalog.default()

    # Validate collections if provided
    if args.collections is not None:
        unknown = sorted(set(args.collections) - set(catalog.keys()))
        if unknown:
            raise SystemExit(f"Unknown collection keys: {unknown}. Available: {catalog.keys()}")
        selected_keys = args.collections
    else:
        selected_keys = None

    finder = ImageFinder(
        client,
        catalog,
        aoi=aoi_gdf,
        year_range=tuple(args.years),
        month_range=tuple(args.months),
        cloud_max=int(args.cloud),
    )

    gdf = finder.search_metadata(collection_keys=selected_keys)
    print("Found images:", len(gdf))
    print(gdf.head())

    paths = finder.bulk_download(
        gdf,
        output_dir=args.out_dir,
        region_mode=args.region,
        scale_override_m=args.scale,
        bands_override=args.bands,
        max_images=args.limit,
        throttle_s=args.throttle,
        max_retries=args.retries,
        retry_backoff_s=args.backoff,
        show_progress=not args.no_progress,
    )
    print(f"Downloaded {len(paths)} image file(s) to {args.out_dir}")
