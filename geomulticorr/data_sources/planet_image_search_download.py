#!/usr/bin/env python
# coding=utf-8
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# planet_image_search_download.py
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
"""Search, preview and *quota-guarded* ordering of PlanetScope imagery.

Planet is a commercial provider: search and previews are free, but every
ordered km² counts against the account quota (e.g. 3 000 km² for Education &
Research plans). This module therefore splits the workflow into three
strictly separated stages:

1. **Search (free)** — :meth:`PlanetImageFinder.search_metadata` queries the
   Planet Data API and returns a GeoDataFrame including acquisition-geometry
   columns (``instrument``, ``signed_view_angle``, sun angles,
   ``clear_percent``); :meth:`PlanetImageFinder.assign_perspective_groups`
   clusters scenes with compatible viewing geometry (after
   UP-RS-ESP/PlanetScope_landslide_tracking, Mueting & Bookhagen) so only
   pairable scenes are worth ordering.
2. **Preview (free)** — :meth:`PlanetImageFinder.download_thumbnails` fetches
   quota-free PNG quicklooks for visual inspection;
   :func:`plot_thumbnail_grid` shows them as a contact sheet.
3. **Order (consumes quota)** — :meth:`PlanetImageFinder.build_order_request`
   assembles an Orders API request (clip to AOI, optional Sentinel-2
   harmonisation) and :meth:`PlanetImageFinder.submit_order` refuses to run
   without an explicit cost estimate below the remaining quota / user budget,
   keeps a local ledger of ordered scenes, and asks for confirmation.
   :meth:`PlanetImageFinder.wait_and_download_order` polls and downloads.

Authentication
--------------
A Planet API key is required (unavoidable — commercial catalog). Provide it
via the ``PL_API_KEY`` environment variable or the ``api_key`` argument.

Quota accounting caveat
-----------------------
With the Orders API ``clip`` tool, quota is charged on the *clipped* area —
but some contract models apply a **minimum charge per asset** (e.g. 100 km²)
regardless of clip size. Before any bulk order, submit ONE clipped test scene
and compare :meth:`PlanetClient.get_quota` before/after; then set
``min_charge_km2`` accordingly in the cost estimate.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import logging as _logging
import os as _os
import time as _time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import requests
from pyproj import Geod as _Geod
from shapely.geometry import mapping as _mapping
from shapely.geometry import shape as _shape
from shapely.geometry.base import BaseGeometry as _BaseGeometry

# Optional progress bar
try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

LOGGER = _logging.getLogger("planet_image_search_download")

PLANET_DATA_URL = "https://api.planet.com/data/v1"
PLANET_ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"
PLANET_TILES_URL = "https://tiles.planet.com/data/v1"
PLANET_QUOTA_URL = "https://api.planet.com/auth/v1/experimental/public/my/subscriptions"

_WGS84_GEOD = _Geod(ellps="WGS84")


def _geodesic_area_km2(geom: _BaseGeometry) -> float:
    """Return the geodesic area (km²) of a WGS84 shapely geometry."""
    area, _ = _WGS84_GEOD.geometry_area_perimeter(geom)
    return abs(area) / 1e6


# -------------------------
# 1) Client (auth + quota)
# -------------------------
class PlanetClient:
    """Thin synchronous REST client for the Planet APIs.

    Parameters
    ----------
    api_key : str, optional
        Planet API key. Falls back to the ``PL_API_KEY`` environment variable.

    Raises
    ------
    ValueError
        If no API key can be found.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or _os.environ.get("PL_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No Planet API key: pass api_key= or set the PL_API_KEY "
                "environment variable."
            )
        self.session = requests.Session()
        # Planet REST APIs use HTTP Basic auth with the key as username
        self.session.auth = (self.api_key, "")

    def get_quota(self) -> dict:
        """Return the account area quota in km².

        Returns
        -------
        dict
            ``{"total_km2", "used_km2", "remaining_km2", "subscriptions"}``
            where ``subscriptions`` is the raw per-subscription list. Sums all
            active subscriptions that expose ``quota_sqkm``.

        Raises
        ------
        requests.HTTPError
            If the (experimental) quota endpoint is unavailable for this
            account type.
        """
        resp = self.session.get(PLANET_QUOTA_URL, timeout=60)
        resp.raise_for_status()
        subs = resp.json()
        total = used = 0.0
        for sub in subs:
            if sub.get("quota_sqkm") is None:
                continue
            if sub.get("state", "active") != "active":
                continue
            total += float(sub["quota_sqkm"])
            used += float(sub.get("quota_used") or 0.0)
        return {
            "total_km2": total,
            "used_km2": used,
            "remaining_km2": total - used,
            "subscriptions": subs,
        }


# -------------------------
# 2) Search / preview / order
# -------------------------
class PlanetImageFinder:
    """Search the Planet catalog, preview scenes and place quota-guarded orders.

    Parameters
    ----------
    client : PlanetClient
        Authenticated Planet client.
    aoi : GeoDataFrame | shapely geometry | GeoJSON-like mapping
        Area of interest (re-projected to WGS84 and dissolved if needed).
    year_range : tuple[int, int], default (2020, 2025)
        Inclusive year range.
    month_range : tuple[int, int], default (1, 12)
        Inclusive month range (applied client-side, like the MPC module).
    cloud_max : float, default 30
        Maximum scene ``cloud_cover`` in percent.
    clear_min : float, optional
        Minimum ``clear_percent`` (UDM2-based usable fraction). Stricter and
        more reliable than cloud cover, but only populated for scenes with a
        UDM2 (roughly post-2018-08); older scenes are dropped when set.
    item_type : str, default "PSScene"
        Planet item type to search.
    instruments : sequence of str, optional
        Restrict to instrument generations (``"PSB.SD"``, ``"PS2.SD"``,
        ``"PS2"``). Never correlate across generations.
    max_view_angle : float, optional
        Maximum absolute view angle in degrees.
    """

    def __init__(
        self,
        client: PlanetClient,
        *,
        aoi: Union[gpd.GeoDataFrame, _BaseGeometry, Mapping],
        year_range: Tuple[int, int] = (2020, 2025),
        month_range: Tuple[int, int] = (1, 12),
        cloud_max: float = 30,
        clear_min: Optional[float] = None,
        item_type: str = "PSScene",
        instruments: Optional[Sequence[str]] = None,
        max_view_angle: Optional[float] = None,
    ) -> None:
        self._client = client
        self.aoi_geom = self._coerce_to_wgs84_geometry(aoi)
        self.aoi_area_km2 = _geodesic_area_km2(self.aoi_geom)
        self.year_range = year_range
        self.month_range = month_range
        self.cloud_max = cloud_max
        self.clear_min = clear_min
        self.item_type = item_type
        self.instruments = list(instruments) if instruments else None
        self.max_view_angle = max_view_angle

    # ---- stage 1: search (free)
    def build_search_filter(self) -> dict:
        """Return the Data API filter tree for the configured constraints."""
        y0, y1 = self.year_range
        clauses: List[dict] = [
            {
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": _mapping(self.aoi_geom),
            },
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": f"{y0}-01-01T00:00:00Z",
                    "lte": f"{y1}-12-31T23:59:59Z",
                },
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": self.cloud_max / 100.0},
            },
            {
                "type": "StringInFilter",
                "field_name": "quality_category",
                "config": ["standard"],
            },
            {
                "type": "StringInFilter",
                "field_name": "ground_control",
                "config": ["true"],
            },
        ]
        if self.clear_min is not None:
            clauses.append(
                {
                    "type": "RangeFilter",
                    "field_name": "clear_percent",
                    "config": {"gte": float(self.clear_min)},
                }
            )
        if self.instruments:
            clauses.append(
                {
                    "type": "StringInFilter",
                    "field_name": "instrument",
                    "config": list(self.instruments),
                }
            )
        if self.max_view_angle is not None:
            clauses.append(
                {
                    "type": "RangeFilter",
                    "field_name": "view_angle",
                    "config": {
                        "gte": -float(self.max_view_angle),
                        "lte": float(self.max_view_angle),
                    },
                }
            )
        return {"type": "AndFilter", "config": clauses}

    def search_metadata(self, page_size: int = 250) -> gpd.GeoDataFrame:
        """Search the Data API (quick-search, paginated) — costs no quota.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per scene: ``image_id``, ``acquisition_date``,
            ``acquired`` (full timestamp), ``instrument``, ``cloud_percent``,
            ``clear_percent``, ``view_angle``, ``signed_view_angle``,
            ``satellite_azimuth``, ``sun_azimuth``, ``sun_elevation``,
            ``quality_category``, ``publishing_stage``,
            ``aoi_coverage_percent``, ``aoi_intersection_km2`` (== the clipped
            area an order would request), ``overlap_ratio``, ``geometry``.
            Sorted by date (desc) then clear_percent (desc).
        """
        url = f"{PLANET_DATA_URL}/quick-search"
        body = {"item_types": [self.item_type], "filter": self.build_search_filter()}
        params = {"_page_size": page_size}

        features: List[dict] = []
        resp = self._client.session.post(url, json=body, params=params, timeout=120)
        resp.raise_for_status()
        page = resp.json()
        features.extend(page.get("features", []))
        while page.get("_links", {}).get("_next"):
            resp = self._client.session.get(page["_links"]["_next"], timeout=120)
            resp.raise_for_status()
            page = resp.json()
            features.extend(page.get("features", []))

        rows: List[dict] = []
        for feat in features:
            props = feat["properties"]
            acquired = _dt.datetime.fromisoformat(
                props["acquired"].replace("Z", "+00:00")
            )
            if not (self.month_range[0] <= acquired.month <= self.month_range[1]):
                continue
            footprint = _shape(feat["geometry"])
            inter = footprint.intersection(self.aoi_geom)
            if inter.is_empty:
                continue
            inter_km2 = _geodesic_area_km2(inter)
            footprint_km2 = _geodesic_area_km2(footprint)
            view_angle = props.get("view_angle")
            sat_az = props.get("satellite_azimuth")
            rows.append(
                {
                    "image_id": feat["id"],
                    "item_type": self.item_type,
                    "acquisition_date": acquired.strftime("%Y-%m-%d"),
                    "acquired": acquired,
                    "instrument": props.get("instrument"),
                    "cloud_percent": 100.0 * float(props.get("cloud_cover") or 0.0),
                    "clear_percent": props.get("clear_percent"),
                    "view_angle": view_angle,
                    "signed_view_angle": self._signed_view_angle(view_angle, sat_az),
                    "satellite_azimuth": sat_az,
                    "sun_azimuth": props.get("sun_azimuth"),
                    "sun_elevation": props.get("sun_elevation"),
                    "quality_category": props.get("quality_category"),
                    "publishing_stage": props.get("publishing_stage"),
                    "footprint_area_km2": footprint_km2,
                    "aoi_intersection_km2": inter_km2,
                    "aoi_coverage_percent": 100.0 * inter_km2 / self.aoi_area_km2,
                    "overlap_ratio": (
                        inter_km2 / footprint_km2 if footprint_km2 else np.nan
                    ),
                    "geometry": footprint,
                }
            )

        if not rows:
            return gpd.GeoDataFrame(
                columns=[
                    "image_id",
                    "item_type",
                    "acquisition_date",
                    "acquired",
                    "instrument",
                    "cloud_percent",
                    "clear_percent",
                    "view_angle",
                    "signed_view_angle",
                    "satellite_azimuth",
                    "sun_azimuth",
                    "sun_elevation",
                    "quality_category",
                    "publishing_stage",
                    "footprint_area_km2",
                    "aoi_intersection_km2",
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
            ["acquisition_date", "clear_percent"], ascending=[False, False]
        ).reset_index(drop=True)
        return gdf

    @staticmethod
    def _signed_view_angle(
        view_angle: Optional[float], satellite_azimuth: Optional[float]
    ) -> Optional[float]:
        """Signed (left/right-looking) view angle.

        Heuristic from UP-RS-ESP/PlanetScope_landslide_tracking: with an
        assumed ~10° orbit-heading reference, a satellite azimuth in
        [10°, 190°) means the scene is imaged from one side of the ground
        track, otherwise from the other. Sign convention: positive for
        azimuth in [10°, 190°).
        """
        if view_angle is None or satellite_azimuth is None:
            return None
        sign = 1.0 if 10.0 <= (satellite_azimuth % 360.0) < 190.0 else -1.0
        return sign * abs(float(view_angle))

    @staticmethod
    def assign_perspective_groups(
        gdf: gpd.GeoDataFrame,
        va_diff_thresh: float = 0.5,
    ) -> gpd.GeoDataFrame:
        """Cluster scenes by compatible acquisition geometry.

        Within each instrument generation, scenes are sorted by
        ``signed_view_angle`` and split wherever the gap between consecutive
        angles exceeds ``va_diff_thresh`` (degrees) — the strategy used by
        UP-RS-ESP/PlanetScope_landslide_tracking (Mueting & Bookhagen) to
        keep orthorectification-induced offsets out of correlation pairs.

        Returns a copy of *gdf* with two new columns: ``perspective_group``
        (label ``"<instrument>#g<N>"``, or ``None`` when the angle is
        missing) and ``group_size``. Only pair scenes within one group.
        """
        gdf = gdf.copy()
        gdf["perspective_group"] = None
        valid = gdf["signed_view_angle"].notna()
        for instrument, sub in gdf[valid].groupby("instrument"):
            sub = sub.sort_values("signed_view_angle")
            angles = sub["signed_view_angle"].to_numpy(dtype=float)
            group_no = np.concatenate(
                [[0], np.cumsum(np.diff(angles) > va_diff_thresh)]
            )
            labels = [f"{instrument}#g{int(n) + 1}" for n in group_no]
            gdf.loc[sub.index, "perspective_group"] = labels
        counts = gdf["perspective_group"].value_counts()
        gdf["group_size"] = gdf["perspective_group"].map(counts).fillna(0).astype(int)
        return gdf

    # ---- stage 2: preview (free)
    def download_thumbnails(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        output_dir: Union[str, Path],
        width: int = 512,
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> List[Path]:
        """Download quota-free PNG quicklooks for every scene in *images_gdf*.

        Parameters
        ----------
        images_gdf : geopandas.GeoDataFrame
            Output of :meth:`search_metadata` (possibly filtered).
        output_dir : str | Path
            Directory for the PNG files (``<date>_<image_id>.png``).
        width : int, default 512
            Thumbnail width in pixels (≤ 512 for all accounts; up to 2048
            with download permission on the visual asset).
        overwrite : bool, default False
            Re-download existing thumbnails.

        Returns
        -------
        list[pathlib.Path]
            Paths of the thumbnails on disk (including pre-existing ones), in
            *images_gdf* row order — index-aligned for grid plotting.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = list(images_gdf.itertuples(index=False))
        iterator = (
            _tqdm(rows, desc="Thumbnails", unit="img")
            if (_HAS_TQDM and show_progress)
            else rows
        )
        paths: List[Path] = []
        for row in iterator:
            image_id = getattr(row, "image_id")
            item_type = getattr(row, "item_type", self.item_type)
            date = getattr(row, "acquisition_date")
            date_str = (
                date.strftime("%Y-%m-%d")
                if hasattr(date, "strftime")
                else str(date)[:10]
            )
            out_path = out_dir / f"{date_str}_{image_id}.png"
            if not out_path.exists() or overwrite:
                url = (
                    f"{PLANET_TILES_URL}/item-types/{item_type}/items/{image_id}/thumb"
                )
                resp = self._client.session.get(
                    url, params={"width": width}, timeout=60
                )
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
            paths.append(out_path)
        return paths

    # ---- stage 3: order (consumes quota!)
    def estimate_order_cost_km2(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        clip: bool = True,
        min_charge_km2: float = 0.0,
    ) -> float:
        """Estimate the quota cost (km²) of ordering the scenes in *images_gdf*.

        With ``clip=True`` each scene costs its AOI-intersection area (that is
        exactly what the clip tool delivers); otherwise its full footprint.
        ``min_charge_km2`` models contracts with a per-asset minimum charge —
        measure it once with a single-scene test order before bulk runs.
        """
        if clip:
            per_scene = images_gdf["aoi_intersection_km2"].astype(float)
        else:
            per_scene = images_gdf["footprint_area_km2"].astype(float)
        return float(np.maximum(per_scene, float(min_charge_km2)).sum())

    def build_order_request(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        order_name: str,
        product_bundle: str = "analytic_sr_udm2",
        clip: bool = True,
        harmonize_to_sentinel2: bool = False,
        archive_zip: bool = True,
    ) -> dict:
        """Assemble (but do NOT submit) an Orders API request body.

        Parameters
        ----------
        images_gdf : geopandas.GeoDataFrame
            Scenes to order (filtered output of :meth:`search_metadata`).
        order_name : str
            Human-readable order name (also used in the local ledger).
        product_bundle : str, default "analytic_sr_udm2"
            Product bundle: surface reflectance + UDM2 usable-data mask.
        clip : bool, default True
            Clip every scene to the AOI (quota charged on clipped area).
        harmonize_to_sentinel2 : bool, default False
            Apply Planet's radiometric harmonisation to the Sentinel-2
            response (recommended when mixing instrument generations).
        archive_zip : bool, default True
            Deliver everything as a single zip archive.
        """
        item_ids = images_gdf["image_id"].tolist()
        if not item_ids:
            raise ValueError("images_gdf contains no scenes")
        request: dict = {
            "name": order_name,
            "source_type": "scenes",
            "products": [
                {
                    "item_ids": item_ids,
                    "item_type": self.item_type,
                    "product_bundle": product_bundle,
                }
            ],
            "tools": [],
        }
        if clip:
            request["tools"].append({"clip": {"aoi": _mapping(self.aoi_geom)}})
        if harmonize_to_sentinel2:
            request["tools"].append({"harmonize": {"target_sensor": "Sentinel-2"}})
        if not request["tools"]:
            del request["tools"]
        if archive_zip:
            request["delivery"] = {"archive_type": "zip", "single_archive": True}
        return request

    def submit_order(
        self,
        order_request: dict,
        *,
        ledger_path: Union[str, Path],
        quota_budget_km2: Optional[float] = None,
        min_charge_km2: float = 0.0,
        images_gdf: Optional[gpd.GeoDataFrame] = None,
        require_confirmation: bool = True,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Submit an order — guarded by a cost estimate, the remaining quota,
        an optional budget, a duplicate-scene check and a confirmation prompt.

        Parameters
        ----------
        order_request : dict
            Body from :meth:`build_order_request`.
        ledger_path : str | Path
            JSON file recording every submitted order (created if missing).
            Scenes already present in the ledger trigger a warning — you may
            be paying for them twice.
        quota_budget_km2 : float, optional
            Hard cap for THIS order; refuse if the estimate exceeds it.
        min_charge_km2 : float, default 0
            Per-asset minimum charge for the cost estimate (see module notes).
        images_gdf : geopandas.GeoDataFrame, optional
            The same scenes passed to :meth:`build_order_request`; enables the
            precise per-scene area estimate. Without it the estimate falls
            back to ``n_scenes × AOI area``.
        require_confirmation : bool, default True
            Ask for interactive ``yes`` before submitting.
        dry_run : bool, default False
            Print the summary and return ``None`` without submitting.

        Returns
        -------
        str or None
            The order id, or ``None`` for a dry run / declined confirmation.

        Raises
        ------
        RuntimeError
            If the estimated cost exceeds the remaining quota or the budget.
        """
        item_ids = order_request["products"][0]["item_ids"]
        clip = any("clip" in t for t in order_request.get("tools", []))

        if images_gdf is not None:
            est_km2 = self.estimate_order_cost_km2(
                images_gdf, clip=clip, min_charge_km2=min_charge_km2
            )
        else:
            per_scene = self.aoi_area_km2 if clip else float("nan")
            est_km2 = len(item_ids) * max(per_scene, min_charge_km2)

        try:
            quota = self._client.get_quota()
            remaining = quota["remaining_km2"]
        except Exception as e:  # experimental endpoint may be unavailable
            LOGGER.warning("could not read quota: %s", e)
            quota, remaining = None, None

        ledger_path = Path(ledger_path)
        ledger = _json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        already = {i for entry in ledger for i in entry.get("item_ids", [])}
        duplicates = sorted(set(item_ids) & already)

        print(f"Order name        : {order_request.get('name')}")
        print(f"Scenes            : {len(item_ids)}  (clip={clip})")
        print(
            f"Estimated cost    : {est_km2:.1f} km2"
            + (f"  (min charge {min_charge_km2} km2/asset)" if min_charge_km2 else "")
        )
        if remaining is not None:
            print(
                f"Quota remaining   : {remaining:.1f} km2 "
                f"({quota['used_km2']:.1f}/{quota['total_km2']:.1f} used)"
            )
        if quota_budget_km2 is not None:
            print(f"Budget for order  : {quota_budget_km2:.1f} km2")
        if duplicates:
            print(
                f"WARNING: {len(duplicates)} scene(s) already in the ledger "
                f"(possible double charge): {duplicates[:5]}"
                + (" ..." if len(duplicates) > 5 else "")
            )

        if remaining is not None and est_km2 > remaining:
            raise RuntimeError(
                f"Estimated cost {est_km2:.1f} km2 exceeds remaining quota "
                f"{remaining:.1f} km2 — order refused."
            )
        if quota_budget_km2 is not None and est_km2 > quota_budget_km2:
            raise RuntimeError(
                f"Estimated cost {est_km2:.1f} km2 exceeds the budget "
                f"{quota_budget_km2:.1f} km2 — order refused."
            )
        if dry_run:
            print("Dry run — order NOT submitted.")
            return None
        if require_confirmation:
            answer = input(f"Submit this order ({est_km2:.1f} km2)? [yes/no] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Declined — order NOT submitted.")
                return None

        resp = self._client.session.post(
            PLANET_ORDERS_URL, json=order_request, timeout=120
        )
        resp.raise_for_status()
        order = resp.json()
        order_id = order["id"]

        ledger.append(
            {
                "order_id": order_id,
                "order_name": order_request.get("name"),
                "submitted_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "item_ids": item_ids,
                "estimated_km2": round(est_km2, 2),
                "clip": clip,
            }
        )
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(_json.dumps(ledger, indent=2))
        print(f"Order submitted: {order_id} (state: {order.get('state')})")
        return order_id

    def wait_and_download_order(
        self,
        order_id: str,
        *,
        output_dir: Union[str, Path],
        poll_interval_s: float = 30.0,
        timeout_s: float = 7200.0,
        overwrite: bool = False,
    ) -> List[Path]:
        """Poll an order until it succeeds, then download all result files.

        Orders are asynchronous on Planet's side (typically minutes). State
        ``partial`` is treated as success for the delivered files; ``failed``
        and ``cancelled`` raise.

        Returns
        -------
        list[pathlib.Path]
            Downloaded files (the zip archive, or per-scene files).
        """
        url = f"{PLANET_ORDERS_URL}/{order_id}"
        deadline = _time.monotonic() + timeout_s
        while True:
            resp = self._client.session.get(url, timeout=60)
            resp.raise_for_status()
            order = resp.json()
            state = order["state"]
            if state in ("success", "partial"):
                break
            if state in ("failed", "cancelled"):
                raise RuntimeError(f"Order {order_id} ended in state {state!r}")
            if _time.monotonic() > deadline:
                raise TimeoutError(
                    f"Order {order_id} still {state!r} after {timeout_s:.0f}s"
                )
            LOGGER.info(
                "order %s state=%s — waiting %.0fs", order_id, state, poll_interval_s
            )
            _time.sleep(poll_interval_s)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = order.get("_links", {}).get("results", [])
        written: List[Path] = []
        iterator = (
            _tqdm(results, desc="Order files", unit="file") if _HAS_TQDM else results
        )
        for result in iterator:
            name = Path(result["name"]).name
            out_path = out_dir / name
            if out_path.exists() and not overwrite:
                written.append(out_path)
                continue
            with self._client.session.get(
                result["location"], stream=True, timeout=300
            ) as r:
                r.raise_for_status()
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        f.write(chunk)
                tmp.rename(out_path)
            written.append(out_path)
        return written

    # ---- helpers
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


def plot_thumbnail_grid(
    thumb_paths: Sequence[Union[str, Path]],
    images_gdf: Optional[gpd.GeoDataFrame] = None,
    *,
    ncols: int = 4,
    figsize_per_cell: float = 3.0,
):
    """Show downloaded thumbnails as a contact sheet for visual selection.

    Parameters
    ----------
    thumb_paths : sequence of paths
        Output of :meth:`PlanetImageFinder.download_thumbnails` (row-aligned
        with *images_gdf* when given).
    images_gdf : geopandas.GeoDataFrame, optional
        Metadata used for the per-image caption (date, instrument,
        clear percent, signed view angle).
    ncols : int, default 4
        Grid columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = len(thumb_paths)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_cell * ncols, figsize_per_cell * nrows)
    )
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(mpimg.imread(thumb_paths[i]))
        if images_gdf is not None and i < len(images_gdf):
            row = images_gdf.iloc[i]
            date = row["acquisition_date"]
            date_str = (
                date.strftime("%Y-%m-%d")
                if hasattr(date, "strftime")
                else str(date)[:10]
            )
            clear = row.get("clear_percent")
            sva = row.get("signed_view_angle")
            caption = f"{date_str} {row.get('instrument', '')}"
            if clear is not None and not (isinstance(clear, float) and np.isnan(clear)):
                caption += f"\nclear {clear:.0f}%"
            if sva is not None and not (isinstance(sva, float) and np.isnan(sva)):
                caption += f" | va {sva:+.1f}\N{DEGREE SIGN}"
            ax.set_title(caption, fontsize=8)
    fig.tight_layout()
    return fig
