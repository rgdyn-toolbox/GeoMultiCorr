#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
   :func:`plot_thumbnail_grid` shows them as a contact sheet, and
   :func:`georeference_thumbnails` turns them into GeoTIFFs from the scene
   footprints (in parallel, see *Thumbnail georeferencing* below).
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

Thumbnail georeferencing
------------------------
A thumbnail from the tiles API is the scene footprint's **EPSG:3857 bounding
box**, drawn north-up with square pixels and letterboxed into the
``width × width`` frame: the long side fills the frame and the short side is
centred, with transparent padding. The geotransform therefore follows exactly
from the footprint and the PNG size (:func:`thumbnail_transform`), with no
corner detection. The alpha channel is only used to check the result: the IoU
of the opaque pixels against the rasterised footprint is ~0.98 on real
thumbnails (edge anti-aliasing), and a low value is reported as a warning.
"""

from __future__ import annotations

import concurrent.futures as _futures
import datetime as _dt
import json as _json
import logging as _logging
import os as _os
import time as _time
import warnings as _warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely as _shapely
from affine import Affine as _Affine
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
PLANET_PLANS_URL = "https://api.planet.com/auth/v1/experimental/public/plans"
PLANET_PRODUCTS_URL = "https://api.planet.com/account/v1/my/products"

_GLOBAL_AOI_KM2 = 5.0e8  # union_aoi at or above this ≈ the whole Earth (5.1e8 km²)

_WGS84_GEOD = _Geod(ellps="WGS84")

THUMBNAIL_FREE_WIDTH = 512
"""Widest thumbnail any account may fetch; wider needs visual-asset download rights."""
VISUAL_DOWNLOAD_PERMISSIONS = ("assets.ortho_visual:download", "assets.visual:download")
"""``_permissions`` entries that allow thumbnails wider than 512 px."""

FOOTPRINTS_FILENAME = "footprints.gpkg"
"""Search table written to the finder's ``workdir`` by :meth:`search_metadata`."""
THUMBNAILS_DIRNAME = "thumbnails"
"""Sub-folder of ``workdir`` holding the ``<image_id>.png`` quicklooks."""


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

    def _get_json(self, url: str):
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def info(self, verbose: bool = True) -> dict:
        """Account summary: organisation, plan(s), quota and download rights.

        Reads the subscriptions endpoint (as :meth:`get_quota`), the plan of
        each subscription (its *policies*: item types, assets, area and time
        of interest) and the documented products endpoint. These ``auth/v1``
        endpoints are **experimental** and undocumented, so any failing call
        is recorded under ``errors`` instead of raising.

        The policy field to look at is ``delay_days``: Education & Research
        plans delay PlanetScope download rights by 30 days. Scenes younger
        than that have no download permission, so their thumbnails are capped
        at 512 px (see :meth:`PlanetImageFinder.download_thumbnails`).

        Parameters
        ----------
        verbose : bool, default True
            Print a short human-readable summary (never the API key).

        Returns
        -------
        dict
            ``organization``, ``subscriptions`` (one summary per
            subscription: plan, validity, ``global_aoi``, ``quota_km2``,
            ``tile_quota``, ``policies``), ``products``, ``raw`` (the
            untouched responses) and ``errors`` (``{section: message}``).
        """
        out: dict = {
            "organization": None,
            "subscriptions": [],
            "products": [],
            "raw": {"subscriptions": None, "plans": {}, "products": None},
            "errors": {},
        }
        try:
            subs = self._get_json(PLANET_QUOTA_URL)
            out["raw"]["subscriptions"] = subs
        except Exception as exc:
            out["errors"]["subscriptions"] = f"{type(exc).__name__}: {exc}"
            subs = []
        for sub in subs or []:
            plan = sub.get("plan") or {}
            plan_id = sub.get("plan_id") or plan.get("id")
            meta = plan.get("meta") or {}
            union_aoi = meta.get("union_aoi")
            total, used = sub.get("quota_sqkm"), sub.get("quota_used")
            summary = {
                "id": sub.get("id"),
                "state": sub.get("state"),
                "plan_name": plan.get("name"),
                "plan_id": plan_id,
                "active_from": sub.get("active_from"),
                "active_to": sub.get("active_to"),
                "item_types": meta.get("item_types"),
                "global_aoi": (
                    bool(union_aoi >= _GLOBAL_AOI_KM2)
                    if union_aoi is not None
                    else None
                ),
                "quota_km2": {
                    "total": total,
                    "used": used,
                    "remaining": (
                        float(total) - float(used or 0.0) if total is not None else None
                    ),
                    "interval": sub.get("quota_interval"),
                    "anchor_date": sub.get("quota_anchor_date"),
                    "reset_at": sub.get("quota_reset_at"),
                    "overage": sub.get("quota_enable_overage"),
                },
                "tile_quota": {
                    kind: {
                        k: (sub.get(f"{kind}_tile_quota") or {}).get(k)
                        for k in ("current", "limit", "reset")
                    }
                    for kind in ("scene", "basemap")
                    if sub.get(f"{kind}_tile_quota")
                },
                "policies": [],
            }
            out["organization"] = out["organization"] or sub.get("organization")
            if plan_id is not None:
                try:
                    plan_full = self._get_json(f"{PLANET_PLANS_URL}/{plan_id}")
                    out["raw"]["plans"][plan_id] = plan_full
                    summary["policies"] = [
                        self._summarise_policy(p)
                        for p in plan_full.get("policies") or []
                    ]
                except Exception as exc:
                    out["errors"][f"plan:{plan_id}"] = f"{type(exc).__name__}: {exc}"
            out["subscriptions"].append(summary)
        try:
            products = self._get_json(PLANET_PRODUCTS_URL)
            out["raw"]["products"] = products
            out["products"] = (
                products.get("results", []) if isinstance(products, dict) else []
            )
        except Exception as exc:
            out["errors"]["products"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(self._format_info(out))
        return out

    @staticmethod
    def _summarise_policy(policy: dict) -> dict:
        """One plan policy → the fields that decide what can be downloaded."""
        cons = {c.get("key"): c.get("value") for c in policy.get("constraints") or []}
        type_map = (cons.get("type_permissions") or {}).get("map") or {}
        assets = {a for lst in type_map.values() for a in (lst or [])}
        toi = cons.get("toi") or {}
        flags = cons.get("flags") or {}
        return {
            "name": policy.get("name"),
            "item_types": sorted(type_map),
            "assets_count": len(assets),
            "visual_download": bool({"ortho_visual", "visual"} & assets),
            "delay_days": toi.get("delay"),
            "toi_ranges": toi.get("ranges"),
            "geotiff_download": flags.get("geotiff_download"),
            "webtile_service": flags.get("webtile_service"),
        }

    @staticmethod
    def _format_info(info: dict) -> str:
        """Human summary of :meth:`info`."""
        lines = []
        org = info.get("organization") or {}
        if org:
            lines.append(f"Organisation: {org.get('name')} (id {org.get('id')})")
        for sub in info.get("subscriptions", []):
            q = sub["quota_km2"]
            lines.append(
                f"Plan: {sub['plan_name']} [{sub['state']}], "
                f"{str(sub['active_from'])[:10]} → {str(sub['active_to'])[:10]}"
            )
            if q["total"] is not None:
                lines.append(
                    f"  Quota: {q['used']:.0f} / {q['total']:.0f} km² used "
                    f"({q['interval'] or 'no'} interval), remaining {q['remaining']:.0f} km²"
                )
            for kind, t in sub["tile_quota"].items():
                lines.append(
                    f"  {kind.capitalize()} tiles: {t['current']} / {t['limit']}"
                    f" (reset {str(t['reset'])[:10]})"
                )
            if sub["global_aoi"] is not None:
                lines.append(
                    "  Area of access: "
                    + ("global" if sub["global_aoi"] else "restricted")
                )
            for p in sub["policies"]:
                types = ", ".join(p["item_types"]) or "-"
                text = f"  Policy {p['name']} ({types}): {p['assets_count']} assets"
                delay = p["delay_days"]
                if delay:
                    text += (
                        f"; download delayed {delay} days → thumbnails > "
                        f"{THUMBNAIL_FREE_WIDTH} px only for scenes older than "
                        f"{delay} days"
                    )
                if p["geotiff_download"] is False:
                    text += "; geotiff_download off"
                lines.append(text)
        lines.append(f"Products: {len(info.get('products') or [])}")
        for section, msg in info.get("errors", {}).items():
            lines.append(f"  ! {section}: {msg}")
        return "\n".join(lines)

    def item_permissions(self, item_id: str, item_type: str = "PSScene") -> dict:
        """Download permissions of one scene (Data API item ``_permissions``).

        Returns ``image_id``, ``acquired``, ``publishing_stage``,
        ``permissions`` (the raw list; empty when the plan grants nothing on
        this scene, e.g. inside the download delay) and ``visual_download``
        (thumbnails wider than 512 px allowed).
        """
        item = self._get_json(
            f"{PLANET_DATA_URL}/item-types/{item_type}/items/{item_id}"
        )
        perms = list(item.get("_permissions") or [])
        props = item.get("properties") or {}
        return {
            "image_id": item.get("id", item_id),
            "acquired": props.get("acquired"),
            "publishing_stage": props.get("publishing_stage"),
            "permissions": perms,
            "visual_download": any(p in perms for p in VISUAL_DOWNLOAD_PERMISSIONS),
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
    workdir : str | Path, optional
        Folder for this AOI's search products. When set,
        :meth:`search_metadata` saves the search table there by default
        (``footprints.gpkg``). Quicklooks are *not* fetched by the search
        unless asked for: download them once the table is cleaned, with
        :meth:`download_thumbnails`, which defaults to ``thumbnails/``
        (``<image_id>.png``); :meth:`georeference_thumbnails` defaults to
        the same places.
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
        workdir: Optional[Union[str, Path]] = None,
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
        self.workdir = Path(workdir) if workdir is not None else None

    @property
    def footprints_path(self) -> Optional[Path]:
        """``<workdir>/footprints.gpkg``, or ``None`` without a workdir."""
        return self.workdir / FOOTPRINTS_FILENAME if self.workdir else None

    @property
    def thumbnail_dir(self) -> Optional[Path]:
        """``<workdir>/thumbnails``, or ``None`` without a workdir."""
        return self.workdir / THUMBNAILS_DIRNAME if self.workdir else None

    def _require_workdir(self, what: str) -> Path:
        if self.workdir is None:
            raise ValueError(
                f"{what}: no workdir — pass PlanetImageFinder(..., workdir=...) "
                "or give the path explicitly."
            )
        return self.workdir

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

    def search_metadata(
        self,
        page_size: int = 250,
        *,
        save: Optional[bool] = None,
        download_thumbnails: bool = False,
        thumbnail_width: int = 512,
        overwrite_thumbnails: bool = False,
        show_progress: bool = True,
    ) -> gpd.GeoDataFrame:
        """Search the Data API (quick-search, paginated) — costs no quota.

        Parameters
        ----------
        page_size : int, default 250
            Results per API page.
        save : bool, optional
            Write the search table to ``<workdir>/footprints.gpkg``
            (rewritten on each search, layer ``footprints``). Defaults to
            ``True`` when the finder has a ``workdir``, ``False`` otherwise;
            ``True`` without a workdir raises.
        download_thumbnails : bool, default False
            Also download **every** found scene's quicklook to
            ``<workdir>/thumbnails/<image_id>.png`` (free, but one request per
            scene). Off by default: clean the table first, then call
            :meth:`download_thumbnails` on the scenes you keep. Needs a
            workdir; independent of *save*. A failed download is logged, not
            raised — the search result is still returned.
        thumbnail_width, overwrite_thumbnails, show_progress
            Passed to :meth:`download_thumbnails` (only with
            ``download_thumbnails=True``).

        Returns
        -------
        geopandas.GeoDataFrame
            One row per scene: ``image_id``, ``acquisition_date``,
            ``acquired`` (full timestamp), ``instrument``, ``cloud_percent``,
            ``clear_percent``, ``view_angle``, ``signed_view_angle``,
            ``satellite_azimuth``, ``sun_azimuth``, ``sun_elevation``,
            ``quality_category``, ``publishing_stage``, ``visual_download``
            (the account may download the visual asset, which thumbnails
            wider than 512 px require),
            ``aoi_coverage_percent``, ``aoi_intersection_km2`` (== the clipped
            area an order would request), ``overlap_ratio``, ``geometry``.
            Sorted by date (desc) then clear_percent (desc).
        """
        if save is None:
            save = self.workdir is not None
        elif save:
            self._require_workdir("search_metadata(save=True)")
        if download_thumbnails:
            self._require_workdir("search_metadata(download_thumbnails=True)")
        gdf = self._search(page_size)
        if gdf.empty:
            if save or download_thumbnails:
                LOGGER.info(
                    "Search returned no scene; nothing saved in %s", self.workdir
                )
            return gdf
        if save:
            self._save_footprints(gdf)
        if download_thumbnails:
            try:
                self.download_thumbnails(
                    gdf,
                    width=thumbnail_width,
                    overwrite=overwrite_thumbnails,
                    show_progress=show_progress,
                )
            except Exception as exc:  # the search itself succeeded: keep it
                LOGGER.warning(
                    "Thumbnail download stopped (%s: %s); resume with "
                    "finder.download_thumbnails(gdf) — existing PNGs are skipped.",
                    type(exc).__name__,
                    exc,
                )
        return gdf

    def _save_footprints(self, gdf: gpd.GeoDataFrame) -> None:
        """Write the search table to ``<workdir>/footprints.gpkg`` (atomic)."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        # GPKG needs its extension, hence ".part.gpkg"; replace() is atomic
        tmp = self.footprints_path.with_name(FOOTPRINTS_FILENAME + ".part.gpkg")
        try:
            gdf.to_file(tmp, layer="footprints", driver="GPKG")
            tmp.replace(self.footprints_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        LOGGER.info("Saved %d footprints to %s", len(gdf), self.footprints_path)

    def _search(self, page_size: int) -> gpd.GeoDataFrame:
        """The quick-search itself (no side effects)."""
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
                    "visual_download": any(
                        p in (feat.get("_permissions") or [])
                        for p in VISUAL_DOWNLOAD_PERMISSIONS
                    ),
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
                    "visual_download",
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
        output_dir: Optional[Union[str, Path]] = None,
        width: int = 512,
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> List[Path]:
        """Download quota-free PNG quicklooks for every scene in *images_gdf*.

        Parameters
        ----------
        images_gdf : geopandas.GeoDataFrame
            Output of :meth:`search_metadata` (possibly filtered).
        output_dir : str | Path, optional
            Directory for the PNG files (``<image_id>.png``, e.g.
            ``20261103_050306_43_255b.png``; the id already carries the date).
            Defaults to ``<workdir>/thumbnails``.
        width : int, default 512
            Thumbnail width in pixels. Up to 512 works for any scene; up to
            2048 only for scenes whose visual asset the account may download
            (``visual_download`` column of :meth:`search_metadata`). This
            varies per scene — e.g. a scene still in preview, or outside the
            plan's area of access. Such scenes are fetched at 512 instead
            (directly when ``visual_download`` is False, else after Planet
            answers 400/403); their ids are kept in
            ``self.last_thumbnail_fallbacks`` and one warning lists them.
        overwrite : bool, default False
            Re-download existing thumbnails.

        Returns
        -------
        list[pathlib.Path]
            Paths of the thumbnails on disk (including pre-existing ones), in
            *images_gdf* row order — index-aligned for grid plotting.
        """
        if output_dir is None:
            self._require_workdir("download_thumbnails")
            output_dir = self.thumbnail_dir
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = list(images_gdf.itertuples(index=False))
        iterator = (
            _tqdm(rows, desc="Thumbnails", unit="img")
            if (_HAS_TQDM and show_progress)
            else rows
        )
        paths: List[Path] = []
        fallbacks: List[str] = []
        wide = width > THUMBNAIL_FREE_WIDTH
        for row in iterator:
            image_id = getattr(row, "image_id")
            item_type = getattr(row, "item_type", self.item_type)
            # The PSScene id already starts with the acquisition date (YYYYMMDD_…),
            # so the file is named after the id alone: that is also the key
            # georeference_thumbnails() matches against the footprints.
            out_path = out_dir / f"{image_id}.png"
            if not out_path.exists() or overwrite:
                url = (
                    f"{PLANET_TILES_URL}/item-types/{item_type}/items/{image_id}/thumb"
                )
                permitted = getattr(row, "visual_download", None)
                known_refused = (
                    wide
                    and permitted is not None
                    and not pd.isna(permitted)
                    and not bool(permitted)
                )
                resp = None
                if not known_refused:
                    resp = self._client.session.get(
                        url, params={"width": width}, timeout=60
                    )
                if known_refused or (
                    wide and getattr(resp, "status_code", 200) in (400, 403)
                ):
                    fallbacks.append(str(image_id))
                    resp = self._client.session.get(
                        url, params={"width": THUMBNAIL_FREE_WIDTH}, timeout=60
                    )
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
            paths.append(out_path)
        self.last_thumbnail_fallbacks = fallbacks
        if fallbacks:
            LOGGER.warning(
                "%d/%d thumbnail(s) fetched at %d px instead of %d: no download "
                "permission on their visual asset. Plans often delay download "
                "rights (30 days for Education & Research), so these are usually "
                "the most recent scenes; see client.info(). Scenes: %s",
                len(fallbacks),
                len(rows),
                THUMBNAIL_FREE_WIDTH,
                width,
                ", ".join(fallbacks[:10]) + (" …" if len(fallbacks) > 10 else ""),
            )
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

    def georeference_thumbnails(
        self,
        images_gdf: Optional[Union[gpd.GeoDataFrame, str, Path]] = None,
        *,
        thumbs: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        crop_to="aoi",
        **kwargs,
    ) -> pd.DataFrame:
        """:func:`georeference_thumbnails` with this finder's defaults.

        *thumbs* defaults to ``<workdir>/thumbnails``, *images_gdf* to
        ``<workdir>/footprints.gpkg``, ``crop_to="aoi"`` to the finder's AOI
        (``None`` keeps the whole scenes). Other keywords (``output_dir``,
        ``dst_crs``, ``resolution``, ``n_jobs``…) are passed through; the
        GeoTIFFs go next to the PNGs unless ``output_dir`` is given.
        """
        if thumbs is None:
            self._require_workdir("georeference_thumbnails")
            thumbs = self.thumbnail_dir
        if images_gdf is None:
            self._require_workdir("georeference_thumbnails")
            images_gdf = self.footprints_path
        if isinstance(crop_to, str) and crop_to == "aoi":
            crop_to = self.aoi_geom
        return georeference_thumbnails(thumbs, images_gdf, crop_to=crop_to, **kwargs)

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


# -------------------------
# 3) Thumbnail georeferencing (free, offline)
# -------------------------
THUMBNAIL_CRS = "EPSG:3857"
"""CRS the tiles API draws thumbnails in (north-up, square pixels)."""

GEOREF_COLUMNS = ["image_id", "png", "path", "crs", "res", "iou", "status", "message"]
"""Columns of the frame returned by :func:`georeference_thumbnails`."""

_EXECUTORS = ("thread", "process")


def thumbnail_transform(bounds: Sequence[float], width: int, height: int) -> _Affine:
    """Geotransform of a Planet thumbnail from its footprint bounds.

    Parameters
    ----------
    bounds : (minx, miny, maxx, maxy)
        Footprint bounding box in :data:`THUMBNAIL_CRS` (EPSG:3857).
    width, height : int
        PNG size in pixels.

    Returns
    -------
    affine.Affine
        North-up transform with square pixels: the long side of the box fills
        the frame and the short side is centred (letterboxing).
    """
    minx, miny, maxx, maxy = (float(b) for b in bounds)
    w_map, h_map = maxx - minx, maxy - miny
    if w_map <= 0 or h_map <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Degenerate thumbnail geometry: bounds={tuple(bounds)}, "
            f"size={width}x{height}."
        )
    s = max(w_map / width, h_map / height)
    x0 = minx - (width * s - w_map) / 2.0
    y0 = maxy + (height * s - h_map) / 2.0
    return _Affine(s, 0.0, x0, 0.0, -s, y0)


def _read_geometry_source(src) -> _BaseGeometry:
    """AOI/footprint (path, GeoDataFrame/GeoSeries, shapely, mapping) → WGS84 geometry."""
    if isinstance(src, (str, Path)):
        src = gpd.read_file(src)
    if isinstance(src, gpd.GeoSeries):
        src = gpd.GeoDataFrame(geometry=src)
    return PlanetImageFinder._coerce_to_wgs84_geometry(src)


def _resolve_dst_crs(dst_crs, reference: _BaseGeometry) -> str:
    """``None`` → EPSG:3857, ``"utm"`` → UTM zone of *reference*, else as given."""
    from rasterio.crs import CRS

    if dst_crs is None:
        return THUMBNAIL_CRS
    if isinstance(dst_crs, str) and dst_crs.lower() == "utm":
        utm = gpd.GeoSeries([reference], crs="EPSG:4326").estimate_utm_crs()
        return utm.to_string()
    return CRS.from_user_input(dst_crs).to_string()


def _georeference_core(
    png_path: Path,
    footprint: _BaseGeometry,
    out_path: Path,
    dst_crs: str,
    crop: Optional[_BaseGeometry],
    min_iou: float,
    resolution: Optional[float] = None,
    resampling: str = "nearest",
) -> Tuple[dict, List[str]]:
    """Georeference one PNG; returns (record, warnings). Raises on I/O errors."""
    import rasterio
    from PIL import Image
    from rasterio.crs import CRS
    from rasterio.enums import ColorInterp, Resampling
    from rasterio.transform import array_bounds
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds
    from rasterio.windows import from_bounds

    messages: List[str] = []
    with Image.open(png_path) as im:
        rgba = np.asarray(im.convert("RGBA"))
    height, width = rgba.shape[:2]

    fp_merc = gpd.GeoSeries([footprint], crs="EPSG:4326").to_crs(THUMBNAIL_CRS).iloc[0]
    transform = thumbnail_transform(fp_merc.bounds, width, height)

    # Check: opaque pixels vs. footprint, tested at pixel centres (the rule of
    # rasterio's rasterize). Pure shapely on purpose: rasterize builds a GDAL
    # MEM dataset and intermittently warns "no geotransform" under threads.
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    xs, ys = transform * (cols, rows)
    fp_mask = _shapely.contains_xy(fp_merc, xs, ys)
    opaque = rgba[..., 3] > 128
    union = int((fp_mask | opaque).sum())
    iou = float((fp_mask & opaque).sum()) / union if union else 0.0
    if iou < min_iou:
        messages.append(
            f"footprint/alpha IoU {iou:.3f} < {min_iou} — the thumbnail may not "
            "match its footprint (wrong id, or a changed thumbnail layout)"
        )

    data = np.moveaxis(rgba, -1, 0)  # (4, h, w)
    crs = CRS.from_user_input(THUMBNAIL_CRS)
    dst = CRS.from_user_input(dst_crs)
    method = Resampling[resampling]

    crop_bounds = None
    if crop is not None:
        # The footprint, not the PNG frame: an AOI inside the transparent
        # letterbox corners would otherwise give an empty file.
        if not footprint.intersects(crop):
            messages.append(
                "the crop AOI does not overlap the scene footprint; no file written"
            )
            res = float(resolution) if resolution is not None else abs(transform.a)
            return (
                {"path": None, "crs": dst.to_string(), "res": res, "iou": iou},
                messages,
            )
        crop_bounds = tuple(
            gpd.GeoSeries([crop], crs="EPSG:4326").to_crs(dst).total_bounds
        )

    if resolution is not None:
        # Common grid: pixel size *resolution*, origin on a multiple of it
        # (gdalwarp -tr -tap), so outputs of different scenes align pixel for
        # pixel. With a crop, the full AOI box is used — every date gets the
        # same bounds and shape, transparent where the scene does not reach.
        r = float(resolution)
        if crop_bounds is not None:
            minx, miny, maxx, maxy = crop_bounds
        else:
            minx, miny, maxx, maxy = transform_bounds(
                crs, dst, *array_bounds(height, width, transform)
            )
        minx, miny = np.floor(minx / r) * r, np.floor(miny / r) * r
        maxx, maxy = np.ceil(maxx / r) * r, np.ceil(maxy / r) * r
        dst_w = max(int(round((maxx - minx) / r)), 1)
        dst_h = max(int(round((maxy - miny) / r)), 1)
        dst_transform = _Affine(r, 0.0, minx, 0.0, -r, maxy)
        warped = np.zeros((4, dst_h, dst_w), dtype=np.uint8)  # alpha 0 outside
        reproject(
            data,
            warped,
            src_transform=transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=dst,
            resampling=method,
        )
        data, transform, crs = warped, dst_transform, dst
    else:
        if dst != crs:
            dst_transform, dst_w, dst_h = calculate_default_transform(
                crs, dst, width, height, *array_bounds(height, width, transform)
            )
            warped = np.zeros((4, dst_h, dst_w), dtype=np.uint8)  # alpha 0 outside
            reproject(
                data,
                warped,
                src_transform=transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs=dst,
                resampling=method,
            )
            data, transform, crs = warped, dst_transform, dst
        if crop_bounds is not None:
            cminx, cminy, cmaxx, cmaxy = crop_bounds
            iminx, iminy, imaxx, imaxy = array_bounds(
                data.shape[1], data.shape[2], transform
            )
            win = from_bounds(
                max(cminx, iminx),
                max(cminy, iminy),
                min(cmaxx, imaxx),
                min(cmaxy, imaxy),
                transform=transform,
            )
            eps = 1e-6
            c0 = max(int(np.floor(win.col_off + eps)), 0)
            r0 = max(int(np.floor(win.row_off + eps)), 0)
            c1 = min(int(np.ceil(win.col_off + win.width - eps)), data.shape[2])
            r1 = min(int(np.ceil(win.row_off + win.height - eps)), data.shape[1])
            data = data[:, r0:r1, c0:c1]
            transform = transform * _Affine.translation(c0, r0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    profile = {
        "driver": "GTiff",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": 4,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "photometric": "RGB",
        "alpha": "YES",
        "compress": "deflate",
    }
    try:
        with rasterio.open(part, "w", **profile) as ds:
            ds.write(data)
            ds.colorinterp = [
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            ]
        part.replace(out_path)
    finally:
        if part.exists():
            part.unlink()
    return (
        {"path": out_path, "crs": crs.to_string(), "res": abs(transform.a), "iou": iou},
        messages,
    )


def _check_grid_args(resolution, resampling: str) -> None:
    """Validate *resolution* / *resampling* in the caller, before any worker."""
    from rasterio.enums import Resampling

    if resolution is not None and not (float(resolution) > 0):
        raise ValueError(f"resolution must be > 0, got {resolution!r}")
    if resampling not in Resampling.__members__:
        raise ValueError(
            f"Unknown resampling {resampling!r}; one of "
            f"{', '.join(Resampling.__members__)}"
        )


def georeference_thumbnail(
    png_path: Union[str, Path],
    footprint,
    *,
    out_path: Optional[Union[str, Path]] = None,
    dst_crs=None,
    crop_to=None,
    resolution: Optional[float] = None,
    resampling: str = "nearest",
    min_iou: float = 0.9,
) -> dict:
    """Write a georeferenced GeoTIFF for one Planet thumbnail.

    Parameters
    ----------
    png_path : str | Path
        Thumbnail from :meth:`PlanetImageFinder.download_thumbnails`.
    footprint : shapely geometry | GeoSeries | GeoDataFrame | mapping | path
        The scene footprint (the ``geometry`` of its :meth:`search_metadata`
        row). A bare shapely geometry or mapping is taken as EPSG:4326.
    out_path : str | Path, optional
        Output GeoTIFF; default ``<png folder>/<png stem>.tif``.
    dst_crs : None | "utm" | CRS-like, optional
        ``None`` keeps EPSG:3857, the thumbnail's own grid (no resampling).
        ``"utm"`` reprojects to the UTM zone of *crop_to* (or of the
        footprint); anything else is read by ``rasterio.crs.CRS``.
    crop_to : AOI-like, optional
        Clip the output to the AOI bounding box (whole pixels, in the output
        CRS). GeoDataFrame, shapely (EPSG:4326), mapping or a vector file path.
        An AOI that misses the scene footprint writes nothing.
    resolution : float, optional
        Pixel size of a **common grid**, in output-CRS units (metres for
        UTM): the origin is snapped to a multiple of it, so outputs of
        different scenes align pixel for pixel. With *crop_to*, the output
        is the whole AOI box snapped outward — the same bounds and shape for
        every scene, transparent where a scene does not reach. ``None`` keeps
        each thumbnail's own pixel size (no resampling in EPSG:3857).
    resampling : str, default "nearest"
        ``rasterio.enums.Resampling`` name used whenever the image is warped
        (``"bilinear"``, ``"cubic"``…; these also soften the alpha edge).
    min_iou : float, default 0.9
        Below this footprint/alpha overlap a warning is logged.

    Returns
    -------
    dict
        ``image_id``, ``png``, ``path`` (``None`` when the crop AOI misses
        the image), ``crs``, ``res`` (pixel size in CRS units — Mercator
        metres for EPSG:3857, i.e. ground size / cos(latitude)), ``iou``.
    """
    _check_grid_args(resolution, resampling)
    png_path = Path(png_path)
    fp = _read_geometry_source(footprint)
    crop = _read_geometry_source(crop_to) if crop_to is not None else None
    crs = _resolve_dst_crs(dst_crs, crop if crop is not None else fp)
    out = Path(out_path) if out_path is not None else png_path.with_suffix(".tif")
    rec, messages = _georeference_core(
        png_path, fp, out, crs, crop, min_iou, resolution, resampling
    )
    for msg in messages:
        LOGGER.warning("%s: %s", png_path.stem, msg)
    return {"image_id": png_path.stem, "png": png_path, **rec}


def _georeference_job(
    png_path: str,
    footprint_wkb: bytes,
    out_path: str,
    dst_crs: str,
    crop_wkb: Optional[bytes],
    min_iou: float,
    resolution: Optional[float] = None,
    resampling: str = "nearest",
) -> dict:
    """Worker for :func:`georeference_thumbnails` (module level so it pickles).

    Never raises and never logs: failures and warnings travel back in the
    record, and the parent process logs them.
    """
    from shapely import wkb as _wkb

    stem = Path(png_path).stem
    try:
        fp = _wkb.loads(footprint_wkb)
        crop = _wkb.loads(crop_wkb) if crop_wkb is not None else None
        rec, messages = _georeference_core(
            Path(png_path),
            fp,
            Path(out_path),
            dst_crs,
            crop,
            min_iou,
            resolution,
            resampling,
        )
        rec["status"] = "written" if rec["path"] is not None else "outside"
        rec["message"] = "; ".join(messages) or None
    except Exception as exc:  # one bad PNG must not cancel the pool
        rec = {
            "path": None,
            "crs": None,
            "res": np.nan,
            "iou": np.nan,
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
        }
    rec["image_id"] = stem
    rec["png"] = Path(png_path)
    return rec


def georeference_thumbnails(
    thumbs: Union[str, Path, Sequence[Union[str, Path]]],
    images_gdf: Union[gpd.GeoDataFrame, str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    dst_crs=None,
    crop_to=None,
    resolution: Optional[float] = None,
    resampling: str = "nearest",
    overwrite: bool = False,
    min_iou: float = 0.9,
    n_jobs: Optional[int] = None,
    executor: str = "thread",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Georeference many Planet thumbnails in parallel from their footprints.

    Each ``<image_id>.png`` is matched to the row of *images_gdf* with that
    ``image_id`` and written as ``<image_id>.tif`` (RGB + alpha GeoTIFF).

    Parameters
    ----------
    thumbs : folder | sequence of paths
        A folder (every ``*.png`` in it) or the list returned by
        :meth:`PlanetImageFinder.download_thumbnails`.
    images_gdf : GeoDataFrame | str | Path
        The :meth:`PlanetImageFinder.search_metadata` frame, or a vector file
        holding it (e.g. the GPKG it was saved to). Needs ``image_id``.
    output_dir : str | Path, optional
        Where to write the GeoTIFFs; default next to each PNG.
    dst_crs, crop_to, resolution, resampling, min_iou
        As in :func:`georeference_thumbnail`. ``"utm"`` is resolved **once**
        for the whole batch (from *crop_to*, else from all footprints), so
        every output shares one CRS. Add *resolution* (and *crop_to*) to put
        every output on one grid — identical shapes, a stack ready to load.
    overwrite : bool, default False
        Rewrite GeoTIFFs that already exist (otherwise ``status="skipped"``).
    n_jobs : int, optional
        Number of workers; default ``min(os.cpu_count(), n_pending)``.
        ``1`` runs serially in this process (no pool).
    executor : {"thread", "process"}, default "thread"
        ``concurrent.futures`` pool type. Threads are the default: GDAL,
        PIL decoding and shapely release the GIL, so on 259 real 512-px
        thumbnails they took 1.7 s against 4.0 s serially. ``"process"``
        was 1.15 s with Linux ``fork``, but forking a multi-threaded parent
        (any Jupyter kernel) can deadlock the children, and the safe start
        methods (``spawn``, ``forkserver``, the default from Python 3.14)
        re-import the package (~4 s) — slower than serial for a few
        hundred thumbnails. Use it for very large batches from a script.
    show_progress : bool, default True
        tqdm progress bar, when tqdm is installed.

    Returns
    -------
    pandas.DataFrame
        One row per PNG, in input order, with :data:`GEOREF_COLUMNS`.
        ``status`` is ``written``, ``skipped``, ``no_footprint``,
        ``outside`` (crop AOI misses the image) or ``failed``; ``message``
        holds the warning or error text.
    """
    if executor not in _EXECUTORS:
        raise ValueError(f"executor must be one of {_EXECUTORS}, got {executor!r}")
    if n_jobs is not None and n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    _check_grid_args(resolution, resampling)

    if isinstance(thumbs, (str, Path)):
        folder = Path(thumbs)
        if not folder.is_dir():
            raise FileNotFoundError(f"Thumbnail folder not found: {folder}")
        pngs = sorted(folder.glob("*.png"))
    else:
        pngs = [Path(p) for p in thumbs]

    if isinstance(images_gdf, (str, Path)):
        images_gdf = gpd.read_file(images_gdf)
    if "image_id" not in images_gdf.columns:
        raise ValueError("images_gdf has no 'image_id' column.")
    gdf = images_gdf
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    footprints: Dict[str, _BaseGeometry] = {}
    for iid, geom in zip(gdf["image_id"].astype(str), gdf.geometry):
        if geom is not None and not geom.is_empty:
            footprints.setdefault(iid, geom)

    crop = _read_geometry_source(crop_to) if crop_to is not None else None
    matched = [footprints[p.stem] for p in pngs if p.stem in footprints]
    if crop is not None:
        reference = crop
    elif matched:
        reference = gpd.GeoSeries(matched, crs="EPSG:4326").union_all()
    else:
        reference = None
    crs = (
        _resolve_dst_crs(dst_crs, reference) if reference is not None else THUMBNAIL_CRS
    )
    crop_wkb = crop.wkb if crop is not None else None

    records: List[Optional[dict]] = [None] * len(pngs)
    jobs: List[Tuple[int, tuple]] = []
    unmatched: List[str] = []
    for i, png in enumerate(pngs):
        out = (Path(output_dir) if output_dir is not None else png.parent) / (
            png.stem + ".tif"
        )
        base = {
            "image_id": png.stem,
            "png": png,
            "path": None,
            "crs": None,
            "res": np.nan,
            "iou": np.nan,
            "message": None,
        }
        if png.stem not in footprints:
            unmatched.append(png.stem)
            records[i] = {**base, "status": "no_footprint"}
        elif out.exists() and not overwrite:
            records[i] = {**base, "path": out, "status": "skipped"}
        else:
            jobs.append(
                (
                    i,
                    (
                        str(png),
                        footprints[png.stem].wkb,
                        str(out),
                        crs,
                        crop_wkb,
                        min_iou,
                        resolution,
                        resampling,
                    ),
                )
            )
    if unmatched:
        LOGGER.warning(
            "%d thumbnail(s) have no footprint in images_gdf (file stem must be "
            "the image_id): %s",
            len(unmatched),
            ", ".join(unmatched[:10]) + (" …" if len(unmatched) > 10 else ""),
        )

    progress = (
        _tqdm(total=len(jobs), desc="Georeferencing", unit="img")
        if (_HAS_TQDM and show_progress and jobs)
        else None
    )
    workers = n_jobs or min(_os.cpu_count() or 1, max(len(jobs), 1))
    try:
        if workers == 1 or len(jobs) <= 1:
            for i, args in jobs:
                records[i] = _georeference_job(*args)
                if progress is not None:
                    progress.update(1)
        else:
            pool_cls = (
                _futures.ThreadPoolExecutor
                if executor == "thread"
                else _futures.ProcessPoolExecutor
            )
            with _warnings.catch_warnings(), pool_cls(max_workers=workers) as pool:
                # Concurrent rasterio reprojects emit about one spurious
                # NotGeoreferencedWarning per batch (GDAL error state crossing
                # threads; never in serial). The source transform is always
                # passed explicitly, and threaded outputs were verified
                # pixel-identical to serial ones (5 × 259 real thumbnails).
                # The filter is process-wide, but only for this block, and the
                # block exits only after every worker has finished.
                from rasterio.errors import NotGeoreferencedWarning

                _warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                future_to_index = {
                    pool.submit(_georeference_job, *args): i for i, args in jobs
                }
                for fut in _futures.as_completed(future_to_index):
                    records[future_to_index[fut]] = fut.result()
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    for rec in records:
        if rec["message"] and rec["status"] in ("written", "outside", "failed"):
            LOGGER.warning(
                "%s [%s]: %s", rec["image_id"], rec["status"], rec["message"]
            )

    frame = pd.DataFrame(records, columns=GEOREF_COLUMNS)
    counts = frame["status"].value_counts().to_dict()
    LOGGER.info("Georeferenced thumbnails: %s", counts)
    return frame
