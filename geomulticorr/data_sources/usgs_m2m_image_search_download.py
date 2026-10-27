#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# usgs_m2m_image_search_download.py
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
"""USGS M2M API client for bulk Landsat Collection-2 downloads.

Supports a free USGS/EROS account (register at https://ers.cr.usgs.gov/register).
Access to Landsat 5/7/8/9 Level-1 and Level-2 collections,
with per-band selection (e.g., Band 8 for 15 m panchromatic imagery).

No quota: download as much as you need.

Authentication
--------------
USGS retired password-based M2M login on 2025-02-26. Authentication now requires
an **application token** (not your account password): generate one at
https://ers.cr.usgs.gov/password/appgenerate while logged in to your USGS
account, then pass it as ``api_token`` / the ``USGS_M2M_TOKEN`` env var,
alongside your ``username`` / ``USGS_M2M_USERNAME``.

API endpoint
------------
The M2M REST API lives at ``https://m2m.cr.usgs.gov/api/api/json/stable`` — note
this is **not** ``m2m.usgs.gov`` (that hostname does not resolve). Every call
after login carries the session token obtained from ``login-token`` in an
``X-Auth-Token`` header (not embedded in the JSON body).
"""

from __future__ import annotations

import json
import logging as _logging
import os
import re
import time as _time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import urlopen, Request

import geopandas as gpd
import numpy as np
from shapely.geometry import box, shape as _shape
from shapely.geometry.base import BaseGeometry as _BaseGeometry

from geomulticorr.data_sources._layout import (
    ORGANIZE_SCHEMES,
    ORGANIZE_SENSOR_PATHROW,
    filename_from_content_disposition,
    landsat_dirs,
    safe_filename,
)

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

LOGGER = _logging.getLogger("usgs_m2m_image_search_download")


def _url_basename(url: str) -> Optional[str]:
    """Filename embedded in a URL path, or None when there is none worth using.

    M2M download URLs are often extensionless redirect endpoints, so a basename
    without a suffix carries no information and is rejected here rather than
    becoming a filename like ``download`` on disk.
    """
    name = Path(url.split("?", 1)[0]).name
    if not name or not Path(name).suffix:
        return None
    return safe_filename(name, fallback="") or None


M2M_BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"
M2M_LOGIN_ENDPOINT = f"{M2M_BASE_URL}/login-token"
M2M_LOGOUT_ENDPOINT = f"{M2M_BASE_URL}/logout"
M2M_SEARCH_ENDPOINT = f"{M2M_BASE_URL}/scene-search"
M2M_DOWNLOAD_OPTIONS_ENDPOINT = f"{M2M_BASE_URL}/download-options"
M2M_DOWNLOAD_REQUEST_ENDPOINT = f"{M2M_BASE_URL}/download-request"
M2M_DOWNLOAD_RETRIEVE_ENDPOINT = f"{M2M_BASE_URL}/download-retrieve"
M2M_PERMISSIONS_ENDPOINT = f"{M2M_BASE_URL}/permissions"
M2M_DOWNLOAD_SEARCH_ENDPOINT = f"{M2M_BASE_URL}/download-search"
M2M_RATE_LIMIT_ENDPOINT = f"{M2M_BASE_URL}/rate-limit-summary"

#: Per-endpoint read timeouts (seconds). One global value cannot serve both:
#: ``login`` answers instantly, whereas ``download-request`` must stage every
#: file server-side before it replies — minutes for a few hundred entries.
#: Timing out on that endpoint does **not** cancel the work; it only loses the
#: URLs, so an over-tight timeout silently burns download quota.
TIMEOUT_DEFAULT_S = 30.0
TIMEOUT_SEARCH_S = 120.0
TIMEOUT_DOWNLOAD_OPTIONS_S = 300.0
TIMEOUT_DOWNLOAD_REQUEST_S = 600.0
TIMEOUT_DOWNLOAD_RETRIEVE_S = 180.0

#: M2M serialises calls per account and answers an overlapping one with HTTP 500
#: ``RATE_LIMIT``. It is a concurrency guard, not a volume ceiling, so it clears
#: on its own — wait and retry rather than failing the batch.
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BACKOFF_S = 20.0

LANDSAT_COLLECTIONS = {
    "L9_L1": {"api_id": "landsat_ot_c2_l1", "name": "Landsat 9 L1", "scale_m": 15},
    "L8_L1": {"api_id": "landsat_ot_c2_l1", "name": "Landsat 8 L1", "scale_m": 15},
    "L7_L1": {"api_id": "landsat_etm_c2_l1", "name": "Landsat 7 L1", "scale_m": 15},
    "L5_L1": {"api_id": "landsat_tm_c2_l1", "name": "Landsat 5 L1", "scale_m": 15},
    "L9_L2": {"api_id": "landsat_ot_c2_l2", "name": "Landsat 9 L2", "scale_m": 30},
    "L8_L2": {"api_id": "landsat_ot_c2_l2", "name": "Landsat 8 L2", "scale_m": 30},
    "L7_L2": {"api_id": "landsat_etm_c2_l2", "name": "Landsat 7 L2", "scale_m": 30},
    "L5_L2": {"api_id": "landsat_tm_c2_l2", "name": "Landsat 5 L2", "scale_m": 30},
}

BAND_NAMES = {
    "B1": "Coastal",
    "B2": "Blue",
    "B3": "Green",
    "B4": "Red",
    "B5": "NIR",
    "B6": "SWIR1",
    "B7": "SWIR2",
    "B8": "Panchromatic (15m)",
    "B9": "Cirrus",
    "B10": "TIRS1",
    "B11": "TIRS2",
    "QA_PIXEL": "Quality Assessment",
    "QA_RADIANCE": "Radiometric Saturation",
    "MTL.txt": "Metadata (MTL)",
    "ANG.txt": "Angle Coefficients",
}


# -------------------------
# 1) Connection utilities
# -------------------------
class M2MClient:
    """USGS M2M API client for Landsat Collection-2 data.

    Parameters
    ----------
    username : str, optional
        USGS/EROS username. If omitted, reads from USGS_M2M_USERNAME env var.
    api_token : str, optional
        M2M application token (**not** your account password — USGS retired
        password-based M2M login on 2025-02-26). Generate one at
        https://ers.cr.usgs.gov/password/appgenerate while logged in. If
        omitted, reads from the USGS_M2M_TOKEN env var.
    password : str, optional
        Deprecated / no longer accepted by the server. Kept only so old calls
        fail with an actionable message instead of a silent auth error.

    Notes
    -----
    Register a free account at https://ers.cr.usgs.gov/register.
    ``login()`` exchanges the (permanent) application token for a short-lived
    session token, sent as an ``X-Auth-Token`` header on every subsequent call.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        api_token: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.username = username or os.environ.get("USGS_M2M_USERNAME")
        self.api_token = api_token or os.environ.get("USGS_M2M_TOKEN")
        self.password = password or os.environ.get("USGS_M2M_PASSWORD")
        #: short-lived token returned by login-token; sent as X-Auth-Token
        self.session_token: Optional[str] = None

    def login(self) -> str:
        """Exchange the application token for a session token.

        Returns
        -------
        str
            Session token (stored internally; valid for a few hours).

        Raises
        ------
        ValueError
            If username/api_token are not provided.
        RuntimeError
            If authentication fails.
        """
        if not self.username:
            raise ValueError(
                "USGS_M2M_USERNAME must be set (via constructor or environment variable)"
            )
        if not self.api_token:
            if self.password:
                raise RuntimeError(
                    "USGS M2M retired password-based login on 2025-02-26. Generate an "
                    "application token at https://ers.cr.usgs.gov/password/appgenerate "
                    "and pass it as api_token= or set USGS_M2M_TOKEN instead of a password."
                )
            raise ValueError(
                "USGS_M2M_TOKEN (or api_token=) must be set. Generate an application "
                "token at https://ers.cr.usgs.gov/password/appgenerate"
            )

        payload = {"username": self.username, "token": self.api_token}
        try:
            response = self._post(M2M_LOGIN_ENDPOINT, payload, authenticated=False)
            self.session_token = response.get("data")
            if not self.session_token:
                raise RuntimeError("Login failed: no session token returned")
            LOGGER.info(f"M2M login successful for user '{self.username}'")
            return self.session_token
        except Exception as e:
            raise RuntimeError(f"M2M login failed: {e}")

    def logout(self) -> None:
        """Invalidate the current session token."""
        if not self.session_token:
            return
        try:
            self._post(M2M_LOGOUT_ENDPOINT, {})
            self.session_token = None
            LOGGER.info("M2M logout successful")
        except Exception as e:
            LOGGER.warning(f"M2M logout failed: {e}")

    def permissions(self) -> List[str]:
        """Roles granted to this account, e.g. ``["user", "download"]``.

        Searching and downloading are authorised separately: ``scene-search``
        works for any authenticated M2M account, while ``download-options`` and
        ``download-request`` need the ``download`` role. That asymmetry is
        exactly what a 403 on the download endpoints — after a search that
        worked — is telling you.

        Returns
        -------
        list[str]
            Granted role names. Missing ``"download"`` means bulk downloads will
            keep returning 403 until access is granted at
            https://ers.cr.usgs.gov/profile/access

        Examples
        --------
        >>> client.permissions()  # doctest: +SKIP
        ['user', 'download']
        """
        if not self.session_token:
            self.login()
        response = self._post(M2M_PERMISSIONS_ENDPOINT, {})
        data = response.get("data") or []
        return [str(p) for p in data]

    def search_scenes(
        self,
        collection_id: str,
        geometry: _BaseGeometry,
        date_range: Tuple[str, str],
        cloud_max: int = 100,
        max_results: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Search for scenes in a collection within an AOI and date range.

        Parameters
        ----------
        collection_id : str
            M2M dataset name (e.g., 'landsat_ot_c2_l1').
        geometry : shapely geometry
            AOI boundary (in EPSG:4326).
        date_range : tuple[str, str]
            (start_date, end_date) as 'YYYY-MM-DD'.
        cloud_max : int, default 100
            Maximum cloud cover percentage (0-100).
        max_results : int, default 5000
            Maximum scenes returned per search.

        Returns
        -------
        list[dict]
            Raw scene dicts from M2M (entityId, displayId, cloudCover,
            temporalCoverage, spatialCoverage, etc.).
        """
        if not self.session_token:
            self.login()

        minx, miny, maxx, maxy = geometry.bounds
        payload = {
            "datasetName": collection_id,
            "sceneFilter": {
                "spatialFilter": {
                    "filterType": "mbr",
                    "lowerLeft": {"latitude": miny, "longitude": minx},
                    "upperRight": {"latitude": maxy, "longitude": maxx},
                },
                "acquisitionFilter": {
                    "start": date_range[0],
                    "end": date_range[1],
                },
                "cloudCoverFilter": {
                    "min": 0,
                    "max": cloud_max,
                    "includeUnknown": False,
                },
            },
            "sortOrder": "ASC",
            "maxResults": max_results,
            "startingNumber": 1,
        }

        try:
            response = self._post(
                M2M_SEARCH_ENDPOINT, payload, timeout=TIMEOUT_SEARCH_S
            )
            results = response.get("data", {}).get("results", [])
            LOGGER.info(
                f"M2M search: {len(results)} scenes in '{collection_id}' "
                f"from {date_range[0]} to {date_range[1]}"
            )
            return results
        except Exception as e:
            LOGGER.error(f"M2M search failed: {e}")
            raise

    def download_options(
        self, collection_id: str, entity_ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """List downloadable products for a set of scenes.

        Parameters
        ----------
        collection_id : str
            M2M dataset name.
        entity_ids : sequence[str]
            Scene entity IDs (from ``search_scenes`` results' ``entityId``).

        Returns
        -------
        list[dict]
            One entry per downloadable product per scene, each with (at least)
            ``id`` (the download option id passed to ``download_request``),
            ``entityId``, ``productName`` and ``available``.
        """
        if not self.session_token:
            self.login()
        payload = {"datasetName": collection_id, "entityIds": list(entity_ids)}
        response = self._post(
            M2M_DOWNLOAD_OPTIONS_ENDPOINT, payload, timeout=TIMEOUT_DOWNLOAD_OPTIONS_S
        )
        return response.get("data", [])

    def download_request(
        self, downloads: Sequence[Dict[str, str]], label: str = "geomulticorr"
    ) -> Dict[str, Any]:
        """Request downloads for a set of product ids.

        Parameters
        ----------
        downloads : sequence[dict]
            ``{"entityId": ..., "productId": ...}`` pairs, from ``download_options``.
        label : str, default "geomulticorr"
            Batch label, also used to poll ``download_retrieve``.

        Returns
        -------
        dict
            Response ``data``: ``availableDownloads`` (ready now, each with a
            ``url``) and ``preparingDownloads`` (need a ``download_retrieve``
            poll before their URL is ready).
        """
        if not self.session_token:
            self.login()
        payload = {"downloads": list(downloads), "label": label}
        response = self._post(
            M2M_DOWNLOAD_REQUEST_ENDPOINT, payload, timeout=TIMEOUT_DOWNLOAD_REQUEST_S
        )
        return response.get("data", {})

    def download_retrieve(self, label: str = "geomulticorr") -> Dict[str, Any]:
        """Poll for downloads that were still preparing at request time.

        Returns
        -------
        dict
            Response ``data``, whose ``available`` list holds now-ready
            ``{"downloadId": ..., "url": ...}`` entries.
        """
        if not self.session_token:
            self.login()
        response = self._post(
            M2M_DOWNLOAD_RETRIEVE_ENDPOINT,
            {"label": label},
            timeout=TIMEOUT_DOWNLOAD_RETRIEVE_S,
        )
        return response.get("data", {})

    def download_search(self, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """List downloads sitting in the account queue, without consuming them.

        Unlike :meth:`download_retrieve`, this does not mark anything retrieved
        and returns **no URLs** — it is the safe way to ask "what did that
        request actually stage?" after a client-side timeout.

        Parameters
        ----------
        label : str, optional
            Restrict to one batch label. ``None`` lists the whole queue.

        Returns
        -------
        list[dict]
            Records with ``downloadId``, ``entityId``, ``displayId``,
            ``productId``, ``filesize`` and ``statusText``.
        """
        if not self.session_token:
            self.login()
        payload: Dict[str, Any] = {"label": label} if label else {}
        response = self._post(
            M2M_DOWNLOAD_SEARCH_ENDPOINT, payload, timeout=TIMEOUT_SEARCH_S
        )
        return response.get("data") or []

    def rate_limit_summary(self) -> Dict[str, Any]:
        """Download quota and current usage for this account and IP.

        Useful for telling a genuine ceiling apart from M2M's ``RATE_LIMIT``
        response, which is a concurrency guard rather than a volume limit.
        """
        if not self.session_token:
            self.login()
        response = self._post(
            M2M_RATE_LIMIT_ENDPOINT, {"authType": [], "ipAddress": []}
        )
        return response.get("data") or {}

    def _post(
        self,
        url: str,
        payload: Dict[str, Any],
        authenticated: bool = True,
        timeout: Optional[float] = None,
        rate_limit_retries: int = RATE_LIMIT_RETRIES,
    ) -> Dict[str, Any]:
        """POST to the M2M API, authenticating via an X-Auth-Token header.

        Retries transparently on M2M's ``RATE_LIMIT`` response, which signals
        "another request from this account is still running" rather than a
        volume ceiling, and therefore clears by waiting.

        Parameters
        ----------
        timeout : float, optional
            Read timeout. Defaults to :data:`TIMEOUT_DEFAULT_S`; slow endpoints
            pass their own — see the ``TIMEOUT_*`` constants.
        rate_limit_retries : int
            Attempts remaining for a ``RATE_LIMIT`` rejection.
        """
        for attempt in range(rate_limit_retries + 1):
            try:
                return self._post_once(url, payload, authenticated, timeout)
            except RuntimeError as exc:
                if "RATE_LIMIT" not in str(exc) or attempt >= rate_limit_retries:
                    raise
                wait = RATE_LIMIT_BACKOFF_S * (attempt + 1)
                LOGGER.warning(
                    "RATE_LIMIT on %s (another request from this account is still "
                    "running); waiting %.0fs then retrying (%d/%d)",
                    url.rsplit("/", 1)[-1],
                    wait,
                    attempt + 1,
                    rate_limit_retries,
                )
                _time.sleep(wait)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _post_once(
        self,
        url: str,
        payload: Dict[str, Any],
        authenticated: bool = True,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One POST attempt, with no rate-limit retry."""
        headers = {"Content-Type": "application/json"}
        if authenticated and self.session_token:
            headers["X-Auth-Token"] = self.session_token

        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, headers=headers)

        try:
            with urlopen(req, timeout=timeout or TIMEOUT_DEFAULT_S) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("errorCode"):
                    raise RuntimeError(
                        f"M2M API error {result['errorCode']}: {result.get('errorMessage', '')}"
                    )
                return result
        except HTTPError as exc:
            # M2M returns its real diagnosis in the JSON body even on a 4xx, and
            # HTTPError is a readable file object. Reporting only the status code
            # turns every distinct cause into an indistinguishable
            # "HTTP Error 403: Forbidden".
            detail = ""
            try:
                payload_err = json.loads(exc.read().decode("utf-8"))
                code = payload_err.get("errorCode") or ""
                message = payload_err.get("errorMessage") or ""
                detail = (
                    f" — {code}: {message}".rstrip(": ") if (code or message) else ""
                )
            except Exception:  # body absent or not JSON
                pass

            endpoint = url.rsplit("/", 1)[-1]
            hint = ""
            if exc.code in (401, 403):
                if endpoint.startswith("download"):
                    hint = (
                        f"\nA valid token is not sufficient here: '{endpoint}' is authorised "
                        f"separately from 'scene-search'. If searching works and this does not, "
                        f"the account is authenticated but lacks the download role. Check with "
                        f"M2MClient.permissions() — it should list 'download' — and request it "
                        f"at https://ers.cr.usgs.gov/profile/access"
                    )
                else:
                    hint = (
                        "\nThe session token is missing, expired (they last a few hours) or "
                        "rejected. Call client.login() again; if it still fails, regenerate the "
                        "application token at https://ers.cr.usgs.gov/password/appgenerate"
                    )
            raise RuntimeError(
                f"M2M API request failed: HTTP {exc.code} on {endpoint}{detail}{hint}"
            ) from exc
        except Exception as e:
            raise RuntimeError(f"M2M API request failed: {e}")


# -------------------------
# 2) Image finder
# -------------------------
class M2MImageFinder:
    """Search and download Landsat Collection-2 imagery via USGS M2M.

    Parameters
    ----------
    client : M2MClient
        An M2M API client (credentials will be used to login if needed).
    aoi : GeoDataFrame | shapely geometry
        Area of interest in EPSG:4326.
    year_range : tuple[int, int], default (2020, 2025)
        Inclusive year range.
    month_range : tuple[int, int], default (1, 12)
        Inclusive month range (1-12).
    cloud_max : int, default 30
        Maximum cloud cover percentage.
    """

    def __init__(
        self,
        client: M2MClient,
        *,
        aoi: Union[gpd.GeoDataFrame, _BaseGeometry],
        year_range: Tuple[int, int] = (2020, 2025),
        month_range: Tuple[int, int] = (1, 12),
        cloud_max: int = 30,
    ) -> None:
        self.client = client
        self.year_range = year_range
        self.month_range = month_range
        self.cloud_max = cloud_max

        if isinstance(aoi, gpd.GeoDataFrame):
            aoi = aoi.to_crs("EPSG:4326").unary_union
        self.aoi = aoi
        self.aoi_area_km2 = aoi.area * 111.32 * 111.32  # rough approximation

    def search_metadata(
        self,
        collections: Optional[Sequence[str]] = None,
    ) -> gpd.GeoDataFrame:
        """Search M2M for Landsat scenes.

        Parameters
        ----------
        collections : sequence[str], optional
            Collection keys to search (e.g., ["L8_L1", "L9_L1"]).
            If omitted, searches L8 and L9 Level-1.

        Returns
        -------
        geopandas.GeoDataFrame
            Columns: image_id, collection_key, acquisition_date, cloud_percent,
            bands_available, geometry (footprint in EPSG:4326).
        """
        if collections is None:
            collections = ["L8_L1", "L9_L1"]

        all_scenes = []
        for coll_key in collections:
            if coll_key not in LANDSAT_COLLECTIONS:
                LOGGER.warning(f"Unknown collection key '{coll_key}', skipping")
                continue

            coll_info = LANDSAT_COLLECTIONS[coll_key]
            api_id = coll_info["api_id"]

            date_start = f"{self.year_range[0]}-{self.month_range[0]:02d}-01"
            date_end = f"{self.year_range[1]}-{self.month_range[1]:02d}-28"

            try:
                scenes = self.client.search_scenes(
                    api_id, self.aoi, (date_start, date_end), self.cloud_max
                )
                for scene in scenes:
                    temporal = scene.get("temporalCoverage") or {}
                    acq_date = (
                        scene.get("acquisitionDate")
                        or temporal.get("startDate")
                        or temporal.get("endDate")
                    )
                    cloud = scene.get("cloudCover")
                    all_scenes.append(
                        {
                            "image_id": scene["entityId"],
                            "collection_key": coll_key,
                            "acquisition_date": acq_date,
                            "cloud_percent": (
                                cloud if cloud not in (None, "-1") else None
                            ),
                            "display_id": scene["displayId"],
                            "bands_available": [
                                "B1",
                                "B2",
                                "B3",
                                "B4",
                                "B5",
                                "B6",
                                "B7",
                                "B8",
                                "B9",
                                "B10",
                                "B11",
                            ],
                        }
                    )
            except Exception as e:
                LOGGER.error(f"Failed to search collection '{coll_key}': {e}")

        if not all_scenes:
            return gpd.GeoDataFrame(
                columns=[
                    "image_id",
                    "collection_key",
                    "acquisition_date",
                    "cloud_percent",
                    "display_id",
                    "bands_available",
                    "geometry",
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )

        rows = []
        for scene in all_scenes:
            cloud = scene["cloud_percent"]
            rows.append(
                {
                    "image_id": scene["image_id"],
                    "collection_key": scene["collection_key"],
                    "acquisition_date": pd.to_datetime(scene["acquisition_date"]),
                    "cloud_percent": float(cloud) if cloud is not None else np.nan,
                    "display_id": scene["display_id"],
                    "bands_available": scene["bands_available"],
                    "geometry": box(*self.aoi.bounds),
                }
            )

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        gdf = gdf.sort_values(
            ["acquisition_date", "cloud_percent"], ascending=[False, True]
        )
        return gdf.reset_index(drop=True)

    def download_images(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        output_dir: Union[str, Path],
        bands: Optional[Sequence[str]] = None,
        download_metadata: bool = True,
        show_progress: bool = True,
        poll_interval_s: float = 5.0,
        max_poll_attempts: int = 12,
        keep_original_names: bool = False,
        organize_by: Optional[str] = None,
        max_files_per_request: int = 100,
    ) -> List[Path]:
        """Download Landsat scenes from M2M.

        Follows the M2M ``download-options`` -> ``download-request`` ->
        ``download-retrieve`` sequence: available products are discovered per
        scene, requested as a batch, and any still "preparing" are polled
        until ready (or ``max_poll_attempts`` is reached, in which case a
        warning names the label to retrieve later with
        ``client.download_retrieve(label=...)``).

        Parameters
        ----------
        images_gdf : GeoDataFrame
            Output from search_metadata (or compatible).
        output_dir : str | Path
            Root directory for downloads (one subdirectory per collection key).
        bands : sequence[str], optional
            Specific bands to download (e.g., ["B8"] for panchromatic), matched
            against each scene's available product names. If omitted, the full
            product bundle is downloaded (which already contains the MTL/ANG
            metadata files). A band with no matching per-band product for a
            given dataset is skipped with a warning (not every Landsat dataset
            exposes every band as a separate download).
        download_metadata : bool, default True
            When downloading specific ``bands`` (a partial scene), also fetch
            the scene's ``MTL.txt`` (product metadata) and ``ANG.txt`` (angle
            coefficients) sidecar files if the dataset exposes them as separate
            downloads. Ignored when ``bands`` is None, because the full bundle
            already bundles those files. Best-effort: skipped with a warning if
            not offered separately.
        show_progress : bool, default True
            Show progress bar.
        poll_interval_s : float, default 5.0
            Seconds to wait between ``download-retrieve`` polls.
        max_poll_attempts : int, default 12
            Max polls per collection batch before giving up (default: 1 minute).
        keep_original_names : bool, default False
            Save each file under the name USGS delivers it as (e.g.
            ``LC08_L1TP_232083_20230301_20230307_02_T1_B8.TIF``) instead of the
            default ``<display_id>_<date>_<product_tag>`` rewrite. Use this when
            building a regional archive that must stay traceable to the catalog;
            the GMC thumb convention is applied later by
            :meth:`~geomulticorr.core.session.Session.sieve_bulk`.
        organize_by : {"sensor_pathrow"}, optional
            Directory layout under *output_dir*. ``None`` (default) keeps the
            historical one-directory-per-collection-key layout.
            ``"sensor_pathrow"`` writes to ``<PLATFORM>/<path>_<row>/``, e.g.
            ``LANDSAT8/232_083/`` — note the platform comes from the product id,
            not the collection key, because Landsat 8 and 9 share the single
            dataset ``landsat_ot_c2_l1``.
        max_files_per_request : int, default 100
            Cap on entries per ``download-request`` call. Counted in **files,
            not scenes**: one scene with ``bands=["B8"]`` and
            ``download_metadata=True`` is four entries, so 100 scenes would
            otherwise be a single 400-entry request. Oversized requests are the
            main cause of ``download-request`` timeouts.

        Returns
        -------
        list[Path]
            Paths to downloaded files.

        Notes
        -----
        A ``download-request`` that times out client-side is **not** cancelled
        server-side, and the quota is spent regardless. This method therefore
        falls back to polling ``download-retrieve`` for the batch label rather
        than abandoning the batch. Anything still unresolved when the polls run
        out is reported with the label, so it can be collected later with
        :meth:`resume_label` without paying for it twice.

        Raises
        ------
        ValueError
            If *organize_by* is not a recognised scheme.
        """
        if organize_by is not None and organize_by not in ORGANIZE_SCHEMES:
            raise ValueError(
                f"organize_by must be one of {ORGANIZE_SCHEMES} or None, got {organize_by!r}"
            )
        if not self.client.session_token:
            self.client.login()

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        downloaded_paths: List[Path] = []

        for collection_key, group in images_gdf.groupby("collection_key"):
            coll_info = LANDSAT_COLLECTIONS[collection_key]
            api_id = coll_info["api_id"]
            coll_dir = out_root / collection_key
            coll_dir.mkdir(parents=True, exist_ok=True)

            entity_ids = list(group["image_id"])
            try:
                options = self.client.download_options(api_id, entity_ids)
            except Exception as e:
                LOGGER.error(f"download_options failed for '{collection_key}': {e}")
                continue

            downloads: List[Dict[str, str]] = []
            # entityId -> (scene display_id, date_str, product_name, delivered
            # filename), for naming and filing each download once
            # download-request / -retrieve resolve its URL.
            #
            # Keyed on entityId ALONE: the download-request response echoes
            # entityId but carries no productId, so a compound key silently
            # misses every lookup and every file lands in the fallback
            # directory. entityId is unique per file — each band and sidecar has
            # its own ("…_B8_TIF", "…_MTL_TXT") — so it is sufficient.
            pre_meta: Dict[str, Tuple[str, str, str, Optional[str]]] = {}
            for row in group.itertuples(index=False):
                entity_id = row.image_id
                matched = self._match_products(options, entity_id, bands)
                if bands and download_metadata:
                    matched += self._match_metadata(options, entity_id)
                for opt in matched:
                    product_id = opt.get("id")
                    if not product_id:
                        continue
                    # Per-file entries carry their own entityId; the scene's own
                    # id addresses the whole bundle, not one band.
                    opt_entity_id = opt.get("entityId") or entity_id
                    downloads.append(
                        {"entityId": opt_entity_id, "productId": product_id}
                    )
                    date_str = row.acquisition_date.strftime("%Y-%m-%d")
                    pre_meta[opt_entity_id] = (
                        row.display_id,
                        date_str,
                        opt.get("productName", "product"),
                        opt.get("displayId"),
                    )

            if not downloads:
                LOGGER.warning(
                    f"No downloadable products matched for collection '{collection_key}'"
                )
                continue

            label = f"gmc_{collection_key}"
            # Chunk by *file* count, not scene count. One scene with bands +
            # metadata is four entries, so a 100-scene group is a 400-entry
            # request — enough to push download-request past any timeout.
            chunks = [
                downloads[i : i + max_files_per_request]
                for i in range(0, len(downloads), max_files_per_request)
            ]
            if len(chunks) > 1:
                LOGGER.info(
                    "%s: %d files -> %d download-request chunks of <=%d",
                    collection_key,
                    len(downloads),
                    len(chunks),
                    max_files_per_request,
                )

            ready: List[Dict[str, Any]] = []
            for chunk_no, chunk in enumerate(chunks, start=1):
                preparing: List[Dict[str, Any]] = []
                recovering = False
                try:
                    result = self.client.download_request(chunk, label=label)
                    ready.extend(result.get("availableDownloads", []))
                    preparing = list(result.get("preparingDownloads", []))
                except Exception as e:
                    # A timeout here does NOT cancel the work: M2M keeps staging
                    # and the quota is spent either way. Losing the response only
                    # loses the URLs, and those are recoverable through the label.
                    LOGGER.warning(
                        "download_request chunk %d/%d for '%s' did not return (%s). "
                        "The server may have staged it anyway — recovering via "
                        "label %r rather than discarding the request.",
                        chunk_no,
                        len(chunks),
                        collection_key,
                        e,
                        label,
                    )
                    recovering = True

                attempt = 0
                idle_polls = 0
                while attempt < max_poll_attempts and (preparing or recovering):
                    _time.sleep(poll_interval_s)
                    try:
                        retrieved = self.client.download_retrieve(label=label)
                    except Exception as e:
                        LOGGER.warning("download_retrieve(%r) failed: %s", label, e)
                        attempt += 1
                        continue
                    newly_ready = retrieved.get("available", []) or []
                    if newly_ready:
                        ready.extend(newly_ready)
                        idle_polls = 0
                        recovering = False
                    else:
                        idle_polls += 1
                    ready_ids = {d.get("downloadId") for d in newly_ready}
                    preparing = [
                        d for d in preparing if d.get("downloadId") not in ready_ids
                    ]
                    attempt += 1
                    # Two quiet polls in a row while nothing is known to be
                    # pending: the queue has drained.
                    if not preparing and idle_polls >= 2:
                        break

                if preparing or recovering:
                    LOGGER.warning(
                        "%s chunk %d/%d: %d download(s) unresolved after %d polls. "
                        "The batch is in the account queue — collect it with "
                        "finder.resume_label(%r, output_dir=...), which reuses the "
                        "queue as a worklist and skips scene-search and "
                        "download-options.",
                        collection_key,
                        chunk_no,
                        len(chunks),
                        len(preparing),
                        max_poll_attempts,
                        label,
                    )

            downloaded_paths.extend(
                self._write_downloads(
                    ready,
                    pre_meta,
                    out_root=out_root,
                    default_dir=coll_dir,
                    keep_original_names=keep_original_names,
                    organize_by=organize_by,
                    show_progress=show_progress,
                    desc=f"Downloading {collection_key}",
                )
            )

        return downloaded_paths

    def _write_downloads(
        self,
        ready: List[Dict[str, Any]],
        pre_meta: Dict[str, Tuple[str, str, str, Optional[str]]],
        *,
        out_root: Path,
        default_dir: Path,
        keep_original_names: bool,
        organize_by: Optional[str],
        show_progress: bool,
        desc: str,
    ) -> List[Path]:
        """Fetch every resolved URL, naming and filing each file.

        Shared by :meth:`download_images` and :meth:`resume_label` so a recovered
        batch lands in exactly the same layout as a first-pass one.
        """
        written: List[Path] = []
        iterator = (
            _tqdm(ready, desc=desc, unit="file")
            if (_HAS_TQDM and show_progress)
            else ready
        )
        for dl in iterator:
            url = dl.get("url")
            if not url:
                continue
            meta = pre_meta.get(dl.get("entityId"))
            delivered_name = None
            if meta:
                display_id, date_str, product_name, delivered_name = meta
                tag = re.sub(r"[^A-Za-z0-9]+", "_", product_name).strip("_")
                filename = f"{display_id}_{date_str}_{tag}"
            else:
                display_id = None
                filename = f"download_{dl.get('downloadId', 'unknown')}"
            ext = Path(url.split("?", 1)[0]).suffix or ".tar.gz"

            # Destination directory. With organize_by, the platform comes
            # from the product id rather than the collection key, so Landsat
            # 8 and 9 separate correctly despite sharing one dataset.
            dest_dir = default_dir
            if organize_by == ORGANIZE_SENSOR_PATHROW and display_id:
                sensor_dir, grid_dir = landsat_dirs(display_id)
                dest_dir = out_root / sensor_dir / grid_dir
                dest_dir.mkdir(parents=True, exist_ok=True)

            # The catalogue already told us the delivered filename in the
            # option's displayId — more reliable than anything the transfer
            # reports, and known before the request is made.
            fallback_name = (
                delivered_name
                if (keep_original_names and delivered_name)
                else f"{filename}{ext}"
            )
            try:
                written.append(
                    self._download_file(
                        url,
                        dest_dir,
                        fallback_name=fallback_name,
                        keep_original_names=keep_original_names,
                    )
                )
            except Exception as e:
                LOGGER.error(f"Failed to download {url}: {e}")
        return written

    def resume_label(
        self,
        label: str,
        *,
        output_dir: Union[str, Path],
        keep_original_names: bool = True,
        organize_by: Optional[str] = ORGANIZE_SENSOR_PATHROW,
        show_progress: bool = True,
        poll_interval_s: float = 10.0,
        max_poll_attempts: int = 30,
        re_request: bool = True,
        max_files_per_request: int = 100,
    ) -> List[Path]:
        """Collect the downloads staged under *label*, reusing the queue as a worklist.

        A ``download-request`` that timed out client-side still ran to completion
        on the USGS side, so the batch exists in the account queue. Recovering it
        through the label skips the two slow, rate-limited steps — ``scene-search``
        and ``download-options`` — and goes straight to fetching.

        Two mechanisms, in order:

        1. ``download-retrieve``, which returns entries that are ready and **not
           yet retrieved**.
        2. For everything else, a fresh ``download-request`` built from the
           queued records' own ``entityId``/``productId``.

        Step 2 is usually the one that does the work. M2M issues a download URL
        exactly once: after that the record's status is ``Proxied`` and
        ``download-retrieve`` reports it under ``requested`` — with no URL —
        rather than ``available``. So a batch whose request response was lost to
        a timeout cannot simply be re-read; its URLs must be re-issued. That
        costs rolling download-quota slots, not money.

        Naming and layout come from each queued record's ``displayId``, which is
        the delivered filename and also carries the platform and WRS-2 path/row,
        so a resumed batch lands exactly where a first-pass one would.

        Parameters
        ----------
        label : str
            Batch label, ``"gmc_<collection_key>"`` for anything written by
            :meth:`download_images` — e.g. ``"gmc_L8_L1"``.
        output_dir : str | Path
            Archive root, the same one passed to :meth:`download_images`.
        keep_original_names, organize_by, show_progress
            As in :meth:`download_images`; the defaults here match the
            catalog-named archive layout.
        poll_interval_s, max_poll_attempts
            Pacing for ``download-retrieve``.
        re_request : bool, default True
            Re-issue URLs for queued records that ``download-retrieve`` will no
            longer hand out. Set ``False`` to collect only what is still
            ``available`` and spend nothing.
        max_files_per_request : int, default 100
            Chunk size for the re-request, in files.

        Returns
        -------
        list[pathlib.Path]
            Paths written. Empty when the queue holds nothing for this label.

        Examples
        --------
        >>> finder.resume_label("gmc_L8_L1", output_dir="downloads_regional")  # doctest: +SKIP
        """
        if organize_by is not None and organize_by not in ORGANIZE_SCHEMES:
            raise ValueError(
                f"organize_by must be one of {ORGANIZE_SCHEMES} or None, got {organize_by!r}"
            )
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        queued = self.client.download_search(label=label)
        if not queued:
            LOGGER.info("Nothing queued under label %r.", label)
            return []

        # Rebuild the naming table from the queue itself. A record's displayId is
        # the filename ("LC08_L1TP_232083_..._B8.TIF"), and it starts with the
        # scene's product id, which is all landsat_dirs() needs.
        pre_meta: Dict[str, Tuple[str, str, str, Optional[str]]] = {}
        for rec in queued:
            entity_id = rec.get("entityId")
            name = rec.get("displayId")
            if not entity_id or not name:
                continue
            pre_meta[entity_id] = (
                name,
                "",
                rec.get("productName", "product"),
                name,
            )
        LOGGER.info("%d record(s) queued under label %r.", len(queued), label)

        ready: List[Dict[str, Any]] = []
        seen: set = set()
        idle_polls = 0
        for _attempt in range(max_poll_attempts):
            try:
                retrieved = self.client.download_retrieve(label=label)
            except Exception as e:
                LOGGER.warning("download_retrieve(%r) failed: %s", label, e)
                _time.sleep(poll_interval_s)
                continue
            newly = [
                d
                for d in (retrieved.get("available") or [])
                if d.get("downloadId") not in seen
            ]
            for d in newly:
                seen.add(d.get("downloadId"))
            ready.extend(newly)
            idle_polls = 0 if newly else idle_polls + 1
            if len(ready) >= len(queued) or idle_polls >= 2:
                break
            _time.sleep(poll_interval_s)

        # Anything download-retrieve will not hand out again: its URL was issued
        # once (status "Proxied") and the response was lost. Re-issue it from the
        # queued record's own ids — no search, no download-options.
        have = {d.get("entityId") for d in ready}
        missing = [
            rec
            for rec in queued
            if rec.get("entityId") not in have and rec.get("productId")
        ]
        if missing and re_request:
            LOGGER.info(
                "%d record(s) under %r are no longer retrievable (URL already "
                "issued); re-requesting them from the queue.",
                len(missing),
                label,
            )
            chunks = [
                missing[i : i + max_files_per_request]
                for i in range(0, len(missing), max_files_per_request)
            ]
            for chunk_no, chunk in enumerate(chunks, start=1):
                payload = [
                    {"entityId": rec["entityId"], "productId": rec["productId"]}
                    for rec in chunk
                ]
                try:
                    result = self.client.download_request(payload, label=label)
                except Exception as e:
                    LOGGER.warning(
                        "re-request chunk %d/%d failed: %s", chunk_no, len(chunks), e
                    )
                    continue
                ready.extend(result.get("availableDownloads") or [])
                still = result.get("preparingDownloads") or []
                for _ in range(max_poll_attempts):
                    if not still:
                        break
                    _time.sleep(poll_interval_s)
                    try:
                        got = self.client.download_retrieve(label=label)
                    except Exception as e:
                        LOGGER.warning("download_retrieve(%r) failed: %s", label, e)
                        continue
                    newly = got.get("available") or []
                    ready.extend(newly)
                    done = {d.get("downloadId") for d in newly}
                    still = [d for d in still if d.get("downloadId") not in done]
        elif missing:
            LOGGER.warning(
                "%d record(s) under %r are queued but their URLs were already "
                "issued, so download-retrieve will not return them. Pass "
                "re_request=True to re-issue.",
                len(missing),
                label,
            )

        if not ready:
            LOGGER.warning("Label %r yielded no fetchable URLs.", label)
            return []

        return self._write_downloads(
            ready,
            pre_meta,
            out_root=out_root,
            default_dir=out_root / label,
            keep_original_names=keep_original_names,
            organize_by=organize_by,
            show_progress=show_progress,
            desc=f"Resuming {label}",
        )

    @staticmethod
    def _scene_files(
        options: List[Dict[str, Any]], entity_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Per-file download entries for one scene, keyed by exact filename.

        Individual band and metadata files are **not** top-level options: M2M
        nests them in each parent option's ``secondaryDownloads``. Every one of
        them reports the same generic ``productName`` — "Landsat Collection 2
        Level-1 Band File" — so the band can only be identified from
        ``displayId``, which is the delivered filename::

            displayId  LC09_L1TP_232083_20231221_20231223_02_T1_B8.TIF
            entityId   L1_LC09_L1TP_232083_20231221_20231223_02_T1_B8_TIF
            id         5e8206bf6bc4a163      (shared by every band file)

        Both fields are needed for the download request, and note ``id`` alone
        does not identify a file — it is the *product type*, identical across
        bands, so requests must pair it with the secondary's own ``entityId``.

        Several parent options carry byte-identical secondary lists (a scene
        typically exposes two "Band File" parents plus a bundle), hence keying
        by filename: without it every file is requested two or three times.
        """
        files: Dict[str, Dict[str, Any]] = {}
        for parent in options:
            if parent.get("entityId") != entity_id:
                continue
            for sec in parent.get("secondaryDownloads") or []:
                name = sec.get("displayId")
                if not name or not sec.get("available"):
                    continue
                files.setdefault(name, sec)
        return files

    @staticmethod
    def _match_products(
        options: List[Dict[str, Any]], entity_id: str, bands: Optional[Sequence[str]]
    ) -> List[Dict[str, Any]]:
        """Pick download entries for one scene: the bundle, or per-band files.

        With *bands*, matching is on the ``displayId`` filename suffix
        (``…_B8.TIF``) of the nested per-file entries, because the product name
        is generic across every band. Suffix matching also keeps ``B1`` from
        matching ``B10``/``B11``.

        Any requested band with no match is skipped with a warning, not fatal.
        """
        entity_options = [
            o for o in options if o.get("entityId") == entity_id and o.get("available")
        ]
        if not entity_options:
            LOGGER.warning(f"No available download options for entity {entity_id}")
            return []

        if not bands:
            for opt in entity_options:
                if "bundle" in (opt.get("productName") or "").lower():
                    return [opt]
            return entity_options[
                :1
            ]  # no explicit bundle: take the first available product

        files = M2MImageFinder._scene_files(options, entity_id)
        matched: List[Dict[str, Any]] = []
        for band in bands:
            token = str(band).upper().removesuffix(".TIF")
            hit = next(
                (
                    entry
                    for name, entry in files.items()
                    if name.upper().removesuffix(".TIF").endswith(f"_{token}")
                ),
                None,
            )
            if hit is None:
                LOGGER.warning(
                    f"No {band!r} file for entity {entity_id}; "
                    f"available: {sorted(files)}"
                )
                continue
            matched.append(hit)
        return matched

    @staticmethod
    def _match_metadata(
        options: List[Dict[str, Any]], entity_id: str
    ) -> List[Dict[str, Any]]:
        """The scene's MTL / ANG metadata sidecars, if the dataset exposes them.

        Matched on the ``displayId`` filename of the nested per-file entries —
        `…_MTL.txt`, `…_MTL.xml`, `…_ANG.txt` — since their ``productName`` is
        the same generic "Band File" string as the imagery. Returns an empty
        list (not an error) when a dataset offers only whole bundles.
        """
        pattern = re.compile(r"_(?:MTL|ANG)\.(?:txt|xml)$", re.IGNORECASE)
        return [
            entry
            for name, entry in M2MImageFinder._scene_files(options, entity_id).items()
            if pattern.search(name)
        ]

    @staticmethod
    def _download_file(
        url: str,
        out_dir: Union[str, Path],
        fallback_name: str,
        keep_original_names: bool = False,
        chunk_size: int = 1 << 20,
    ) -> Path:
        """Download a file from URL with chunking, atomically.

        The destination filename is resolved from the *same* response that
        carries the bytes. M2M download URLs are single-use redirects, so a HEAD
        request to read the name first would consume the URL and leave nothing
        to download.

        Resolution order when *keep_original_names* is set:
        ``Content-Disposition`` -> URL basename (only if it has a suffix, since
        these URLs are frequently extensionless) -> *fallback_name*.

        Parameters
        ----------
        url : str
            Ready download URL from ``download-request`` / ``download-retrieve``.
        out_dir : str | Path
            Destination directory; created if missing.
        fallback_name : str
            Filename to use when the server reports none, and the only name used
            when *keep_original_names* is False.
        keep_original_names : bool, default False
            Prefer the server-reported filename over *fallback_name*.
        chunk_size : int
            Streaming chunk size in bytes.

        Returns
        -------
        pathlib.Path
            Path actually written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with urlopen(url, timeout=300) as resp:
            name = fallback_name
            if keep_original_names:
                name = (
                    filename_from_content_disposition(
                        resp.headers.get("Content-Disposition")
                    )
                    or _url_basename(url)
                    or fallback_name
                )
            out_path = out_dir / safe_filename(name, fallback=fallback_name)
            tmp_path = out_path.with_name(f"{out_path.name}.part")
            LOGGER.info(f"Downloading {out_path.name}")
            try:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                os.replace(tmp_path, out_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
        return out_path


# Import pandas late to avoid issues if not installed
import pandas as pd
