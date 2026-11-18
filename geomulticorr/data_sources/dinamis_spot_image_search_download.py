#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# dinamis_spot_image_search_download.py
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
"""Search and bulk-download SPOT 6/7 imagery from the DINAMIS STAC catalog.

DINAMIS (via the Théia ``api.stac.teledetection.fr`` STAC endpoint) hosts the
national SPOT 6/7 ortho coverage (collection ``dinamis-openspot-6-7``):
metropolitan France and Corsica, 2014-06 onwards (2025-10 at the time of
writing; the extent grows as new years are ingested). Assets are
Cloud-Optimized GeoTIFFs on MinIO object storage plus DIMAP XML metadata and
GML cloud/ROI masks.

The catalogue was restructured in 2026: the former ``spot-6-7-drs`` collection
(2017-2023, asset keys ``src_pan``/``src_xs``/``dimap_pan``/…) was removed and
its content re-published under ``dinamis-openspot-6-7`` with the asset keys
``PAN``/``MS``/``PAN-dimap``/``MS-dimap``/``cld-mask``/``roi-mask``/``QL``/
``mask``/``lineage``. The old key names are still accepted everywhere as
aliases (:data:`_ASSET_ALIASES`), so existing calls keep working. A search
against a collection that no longer exists raises instead of silently
returning nothing (pystac does the latter).

Two download modes, mirroring :mod:`geomulticorr.data_sources.mpc_image_search_download`:

* :meth:`DinamisImageFinder.download_images` — stream full-extent asset files
  (panchromatic 1.5 m and/or multispectral 6 m), renamed via the
  ``filename_template`` parameter (default ``IMG_S6P_201707261036189.tif``).
* :meth:`DinamisImageFinder.download_cog_clip` — windowed COG read: only the
  AOI window is fetched over HTTP, so an AOI-scoped analysis never pays for a
  full 400-800 MB scene.

Authentication
--------------
Requires a free DINAMIS account (https://dinamis.data-terra.org). Provide an
access/secret key pair via ``access_key``/``secret_key`` or the
``MDT_SPOT67_STAC_ACCESS_KEY`` / ``MDT_SPOT67_STAC_SECRET_KEY`` environment
variables (e.g. loaded from a ``.env`` file with ``python-dotenv`` — not a
dependency of this module, load it in the calling notebook/script).

**The key pair authenticates the STAC API only.** Assets live on MinIO object
storage, which ignores those headers and requires a *presigned* URL carrying
``X-Amz-Signature`` query parameters, issued by the signing endpoint
``https://signing.stac.teledetection.fr``. Sending the keys as headers to an
asset URL returns ``403 Forbidden``. Every download path here therefore signs
its href through the ``teledetection`` SDK (``pip install teledetection``)
immediately before the request — never at search time, because presigned URLs
expire and a bulk run over multi-GB scenes would start failing partway through.
The SDK caches each signature and re-signs when less than 30 minutes of validity
remain.

Output naming
-------------
Downloads are named by :data:`DEFAULT_FILENAME_TEMPLATE`, the Airbus-style
``IMG_<mission><sensor>_<acquisition stamp>`` convention::

    IMG_S7P_201806231000132.tif            # panchromatic scene
    IMG_S7MS_201806231000132.tif           # multispectral scene
    IMG_S7P_201806231000132_metadata.xml   # its DIMAP sibling
    IMG_S7P_201806231000132_SCENE2.tif     # 2nd scene cut from the same strip

The acquisition stamp is **not** unique per item: one strip is cut into
``SCENE1``, ``SCENE2``… items that all carry it. ``{scene_part}`` is empty for
``SCENE1`` (so single-scene names are unchanged) and ``_SCENE<n>`` otherwise.
The download methods refuse to start when two assets of one call would render
to the same filename, since the second would overwrite (or be skipped as
"existing") without a trace.

``DinamisImageFinder(..., filename_template=...)`` overrides it with any
``str.format`` template over the fields listed in :func:`_name_fields`, so a
local convention needs no edit to this module.
:data:`LEGACY_FILENAME_TEMPLATE` restores the pre-0.4.3 date-based names.

Use :meth:`DinamisImageFinder.preview_filenames` to check a template before
paying for a download, and :meth:`DinamisImageFinder.name_fields` to list what a
given asset can reference.

Note that DINAMIS ships **one mosaicked COG per scene**. The ``R1C1``/``R3C4``
tile indices in the original Airbus DIMAP delivery (12 tiles for a typical pan
scene) do not survive that mosaicking, so no field here can reproduce them; a
template wanting that token must spell it out as literal text.
"""

from __future__ import annotations

import logging as _logging
import os as _os
import re as _re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd
import pystac_client
import rasterio
import requests
from pyproj import Geod as _Geod
from rasterio.mask import mask as _rio_mask
from rasterio.warp import transform_geom as _transform_geom
from shapely.geometry import box as _box
from shapely.geometry import mapping as _mapping
from shapely.geometry import shape as _shape
from shapely.geometry.base import BaseGeometry as _BaseGeometry

# Optional progress bar
try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

LOGGER = _logging.getLogger("dinamis_spot_image_search_download")

DINAMIS_STAC_URL = "https://api.stac.teledetection.fr"
SPOT67_COLLECTION = "dinamis-openspot-6-7"

#: pre-2026 asset key (``spot-6-7-drs``) -> current key (``dinamis-openspot-6-7``).
#: Both spellings are accepted; :func:`_resolve_asset_key` picks whichever the
#: item actually carries.
_ASSET_ALIASES = {
    "src_pan": "PAN",
    "src_xs": "MS",
    "dimap_pan": "PAN-dimap",
    "dimap_xs": "MS-dimap",
    "cld_msk_vec_pan": "cld-mask",
    "cld_msk_vec_xs": "cld-mask",
    "roi_msk_vec_pan": "roi-mask",
    "roi_msk_vec_xs": "roi-mask",
    "rendered_preview": "QL",
}

#: file extension for each STAC asset key
_ASSET_EXTENSIONS = {
    "PAN": "tif",
    "MS": "tif",
    "PAN-dimap": "xml",
    "MS-dimap": "xml",
    "cld-mask": "gml",
    "roi-mask": "gml",
    "QL": "jpg",
    "mask": "zip",
    "lineage": "zip",
    "src_pan": "tif",
    "src_xs": "tif",
    "dimap_pan": "xml",
    "dimap_xs": "xml",
    "cld_msk_vec_pan": "gpkg",
    "cld_msk_vec_xs": "gpkg",
    "roi_msk_vec_pan": "gpkg",
    "roi_msk_vec_xs": "gpkg",
    "rendered_preview": "png",
}
#: sensor label (Panchromatic / Multispectral) for each STAC asset key
_ASSET_SENSOR = {
    "PAN": "P",
    "PAN-dimap": "P",
    "MS": "MS",
    "MS-dimap": "MS",
    "src_pan": "P",
    "dimap_pan": "P",
    "cld_msk_vec_pan": "P",
    "roi_msk_vec_pan": "P",
    "src_xs": "MS",
    "dimap_xs": "MS",
    "cld_msk_vec_xs": "MS",
    "roi_msk_vec_xs": "MS",
}
#: filename suffix for non-primary assets. Every non-image asset needs a distinct
#: one: the cloud and ROI masks are both .gpkg and would otherwise collide.
_ASSET_SUFFIX = {
    "PAN-dimap": "_metadata",
    "MS-dimap": "_metadata",
    "cld-mask": "_cloudmask",
    "roi-mask": "_roimask",
    "QL": "_preview",
    "mask": "_masks",
    "lineage": "_lineage",
    "dimap_pan": "_metadata",
    "dimap_xs": "_metadata",
    "cld_msk_vec_pan": "_cloudmask",
    "cld_msk_vec_xs": "_cloudmask",
    "roi_msk_vec_pan": "_roimask",
    "roi_msk_vec_xs": "_roimask",
    "rendered_preview": "_preview",
}
#: image asset -> its DIMAP metadata asset
_METADATA_ASSET_FOR = {
    "PAN": "PAN-dimap",
    "MS": "MS-dimap",
    "src_pan": "dimap_pan",
    "src_xs": "dimap_xs",
}

#: environment variables the ``teledetection`` SDK reads for its API key pair
_TLD_ACCESS_KEY_ENV = "TLD_ACCESS_KEY"
_TLD_SECRET_KEY_ENV = "TLD_SECRET_KEY"

#: default output naming: ``IMG_S6P_201707261036189.tif`` — the Airbus-style
#: ``IMG_<mission><sensor>_<acquisition stamp>`` convention. ``{suffix}`` keeps
#: the metadata/mask siblings from colliding with the image they describe.
DEFAULT_FILENAME_TEMPLATE = "IMG_{sensor_tag}_{stamp}{scene_part}{suffix}.{ext}"

#: the pre-0.4.3 convention, kept for projects already named that way
LEGACY_FILENAME_TEMPLATE = "IMG_SPOT{platform_num}{sensor_part}_{date}{suffix}.{ext}"

#: DINAMIS asset basenames follow the Airbus DIMAP grammar, e.g.
#: ``COG_SPOT7_P_201806231000132_ORT_SPOT7_20180705_0813381atzn5pa9ztdm_1.tif``
#: (role, mission, spectral mode, acquisition stamp, geometric level, job id).
_ASSET_NAME_RE = _re.compile(
    r"^(?P<role>[A-Z]+)_"
    r"(?P<mission>SPOT\d)_"
    r"(?P<sensor>MS|P)_"
    r"(?P<stamp>\d+)_"
    r"(?P<geom_level>[A-Z0-9]+)_"
    r"(?P<job>.+?)"
    r"(?:_MSK)?(?:_preview)?"
    r"\.[A-Za-z]+$"
)

#: item ids of ``dinamis-openspot-6-7``, e.g.
#: ``SPOT67_FRANCE_202411171007349_GS1_E006N46_SCENE1_ORTHO-HPQ``. The stamp is
#: shared by every scene cut from one strip; ``(stamp, scene)`` is unique.
_ITEM_ID_RE = _re.compile(r"_(?P<stamp>\d{15})_.*?(?:_(?P<scene>SCENE\d+)_|$)")

#: search_metadata column -> STAC property (acquisition geometry, which matters
#: for correlation: off-nadir incidence × DEM error is a per-image artefact)
_VIEW_COLUMNS = {
    "snow_percent": "eo:snow_cover",
    "off_nadir_deg": "view:off_nadir",
    "incidence_deg": "view:incidence_angle",
    "sun_azimuth_deg": "view:sun_azimuth",
    "sun_elevation_deg": "view:sun_elevation",
}

_WGS84_GEOD = _Geod(ellps="WGS84")


def _resolve_asset_key(item: dict, asset_key: str) -> str:
    """Return the spelling of ``asset_key`` that ``item`` actually carries.

    Accepts the current keys (``PAN``) and the pre-2026 ones (``src_pan``) in
    both directions; a key the item has under neither spelling is returned
    unchanged, so the caller's "asset not found" path still reports what was
    asked for.
    """
    assets = item.get("assets", {})
    if asset_key in assets:
        return asset_key
    current = _ASSET_ALIASES.get(asset_key)
    if current and current in assets:
        return current
    for legacy, cur in _ASSET_ALIASES.items():
        if cur == asset_key and legacy in assets:
            return legacy
    return asset_key


def _mission_num(props: Mapping) -> str:
    """SPOT mission number (``"6"``/``"7"``) from STAC properties.

    ``dinamis-openspot-6-7`` reports ``platform: "SPOT"`` plus ``mission: "6"``;
    the removed ``spot-6-7-drs`` reported ``platform: "SPOT-6"``.
    """
    mission = props.get("mission")
    if mission not in (None, ""):
        return str(mission)
    return str(props.get("platform", "spot")).upper().split("-")[-1]


def _platform_label(props: Mapping) -> Optional[str]:
    """``"spot-6"``/``"spot-7"`` whichever catalogue generation the item comes from."""
    num = _mission_num(props)
    return f"spot-{num}" if num.isdigit() else props.get("platform")


def _geodesic_area_km2(geom: _BaseGeometry) -> float:
    """Return the geodesic area (km²) of a WGS84 shapely geometry."""
    area, _ = _WGS84_GEOD.geometry_area_perimeter(geom)
    return abs(area) / 1e6


def _name_fields(item: dict, asset_key: str) -> Dict[str, str]:
    """Return every field available for building an output filename.

    Two sources are merged. The STAC item carries almost nothing — only
    ``datetime``, ``platform`` and ``instruments`` — so the richer identifiers
    are recovered by parsing the asset's own basename, which follows the Airbus
    DIMAP grammar. Fields that cannot be recovered come back as empty strings
    rather than raising, so a template never explodes mid-download.

    Returns
    -------
    dict[str, str]
        ``mission`` (``SPOT7``), ``mission_short`` (``S7``), ``platform_num``
        (``7``), ``sensor`` (``P``/``MS``), ``sensor_tag`` (``S7P``),
        ``sensor_part`` (``_P``, or ``""`` for assets with no sensor),
        ``date`` (``20180623``), ``year``, ``month``, ``day``, ``time``
        (``1000132``), ``stamp`` (``201806231000132`` — the full acquisition
        timestamp Airbus uses as a scene identifier), ``geom_level`` (``ORT``),
        ``job`` (the production job id), ``scene`` (``SCENE2``, ``""`` when
        the item id has none), ``scene_part`` (``""`` for ``SCENE1``,
        ``_SCENE<n>`` otherwise), ``item_id``, ``asset`` (the STAC asset key
        the item carries), ``suffix`` (``_metadata``/``_cloudmask``/``_preview``)
        and ``ext``.
    """
    asset_key = _resolve_asset_key(item, asset_key)
    props = item.get("properties", {})
    dt = pd.to_datetime(props.get("datetime"))

    href = item.get("assets", {}).get(asset_key, {}).get("href", "")
    basename = _os.path.basename(href.split("?")[0])
    parsed = _ASSET_NAME_RE.match(basename)
    grp = parsed.groupdict() if parsed else {}

    mission = grp.get("mission") or f"SPOT{_mission_num(props)}"
    platform_num = mission[len("SPOT") :]
    sensor = _ASSET_SENSOR.get(asset_key) or grp.get("sensor") or ""
    mission_short = f"S{platform_num}"

    id_match = _ITEM_ID_RE.search(item.get("id", ""))
    id_grp = id_match.groupdict() if id_match else {}
    scene = id_grp.get("scene") or ""

    # Sidecars such as the MASKS/LINEAGE zips are named after the item, not the
    # Airbus product, so the stamp comes from the item id — exact, unlike the
    # STAC datetime, which this collection rounds to midnight.
    stamp = grp.get("stamp") or id_grp.get("stamp") or ""
    if not stamp:
        # The default template keys on {stamp}; leaving it empty would collapse
        # every scene of a campaign onto one filename and silently overwrite.
        # The STAC datetime is a near-miss (it reports the MS acquisition, while
        # a pan asset stamps its own, seconds apart) but it is unique per scene.
        stamp = dt.strftime("%Y%m%d%H%M%S") if dt is not None else ""
        LOGGER.warning(
            "could not parse an acquisition stamp from asset %r of item %s "
            "(basename %r); falling back to the STAC datetime %r",
            asset_key,
            item.get("id", "?"),
            basename,
            stamp,
        )

    return {
        "mission": mission,
        "mission_short": mission_short,
        "platform_num": platform_num,
        "sensor": sensor,
        "sensor_tag": f"{mission_short}{sensor}",
        "sensor_part": f"_{sensor}" if sensor else "",
        "date": dt.strftime("%Y%m%d") if dt is not None else "",
        "year": dt.strftime("%Y") if dt is not None else "",
        "month": dt.strftime("%m") if dt is not None else "",
        "day": dt.strftime("%d") if dt is not None else "",
        "time": stamp[8:] if len(stamp) > 8 else "",
        "stamp": stamp,
        "geom_level": grp.get("geom_level", ""),
        "job": grp.get("job", ""),
        "scene": scene,
        "scene_part": f"_{scene}" if scene and scene != "SCENE1" else "",
        "item_id": item.get("id", ""),
        "asset": asset_key,
        "suffix": _ASSET_SUFFIX.get(asset_key, ""),
        "ext": _ASSET_EXTENSIONS.get(asset_key, "bin"),
    }


def _sign_href(href: str) -> str:
    """Return a presigned MinIO URL for ``href``, valid for a limited time.

    The DINAMIS key pair authenticates the STAC API but *not* the object store:
    an unsigned asset URL returns ``403 Forbidden`` no matter what headers are
    attached. Signing mints the ``X-Amz-Signature`` query parameters MinIO
    actually checks.

    Call this immediately before each request rather than caching the result:
    signatures expire, and the SDK already caches them internally (re-signing
    when under 30 minutes of validity remain). Signing an already-signed URL is
    a no-op, and URLs outside the DINAMIS storage domain are returned unchanged.

    ``teledetection`` is imported lazily so that :mod:`geomulticorr.data_sources`
    still imports — and :meth:`DinamisImageFinder.search_metadata` still runs —
    in an environment where the SDK is not installed.
    """
    try:
        import teledetection as _tld
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Downloading DINAMIS assets requires the 'teledetection' SDK, which "
            "mints the presigned MinIO URLs the object store demands "
            "(the API key pair alone yields 403 Forbidden). "
            "Install it with: pip install teledetection"
        ) from exc
    return _tld.sign(href)


def _coerce_to_wgs84_geometry(
    aoi: Union[gpd.GeoDataFrame, _BaseGeometry, Mapping, Sequence[float]],
) -> _BaseGeometry:
    """Coerce AOI input (GeoDataFrame / shapely / GeoJSON-like / bbox) to a WGS84 shapely geometry."""
    if isinstance(aoi, gpd.GeoDataFrame):
        gdf = aoi
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf.union_all()
    if isinstance(aoi, _BaseGeometry):
        return aoi
    if isinstance(aoi, Mapping):
        return _shape(aoi)
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        return _box(*aoi)
    raise TypeError(
        "AOI must be a GeoDataFrame, shapely geometry, GeoJSON-like mapping, "
        "or a (west, south, east, north) bbox"
    )


# -------------------------
# 1) Client (auth)
# -------------------------
class DinamisClient:
    """Authenticate against the DINAMIS/Théia STAC API.

    Parameters
    ----------
    access_key, secret_key : str, optional
        DINAMIS API key pair. Falls back to the ``MDT_SPOT67_STAC_ACCESS_KEY``
        / ``MDT_SPOT67_STAC_SECRET_KEY`` environment variables.
    stac_url : str
        STAC API endpoint (defaults to the public Théia endpoint).

    Notes
    -----
    Register a free account at https://dinamis.data-terra.org to obtain a key
    pair. These headers authenticate the **STAC search only** — asset downloads
    go to MinIO object storage, which rejects them with ``403 Forbidden`` and
    requires a presigned URL instead (see :func:`_sign_href`).

    Because the same key pair can serve both, the constructor exports it under
    the ``TLD_ACCESS_KEY`` / ``TLD_SECRET_KEY`` names the ``teledetection`` SDK
    reads, unless those are already set. Without them the SDK falls back to its
    OAuth2 device flow, which tries to open a browser — unusable in a script or
    on a cluster node.
    """

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        stac_url: str = DINAMIS_STAC_URL,
    ) -> None:
        self.access_key = access_key or _os.environ.get("MDT_SPOT67_STAC_ACCESS_KEY")
        self.secret_key = secret_key or _os.environ.get("MDT_SPOT67_STAC_SECRET_KEY")
        if not self.access_key or not self.secret_key:
            raise ValueError(
                "DINAMIS access/secret key not found. Pass access_key/secret_key, "
                "or set MDT_SPOT67_STAC_ACCESS_KEY / MDT_SPOT67_STAC_SECRET_KEY "
                "(e.g. in a .env file loaded with python-dotenv). "
                "Register a free account at https://dinamis.data-terra.org."
            )
        self.headers: Dict[str, str] = {
            "access-key": self.access_key,
            "secret-key": self.secret_key,
        }
        _os.environ.setdefault(_TLD_ACCESS_KEY_ENV, self.access_key)
        _os.environ.setdefault(_TLD_SECRET_KEY_ENV, self.secret_key)
        self.stac_url = stac_url
        self._stac = pystac_client.Client.open(stac_url, headers=self.headers)

    @property
    def stac(self) -> pystac_client.Client:
        """The underlying authenticated STAC client."""
        return self._stac

    @staticmethod
    def sign(href: str) -> str:
        """Presign a DINAMIS asset URL for direct use with requests/rasterio/GDAL.

        The download methods sign their hrefs already; this is for reaching an
        asset by hand, e.g. ``rasterio.open(client.sign(item["assets"][k]["href"]))``.
        """
        return _sign_href(href)


# -------------------------
# 2) Image search and download
# -------------------------
class DinamisImageFinder:
    """Search the SPOT 6/7 DINAMIS collection and bulk-download scenes.

    Parameters
    ----------
    client : DinamisClient
        Authenticated STAC client.
    aoi : GeoDataFrame | shapely geometry | GeoJSON-like mapping | (w, s, e, n)
        Area of interest. A GeoDataFrame is re-projected to WGS84 and dissolved.
    year_range : tuple[int | None, int | None], default (2014, None)
        Inclusive year range filter; ``None`` leaves that end open (the
        collection spans 2014-06 to the latest ingested year).
    month_range : tuple[int, int], default (1, 12)
        Inclusive month (1-12) range filter (applied client-side).
    cloud_max : float, default 100
        Maximum ``eo:cloud_cover`` percentage. Scenes with no reported cloud
        cover are always kept.
    collection : str, default "dinamis-openspot-6-7"
        STAC collection id.
    filename_template : str, optional
        ``str.format``-style template for downloaded filenames. Defaults to
        :data:`DEFAULT_FILENAME_TEMPLATE` (``IMG_S7P_201806231000132.tif``);
        pass :data:`LEGACY_FILENAME_TEMPLATE` for the older date-based names.
        See :meth:`name_fields` for the available placeholders and
        :meth:`preview_filenames` to check a template before downloading.
    """

    def __init__(
        self,
        client: DinamisClient,
        aoi: Union[gpd.GeoDataFrame, _BaseGeometry, Mapping, Sequence[float]],
        *,
        year_range: Tuple[Optional[int], Optional[int]] = (2014, None),
        month_range: Tuple[int, int] = (1, 12),
        cloud_max: float = 100.0,
        collection: str = SPOT67_COLLECTION,
        filename_template: str = DEFAULT_FILENAME_TEMPLATE,
    ) -> None:
        self.client = client
        self.collection = collection
        self.year_range = year_range
        self.month_range = month_range
        self.cloud_max = cloud_max
        self.filename_template = filename_template
        self.aoi_geometry: _BaseGeometry = _coerce_to_wgs84_geometry(aoi)
        self.bbox: Tuple[float, float, float, float] = tuple(self.aoi_geometry.bounds)
        # Raw STAC item dicts from the last search, keyed by item id.
        self._item_cache: Dict[str, dict] = {}

    @property
    def aoi_area_km2(self) -> float:
        """Geodesic area of the AOI in km²."""
        return _geodesic_area_km2(self.aoi_geometry)

    # ---- public API
    def search_metadata(
        self,
        *,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        limit: int = 100,
    ) -> gpd.GeoDataFrame:
        """Search the collection and build a GeoDataFrame of per-scene metadata.

        Parameters
        ----------
        date_start, date_end : str, optional
            ISO date strings (``"YYYY-MM-DD"``) overriding ``year_range``.
        limit : int, default 100
            Page size for the STAC search (results are paginated automatically).

        Returns
        -------
        geopandas.GeoDataFrame
            One row per scene: ``image_id``, ``acquisition_date``, ``platform``
            (``"spot-6"`` / ``"spot-7"``), ``cloud_percent``, ``snow_percent``,
            ``off_nadir_deg``, ``incidence_deg``, ``sun_azimuth_deg``,
            ``sun_elevation_deg`` (``None`` when the item does not report
            them), ``assets`` (list of available asset keys), ``geometry``
            (footprint, EPSG:4326). Sorted by date ascending.

        Raises
        ------
        ValueError
            If the search is empty because ``collection`` does not exist on the
            server. pystac returns zero items for an unknown collection, which
            is indistinguishable from "no scene here" — exactly how the 2026
            removal of ``spot-6-7-drs`` went unnoticed.
        """
        if date_start or date_end:
            datetime_range = f"{date_start or '..'}/{date_end or '..'}"
        else:
            y0, y1 = self.year_range
            start = f"{y0}-01-01" if y0 is not None else ".."
            end = f"{y1}-12-31" if y1 is not None else ".."
            datetime_range = f"{start}/{end}"

        search = self.client.stac.search(
            collections=[self.collection],
            bbox=self.bbox,
            datetime=datetime_range,
            limit=limit,
        )
        items = list(search.items_as_dicts())
        if not items:
            self._check_collection_exists()

        rows: List[dict] = []
        for it in items:
            props = it["properties"]
            dt = pd.to_datetime(props.get("datetime"))
            if dt is None or not (
                self.month_range[0] <= dt.month <= self.month_range[1]
            ):
                continue
            cloud = props.get("eo:cloud_cover")
            if cloud is not None and cloud > self.cloud_max:
                continue
            self._item_cache[it["id"]] = it
            rows.append(
                {
                    "image_id": it["id"],
                    "acquisition_date": dt,
                    "platform": _platform_label(props),
                    "cloud_percent": cloud,
                    **{col: props.get(key) for col, key in _VIEW_COLUMNS.items()},
                    "assets": list(it["assets"].keys()),
                    "geometry": _shape(it["geometry"]),
                }
            )

        if not rows:
            return gpd.GeoDataFrame(
                columns=[
                    "image_id",
                    "acquisition_date",
                    "platform",
                    "cloud_percent",
                    *_VIEW_COLUMNS,
                    "assets",
                    "geometry",
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        gdf = gdf.sort_values("acquisition_date").reset_index(drop=True)
        return gdf

    def download_images(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        output_dir: Union[str, Path],
        assets: Sequence[str] = ("PAN",),
        download_metadata: bool = True,
        max_images: Optional[int] = None,
        show_progress: bool = True,
        overwrite: bool = False,
        timeout: float = 300.0,
    ) -> List[Path]:
        """Stream full-extent asset files for the given scenes.

        Each asset URL is presigned immediately before its own request, so a run
        spanning many multi-hundred-MB scenes cannot outlive its signatures.

        Parameters
        ----------
        images_gdf : geopandas.GeoDataFrame
            Output of :meth:`search_metadata` (possibly filtered). Must contain
            ``image_id``.
        output_dir : str | Path
            Destination directory (flat — filenames already encode platform,
            sensor and date).
        assets : sequence[str], default ("PAN",)
            STAC asset keys to download. Common choices: ``"PAN"``
            (panchromatic, 1.5 m), ``"MS"`` (multispectral, 6 m), or both;
            ``("PAN-dimap",)`` with ``download_metadata=False`` fetches the
            metadata alone. The pre-2026 names (``"src_pan"``…) are accepted.
        download_metadata : bool, default True
            Also fetch each image asset's DIMAP XML metadata sibling.
        max_images : int, optional
            Limit the number of scenes processed.
        show_progress : bool, default True
            Show a tqdm progress bar when available.
        overwrite : bool, default False
            Re-download files that already exist on disk.
        timeout : float, default 300
            Per-request timeout in seconds (full scenes can be several hundred MB).

        Returns
        -------
        list[pathlib.Path]
            Paths of all files written.

        Raises
        ------
        ValueError
            If two assets would be written to the same filename (see
            :meth:`_plan_downloads`). Nothing is downloaded in that case.
        """
        if "image_id" not in images_gdf.columns:
            raise ValueError("images_gdf must contain an 'image_id' column")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = list(images_gdf.itertuples(index=False))
        if max_images is not None:
            rows = rows[: int(max_images)]

        def keys_for(item: dict) -> List[str]:
            keys = [_resolve_asset_key(item, k) for k in assets]
            if download_metadata:
                for key in list(keys):
                    meta_key = _METADATA_ASSET_FOR.get(key)
                    if meta_key:
                        keys.append(_resolve_asset_key(item, meta_key))
            return list(dict.fromkeys(keys))

        plan = self._plan_downloads(rows, keys_for, out_dir)
        iterator = (
            _tqdm(plan, desc="Scenes", unit="scene")
            if (_HAS_TQDM and show_progress)
            else plan
        )

        written: List[Path] = []
        for image_id, jobs in iterator:
            for href, out_path in jobs:
                if out_path.exists() and not overwrite:
                    LOGGER.info("skip existing %s", out_path)
                    written.append(out_path)
                    continue
                self._stream_download(href, out_path, timeout=timeout)
                written.append(out_path)

        return written

    def download_cog_clip(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        output_dir: Union[str, Path],
        assets: Sequence[str] = ("MS",),
        max_images: Optional[int] = None,
        show_progress: bool = True,
        overwrite: bool = False,
    ) -> List[Path]:
        """Windowed COG read of the AOI only — avoids downloading full scenes.

        Because assets are Cloud-Optimized GeoTIFFs, GDAL/rasterio can request
        only the tiles overlapping the AOI over HTTP range requests, instead
        of streaming the full 400-800 MB scene.

        Each href is presigned before it is opened — GDAL's ``GDAL_HTTP_HEADERS``
        cannot authenticate against MinIO, which only honours the signature
        carried in the URL's query string.

        Parameters
        ----------
        images_gdf : geopandas.GeoDataFrame
            Output of :meth:`search_metadata` (possibly filtered).
        output_dir : str | Path
            Destination directory.
        assets : sequence[str], default ("MS",)
            Raster asset keys to clip (``"MS"`` or ``"PAN"``; the pre-2026
            ``"src_xs"``/``"src_pan"`` are accepted).
        max_images : int, optional
            Limit the number of scenes processed.
        show_progress : bool, default True
            Show a tqdm progress bar when available.
        overwrite : bool, default False
            Re-clip files that already exist on disk.

        Returns
        -------
        list[pathlib.Path]
            Paths of all clipped GeoTIFFs written.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        aoi_geom_wgs84 = _mapping(self.aoi_geometry)

        rows = list(images_gdf.itertuples(index=False))
        if max_images is not None:
            rows = rows[: int(max_images)]

        # Same template as download_images, with "_clip" folded into the
        # suffix so a clip is never mistaken for a full scene.
        plan = self._plan_downloads(
            rows,
            lambda item: [_resolve_asset_key(item, k) for k in assets],
            out_dir,
            overrides=lambda key: {
                "suffix": f"{_ASSET_SUFFIX.get(key, '')}_clip",
                "ext": "tif",
            },
        )
        iterator = (
            _tqdm(plan, desc="COG clips", unit="scene")
            if (_HAS_TQDM and show_progress)
            else plan
        )

        written: List[Path] = []
        for image_id, jobs in iterator:
            for href, out_path in jobs:
                if out_path.exists() and not overwrite:
                    LOGGER.info("skip existing %s", out_path)
                    written.append(out_path)
                    continue
                # GDAL_DISABLE_READDIR_ON_OPEN spares a directory listing MinIO
                # cannot serve; the signature travels in the URL, not in headers.
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                    with rasterio.open(_sign_href(href)) as src:
                        # rasterio.mask reads geometry coordinates in the
                        # raster's own CRS. The AOI is WGS84 while SPOT COGs
                        # are Lambert-93 (EPSG:2154), so an unprojected AOI
                        # lands far off-grid — reproject per scene rather than
                        # assuming a single target CRS for the whole archive.
                        aoi_geom = [
                            _transform_geom("EPSG:4326", src.crs, aoi_geom_wgs84)
                        ]
                        try:
                            clipped, transform = _rio_mask(src, aoi_geom, crop=True)
                        except ValueError as exc:
                            # The search matched on the AOI *bounding box*; a
                            # scene can miss the AOI geometry itself. Skip it
                            # instead of aborting the whole batch.
                            LOGGER.warning(
                                "scene %s does not overlap the AOI, skipping (%s)",
                                image_id,
                                exc,
                            )
                            continue
                        profile = src.profile.copy()
                        profile.update(
                            {
                                "height": clipped.shape[1],
                                "width": clipped.shape[2],
                                "transform": transform,
                                "driver": "GTiff",
                            }
                        )
                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(clipped)
                LOGGER.info("wrote %s", out_path)
                written.append(out_path)

        return written

    # ---- internals
    def _check_collection_exists(self) -> None:
        """Raise if ``self.collection`` is unknown to the server.

        Called only after an empty search, so a normal search costs nothing
        extra. Any failure other than "not found" is left alone: an empty
        result is then reported as such.
        """
        try:
            self.client.stac.get_collection(self.collection)
        except Exception as exc:
            msg = str(exc).lower()
            if not any(s in msg for s in ("notfound", "not found", "does not exist")):
                return
            try:
                ids = sorted(c.id for c in self.client.stac.get_collections())
            except Exception:
                ids = []
            spot = [i for i in ids if "spot" in i.lower()]
            raise ValueError(
                f"STAC collection {self.collection!r} does not exist on "
                f"{getattr(self.client, 'stac_url', DINAMIS_STAC_URL)}"
                + (f"; SPOT collections available: {', '.join(spot)}" if spot else "")
                + f". The default is {SPOT67_COLLECTION!r}."
            ) from exc

    def _plan_downloads(self, rows, keys_for, out_dir: Path, overrides=None):
        """Resolve every (item, asset) → (href, path) before anything is fetched.

        Returns ``[(image_id, [(href, out_path), ...]), ...]``. Missing assets
        are warned about and dropped. Two assets rendering to the same path
        raise ``ValueError``: written in sequence, the second would overwrite
        the first, or be skipped as "existing" — a scene lost either way. That
        happened with the stamp-only default name, whose stamp is shared by the
        ``SCENE1``/``SCENE2`` items of one strip.
        """
        plan = []
        owner: Dict[Path, Tuple[str, str]] = {}
        for row in rows:
            image_id: str = getattr(row, "image_id")
            item = self._resolve_item(image_id)
            jobs = []
            for asset_key in keys_for(item):
                href = item["assets"].get(asset_key, {}).get("href")
                if not href:
                    LOGGER.warning("asset %r not found on item %s", asset_key, image_id)
                    continue
                extra = overrides(asset_key) if overrides else {}
                out_path = out_dir / self._clean_filename(item, asset_key, **extra)
                if out_path in owner and owner[out_path] != (image_id, asset_key):
                    other_id, other_key = owner[out_path]
                    raise ValueError(
                        f"filename collision: {other_id!r}[{other_key}] and "
                        f"{image_id!r}[{asset_key}] both render to {out_path.name!r} "
                        f"with template {self.filename_template!r}. Add "
                        "{scene_part} (or {item_id}) to the template so each "
                        "scene gets its own file."
                    )
                owner[out_path] = (image_id, asset_key)
                jobs.append((href, out_path))
            plan.append((image_id, jobs))
        return plan

    def _resolve_item(self, image_id: str) -> dict:
        """Return the raw STAC item dict, from cache or a targeted id search."""
        cached = self._item_cache.get(image_id)
        if cached is not None:
            return cached
        search = self.client.stac.search(collections=[self.collection], ids=[image_id])
        items = list(search.items_as_dicts())
        if not items:
            raise RuntimeError(
                f"Item {image_id!r} not found in collection {self.collection!r}."
            )
        self._item_cache[image_id] = items[0]
        return items[0]

    @staticmethod
    def name_fields(item: dict, asset_key: str) -> Dict[str, str]:
        """Return the placeholders available to ``filename_template`` for one asset.

        Useful for discovering what a template can reference::

            item = finder._resolve_item(gdf.loc[0, "image_id"])
            finder.name_fields(item, "PAN")

        See :func:`_name_fields` for the full list of keys.
        """
        return _name_fields(item, asset_key)

    def preview_filenames(
        self,
        images_gdf: gpd.GeoDataFrame,
        *,
        assets: Sequence[str] = ("PAN",),
        max_images: Optional[int] = None,
    ) -> pd.DataFrame:
        """Render the output filenames a download *would* produce, without downloading.

        Checking a naming convention should not cost a multi-hundred-MB transfer.

        Returns
        -------
        pandas.DataFrame
            Columns ``image_id``, ``asset``, ``filename``.
        """
        rows = list(images_gdf.itertuples(index=False))
        if max_images is not None:
            rows = rows[: int(max_images)]
        out: List[dict] = []
        for row in rows:
            image_id: str = getattr(row, "image_id")
            item = self._resolve_item(image_id)
            for asset_key in assets:
                out.append(
                    {
                        "image_id": image_id,
                        "asset": asset_key,
                        "filename": self._clean_filename(item, asset_key),
                    }
                )
        return pd.DataFrame(out, columns=["image_id", "asset", "filename"])

    def _clean_filename(self, item: dict, asset_key: str, **overrides: str) -> str:
        """Render ``filename_template`` against this asset's metadata fields.

        The default template reproduces the historical convention
        ``IMG_SPOT<6|7>[_<P|MS>]_<YYYYMMDD>[_metadata|_cloudmask|_preview].<ext>``.
        ``overrides`` replaces individual fields — used by the clip path to
        stamp ``_clip`` into the name, so both download paths honour the same
        template.
        """
        fields = _name_fields(item, asset_key)
        fields.update(overrides)
        try:
            return self.filename_template.format(**fields)
        except KeyError as exc:
            raise ValueError(
                f"Unknown placeholder {exc} in filename_template "
                f"{self.filename_template!r}. Available placeholders: "
                f"{', '.join(sorted(fields))}."
            ) from exc

    def _stream_download(
        self, url: str, out_path: Path, *, timeout: float, chunk_bytes: int = 1 << 20
    ) -> None:
        """Stream one asset to disk, atomically (temp sibling, then rename).

        ``url`` is signed here, one file at a time, so that a long bulk run
        never reaches a scene with an expired signature.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f"{out_path.name}.part")
        try:
            with requests.get(_sign_href(url), stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_bytes):
                        f.write(chunk)
            tmp_path.replace(out_path)
            LOGGER.info("wrote %s", out_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
