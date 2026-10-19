#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# __init__.py
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
"""Satellite-imagery and DEM data sources.

**Satellite imagery:**

* Microsoft Planetary Computer (free, no API key):
  :mod:`geomulticorr.data_sources.mpc_image_search_download`
* Planet / PlanetScope (commercial, API key + area quota):
  :mod:`geomulticorr.data_sources.planet_image_search_download`
* USGS M2M (free Landsat/Level-1, no quota, per-band selection):
  :mod:`geomulticorr.data_sources.usgs_m2m_image_search_download`
* Google Earth Engine (quota-limited exports, multi-sensor):
  :mod:`geomulticorr.data_sources.gee_image_search_download`
* DINAMIS SPOT 6/7 (free w/ account, metropolitan France 2017-2023, 1.5 m pan):
  :mod:`geomulticorr.data_sources.dinamis_spot_image_search_download`

**Digital Elevation Models:**

* SRTM GL1 (30 m, 56S–60N, ESA STEP):
* Copernicus DEM GLO-30 (30 m, global, AWS):
  :mod:`geomulticorr.data_sources.global_dem_download`
"""

from geomulticorr.data_sources.mpc_image_search_download import (
    CollectionSpec as MPC_CollectionSpec,
    ImageCollectionCatalog as MPC_ImageCollectionCatalog,
    ImageFinder as MPC_ImageFinder,
    MPC_STAC_URL,
)
from geomulticorr.data_sources.planet_image_search_download import (
    PlanetClient,
    PlanetImageFinder,
    plot_thumbnail_grid,
)
from geomulticorr.data_sources.usgs_m2m_image_search_download import (
    M2MClient,
    M2MImageFinder,
)
from geomulticorr.data_sources.gee_image_search_download import (
    GEEClient,
    ImageCollectionCatalog as GEE_ImageCollectionCatalog,
    ImageFinder as GEE_ImageFinder,
    CollectionSpec as GEE_CollectionSpec,
)
from geomulticorr.data_sources.global_dem_download import (
    DEMClient,
    DEMFinder,
)
from geomulticorr.data_sources.dinamis_spot_image_search_download import (
    DinamisClient,
    DinamisImageFinder,
    DEFAULT_FILENAME_TEMPLATE as DINAMIS_DEFAULT_FILENAME_TEMPLATE,
    LEGACY_FILENAME_TEMPLATE as DINAMIS_LEGACY_FILENAME_TEMPLATE,
)

__all__ = [
    # MPC (Microsoft Planetary Computer)
    "MPC_CollectionSpec",
    "MPC_ImageCollectionCatalog",
    "MPC_ImageFinder",
    "MPC_STAC_URL",
    # Planet
    "PlanetClient",
    "PlanetImageFinder",
    "plot_thumbnail_grid",
    # USGS M2M
    "M2MClient",
    "M2MImageFinder",
    # Google Earth Engine
    "GEEClient",
    "GEE_ImageCollectionCatalog",
    "GEE_ImageFinder",
    "GEE_CollectionSpec",
    # Global DEMs
    "DEMClient",
    "DEMFinder",
    # DINAMIS SPOT 6/7
    "DinamisClient",
    "DinamisImageFinder",
    "DINAMIS_DEFAULT_FILENAME_TEMPLATE",
    "DINAMIS_LEGACY_FILENAME_TEMPLATE",
]
