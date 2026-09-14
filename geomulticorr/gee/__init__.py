#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# __init__.py
# creation date: 2026-05-12.
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
"""Backward compatibility: GEE module has been moved to data_sources.

Deprecated: import from geomulticorr.data_sources.gee_image_search_download instead.
This module re-exports the moved classes for compatibility.
"""

import warnings

warnings.warn(
    "geomulticorr.gee is deprecated; import from geomulticorr.data_sources instead. "
    "Example: from geomulticorr.data_sources import GEEClient, GEE_ImageFinder",
    DeprecationWarning,
    stacklevel=2,
)

from geomulticorr.data_sources.gee_image_search_download import (
    GEEClient,
    CollectionSpec,
    ImageCollectionCatalog,
    ImageFinder,
)

__all__ = ["GEEClient", "CollectionSpec", "ImageCollectionCatalog", "ImageFinder"]