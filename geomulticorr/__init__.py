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
__version__ = "0.5.2"

print(f"geomulticorr {__version__}")

from geomulticorr.core.session import open_gmc_session
from geomulticorr.inversion import TIOInversion
from geomulticorr.corrections import (
    CorrectionPipeline,
    OutlierFilter,
    CCFilter,
    MedianCentering,
    RampCorrection,
    TopoCorrection,
    DirectionalBiasCorrection,
    AlongTrackDestriping,
    AcrossTrackDestriping,
    SnowMask,
    CloudMask,
    SlopeMask,
    ShadowMask,
)
