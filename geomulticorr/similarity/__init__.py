#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# __init__.py
# creation date: 2026-10-30.
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
"""Image similarity as a predictor of correlability, for any pair of images.

``preprocess`` and ``metrics`` are pure array code with no GMC imports, so the
package is usable on any two co-registered images (Landsat, Sentinel-2,
PlanetScope, SPOT, Pléiades, hillshades) without opening a GMC session.
"""

from geomulticorr.similarity import metrics, preprocess
from geomulticorr.similarity.metrics import (
    SIMILARITY_KEYS,
    histogram_comparison,
    joint_valid,
    local_zncc,
    mutual_information,
    nmad,
    orientation_correlation,
    pearson,
    similarity_report,
    spearman,
    spectral_coherence,
    ssim_components,
    trial_correlation,
)
from geomulticorr.similarity.preprocess import (
    antialias_sigma,
    as_float,
    asp_prefilter,
    band_pass,
    match_histogram,
    nan_gaussian,
    nan_uniform,
    wallis,
)
from geomulticorr.similarity.raster import compare_rasters
from geomulticorr.similarity.selection import (
    METRIC_HIGHER_IS_BETTER,
    bridge_coverage,
    rank_score,
    select_bridge_couples,
    select_couples,
    sensor_family,
)
