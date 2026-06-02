#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ---------------------------------------------------------------------------- #
"""Correction and mask pipeline for GeoMultiCorr displacement fields."""

from geomulticorr.corrections import fit
from geomulticorr.corrections.corrections import (
    BaseCorrection,
    CorrectionPipeline,
    MedianCentering,
    RampCorrection,
    TopoCorrection,
    TopoRampCorrection,
    SlopeRampCorrection,
    AlongTrackDestriping,
    AcrossTrackDestriping,
    make_corrections,
)
from geomulticorr.corrections.masks import (
    BaseMask,
    FilterPipeline,
    OutlierFilter,
    CCFilter,
    StableAreaMask,
    SnowMask,
    CloudMask,
    SlopeMask,
    ShadowMask,
)

__all__ = [
    "BaseCorrection",
    "CorrectionPipeline",
    "MedianCentering",
    "RampCorrection",
    "TopoCorrection",
    "TopoRampCorrection",
    "SlopeRampCorrection",
    "AlongTrackDestriping",
    "AcrossTrackDestriping",
    "make_corrections",
    "BaseMask",
    "FilterPipeline",
    "OutlierFilter",
    "CCFilter",
    "StableAreaMask",
    "SnowMask",
    "CloudMask",
    "SlopeMask",
    "ShadowMask",
]
