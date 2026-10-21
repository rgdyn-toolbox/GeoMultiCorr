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
from geomulticorr.inversion.tio_inversion import TIOInversion, tiff2bin
from geomulticorr.inversion.pytio import TIOConfig, input_tio_text
from geomulticorr.inversion import pytio

__all__ = ["TIOInversion", "tiff2bin", "TIOConfig", "input_tio_text", "pytio"]
from geomulticorr.inversion._stack import load_cumulative_stack, write_cumulative_stack
from geomulticorr.inversion.tio_inversion import resolve_inversion_dir

__all__ += ["load_cumulative_stack", "write_cumulative_stack", "resolve_inversion_dir"]
from geomulticorr.inversion.fusion import (
    calibrate_to_reference,
    closure_summary,
    fuse_inversions,
    fuse_series,
    pair_closure_against_series,
)

__all__ += ["calibrate_to_reference", "fuse_series", "pair_closure_against_series",
            "closure_summary", "fuse_inversions"]
