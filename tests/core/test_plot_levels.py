#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_plot_levels.py
# creation date: 2026-08-14.
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
"""Tests for the PLOT_LEVELS presets used by Session.apply_pairs_corrections."""

from __future__ import annotations

import pytest

from geomulticorr.core.session import PLOT_LEVELS
from geomulticorr.utils.gmc_functions import PLOT_PREVIEW_PX


class TestPlotLevels:
    def test_expected_levels_exist(self):
        assert set(PLOT_LEVELS) == {"light", "full"}

    @pytest.mark.parametrize("level", ["light", "full"])
    def test_every_level_defines_every_knob(self, level):
        # apply_pairs_corrections indexes these three keys unconditionally.
        assert set(PLOT_LEVELS[level]) == {"preview_px", "dpi", "hexbin_gridsize"}

    def test_light_is_the_cheaper_preset(self):
        light, full = PLOT_LEVELS["light"], PLOT_LEVELS["full"]
        assert light["dpi"] < full["dpi"]
        assert light["hexbin_gridsize"] < full["hexbin_gridsize"]
        assert light["preview_px"] is not None

    def test_full_disables_decimation(self):
        assert PLOT_LEVELS["full"]["preview_px"] is None

    def test_light_matches_the_plotting_module_default(self):
        # A drift here would make the batch path and a direct plot_* call
        # render at different resolutions for no visible reason.
        assert PLOT_LEVELS["light"]["preview_px"] == PLOT_PREVIEW_PX
