#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
