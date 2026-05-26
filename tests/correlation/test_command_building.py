#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# test_command_building.py
# creation date: 2026-05-21.
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
"""Tests for correlation command building and parameter validation."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import logging
import shutil

from geomulticorr.correlation import ASP
from geomulticorr._logging import logger
# ---------------------------------------------------------------------------- #
# COMMAND BUILDING TESTS
# ---------------------------------------------------------------------------- #
class TestBuildStereoCmd:
    """Test _build_stereo_cmd method."""

    @pytest.fixture
    def asp_helper(self):
        """Create ASP instance with mocked dependencies."""
        with patch('shutil.which', return_value="/fake/parallel_stereo"):
            yield ASP(session=None, asp_bin_dir=None)
    # END def

    def test_returns_list_of_strings(self, asp_helper):
        """Should return a list of string command parts."""
        result = asp_helper._build_stereo_cmd(
            left="left.tif",
            right="right.tif",
            out_prefix="/tmp/output"
        )

        assert isinstance(result, list)
        assert all(isinstance(part, str) for part in result)
    # END def

    def test_includes_binary_name(self, asp_helper):
        """Command should start with binary name."""
        result = asp_helper._build_stereo_cmd(
            left="left.tif",
            right="right.tif",
            out_prefix="/tmp/output",
            stereo_bin="parallel_stereo"
        )

        assert result[0] == "parallel_stereo"
    # END def

    def test_includes_image_and_output_paths(self, asp_helper):
        """Command should include all required paths."""
        result = asp_helper._build_stereo_cmd(
            left="path/to/left.tif",
            right="path/to/right.tif",
            out_prefix="path/to/output"
        )

        assert "path/to/left.tif" in result
        assert "path/to/right.tif" in result
        assert "path/to/output" in result
    # END def

    def test_includes_correlator_mode_flag(self, asp_helper):
        """Command should include --correlator-mode flag."""
        result = asp_helper._build_stereo_cmd(
            left="left.tif",
            right="right.tif",
            out_prefix="/tmp/output"
        )

        assert "--correlator-mode" in result
    # END def

    @pytest.mark.parametrize("processes,expected", [
        (1, "--processes 1"),
        (4, "--processes 4"),
        (8, "--processes 8"),
    ])
    def test_processes_parameter(self, asp_helper, processes, expected):
        """Should include processes parameter in command."""
        result = asp_helper._build_stereo_cmd(
            left="left.tif",
            right="right.tif",
            out_prefix="/tmp/output",
            processes=processes
        )

        assert expected in result
    # END def
# END class

class TestBuildCorrevalCmd:
    """Test _build_correval_cmd method."""

    @pytest.fixture
    def asp_helper(self):
        """Create ASP instance with mocked dependencies."""
        with patch('shutil.which', return_value="/fake/parallel_stereo"):
            yield ASP(session=None, asp_bin_dir=None)
    # END def

    def test_returns_list_of_strings(self, asp_helper):
        """Should return a list of string command parts."""
        result = asp_helper._build_correval_cmd(
            out_prefix="/tmp/output"
        )

        assert isinstance(result, list)
        assert all(isinstance(part, str) for part in result)
    # END def

    def test_includes_correval_binary(self, asp_helper):
        """Command should start with corr_eval binary."""
        result = asp_helper._build_correval_cmd(
            out_prefix="/tmp/output",
            correval_bin="corr_eval"
        )

        assert result[0] == "corr_eval"
    # END def

    @pytest.mark.parametrize("algorithm,expected_mode", [
        ("asp_bm", "2"),
        ("asp_sgm", "0"),
        ("asp_mgm", "0"),
        ("asp_final_mgm", "0"),
    ])
    def test_prefilter_mode_by_algorithm(self, asp_helper, algorithm, expected_mode):
        """Prefilter mode should depend on algorithm."""
        result = asp_helper._build_correval_cmd(
            out_prefix="/tmp/output",
            corr_algorithm=algorithm
        )

        assert f"--prefilter-mode {expected_mode}" in result, f"Expected '--prefilter-mode {expected_mode}' not found in {result}"
    # END def

    def test_output_filename_includes_metric(self, asp_helper):
        """Output should include metric in filename."""
        result = asp_helper._build_correval_cmd(
            out_prefix="/tmp/output",
            metric="ncc"
        )

        # The output filename should be in the last argument
        # and should include the metric
        assert any("-F-ncc" in part for part in result)
    # END def
# END class