#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# test_parameters.py
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
"""Unit tests for correlation parameters."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import logging
import shutil

from geomulticorr.correlation import ASP
from geomulticorr._logging import logger

# ---------------------------------------------------------------------------- #
# PARAMETER BUILDING TESTS
# ---------------------------------------------------------------------------- #
class TestBuildCorrelationParams:
    """Test build_correlation_params method."""

    @pytest.fixture
    def asp_helper(self):
        """Create ASP instance with mocked dependencies."""
        with patch('shutil.which', return_value="/fake/parallel_stereo"):
            yield ASP(session=None, asp_bin_dir=None)
    # END def

    @pytest.fixture
    def temp_params_file(self):
        """Create temporary file for parameters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    # END def

    def test_returns_list_of_strings(self, asp_helper, temp_params_file):
        """Should return list of parameter lines."""
        result = asp_helper.build_correlation_params(
            params_file_path=temp_params_file
        )

        assert isinstance(result, list)
        assert all(isinstance(line, str) for line in result)
    # END def

    def test_writes_to_file(self, asp_helper, temp_params_file):
        """Should write parameters to file."""
        asp_helper.build_correlation_params(
            params_file_path=temp_params_file
        )

        assert temp_params_file.exists()
        content = temp_params_file.read_text()
        assert len(content) > 0
        assert "stereo-algorithm" in content
    # END def

    def test_includes_algorithm_parameter(self, asp_helper, temp_params_file):
        """Parameters should include correlation algorithm."""
        result = asp_helper.build_correlation_params(
            corr_algorithm="asp_sgm",
            params_file_path=temp_params_file
        )

        content = "\n".join(result)
        assert "stereo-algorithm asp_sgm" in content
    # END def

    def test_sgm_algorithm_override(self, asp_helper, temp_params_file, caplog):
        """SGM algorithm should trigger kernel override warning."""
        result = asp_helper.build_correlation_params(
            corr_algorithm="asp_sgm",
            corr_kernel=(21, 21),  # Will be overridden
            params_file_path=temp_params_file
        )

        content = "\n".join(result)
        # Kernel should be overridden to 9x9
        assert "corr-kernel 9 9" in content
    # END def
# END class