#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# conftest.py
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
"""Pytest fixtures for correlation module tests."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from geomulticorr.correlation import ASP

@pytest.fixture
def asp_helper():
    """ASP instance with mocked ASP binaries."""
    with patch('shutil.which', return_value="/fake/parallel_stereo"):
        return ASP(session=None, asp_bin_dir=None)
    #END with
#END def

@pytest.fixture
def mock_subprocess_success():
    """Successful subprocess result."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = "Success"
    result.stderr = ""
    return result
#END def

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for correlation outputs."""
    output_dir = tmp_path / "correlation_output"
    output_dir.mkdir()
    return output_dir
#END def


