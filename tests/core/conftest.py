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
"""Pytest fixtures for core module tests."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

@pytest.fixture
def mock_session():
    """Mock Session object"""
    session = MagicMock()
    session.session_path = Path("/tmp/test_session")
    session.geodatabase_path = Path("/tmp/test_session/gmc.gpkg")
    session.mapset_path = Path("/tmp/test_session/mapset.qgz")
    return session
#END def

@pytest.fixture
def mock_pzone(mock_session):
    """Mock ProcessingZone object"""
    pzone = MagicMock()
    pzone.name = "test_zone"
    pzone.session = mock_session
    pzone.crs = "EPSG:4326"
    return pzone
#END def

@pytest.fixture
def mock_pair(mock_pzone):
    """Mock Pair object."""
    pair = MagicMock()
    pair.pa_key = "TestZone_2020-01-01_2020-01-02"
    pair.pa_path = Path("/tmp/pair_output")
    pair.pa_status = "clipped"
    pair.pzone = mock_pzone
    pair.get_status.return_value = "clipped"
    return pair