#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# conftest.py
# creation date: 2026-05-19.
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
"""
tests/conftest.py

Shared test fixtures - automatically available to all tests.
"""

import pytest
import logging

# GeoMultiCorr imports
from geomulticorr._logging import (
    GMCFormatter, logger, SUCCESS, FOLDER, SETTINGS,
    SEARCH, FILE, STATISTICS, TIMER, SAVE, LIST, LAUNCH
    )
# -------------------------------------------------------------- #
# FIXTURE 1: Formatter without visual elements (easiest to test)
# -------------------------------------------------------------- #
@pytest.fixture
def formatter_plain():
    """A GMCFormatter instance with colors and icons disabled."""
    return GMCFormatter(
        use_color=False,
        use_icons=False
        )
# END def

# ------------------------------------------------------------------- #
# FIXTURE 2: Formatter with visual elements (colors and icons enabled)
# ------------------------------------------------------------------- #
@pytest.fixture
def formatter_with_icons():
    """A GMCFormatter instance with icons enabled, but no colors."""
    return GMCFormatter(
        use_color=False,
        use_icons=True
        )
# END def

# ------------------------------------------------------------------- #
# FIXTURE 3: Formatter with colors
# ------------------------------------------------------------------- #
@pytest.fixture
def formatter_with_colors():
    """A GMCFormatter instance with colors enabled, but no icons."""
    return GMCFormatter(
        use_color=True,
        use_icons=False
        )
# END def

# ------------------------------------------------------------------- #
# FIXTURE 4: Formatter with colors and icons enabled (full visual style)
# ------------------------------------------------------------------- #
@pytest.fixture
def make_log_record():
    """A factory fixture that creates log records on demand."""
    def _create(msg="Test message", level=logging.INFO):
        return logging.LogRecord(
            name="GMC",
            level=level,
            pathname="test.py",
            lineno=42,
            msg=msg,
            args=(),
            exc_info=None
        )
    return _create
    # END def
#END def


