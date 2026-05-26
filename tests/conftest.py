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
import tempfile

from pathlib import Path

# GeoMultiCorr imports
from geomulticorr._logging import (
    GMCFormatter, logger, SUCCESS, FOLDER, SETTINGS,
    SEARCH, FILE, STATISTICS, TIMER, SAVE, LIST, LAUNCH
    )
# -------------------------------------------------------------- #
# ------------------------------------------------------------------
# FIXTURE 1: Formatter without visual elements (easiest to test)
# ------------------------------------------------------------------
@pytest.fixture
def formatter_plain():
    """GMCFormatter instance with colors and icons disabled.

    Useful for testing formatter output without ANSI codes or emoji icons,
    making assertions on text content clearer.

    :return: Configured GMCFormatter instance.
    :rtype: geomulticorr._logging.GMCFormatter
    """
    return GMCFormatter(
        use_color=False,
        use_icons=False
    )
# END def


@pytest.fixture
def temp_dir():
    """Temporary directory for file system operations in tests.

    Creates a temporary directory that is automatically cleaned up after
    the test completes. Useful for testing file creation, deletion, and
    content verification without polluting the filesystem.

    :return: Path to the temporary directory.
    :rtype: pathlib.Path

    Example:
        def test_writes_file(temp_dir):
            output_file = temp_dir / "output.txt"
            output_file.write_text("test data")
            assert output_file.exists()
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
# END def


# ------------------------------------------------------------------
# FIXTURE 2: Formatter with icons enabled
# ------------------------------------------------------------------
@pytest.fixture
def formatter_with_icons():
    """GMCFormatter instance with emoji icons enabled.

    Icons are disabled for colors, enabling verification that emoji icons
    are properly inserted into log messages at the correct positions.

    :return: Configured GMCFormatter instance.
    :rtype: geomulticorr._logging.GMCFormatter
    """
    return GMCFormatter(
        use_color=False,
        use_icons=True
    )
# END def


# ------------------------------------------------------------------
# FIXTURE 3: Formatter with colors enabled
# ------------------------------------------------------------------
@pytest.fixture
def formatter_with_colors():
    """GMCFormatter instance with ANSI colors enabled.

    Icons are disabled for colors, enabling verification that ANSI escape
    codes are properly inserted for terminal coloring.

    :return: Configured GMCFormatter instance.
    :rtype: geomulticorr._logging.GMCFormatter
    """
    return GMCFormatter(
        use_color=True,
        use_icons=False
    )
# END def


# ------------------------------------------------------------------
# FIXTURE 4: Log record factory
# ------------------------------------------------------------------
@pytest.fixture
def make_log_record():
    """Factory fixture to create logging.LogRecord instances on demand.

    Returns a callable that creates log records with customizable message
    and log level. Useful for testing formatter output with different
    message content and severity levels.

    :return: Callable that creates LogRecord instances.
    :rtype: callable

    Example:
        def test_format_warning(formatter_plain, make_log_record):
            record = make_log_record(msg="Warning!", level=logging.WARNING)
            result = formatter_plain.format(record)
            assert "Warning!" in result
    """
    def _create(msg="Test message", level=logging.INFO):
        """Create a LogRecord with given message and level.

        :param msg: Message content for the log record.
        :type msg: str
        :param level: Log level (logging.DEBUG, logging.INFO, etc.).
        :type level: int
        :return: Configured LogRecord instance.
        :rtype: logging.LogRecord
        """
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

@pytest.fixture
def caplog_gmc(caplog):
    """Caplog fixture that works with GMC logger's propagate=False.
    
    The GMC logger disables propagation to prevent duplicate messages.
    This fixture temporarily enables propagation so pytest's caplog
    can capture logs during testing.
    
    Usage:
        def test_something(caplog_gmc):
            some_function()
            assert "expected message" in caplog_gmc.text
    
    :param caplog: Pytest's built-in caplog fixture.
    :type caplog: pytest.LogCaptureFixture
    :return: The caplog fixture with GMC logger propagation enabled.
    :rtype: pytest.LogCaptureFixture
    """
    gmc_logger = logging.getLogger("GMC")
    original_propagate = gmc_logger.propagate
    gmc_logger.propagate = True
    try:
        yield caplog
    finally:
        gmc_logger.propagate = original_propagate


