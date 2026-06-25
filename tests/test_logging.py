#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# test_logging.py
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
"""tests/test_logging.py

Unit tests for geomulticorr._logging module.
"""

import pytest
import logging

from geomulticorr._logging import (
    GMCFormatter, logger, SUCCESS, FOLDER, SETTINGS,
    SEARCH, FILE, STATISTICS, TIMER, SAVE, LIST, LAUNCH
    )

# -------------------------------------------------------------- #
# TEST GROUP 1: Custom Loggings Levels
# -------------------------------------------------------------- #
class TestCustomLogLevels:
    """Test that custom log levels (like SUCCESS) are properly registered"""
    def test_success_level_constant_exists(self):
        """SUCCESS constant should be defined"""
        assert SUCCESS == 25
    # END def

    def test_folder_level_constant_exists(self):
        """FOLDER constant should be defined"""
        assert FOLDER == 21
    # END def

    def test_settings_level_constant_exists(self):
        """SETTINGS constant should be defined"""
        assert SETTINGS == 22
    # END def

    def test_search_level_constant_exists(self):
        """SEARCH constant should be defined"""
        assert SEARCH == 23
    # END def

    def test_file_level_constant_exists(self):
        """FILE constant should be defined"""
        assert FILE == 24
    # END def

    def test_statistics_level_constant_exists(self):
        """STATISTICS constant should be defined"""
        assert STATISTICS == 26
    # END def

    def test_timer_level_constant_exists(self):
        """TIMER constant should be defined"""
        assert TIMER == 27
    # END def

    def test_save_level_constant_exists(self):
        """SAVE constant should be defined"""
        assert SAVE == 28
    # END def

    def test_list_level_constant_exists(self):
        """LIST constant should be defined"""
        assert LIST == 31
    # END def

    def test_launch_level_constant_exists(self):
        """LAUNCH constant should be defined"""
        assert LAUNCH == 32
    # END def

    def test_custom_level_names_registered(self):
        """Custom log levels should be registered in logging module"""
        assert logging.getLevelName(SUCCESS) == "SUCCESS", "SUCCESS level should be registered"
        assert logging.getLevelName(FOLDER) == "FOLDER", "FOLDER level should be registered"
        assert logging.getLevelName(SETTINGS) == "SETTINGS", "SETTINGS level should be registered"
        assert logging.getLevelName(SEARCH) == "SEARCH", "SEARCH level should be registered"
        assert logging.getLevelName(FILE) == "FILE", "FILE level should be registered"
        assert logging.getLevelName(STATISTICS) == "STATISTICS", "STATISTICS level should be registered"
        assert logging.getLevelName(TIMER) == "TIMER", "TIMER level should be registered"
        assert logging.getLevelName(SAVE) == "SAVE", "SAVE level should be registered"
        assert logging.getLevelName(LIST) == "LIST", "LIST level should be registered"
        assert logging.getLevelName(LAUNCH) == "LAUNCH", "LAUNCH level should be registered"
    # END def
# END class

# -------------------------------------------------------------- #
# TEST GROUP 2: Logger has custom methods
# -------------------------------------------------------------- #
class TestLoggerCustomMethods:
    """Test that the logger has the custom methods we added"""

    def test_logger_has_all_custom_methods(self):
        """Logger should have all custom methods for custom levels"""
        methods = ["success", "folder", "settings", "search", "file", "statistics", "timer", "save", "list", "launch"]
        for method in methods:
            assert hasattr(logger, method), f"Logger should have a '{method}' method"
            assert callable(getattr(logger, method)), f"'{method}' should be a method"
        #END for
    # END def
# END class

# -------------------------------------------------------------- #
# TEST GROUP 3: GMCFormatter basic functionality
# -------------------------------------------------------------- #
class TestGMCFormatterBasics:
    """Test basic functionality of the GMCFormatter"""

    def test_formatter_instantiation(self):
        """Formatter should instantiate without errors"""
        formatter = GMCFormatter(
            use_color=False,
            use_icons=False
        )
        assert isinstance(formatter, GMCFormatter)
    # END def

    def test_formatter_attibutes_preserved(self):
        """Formatter should remember its configuration options"""
        formatter = GMCFormatter(
            use_color=True,
            use_icons=True
        )
        assert formatter.use_color is True
        assert formatter.use_icons is True
    # END def

    def test_formatter_has_icons_dict(self):
        """Formatter should have an ICONS mapping"""
        assert hasattr(GMCFormatter, "ICONS")
        assert isinstance(GMCFormatter.ICONS, dict)
        assert logging.DEBUG in GMCFormatter.ICONS
        assert logging.INFO in GMCFormatter.ICONS
        assert logging.WARNING in GMCFormatter.ICONS
        assert logging.ERROR in GMCFormatter.ICONS
        assert logging.CRITICAL in GMCFormatter.ICONS
        assert SUCCESS in GMCFormatter.ICONS
        assert FOLDER in GMCFormatter.ICONS
        assert SETTINGS in GMCFormatter.ICONS
        assert SEARCH in GMCFormatter.ICONS
        assert FILE in GMCFormatter.ICONS
        assert STATISTICS in GMCFormatter.ICONS
        assert TIMER in GMCFormatter.ICONS
        assert SAVE in GMCFormatter.ICONS
        assert LIST in GMCFormatter.ICONS
        assert LAUNCH in GMCFormatter.ICONS
    # END def

    def test_formatter_has_styles_dict(self):
        """Formatter should have a STYLES mapping covering all levels"""
        assert hasattr(GMCFormatter, "STYLES")
        assert isinstance(GMCFormatter.STYLES, dict)
        for level in (
            logging.DEBUG, logging.INFO, logging.WARNING,
            logging.ERROR, logging.CRITICAL,
            SUCCESS, FOLDER, SETTINGS, SEARCH, FILE,
            STATISTICS, TIMER, SAVE, LIST, LAUNCH,
        ):
            assert level in GMCFormatter.STYLES, f"Level {level} missing from STYLES"
            assert isinstance(GMCFormatter.STYLES[level], str)
            assert len(GMCFormatter.STYLES[level]) > 0
    # END def
# END class
# -------------------------------------------------------------- #
# TEST GROUP 4: GMCFormatter output
# -------------------------------------------------------------- #
class TestGMCFormatterOutput:
    """Test what GMCFormatter actually outputs"""

    def test_format_includes_message(self, formatter_plain, make_log_record):
        """Formatted output should include the log message."""
        record = make_log_record(msg="Test message")  # Call as function
        result = formatter_plain.format(record)
        assert "Test message" in result
    
    def test_format_includes_separator_colon(self, formatter_plain, make_log_record):
        """Formatted output should have : separator after prefix."""
        record = make_log_record(msg="Test message")  # Call as function
        result = formatter_plain.format(record)
        assert "GMC" in result
        assert ":" in result
    
    def test_format_structure(self, formatter_plain, make_log_record):
        """Full format structure."""
        record = make_log_record(msg="Test message")  # Call as function
        result = formatter_plain.format(record)
        
        assert result.startswith("[")
        assert "]" in result
        assert ":" in result
        assert result.endswith("Test message")
    
    def test_format_returns_string(self, formatter_plain, make_log_record):
        """Format method should always return a string."""
        record = make_log_record()  # Call as function, uses default "Test message"
        result = formatter_plain.format(record)
        assert isinstance(result, str)
# END class
# -------------------------------------------------------------- #
# TEST GROUP 5: GMCFormatter with icons enabled
# -------------------------------------------------------------- #
class TestGMCFormatterIcons:
    """Test that GMCFormatter icons are included/excluded"""

    def test_formatter_excludes_icons_when_disabled(
            self,
            formatter_plain,
            make_log_record):
        """Formatter should exclude icons when use_icons=False"""
        record = make_log_record(msg="Test message")
        result = formatter_plain.format(record)

        # None of the Unicode symbols used as icons should appear
        symbol_list = ["·", "ℹ", "⚠", "✗", "✗✗", "✓", "»", "~", "?", "≈", "◷", "↓", "▸"]
        for symbol in symbol_list:
            assert symbol not in result, f"Found symbol {symbol!r} in: {result}"
    # END def

    def test_correct_icon_for_debug_level(self, formatter_with_icons):
        """DEBUG level should use · icon."""
        record = logging.LogRecord(
            name="GMC", level=logging.DEBUG, pathname="test.py", lineno=42, msg="Debug message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "·" in result, f"Did not find DEBUG icon in: {result}"
    # END def

    def test_correct_icon_for_info_level(self, formatter_with_icons):
        """INFO level should use ℹ icon."""
        record = logging.LogRecord(
            name="GMC", level=logging.INFO, pathname="test.py", lineno=42, msg="Info message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "ℹ" in result, f"Did not find INFO icon in: {result}"
    # END def

    def test_correct_icon_for_warning_level(self, formatter_with_icons):
        """WARNING level should use ⚠ icon."""
        record = logging.LogRecord(
            name="GMC", level=logging.WARNING, pathname="test.py", lineno=42, msg="Warning message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "⚠" in result, f"Did not find WARNING icon in: {result}"
    # END def

    def test_correct_icon_for_error_level(self, formatter_with_icons):
        """ERROR level should use ✗ icon."""
        record = logging.LogRecord(
            name="GMC", level=logging.ERROR, pathname="test.py", lineno=42, msg="Error message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "✗" in result, f"Did not find ERROR icon in: {result}"
    # END def

    def test_correct_icon_for_critical_level(self, formatter_with_icons):
        """CRITICAL level should use ✗✗ icon."""
        record = logging.LogRecord(
            name="GMC", level=logging.CRITICAL, pathname="test.py", lineno=42, msg="Critical message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "✗✗" in result, f"Did not find CRITICAL icon in: {result}"
    # END def

    def test_correct_icon_for_success_level(self, formatter_with_icons):
        """SUCCESS level should use ✓ icon."""
        record = logging.LogRecord(
            name="GMC", level=SUCCESS, pathname="test.py", lineno=42, msg="Success message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "✓" in result, f"Did not find SUCCESS icon in: {result}"
    # END def

    def test_correct_icon_for_folder_level(self, formatter_with_icons):
        """FOLDER level should use » icon."""
        record = logging.LogRecord(
            name="GMC", level=FOLDER, pathname="test.py", lineno=42, msg="Folder message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "»" in result, f"Did not find FOLDER icon in: {result}"
    # END def

    def test_correct_icon_for_settings_level(self, formatter_with_icons):
        """SETTINGS level should use ~ icon."""
        record = logging.LogRecord(
            name="GMC", level=SETTINGS, pathname="test.py", lineno=42, msg="Settings message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "~" in result, f"Did not find SETTINGS icon in: {result}"
    # END def

    def test_correct_icon_for_search_level(self, formatter_with_icons):
        """SEARCH level should use ? icon."""
        record = logging.LogRecord(
            name="GMC", level=SEARCH, pathname="test.py", lineno=42, msg="Search message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "?" in result, f"Did not find SEARCH icon in: {result}"
    # END def

    def test_correct_icon_for_file_level(self, formatter_with_icons):
        """FILE level should use » icon."""
        record = logging.LogRecord(
            name="GMC", level=FILE, pathname="test.py", lineno=42, msg="File message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "»" in result, f"Did not find FILE icon in: {result}"
    # END def

    def test_correct_icon_for_statistics_level(self, formatter_with_icons):
        """STATISTICS level should use ≈ icon."""
        record = logging.LogRecord(
            name="GMC", level=STATISTICS, pathname="test.py", lineno=42, msg="Statistics message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "≈" in result, f"Did not find STATISTICS icon in: {result}"
    # END def

    def test_correct_icon_for_timer_level(self, formatter_with_icons):
        """TIMER level should use ◷ icon."""
        record = logging.LogRecord(
            name="GMC", level=TIMER, pathname="test.py", lineno=42, msg="Timer message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "◷" in result, f"Did not find TIMER icon in: {result}"
    # END def

    def test_correct_icon_for_save_level(self, formatter_with_icons):
        """SAVE level should use ↓ icon."""
        record = logging.LogRecord(
            name="GMC", level=SAVE, pathname="test.py", lineno=42, msg="Save message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "↓" in result, f"Did not find SAVE icon in: {result}"
    # END def

    def test_correct_icon_for_list_level(self, formatter_with_icons):
        """LIST level should use · icon."""
        record = logging.LogRecord(
            name="GMC", level=LIST, pathname="test.py", lineno=42, msg="List message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "·" in result, f"Did not find LIST icon in: {result}"
    # END def

    def test_correct_icon_for_launch_level(self, formatter_with_icons):
        """LAUNCH level should use ▸ icon."""
        record = logging.LogRecord(
            name="GMC", level=LAUNCH, pathname="test.py", lineno=42, msg="Launch message", args=(), exc_info=None
        )
        result = formatter_with_icons.format(record)
        assert "▸" in result, f"Did not find LAUNCH icon in: {result}"
    # END def
# END class

# -------------------------------------------------------------- #
# TEST GROUP 6: GMCFormatter colors
# -------------------------------------------------------------- #
class TestGMCFormatterColors:
    """Test that GMCFormatter colors are included/excluded"""

    def test_formatter_excludes_colors_when_disabled(self, formatter_plain):
        """WHen use_color=False, no ANSI codes should appear"""
        record = logging.LogRecord(
            name="GMC", level=logging.INFO, pathname="test.py", lineno=42,
            msg="Info message", args=(), exc_info=None
        )
        result = formatter_plain.format(record)
        # ANSI escape codes look like \033[31m for red, \033[32m for green, etc.
        assert "\033[" not in result, f"Found ANSI color code in: {result}"
    # END def

    def test_formatter_includes_colors_when_enabled(self, formatter_with_colors):
        """When use_color=True, ANSI escape codes should wrap the prefix."""
        record = logging.LogRecord(
            name="GMC", level=logging.INFO, pathname="test.py", lineno=42,
            msg="Info message", args=(), exc_info=None
        )
        result = formatter_with_colors.format(record)
        assert "\033[" in result, f"Did not find ANSI color code in: {result}"
    # END def
# END class

# -------------------------------------------------------------- #
# TEST GROUP 7: Edge cases
# -------------------------------------------------------------- #
class TestGMCFormatterEdgeCases:
    """Test unusual or edge scenarios"""
    def test_format_with_empty_message(self, formatter_plain):
        """Should handle empty message gracefully"""
        record = logging.LogRecord(
            name="GMC", level=logging.INFO, pathname="test.py", lineno=42,
            msg="", args=(), exc_info=None
        )
        result = formatter_plain.format(record)
        assert isinstance(result, str), "Formatter should return a string even for empty message"
    # END def

    def test_format_with_very_long_message(self, formatter_plain):
        """Should handle very long messages"""
        long_msg = "A" * 1000  # 1000 characters
        record = logging.LogRecord(
            name="GMC", level=logging.INFO, pathname="test.py", lineno=42,
            msg=long_msg, args=(), exc_info=None
        )
        result = formatter_plain.format(record)
        assert long_msg in result, "Formatter should include the full long message"
    # END def

    def test_format_with_special_characters(self, formatter_plain):
        """Should handle messages with special characters"""
        special_msg = "Special chars: \n\t\r!@#$%^&*()_+-=[]{}|;':\",.<>/?"
        record = logging.LogRecord(
            name="GMC", level=logging.INFO, pathname="test.py", lineno=42,
            msg=special_msg, args=(), exc_info=None
        )
        result = formatter_plain.format(record)
        assert special_msg in result, "Formatter should include the message with special characters"
    # END def

    def test_format_with_newlines(self, formatter_plain):
        """Should handle messages with newlines properly"""
        multiline_msg = "Line 1\nLine 2\nLine 3"
        record = logging.LogRecord(
            name="GMC", level=logging.INFO, pathname="test.py", lineno=42,
            msg=multiline_msg, args=(), exc_info=None
        )
        result = formatter_plain.format(record)
        assert multiline_msg in result, "Formatter should include the message with newlines"
    # END def
# END class