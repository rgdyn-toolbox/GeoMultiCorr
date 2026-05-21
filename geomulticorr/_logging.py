#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# _logging.py
# creation date: 2026-05-07.
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
GeoMultiCorr Logging Module
============================

Usage:
    from geomulticorr._logging import logger

    logger.debug("CRS: EPSG:4326")
    logger.info("Session initialized.")
    logger.warning("No date found, using filename.")
    logger.error("Geodatabase not found.")
    logger.critical("Cannot recover, aborting.")

Control verbosity externally:
    import logging
    logging.getLogger("GMC").setLevel(logging.WARNING)

Available emoji icons by level:
    🔧 DEBUG
    📋 INFO
    ⚠️  WARNING
    ✗ ERROR
    🚨 CRITICAL

Other useful symbols for inline use:
    ✓ Checkmark
    ⚙️ Settings/Config
    🚀 Launch/Start
    ⏱️ Timer/Duration
    📁 Folder
    📄 File
    🔍 Search
    🎯 Target/Goal
    💾 Save
    📊 Statistics
    📋 Clipboard/List
"""
from __future__ import annotations

import logging
import sys

# Define custom logging levels (between INFO=20 and WARNING=30)
# Check https://docs.python.org/3/library/logging.html#logging-levels for standard levels
# TODO: Check if new custom levels can cause conflicts with other libraries
SUCCESS = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS, "SUCCESS")
FOLDER = 21
logging.addLevelName(FOLDER, "FOLDER")
SETTINGS = 22
logging.addLevelName(SETTINGS, "SETTINGS")
SEARCH = 23
logging.addLevelName(SEARCH, "SEARCH")
FILE = 24
logging.addLevelName(FILE, "FILE")
STATISTICS = 26
logging.addLevelName(STATISTICS, "STATISTICS")
TIMER = 27
logging.addLevelName(TIMER, "TIMER")
SAVE = 28
logging.addLevelName(SAVE, "SAVE")
LIST = 31
logging.addLevelName(LIST, "LIST")
LAUNCH = 32
logging.addLevelName(LAUNCH, "LAUNCH")

def _add_success_method(logger_class: type[logging.Logger]) -> None:
    """Dynamically add custom logging level methods to the Logger class.

    Adds methods: success, folder, settings, search, file, statistics, timer, save, list, launch.
    """

    def success(self, message: str, *args, **kwargs) -> None:
        """Log at ``SUCCESS`` level (25)."""
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, message, args, **kwargs)

    def folder(self, message: str, *args, **kwargs) -> None:
        """Log at ``FOLDER`` level (21)."""
        if self.isEnabledFor(FOLDER):
            self._log(FOLDER, message, args, **kwargs)

    def settings(self, message: str, *args, **kwargs) -> None:
        """Log at ``SETTINGS`` level (22)."""
        if self.isEnabledFor(SETTINGS):
            self._log(SETTINGS, message, args, **kwargs)

    def search(self, message: str, *args, **kwargs) -> None:
        """Log at ``SEARCH`` level (23)."""
        if self.isEnabledFor(SEARCH):
            self._log(SEARCH, message, args, **kwargs)

    def file(self, message: str, *args, **kwargs) -> None:
        """Log at ``FILE`` level (24)."""
        if self.isEnabledFor(FILE):
            self._log(FILE, message, args, **kwargs)

    def statistics(self, message: str, *args, **kwargs) -> None:
        """Log at ``STATISTICS`` level (26)."""
        if self.isEnabledFor(STATISTICS):
            self._log(STATISTICS, message, args, **kwargs)

    def timer(self, message: str, *args, **kwargs) -> None:
        """Log at ``TIMER`` level (27)."""
        if self.isEnabledFor(TIMER):
            self._log(TIMER, message, args, **kwargs)

    def save(self, message: str, *args, **kwargs) -> None:
        """Log at ``SAVE`` level (28)."""
        if self.isEnabledFor(SAVE):
            self._log(SAVE, message, args, **kwargs)

    def list(self, message: str, *args, **kwargs) -> None:
        """Log at ``LIST`` level (31)."""
        if self.isEnabledFor(LIST):
            self._log(LIST, message, args, **kwargs)

    def launch(self, message: str, *args, **kwargs) -> None:
        """Log at ``LAUNCH`` level (32)."""
        if self.isEnabledFor(LAUNCH):
            self._log(LAUNCH, message, args, **kwargs)

    logger_class.success = success
    logger_class.folder = folder
    logger_class.settings = settings
    logger_class.search = search
    logger_class.file = file
    logger_class.statistics = statistics
    logger_class.timer = timer
    logger_class.save = save
    logger_class.list = list
    logger_class.launch = launch

_add_success_method(logging.Logger)

class GMCFormatter(logging.Formatter):
    """Custom formatter that adds GeoMultiCorr prefix, ANSI colors, and emoji icons to log records."""

    ICONS = {
        logging.DEBUG:    "🔧",
        logging.INFO:     "📋",
        logging.WARNING:  "⚠️",
        logging.ERROR:    "✗",
        logging.CRITICAL: "🚨",
        SUCCESS:          "✓",
        FOLDER:           "📁",
        SETTINGS:         "⚙️",
        SEARCH:           "🔍",
        FILE:             "📄",
        STATISTICS:       "📊",
        TIMER:            "⏱️",
        SAVE:             "💾",
        LIST:             "📋",
        LAUNCH:           "🚀",
    }
    COLORS = {
        # ANSI escape codes for colors
        # red: \033[31m,
        # green: \033[32m,
        # yellow: \033[33m,
        # blue: \033[34m,
        # default: \033[0m,
        logging.DEBUG:    "\033[0m",    # white
        logging.INFO:     "\033[0m",     # default
        logging.WARNING:  "\033[33m",    # yellow
        logging.ERROR:    "\033[31m",    # red
        logging.CRITICAL: "\033[1;31m",  # bold red
        SUCCESS:          "\033[0m",    # green
        FOLDER:           "\033[0m",     # default
        SETTINGS:         "\033[0m",     # default
        SEARCH:           "\033[0m",     # default
        FILE:             "\033[0m",     # default
        STATISTICS:       "\033[0m",     # default
        TIMER:            "\033[0m",     # default
        SAVE:             "\033[0m",     # default
        LIST:             "\033[0m",     # default
        LAUNCH:           "\033[0m",     # default

    }
    BOLD = "\033[1m"
    RESET = "\033[0m"
    PREFIX = "GMC"

    def __init__(self, use_color: bool = True, use_icons: bool = True):
        super().__init__()
        self.use_color = use_color
        self.use_icons = use_icons

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with colors, icons, and GMC prefix."""
        msg = record.getMessage()
        icon = self.ICONS.get(record.levelno, "") if self.use_icons else ""
        if self.use_color:
            color = self.COLORS.get(record.levelno, self.RESET)
            bold = self.BOLD if getattr(record, "bold", False) else ""
            return f"{color}{bold}[ {self.PREFIX} {icon} ] : {msg}{self.RESET}"
        return f"[ {self.PREFIX} {icon} ] : {msg}"

def _setup_logger() -> logging.Logger:
    """Create and configure the GMC logger (called once at import time)."""
    _logger = logging.getLogger("GMC")
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(GMCFormatter())
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
    return _logger

logger = _setup_logger()