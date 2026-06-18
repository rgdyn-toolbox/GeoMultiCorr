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

Available symbols by level:
    · DEBUG
    ℹ INFO
    ⚠ WARNING
    ✗ ERROR
    ✗✗ CRITICAL
    ✓ SUCCESS
    » FOLDER / FILE
    ~ SETTINGS
    ? SEARCH
    ≈ STATISTICS
    ◷ TIMER
    ↓ SAVE
    · LIST
    ▸ LAUNCH
"""
from __future__ import annotations

import logging

from rich.console import Console

# Define custom logging levels (between INFO=20 and WARNING=30)
SUCCESS = 25
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

_gmc_console = Console(highlight=False)


def _add_success_method(logger_class: type[logging.Logger]) -> None:
    """Dynamically add custom logging level methods to the Logger class."""

    def success(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, message, args, **kwargs)

    def folder(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(FOLDER):
            self._log(FOLDER, message, args, **kwargs)

    def settings(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(SETTINGS):
            self._log(SETTINGS, message, args, **kwargs)

    def search(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(SEARCH):
            self._log(SEARCH, message, args, **kwargs)

    def file(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(FILE):
            self._log(FILE, message, args, **kwargs)

    def statistics(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(STATISTICS):
            self._log(STATISTICS, message, args, **kwargs)

    def timer(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(TIMER):
            self._log(TIMER, message, args, **kwargs)

    def save(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(SAVE):
            self._log(SAVE, message, args, **kwargs)

    def list(self, message: str, *args, **kwargs) -> None:
        if self.isEnabledFor(LIST):
            self._log(LIST, message, args, **kwargs)

    def launch(self, message: str, *args, **kwargs) -> None:
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
    """Custom formatter that renders the GMC prefix with rich markup or plain text.

    Output format:
        use_color=False : [ GMC ✓ ] : message
        use_color=True  : [bold green][ GMC ✓ ][/bold green] : message
    Only the prefix bracket is styled; the message text is always plain.
    """

    ICONS = {
        logging.DEBUG:    "·",
        logging.INFO:     "ℹ",
        logging.WARNING:  "⚠",
        logging.ERROR:    "✗",
        logging.CRITICAL: "✗✗",
        SUCCESS:          "✓",
        FOLDER:           "»",
        SETTINGS:         "~",
        SEARCH:           "?",
        FILE:             "»",
        STATISTICS:       "≈",
        TIMER:            "◷",
        SAVE:             "↓",
        LIST:             "·",
        LAUNCH:           "▸",
    }

    STYLES = {
        logging.DEBUG:    "dim",
        logging.INFO:     "default",
        logging.WARNING:  "bold yellow",
        logging.ERROR:    "bold red",
        logging.CRITICAL: "bold red",
        SUCCESS:          "bold green",
        FOLDER:           "blue",
        SETTINGS:         "cyan",
        SEARCH:           "cyan",
        FILE:             "cyan",
        STATISTICS:       "magenta",
        TIMER:            "magenta",
        SAVE:             "green",
        LIST:             "dim",
        LAUNCH:           "bold cyan",
    }

    PREFIX = "GMC"

    def __init__(self, use_color: bool = True, use_icons: bool = True):
        super().__init__()
        self.use_color = use_color
        self.use_icons = use_icons

    def format(self, record: logging.LogRecord) -> str:
        """Return the formatted log line as a plain string or rich markup string."""
        msg = record.getMessage()
        icon = self.ICONS.get(record.levelno, "") if self.use_icons else ""
        prefix = f"[ {self.PREFIX} {icon} ]" if icon else f"[ {self.PREFIX} ]"
        if self.use_color:
            style = self.STYLES.get(record.levelno, "default")
            return f"[{style}]{prefix}[/{style}] : {msg}"
        return f"{prefix} : {msg}"


class GMCHandler(logging.Handler):
    """Logging handler that renders GMC log lines via rich Console."""

    def __init__(self, use_color: bool = True, use_icons: bool = True):
        super().__init__()
        self._fmt = GMCFormatter(use_color=use_color, use_icons=use_icons)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _gmc_console.print(self._fmt.format(record), markup=self._fmt.use_color)
        except Exception:
            self.handleError(record)


def _setup_logger() -> logging.Logger:
    """Create and configure the GMC logger (called once at import time)."""
    _logger = logging.getLogger("GMC")
    if not _logger.handlers:
        _logger.addHandler(GMCHandler(use_color=True, use_icons=True))
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
    return _logger


logger = _setup_logger()
