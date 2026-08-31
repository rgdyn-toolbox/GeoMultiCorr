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

Silence a chatty batch loop temporarily:
    from geomulticorr._logging import quiet

    with quiet():                 # only WARNING and above get through
        for pair in pairs:
            pair.extract_raw_displacements()

Note that ``setLevel`` alone is a blunt instrument: LIST and LAUNCH sit
numerically *above* WARNING and would slip through. Prefer :func:`quiet`,
which ranks records by :func:`severity` rather than by raw level number.

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
import sys
from contextlib import contextmanager
from typing import Any, Iterator

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


# ── Severity remap ───────────────────────────────────────────────────────────
# The custom levels are NOT ordered by importance. They were numbered to slot
# between INFO=20 and WARNING=30 so they would print by default, but LIST=31 and
# LAUNCH=32 overflowed past WARNING — they are chatter, not alerts. Anything
# reasoning about "how important is this record" must go through severity(),
# never through record.levelno directly, or a plain setLevel(WARNING) leaks them.
_SEVERITY: dict[int, int] = {
    logging.DEBUG:    logging.DEBUG,
    LIST:             15,
    logging.INFO:     logging.INFO,
    FOLDER:           logging.INFO,
    SETTINGS:         logging.INFO,
    SEARCH:           logging.INFO,
    FILE:             logging.INFO,
    SUCCESS:          logging.INFO,
    STATISTICS:       logging.INFO,
    TIMER:            logging.INFO,
    SAVE:             logging.INFO,
    LAUNCH:           logging.INFO,
    logging.WARNING:  logging.WARNING,
    logging.ERROR:    logging.ERROR,
    logging.CRITICAL: logging.CRITICAL,
}

_ALWAYS_ATTR = "gmc_always"

#: ``extra=`` payload marking a record that must survive :func:`quiet`.
#:
#: Used by the batch-progress fallback, which has to report progress from
#: inside the very block that silences the loop's per-item chatter. Treat it as
#: immutable — :mod:`logging` copies its keys onto the record, so one shared
#: dict is safe, but mutating it would leak into every record already emitted.
ALWAYS: dict[str, Any] = {_ALWAYS_ATTR: True}


def severity(levelno: int) -> int:
    """Return the importance rank of *levelno*, correcting the level ordering.

    The GMC custom levels are numbered for print-by-default behaviour, not by
    importance: ``LIST=31`` and ``LAUNCH=32`` sit above ``WARNING=30`` despite
    being routine chatter. This maps every level onto a rank that *is* ordered
    by importance, so callers can compare records meaningfully.

    :param levelno: A logging level number.
    :type levelno: int
    :return: The importance rank (unknown levels are returned unchanged).
    :rtype: int
    """
    return _SEVERITY.get(levelno, levelno)


class _QuietFilter(logging.Filter):
    """Drop records ranked below *min_severity*, unless flagged ``gmc_always``."""

    def __init__(self, min_severity: int) -> None:
        super().__init__()
        self.min_severity = min_severity

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, _ALWAYS_ATTR, False):
            return True
        return severity(record.levelno) >= self.min_severity


@contextmanager
def quiet(min_level: int = logging.WARNING, *, enabled: bool = True) -> Iterator["GMCLogger"]:
    """Temporarily silence low-severity GMC records.

    Wrap a batch loop whose per-item helpers are chatty by design. Those helpers
    (``Pair.extract_raw_displacements``, the ``stats`` savers,
    ``TIOInversion.export_pair_to_binary``) are also driven on a single pair from
    notebooks, where their messages *are* useful — so they are suppressed here
    for the duration of the loop rather than removed. Warnings and errors still
    surface, as do records logged with ``extra=``:data:`ALWAYS`.

    The filter is attached to the *logger*, not to a handler: :meth:`Logger.handle`
    consults ``self.filter()`` before ``callHandlers``, so one filter covers the
    console handler and any file handler a user has added.

    The logger's *level* is deliberately left untouched, so this can only ever
    make output quieter, never louder. Nesting is safe: each block installs and
    removes its own filter.

    .. warning::
       The GMC logger is a process-wide singleton, so this also silences records
       emitted from other threads while the block is active. Safe for sequential
       loops; pass ``enabled=False`` around threaded work you still want to hear
       from.

    :param min_level: Lowest level (ranked by :func:`severity`) still emitted.
        Only DEBUG / INFO / WARNING / ERROR / CRITICAL are meaningful — every
        custom level between INFO and WARNING ranks as INFO.
    :type min_level: int
    :param enabled: ``False`` makes the block a no-op, so callers can write
        ``with quiet(enabled=not verbose):`` without branching.
    :type enabled: bool
    :return: The GMC logger, for convenience.
    :rtype: GMCLogger
    """
    if not enabled:
        yield logger
        return

    # Suppression is done purely by the filter — deliberately NOT by raising the
    # logger level. setLevel(WARNING) would make isEnabledFor(INFO) false, so an
    # ALWAYS-flagged record would never be constructed and the filter would never
    # get to let it through. The level is left alone, which also means quiet()
    # can never lower a level the caller deliberately raised.
    flt = _QuietFilter(severity(min_level))
    logger.addFilter(flt)
    try:
        yield logger
    finally:
        logger.removeFilter(flt)


class GMCLogger(logging.Logger):
    """GMC logger subclass with custom logging levels and methods."""

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


class GMCFormatter(logging.Formatter):
    """Custom formatter that adds GMC prefix, ANSI colors, and Unicode symbols.

    Output format:
        use_color=False : [GMC ✓] : message
        use_color=True  : <ansi>[GMC ✓]<reset> : message
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
        logging.DEBUG:    "\033[2m",     # dim
        logging.INFO:     "\033[0m",     # default
        logging.WARNING:  "\033[1;33m",  # bold yellow
        logging.ERROR:    "\033[1;31m",  # bold red
        logging.CRITICAL: "\033[1;31m",  # bold red
        SUCCESS:          "\033[1;32m",  # bold green
        FOLDER:           "\033[34m",    # blue
        SETTINGS:         "\033[36m",    # cyan
        SEARCH:           "\033[36m",    # cyan
        FILE:             "\033[36m",    # cyan
        STATISTICS:       "\033[35m",    # magenta
        TIMER:            "\033[35m",    # magenta
        SAVE:             "\033[32m",    # green
        LIST:             "\033[2m",     # dim
        LAUNCH:           "\033[1;36m",  # bold cyan
    }

    RESET = "\033[0m"
    PREFIX = "GMC"

    def __init__(self, use_color: bool = True, use_icons: bool = True):
        super().__init__()
        self.use_color = use_color
        self.use_icons = use_icons

    def format(self, record: logging.LogRecord) -> str:
        """Return the formatted log line as a plain or ANSI-colored string."""
        msg = record.getMessage()
        icon = self.ICONS.get(record.levelno, "") if self.use_icons else ""
        prefix = f"[{self.PREFIX} {icon}]" if icon else f"[{self.PREFIX}]"
        if self.use_color:
            ansi = self.STYLES.get(record.levelno, self.RESET)
            return f"{ansi}{prefix}{self.RESET} : {msg}"
        return f"{prefix} : {msg}"


_LAZY_STREAM = object()  # sentinel: "resolve sys.stdout at emit time"


class LazyStdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stdout`` at emit time, not at import.

    The stock ``logging.StreamHandler(sys.stdout)`` captures whatever
    ``sys.stdout`` was bound to when it was constructed and holds that object
    forever. Since this handler is built at import time, it never sees a later
    swap — which breaks two things GMC relies on:

    * ``contextlib.redirect_stdout`` and notebook output capture never receive
      GMC output, because the handler still writes to the original stream;
    * inside a :class:`rich.live.Live` region, rich replaces ``sys.stdout`` with
      a :class:`rich.file_proxy.FileProxy` precisely so that ordinary writes are
      rendered *above* the live region. A handler holding the original stream
      writes underneath rich's cursor bookkeeping and smears the progress bar —
      the bar appears stuck or duplicated while work proceeds normally.

    Resolving lazily routes GMC log lines through that FileProxy, which
    ANSI-decodes each line into a rich ``Text``. Colours survive, and because
    rich never applies console markup to a ``Text`` instance, bracketed messages
    such as ``[MedianCentering]`` are passed through rather than eaten as markup.

    ``setStream()`` still works and pins an explicit stream; call
    :meth:`reset_stream` to go back to following ``sys.stdout``.
    """

    def __init__(self) -> None:
        self._pinned: Any = _LAZY_STREAM
        super().__init__(stream=_LAZY_STREAM)

    @property
    def stream(self) -> Any:  # type: ignore[override]
        return sys.stdout if self._pinned is _LAZY_STREAM else self._pinned

    @stream.setter
    def stream(self, value: Any) -> None:
        # Both StreamHandler.__init__ and StreamHandler.setStream() assign here;
        # recording the value (rather than ignoring it) keeps setStream working.
        self._pinned = value

    def reset_stream(self) -> None:
        """Drop any pinned stream and follow ``sys.stdout`` again."""
        self.acquire()
        try:
            self._pinned = _LAZY_STREAM
        finally:
            self.release()


def _setup_logger() -> GMCLogger:
    """Create and configure the GMC logger (called once at import time)."""
    logging.setLoggerClass(GMCLogger)
    _logger = logging.getLogger("GMC")
    if not _logger.handlers:
        handler = LazyStdoutHandler()
        handler.setFormatter(GMCFormatter())
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
    return _logger  # type: ignore[return-value]


logger: GMCLogger = _setup_logger()
