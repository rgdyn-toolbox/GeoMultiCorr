#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# progress.py
# creation date: 2026-08-30.
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
"""Shared batch-progress reporter.

One helper replaces the ~20-line rich ``Live`` + ``Progress`` block that was
copy-pasted across every per-pair batch loop in :mod:`geomulticorr.core.session`
and :mod:`geomulticorr.inversion.tio_inversion`. Beyond removing the duplication
it fixes two things the copies all got wrong.

**It quiets the per-item log spam.** The helpers a batch loop calls
(``Pair.extract_raw_displacements``, the ``stats`` savers,
``TIOInversion.export_pair_to_binary``) log one to five lines *per pair* — for a
few hundred pairs that buries the progress bar. Those lines are genuinely useful
when the same helper is driven on a single pair from a notebook, so they are not
removed: they are suppressed for the duration of the loop via
:func:`geomulticorr._logging.quiet`, and the caller prints one header up front
instead. Pass ``verbose=True`` to get them back.

**It degrades usefully when there is no terminal.** ``rich.live.Live.refresh()``
is a no-op unless the console is a terminal or a Jupyter kernel: on a batch job
whose stdout is a file (``nohup``, OAR, SLURM) the bar renders *nothing* until
the loop ends, then dumps one final frame. Here, a non-tty console skips ``Live``
entirely and emits throttled one-line progress records instead.
"""
from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Any, Callable, Iterable, Iterator, TypeVar

from rich.console import Group as _RichGroup
from rich.live import Live as _RichLive
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text as _RichText

from geomulticorr._logging import ALWAYS, logger, quiet
from geomulticorr.core.console.console import _rich_console

T = TypeVar("T")

#: Non-tty fallback: never more than one progress line per this many seconds.
FALLBACK_INTERVAL_S = 15.0
#: Non-tty fallback: at least one progress line per this fraction of the batch.
FALLBACK_FRACTION = 0.05


def _live_capable(console: Any) -> bool:
    """Return ``True`` when a rich ``Live`` region actually renders while running.

    Mirrors the guard inside :meth:`rich.live.Live.refresh`: on a Jupyter kernel
    it renders into an ``ipywidgets.Output``, on a terminal it repaints in place,
    and *everywhere else* ``refresh()` does nothing until ``stop()`` prints one
    final frame. The Jupyter half additionally needs ``ipywidgets`` importable —
    without it rich only issues a ``warnings.warn`` and renders nothing at all.

    :param console: The rich console the ``Live`` would be attached to.
    :return: ``True`` if a live region would animate, ``False`` if it would not.
    :rtype: bool
    """
    if console.is_terminal:
        return True
    if console.is_jupyter:
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            return False
        return True
    return False


def _fmt_duration(seconds: float) -> str:
    """Format *seconds* as ``M:SS``, or ``H:MM:SS`` once past an hour."""
    total = int(max(0.0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


class BatchProgress:
    """Two-row progress reporter for a per-item batch loop.

    The rendered live region is a :class:`rich.console.Group` of exactly two
    renderables — the item label on the first row, the bar on the second::

        PasDeLours_2016-08-16-planetscope_2017-07-03-planetscope
         42% ━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━ 105/250 • 0:04:12

    This is the layout already used by ``Session.map_georasters`` and
    ``Session.sieve_bulk``; collecting it here is what stops it drifting between
    call sites. Note the ``Progress`` is deliberately never ``.start()``-ed — it
    is a renderable driven by the ``Live``, which owns the refresh loop. Starting
    it too would spawn a second refresh thread fighting the first.

    Usable two ways. **Iterator form** — the common case. :meth:`track` labels
    the item before the body runs and advances after it, *including when the body
    hits* ``continue`` (a ``continue`` resumes the generator right after its
    ``yield``), so early-exit branches need no bookkeeping::

        with BatchProgress(len(pairs), description="extract", verbose=verbose) as bar:
            for pair in bar.track(pairs, key=lambda p: p.pa_key):
                if not pair.pa_disparity_f_path.exists():
                    continue          # still advances

    A ``break``, by contrast, skips the final advance — deliberate, since the
    item was not processed.

    **Explicit form** — for pooled loops where completion order is not iteration
    order::

        with BatchProgress(len(futures), description="map_georasters") as bar:
            for fut in as_completed(futures):
                try:
                    ...
                finally:
                    bar.advance(label=names[fut])

    :param total: Number of items. ``0`` builds no bar and emits nothing.
    :type total: int
    :param description: Short loop name, used in the non-tty fallback lines.
    :type description: str
    :param label_prefix: Fixed text prepended to every label, e.g.
        ``"Mapping georasters: "``.
    :type label_prefix: str
    :param verbose: ``True`` keeps the per-item log messages the loop's helpers
        emit. Default ``False`` suppresses everything below WARNING for the
        duration of the block.
    :type verbose: bool
    :param console: rich console; defaults to the package singleton.
    :param interval: Non-tty fallback — minimum seconds between progress lines.
    :type interval: float
    :param fraction: Non-tty fallback — emit at least every this fraction of the
        batch.
    :type fraction: float
    """

    def __init__(
        self,
        total: int,
        *,
        description: str = "",
        label_prefix: str = "",
        verbose: bool = False,
        console: Any = None,
        interval: float = FALLBACK_INTERVAL_S,
        fraction: float = FALLBACK_FRACTION,
    ) -> None:
        self.total = int(total)
        self.description = description or "progress"
        self.label_prefix = label_prefix
        self.completed = 0
        self.console = _rich_console if console is None else console
        self._verbose = bool(verbose)
        self._interval = float(interval)
        self._step = max(1, int(self.total * fraction)) if self.total else 1
        self._label = ""
        self._stack: ExitStack | None = None
        self._live: Any = None
        self._progress: Progress | None = None
        self._task: Any = None
        self._text: _RichText | None = None
        self._group: _RichGroup | None = None
        self._t0 = 0.0
        self._last_t = 0.0
        self._last_n = -1

    @property
    def renderable(self) -> "_RichGroup | None":
        """The two-row live region: ``Group(Text, Progress)``, or ``None``.

        ``None`` when no live region was built — an empty batch, or a console
        that is not live-capable (see :func:`_live_capable`). Prefer this over
        reaching into ``Live.renderable``, which rich nests inside a group of
        its own.
        """
        return self._group

    # ── context manager ──────────────────────────────────────────────────────
    def __enter__(self) -> "BatchProgress":
        self._t0 = self._last_t = time.monotonic()
        self._stack = ExitStack()
        self._stack.enter_context(quiet(enabled=not self._verbose))
        if self.total and _live_capable(self.console):
            self._text = _RichText("", style="cyan")
            self._progress = Progress(
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=self.console,
            )
            self._task = self._progress.add_task("", total=self.total)
            # Row 1 = label, row 2 = bar.
            self._group = _RichGroup(self._text, self._progress)
            self._live = self._stack.enter_context(
                _RichLive(
                    self._group,
                    console=self.console,
                    transient=False,
                    refresh_per_second=10,
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Non-tty runs get a closing line so the log records the final count —
        # unless throttling already happened to emit one at exactly that count.
        if (
            exc_type is None
            and self._live is None
            and self.total
            and self._last_n != self.completed
        ):
            self._emit(final=True)
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()
        return False

    # ── driving the bar ──────────────────────────────────────────────────────
    def track(
        self,
        items: Iterable[T],
        key: Callable[[T], str] | None = None,
    ) -> Iterator[T]:
        """Yield *items*, labelling before each one and advancing after it.

        :param items: The batch to iterate.
        :param key: Maps an item to its display label; defaults to ``str``.
        :type key: callable or None
        """
        for item in items:
            self.set_label(key(item) if key is not None else str(item))
            yield item
            self.advance()

    def set_label(self, label: str) -> None:
        """Show *label* (after ``label_prefix``) as the item being processed."""
        self._label = f"{self.label_prefix}{label}"
        if self._live is not None:
            self._text.plain = self._label  # type: ignore[union-attr]
            self._live.refresh()
        else:
            self._emit()

    def advance(self, n: int = 1, label: str | None = None) -> None:
        """Mark *n* items complete, optionally updating the label first.

        :param n: How many items completed.
        :type n: int
        :param label: New label to show; ``None`` keeps the current one.
        :type label: str or None
        """
        if label is not None:
            self._label = f"{self.label_prefix}{label}"
        self.completed += n
        if self._live is not None:
            if label is not None:
                self._text.plain = self._label  # type: ignore[union-attr]
            self._progress.advance(self._task, n)  # type: ignore[union-attr]
            self._live.refresh()
        else:
            self._emit()

    # ── non-tty fallback ─────────────────────────────────────────────────────
    def _emit(self, final: bool = False) -> None:
        """Log a throttled progress line (no-terminal path only).

        The two rows necessarily collapse into one here: a log file has no cursor
        to repaint, so the label is appended after an em-dash instead of living
        on its own row.
        """
        now = time.monotonic()
        n = self.completed
        due = (
            final
            or self._last_n < 0
            or (now - self._last_t) >= self._interval
            or (n - self._last_n) >= self._step
        )
        if not due:
            return
        self._last_t, self._last_n = now, n
        elapsed = now - self._t0
        pct = (100.0 * n / self.total) if self.total else 100.0
        left = f", ~{_fmt_duration(elapsed / n * (self.total - n))} left" if n else ""
        item = f" — {self._label}" if (self._label and not final) else ""
        # extra=ALWAYS: this record must survive the quiet() block opened above.
        logger.info(
            f"{self.description}: {n}/{self.total} ({pct:>3.0f}%) • "
            f"{_fmt_duration(elapsed)} elapsed{left}{item}",
            extra=ALWAYS,
        )
