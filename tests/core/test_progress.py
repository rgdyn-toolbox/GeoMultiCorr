#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_progress.py
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
"""tests/core/test_progress.py

Unit tests for :class:`geomulticorr.core.console.progress.BatchProgress`.

Two things make this testable without a terminal:

* rich accepts any file-like object as a console target, and ``force_terminal``
  decides which rendering path ``Live`` takes — so both the animated and the
  non-tty fallback branches are reachable from a ``StringIO``;
* ``BatchProgress.renderable`` exposes the two-row group directly, so the layout
  contract can be asserted structurally rather than by scraping rendered text.

Note that inside a ``Live`` region rich redirects **both** ``sys.stdout`` and
``sys.stderr`` into its console. Anything a test wants to inspect afterwards must
therefore be stashed in a variable inside the block and asserted after it —
printing from inside the block goes into the console's buffer, not the terminal.
"""

import io
import logging

import pytest
from rich.console import Console, Group
from rich.progress import Progress
from rich.text import Text

from geomulticorr._logging import logger
from geomulticorr.core.console.progress import (
    BatchProgress,
    _fmt_duration,
    _live_capable,
)


# -------------------------------------------------------------- #
# Fixtures / helpers
# -------------------------------------------------------------- #
@pytest.fixture
def tty_console():
    """A console rich treats as an animating terminal."""
    return Console(file=io.StringIO(), force_terminal=True, width=100, highlight=False)


@pytest.fixture
def file_console():
    """A console rich treats as a plain file — Live.refresh() is a no-op here."""
    return Console(file=io.StringIO(), highlight=False)


class _RecordCollector(logging.Handler):
    """Collects formatted GMC messages without touching the real handlers."""

    def __init__(self):
        super().__init__(level=0)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def gmc_records():
    """Capture GMC records for the duration of a test."""
    collector = _RecordCollector()
    logger.addHandler(collector)
    try:
        yield collector.messages
    finally:
        logger.removeHandler(collector)


# -------------------------------------------------------------- #
# TEST GROUP 1: the two-row layout contract
# -------------------------------------------------------------- #
class TestTwoRowLayout:
    """Row 1 is the item label, row 2 is the bar — the same shape every call site uses."""

    def test_renderable_is_a_group_of_exactly_two(self, tty_console):
        captured = {}
        with BatchProgress(4, description="d", console=tty_console) as bar:
            for _ in bar.track(range(4), key=lambda i: f"pair_{i:02d}"):
                pass
            captured["group"] = bar.renderable

        group = captured["group"]
        assert isinstance(group, Group)
        assert len(group.renderables) == 2, "layout must stay exactly label + bar"
    # END def

    def test_row_one_is_text_row_two_is_progress(self, tty_console):
        captured = {}
        with BatchProgress(2, console=tty_console) as bar:
            for _ in bar.track(range(2)):
                pass
            captured["group"] = bar.renderable

        first, second = captured["group"].renderables
        assert isinstance(first, Text), "row 1 must be the item label"
        assert isinstance(second, Progress), "row 2 must be the bar"
    # END def

    def test_label_row_shows_the_current_item(self, tty_console):
        captured = {}
        with BatchProgress(3, console=tty_console) as bar:
            for _ in bar.track(range(3), key=lambda i: f"pair_{i:02d}"):
                pass
            captured["label"] = bar.renderable.renderables[0].plain
        assert captured["label"] == "pair_02"
    # END def

    def test_label_prefix_is_prepended(self, tty_console):
        captured = {}
        with BatchProgress(2, console=tty_console,
                           label_prefix="Mapping georasters: ") as bar:
            for _ in bar.track(range(2), key=lambda i: f"f{i}.tif"):
                pass
            captured["label"] = bar.renderable.renderables[0].plain
        assert captured["label"] == "Mapping georasters: f1.tif"
    # END def

    def test_renders_label_and_count_during_the_loop(self, tty_console):
        """Structural assertions cannot catch an unrendered bar — check the output too."""
        with BatchProgress(4, console=tty_console) as bar:
            for _ in bar.track(range(4), key=lambda i: f"pair_{i:02d}"):
                pass
        out = tty_console.file.getvalue()
        assert "pair_03" in out, "the label must actually be drawn"
        assert "4/4" in out, "MofNCompleteColumn must actually be drawn"
    # END def

    def test_renderable_is_none_without_a_live_region(self, file_console):
        captured = {}
        with BatchProgress(3, console=file_console) as bar:
            captured["r"] = bar.renderable
        assert captured["r"] is None
    # END def
# END class


# -------------------------------------------------------------- #
# TEST GROUP 2: advancing
# -------------------------------------------------------------- #
class TestAdvance:
    """track() must advance on every loop exit the call sites actually use."""

    def test_track_advances_on_continue(self, tty_console):
        """The property all three rewritten call sites depend on.

        A ``continue`` resumes the generator right after its ``yield``, so the
        advance still runs — which is why the early-exit branches in
        ``extract_pairs_raw_displacements`` need no bookkeeping of their own.
        """
        with BatchProgress(10, console=tty_console) as bar:
            for i in bar.track(range(10)):
                if i % 2 == 0:
                    continue
        assert bar.completed == 10
    # END def

    def test_track_does_not_advance_after_break(self, tty_console):
        """Deliberate: a broken-out item was not processed."""
        with BatchProgress(10, console=tty_console) as bar:
            for i in bar.track(range(10)):
                if i == 3:
                    break
        assert bar.completed == 3
    # END def

    def test_explicit_advance_with_label(self, tty_console):
        """The form pooled loops need, where completion order != iteration order."""
        captured = {}
        with BatchProgress(3, console=tty_console) as bar:
            for name in ("a", "b", "c"):
                bar.advance(label=name)
            captured["label"] = bar.renderable.renderables[0].plain
        assert bar.completed == 3
        assert captured["label"] == "c"
    # END def

    def test_advance_accepts_a_step(self, tty_console):
        with BatchProgress(10, console=tty_console) as bar:
            bar.advance(4)
        assert bar.completed == 4
    # END def

    def test_zero_total_builds_no_live_and_logs_nothing(self, file_console, gmc_records):
        with BatchProgress(0, console=file_console) as bar:
            for _ in bar.track([]):
                pass
            assert bar.renderable is None
        assert gmc_records == []
    # END def
# END class


# -------------------------------------------------------------- #
# TEST GROUP 3: the non-tty fallback
# -------------------------------------------------------------- #
class TestNonTtyFallback:
    """rich's Live renders nothing until stop() when there is no terminal."""

    def test_live_capable_true_for_a_terminal(self, tty_console):
        assert _live_capable(tty_console) is True
    # END def

    def test_live_capable_false_for_a_plain_file(self, file_console):
        assert _live_capable(file_console) is False
    # END def

    def test_emits_progress_during_the_loop_not_only_at_the_end(
        self, file_console, gmc_records
    ):
        """The direct regression test for the reported server symptom.

        A frozen bar while files land on disk means nothing was reported until
        the loop finished. At least one line must appear mid-loop.
        """
        seen_midway = []
        with BatchProgress(100, description="extract", console=file_console,
                           interval=1e9) as bar:
            for i in bar.track(range(100)):
                if i == 50:
                    seen_midway.append(len(gmc_records))
        assert seen_midway[0] > 0, "no progress was reported before the loop ended"
    # END def

    def test_final_line_reports_the_full_count(self, file_console, gmc_records):
        with BatchProgress(10, description="x", console=file_console,
                           interval=1e9) as bar:
            for _ in bar.track(range(10)):
                pass
        assert "10/10 (100%)" in gmc_records[-1]
    # END def

    def test_output_is_throttled(self, file_console, gmc_records):
        """500 items must not produce 500 log lines."""
        with BatchProgress(500, description="x", console=file_console,
                           interval=1e9, fraction=0.05) as bar:
            for _ in bar.track(range(500)):
                pass
        # ~1 line per 5% of the batch, plus the opening line.
        assert len(gmc_records) <= 25, f"expected ~21 lines, got {len(gmc_records)}"
    # END def

    def test_no_duplicate_final_line(self, file_console, gmc_records):
        with BatchProgress(100, description="x", console=file_console,
                           interval=1e9) as bar:
            for _ in bar.track(range(100)):
                pass
        assert gmc_records[-1] != gmc_records[-2]
    # END def

    def test_line_carries_description_and_label(self, file_console, gmc_records):
        with BatchProgress(20, description="apply_pairs_corrections",
                           console=file_console, interval=1e9) as bar:
            for _ in bar.track(range(20), key=lambda i: f"pair_{i:02d}"):
                pass
        assert any("apply_pairs_corrections" in m for m in gmc_records)
        assert any("pair_" in m for m in gmc_records)
    # END def

    def test_nothing_is_emitted_on_an_exception(self, file_console, gmc_records):
        """A crashed loop should not report a tidy closing line."""
        with pytest.raises(RuntimeError):
            with BatchProgress(100, console=file_console, interval=1e9) as bar:
                for i in bar.track(range(100)):
                    if i == 5:
                        raise RuntimeError("boom")
        assert not any("100/100" in m for m in gmc_records)
    # END def
# END class


# -------------------------------------------------------------- #
# TEST GROUP 4: log quieting
# -------------------------------------------------------------- #
class TestQuieting:
    """Per-item helper chatter is suppressed for the duration of the loop."""

    def test_chatter_is_suppressed_inside_and_lifts_after(self, tty_console, gmc_records):
        with BatchProgress(2, console=tty_console) as bar:
            for _ in bar.track(range(2)):
                logger.file("Saved EW displacement: x.tif")
        logger.file("after the loop")
        assert not any("Saved EW displacement" in m for m in gmc_records)
        assert any("after the loop" in m for m in gmc_records)
    # END def

    def test_verbose_true_keeps_per_item_messages(self, tty_console, gmc_records):
        with BatchProgress(2, console=tty_console, verbose=True) as bar:
            for _ in bar.track(range(2)):
                logger.file("Saved EW displacement: x.tif")
        assert sum("Saved EW displacement" in m for m in gmc_records) == 2
    # END def

    def test_warnings_survive_the_quieting(self, tty_console, gmc_records):
        with BatchProgress(2, console=tty_console) as bar:
            for _ in bar.track(range(2)):
                logger.warning("misaligned DEM")
        assert sum("misaligned DEM" in m for m in gmc_records) == 2
    # END def

    def test_quieting_is_lifted_on_an_exception(self, tty_console, gmc_records):
        with pytest.raises(RuntimeError):
            with BatchProgress(3, console=tty_console) as bar:
                for _ in bar.track(range(3)):
                    raise RuntimeError("boom")
        logger.file("after the crash")
        assert any("after the crash" in m for m in gmc_records)
    # END def
# END class


# -------------------------------------------------------------- #
# TEST GROUP 5: duration formatting
# -------------------------------------------------------------- #
class TestFormatDuration:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "0:00"),
        (9, "0:09"),
        (65, "1:05"),
        (599, "9:59"),
        (3600, "1:00:00"),
        (3661, "1:01:01"),
        (-5, "0:00"),
    ])
    def test_formatting(self, seconds, expected):
        assert _fmt_duration(seconds) == expected
    # END def
# END class
