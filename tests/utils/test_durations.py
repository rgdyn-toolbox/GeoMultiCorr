#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Duration strings for temporal-baseline filters.

This parser shipped with ``update_pairs`` and had no tests at all. It is the
single source of truth for how ``"1Y"`` becomes a number of days, and it now
feeds both explorers and the conditional-search threshold, so its edges are
worth pinning.
"""
from __future__ import annotations

import pytest

from geomulticorr.utils._durations import (
    DT_UNIT_TO_DAYS,
    format_dt_days,
    parse_dt_days,
)


class TestUnits:
    @pytest.mark.parametrize(
        "text, days",
        [("30D", 30), ("1W", 7), ("2W", 14), ("1M", 30), ("6M", 180),
         ("1Y", 365), ("10Y", 3650)],
    )
    def test_every_unit(self, text, days):
        assert parse_dt_days(text) == days

    @pytest.mark.parametrize("text", ["1y", "6m", "2w", "30d"])
    def test_case_insensitive(self, text):
        assert parse_dt_days(text) == parse_dt_days(text.upper())

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_dt_days("  1Y  ") == 365

    def test_the_table_is_deliberately_approximate(self):
        """M=30 and Y=365, not 30.44/365.25 — see the note in _durations."""
        assert DT_UNIT_TO_DAYS == {"D": 1, "W": 7, "M": 30, "Y": 365}


class TestPassThrough:
    def test_integer_passes_through(self):
        assert parse_dt_days(30) == 30

    def test_none_passes_through(self):
        assert parse_dt_days(None) is None

    def test_zero_passes_through(self):
        """0 is the explorers' 'off' sentinel and must survive intact."""
        assert parse_dt_days(0) == 0


class TestRejections:
    """Raising matters: silently reinterpreting an unparseable threshold would
    filter a pair network in a way the caller never asked for."""

    @pytest.mark.parametrize(
        "bad",
        ["30",        # unit is mandatory
         "1Y6M",      # no compound forms
         "-5D",       # no sign
         "1 Y",       # no internal whitespace
         "Y",         # no bare unit
         "",          # empty string
         "abc",
         "1.5Y",      # no fractions
         "1H",        # unsupported unit
        ],
    )
    def test_rejected(self, bad):
        with pytest.raises(ValueError, match="Cannot parse duration"):
            parse_dt_days(bad)

    def test_float_is_rejected_rather_than_truncated(self):
        with pytest.raises(ValueError):
            parse_dt_days(30.0)

    def test_the_message_names_the_accepted_forms(self):
        with pytest.raises(ValueError, match="30D"):
            parse_dt_days("nope")


class TestFormat:
    @pytest.mark.parametrize(
        "days, text",
        [(365, "1Y"), (3650, "10Y"), (180, "6M"), (30, "1M"), (14, "2W"),
         (7, "1W"), (45, "45D"), (1, "1D")],
    )
    def test_picks_the_most_readable_unit(self, days, text):
        assert format_dt_days(days) == text

    def test_round_trips_through_the_parser(self):
        for days in (7, 14, 30, 45, 180, 365, 1000, 3650):
            assert parse_dt_days(format_dt_days(days)) == days

    def test_none_is_empty(self):
        assert format_dt_days(None) == ""

    def test_zero_and_negative_stay_numeric(self):
        assert format_dt_days(0) == "0"
        assert format_dt_days(-5) == "-5"
