#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_weights_frame.py
# creation date: 2026-08-31.
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
"""tests/utils/test_weights_frame.py

The weights frame is the single contract both plotting backends consume, so its
column set, its dtypes and its refusal to broadcast a short vector are what keep
the plotly figure and its matplotlib twin describing the same pairs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from geomulticorr.utils._weights_frame import (
    WEIGHT_DIRECTION_COLORS,
    WEIGHT_DIRECTION_MARKERS,
    WEIGHT_MODE_KEYS,
    WEIGHTS_FRAME_COLUMNS,
    empty_weights_frame,
    format_weights_summary,
    relevant_weight_keys,
    weight_summary,
    weights_frame,
    weights_stats,
)


def _frame(n: int = 3):
    return weights_frame(
        [f"pz_p{i}" for i in range(n)],
        [30.0 * (i + 1) for i in range(n)],
        [0.9 - 0.1 * i for i in range(n)],
        [0.8 - 0.1 * i for i in range(n)],
        nmad_ew=[0.2] * n, nmad_ns=[0.3] * n, cc=[0.7] * n,
        corr_direction=["Forward"] * n,
    )


class TestFrameContract:
    """Columns, order and dtypes — a renderer indexes these by name."""

    def test_columns_exactly_and_in_order(self):
        assert list(_frame().columns) == list(WEIGHTS_FRAME_COLUMNS)

    def test_empty_frame_shares_the_contract(self):
        empty = empty_weights_frame()
        assert list(empty.columns) == list(WEIGHTS_FRAME_COLUMNS)
        assert len(empty) == 0

    def test_zero_pairs_returns_the_empty_frame(self):
        assert len(weights_frame([], [], [], [])) == 0

    def test_numeric_columns_are_floats(self):
        frame = _frame()
        for column in ("dt_days", "w_ew", "w_ns", "nmad_ew", "nmad_ns", "cc"):
            assert frame[column].dtype == np.dtype("float64"), column

    def test_optional_columns_default_to_nan_and_empty_string(self):
        frame = weights_frame(["a"], [10], [1.0], [1.0])
        assert math.isnan(frame["nmad_ew"].iloc[0])
        assert math.isnan(frame["cc"].iloc[0])
        assert frame["corr_direction"].iloc[0] == ""

    def test_values_round_trip(self):
        frame = _frame(2)
        assert frame["pa_key"].tolist() == ["pz_p0", "pz_p1"]
        assert frame["w_ew"].tolist() == pytest.approx([0.9, 0.8])
        assert frame["w_ns"].tolist() == pytest.approx([0.8, 0.7])


class TestLengthValidation:
    """A silent broadcast here would mislabel every point in the figure."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"nmad_ew": [0.1]},
            {"nmad_ns": [0.1]},
            {"cc": [0.5]},
            {"corr_direction": ["Forward"]},
        ],
    )
    def test_short_optional_vector_raises(self, kwargs):
        with pytest.raises(ValueError, match="3 pairs"):
            weights_frame(["a", "b", "c"], [1, 2, 3], [1, 1, 1], [1, 1, 1], **kwargs)

    def test_short_weight_vector_raises(self):
        with pytest.raises(ValueError, match="w_ns"):
            weights_frame(["a", "b"], [1, 2], [1.0, 1.0], [1.0])


class TestRelevantWeightKeys:
    """Drives the widget visibility, the file stem and the JSON listing alike."""

    def test_uniform_has_no_knobs(self):
        assert relevant_weight_keys("uniform") == set()

    def test_unknown_mode_is_empty_not_an_error(self):
        assert relevant_weight_keys("not-a-mode") == set()

    def test_returns_a_copy_callers_may_mutate(self):
        first = relevant_weight_keys("sigmoid")
        first.add("bogus")
        assert "bogus" not in relevant_weight_keys("sigmoid")

    def test_abg_only_under_wmean(self):
        assert relevant_weight_keys("quality", "geomean").isdisjoint(
            {"alpha", "beta", "gamma"})
        assert {"alpha", "beta", "gamma"} <= relevant_weight_keys("quality", "wmean")

    def test_quality_spatial_wmean_has_no_gamma(self):
        """quality_spatial ignores the Δt term entirely, so γ is meaningless."""
        keys = relevant_weight_keys("quality_spatial", "wmean")
        assert {"alpha", "beta"} <= keys
        assert "gamma" not in keys

    @pytest.mark.parametrize("mode", ["relative_temporal", "sigmoid"])
    def test_dt_range_is_relevant_to_the_normalised_modes(self, mode):
        """Without it a dt_range=(300,400) run collides with a dt_range=None one."""
        assert "dt_range" in relevant_weight_keys(mode)

    def test_dt_range_is_irrelevant_elsewhere(self):
        for mode in ("uniform", "temporal", "parametric", "quality",
                     "quality_spatial"):
            assert "dt_range" not in relevant_weight_keys(mode), mode


class TestModeTableIsShared:
    """One table, not two — the explorer, the stem and the JSON must agree."""

    def test_tio_mode_controls_is_the_same_object(self):
        from geomulticorr.inversion.tio_inversion import TIOInversion

        assert TIOInversion._MODE_CONTROLS is WEIGHT_MODE_KEYS

    def test_every_mode_is_covered(self):
        from geomulticorr.inversion.tio_inversion import TIOInversion

        inv_modes = {"uniform", "temporal", "relative_temporal", "sigmoid",
                     "parametric", "quality", "quality_spatial"}
        assert set(WEIGHT_MODE_KEYS) == inv_modes
        assert TIOInversion is not None  # import guard, keeps the alias honest


class TestStyleIsShared:
    """Both backends read these, so a PNG cannot style its series differently."""

    def test_both_directions_have_a_colour_and_two_markers(self):
        for direction in ("EW", "NS"):
            assert WEIGHT_DIRECTION_COLORS[direction].startswith("#")
            plotly_symbol, mpl_marker = WEIGHT_DIRECTION_MARKERS[direction]
            assert plotly_symbol and mpl_marker

    def test_the_two_directions_are_distinguishable(self):
        assert WEIGHT_DIRECTION_COLORS["EW"] != WEIGHT_DIRECTION_COLORS["NS"]
        assert WEIGHT_DIRECTION_MARKERS["EW"] != WEIGHT_DIRECTION_MARKERS["NS"]


class TestSummaries:
    """User-facing counts, and the JSON-safety the trace file depends on."""

    def test_weight_summary_ignores_nan(self):
        summary = weight_summary([0.5, float("nan"), 1.0])
        assert summary["n"] == 3
        assert summary["min"] == pytest.approx(0.5)
        assert summary["max"] == pytest.approx(1.0)

    def test_weight_summary_of_all_nan_is_none_not_nan(self):
        """NaN is not JSON-serialisable to valid JSON — None is."""
        summary = weight_summary([float("nan"), float("nan")])
        assert summary["min"] is None and summary["mean"] is None

    def test_weight_summary_of_nothing(self):
        assert weight_summary([])["n"] == 0

    def test_weight_summary_counts_zeros(self):
        assert weight_summary([0.0, 0.0, 0.5])["n_zero"] == 2

    def test_weight_summary_is_json_safe(self):
        import json

        json.dumps(weight_summary(np.array([0.25, 0.5])))

    def test_weights_stats_reports_both_directions(self):
        stats = weights_stats(_frame(3), weight_mode="quality", combine="wmean")
        assert stats["n_pairs"] == 3
        assert stats["EW"]["max"] == pytest.approx(0.9)
        assert stats["NS"]["max"] == pytest.approx(0.8)

    def test_weights_stats_on_an_empty_frame(self):
        stats = weights_stats(empty_weights_frame())
        assert stats["n_pairs"] == 0
        assert stats["EW"]["mean"] is None

    def test_summary_line_names_the_combine_only_for_quality(self):
        quality = format_weights_summary(
            weights_stats(_frame(), weight_mode="quality", combine="wmean"))
        assert "quality:wmean" in quality

        sigmoid = format_weights_summary(
            weights_stats(_frame(), weight_mode="sigmoid", combine="wmean"))
        assert "sigmoid" in sigmoid and "wmean" not in sigmoid

    def test_summary_line_survives_missing_stats(self):
        """One pair with no stats JSON must not blank the whole summary."""
        frame = weights_frame(["a"], [10], [float("nan")], [float("nan")])
        assert "n/a" in format_weights_summary(weights_stats(frame))
