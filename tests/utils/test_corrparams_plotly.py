#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the correlation-parameter plotly views.

These builders run inside the widget loop, where an exception reaches only the
kernel log that VSCode hides — so "never raises" is the headline contract here,
not a nicety.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomulticorr.utils._corrparams_frame import (
    corrparams_frame_from_pairs,
    empty_corrparams_frame,
)
from geomulticorr.utils._corrparams_plotly import (
    DESIGN_COLORS,
    GROUP_PALETTE,
    VIEW_BUILDERS,
    VIEW_LABELS,
    figure_cost,
    figure_design_map,
    figure_kernel_search,
    figure_snr,
)


@pytest.fixture
def frame():
    return corrparams_frame_from_pairs(
        ["a", "b", "c", "d"], [30.0, 400.0, 1200.0, 2500.0], [3.0, 3.0, 1.5, 1.5],
        sensor_i=["ps", "ps", "spot", "spot"], velocity_m_yr=2.0,
    )


class TestDispatch:
    def test_labels_cover_every_builder(self):
        assert set(VIEW_BUILDERS) == set(VIEW_LABELS)

    def test_design_map_is_the_default_first_view(self):
        assert next(iter(VIEW_LABELS)) == "design_map"

    @pytest.mark.parametrize("view", list(VIEW_BUILDERS))
    def test_every_view_builds(self, view, frame):
        fig = VIEW_BUILDERS[view](frame)
        assert len(fig.data) >= 1

    @pytest.mark.parametrize("view", list(VIEW_BUILDERS))
    def test_every_view_accepts_a_title(self, view, frame):
        assert VIEW_BUILDERS[view](frame, title="hello").layout.title.text == "hello"


class TestNeverRaises:
    @pytest.mark.parametrize("view", list(VIEW_BUILDERS))
    def test_empty_frame_returns_an_annotated_figure(self, view):
        fig = VIEW_BUILDERS[view](empty_corrparams_frame())
        assert fig is not None
        assert fig.layout.annotations

    @pytest.mark.parametrize("view", list(VIEW_BUILDERS))
    def test_none_frame_does_not_raise(self, view):
        assert VIEW_BUILDERS[view](None) is not None

    def test_non_finite_baselines_do_not_raise(self, frame):
        broken = frame.copy()
        broken["dt_days"] = 0
        assert figure_design_map(broken) is not None

    def test_all_nan_search_does_not_raise(self, frame):
        broken = frame.copy()
        broken["search_px"] = np.nan
        assert figure_design_map(broken) is not None


class TestDesignMap:
    def test_draws_band_floor_and_ceiling(self, frame):
        fig = figure_design_map(frame)
        names = [t.name for t in fig.data]
        assert "measurable" in names
        assert any("detection floor" in str(n) for n in names)
        assert any("search reach" in str(n) for n in names)

    def test_both_axes_are_log(self, frame):
        """The two boundaries are only straight parallel lines in log-log."""
        fig = figure_design_map(frame)
        assert fig.layout.xaxis.type == "log"
        assert fig.layout.yaxis.type == "log"

    def test_one_marker_trace_per_group_not_per_pair(self, frame):
        fig = figure_design_map(frame, color_by="group")
        marker_traces = [t for t in fig.data if t.mode == "markers"]
        assert len(marker_traces) == frame["group"].nunique()
        assert sum(len(t.x) for t in marker_traces) == len(frame)

    def test_color_by_sensor_groups_by_sensor(self, frame):
        fig = figure_design_map(frame, color_by="sensor")
        assert len([t for t in fig.data if t.mode == "markers"]) == 2

    def test_color_by_none_is_a_single_trace(self, frame):
        fig = figure_design_map(frame, color_by="none")
        assert len([t for t in fig.data if t.mode == "markers"]) == 1

    def test_below_floor_pairs_get_a_distinct_symbol(self, frame):
        fig = figure_design_map(frame, k=3.0, c_px=0.15)
        symbols = np.concatenate(
            [np.asarray(t.marker.symbol).ravel() for t in fig.data if t.mode == "markers"]
        )
        assert "x" in set(symbols)

    def test_tau_draws_the_decorrelation_cut(self, frame):
        fig = figure_design_map(frame, tau_days=900.0)
        assert any("τ" in str(a.text) for a in fig.layout.annotations)

    def test_no_tau_draws_no_cut(self, frame):
        assert not figure_design_map(frame).layout.annotations


class TestKernelSearch:
    def test_draws_kernel_and_search_series(self, frame):
        names = [t.name for t in figure_kernel_search(frame).data]
        assert "kernel (px)" in names
        assert "search half-width (px)" in names

    def test_y_axis_is_log(self, frame):
        """A linear axis lets the strain ceiling crush the data into a sliver."""
        assert figure_kernel_search(frame).layout.yaxis.type == "log"

    def test_strain_ceiling_appears_only_when_asked(self, frame):
        assert "strain ceiling" not in [t.name for t in figure_kernel_search(frame).data]
        with_strain = figure_kernel_search(frame, strain_rate_per_yr=0.02)
        assert "strain ceiling" in [t.name for t in with_strain.data]

    def test_texture_floor_is_annotated(self, frame):
        fig = figure_kernel_search(frame)
        assert any("texture floor" in str(a.text) for a in fig.layout.annotations)


class TestSnr:
    def test_threshold_line_is_annotated(self, frame):
        fig = figure_snr(frame, k=3.0)
        assert any("k=3" in str(a.text) for a in fig.layout.annotations)

    def test_title_counts_pairs_below_the_floor(self, frame):
        assert "below the floor" in figure_snr(frame, k=3.0).layout.title.text

    def test_clean_frame_gets_no_warning_in_the_title(self):
        good = corrparams_frame_from_pairs(
            ["a"], [2000.0], [3.0], sensor_i=["ps"], velocity_m_yr=5.0
        )
        assert "below the floor" not in figure_snr(good, k=3.0).layout.title.text


class TestCost:
    def test_one_bar_per_group(self, frame):
        fig = figure_cost(frame)
        assert len(fig.data[0].x) == frame["group"].nunique()

    def test_bars_sum_the_group_cost(self, frame):
        fig = figure_cost(frame)
        np.testing.assert_allclose(
            float(np.sum(fig.data[0].y)), float(frame["cost_index"].sum())
        )

    def test_title_reports_the_total(self, frame):
        assert "total" in figure_cost(frame).layout.title.text


class TestStyleTables:
    def test_palette_is_non_empty_and_stable(self):
        assert len(GROUP_PALETTE) >= 4
        assert all(c.startswith("#") for c in GROUP_PALETTE)

    def test_design_colors_cover_every_role(self):
        assert {"floor", "ceiling", "band", "pairs", "flagged", "limit"} <= set(DESIGN_COLORS)


class TestAutoSearchRendering:
    """When no pair has an explicit box there is no search reach to draw, and a
    line from a fabricated default would describe a constraint that does not
    exist."""

    @pytest.fixture
    def all_auto(self):
        return corrparams_frame_from_pairs(
            ["a", "b"], [30.0, 60.0], [3.0, 3.0], sensor_i=["ps", "ps"],
            velocity_m_yr=2.0, explicit_search_above_days=365,
        )

    @pytest.fixture
    def mixed(self):
        return corrparams_frame_from_pairs(
            ["a", "b"], [30.0, 2000.0], [3.0, 3.0], sensor_i=["ps", "ps"],
            velocity_m_yr=2.0, explicit_search_above_days=365,
        )

    def test_all_auto_omits_the_ceiling_line(self, all_auto):
        names = [str(t.name) for t in figure_design_map(all_auto).data]
        assert not any(n.startswith("search reach (S=") for n in names)

    def test_all_auto_says_so_in_the_legend(self, all_auto):
        names = [str(t.name) for t in figure_design_map(all_auto).data]
        assert any("ASP auto" in n for n in names)

    def test_all_auto_omits_the_shaded_band(self, all_auto):
        """A band needs two boundaries; with one it would be unbounded."""
        assert "measurable" not in [str(t.name) for t in figure_design_map(all_auto).data]

    def test_the_detection_floor_is_always_drawn(self, all_auto):
        names = [str(t.name) for t in figure_design_map(all_auto).data]
        assert any("detection floor" in n for n in names)

    def test_a_mixed_frame_still_draws_the_ceiling(self, mixed):
        names = [str(t.name) for t in figure_design_map(mixed).data]
        assert any(n.startswith("search reach (S=") for n in names)

    def test_all_auto_never_raises_in_any_view(self, all_auto):
        for builder in VIEW_BUILDERS.values():
            assert builder(all_auto) is not None
