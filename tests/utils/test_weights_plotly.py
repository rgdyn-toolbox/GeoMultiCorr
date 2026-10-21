#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``figure_weights`` and its matplotlib twin: trace layout, sensor grouping, never-raise.

The two backends share one grouping rule — a single sensor group draws exactly
the two direction series it always did; several groups draw one series per
(direction, sensor) — so the tests drive both from the same frames.
"""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

from geomulticorr.utils._weights_frame import (  # noqa: E402
    empty_weights_frame,
    weights_frame,
)
from geomulticorr.utils._weights_plotly import figure_weights  # noqa: E402
from geomulticorr.utils.gmc_functions import plot_inversion_weights  # noqa: E402


def _frame(sensors=None):
    n = 4
    return weights_frame(
        [f"pz_p{i}" for i in range(n)], [30.0, 90.0, 200.0, 400.0],
        [0.9, 0.6, 0.3, 0.2], [0.8, 0.5, 0.2, 0.1],
        nmad_ew=[0.2] * n, nmad_ns=[0.3] * n, cc=[0.7] * n,
        corr_direction=["Forward"] * n, sensor=sensors,
    )


class TestPlotlyGrouping:
    def test_single_group_is_the_two_direction_traces(self):
        fig = figure_weights(_frame())
        assert [t.name for t in fig.data] == ["EW", "NS"]

    def test_single_named_sensor_is_still_two_traces(self):
        fig = figure_weights(_frame(["spot6"] * 4))
        assert [t.name for t in fig.data] == ["EW", "NS"]

    def test_mixed_sensors_give_one_trace_per_direction_and_sensor(self):
        fig = figure_weights(_frame(["planetscope", "spot6", "planetscope", "spot6"]))
        names = [t.name for t in fig.data]
        assert names == ["EW · planetscope", "EW · spot6", "NS · planetscope", "NS · spot6"]
        assert [t.legendgroup for t in fig.data] == ["EW", "EW", "NS", "NS"]
        assert len(fig.data[0].x) == 2
        assert fig.data[0].marker.color != fig.data[1].marker.color    # colour by sensor
        assert fig.data[0].marker.color == fig.data[2].marker.color    # same sensor, both dirs
        assert fig.data[0].marker.symbol == fig.data[1].marker.symbol  # symbol by direction
        assert fig.data[0].marker.symbol != fig.data[2].marker.symbol

    def test_visibility_follows_directions_across_groups(self):
        fig = figure_weights(_frame(["a", "b", "a", "b"]), directions="EW")
        assert [t.visible for t in fig.data] == [True, True, False, False]

    def test_hover_carries_the_sensor(self):
        fig = figure_weights(_frame(["a", "b", "a", "b"]))
        assert "sensor=%{customdata[3]}" in fig.data[0].hovertemplate
        assert fig.data[0].customdata[0][3] == "a"

    def test_unknown_sensor_group_is_labelled(self):
        fig = figure_weights(_frame(["", "spot6", "", "spot6"]))
        assert fig.data[0].name == "EW · ?"

    def test_empty_frame_never_raises(self):
        fig = figure_weights(empty_weights_frame())
        assert len(fig.data) == 0
        assert figure_weights(None).layout.annotations


class TestMatplotlibTwin:
    def test_single_group_two_collections(self):
        fig, ax = plot_inversion_weights(_frame())
        try:
            assert len(ax.collections) == 2
        finally:
            plt.close(fig)

    def test_mixed_sensors_four_collections_with_legend_labels(self):
        fig, ax = plot_inversion_weights(_frame(["planetscope", "spot6", "planetscope", "spot6"]))
        try:
            assert len(ax.collections) == 4
            labels = [h.get_label() for h in ax.collections]
            assert labels == ["EW · planetscope", "EW · spot6", "NS · planetscope", "NS · spot6"]
        finally:
            plt.close(fig)

    def test_no_figure_leak(self):
        plt.close("all")
        fig, _ = plot_inversion_weights(_frame(["a", "b", "a", "b"]))
        try:
            assert plt.get_fignums() == [fig.number]
        finally:
            plt.close(fig)
