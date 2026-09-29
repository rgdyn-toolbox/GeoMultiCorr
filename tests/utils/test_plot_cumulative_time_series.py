#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``plot_cumulative_time_series`` — per-point panels, one line per inversion.

Structural checks only (axes count, one line per label, band on/off, no
figure leak); the rendered PNG was looked at when the function was written.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

from geomulticorr.utils import plot_cumulative_time_series  # noqa: E402

DATES = ["2016-08-16", "2018-07-29", "2020-07-25", "2022-07-22", "2024-07-08"]


def _frame(points=(0, 1, 2), components=("EW", "NS", "magn"), slope=1.0, std=0.3):
    rows = []
    for pid in points:
        for comp in components:
            for k, date in enumerate(DATES):
                v = slope * k * (1 + 0.1 * pid) * (1 if comp != "NS" else -1)
                rows.append({"point_id": pid, "date": date, "component": comp,
                             "mean": v, "median": v, "min": v - 1, "max": v + 1,
                             "std": std, "count": 9})
    return pd.DataFrame(rows)


class TestLayout:
    def test_grid_and_lines(self):
        frames = {"SPOT": _frame(), "PlanetScope": _frame(slope=0.9), "Fused": _frame(slope=1.05)}
        fig = plot_cumulative_time_series(frames, fig_name="Cumulative displacement — test")
        try:
            assert len(fig.axes) == 3 * 3
            for a in fig.axes:
                assert len(a.lines) == 3 + 1                  # three series + the zero line
                assert len(a.collections) == 3                # one band per series
            assert fig._suptitle.get_text() == "Cumulative displacement — test"
            assert fig.axes[0].get_title() == "Point 0"
            assert fig.axes[0].get_ylabel() == "EW displacement (m)"
            assert fig.axes[6].get_ylabel() == "Magnitude (m)"
            assert fig.axes[0].get_legend() is not None
        finally:
            plt.close(fig)

    def test_band_none_draws_no_collections(self):
        fig = plot_cumulative_time_series({"only": _frame()}, band=None)
        try:
            assert all(len(a.collections) == 0 for a in fig.axes)
        finally:
            plt.close(fig)

    def test_point_ids_and_components_select_the_grid(self):
        fig = plot_cumulative_time_series({"a": _frame()}, components=("EW",), point_ids=[2, 0])
        try:
            assert len(fig.axes) == 2
            assert fig.axes[0].get_title() == "Point 2"
        finally:
            plt.close(fig)

    def test_missing_component_is_annotated_not_raised(self):
        frames = {"SPOT": _frame(), "PS": _frame(components=("EW", "NS"))}
        fig = plot_cumulative_time_series(frames)
        try:
            magn_axes = fig.axes[6:]
            for a in magn_axes:
                assert len(a.lines) == 1 + 1                  # SPOT only + zero line
                assert any("n/a: PS" in t.get_text() for t in a.texts)
        finally:
            plt.close(fig)

    def test_median_stat_and_colour_override(self):
        fig = plot_cumulative_time_series({"x": _frame()}, stat="median",
                                          colors={"x": "#123456"})
        try:
            assert fig.axes[0].lines[0].get_color() == "#123456"
        finally:
            plt.close(fig)

    def test_existing_axes_are_used(self):
        fig, axes = plt.subplots(2, 3)
        try:
            out = plot_cumulative_time_series({"x": _frame()}, components=("EW", "NS"), ax=axes)
            assert out is fig
            assert len(axes[0, 0].lines) == 2
        finally:
            plt.close(fig)

    def test_wrong_axes_shape_raises(self):
        fig, axes = plt.subplots(1, 3)
        try:
            with pytest.raises(ValueError, match="array of Axes"):
                plot_cumulative_time_series({"x": _frame()}, ax=axes)
        finally:
            plt.close(fig)

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="frames is empty"):
            plot_cumulative_time_series({})
        with pytest.raises(ValueError, match="point_id"):
            plot_cumulative_time_series({"x": pd.DataFrame()})


class TestNoFigureLeak:
    def test_exactly_one_figure_is_created(self):
        plt.close("all")
        fig = plot_cumulative_time_series({"a": _frame(), "b": _frame(slope=2)})
        try:
            assert plt.get_fignums() == [fig.number]
        finally:
            plt.close(fig)
