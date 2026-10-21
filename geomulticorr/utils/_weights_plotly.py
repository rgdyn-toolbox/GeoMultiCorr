#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _weights_plotly.py
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
"""Interactive plotly view of TIO pair weights.

The counterpart of :mod:`geomulticorr.utils._pairs_plotly` for the inversion
weights: it takes a *weights frame*
(:mod:`geomulticorr.utils._weights_frame`) and returns a plain
:class:`plotly.graph_objects.Figure`.

**A plain ``Figure``, never a ``go.FigureWidget``.** Since plotly 6 the latter is
an *anywidget*, whose front-end JS is not bundled with the VSCode Jupyter
extension and fails to load ("No version of module anywidget is registered").
:meth:`~geomulticorr.inversion.tio_inversion.TIOInversion.explore_weights`
redraws this into a ``widgets.Output`` instead, the same way the pairing-strategy
explorer does.

**This module must never raise.** It runs inside the widget loop, where an
exception only reaches the kernel log that VSCode usually hides — degenerate
input returns an empty figure carrying the reason as an annotation.

**Keep it out of** :mod:`geomulticorr.utils` ``__init__`` — that package imports
``gmc_functions`` on every ``import geomulticorr``, and re-exporting from here
would make plotly a hard import-time requirement of the whole library.
"""
from __future__ import annotations

import pandas as pd

from geomulticorr.utils._weights_frame import (
    WEIGHT_DIRECTION_COLORS,
    WEIGHT_DIRECTION_MARKERS,
    sensor_colors,
)

#: Hover fields shared by both series, in ``customdata`` column order.
_CUSTOMDATA_COLUMNS: tuple[str, ...] = ("nmad", "cc", "corr_direction", "sensor")

#: Hover template shared by both series.  ``$DIRECTION`` is substituted rather
#: than formatted: the template is full of plotly's own ``%{…}`` fields, which
#: ``str.format`` reads as format specs and rejects with ``KeyError: 'text'``.
_HOVER = (
    "pair_key=%{text}"
    "<br>weight=%{y:.3f}"
    "<br>map=$DIRECTION"
    "<br>corr_dir=%{customdata[2]}"
    "<br>sensor=%{customdata[3]}"
    "<br>NMAD=%{customdata[0]:.3f}"
    "<br>CC=%{customdata[1]:.2f}"
    "<br>Δt=%{x:.0f} d<extra></extra>"
)


def _empty_figure(message: str = "No pairs to plot"):
    """A blank figure carrying *message*, for input nothing can be drawn from."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                 showarrow=False, font=dict(size=14, color="#888")),
        ],
    )
    return fig


def figure_weights(
    frame: pd.DataFrame,
    *,
    title: str = "",
    directions: str | tuple[str, ...] = ("EW", "NS"),
    height: int = 460,
):
    """Weight-vs-Δt scatter, one series per direction (and per sensor when mixed).

    With a single sensor group in *frame* (or none recorded) the figure is
    exactly two traces, coloured by direction. With several groups each
    direction gets one trace **per sensor**, coloured by
    :func:`~geomulticorr.utils._weights_frame.sensor_colors` and still marked
    by direction, in a legend group per direction — so a ``sensor_weights``
    factor is visible as a level shift of one colour.

    :param frame: A weights frame (see :mod:`geomulticorr.utils._weights_frame`).
    :param title: Figure title.
    :param directions: Which series to draw — ``"both"``, ``"EW"``, ``"NS"``, or
        any tuple of those direction names.
    :param height: Figure height in pixels.
    :returns: A :class:`plotly.graph_objects.Figure`; an annotated empty one when
        *frame* is empty, since this must not raise inside the widget loop.
    """
    import plotly.graph_objects as go

    if frame is None or len(frame) == 0:
        return _empty_figure()

    if isinstance(directions, str):
        wanted = ("EW", "NS") if directions == "both" else (directions,)
    else:
        wanted = tuple(directions)

    sensors = (frame["sensor"].fillna("").astype(str).tolist()
               if "sensor" in frame.columns else [""] * len(frame))
    groups = sorted(set(sensors))
    by_sensor = len(groups) > 1
    palette = sensor_colors(groups) if by_sensor else {}

    fig = go.Figure()
    for direction, w_col, nmad_col in (("EW", "w_ew", "nmad_ew"),
                                       ("NS", "w_ns", "nmad_ns")):
        symbol, _ = WEIGHT_DIRECTION_MARKERS[direction]
        for group in (groups if by_sensor else [None]):
            sub = frame if group is None else frame[[s == group for s in sensors]]
            sub_sensors = [s for s in sensors if group is None or s == group]
            # list-of-lists rather than np.column_stack, so the float NMAD/CC
            # keep their dtype alongside the string columns.
            customdata = [
                [n, c, d, sen] for n, c, d, sen in zip(
                    sub[nmad_col].tolist(), sub["cc"].tolist(),
                    sub["corr_direction"].tolist(), sub_sensors)
            ]
            fig.add_trace(
                go.Scatter(
                    x=sub["dt_days"].tolist(), y=sub[w_col].tolist(), mode="markers",
                    name=direction if group is None else f"{direction} · {group or '?'}",
                    legendgroup=direction,
                    text=sub["pa_key"].tolist(), customdata=customdata, opacity=0.75,
                    visible=direction in wanted,
                    marker=dict(size=9, symbol=symbol,
                                color=(WEIGHT_DIRECTION_COLORS[direction]
                                       if group is None else palette[group])),
                    hovertemplate=_HOVER.replace("$DIRECTION", direction),
                )
            )

    fig.update_layout(
        title=title or "TIO pair weights",
        xaxis_title="Temporal baseline Δt (days)",
        yaxis_title="weight",
        yaxis_range=[-0.02, 1.02],
        template="plotly_white",
        height=height,
    )
    return fig
