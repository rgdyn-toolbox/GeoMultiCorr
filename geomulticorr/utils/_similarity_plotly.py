#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _similarity_plotly.py
# creation date: 2026-10-29.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
# ---------------------------------------------------------------------------- #
"""
Interactive views for ``Session.explore_similarity_pairs``.

Every builder takes the similarity frame and the current selection and returns
a plain ``go.Figure``. They run inside the widget loop, so **they never raise**:
degenerate input gives a figure carrying a message. The network view reuses the
pairing explorer's builder on a pairs frame, so a similarity selection is drawn
exactly like a strategy preview.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from geomulticorr.similarity.selection import (
    similarity_matrix,
    thumb_order,
    thumb_sensors,
)
from geomulticorr.utils._pairs_frame import empty_pairs_frame, pairs_frame_from_indices
from geomulticorr.utils._pairs_plotly import (
    _base_layout,
    _empty_figure,
    figure_network,
)

#: Views offered by the explorer, in dropdown order.
SIMILARITY_VIEWS: dict[str, str] = {
    "matrix": "Similarity matrix",
    "network": "Network (selected pairs)",
}


def selection_pairs_frame(
    selected: pd.DataFrame, both_directions: bool = True, pz: str = ""
) -> pd.DataFrame:
    """A pairs frame (see ``_pairs_frame``) of the selected couples.

    Left is the earlier date; with *both_directions* each couple also appears
    backward, as ``update_pairs(strategy="similarity")`` would commit it.
    """
    if selected is None or selected.empty:
        return empty_pairs_frame()
    order = thumb_order(selected)
    sensors = thumb_sensors(selected)
    dates = {}
    for k, d in zip(selected["left_key"], selected["left_date"]):
        dates[k] = d
    for k, d in zip(selected["right_key"], selected["right_date"]):
        dates[k] = d
    pos = {k: i for i, k in enumerate(order)}
    idx = [
        (pos[a], pos[b]) for a, b in zip(selected["left_key"], selected["right_key"])
    ]
    if both_directions:
        idx += [(j, i) for i, j in idx]
    return pairs_frame_from_indices(
        [dates[k] for k in order], idx, sensors=[sensors[k] for k in order], pz=pz
    )


def figure_similarity_matrix(
    sim: pd.DataFrame,
    selected: pd.DataFrame | None = None,
    *,
    metric: str = "score",
    title: str = "",
    height: int = 640,
) -> go.Figure:
    """Thumb × thumb heatmap of *metric*, selected couples outlined."""
    try:
        if sim is None or sim.empty or metric not in sim.columns:
            return _empty_figure("No similarity values to show.", title=title)
        order = thumb_order(sim)
        m = similarity_matrix(sim, metric, order)
        labels = [k.split("_", 1)[1] if "_" in k else k for k in order]
        reverse = metric == "trial_sigma_px"
        fig = go.Figure(
            go.Heatmap(
                z=m.to_numpy(),
                x=labels,
                y=labels,
                colorscale="Viridis",
                reversescale=reverse,
                colorbar=dict(title=metric),
                hovertemplate="%{y}<br>%{x}<br>"
                + metric
                + " = %{z:.3f}<extra></extra>",
            )
        )
        if selected is not None and not selected.empty:
            pos = {k: i for i, k in enumerate(order)}
            xs, ys = [], []
            for a, b in zip(selected["left_key"], selected["right_key"]):
                if a in pos and b in pos:
                    xs += [labels[pos[b]], labels[pos[a]]]
                    ys += [labels[pos[a]], labels[pos[b]]]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(
                        symbol="square-open",
                        size=max(6, min(18, int(500 / max(len(order), 1)))),
                        color="red",
                        line=dict(width=2),
                    ),
                    name="selected",
                    hoverinfo="skip",
                )
            )
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=8))
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=8))
        return _base_layout(fig, title, height)
    except Exception as exc:  # never raise in the widget loop
        return _empty_figure(f"Cannot draw the matrix: {exc}", title=title)


def figure_similarity_view(
    view: str,
    sim: pd.DataFrame,
    selected: pd.DataFrame | None,
    *,
    metric: str = "score",
    both_directions: bool = True,
    title: str = "",
) -> go.Figure:
    """Dispatch to the view's builder; unknown views and errors give a message."""
    try:
        if view == "matrix":
            return figure_similarity_matrix(sim, selected, metric=metric, title=title)
        if view == "network":
            frame = selection_pairs_frame(selected, both_directions)
            if frame.empty:
                return _empty_figure(
                    "No couple passes the current selection.", title=title
                )
            return figure_network(frame, title=title, color_by="sensor")
        return _empty_figure(f"Unknown view {view!r}.", title=title)
    except Exception as exc:
        return _empty_figure(f"Cannot draw {view!r}: {exc}", title=title)
