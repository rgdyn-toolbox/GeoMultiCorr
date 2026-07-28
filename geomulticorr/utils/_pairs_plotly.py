#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _pairs_plotly.py
# creation date: 2026-07-27.
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
"""Interactive plotly views of a pair network, for the pairing-strategy explorer.

Every builder is a **pure function of a pairs frame** (see
:mod:`geomulticorr.utils._pairs_frame`) returning a :class:`plotly.graph_objects.Figure`
— no widgets, no ``Session``, no display side effects.  The explorer merely picks
one and displays it, which keeps the views unit-testable and reusable outside the
notebook.

Coordinates come from :mod:`geomulticorr.utils._pairs_geometry`, the same module
the matplotlib figures in :mod:`geomulticorr.utils.gmc_functions` use, so the two
backends draw identical curves.

Performance: every repeated element is collapsed into a **single trace** whose
segments are separated by ``NaN``.  A few hundred chords then build in ~40 ms
instead of one trace per pair, and hover stays precise because each pair also
gets one invisible anchor marker at its curve midpoint (hovering a
several-thousand-point polyline is useless).

Views never raise on degenerate input — an empty or too-small frame yields a
figure carrying an explanatory annotation, so over-restrictive explorer filters
cannot break the interactive loop.

Public API:

- :func:`figure_baseline` — Δt vs pair midpoint date (the default view).
- :func:`figure_chord` — time-proportional circular chord diagram.
- :func:`figure_network` — horizontal timeline with Bézier arcs.
- :func:`figure_dt_hist` — distribution of temporal baselines.
- :data:`VIEW_BUILDERS` / :data:`VIEW_LABELS` — dispatch tables for the explorer.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geomulticorr.utils._pairs_frame import date_summary, unique_couples, unique_dates
from geomulticorr.utils._pairs_geometry import (
    CircularTimeScale,
    arc_heights,
    arc_ylim,
    chord_polylines,
    polar_to_xy,
    polyline_point_and_tangent,
    ring_centerline,
    tangent_marker_angles,
    timeline_arc_polylines,
    to_decimal_year,
    year_label_placement,
    year_tick_segments,
)

#: Colours of the forward/backward split, matching the matplotlib figures.
DIRECTION_COLORS: dict[str, str] = {"forward": "#1f77b4", "backward": "#ff7f0e"}

#: Customdata columns carried by every per-pair hover.
_CUSTOM_COLS = ("pz", "date_i", "date_j", "sensor_i", "sensor_j", "dt_days", "direction")

#: Hover body shared by the baseline view — kept verbatim from the original explorer.
_BASELINE_HOVER = (
    "pzone=%{customdata[0]}<br>Δt=%{y:.0f} d"
    "<br>%{customdata[1]} (%{customdata[3]})"
    " → %{customdata[2]} (%{customdata[4]})<extra></extra>"
)

#: Hover body for views whose y axis is not Δt (chord, network) — Δt from customdata.
_PAIR_HOVER = (
    "dir=%{customdata[6]}<br>pzone=%{customdata[0]}<br>Δt=%{customdata[5]:.0f} d"
    "<br>%{customdata[1]} (%{customdata[3]})"
    " → %{customdata[2]} (%{customdata[4]})<extra></extra>"
)

_NODE_HOVER = "%{customdata[0]}<br>sensor=%{customdata[1]}<br>%{customdata[2]} pairs<extra></extra>"


def _customdata(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(_CUSTOM_COLS)].to_numpy()


def _auto_bezier_points(n_pairs: int, requested: int | None) -> int:
    """Fewer samples for very large networks, keeping the redraw payload bounded."""
    if requested is not None:
        return int(requested)
    return 16 if n_pairs <= 1000 else 8


def _flatten_with_gaps(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(m, n, 2)`` polylines → two flat arrays with a ``NaN`` break between curves."""
    if len(curves) == 0:
        return np.array([]), np.array([])
    m, n, _ = curves.shape
    padded = np.full((m, n + 1, 2), np.nan)
    padded[:, :n, :] = curves
    return padded[:, :, 0].ravel(), padded[:, :, 1].ravel()


def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title or None,
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=30, t=60 if title else 30, b=50),
    )
    return fig


def _empty_figure(message: str, *, title: str = "", height: int = 460) -> go.Figure:
    """A blank figure carrying an explanatory message instead of raising."""
    fig = go.Figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(size=13, color="#888"))
    return _base_layout(fig, title, height)


# ── baseline: Δt vs midpoint date ───────────────────────────────────────────────

def figure_baseline(frame: pd.DataFrame, *, title: str = "", height: int = 460) -> go.Figure:
    """Scatter of temporal baseline against pair midpoint date, split by direction.

    Answers "how many pairs, at which baselines, and when" — the view to check
    that a strategy gives even temporal coverage.

    :param frame: A pairs frame.
    :param title: Figure title.
    :param height: Figure height in pixels.
    """
    if len(frame) == 0:
        return _empty_figure("No pairs match the current filters.", title=title, height=height)

    fig = go.Figure()
    for name, symbol in (("forward", "circle"), ("backward", "diamond")):
        sub = frame[frame["direction"] == name]
        fig.add_trace(
            go.Scatter(
                x=sub["mid_dec"], y=sub["dt_days"], mode="markers", name=name,
                customdata=_customdata(sub), opacity=0.75,
                marker=dict(size=9, symbol=symbol, color=DIRECTION_COLORS[name]),
                hovertemplate=f"dir={name}<br>" + _BASELINE_HOVER,
            )
        )
    fig.update_layout(
        xaxis_title="Pair midpoint date (decimal year)",
        yaxis_title="Temporal baseline Δt (days)",
    )
    return _base_layout(fig, title, height)


# ── chord: time-proportional circle ─────────────────────────────────────────────

def figure_chord(
    frame: pd.DataFrame,
    *,
    title: str = "",
    height: int = 620,
    colorscale: str = "Jet",
    n_ring_points: int = 720,
    outer_radius: float = 1.0,
    ring_width: float = 0.045,
    link_radius: float | None = None,
    curvature: float = 0.85,
    n_bezier_points: int | None = None,
    link_color: str = "rgba(0,0,0,0.35)",
    link_width: float = 0.8,
    label_radius: float = 1.08,
    show_year_ticks: bool = True,
    show_year_labels: bool = True,
    show_arrows: bool = False,
    arrow_position: float = 0.5,
    arrow_size: float | None = None,
    arrow_color: str = "rgba(0,0,0,0.75)",
    xylim: float = 1.30,
    dedupe: bool = True,
) -> go.Figure:
    """Circular chord diagram with dates placed proportionally to time.

    Jan 1 sits at the top and time runs clockwise, so a full turn is one year and
    seasonal clustering in the archive shows up as angular clustering.

    :param frame: A pairs frame.
    :param colorscale: Plotly colorscale of the time ring.
    :param n_ring_points: Ring resolution (a single marker trace).
    :param link_radius: Chord endpoint radius; defaults to the ring mid-radius.
    :param curvature: 0 → chords hug the rim, 1 → antipodal chords cross the centre.
    :param n_bezier_points: Samples per chord; ``None`` picks 16 or 8 by pair count.
    :param show_arrows: Draw an arrowhead per chord, pointing from the left image
        to the right image of that pair.  Off by default: past roughly 150 chords
        the heads crowd each other and cost more legibility than they add.
    :param arrow_position: Where along each chord the head sits, ``0``–``1``.  The
        default of mid-chord matters — chord *endpoints* coincide with the
        acquisition angles, so end-placed heads from many pairs would pile up on a
        handful of points.
    :param arrow_size: Marker size in px; ``None`` scales it down as chords multiply.
    :param dedupe: Drop duplicate unordered couples (default ``True`` — otherwise
        redundancy strategies overplot every chord twice).  With ``dedupe=False``
        both directions of a couple are drawn, each carrying its own arrowhead.
    """
    if dedupe:
        frame = unique_couples(frame)
    if len(frame) == 0:
        return _empty_figure("No pairs match the current filters.", title=title, height=height)

    dates = unique_dates(frame)
    if len(dates) < 2:
        return _empty_figure("A chord diagram needs at least two acquisition dates.",
                             title=title, height=height)

    scale = CircularTimeScale.from_dates(dates)
    if link_radius is None:
        link_radius = outer_radius - ring_width / 2
    n_pts = _auto_bezier_points(len(frame), n_bezier_points)

    fig = go.Figure()

    # 1. time ring — plotly cannot gradient *within* a line, so a dense marker ring it is
    rx, ry, rfrac = ring_centerline(n_ring_points, outer_radius=outer_radius,
                                    ring_width=ring_width)
    fig.add_trace(
        go.Scatter(
            x=rx, y=ry, mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(color=rfrac, colorscale=colorscale, cmin=0, cmax=1,
                        size=max(ring_width * height * 0.6, 4), showscale=False),
        )
    )

    # 2. all chords in ONE trace
    curves = chord_polylines(scale.angle(frame["t_i"]), scale.angle(frame["t_j"]),
                             link_radius=link_radius, curvature=curvature, n_points=n_pts)
    cx, cy = _flatten_with_gaps(curves)
    fig.add_trace(
        go.Scatter(x=cx, y=cy, mode="lines", showlegend=False, hoverinfo="skip",
                   line=dict(color=link_color, width=link_width))
    )

    # 3. invisible per-chord anchor carrying the pair hover
    mid = curves[:, n_pts // 2, :]
    fig.add_trace(
        go.Scatter(
            x=mid[:, 0], y=mid[:, 1], mode="markers", showlegend=False,
            marker=dict(size=7, color="rgba(0,0,0,0)"),
            customdata=_customdata(frame), hovertemplate=_PAIR_HOVER,
        )
    )

    # 4. acquisition nodes
    summary = date_summary(frame)
    nx, ny = polar_to_xy(link_radius, scale.angle(dates))
    fig.add_trace(
        go.Scatter(
            x=nx, y=ny, mode="markers", showlegend=False,
            marker=dict(size=6, color=scale.fraction(dates), colorscale=colorscale,
                        cmin=0, cmax=1, line=dict(color="black", width=0.5)),
            customdata=summary[["date", "sensor", "n_pairs"]].to_numpy(),
            hovertemplate=_NODE_HOVER,
        )
    )

    # 5. year ticks
    if show_year_ticks:
        segs = year_tick_segments(scale, outer_radius=outer_radius, ring_width=ring_width)
        tx, ty = _flatten_with_gaps(segs)
        fig.add_trace(
            go.Scatter(x=tx, y=ty, mode="lines", showlegend=False, hoverinfo="skip",
                       line=dict(color="black", width=1.4))
        )

    # 6. arrowheads — appended LAST so trace indices 0-4 stay stable
    if show_arrows:
        tips, tangents = polyline_point_and_tangent(curves, position=arrow_position)
        size = (
            arrow_size
            if arrow_size is not None
            else float(np.clip(10.0 * np.sqrt(80.0 / max(len(curves), 1)), 4.0, 11.0))
        )
        fig.add_trace(
            go.Scatter(
                x=tips[:, 0], y=tips[:, 1], mode="markers", showlegend=False,
                hoverinfo="skip",
                marker=dict(symbol="triangle-up", size=size, color=arrow_color,
                            angle=tangent_marker_angles(tangents), angleref="up"),
            )
        )

    if show_year_labels:
        lx, ly, rot, ha = year_label_placement(scale, label_radius=label_radius)
        for x, y, r, a, year in zip(lx, ly, rot, ha, scale.years):
            fig.add_annotation(
                x=float(x), y=float(y), text=str(int(year)), showarrow=False,
                textangle=float(-r), xanchor={"left": "left", "right": "right"}[a],
                yanchor="middle", font=dict(size=10),
            )

    fig.update_layout(
        xaxis=dict(visible=False, range=[-xylim, xylim], fixedrange=True),
        yaxis=dict(visible=False, range=[-xylim, xylim], scaleanchor="x", scaleratio=1,
                   fixedrange=True),
        showlegend=False,
    )
    return _base_layout(fig, title, height)


# ── network: timeline + arcs ────────────────────────────────────────────────────

def _network_groups(frame: pd.DataFrame, color_by: str, n_dt_classes: int) -> list[tuple]:
    """``[(label, mask, color)]`` for the arc traces."""
    if color_by == "direction":
        return [
            (d, (frame["direction"] == d).to_numpy(), c)
            for d, c in DIRECTION_COLORS.items()
            if (frame["direction"] == d).any()
        ]
    if color_by in ("sensor", "pzone"):
        col = "sensor_i" if color_by == "sensor" else "pz"
        levels = sorted(frame[col].unique())
        palette = _qualitative_colors(len(levels))
        return [(str(lv), (frame[col] == lv).to_numpy(), palette[i]) for i, lv in enumerate(levels)]
    if color_by == "dt":
        # Plotly cannot vary colour within a line trace, so Δt is binned into
        # quantile classes — one trace each. (The matplotlib view is continuous.)
        dt = frame["dt_years"].to_numpy()
        n_cls = max(1, min(int(n_dt_classes), len(np.unique(dt))))
        edges = np.quantile(dt, np.linspace(0, 1, n_cls + 1))
        edges[-1] += 1e-9
        groups = []
        for k in range(n_cls):
            mask = (dt >= edges[k]) & (dt < edges[k + 1])
            if mask.any():
                shade = k / max(n_cls - 1, 1)
                groups.append((f"{edges[k]:.2f}–{edges[k + 1]:.2f} yr", mask,
                               _sample_colorscale("Plasma_r", shade)))
        return groups
    raise ValueError(
        f"Unknown color_by '{color_by}'. Valid options: 'direction', 'sensor', 'pzone', 'dt'."
    )


def _qualitative_colors(n: int) -> list[str]:
    from plotly.colors import qualitative

    palette = qualitative.T10 if n <= len(qualitative.T10) else qualitative.Alphabet
    return [palette[i % len(palette)] for i in range(max(n, 1))]


def _sample_colorscale(name: str, value: float) -> str:
    from plotly.colors import sample_colorscale

    return sample_colorscale(name, [float(np.clip(value, 0, 1))])[0]


def figure_network(
    frame: pd.DataFrame,
    *,
    title: str = "",
    height: int = 460,
    color_by: str = "dt",
    n_bezier_points: int | None = None,
    n_dt_classes: int = 5,
    mirror_direction: bool = True,
    dedupe: bool = False,
) -> go.Figure:
    """Horizontal timeline with one Bézier arc per pair.

    Arc height grows with the temporal baseline, so the network's redundancy
    structure — which dates are connected, and how far apart — reads directly.
    With *mirror_direction* the timeline becomes two-sided: forward pairs arch
    above it and backward pairs below, making direction visible as geometry.

    :param frame: A pairs frame.
    :param color_by: ``"dt"`` | ``"sensor"`` | ``"pzone"`` | ``"direction"``.
        Defaults to ``"dt"`` because mirroring already encodes direction.
    :param n_dt_classes: Number of Δt quantile classes when ``color_by="dt"``
        (plotly needs discrete classes; the matplotlib view is continuous).
    :param mirror_direction: Draw backward pairs below the axis.
    :param dedupe: Drop duplicate unordered couples; ``False`` by default so the
        drawn arcs match the number of pairs actually built — and so mirroring
        has both directions to show.
    """
    if dedupe:
        frame = unique_couples(frame)
    if len(frame) == 0:
        return _empty_figure("No pairs match the current filters.", title=title, height=height)

    n_pts = _auto_bezier_points(len(frame), n_bezier_points)
    x1 = frame["dec_i"].to_numpy()
    x2 = frame["dec_j"].to_numpy()
    backward = (frame["direction"] == "backward").to_numpy()
    heights = arc_heights(frame["dt_years"].to_numpy(), backward=backward,
                          mirror=mirror_direction)
    curves = timeline_arc_polylines(x1, x2, heights, n_points=n_pts)

    fig = go.Figure()
    for label, mask, color in _network_groups(frame, color_by, n_dt_classes):
        gx, gy = _flatten_with_gaps(curves[mask])
        fig.add_trace(
            go.Scatter(x=gx, y=gy, mode="lines", name=label, hoverinfo="skip",
                       line=dict(color=color, width=1.4), opacity=0.75)
        )

    # invisible per-arc anchor at the apex, carrying the pair hover
    apex = curves[:, n_pts // 2, :]
    fig.add_trace(
        go.Scatter(x=apex[:, 0], y=apex[:, 1], mode="markers", showlegend=False,
                   marker=dict(size=7, color="rgba(0,0,0,0)"),
                   customdata=_customdata(frame), hovertemplate=_PAIR_HOVER)
    )

    # acquisition nodes on the timeline
    summary = date_summary(frame)
    node_dec = to_decimal_year(pd.DatetimeIndex(summary["t"]))
    fig.add_trace(
        go.Scatter(x=node_dec, y=np.zeros_like(node_dec), mode="markers", showlegend=False,
                   marker=dict(size=8, color="black", line=dict(color="white", width=0.5)),
                   customdata=summary[["date", "sensor", "n_pairs"]].to_numpy(),
                   hovertemplate=_NODE_HOVER)
    )

    pad = (node_dec.max() - node_dec.min()) * 0.03 if len(node_dec) > 1 else 0.1
    y_bottom, y_top = arc_ylim(heights)
    fig.update_layout(
        xaxis=dict(title="Acquisition date (decimal year)",
                   range=[node_dec.min() - pad, node_dec.max() + pad]),
        yaxis=dict(visible=False, range=[y_bottom, y_top]),
        legend=dict(title=color_by, orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0),
        shapes=[dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=0,
                     line=dict(color="black", width=1))],
    )

    # Direction legend as annotations (not traces), so trace counts stay stable.
    if mirror_direction:
        # Forward-only pair sets have nothing to put below the axis — say so, or
        # the mirror looks broken rather than simply inapplicable.
        forward_label = (
            "▲ above: forward (i → j)" if backward.any()
            else "▲ forward-only pair set — no backward pairs to mirror"
        )
        if (~backward).any():
            fig.add_annotation(
                xref="paper", yref="paper", x=0.004, y=0.99, xanchor="left",
                yanchor="top", showarrow=False, text=forward_label,
                font=dict(size=10, color=DIRECTION_COLORS["forward"]),
            )
        if backward.any():
            fig.add_annotation(
                xref="paper", yref="paper", x=0.004, y=0.01, xanchor="left",
                yanchor="bottom", showarrow=False, text="▼ below: backward (j → i)",
                font=dict(size=10, color=DIRECTION_COLORS["backward"]),
            )
    return _base_layout(fig, title, height)


# ── Δt histogram ────────────────────────────────────────────────────────────────

def figure_dt_hist(
    frame: pd.DataFrame,
    *,
    title: str = "",
    height: int = 460,
    nbins: int = 40,
    split_direction: bool = True,
) -> go.Figure:
    """Distribution of temporal baselines, with median and mean markers.

    The quickest check that a strategy delivers the Δt coverage a study needs.

    :param frame: A pairs frame.
    :param nbins: Approximate number of histogram bins.
    :param split_direction: Overlay forward and backward separately.
    """
    if len(frame) == 0:
        return _empty_figure("No pairs match the current filters.", title=title, height=height)

    fig = go.Figure()
    if split_direction:
        for name, color in DIRECTION_COLORS.items():
            sub = frame[frame["direction"] == name]
            if len(sub) == 0:
                continue
            fig.add_trace(
                go.Histogram(x=sub["dt_days"], name=name, nbinsx=nbins, opacity=0.7,
                             marker=dict(color=color),
                             hovertemplate="Δt %{x} d<br>%{y} pairs<extra>%{fullData.name}</extra>")
            )
    else:
        fig.add_trace(
            go.Histogram(x=frame["dt_days"], name="all pairs", nbinsx=nbins, opacity=0.8,
                         marker=dict(color="#7f7f7f"),
                         hovertemplate="Δt %{x} d<br>%{y} pairs<extra></extra>")
        )

    dt = frame["dt_days"].to_numpy()
    for value, label, dash in ((np.median(dt), "median", "dash"), (dt.mean(), "mean", "dot")):
        fig.add_vline(x=float(value), line=dict(color="black", width=1.2, dash=dash),
                      annotation_text=f"{label} {value:.0f} d", annotation_position="top")

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Temporal baseline Δt (days)",
        yaxis_title="Number of pairs",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return _base_layout(fig, title, height)


#: Explorer dispatch table — view key → builder.
VIEW_BUILDERS: dict[str, Callable[..., go.Figure]] = {
    "baseline": figure_baseline,
    "chord": figure_chord,
    "network": figure_network,
    "dt_hist": figure_dt_hist,
}

#: Human-readable labels for the explorer dropdown, in display order.
VIEW_LABELS: dict[str, str] = {
    "baseline": "Δt vs midpoint",
    "chord": "Chord",
    "network": "Network",
    "dt_hist": "Δt histogram",
}
