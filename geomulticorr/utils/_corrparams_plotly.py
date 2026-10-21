#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _corrparams_plotly.py
# creation date: 2026-09-10.
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
"""Interactive plotly views of a correlation-parameters frame.

Four views, dispatched by :data:`VIEW_BUILDERS`.  The important one is
:func:`figure_design_map`: because both feasibility boundaries are ``v ∝ 1/Δt``,
they plot as parallel straight lines in log-log, and every pair lands visibly
below the detection floor, inside the band, or above the search ceiling.  That
single picture is what the explorer exists to show.

**These functions must never raise** — they run inside the widget loop, where an
exception reaches only the kernel log, which VSCode hides.  Degenerate input
returns :func:`_empty_figure` carrying an explanation.

Plain ``go.Figure`` throughout, never ``go.FigureWidget``: since plotly 6 the
latter is an anywidget whose front-end JS is not bundled with the VSCode Jupyter
extension.

Public API: :func:`figure_design_map`, :func:`figure_kernel_search`,
:func:`figure_snr`, :func:`figure_cost`, plus :data:`VIEW_BUILDERS` and
:data:`VIEW_LABELS`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geomulticorr.correlation.corr_params import (
    DAYS_PER_YEAR,
    TEXTURE_FLOOR_PX,
    feasibility_bounds,
    strain_kernel_limit_px,
)

#: Colour of each boundary and the pairs, shared with the matplotlib twins so a
#: PNG cannot diverge from its HTML counterpart.
DESIGN_COLORS: dict[str, str] = {
    "floor": "#c0392b",      # detection floor — below it is noise
    "ceiling": "#2471a3",    # search reach — above it the box rails
    "band": "rgba(39,174,96,0.10)",
    "pairs": "#34495e",
    "flagged": "#e67e22",
    "limit": "#7f8c8d",
}

#: Qualitative palette for per-group series.  Deliberately short: a design map
#: with more than a handful of groups is telling you the binning is too fine.
GROUP_PALETTE: tuple[str, ...] = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
)


def _empty_figure(message: str, *, title: str = "", height: int = 460) -> go.Figure:
    """A blank figure carrying an explanatory message instead of raising."""
    fig = go.Figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper",
        x=0.5, y=0.5, font=dict(size=13, color="#888"),
    )
    return _base_layout(fig, title, height)


def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title or None,
        template="plotly_white",
        height=height,
        margin=dict(l=70, r=30, t=60 if title else 30, b=55),
    )
    return fig


def _group_color(group: str, groups: list[str]) -> str:
    """Stable colour for *group* — by position, so a redraw does not reshuffle."""
    return GROUP_PALETTE[groups.index(group) % len(GROUP_PALETTE)]


def _explicit_search_px(frame: pd.DataFrame) -> float | None:
    """Largest explicit search half-width in *frame*, or ``None``.

    ``None`` means **no pair has an explicit box** — they are all on ASP's
    automatic range. That is a legitimate, common state once the Δt threshold is
    in play, and it must be distinguishable from "the box is small": a search
    ceiling drawn from a fabricated default would describe a constraint that
    does not exist.

    ``np.nanmax`` is avoided deliberately — it warns and returns nan on an
    all-NaN slice, which would then poison every boundary value.
    """
    values = frame["search_px"].to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else None


def _safe_search_px(frame: pd.DataFrame, default: float = 5.0) -> float:
    """Backwards-compatible wrapper: :func:`_explicit_search_px` with a default."""
    value = _explicit_search_px(frame)
    return float(default) if value is None else value


def _dt_grid(frame: pd.DataFrame, n: int = 64) -> np.ndarray:
    """A log-spaced Δt grid spanning the frame, padded so lines reach the edges."""
    dt = frame["dt_days"].to_numpy(dtype="float64")
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.array([])
    lo, hi = float(dt.min()), float(dt.max())
    if hi <= lo:
        lo, hi = lo / 2.0, hi * 2.0
    return np.logspace(np.log10(lo * 0.7), np.log10(hi * 1.4), n)


# ── design map: the key figure ──────────────────────────────────────────────── #

def figure_design_map(
    frame: pd.DataFrame,
    *,
    title: str = "",
    height: int = 520,
    k: float = 3.0,
    c_px: float = 0.15,
    coreg_px: float = 2.0,
    margin_px: float = 2.0,
    tau_days: float | None = None,
    color_by: str = "group",
) -> go.Figure:
    """Log-log Δt versus velocity, with the feasibility band and every pair.

    The two boundaries are parallel lines of slope -1 (guide §2.1), so the band
    between them has a constant width — the dynamic range ``S/(k·c)``.  Pairs
    below the floor measure noise; pairs above the ceiling rail against the
    search box.

    :param frame: A correlation-parameters frame.
    :param title: Figure title.
    :param height: Figure height, pixels.
    :param k: Confidence multiplier, positioning the detection floor.
    :param c_px: Subpixel matching precision, pixels.
    :param coreg_px: Co-registration allowance, pixels.
    :param margin_px: Search safety margin, pixels.
    :param tau_days: Decorrelation time; draws the baseline cut when given.
    :param color_by: ``"group"``, ``"sensor"`` or ``"none"``.
    :returns: A figure; an annotated empty one for degenerate input.
    """
    if frame is None or len(frame) == 0:
        return _empty_figure("No pairs to plot", title=title, height=height)

    grid = _dt_grid(frame)
    if grid.size == 0:
        return _empty_figure("No usable temporal baselines", title=title, height=height)

    fig = go.Figure()

    # Boundaries are drawn from the median resolution and the largest search in
    # the frame: a mixed-sensor frame has no single band, and showing the most
    # permissive one would hide pairs that are actually out of reach.
    resolution = float(np.nanmedian(frame["resolution_m"].to_numpy(dtype="float64")))
    # None when every pair is on ASP's automatic range: there is then no search
    # reach to draw, and a line from a fabricated default would describe nothing.
    search = _explicit_search_px(frame)
    bounds = feasibility_bounds(
        grid, resolution, c_px, search if search is not None else 0.0,
        k=k, coreg_px=coreg_px, margin_px=margin_px,
    )

    if search is not None:
        # Shaded band first, so markers and lines sit on top of it.
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([bounds["v_max"], bounds["v_min"][::-1]]),
                fill="toself", fillcolor=DESIGN_COLORS["band"],
                line=dict(width=0), hoverinfo="skip",
                name="measurable", showlegend=True,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=grid, y=bounds["v_min"], mode="lines",
            line=dict(color=DESIGN_COLORS["floor"], width=2),
            name=f"detection floor (k={k:g})",
            hovertemplate="Δt=%{x:.0f} d<br>v_min=%{y:.2f} m/yr<extra></extra>",
        )
    )
    if search is not None:
        fig.add_trace(
            go.Scatter(
                x=grid, y=bounds["v_max"], mode="lines",
                line=dict(color=DESIGN_COLORS["ceiling"], width=2, dash="dash"),
                name=f"search reach (S={search:.0f} px)",
                hovertemplate="Δt=%{x:.0f} d<br>v_max=%{y:.2f} m/yr<extra></extra>",
            )
        )
    else:
        # Say so, rather than leaving a reader to wonder where the ceiling went.
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=DESIGN_COLORS["limit"], width=2, dash="dash"),
                name="search reach: ASP auto (no explicit box)",
                hoverinfo="skip",
            )
        )

    if tau_days:
        fig.add_vline(
            x=float(tau_days), line=dict(color=DESIGN_COLORS["limit"], width=1.5, dash="dot"),
            annotation_text=f"τ = {float(tau_days):.0f} d", annotation_position="top",
        )

    _add_pair_markers(fig, frame, color_by, k)

    fig.update_xaxes(type="log", title_text="Temporal baseline Δt (days)")
    fig.update_yaxes(type="log", title_text="Surface velocity v (m/yr)")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return _base_layout(fig, title or "Correlation design map", height)


def _add_pair_markers(fig: go.Figure, frame: pd.DataFrame, color_by: str, k: float) -> None:
    """One batched trace per colour group — never one trace per pair."""
    hover = (
        "%{text}"
        "<br>Δt=%{x:.0f} d"
        "<br>v=%{y:.2f} m/yr"
        "<br>D=%{customdata[0]:.2f} m (%{customdata[1]:.1f} px)"
        "<br>SNR=%{customdata[2]:.2f}"
        "<br>kernel=%{customdata[3]} px · search=%{customdata[4]:.0f} px"
        "<extra></extra>"
    )
    column = {"group": "group", "sensor": "sensor_i"}.get(color_by)
    keys = [""] if column is None else sorted(frame[column].astype(str).unique())

    for key in keys:
        sub = frame if column is None else frame[frame[column].astype(str) == key]
        if len(sub) == 0:
            continue
        below = sub["snr"].to_numpy(dtype="float64") < float(k)
        fig.add_trace(
            go.Scatter(
                x=sub["dt_days"].tolist(),
                y=sub["velocity_m_yr"].tolist(),
                mode="markers",
                name=key or "pairs",
                text=sub["pa_key"].tolist(),
                customdata=[
                    [d, p, s, kp, sp]
                    for d, p, s, kp, sp in zip(
                        sub["disp_m"], sub["disp_px"], sub["snr"],
                        sub["kernel_px"], sub["search_px"],
                    )
                ],
                marker=dict(
                    size=10,
                    color=_group_color(key, keys) if column is not None else DESIGN_COLORS["pairs"],
                    symbol=np.where(below, "x", "circle"),
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=hover,
            )
        )


# ── kernel and search versus baseline ───────────────────────────────────────── #

def figure_kernel_search(
    frame: pd.DataFrame,
    *,
    title: str = "",
    height: int = 520,
    strain_rate_per_yr: float | None = None,
) -> go.Figure:
    """Derived kernel and search half-width against temporal baseline.

    Shows the texture floor and, when a strain rate is given, the strain ceiling
    — the two bounds that squeeze the kernel from below and above (guide §3.1).

    :param frame: A correlation-parameters frame.
    :param title: Figure title.
    :param height: Figure height, pixels.
    :param strain_rate_per_yr: Strain rate for the ceiling curve, or None.
    :returns: A figure; an annotated empty one for degenerate input.
    """
    if frame is None or len(frame) == 0:
        return _empty_figure("No pairs to plot", title=title, height=height)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["dt_days"].tolist(), y=frame["kernel_px"].tolist(),
            mode="markers", name="kernel (px)",
            text=frame["pa_key"].tolist(),
            marker=dict(size=9, color=GROUP_PALETTE[0]),
            hovertemplate="%{text}<br>Δt=%{x:.0f} d<br>kernel=%{y} px<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["dt_days"].tolist(), y=frame["search_px"].tolist(),
            mode="markers", name="search half-width (px)",
            text=frame["pa_key"].tolist(),
            marker=dict(size=9, symbol="diamond", color=GROUP_PALETTE[1]),
            hovertemplate="%{text}<br>Δt=%{x:.0f} d<br>search=%{y:.0f} px<extra></extra>",
        )
    )

    fig.add_hline(
        y=TEXTURE_FLOOR_PX,
        line=dict(color=DESIGN_COLORS["limit"], width=1.5, dash="dot"),
        annotation_text=f"texture floor ({TEXTURE_FLOOR_PX} px)",
        annotation_position="bottom right",
    )

    if strain_rate_per_yr:
        grid = _dt_grid(frame)
        if grid.size:
            limit = [strain_kernel_limit_px(strain_rate_per_yr, d) for d in grid]
            fig.add_trace(
                go.Scatter(
                    x=grid, y=limit, mode="lines", name="strain ceiling",
                    line=dict(color=DESIGN_COLORS["floor"], width=2, dash="dash"),
                    hovertemplate="Δt=%{x:.0f} d<br>K ≤ %{y:.1f} px<extra></extra>",
                )
            )

    fig.update_xaxes(type="log", title_text="Temporal baseline Δt (days)")
    # Log y, for the same reason as the matplotlib twin: the strain ceiling
    # spans hundreds of pixels while the derived kernels span tens.
    fig.update_yaxes(type="log", title_text="pixels")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return _base_layout(fig, title or "Kernel and search versus baseline", height)


# ── SNR ─────────────────────────────────────────────────────────────────────── #

def figure_snr(
    frame: pd.DataFrame, *, title: str = "", height: int = 460, k: float = 3.0
) -> go.Figure:
    """Signal-to-noise ratio against baseline, with the ``k`` threshold.

    :param frame: A correlation-parameters frame.
    :param title: Figure title.
    :param height: Figure height, pixels.
    :param k: Confidence multiplier, drawn as the detection threshold.
    :returns: A figure; an annotated empty one for degenerate input.
    """
    if frame is None or len(frame) == 0:
        return _empty_figure("No pairs to plot", title=title, height=height)

    snr_values = frame["snr"].to_numpy(dtype="float64")
    below = snr_values < float(k)
    fig = go.Figure(
        go.Scatter(
            x=frame["dt_days"].tolist(), y=snr_values.tolist(), mode="markers",
            name="pairs", text=frame["pa_key"].tolist(),
            marker=dict(
                size=10,
                color=np.where(below, DESIGN_COLORS["flagged"], DESIGN_COLORS["pairs"]),
                symbol=np.where(below, "x", "circle"),
            ),
            hovertemplate="%{text}<br>Δt=%{x:.0f} d<br>SNR=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=float(k), line=dict(color=DESIGN_COLORS["floor"], width=2),
        annotation_text=f"detection threshold k={k:g}", annotation_position="top right",
    )
    fig.update_xaxes(type="log", title_text="Temporal baseline Δt (days)")
    fig.update_yaxes(type="log", title_text="SNR = v·Δt / σ_D")
    n_below = int(np.sum(below))
    suffix = f" — {n_below} below the floor" if n_below else ""
    return _base_layout(fig, title or f"Detectability{suffix}", height)


# ── cost ────────────────────────────────────────────────────────────────────── #

def figure_cost(frame: pd.DataFrame, *, title: str = "", height: int = 460) -> go.Figure:
    """Summed relative correlation cost per parameter group.

    Cost scales as ``K² · S_x · S_y``, so this is where an oversized search box
    announces itself before it costs a night of compute (guide §6.3).

    :param frame: A correlation-parameters frame.
    :param title: Figure title.
    :param height: Figure height, pixels.
    :returns: A figure; an annotated empty one for degenerate input.
    """
    if frame is None or len(frame) == 0:
        return _empty_figure("No pairs to plot", title=title, height=height)

    grouped = frame.groupby("group", sort=True)
    groups = list(grouped.groups.keys())
    totals = grouped["cost_index"].sum().to_numpy()
    counts = grouped.size().to_numpy()

    fig = go.Figure(
        go.Bar(
            x=groups, y=totals,
            marker_color=[_group_color(g, groups) for g in groups],
            customdata=np.column_stack([counts, totals / np.maximum(counts, 1)]),
            hovertemplate=(
                "%{x}<br>total cost=%{y:.1f}×"
                "<br>%{customdata[0]} pairs · %{customdata[1]:.2f}× each<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="parameter group")
    fig.update_yaxes(title_text="relative cost (1.0 = ASP defaults)")
    total = float(np.nansum(totals))
    return _base_layout(fig, title or f"Correlation cost — {total:.0f}× total", height)


#: Explorer dispatch table — view key → builder.
VIEW_BUILDERS: dict[str, Callable[..., go.Figure]] = {
    "design_map": figure_design_map,
    "kernel_search": figure_kernel_search,
    "snr": figure_snr,
    "cost": figure_cost,
}

#: Human-readable labels for the explorer dropdown, in display order.
VIEW_LABELS: dict[str, str] = {
    "design_map": "Design map",
    "kernel_search": "Kernel & search",
    "snr": "Detectability",
    "cost": "Cost",
}
