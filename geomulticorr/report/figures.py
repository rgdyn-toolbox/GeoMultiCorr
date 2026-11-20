#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# figures.py
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
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------- #
"""Report figures drawn from stage records (static PNG, matplotlib only).

The ``plot_*`` functions follow the house plotting conventions — ``ax=None``,
return ``(fig, ax)``, ``fig.tight_layout()``, never save themselves; the
report's :func:`save` owns the file and closes the figure in a ``finally``.
They take record data (lists of dicts), never rasters.

PNG only: the PDF renderer does not run JavaScript, so the interactive plotly
twins are linked from the report, never embedded (and ``kaleido`` is not a
dependency).
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import matplotlib

matplotlib.use("agg")  # noqa: E402  (report rendering is always headless)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _dates(values) -> list[datetime]:
    out = []
    for v in values:
        try:
            out.append(datetime.fromisoformat(str(v)[:10]))
        except ValueError:
            out.append(None)
    return out


def plot_archive_timeline(thumbs: list[dict], *, ax=None, fig_name: str | None = None):
    """One row per sensor, one tick per acquisition."""
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8, 0.6 + 0.45 * max(1, len({t["sensor"] for t in thumbs})))
        )
    else:
        fig = ax.figure
    sensors = sorted({t["sensor"] for t in thumbs})
    cmap = matplotlib.colormaps["tab10"].resampled(max(1, len(sensors)))
    for i, sensor in enumerate(sensors):
        dates = [
            d for d in _dates(t["date"] for t in thumbs if t["sensor"] == sensor) if d
        ]
        ax.scatter(dates, [i] * len(dates), marker="|", s=220, color=cmap(i), lw=1.5)
    ax.set_yticks(
        range(len(sensors)),
        [f"{s} ({sum(t['sensor'] == s for t in thumbs)})" for s in sensors],
    )
    ax.set_ylim(-0.6, len(sensors) - 0.4)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(fig_name or "Acquisitions per sensor")
    fig.tight_layout()
    return fig, ax


def plot_dt_histogram(pairs: list[dict], *, ax=None, fig_name: str | None = None):
    """Temporal baselines of the committed pairs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure
    dts = np.array(
        [p["dt_days"] for p in pairs if isinstance(p.get("dt_days"), (int, float))]
    )
    dts = np.abs(dts)
    if dts.size:
        ax.hist(dts, bins=min(40, max(10, dts.size)), color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Δt (days)")
    ax.set_ylabel("pairs")
    ax.set_title(fig_name or "Temporal baselines")
    fig.tight_layout()
    return fig, ax


def plot_nmad_before_after(pairs: list[dict], *, ax=None, fig_name: str | None = None):
    """Per-pair NMAD on stable ground before vs after the corrections (EW and NS)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    top = 0.0
    for comp, color in (("ew", "#C44E52"), ("ns", "#4C72B0")):
        raw = np.array([p.get(f"{comp}_nmad_raw") for p in pairs], dtype=float)
        corr = np.array([p.get(f"{comp}_nmad_corr") for p in pairs], dtype=float)
        ok = np.isfinite(raw) & np.isfinite(corr)
        if ok.any():
            ax.scatter(
                raw[ok], corr[ok], s=12, alpha=0.6, color=color, label=comp.upper()
            )
            top = max(top, float(np.nanmax(raw[ok])), float(np.nanmax(corr[ok])))
    if top > 0:
        ax.plot([0, top], [0, top], color="0.5", lw=1, ls="--", label="no change")
        ax.set_xlim(0, top * 1.05)
        ax.set_ylim(0, top * 1.05)
        ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel("NMAD raw (m)")
    ax.set_ylabel("NMAD corrected (m)")
    ax.set_title(fig_name or "Stable-ground NMAD per pair")
    fig.tight_layout()
    return fig, ax


def save(fig, path: pathlib.Path, dpi: int = 150) -> pathlib.Path:
    """Write *fig* as PNG and close it (always)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    return path
