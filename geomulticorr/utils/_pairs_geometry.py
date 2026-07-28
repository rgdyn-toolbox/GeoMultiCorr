#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _pairs_geometry.py
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
"""Pure-geometry helpers for pair-network layouts.

All functions operate on plain NumPy arrays and pandas timestamps — no
matplotlib, no plotly, no geoutils.  They exist so the matplotlib static figures
(:mod:`geomulticorr.utils.gmc_functions`) and the interactive plotly views
(:mod:`geomulticorr.utils._pairs_plotly`) compute *identical* coordinates from a
single implementation.

Same design intent as :mod:`geomulticorr.corrections.fit`: the maths lives in a
dependency-light module underneath the rendering layer.

Layout conventions
------------------

- **Decimal year** — ``year + (t - Jan1(year)) / 365.25 days``.  This is the
  formula the pair plots have always used; it is *not* leap-year aware, and is
  kept bit-for-bit so existing figures do not shift.
- **Circular time scale** — the circle spans Jan 1 of the first year to Jan 1 of
  (last year + 1).  Jan 1 sits at the **top** and time runs **clockwise**, i.e.
  ``angle = pi/2 - 2*pi*fraction``.
- **Chords** are quadratic Bézier curves whose control point is pulled towards
  the centre by ``curvature``, so short-Δt pairs stay near the rim and long-Δt
  pairs bow across the disc.
- **Timeline arcs** are cubic Béziers over a horizontal date axis.

Every function returns **coordinate arrays, never artists**.

Public API:

- :func:`to_decimal_year` — the single decimal-year implementation.
- :class:`CircularTimeScale` — date → angle mapping shared by both backends.
- :func:`polar_to_xy`, :func:`bezier_sample` — low-level primitives.
- :func:`chord_control_points` / :func:`chord_polylines` — circular chords.
- :func:`polyline_point_and_tangent` / :func:`tangent_marker_angles` — arrowheads.
- :func:`arc_heights` / :func:`arc_ylim` — signed timeline-arc heights and limits.
- :func:`timeline_arc_control_points` / :func:`timeline_arc_polylines` — arcs.
- :func:`color_ring_mesh` / :func:`ring_centerline` — the time colour ring.
- :func:`year_tick_segments` / :func:`year_label_placement` — ring decorations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

#: Days per year used by the decimal-year convention (not leap-aware — see module docstring).
DAYS_PER_YEAR: float = 365.25

#: Default radius at which chord endpoints sit (ring mid-radius for the default ring).
DEFAULT_LINK_RADIUS: float = 0.9775

DateLike = str | pd.Timestamp | np.datetime64


def _as_index(dates) -> pd.DatetimeIndex:
    """Coerce anything date-like (scalar or sequence) to a ``DatetimeIndex``."""
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    if isinstance(dates, (str, pd.Timestamp, np.datetime64)):
        return pd.DatetimeIndex([pd.Timestamp(dates)])
    return pd.DatetimeIndex(pd.to_datetime(list(dates)))


def _is_scalar_date(dates) -> bool:
    return isinstance(dates, (str, pd.Timestamp, np.datetime64))


# ── decimal year ────────────────────────────────────────────────────────────────

def to_decimal_year(dates: DateLike | Sequence[DateLike]):
    """Convert date(s) to decimal year.

    ``year + (t - Jan1(year)) / 365.25 days`` — the convention used by every pair
    plot in the package.  Consolidates three previously duplicated inline
    implementations.

    :param dates: A single date-like, or any sequence of them.
    :returns: ``float`` for a scalar input, ``np.ndarray`` for a sequence.
    """
    idx = _as_index(dates)
    jan1 = pd.to_datetime(pd.Series(idx.year, dtype="int64").astype(str) + "-01-01")
    dec = idx.year.to_numpy(dtype=float) + (
        (idx.to_numpy() - jan1.to_numpy()) / np.timedelta64(1, "ns")
    ) / (DAYS_PER_YEAR * 24 * 3600 * 1e9)
    if _is_scalar_date(dates):
        return float(dec[0])
    return dec


# ── circular time scale ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CircularTimeScale:
    """Maps dates onto a circle: Jan 1 at the top, time running clockwise.

    The circle spans whole years — from Jan 1 of the first year to Jan 1 of the
    year *after* the last one — so a full turn is always an integer number of
    years and seasonal alignment is readable.

    :ivar start: Jan 1 of the first year covered.
    :ivar end: Jan 1 of the year after the last one covered.
    """

    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_dates(cls, dates: Sequence[DateLike]) -> "CircularTimeScale":
        """Build the scale spanning the whole years touched by *dates*."""
        idx = _as_index(dates)
        if len(idx) == 0:
            raise ValueError("Cannot build a circular time scale from an empty date list.")
        return cls(
            start=pd.Timestamp(year=int(idx.min().year), month=1, day=1),
            end=pd.Timestamp(year=int(idx.max().year) + 1, month=1, day=1),
        )

    @property
    def total_seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    def fraction(self, dates: Sequence[DateLike]) -> np.ndarray:
        """Position of *dates* along the circle, in ``[0, 1)`` for in-range dates."""
        idx = _as_index(dates)
        return (
            (idx.to_numpy() - np.datetime64(self.start)) / np.timedelta64(1, "s")
        ).astype(float) / self.total_seconds

    def angle(self, dates: Sequence[DateLike]) -> np.ndarray:
        """Angle (radians) of *dates*: ``pi/2`` at Jan 1, decreasing clockwise."""
        return np.pi / 2 - 2 * np.pi * self.fraction(dates)

    @property
    def years(self) -> np.ndarray:
        """Integer years covered, first to last inclusive."""
        return np.arange(int(self.start.year), int(self.end.year))

    def year_angles(self) -> np.ndarray:
        """Angle of Jan 1 of each year in :attr:`years`."""
        return self.angle([pd.Timestamp(year=int(y), month=1, day=1) for y in self.years])


# ── primitives ──────────────────────────────────────────────────────────────────

def polar_to_xy(radius, theta) -> tuple[np.ndarray, np.ndarray]:
    """Polar → cartesian, broadcasting *radius* against *theta*."""
    radius = np.asarray(radius, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return radius * np.cos(theta), radius * np.sin(theta)


def bezier_sample(control_pts: np.ndarray, n_points: int = 16) -> np.ndarray:
    """Sample quadratic or cubic Bézier curves via the Bernstein basis.

    :param control_pts: ``(m, k, 2)`` control nets, ``k`` = 3 (quadratic) or
        4 (cubic).  A single ``(k, 2)`` net is accepted and promoted to ``m=1``.
    :param n_points: Samples per curve, endpoints included.
    :returns: ``(m, n_points, 2)`` polylines.
    """
    pts = np.asarray(control_pts, dtype=float)
    if pts.ndim == 2:
        pts = pts[None, :, :]
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError(f"control_pts must have shape (m, k, 2); got {pts.shape}.")
    k = pts.shape[1]
    if k not in (3, 4):
        raise ValueError(f"Only quadratic (k=3) and cubic (k=4) curves are supported; got k={k}.")
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")

    s = np.linspace(0.0, 1.0, n_points)[None, :, None]  # (1, n, 1)
    if k == 3:
        basis = [(1 - s) ** 2, 2 * (1 - s) * s, s ** 2]
    else:
        basis = [(1 - s) ** 3, 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s ** 2, s ** 3]
    return sum(b * pts[:, i, None, :] for i, b in enumerate(basis))


# ── circular chords ─────────────────────────────────────────────────────────────

def chord_control_points(
    theta_i,
    theta_j,
    *,
    link_radius: float = DEFAULT_LINK_RADIUS,
    curvature: float = 0.85,
) -> np.ndarray:
    """Quadratic control nets for chords between angles *theta_i* and *theta_j*.

    The control point sits on the angular bisector, pulled inwards by an amount
    proportional to the angular separation: neighbouring dates give an almost
    straight rim-hugging link, opposite dates bow through the centre.

    :param theta_i: Start angles (radians), array-like.
    :param theta_j: End angles (radians), array-like, same length.
    :param link_radius: Radius of the chord endpoints.
    :param curvature: 0 → straight-ish chords along the rim, 1 → control point at
        the centre for antipodal dates.
    :returns: ``(m, 3, 2)`` control nets.
    """
    ti = np.atleast_1d(np.asarray(theta_i, dtype=float))
    tj = np.atleast_1d(np.asarray(theta_j, dtype=float))
    if ti.shape != tj.shape:
        raise ValueError(f"theta_i and theta_j must have the same shape; got {ti.shape} vs {tj.shape}.")

    x1, y1 = polar_to_xy(link_radius, ti)
    x2, y2 = polar_to_xy(link_radius, tj)

    # Angular bisector, and the (signed-free) angular separation in [0, pi].
    mid_theta = np.arctan2(np.sin(ti) + np.sin(tj), np.cos(ti) + np.cos(tj))
    delta = np.abs(np.arctan2(np.sin(tj - ti), np.cos(tj - ti)))

    control_radius = link_radius * (1 - curvature * np.sin(delta / 2))
    xc, yc = polar_to_xy(control_radius, mid_theta)

    return np.stack(
        [np.column_stack([x1, y1]), np.column_stack([xc, yc]), np.column_stack([x2, y2])],
        axis=1,
    )


def chord_polylines(
    theta_i,
    theta_j,
    *,
    link_radius: float = DEFAULT_LINK_RADIUS,
    curvature: float = 0.85,
    n_points: int = 16,
) -> np.ndarray:
    """Sampled chord curves — :func:`chord_control_points` + :func:`bezier_sample`.

    :returns: ``(m, n_points, 2)`` polylines, ready for a matplotlib
        ``LineCollection`` or a NaN-separated plotly line trace.
    """
    return bezier_sample(
        chord_control_points(theta_i, theta_j, link_radius=link_radius, curvature=curvature),
        n_points=n_points,
    )


# ── arrowheads ──────────────────────────────────────────────────────────────────

def polyline_point_and_tangent(
    curves: np.ndarray,
    *,
    position: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit direction of travel at a fractional position along polylines.

    Used to place arrowheads.  ``position=0.5`` (the default) puts the head at the
    middle of each curve, which matters for chords: their endpoints sit *on* the
    acquisition-date angles, so end-placed heads from many pairs would stack on a
    handful of points, while mid-curve they fan out and stay legible.

    :param curves: ``(m, n, 2)`` polylines, e.g. from :func:`chord_polylines`.
    :param position: Where along each curve, ``0.0`` = start, ``1.0`` = end.
    :returns: ``((m, 2) points, (m, 2) unit tangents)``.  Degenerate curves (all
        samples identical) yield a zero tangent rather than NaN.
    """
    pts = np.asarray(curves, dtype=float)
    if pts.ndim == 2:
        pts = pts[None, :, :]
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError(f"curves must have shape (m, n, 2); got {pts.shape}.")
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must lie in [0, 1]; got {position}.")

    n = pts.shape[1]
    if n < 2:
        raise ValueError("curves must hold at least two samples per polyline.")

    k = int(round(position * (n - 1)))
    points = pts[:, k, :]

    # Central difference where possible, one-sided at the ends; widen the stencil
    # while a curve is locally degenerate so a repeated sample never yields NaN.
    lo, hi = max(k - 1, 0), min(k + 1, n - 1)
    tang = pts[:, hi, :] - pts[:, lo, :]
    for _ in range(n):
        norm = np.hypot(tang[:, 0], tang[:, 1])
        if np.all(norm > 1e-12) or (lo == 0 and hi == n - 1):
            break
        lo, hi = max(lo - 1, 0), min(hi + 1, n - 1)
        tang = pts[:, hi, :] - pts[:, lo, :]

    norm = np.hypot(tang[:, 0], tang[:, 1])
    good = norm > 1e-12
    unit = np.zeros_like(tang)
    unit[good] = tang[good] / norm[good, None]
    return points, unit


def tangent_marker_angles(tangents: np.ndarray) -> np.ndarray:
    """Convert unit tangents to plotly ``marker.angle`` degrees.

    Plotly measures ``marker.angle`` **clockwise from "up"** (with
    ``angleref="up"`` and a ``triangle-up`` symbol), whereas :func:`numpy.arctan2`
    gives degrees counter-clockwise from ``+x``::

        phi   = degrees(arctan2(ty, tx))   # CCW from +x, in (-180, 180]
        angle = (90 - phi) % 360           # CW from "up", in [0, 360)

    These are **screen** angles, so they are only faithful on axes with a 1:1 data
    aspect ratio — which is why arrowheads are offered on the chord view (square,
    ``scaleanchor="x"``) and not on the timeline network view.

    :param tangents: ``(m, 2)`` unit vectors; zero vectors map to ``0.0``.
    :returns: ``(m,)`` angles in degrees.
    """
    t = np.atleast_2d(np.asarray(tangents, dtype=float))
    phi = np.degrees(np.arctan2(t[:, 1], t[:, 0]))
    angles = np.mod(90.0 - phi, 360.0)
    return np.where(np.hypot(t[:, 0], t[:, 1]) > 1e-12, angles, 0.0)


# ── timeline arcs ───────────────────────────────────────────────────────────────

def arc_heights(
    dt_years,
    *,
    backward=None,
    scale: float = 0.5,
    min_height: float = 0.05,
    mirror: bool = True,
) -> np.ndarray:
    """Per-pair timeline-arc height, **negative for backward pairs** when mirroring.

    Mirroring turns the network view into a two-sided plot — forward pairs arch
    above the timeline, backward pairs below — so the direction of every pair is
    readable from its geometry alone.

    :param dt_years: Temporal baselines in years.
    :param backward: Boolean mask, ``True`` where the pair runs j → i.  ``None``
        (or ``mirror=False``) keeps every height positive, the one-sided layout.
    :param scale: Height per year of baseline.
    :param min_height: Floor applied to the **magnitude**, so a same-day backward
        pair still dips below the axis.
    :returns: ``(m,)`` signed heights.
    """
    h = np.maximum(np.asarray(dt_years, dtype=float) * scale, min_height)
    if mirror and backward is not None:
        h = np.where(np.asarray(backward, dtype=bool), -h, h)
    return h


def arc_ylim(
    heights,
    *,
    extent: float = 0.85,
    floor: float = 0.05,
) -> tuple[float, float]:
    """Axis limits for a set of (possibly signed) arc heights.

    A cubic Bézier with both control points at ``h`` peaks at ``0.75 * h``, so the
    drawn extent sits well below the nominal height; the default *extent* of 0.85
    is that 0.75 plus a little headroom, and reproduces the historical one-sided
    range ``[-floor, h.max() * 0.85]`` exactly.

    Signed heights are why this is a shared helper: with every pair backward,
    ``heights.max()`` is *negative* and a naive ``[-floor, max]`` would invert the
    axis.

    :param extent: Fraction of the nominal height the axis reaches to.
    :param floor: Minimum half-range, keeping the ``y=0`` date nodes off the edge.
    :returns: ``(bottom, top)`` with ``bottom <= -floor <= floor <= top``.
    """
    h = np.asarray(heights, dtype=float)
    top = float(extent * h[h > 0].max()) if np.any(h > 0) else floor
    bottom = float(extent * h[h < 0].min()) if np.any(h < 0) else -floor
    return min(bottom, -floor), max(top, floor)


def timeline_arc_control_points(x1, x2, height) -> np.ndarray:
    """Cubic control nets for timeline arcs rising from ``y=0``.

    Vectorised form of the classic ``[(x1, 0), (x1, h), (x2, h), (x2, 0)]`` net:
    the arc leaves and re-enters the timeline vertically and peaks near *height*.

    :returns: ``(m, 4, 2)`` control nets.
    """
    x1 = np.atleast_1d(np.asarray(x1, dtype=float))
    x2 = np.atleast_1d(np.asarray(x2, dtype=float))
    h = np.broadcast_to(np.atleast_1d(np.asarray(height, dtype=float)), x1.shape)
    zeros = np.zeros_like(x1)
    return np.stack(
        [
            np.column_stack([x1, zeros]),
            np.column_stack([x1, h]),
            np.column_stack([x2, h]),
            np.column_stack([x2, zeros]),
        ],
        axis=1,
    )


def timeline_arc_polylines(x1, x2, height, *, n_points: int = 16) -> np.ndarray:
    """Sampled timeline arcs. :returns: ``(m, n_points, 2)`` polylines."""
    return bezier_sample(timeline_arc_control_points(x1, x2, height), n_points=n_points)


# ── colour ring and decorations ─────────────────────────────────────────────────

def color_ring_mesh(
    n_segments: int = 720,
    *,
    outer_radius: float = 1.0,
    ring_width: float = 0.045,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quad mesh for the circular time colour ring — **one** matplotlib artist.

    Drawing the ring as ``n_segments`` individual wedges costs ~1 s per redraw at
    ``n_segments=1000``; a single ``pcolormesh`` fed by this mesh costs ~20 ms.

    :returns: ``(X, Y, frac)`` where ``X``/``Y`` are ``(2, n_segments + 1)``
        (row 0 = inner rim, row 1 = outer rim) and ``frac`` is ``(n_segments,)``
        holding the time fraction at each segment centre.  Feed directly to
        ``ax.pcolormesh(X, Y, frac[None, :], shading="flat")``.
    """
    edges = np.linspace(0.0, 1.0, n_segments + 1)
    theta = np.pi / 2 - 2 * np.pi * edges
    radii = np.array([outer_radius - ring_width, outer_radius])[:, None]
    X = radii * np.cos(theta)[None, :]
    Y = radii * np.sin(theta)[None, :]
    frac = 0.5 * (edges[:-1] + edges[1:])
    return X, Y, frac


def ring_centerline(
    n_points: int = 720,
    *,
    outer_radius: float = 1.0,
    ring_width: float = 0.045,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points along the ring mid-radius — the plotly ring (one marker trace).

    :returns: ``(x, y, frac)``, each ``(n_points,)``.
    """
    frac = np.linspace(0.0, 1.0, n_points, endpoint=False)
    theta = np.pi / 2 - 2 * np.pi * frac
    x, y = polar_to_xy(outer_radius - ring_width / 2, theta)
    return x, y, frac


def year_tick_segments(
    scale: CircularTimeScale,
    *,
    outer_radius: float = 1.0,
    ring_width: float = 0.045,
    tick_length: float = 0.075,
) -> np.ndarray:
    """Radial ``|`` tick marks at Jan 1 of each year, centred on the ring.

    :returns: ``(n_years, 2, 2)`` segments — one ``[[x1, y1], [x2, y2]]`` per year.
    """
    theta = scale.year_angles()
    r_center = outer_radius - ring_width / 2
    r1, r2 = r_center - tick_length / 2, r_center + tick_length / 2
    x1, y1 = polar_to_xy(r1, theta)
    x2, y2 = polar_to_xy(r2, theta)
    return np.stack([np.column_stack([x1, y1]), np.column_stack([x2, y2])], axis=1)


def year_label_placement(
    scale: CircularTimeScale,
    *,
    label_radius: float = 1.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Positions, rotations and alignments for the year labels around the ring.

    Labels are rotated tangentially and flipped in the left half so they always
    read left-to-right.

    :returns: ``(x, y, rotation_deg, ha)`` — the last is a list of matplotlib
        horizontal-alignment strings (plotly maps them to ``xanchor``).
    """
    theta = scale.year_angles()
    x, y = polar_to_xy(label_radius, theta)

    rotation = np.degrees(theta)
    ha: list[str] = []
    for k, rot in enumerate(rotation):
        if rot < -90:
            rotation[k] = rot + 180
            ha.append("right")
        elif rot > 90:
            rotation[k] = rot - 180
            ha.append("right")
        else:
            ha.append("left")
    return x, y, rotation, ha
