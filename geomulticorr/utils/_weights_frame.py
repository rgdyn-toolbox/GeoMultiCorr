#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _weights_frame.py
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
"""The canonical *weights frame* — one shape of input for both plotting backends.

This mirrors :mod:`geomulticorr.utils._pairs_frame`: the interactive plotly view
and the static matplotlib twin both consume a frame built here, so the figure on
screen and the figure written to disk cannot drift apart.

One row per pair, carrying both directions' weights side by side, because the
figure draws EW and NS as two series over the same Δt axis.

This module is **pure pandas/numpy** — no plotting, no ``geomulticorr.core``, no
``geomulticorr.inversion``.  ``TIOInversion`` computes the weights and passes the
vectors in.

Public API:

- :data:`WEIGHTS_FRAME_COLUMNS` — the contract.
- :func:`weights_frame` — build one from per-pair vectors.
- :data:`WEIGHT_DIRECTION_COLORS` / :data:`WEIGHT_DIRECTION_MARKERS` — shared
  styling, so plotly and matplotlib draw the same thing.
- :data:`WEIGHT_MODE_KEYS` / :func:`relevant_weight_keys` — which parameters
  actually affect a given weighting mode.
- :func:`parse_sensor_weights` / :func:`format_sensor_weights` — the
  ``"spot=0.5, planetscope=1"`` text form of the ``sensor_weights`` mapping,
  used by the explorer's Text control.
- :func:`sensor_colors` — one deterministic colour per sensor group, for both backends.
- :func:`weights_stats` / :func:`format_weights_summary` — user-facing counts.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: Columns every weights frame carries, in order.
WEIGHTS_FRAME_COLUMNS: tuple[str, ...] = (
    "pa_key",          # pair key, the hover/debug identity
    "dt_days",         # temporal baseline in whole days — the figure's x axis
    "w_ew",            # EW weight in [0, 1]
    "w_ns",            # NS weight in [0, 1]
    "nmad_ew",         # EW NMAD (m) from the pair stats JSON, nan when missing
    "nmad_ns",         # NS NMAD (m), nan when missing
    "cc",              # cc_quality_gte_050, nan when missing
    "corr_direction",  # the pair's correlation direction ("" when unknown)
    "sensor",          # lower-case sensor of the pair ("" when unknown, "a+b" if mixed)
)

_DTYPES: dict[str, str] = {
    "pa_key": "object",
    "dt_days": "float64",
    "w_ew": "float64",
    "w_ns": "float64",
    "nmad_ew": "float64",
    "nmad_ns": "float64",
    "cc": "float64",
    "corr_direction": "object",
    "sensor": "object",
}

#: Series colour per direction.  Read by **both** backends so a saved PNG cannot
#: use different colours from the plotly figure it is a twin of.
WEIGHT_DIRECTION_COLORS: dict[str, str] = {"EW": "#1f77b4", "NS": "#ff7f0e"}

#: Marker per direction — plotly symbol name on the left, matplotlib on the right.
WEIGHT_DIRECTION_MARKERS: dict[str, tuple[str, str]] = {
    "EW": ("circle", "o"),
    "NS": ("diamond", "D"),
}

#: Palette for sensor groups (a colour-blind-safe qualitative set).  Assigned
#: by :func:`sensor_colors` in **sorted** sensor order, so the same archive gets
#: the same colours whatever order its pairs come in.
SENSOR_PALETTE: tuple[str, ...] = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#ff7f0e",
)


def sensor_colors(sensors: Sequence[str]) -> dict[str, str]:
    """``{sensor: hex}`` — deterministic under reordering, cycling the palette.

    Read by both plotting backends so a saved PNG and the plotly figure agree.
    The empty label (unknown sensor) is included when present.
    """
    unique = sorted({("" if s is None else str(s)) for s in sensors})
    return {s: SENSOR_PALETTE[i % len(SENSOR_PALETTE)] for i, s in enumerate(unique)}


def parse_sensor_weights(text: str | None) -> dict[str, float] | None:
    """``"spot=0.5, planetscope=1"`` → ``{"planetscope": 1.0, "spot": 0.5}``.

    Accepts ``=`` or ``:`` between name and factor, commas or whitespace between
    entries, and any case (keys are lower-cased). Empty or whitespace-only text
    means "no sensor weighting" and returns ``None``. A malformed token raises
    ``ValueError`` naming it — the explorer catches that and keeps the last good
    value rather than letting the callback die silently.
    """
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    out: dict[str, float] = {}
    for token in text.replace(",", " ").split():
        sep = "=" if "=" in token else (":" if ":" in token else None)
        if sep is None:
            raise ValueError(f"expected name=factor, got {token!r}")
        key, _, value = token.partition(sep)
        key = key.strip().lower()
        if not key:
            raise ValueError(f"missing sensor name in {token!r}")
        try:
            factor = float(value)
        except ValueError:
            raise ValueError(f"factor for {key!r} must be a number, got {value!r}") from None
        if not np.isfinite(factor) or factor < 0.0:
            raise ValueError(f"factor for {key!r} must be finite and ≥ 0, got {value!r}")
        out[key] = factor
    return dict(sorted(out.items())) or None


def format_sensor_weights(sensor_weights: dict[str, float] | None) -> str:
    """The inverse of :func:`parse_sensor_weights`: sorted ``name=factor`` list, ``""`` for none."""
    if not sensor_weights:
        return ""
    return ", ".join(f"{k}={float(v):g}" for k, v in sorted(sensor_weights.items()))

#: Weighting parameters that actually affect each mode.
#:
#: Drives three things that must agree: the explorer's show/hide logic, the
#: pruning in :func:`~geomulticorr.utils._weights_export.weights_figure_stem`, and
#: the ``relevant_params`` field of the run-parameters JSON.
#: :attr:`~geomulticorr.inversion.tio_inversion.TIOInversion._MODE_CONTROLS` is an
#: alias of this table rather than a second copy.
#:
#: ``dt_range`` appears here for ``relative_temporal``/``sigmoid`` even though no
#: widget exposes it: without it a ``dt_range=(300, 400)`` run and a
#: ``dt_range=None`` run collide on the same file stem.  The explorer looks names
#: up by membership, so an entry with no widget is simply ignored there.
WEIGHT_MODE_KEYS: dict[str, set[str]] = {
    "uniform": set(),
    "temporal": {"w_min"},
    "relative_temporal": {"w_min", "invert", "dt_range"},
    "sigmoid": {"sharpness", "w_min", "invert", "dt_range"},
    "parametric": {"slope", "min_weight"},
    "quality": {"combine", "cc_gamma", "invert"},
    "quality_spatial": {"combine", "cc_gamma"},
}


def relevant_weight_keys(
    weight_mode: str, combine: str | None = None,
    sensor_weights: dict[str, float] | None = None,
) -> set[str]:
    """Parameters that change the weights for *weight_mode*.

    Folds in the α/β/γ rule the explorer applies: those three only matter when a
    quality mode is combined with ``'wmean'``, so a stem or a parameter listing
    must not mention them otherwise. ``sensor_weights`` is mode-independent —
    it multiplies every mode's result, ``uniform`` included — so it is folded in
    here rather than listed under each mode in :data:`WEIGHT_MODE_KEYS`, and
    only when actually set.

    :param weight_mode: One of the keys of :data:`WEIGHT_MODE_KEYS`.
    :param combine: The combination method, when the mode is a quality mode.
    :param sensor_weights: The sensor factors in force, or ``None``/``{}``.
    :returns: A new set (never the stored one — callers mutate it).
    """
    relevant = set(WEIGHT_MODE_KEYS.get(weight_mode, set()))
    if combine == "wmean":
        if weight_mode == "quality":
            relevant |= {"alpha", "beta", "gamma"}
        elif weight_mode == "quality_spatial":
            relevant |= {"alpha", "beta"}
    if sensor_weights:
        relevant.add("sensor_weights")
    return relevant


def empty_weights_frame() -> pd.DataFrame:
    """A zero-row weights frame with the correct columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _DTYPES.items()})


def weights_frame(
    pa_keys: Sequence[str],
    dt_days: Sequence[float],
    w_ew: Sequence[float],
    w_ns: Sequence[float],
    *,
    nmad_ew: Sequence[float] | None = None,
    nmad_ns: Sequence[float] | None = None,
    cc: Sequence[float] | None = None,
    corr_direction: Sequence[str] | None = None,
    sensor: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a weights frame from per-pair vectors, all aligned to *pa_keys*.

    :param pa_keys: One pair key per row.
    :param dt_days: Temporal baseline per pair, in days.
    :param w_ew: EW weight per pair.
    :param w_ns: NS weight per pair.
    :param nmad_ew: EW NMAD per pair; ``nan`` where absent (the default).
    :param nmad_ns: NS NMAD per pair.
    :param cc: CC quality per pair.
    :param corr_direction: The pair's correlation direction; ``""`` by default.
    :param sensor: The pair's sensor label (lower-case); ``""`` by default.
    :returns: A frame with :data:`WEIGHTS_FRAME_COLUMNS`.
    :raises ValueError: If any vector's length differs from *pa_keys*.  A silent
        broadcast here would mislabel every point in the figure.
    """
    n = len(pa_keys)
    if n == 0:
        return empty_weights_frame()

    def _floats(values, label: str) -> np.ndarray:
        if values is None:
            return np.full(n, np.nan, dtype="float64")
        if len(values) != n:
            raise ValueError(
                f"{label} has {len(values)} entries but there are {n} pairs."
            )
        return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
            dtype="float64"
        )

    def _labels(values, label: str) -> list[str]:
        if values is None:
            return [""] * n
        if len(values) != n:
            raise ValueError(
                f"{label} has {len(values)} entries but there are {n} pairs."
            )
        return [("" if v is None else str(v)) for v in values]

    directions = _labels(corr_direction, "corr_direction")
    sensors = _labels(sensor, "sensor")

    frame = pd.DataFrame(
        {
            "pa_key": np.asarray([str(k) for k in pa_keys], dtype=object),
            "dt_days": _floats(dt_days, "dt_days"),
            "w_ew": _floats(w_ew, "w_ew"),
            "w_ns": _floats(w_ns, "w_ns"),
            "nmad_ew": _floats(nmad_ew, "nmad_ew"),
            "nmad_ns": _floats(nmad_ns, "nmad_ns"),
            "cc": _floats(cc, "cc"),
            "corr_direction": np.asarray(directions, dtype=object),
            "sensor": np.asarray(sensors, dtype=object),
        }
    )
    return frame.astype(_DTYPES)[list(WEIGHTS_FRAME_COLUMNS)]


def weights_stats(frame: pd.DataFrame, *, weight_mode: str = "",
                  combine: str = "") -> dict:
    """Counts and per-direction weight statistics for a user-facing summary.

    :returns: ``{"n_pairs", "weight_mode", "combine", "EW": {...}, "NS": {...}}``
        where each direction block holds ``min``/``max``/``mean``/``n_zero``.
        Numbers are plain Python floats, so the dict is JSON-safe.
    """
    out: dict = {
        "n_pairs": int(len(frame)),
        "weight_mode": weight_mode,
        "combine": combine,
    }
    for direction, column in (("EW", "w_ew"), ("NS", "w_ns")):
        out[direction] = weight_summary(
            frame[column].to_numpy() if len(frame) else []
        )
    return out


def weight_summary(values) -> dict:
    """``{"n", "min", "max", "mean", "n_zero"}`` for one weight vector.

    NaNs are ignored rather than propagated, so one pair with missing stats
    cannot blank the whole summary.  All values are plain floats/ints for JSON.
    """
    arr = np.asarray(list(values), dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n": int(arr.size), "min": None, "max": None,
                "mean": None, "n_zero": 0}
    return {
        "n": int(arr.size),
        "min": float(round(float(finite.min()), 6)),
        "max": float(round(float(finite.max()), 6)),
        "mean": float(round(float(finite.mean()), 6)),
        "n_zero": int((finite == 0.0).sum()),
    }


def format_weights_summary(stats: dict) -> str:
    """One-line HTML summary of *stats*, for the explorer's status area."""
    label = stats.get("weight_mode") or "?"
    if stats.get("combine") and label in ("quality", "quality_spatial"):
        label = f"{label}:{stats['combine']}"

    parts = [f"<b>{stats.get('n_pairs', 0)}</b> pairs &middot; mode <code>{label}</code>"]
    for direction in ("EW", "NS"):
        block = stats.get(direction) or {}
        if block.get("mean") is None:
            parts.append(f"{direction}: n/a")
            continue
        piece = (f"{direction}: mean <b>{block['mean']:.3f}</b> "
                 f"[{block['min']:.3f}–{block['max']:.3f}]")
        if block.get("n_zero"):
            piece += f" &middot; {block['n_zero']} at zero"
        parts.append(piece)
    return " &nbsp;|&nbsp; ".join(parts)
