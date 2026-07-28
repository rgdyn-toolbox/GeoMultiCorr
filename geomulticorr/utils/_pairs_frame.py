#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _pairs_frame.py
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
"""The canonical *pairs frame* — one table contract for every pair plot.

Pairs reach the plotting code from two very different places:

- **candidate** pairs, computed on the fly by
  :meth:`~geomulticorr.core.session.Session.explore_pairs_strategy` from
  :func:`~geomulticorr.core.pzone._strategy_pair_indices`, never written to disk;
- **committed** pairs, read back from the geodatabase via
  :meth:`~geomulticorr.core.session.Session.get_pairs_overview`.

Both are normalised here into a single ``pandas.DataFrame`` with the columns in
:data:`PAIRS_FRAME_COLUMNS`, so every renderer — matplotlib or plotly, static or
interactive — consumes exactly one shape of input.

This module is pure pandas/numpy and deliberately does **not** import
:mod:`geomulticorr.core`: the caller runs the pairing strategy and hands the
index pairs in, which keeps the dependency edge pointing one way (``core`` →
``utils``).

Public API:

- :data:`PAIRS_FRAME_COLUMNS` — the contract.
- :func:`pairs_frame_from_indices` — candidate pairs.
- :func:`pairs_frame_from_overview` — committed pairs.
- :func:`empty_pairs_frame` — correctly typed zero-row frame.
- :func:`unique_dates`, :func:`unique_couples`, :func:`date_summary` — views.
- :func:`pairs_stats` / :func:`format_pairs_summary` — unambiguous counts.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from geomulticorr.utils._pairs_geometry import DAYS_PER_YEAR, to_decimal_year

#: Columns every pairs frame carries, in order.
PAIRS_FRAME_COLUMNS: tuple[str, ...] = (
    "pz",          # pzone name ("" when unknown)
    "date_i",      # ISO date of the left/first acquisition
    "date_j",      # ISO date of the right/second acquisition
    "t_i",         # datetime64 of date_i
    "t_j",         # datetime64 of date_j
    "dec_i",       # decimal year of date_i
    "dec_j",       # decimal year of date_j
    "dt_days",     # |t_j - t_i| in whole days
    "dt_years",    # dt_days / 365.25, rounded to 4 (matches Pair.pa_dt_years)
    "mid_dec",     # decimal year of the pair midpoint
    "direction",   # "forward" when t_i <= t_j, else "backward"
    "sensor_i",    # sensor of date_i ("" when unknown)
    "sensor_j",    # sensor of date_j ("" when unknown)
    "label",       # "<pz>_<date_i>_<date_j>" — hover/debug key
)

_DTYPES: dict[str, str] = {
    "pz": "object", "date_i": "object", "date_j": "object",
    "t_i": "datetime64[ns]", "t_j": "datetime64[ns]",
    "dec_i": "float64", "dec_j": "float64",
    "dt_days": "int64", "dt_years": "float64", "mid_dec": "float64",
    "direction": "object", "sensor_i": "object", "sensor_j": "object",
    "label": "object",
}


def empty_pairs_frame() -> pd.DataFrame:
    """A zero-row pairs frame with the correct columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _DTYPES.items()})


def _assemble(
    pz: Sequence[str],
    t_i: pd.DatetimeIndex,
    t_j: pd.DatetimeIndex,
    sensor_i: Sequence[str],
    sensor_j: Sequence[str],
) -> pd.DataFrame:
    """Derive every computed column from the raw endpoints. Single source of truth."""
    if len(t_i) == 0:
        return empty_pairs_frame()

    date_i = t_i.strftime("%Y-%m-%d").to_numpy()
    date_j = t_j.strftime("%Y-%m-%d").to_numpy()

    dt_days = np.abs((t_j - t_i).days.to_numpy()).astype("int64")
    mid = t_i + (t_j - t_i) / 2
    pz_arr = np.asarray(pz, dtype=object)

    frame = pd.DataFrame(
        {
            "pz": pz_arr,
            "date_i": date_i,
            "date_j": date_j,
            "t_i": t_i,
            "t_j": t_j,
            "dec_i": to_decimal_year(t_i),
            "dec_j": to_decimal_year(t_j),
            "dt_days": dt_days,
            "dt_years": np.round(dt_days / DAYS_PER_YEAR, 4),
            "mid_dec": to_decimal_year(pd.DatetimeIndex(mid)),
            "direction": np.where(t_i.to_numpy() <= t_j.to_numpy(), "forward", "backward"),
            "sensor_i": np.asarray(sensor_i, dtype=object),
            "sensor_j": np.asarray(sensor_j, dtype=object),
            "label": [f"{p}_{a}_{b}" for p, a, b in zip(pz_arr, date_i, date_j)],
        }
    )
    return frame.astype(_DTYPES)[list(PAIRS_FRAME_COLUMNS)]


def pairs_frame_from_indices(
    dates: Sequence,
    index_pairs: Sequence[tuple[int, int]],
    *,
    sensors: Sequence[str] | None = None,
    pz: str = "",
) -> pd.DataFrame:
    """Build a pairs frame from *candidate* index pairs.

    :param dates: Acquisition dates, **sorted ascending** — the same list the
        indices were generated against.
    :param index_pairs: Output of
        :func:`~geomulticorr.core.pzone._strategy_pair_indices`, i.e. ``(i, j)``
        positions into *dates*.  ``i > j`` marks a backward pair.
    :param sensors: Optional sensor name per date; defaults to ``""``.
    :param pz: Pzone name stamped on every row.
    :returns: A frame with :data:`PAIRS_FRAME_COLUMNS`.
    """
    if len(index_pairs) == 0:
        return empty_pairs_frame()

    t = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if sensors is None:
        sensors = [""] * len(t)
    elif len(sensors) != len(t):
        raise ValueError(f"sensors has {len(sensors)} entries but dates has {len(t)}.")

    i_idx = np.array([i for i, _ in index_pairs], dtype=int)
    j_idx = np.array([j for _, j in index_pairs], dtype=int)
    if i_idx.max(initial=0) >= len(t) or j_idx.max(initial=0) >= len(t):
        raise ValueError("index_pairs references a position beyond the end of dates.")

    sensors_arr = np.asarray(sensors, dtype=object)
    return _assemble(
        pz=[pz] * len(i_idx),
        t_i=t[i_idx],
        t_j=t[j_idx],
        sensor_i=sensors_arr[i_idx],
        sensor_j=sensors_arr[j_idx],
    )


def pairs_frame_from_overview(
    pairs_gdf: pd.DataFrame,
    *,
    date_sensor: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a pairs frame from *committed* pairs read out of the geodatabase.

    :param pairs_gdf: The output of
        :meth:`~geomulticorr.core.session.Session.get_pairs_overview` — needs at
        least ``pa_left_date`` and ``pa_right_date``.
    :param date_sensor: Fallback ISO-date → sensor mapping, used only when the
        layer predates the ``pa_left_sensor`` / ``pa_right_sensor`` columns.
        Without it, sensors on such layers degrade to ``""``.
    :returns: A frame with :data:`PAIRS_FRAME_COLUMNS`.
    """
    if pairs_gdf is None or len(pairs_gdf) == 0:
        return empty_pairs_frame()

    missing = {"pa_left_date", "pa_right_date"} - set(pairs_gdf.columns)
    if missing:
        raise ValueError(f"pairs overview is missing required column(s): {sorted(missing)}.")

    left = pairs_gdf["pa_left_date"].astype(str)
    right = pairs_gdf["pa_right_date"].astype(str)

    def _sensors(col: str, dates: pd.Series) -> np.ndarray:
        if col in pairs_gdf.columns:
            return pairs_gdf[col].fillna("").astype(str).to_numpy(dtype=object)
        if date_sensor:  # older geodatabase: fall back to the thumbs lookup
            return dates.map(lambda d: date_sensor.get(d, "")).to_numpy(dtype=object)
        return np.full(len(pairs_gdf), "", dtype=object)

    pz = (
        pairs_gdf["pa_pz_name"].fillna("").astype(str).to_numpy(dtype=object)
        if "pa_pz_name" in pairs_gdf.columns
        else np.full(len(pairs_gdf), "", dtype=object)
    )

    return _assemble(
        pz=pz,
        t_i=pd.DatetimeIndex(pd.to_datetime(left)),
        t_j=pd.DatetimeIndex(pd.to_datetime(right)),
        sensor_i=_sensors("pa_left_sensor", left),
        sensor_j=_sensors("pa_right_sensor", right),
    )


# ── derived views ───────────────────────────────────────────────────────────────

def unique_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted unique acquisition dates appearing on either side of any pair."""
    if len(frame) == 0:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(
        pd.unique(pd.concat([frame["t_i"], frame["t_j"]], ignore_index=True))
    ).sort_values()


def unique_couples(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate *unordered* couples, keeping the first occurrence.

    The ``redundancy`` and ``forward-backward`` strategies emit both ``(i, j)``
    and ``(j, i)``.  Drawing both puts two identical curves on top of each other,
    which doubles the effective alpha and misleads the eye about density.
    """
    if len(frame) == 0:
        return frame.copy()
    key = [tuple(sorted((a, b))) for a, b in zip(frame["date_i"], frame["date_j"])]
    return frame.loc[~pd.Index(key).duplicated(keep="first")].reset_index(drop=True)


#: How each strategy emits couples.  Surfaced in the explorer summary so the
#: forward-only vs bidirectional distinction never has to be guessed at — the
#: usual confusion is that ``step`` returns exactly half of ``redundancy``.
STRATEGY_SEMANTICS: dict[str, str] = {
    "consecutive": "forward only: each acquisition paired with the next",
    "step": "forward only: every (i, j) with 0 < j - i ≤ max_step",
    "redundancy": "bidirectional: every 'step' couple is also emitted as (j, i)",
    "forward-backward": "bidirectional: consecutive couples in both directions",
}


def pairs_stats(frame: pd.DataFrame, *, strategy: str | None = None) -> dict:
    """Counts that describe a pairs frame unambiguously.

    Pure computation, no formatting — :func:`format_pairs_summary` renders it.

    The important subtlety is ``sensor_dates``: it counts **acquisitions** per
    sensor and therefore sums to ``n_dates``.  Counting pair *endpoints* instead
    (two per pair) yields twice the pair count, which reads like a pair total and
    is exactly the misreading this function exists to prevent.

    :param frame: A pairs frame.
    :param strategy: Optional strategy name, echoed back with its semantics.
    :returns: A dict of counts; see the keys below.
    """
    dates = date_summary(frame)
    n_pairs = int(len(frame))
    n_couples = int(len(unique_couples(frame)))
    directions = frame["direction"] if n_pairs else pd.Series(dtype="object")
    dts = frame["dt_days"] if n_pairs else pd.Series(dtype="int64")

    return {
        "n_pairs": n_pairs,
        "n_couples": n_couples,
        "n_reversed": n_pairs - n_couples,
        "n_forward": int((directions == "forward").sum()),
        "n_backward": int((directions == "backward").sum()),
        "n_dates": int(len(dates)),
        "n_pzones": int(frame["pz"].nunique()) if n_pairs else 0,
        "dt_min": int(dts.min()) if n_pairs else 0,
        "dt_median": float(dts.median()) if n_pairs else 0.0,
        "dt_mean": float(dts.mean()) if n_pairs else 0.0,
        "dt_max": int(dts.max()) if n_pairs else 0,
        # acquisitions per sensor — sums to n_dates, NOT to 2 * n_pairs
        "sensor_dates": dates["sensor"].value_counts().to_dict() if len(dates) else {},
        "strategy": strategy,
        "bidirectional": (n_pairs - n_couples) > 0 if n_pairs else None,
    }


def format_pairs_summary(
    stats: dict,
    *,
    drawn: int | None = None,
    drawn_label: str = "curves drawn",
    html: bool = True,
) -> str:
    """Render :func:`pairs_stats` as a short, unambiguous multi-line summary.

    :param stats: Output of :func:`pairs_stats`.
    :param drawn: How many curves/points the current figure actually shows.  A
        line is emitted **only when this differs from the pair count** — e.g. the
        chord view deduplicates, so 250 bidirectional pairs render as 125 chords.
    :param drawn_label: Noun for the drawn count ("chords drawn", "arcs drawn"…).
    :param html: Render with HTML tags (for ``ipywidgets.HTML``) or as plain text.
    :returns: The formatted summary.
    """
    bold = (lambda s: f"<b>{s}</b>") if html else (lambda s: f"**{s}**")
    faint = (lambda s: f"<i>{s}</i>") if html else (lambda s: s)
    sep = " &nbsp;|&nbsp; " if html else " | "
    br = "<br>" if html else "\n"

    n_pairs = stats["n_pairs"]
    if n_pairs == 0:
        return f"{bold('0 pairs')} — no pair matches the current filters."

    lines: list[str] = []

    n_couples, n_reversed = stats["n_couples"], stats["n_reversed"]
    if n_reversed:
        lines.append(
            f"{bold(f'{n_pairs} pairs')}{sep}{n_couples} unique couples × 2 directions "
            f"({stats['n_forward']} forward, {stats['n_backward']} backward)"
        )
    else:
        lines.append(
            f"{bold(f'{n_pairs} pairs')}{sep}{n_couples} unique couples "
            "(one direction each — forward only)"
        )

    if drawn is not None and drawn != n_pairs:
        lines.append(f"{drawn} {drawn_label} (duplicate couples merged)")

    sensors = ", ".join(f"{s}: {c}" for s, c in sorted(stats["sensor_dates"].items()))
    dates_line = f"{stats['n_dates']} acquisition dates"
    if sensors:
        dates_line += f" — acquisitions per sensor: {sensors}"
    if stats["n_pzones"] > 1:
        dates_line += f"{sep}{stats['n_pzones']} pzones"
    lines.append(dates_line)

    lines.append(
        "Δt min/median/mean/max = "
        f"{stats['dt_min']} / {stats['dt_median']:.0f} / "
        f"{stats['dt_mean']:.0f} / {stats['dt_max']} d"
    )

    semantics = STRATEGY_SEMANTICS.get(stats.get("strategy") or "")
    if semantics:
        lines.append(faint(f"{stats['strategy']} — {semantics}"))

    return br.join(lines)


def date_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-acquisition summary used for node hover: ``date``, ``t``, ``sensor``, ``n_pairs``.

    ``n_pairs`` counts every pair touching that date, so it sums to ``2 * len(frame)``.
    """
    if len(frame) == 0:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="object"),
                "t": pd.Series(dtype="datetime64[ns]"),
                "sensor": pd.Series(dtype="object"),
                "n_pairs": pd.Series(dtype="int64"),
            }
        )

    endpoints = pd.DataFrame(
        {
            "date": pd.concat([frame["date_i"], frame["date_j"]], ignore_index=True),
            "t": pd.concat([frame["t_i"], frame["t_j"]], ignore_index=True),
            "sensor": pd.concat([frame["sensor_i"], frame["sensor_j"]], ignore_index=True),
        }
    )
    grouped = (
        endpoints.groupby("date", as_index=False)
        .agg(t=("t", "first"), sensor=("sensor", "first"), n_pairs=("date", "size"))
        .sort_values("t")
        .reset_index(drop=True)
    )
    return grouped.astype({"n_pairs": "int64"})
