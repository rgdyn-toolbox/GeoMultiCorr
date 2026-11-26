#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# validation.py
# creation date: 2026-10-30.
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
"""
Checking similarity predictions against what the correlator actually achieved.

Pure pandas/SciPy, no GMC imports. The caller gathers, per correlated pair, the
realised quality (median CC, fraction of CC ≥ 0.5, stable-ground NMAD) and the
two thumbs it was built from; this module matches them to similarity rows and
ranks each index against each outcome.

After correlation, the same indices can be measured again on the pair aligned
by ASP's raw disparity and by the corrected displacement maps; the session
stores them in each pair's stats JSON under ``similarity`` (``before`` /
``after.raw`` / ``after.corrected``). :func:`flatten_similarity_block` turns
that block into one row of ``<index>_raw`` / ``<index>_corr`` columns and
:func:`before_after_table` ranks each index against its own twin.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

#: Indices compared against the outcomes, in display order. ``dt_days`` is
#: the naive baseline every index has to beat.
VALIDATION_PREDICTORS: tuple[str, ...] = (
    "dt_days",
    "doy_diff",
    "pearson",
    "spearman",
    "linfoot",
    "hist_intersection",
    "ssim_luminance",
    "ssim_contrast",
    "ssim_structure",
    "zncc_local_median",
    "oc",
    "oc_axial",
    "coh_mean",
    "trial_valid_frac",
    "trial_peak_median",
    "trial_sigma_px",
    "score",
)

#: Realised outcomes. Good correlation means high CC and **low** NMAD.
VALIDATION_OUTCOMES: tuple[str, ...] = ("cc_median", "cc_q50", "nmad_px")

#: Below this many matched couples a rank correlation is not reported.
MIN_VALIDATION_COUPLES = 5

#: Column suffix of each after-correlation variant, keyed by its name in the
#: pair JSON: aligned by the raw disparity, or by the corrected maps.
ASP_VARIANTS: dict[str, str] = {"raw": "_raw", "corrected": "_corr"}

#: Outcomes measured on the ASP rasters over the similarity's own stable mask
#: (the stats JSON outcomes use the corrections' mask, another set of pixels).
RASTER_OUTCOMES: tuple[str, ...] = ("ncc_median", "ncc_q50", "nmad_corr_px")


def couple_key_from_pair_name(pa_name: str) -> tuple[str, str] | None:
    """The two thumb stems a pair was built from, sorted, or None.

    Parses the pair key exactly like ``Pair(path)``: the pzone appears once,
    then the two thumb keys joined by ``__``
    (``Chimborazo_2016-11-17_planetscope_3m__2017-08-14_planetscope_3m`` →
    ``("Chimborazo_2016-11-17_planetscope_3m", "Chimborazo_2017-08-14_planetscope_3m")``).
    Keys in the older ``-`` format carry no ``__`` and return None; the caller
    then falls back on dates and sensors.
    """
    pz, _, rest = str(pa_name).partition("_")
    parts = rest.split("__")
    if not pz or len(parts) != 2 or not all(parts):
        return None
    return tuple(sorted((f"{pz}_{parts[0]}", f"{pz}_{parts[1]}")))


def spearman_table(
    frame: pd.DataFrame,
    predictors: Sequence[str] = VALIDATION_PREDICTORS,
    outcomes: Sequence[str] = VALIDATION_OUTCOMES,
) -> pd.DataFrame:
    """Spearman ρ of every predictor against every outcome present in *frame*.

    Columns absent from *frame* are skipped; a pair of columns with fewer than
    :data:`MIN_VALIDATION_COUPLES` finite rows, or constant, gives NaN.
    """
    preds = [p for p in predictors if p in frame.columns]
    outs = [o for o in outcomes if o in frame.columns]
    table = pd.DataFrame(np.nan, index=preds, columns=outs, dtype=float)
    for p in preds:
        for o in outs:
            table.loc[p, o] = _rho(frame[p], frame[o])
    return table


def _rho(a, b) -> float:
    """Spearman ρ over the rows finite in both; NaN when too few or constant."""
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_VALIDATION_COUPLES:
        return np.nan
    if np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def with_asp_twins(names: Sequence[str]) -> list[str]:
    """*names* followed by their ``_raw`` and ``_corr`` twins."""
    return list(names) + [n + sfx for sfx in ASP_VARIANTS.values() for n in names]


def flatten_similarity_block(block) -> dict:
    """One row of columns from a pair JSON ``similarity`` block.

    ``after.raw.indices`` → ``<index>_raw``, ``after.corrected.indices`` →
    ``<index>_corr``, each variant's ``disp_valid_frac`` → ``disp_valid_frac_<sfx>``,
    and the raster outcomes ``ncc_median``, ``ncc_q50`` (raw) and
    ``nmad_corr_m``, ``nmad_corr_px`` (corrected). A missing variant
    (``corrected: null`` before corrections ran) gives no column; ``null``
    values become NaN.
    """
    after = (block or {}).get("after") or {}
    row: dict = {}
    for variant, sfx in ASP_VARIANTS.items():
        part = after.get(variant) or {}
        for key, value in (part.get("indices") or {}).items():
            row[key + sfx] = value
        if part:
            row["disp_valid_frac" + sfx] = part.get("disp_valid_frac")
    raw, corr = after.get("raw") or {}, after.get("corrected") or {}
    if raw:
        row["ncc_median"] = raw.get("ncc_median")
        row["ncc_q50"] = raw.get("ncc_q50")
    if corr:
        row["nmad_corr_m"] = corr.get("nmad_m")
        row["nmad_corr_px"] = corr.get("nmad_px")
    return {k: (np.nan if v is None else v) for k, v in row.items()}


def before_after_table(
    frame: pd.DataFrame, predictors: Sequence[str] = VALIDATION_PREDICTORS
) -> pd.DataFrame:
    """Spearman ρ of each index against its own after-correlation twins.

    Columns ``before_vs_raw`` and ``before_vs_corr``; indices without a twin in
    *frame* (``dt_days``, ``score``…) are left out. A high ρ means the index
    measured before correlation already ranked the couples as they rank once
    ASP has aligned them.
    """
    cols = {f"before_vs{sfx}": sfx for sfx in ASP_VARIANTS.values()}
    preds = [
        p
        for p in predictors
        if p in frame.columns and any(p + s in frame.columns for s in cols.values())
    ]
    table = pd.DataFrame(np.nan, index=preds, columns=list(cols), dtype=float)
    for p in preds:
        for col, sfx in cols.items():
            if p + sfx in frame.columns:
                table.loc[p, col] = _rho(frame[p], frame[p + sfx])
    return table


def match_outcomes(sim: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Join similarity rows with realised outcomes on the unordered thumb couple.

    :param sim: Similarity frame with ``left_key`` / ``right_key`` and ``res``.
    :param outcomes: Indexed by the sorted ``(stem, stem)`` couple, with any of
        ``cc_median``, ``cc_q50``, ``nmad_m``.
    :returns: Matched rows of *sim* with the outcome columns, plus ``nmad_px``
        (``nmad_m / res``) when ``nmad_m`` is present.
    """
    # Key on plain tuples: a list of tuples given to pd.Index becomes a
    # MultiIndex, which an ordinary join cannot match against a tuple column.
    lookup = {
        tuple(sorted(k)): row
        for k, row in zip(list(outcomes.index), outcomes.to_dict(orient="records"))
    }
    keys = [tuple(sorted(k)) for k in zip(sim["left_key"], sim["right_key"])]
    hit = np.array([k in lookup for k in keys], dtype=bool)
    out = sim[hit].copy()
    for col in outcomes.columns:
        out[col] = [lookup[k][col] for k, h in zip(keys, hit) if h]
    if "nmad_m" in out.columns and "res" in out.columns:
        out["nmad_px"] = out["nmad_m"] / out["res"]
    return out
