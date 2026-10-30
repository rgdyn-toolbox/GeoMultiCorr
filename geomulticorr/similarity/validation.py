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
        x = pd.to_numeric(frame[p], errors="coerce").to_numpy(dtype=float)
        for o in outs:
            y = pd.to_numeric(frame[o], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MIN_VALIDATION_COUPLES:
                continue
            if np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
                continue
            table.loc[p, o] = stats.spearmanr(x[ok], y[ok]).statistic
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
