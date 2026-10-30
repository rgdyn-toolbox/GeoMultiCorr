#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# selection.py
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
Choosing cross-sensor couples to correlate, from a similarity frame.

Pure pandas, no GMC imports: the input is any frame with the columns
:func:`~geomulticorr.similarity.raster.compare_rasters` produces plus
``left_key``, ``right_key``, ``left_sensor``, ``right_sensor`` and
``dt_days`` — which is what ``Session.compute_thumbs_similarity`` returns.

Why not simply sort by one score and take the top N:

- **A ratio of two metrics is dominated by one of them.** On Pas de l'Ours,
  ``trial_valid_frac / trial_sigma_px`` ranked like ``1/σ`` alone (Spearman
  −0.96) and ignored the NCC peak, the best-validated predictor. A mean of
  percentile ranks weighs each validated predictor equally, whatever its units.
- **The best couples cluster on one image.** A top-N list bridged one SPOT date
  seven times and left the others unconnected. Bridges exist to tie *every*
  minority-sensor date to the other network, so selection is per image.
- **Similarity is measured on stable ground** and cannot see the displacement
  a long baseline accumulates on the moving terrain. Δt is capped separately.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: Predictors combined by :func:`rank_score`: column → ``True`` if higher is better.
#: The two validated best (peak and coherence) plus the two that describe the
#: matching itself (how many tiles match, how noisily).
DEFAULT_SCORE_COLUMNS: Mapping[str, bool] = {
    "trial_peak_median": True,
    "coh_mean": True,
    "trial_valid_frac": True,
    "trial_sigma_px": False,
}


def sensor_family(sensor: str) -> str:
    """``spot6``/``spot7`` → ``spot``; ``landsat8`` → ``landsat``; others unchanged.

    Strips a trailing platform number (with an optional ``-``/``_`` and letter
    suffix), so two satellites of one constellation count as one sensor.
    """
    return re.sub(r"[-_]?\d+[a-z]?$", "", str(sensor).lower()) or str(sensor).lower()


def rank_score(
    frame: pd.DataFrame, columns: Mapping[str, bool] = DEFAULT_SCORE_COLUMNS
) -> pd.Series:
    """Mean percentile rank (0–1, 1 = best) over *columns*, within *frame*.

    Scale-free, so no metric dominates through its units; NaN ranks are
    skipped, and a row with no ranked column scores NaN. Ranks are relative
    to the rows passed in: score a candidate set, not the whole archive, when
    the candidates are what is being compared.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"similarity frame lacks column(s): {missing}")
    ranks = pd.concat(
        [frame[c].rank(pct=True, ascending=up) for c, up in columns.items()], axis=1
    )
    return ranks.mean(axis=1, skipna=True)


#: Metrics a selection may threshold on → ``True`` if higher is better.
#: ``score`` is the stored :func:`rank_score` (a percentile, 0–1).
METRIC_HIGHER_IS_BETTER: Mapping[str, bool] = {
    "score": True,
    "trial_peak_median": True,
    "trial_valid_frac": True,
    "trial_sigma_px": False,
    "coh_mean": True,
    "zncc_local_median": True,
    "ssim_structure": True,
    "oc": True,
    "pearson": True,
    "linfoot": True,
}


def _with_families(sim: pd.DataFrame, family: Callable[[str], str]) -> pd.DataFrame:
    need = {"left_key", "right_key", "left_sensor", "right_sensor", "dt_days"}
    missing = need - set(sim.columns)
    if missing:
        raise KeyError(f"similarity frame lacks column(s): {sorted(missing)}")
    c = sim.copy()
    if "score" not in c.columns:
        # Over the whole input, so a threshold means "better than x % of these
        # couples" whatever filters follow.
        c["score"] = rank_score(c)
    c["family_left"] = c["left_sensor"].map(family)
    c["family_right"] = c["right_sensor"].map(family)
    return c


def _anchor_family(c: pd.DataFrame, anchor: str) -> str:
    families = pd.concat(
        [
            c[["left_key", "family_left"]].set_axis(["key", "fam"], axis=1),
            c[["right_key", "family_right"]].set_axis(["key", "fam"], axis=1),
        ]
    ).drop_duplicates("key")
    if anchor == "minority":
        counts = families["fam"].value_counts()
        return counts.index[counts.to_numpy().argmin()]
    if anchor not in set(families["fam"]):
        raise ValueError(
            f"anchor {anchor!r} is not a sensor family here: "
            f"{sorted(set(families['fam']))}"
        )
    return anchor


def select_couples(
    sim: pd.DataFrame,
    *,
    metric: str = "score",
    threshold: float | None = None,
    cross_sensor_only: bool = False,
    per_image: int | None = None,
    anchor: str = "minority",
    max_dt_days: float | None = None,
    min_dt_days: float | None = None,
    family: Callable[[str], str] = sensor_family,
) -> pd.DataFrame:
    """Select couples from a similarity frame by threshold, Δt and per-image cap.

    The one selection rule behind ``Session.explore_similarity_pairs`` and
    ``Session.update_pairs(strategy="similarity")``.

    1. ``cross_sensor_only`` keeps couples whose sensor **families** differ
       (``spot6``↔``spot7`` is not cross-sensor).
    2. Δt gates, then ``metric`` ≥ ``threshold`` (≤ for a lower-is-better
       metric, see :data:`METRIC_HIGHER_IS_BETTER`). ``None`` turns a gate off.
    3. ``per_image`` caps how many couples each image keeps, best first by
       ``metric``. With ``cross_sensor_only`` it applies to the images of the
       *anchor* family (the minority sensor by default), so every image of the
       sensor at risk of being disconnected gets its own partners. Otherwise a
       couple is kept when it is among the best ``per_image`` of **either** of
       its two images.

    ``score`` is used as stored (a percentile over the whole run, so ``0.8``
    keeps the same meaning while filters change); it is computed over *sim*
    only when absent.

    :returns: The selected rows plus ``family_left``, ``family_right`` and,
        with ``cross_sensor_only``, ``anchor_key``, ``partner_key`` and
        ``rank_in_anchor``; sorted best first.
    """
    if metric not in METRIC_HIGHER_IS_BETTER:
        raise ValueError(
            f"unknown metric {metric!r}; choose from {sorted(METRIC_HIGHER_IS_BETTER)}"
        )
    if per_image is not None and per_image < 1:
        raise ValueError(f"per_image must be >= 1 or None, got {per_image}")
    up = METRIC_HIGHER_IS_BETTER[metric]
    c = _with_families(sim, family)
    if metric not in c.columns:
        raise KeyError(f"similarity frame lacks column {metric!r}")
    if cross_sensor_only:
        c = c[c["family_left"] != c["family_right"]]
    if max_dt_days is not None:
        c = c[c["dt_days"].to_numpy(dtype=float) <= max_dt_days]
    if min_dt_days is not None:
        c = c[c["dt_days"].to_numpy(dtype=float) >= min_dt_days]
    if threshold is not None:
        v = c[metric].to_numpy(dtype=float)
        c = c[v >= threshold] if up else c[v <= threshold]
    if c.empty:
        return _empty_selection(sim)
    c = c.sort_values(metric, ascending=not up, kind="mergesort")

    if cross_sensor_only:
        fam = _anchor_family(c, anchor)
        left_is_anchor = c["family_left"] == fam
        c["anchor_key"] = np.where(left_is_anchor, c["left_key"], c["right_key"])
        c["partner_key"] = np.where(left_is_anchor, c["right_key"], c["left_key"])
        c["rank_in_anchor"] = c.groupby("anchor_key").cumcount() + 1
        if per_image is not None:
            c = c[c["rank_in_anchor"] <= per_image]
        c = c.sort_values(["anchor_key", "rank_in_anchor"])
    elif per_image is not None:
        # rank among *all* couples of an image, whichever side it is on
        ends = pd.concat(
            [
                pd.DataFrame({"row": c.index, "img": c["left_key"]}),
                pd.DataFrame({"row": c.index, "img": c["right_key"]}),
            ]
        )
        order = {r: i for i, r in enumerate(c.index)}
        ends["pos"] = ends["row"].map(order)
        ends = ends.sort_values("pos", kind="mergesort")
        ends["rank"] = ends.groupby("img").cumcount() + 1
        best = ends.groupby("row")["rank"].min()
        c = c[c.index.map(best) <= per_image]
    return c.reset_index(drop=True)


def select_bridge_couples(
    sim: pd.DataFrame,
    *,
    per_image: int = 2,
    anchor: str = "minority",
    family: Callable[[str], str] = sensor_family,
    min_valid_frac: float | None = 0.9,
    min_peak: float | None = None,
    max_sigma_px: float | None = None,
    max_dt_days: float | None = None,
    min_dt_days: float | None = None,
) -> pd.DataFrame:
    """Cross-sensor bridges: :func:`select_couples` with ``cross_sensor_only``.

    For every image of the anchor sensor (the minority one by default), its
    ``per_image`` best partners in the other sensor, ranked by ``score``,
    after hard gates on the raw indices (``min_valid_frac``, ``min_peak``,
    ``max_sigma_px``; ``None`` turns one off). Anchor images left without a
    partner are absent — check :func:`bridge_coverage`.
    """
    if per_image < 1:
        raise ValueError(f"per_image must be >= 1, got {per_image}")
    c = _with_families(sim, family)
    for col, limit, op in (
        ("trial_valid_frac", min_valid_frac, np.greater_equal),
        ("trial_peak_median", min_peak, np.greater_equal),
        ("trial_sigma_px", max_sigma_px, np.less_equal),
    ):
        if limit is not None:
            c = c[op(c[col].to_numpy(dtype=float), limit)]
    return select_couples(
        c.drop(columns=["family_left", "family_right"]),
        metric="score",
        cross_sensor_only=True,
        per_image=per_image,
        anchor=anchor,
        max_dt_days=max_dt_days,
        min_dt_days=min_dt_days,
        family=family,
    )


def _empty_selection(sim: pd.DataFrame) -> pd.DataFrame:
    extra = ["family_left", "family_right", "anchor_key", "partner_key", "score"]
    extra += ["rank_in_anchor"]
    cols = list(dict.fromkeys(list(sim.columns) + extra))
    return pd.DataFrame(columns=cols)


def bridge_coverage(
    sim: pd.DataFrame,
    selected: pd.DataFrame,
    anchor_keys: Iterable[str] | None = None,
    family: Callable[[str], str] = sensor_family,
) -> pd.DataFrame:
    """How many bridges each anchor image received — zeros included.

    :param sim: The full similarity frame (to know every anchor image).
    :param selected: Output of :func:`select_couples` (``cross_sensor_only``) or
        :func:`select_bridge_couples`.
    :param anchor_keys: The anchor images to report; by default every image of
        the anchor family that appears in *sim*.
    :param family: The same sensor → family mapping given to the selection.
    :returns: One row per anchor image: ``n_bridges`` and the best ``score``.
        An image with ``n_bridges == 0`` stays disconnected from the other
        network — relax a gate for it rather than ignore it.
    """
    if anchor_keys is None:
        if selected.empty:
            return pd.DataFrame(columns=["anchor_key", "n_bridges", "best_score"])
        fam = (
            selected["family_left"]
            .where(
                selected["anchor_key"] == selected["left_key"], selected["family_right"]
            )
            .iloc[0]
        )
        keys = pd.concat(
            [
                sim.loc[sim["left_sensor"].map(family) == fam, "left_key"],
                sim.loc[sim["right_sensor"].map(family) == fam, "right_key"],
            ]
        )
        anchor_keys = sorted(set(keys))
    g = selected.groupby("anchor_key")["score"]
    out = pd.DataFrame({"anchor_key": list(anchor_keys)})
    out["n_bridges"] = out["anchor_key"].map(g.size()).fillna(0).astype(int)
    out["best_score"] = out["anchor_key"].map(g.max())
    return out


def thumb_order(sim: pd.DataFrame) -> list[str]:
    """Every thumb of *sim*, sorted by acquisition date then name."""
    ends = pd.concat(
        [
            sim[["left_key", "left_date"]].set_axis(["key", "date"], axis=1),
            sim[["right_key", "right_date"]].set_axis(["key", "date"], axis=1),
        ]
    ).drop_duplicates("key")
    return ends.sort_values(["date", "key"])["key"].tolist()


def thumb_sensors(sim: pd.DataFrame) -> dict[str, str]:
    """``{thumb key: sensor}`` from both sides of *sim*."""
    pairs = list(zip(sim["left_key"], sim["left_sensor"]))
    pairs += list(zip(sim["right_key"], sim["right_sensor"]))
    return dict(pairs)


def similarity_matrix(
    sim: pd.DataFrame, metric: str, order: Sequence[str] | None = None
) -> pd.DataFrame:
    """Symmetric thumb × thumb matrix of *metric* (NaN on the diagonal and for
    couples not measured). Rows and columns follow *order*, by default
    :func:`thumb_order`. Shared by the static figure and the explorer heatmap
    so both lay thumbs out identically.
    """
    keys = list(order) if order is not None else thumb_order(sim)
    pos = {k: i for i, k in enumerate(keys)}
    m = np.full((len(keys), len(keys)), np.nan)
    for a, b, v in zip(sim["left_key"], sim["right_key"], sim[metric]):
        if a in pos and b in pos:
            m[pos[a], pos[b]] = m[pos[b], pos[a]] = v
    return pd.DataFrame(m, index=keys, columns=keys)
