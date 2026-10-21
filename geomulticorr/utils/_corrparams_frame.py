#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _corrparams_frame.py
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
"""The canonical *correlation-parameters frame* contract.

One row per pair, carrying both the inputs the derivation used (resolution,
baseline, assumed velocity) and everything it produced (kernel, search box, SNR,
cost).  Every renderer — plotly or matplotlib, interactive or static — takes
exactly this shape, so a design map drawn in the explorer and one saved to PNG
cannot disagree.

Same layering as :mod:`geomulticorr.utils._pairs_frame`: pure pandas/NumPy, and
**no imports from** :mod:`geomulticorr.core`.  The physics comes from
:mod:`geomulticorr.correlation.corr_params`, which is equally dependency-light;
the caller supplies the per-pair facts.

Two constructors, one contract:

- :func:`corrparams_frame_from_pairs` — **derived** parameters, computed on the
  fly by the explorer from resolution/Δt/velocity.
- :func:`corrparams_frame_from_plan` — the same rows **replayed** from a
  correlation plan JSON.

A test asserts the two agree for the same logical inputs, exactly as
``test_pairs_frame.py`` does for candidate versus committed pairs.

Public API:

- :data:`CORRPARAMS_FRAME_COLUMNS` — the column contract.
- :func:`empty_corrparams_frame` — correctly typed zero-row frame.
- :func:`corrparams_frame_from_pairs` / :func:`corrparams_frame_from_plan`.
- :func:`group_table` — one row per parameter group, for the explorer's table.
- :data:`CORR_MODE_KEYS` / :func:`relevant_corr_keys` — which controls matter.
- :func:`corrparams_stats` / :func:`format_corrparams_summary` — user-facing
  counts.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from geomulticorr.correlation.corr_params import (
    DAYS_PER_YEAR,
    assign_dt_bin,
    cost_index,
    format_dt_span,
    geometric_dt_bins,
    search_half_width_px,
    suggest_parameters,
    uniform_conflicts,
)

#: Columns every correlation-parameters frame carries, in order.
CORRPARAMS_FRAME_COLUMNS: tuple[str, ...] = (
    "pa_key",         # pair key, the join back to Pair / the geodatabase
    "pz",             # pzone name ("" when unknown)
    "sensor_i",       # left sensor  ("" when unknown)
    "sensor_j",       # right sensor ("" when unknown)
    "dt_days",        # temporal baseline in whole days
    "dt_years",       # dt_days / 365.25, rounded to 4 (matches Pair.pa_dt_years)
    "resolution_m",   # GSD of the LEFT image — ASP disparity is in left pixels
    "group",          # "<sensor>|<dt-bin>" — the parameter-set identity
    "velocity_m_yr",  # assumed surface velocity that drove the derivation
    "disp_m",         # expected displacement, metres  (v * dt)
    "disp_px",        # expected displacement, pixels  (disp_m / resolution_m)
    "kernel_px",      # derived corr-kernel side
    "footprint_m",    # kernel_px * resolution_m — the physical template size
    "search_px",      # derived corr-search half-width
    "snr",            # disp_m / noise_floor_m
    "cost_index",     # relative runtime, 1.0 = ASP defaults
    "source",         # "auto" | "group" | "override" — where the numbers came from
    "flags",          # "; "-joined warnings, "" when clean
)

_DTYPES: dict[str, str] = {
    "pa_key": "object", "pz": "object",
    "sensor_i": "object", "sensor_j": "object",
    "dt_days": "int64", "dt_years": "float64", "resolution_m": "float64",
    "group": "object", "velocity_m_yr": "float64",
    "disp_m": "float64", "disp_px": "float64",
    "kernel_px": "int64", "footprint_m": "float64", "search_px": "float64",
    "snr": "float64", "cost_index": "float64",
    "source": "object", "flags": "object",
}

#: Values of ``source``, in precedence order — later wins.  ``"auto"`` is the
#: per-pair derivation, ``"group"`` a value edited for a whole group, and
#: ``"override"`` a single pair the user pinned by hand.
SOURCE_ORDER: tuple[str, ...] = ("auto", "group", "override")

#: Which explorer controls actually change the derived parameters, keyed by the
#: axis they act on.  Read by three things that must agree: widget visibility,
#: the figure-stem pruning in
#: :func:`~geomulticorr.utils._corrparams_export.corrparams_figure_stem`, and the
#: ``relevant_params`` field of the correlation plan JSON.
#:
#: ``flow_azimuth_deg`` and ``strain_rate_per_yr`` are optional refinements: they
#: are listed so a run that sets them cannot collide on a stem with one that
#: does not, and :func:`relevant_corr_keys` drops them when they are None.
CORR_MODE_KEYS: dict[str, set[str]] = {
    "asp_bm": {
        "footprint_m", "c_px", "k", "coreg_px", "margin_px",
        "velocity_m_yr", "flow_azimuth_deg", "strain_rate_per_yr",
        "subpixel_mode", "dt_bin_ratio",
        "explicit_search", "explicit_search_above_days",
        "uniform_kernel", "uniform_kernel_px",
        "uniform_search", "uniform_search_box",
    },
    # SGM/MGM force the kernel to 9x9, so footprint is inert there — listing it
    # would put a number in the stem that changed nothing about the run.
    "asp_sgm": {
        "c_px", "k", "coreg_px", "margin_px",
        "velocity_m_yr", "flow_azimuth_deg", "dt_bin_ratio",
        "explicit_search", "explicit_search_above_days",
        "uniform_search", "uniform_search_box",
    },
}
CORR_MODE_KEYS["asp_mgm"] = set(CORR_MODE_KEYS["asp_sgm"])
CORR_MODE_KEYS["asp_final_mgm"] = set(CORR_MODE_KEYS["asp_sgm"])


def relevant_corr_keys(
    corr_algorithm: str,
    *,
    flow_azimuth_deg: float | None = None,
    strain_rate_per_yr: float | None = None,
    explicit_search: bool = True,
    uniform_kernel: bool = False,
    uniform_search: bool = False,
) -> set[str]:
    """Parameters that change the derivation for *corr_algorithm*.

    Drops the two optional refinements when they are unset, so a run that leaves
    them at None does not carry a ``flowNone`` fragment through every filename.

    :param corr_algorithm: ASP ``stereo-algorithm`` value.
    :param flow_azimuth_deg: Flow azimuth, or None when not used.
    :param strain_rate_per_yr: Strain rate, or None when not used.
    :returns: A new set (never the stored one — callers mutate it).
    """
    relevant = set(CORR_MODE_KEYS.get(corr_algorithm, CORR_MODE_KEYS["asp_bm"]))
    if flow_azimuth_deg is None:
        relevant.discard("flow_azimuth_deg")
    if strain_rate_per_yr is None:
        relevant.discard("strain_rate_per_yr")
    # A threshold nobody applied, and a fixed value nobody asked for, are not
    # part of the recipe — naming them in a stem would invent a distinction.
    if not explicit_search:
        relevant.discard("explicit_search_above_days")
    if not uniform_kernel:
        relevant.discard("uniform_kernel_px")
    if not uniform_search:
        relevant.discard("uniform_search_box")
    return relevant


def empty_corrparams_frame() -> pd.DataFrame:
    """A zero-row correlation-parameters frame with the right columns/dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _DTYPES.items()})


def group_key(sensor: str, dt_label: str) -> str:
    """Identity of one parameter group: ``"<sensor>|<dt-bin>"``.

    Sensor is the second grouping axis because resolution differs between
    sensors, and the pixel conversion of every physical scale depends on it.

    :param sensor: Sensor name, or ``""`` when unknown.
    :param dt_label: Δt bin label from
        :func:`~geomulticorr.correlation.corr_params.geometric_dt_bins`.
    :returns: The group key.
    """
    return f"{sensor or 'unknown'}|{dt_label or 'all'}"


def _assemble(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build the frame from per-pair dicts. Single source of truth for dtypes.

    Both public constructors funnel through here, which is what keeps a derived
    frame and a replayed one byte-comparable.
    """
    if not rows:
        return empty_corrparams_frame()

    frame = pd.DataFrame(list(rows))
    for column in CORRPARAMS_FRAME_COLUMNS:
        if column not in frame.columns:
            frame[column] = _default_for(column)
    # dt_years is always derived, never trusted from the caller: it must match
    # Pair.pa_dt_years bit-for-bit or a joined table shows two different years
    # for one pair.
    frame["dt_years"] = np.round(
        frame["dt_days"].to_numpy(dtype="float64") / DAYS_PER_YEAR, 4
    )
    return frame.astype(_DTYPES)[list(CORRPARAMS_FRAME_COLUMNS)]


def _default_for(column: str) -> Any:
    """Neutral value for a column the caller did not supply."""
    dtype = _DTYPES[column]
    if dtype == "object":
        return ""
    if dtype == "int64":
        return 0
    return np.nan


def corrparams_frame_from_pairs(
    pa_keys: Sequence[str],
    dt_days: Sequence[float],
    resolution_m: Sequence[float],
    *,
    pz: Sequence[str] | str = "",
    sensor_i: Sequence[str] | None = None,
    sensor_j: Sequence[str] | None = None,
    velocity_m_yr: float = 1.0,
    footprint_m: float = 60.0,
    c_px: float = 0.15,
    k: float = 3.0,
    coreg_px: float = 2.0,
    margin_px: float = 2.0,
    flow_azimuth_deg: float | None = None,
    strain_rate_per_yr: float | None = None,
    corr_algorithm: str = "asp_bm",
    subpixel_mode: int = 2,
    tau_days: float | None = None,
    dt_bin_ratio: float = 3.0,
    explicit_search: bool = True,
    explicit_search_above_days: float = 0.0,
    group_params: Mapping[str, Mapping[str, Any]] | None = None,
    params_by_pair: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Derive parameters for every pair and return them as a frame.

    This is the **derived** constructor: it runs
    :func:`~geomulticorr.correlation.corr_params.suggest_parameters` per pair,
    bins the baselines geometrically, and then lets group edits and per-pair
    overrides win in that order — the ``auto -> group -> override`` precedence
    that :meth:`Session.prepare_pairs_correlation` applies at write time.

    :param pa_keys: Pair keys, one per row.
    :param dt_days: Temporal baselines, days.
    :param resolution_m: GSD of each pair's **left** image, m/px.
    :param pz: Pzone name(s); a bare string is broadcast.
    :param sensor_i: Left sensors; defaults to ``""``.
    :param sensor_j: Right sensors; defaults to ``""``.
    :param velocity_m_yr: Assumed surface velocity, m/yr.
    :param footprint_m: Desired template ground footprint, metres.
    :param c_px: Subpixel matching precision, pixels.
    :param k: Confidence multiplier.
    :param coreg_px: Co-registration allowance, pixels.
    :param margin_px: Search safety margin, pixels.
    :param flow_azimuth_deg: Flow azimuth for an asymmetric box, or None.
    :param strain_rate_per_yr: Strain rate for the kernel ceiling, or None.
    :param corr_algorithm: ASP ``stereo-algorithm`` value.
    :param subpixel_mode: ASP ``subpixel-mode``.
    :param tau_days: Decorrelation time for coherence warnings, or None.
    :param dt_bin_ratio: Multiplicative width of each Δt bin.
    :param group_params: ``{group_key: {"corr_kernel": …, "corr_search": …}}``
        overriding the derivation for a whole group.
    :param params_by_pair: ``{pa_key: {...}}`` overriding a single pair.
    :returns: A frame matching :data:`CORRPARAMS_FRAME_COLUMNS`.
    """
    n = len(pa_keys)
    if n == 0:
        return empty_corrparams_frame()
    if len(dt_days) != n or len(resolution_m) != n:
        raise ValueError(
            f"pa_keys, dt_days and resolution_m must be the same length; "
            f"got {n}, {len(dt_days)}, {len(resolution_m)}"
        )

    pz_seq = [pz] * n if isinstance(pz, str) else list(pz)
    si = list(sensor_i) if sensor_i is not None else [""] * n
    sj = list(sensor_j) if sensor_j is not None else [""] * n
    for name, seq in (("pz", pz_seq), ("sensor_i", si), ("sensor_j", sj)):
        if len(seq) != n:
            raise ValueError(f"{name} must have length {n}, got {len(seq)}")

    edges, labels = geometric_dt_bins(dt_days, ratio=dt_bin_ratio)
    group_params = group_params or {}
    params_by_pair = params_by_pair or {}

    rows: list[dict[str, Any]] = []
    for i in range(n):
        key = str(pa_keys[i])
        dt = float(dt_days[i])
        res = float(resolution_m[i])
        group = group_key(si[i], assign_dt_bin(dt, edges, labels))

        derived = suggest_parameters(
            res,
            dt,
            velocity_m_yr,
            footprint_m=footprint_m,
            c_px=c_px,
            k=k,
            coreg_px=coreg_px,
            margin_px=margin_px,
            flow_azimuth_deg=flow_azimuth_deg,
            strain_rate_per_yr=strain_rate_per_yr,
            corr_algorithm=corr_algorithm,
            subpixel_mode=subpixel_mode,
            tau_days=tau_days,
            explicit_search=explicit_search,
            explicit_search_above_days=explicit_search_above_days,
        )

        source = "auto"
        kernel = max(derived["corr_kernel"])
        search = derived["search_px"]
        corr_search = derived["corr_search"]
        cost = derived["cost_index"]
        flags = list(derived["warnings"])

        for level, table, lookup in (
            ("group", group_params, group),
            ("override", params_by_pair, key),
        ):
            patch = table.get(lookup)
            if not patch:
                continue
            # A plan nests its group parameters under "params"; a hand-written
            # mapping may be flat.  Accept both, exactly as resolve_pair_params
            # does -- reading only the flat shape set `source` to "group"
            # while silently changing no value, so a replayed plan reported a
            # reduction it had not applied.
            patch = dict(patch.get("params") or patch)
            source = level
            patched_kernel = patched_search = None
            if "corr_kernel" in patch and patch["corr_kernel"] is not None:
                kernel = max(_as_pair(patch["corr_kernel"]))
                patched_kernel = kernel
            if "corr_search" in patch:
                corr_search = patch["corr_search"]
                search = search_half_width_px(corr_search)
                patched_search = corr_search
            # Cost is a function of the kernel and the box, so it has to follow
            # them.  Leaving the derived value in place made the cost view
            # under-report a patched group by orders of magnitude -- exactly the
            # figure someone consults before committing a night of compute.
            cost = cost_index(
                (kernel, kernel), corr_search,
                str(patch.get("corr_algorithm", corr_algorithm)),
            )
            # A patched value that violates what this pair actually needs is
            # reported, never clamped: the caller gets the parameters they asked
            # for plus a record of which pairs they are wrong for.  In the normal
            # flow this stays silent -- the group reduction takes min(kernel) and
            # max(search), so it cannot exceed any member's limits -- and fires
            # only on a genuine override.
            flags.extend(
                uniform_conflicts(
                    derived,
                    kernel_px=patched_kernel,
                    corr_search=patched_search,
                    corr_algorithm=str(patch.get("corr_algorithm", corr_algorithm)),
                )
            )

        rows.append(
            {
                "pa_key": key,
                "pz": pz_seq[i],
                "sensor_i": si[i],
                "sensor_j": sj[i],
                "dt_days": int(round(dt)),
                "resolution_m": res,
                "group": group,
                "velocity_m_yr": float(velocity_m_yr),
                "disp_m": derived["disp_m"],
                "disp_px": derived["disp_px"],
                "kernel_px": int(kernel),
                "footprint_m": float(kernel) * res,
                "search_px": float(search),
                "snr": derived["snr"],
                "cost_index": cost,
                "source": source,
                "flags": "; ".join(flags),
            }
        )

    return _assemble(rows)


def corrparams_frame_from_plan(plan: Mapping[str, Any]) -> pd.DataFrame:
    """Replay a correlation plan JSON back into a frame.

    The inverse of writing a plan: given the assumptions and the per-pair facts
    the plan recorded, re-derive the same rows
    :func:`corrparams_frame_from_pairs` produced.  This is what makes a plan a
    genuine trace rather than a summary — the figures can be regenerated from
    the file alone, months later, with no session open.

    :param plan: A plan mapping as built by
        :func:`~geomulticorr.correlation._correlation_plan.build_correlation_plan`.
    :returns: A frame matching :data:`CORRPARAMS_FRAME_COLUMNS`.
    """
    pairs = list(plan.get("pairs") or [])
    if not pairs:
        return empty_corrparams_frame()

    assumptions = dict(plan.get("assumptions") or {})
    groups = {
        str(name): dict(block.get("params") or {})
        for name, block in (plan.get("groups") or {}).items()
    }
    overrides = {
        str(name): dict(block or {}) for name, block in (plan.get("overrides") or {}).items()
    }

    return corrparams_frame_from_pairs(
        [p["pa_key"] for p in pairs],
        [p["dt_days"] for p in pairs],
        [p["resolution_m"] for p in pairs],
        pz=[p.get("pz", "") for p in pairs],
        sensor_i=[p.get("sensor_i", "") for p in pairs],
        sensor_j=[p.get("sensor_j", "") for p in pairs],
        group_params=groups,
        params_by_pair=overrides,
        **{key: assumptions[key] for key in _PLAN_ASSUMPTION_KEYS if key in assumptions},
    )


#: Assumption keys a plan carries through to the derivation.  Kept explicit so
#: an unknown key in an old plan file is ignored rather than raising a
#: ``TypeError`` deep inside ``suggest_parameters``.
_PLAN_ASSUMPTION_KEYS: tuple[str, ...] = (
    "velocity_m_yr", "footprint_m", "c_px", "k", "coreg_px", "margin_px",
    "flow_azimuth_deg", "strain_rate_per_yr", "corr_algorithm",
    "subpixel_mode", "tau_days", "dt_bin_ratio",
    "explicit_search", "explicit_search_above_days",
)


def _as_pair(value: Any) -> tuple[int, int]:
    """Coerce a kernel spec to ``(h, w)``, accepting a scalar or a sequence."""
    if isinstance(value, (tuple, list, np.ndarray)):
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def group_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per parameter group, for the explorer's editable table.

    Groups are the unit a parameter set applies to, so this is the view the user
    actually tunes: how many pairs it covers, the baseline span it has to serve,
    and the kernel/search it resolved to.

    :param frame: A correlation-parameters frame.
    :returns: A frame indexed by ``group`` with the per-group summary.
    """
    if len(frame) == 0:
        return pd.DataFrame(
            {
                c: pd.Series(dtype=t)
                for c, t in {
                    "group": "object", "n_pairs": "int64",
                    "sensor": "object", "resolution_m": "float64",
                    "dt_min_days": "int64", "dt_max_days": "int64",
                    "dt_span": "object",
                    "kernel_px": "int64", "footprint_m": "float64",
                    "search_px": "float64", "snr_min": "float64",
                    "cost_index": "float64", "n_flagged": "int64",
                }.items()
            }
        )

    grouped = frame.groupby("group", sort=True)
    out = pd.DataFrame(
        {
            "group": list(grouped.groups.keys()),
            "n_pairs": grouped.size().to_numpy().astype("int64"),
            "sensor": grouped["sensor_i"].first().to_numpy(),
            "resolution_m": grouped["resolution_m"].median().to_numpy(),
            "dt_min_days": grouped["dt_days"].min().to_numpy().astype("int64"),
            "dt_max_days": grouped["dt_days"].max().to_numpy().astype("int64"),
            # Presentation only — the group KEY stays the canonical days form,
            # because it is also a plan-JSON name and a figure-stem fragment.
            "dt_span": [
                format_dt_span(lo, hi) for lo, hi in zip(
                    grouped["dt_days"].min().to_numpy(),
                    grouped["dt_days"].max().to_numpy(),
                )
            ],
            # The two extremes go in OPPOSITE directions, and getting this
            # backwards silently breaks the longest-baseline pairs in a bin.
            #
            # The search box must REACH the farthest-moving pair, so it takes
            # the maximum -- a box sized for the median rails on the longest
            # baseline sharing the bin.
            #
            # The kernel is a CEILING, not a reach: each pair's strain limit
            # caps it, and exceeding that limit breaks matching outright.  So
            # the group takes the minimum -- the most constrained pair.  A max
            # here would write a 21 px kernel into a pair whose strain ceiling
            # was 8 px, which is precisely the failure the ceiling exists to
            # prevent.  (kernel_from_footprint already clamps at the texture
            # floor, so the minimum is never below it.)
            "kernel_px": grouped["kernel_px"].min().to_numpy().astype("int64"),
            "footprint_m": grouped["footprint_m"].min().to_numpy(),
            "search_px": grouped["search_px"].max().to_numpy(),
            "snr_min": grouped["snr"].min().to_numpy(),
            "cost_index": grouped["cost_index"].sum().to_numpy(),
            "n_flagged": grouped["flags"]
            .apply(lambda s: int((s.astype(str) != "").sum()))
            .to_numpy()
            .astype("int64"),
        }
    )
    return out.reset_index(drop=True)


def _finite_max(series: pd.Series) -> float | None:
    """Largest finite value, or None when there is none.

    ``np.nanmax`` warns and returns nan on an all-NaN slice, which happens
    legitimately whenever every pair is left on ASP's automatic search range.
    """
    values = series.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else None


def corrparams_stats(frame: pd.DataFrame, *, k: float = 3.0) -> dict:
    """Counts and spans for a user-facing summary.

    :param frame: A correlation-parameters frame.
    :param k: Confidence multiplier, so ``n_below_floor`` matches the design map.
    :returns: A JSON-safe dict of plain Python scalars.
    """
    n = int(len(frame))
    if n == 0:
        return {
            "n_pairs": 0, "n_groups": 0, "n_below_floor": 0, "n_flagged": 0,
            "dt_min_days": None, "dt_max_days": None,
            "kernel_min_px": None, "kernel_max_px": None,
            "search_max_px": None, "n_auto_search": 0,
            "total_cost": 0.0, "sensors": [],
        }

    snr_values = frame["snr"].to_numpy(dtype="float64")
    return {
        "n_pairs": n,
        "n_groups": int(frame["group"].nunique()),
        "n_below_floor": int(np.sum(np.isfinite(snr_values) & (snr_values < float(k)))),
        "n_flagged": int((frame["flags"].astype(str) != "").sum()),
        "dt_min_days": int(frame["dt_days"].min()),
        "dt_max_days": int(frame["dt_days"].max()),
        "kernel_min_px": int(frame["kernel_px"].min()),
        "kernel_max_px": int(frame["kernel_px"].max()),
        # None, not nan: every pair may be on ASP's automatic range, and
        # np.nanmax warns and returns nan on an all-NaN slice.
        "search_max_px": _finite_max(frame["search_px"]),
        "n_auto_search": int(
            (~np.isfinite(frame["search_px"].to_numpy(dtype="float64"))).sum()
        ),
        "total_cost": float(round(float(frame["cost_index"].sum()), 3)),
        "sensors": sorted({s for s in frame["sensor_i"].astype(str) if s}),
    }


def format_corrparams_summary(stats: dict, *, html: bool = True) -> str:
    """One-line summary of *stats*, for the explorer's status area.

    Reports pairs **below the detection floor** prominently: those are the pairs
    that will contribute noise rather than signal to the inversion, and pruning
    them is usually a better fix than adding parameter sets (guide §2.4).

    :param stats: Output of :func:`corrparams_stats`.
    :param html: Emit HTML when True, plain text otherwise.
    :returns: The formatted summary.
    """
    n = stats.get("n_pairs", 0)
    if not n:
        return "no pairs" if not html else "<i>no pairs</i>"

    def bold(text: str) -> str:
        return f"<b>{text}</b>" if html else text

    parts = [
        f"{bold(str(n))} pairs in {bold(str(stats['n_groups']))} group(s)",
        f"Δt {stats['dt_min_days']}–{stats['dt_max_days']} d",
        f"kernel {stats['kernel_min_px']}–{stats['kernel_max_px']} px",
        (f"search ≤ {stats['search_max_px']:.0f} px"
         if stats.get("search_max_px") is not None else "search: ASP auto"),
        f"cost {stats['total_cost']:.1f}×",
    ]
    if stats.get("n_below_floor"):
        warn = f"{stats['n_below_floor']} below detection floor"
        parts.append(f"<span style='color:#c33'>{warn}</span>" if html else warn)
    if stats.get("n_flagged"):
        parts.append(f"{stats['n_flagged']} flagged")

    separator = " &nbsp;|&nbsp; " if html else " | "
    return separator.join(parts)
