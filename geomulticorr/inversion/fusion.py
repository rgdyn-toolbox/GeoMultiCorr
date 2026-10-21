#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# fusion.py
# creation date: 2026-09-14.
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
"""Fusing two same-sensor inversions into one series — and deciding which to trust.

Why this exists
---------------
Pairs are built within a sensor (SPOT–SPOT, PlanetScope–PlanetScope), so a
joint inversion of both archives is two **disconnected** sub-networks: no pair
measures the displacement between a SPOT date and a PlanetScope date. The
solver still returns one series — the smoothing prior fills the gap — but the
relative offset between the sub-series is not a measurement, and any *rate*
disagreement between the sensors lands on the intervals adjacent to the
minority dates as a spike-and-return. The solver's own diagnostics do not flag
it (``rank_defect`` stays 0, the closure RMS stays small); see
``TIOInversion.network_components()``.

The honest alternative is to invert each sensor on its own and then decide
explicitly how the two series relate:

1. :func:`pair_closure_against_series` — before choosing, measure how the
   *other* sensor's pairs disagree with one sensor's series, per pixel and
   per pair (run it both ways). This is what tells you which sensor to make
   the reference; a higher-resolution sensor is not automatically the better
   one on moving terrain (SPOT6/7's off-nadir incidence × a DEM error gives
   per-image EW artefacts a near-nadir constellation does not have).
2. :func:`calibrate_to_reference` — fit, per pixel, an offset (always: each
   run carries its own constant reference through ``lect_depl_cumule_lin``'s
   line fit) and optionally a linear drift of ``secondary − reference`` over
   the epochs where both exist, and subtract it from the secondary series.
3. :func:`fuse_series` — union of dates, the reference winning on a shared
   date, with a ``source`` coordinate saying which run each epoch came from.
4. :func:`fuse_inversions` — the three above plus
   :func:`~geomulticorr.inversion._stack.write_cumulative_stack`, so the
   product lands in the ``TOT_*.tif`` layout with its calibration maps and a
   JSON trace beside it.

Conventions
-----------
- Every series is an ``xr.Dataset`` from
  :func:`~geomulticorr.inversion._stack.load_cumulative_stack`: data
  variables ``EW``/``NS`` on ``(time, y, x)``, ``time`` datetime64, one grid.
  The two inputs must be on the **same** grid — ``xr.align(join="exact")`` —
  which also means the same solver backend (the Fortran path drops one row).
- Magnitude is never fused: it is re-derived from the fused EW/NS by the writer.
- The calibration is fitted on a float "years since the first overlap epoch"
  axis, not on datetime64: xarray's ``polyfit`` would otherwise put the
  intercept at 1970 in nanoseconds. ``offset`` means the bias at that first
  epoch, ``rate`` is in m/yr.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from geomulticorr._logging import logger
from geomulticorr.inversion._run_parameters import _jsonable
from geomulticorr.inversion._stack import (COMPONENTS, load_cumulative_stack,
                                           write_cumulative_stack)
from geomulticorr.stats.inversion_extractor import resolve_inversion_dir

#: Days per year used to turn the datetime axis into decimal years.
_DAYS_PER_YEAR = 365.25

try:  # numpy ≥ 1.25 moved it; numpy 2 removed the top-level alias
    _RankWarning = np.exceptions.RankWarning
except AttributeError:  # pragma: no cover
    _RankWarning = np.RankWarning

_MODELS = ("offset+rate", "offset")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_same_grid(a: xr.Dataset, b: xr.Dataset, what: str = "the two series") -> None:
    """Refuse two datasets that are not on exactly one grid."""
    try:
        xr.align(a, b, join="exact", exclude={"time", "component", "pair"})
    except ValueError as exc:
        raise ValueError(
            f"{what} are not on the same grid — the same pzone grid (and the same "
            f"solver backend) is required; regrid before fusing. ({exc})"
        ) from None


def _check_series(ds: xr.Dataset, components: Sequence[str], label: str) -> None:
    missing = [c for c in components if c not in ds.data_vars]
    if missing:
        raise KeyError(f"{label} series lacks {missing}; has {list(ds.data_vars)}")
    if "time" not in ds.dims:
        raise ValueError(f"{label} series has no 'time' dimension")
    for c in components:
        if ds[c].dims != ("time", "y", "x"):
            raise ValueError(f"{label}[{c}] must be (time, y, x), got {ds[c].dims}")


def _years_since(time_values, t0) -> np.ndarray:
    t = pd.to_datetime(np.asarray(time_values))
    return ((t - pd.Timestamp(t0)) / pd.Timedelta(days=_DAYS_PER_YEAR)).to_numpy(dtype="float64")


def _dates_of(ds: xr.Dataset) -> list[str]:
    if "date" in ds.coords:
        return [str(d) for d in ds["date"].values]
    return list(pd.to_datetime(ds["time"].values).strftime("%Y%m%d"))


def _as_timestamp(value) -> pd.Timestamp:
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        return pd.Timestamp(datetime.strptime(value, "%Y%m%d"))
    return pd.Timestamp(value)


# ─────────────────────────────────────────────────────────────────────────────
# 1. calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_to_reference(
    secondary: xr.Dataset,
    reference: xr.Dataset,
    *,
    model: Literal["offset+rate", "offset"] = "offset+rate",
    overlap: tuple | None = None,
    min_epochs: int = 2,
    components: Sequence[str] = COMPONENTS,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Fit and remove, per pixel, the bias of *secondary* relative to *reference*.

    Over the *overlap* — the secondary epochs that fall inside the reference's
    time span (the reference is interpolated linearly to them, never
    extrapolated) — the difference ``secondary − reference`` is fitted with a
    constant (``model="offset"``) or a constant plus a linear drift
    (``"offset+rate"``, the default). The fitted model is then subtracted from
    the secondary series on its **whole** time axis, so a rate correction is
    extrapolated outside the overlap; the trace records that.

    :param secondary: The series to correct.
    :param reference: The series to correct it towards.
    :param model: ``"offset+rate"`` or ``"offset"``.
    :param overlap: Optional ``(start, end)`` (``YYYYMMDD`` strings or
        anything :class:`pandas.Timestamp` accepts) restricting the epochs used
        for the fit within the natural overlap.
    :param min_epochs: For ``"offset+rate"``, the minimum number of overlap
        epochs; fewer raises rather than silently degrading to an offset —
        that would change what the product means.
    :param components: The data variables to calibrate.
    :returns: ``(corrected, fit)`` — the corrected secondary (same shape and
        coords, attrs gain ``calibration``), and a dataset with ``offset``,
        ``rate``, ``rms`` and ``n_valid`` on ``(component, y, x)``. Pixels with
        fewer valid overlap epochs than the model needs fall back per pixel:
        one epoch → ``rate = 0`` and ``offset`` = that single difference; none
        → NaN (the corrected pixel is NaN; ``n_valid`` says why).
    :raises ValueError: On grid mismatch, no overlap, or too few epochs.
    """
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    components = tuple(components)
    _check_series(secondary, components, "secondary")
    _check_series(reference, components, "reference")
    _require_same_grid(secondary, reference)

    ref_t = pd.to_datetime(reference["time"].values)
    sec_t = pd.to_datetime(secondary["time"].values)
    lo, hi = ref_t.min(), ref_t.max()
    if overlap is not None:
        lo = max(lo, _as_timestamp(overlap[0]))
        hi = min(hi, _as_timestamp(overlap[1]))
    in_overlap = (sec_t >= lo) & (sec_t <= hi)
    t_ov = secondary["time"].values[in_overlap]
    n_overlap = int(in_overlap.sum())
    if n_overlap == 0:
        raise ValueError(
            "no overlap: no secondary epoch falls within the reference span "
            f"{lo.date()} – {hi.date()} (reference dates {_dates_of(reference)[0]}…"
            f"{_dates_of(reference)[-1]}, secondary {_dates_of(secondary)[0]}…"
            f"{_dates_of(secondary)[-1]})"
        )
    deg = 1 if model == "offset+rate" else 0
    if deg == 1 and n_overlap < min_epochs:
        raise ValueError(
            f"model='offset+rate' needs at least {min_epochs} overlap epochs, found "
            f"{n_overlap} — use model='offset', or lower min_epochs."
        )

    t0 = pd.Timestamp(t_ov[0])
    t_yr_ov = _years_since(t_ov, t0)
    t_yr_full = _years_since(secondary["time"].values, t0)

    ref = reference[list(components)]
    if any(getattr(ref[c].data, "chunks", None) is not None for c in components):
        ref = ref.chunk({"time": -1})
    ref_on_sec = ref.interp(time=t_ov, method="linear")
    diff = (secondary[list(components)].sel(time=t_ov) - ref_on_sec)
    diff = diff.assign_coords(t_yr=("time", t_yr_ov)).swap_dims({"time": "t_yr"})

    offsets, rates, rmss, n_valids, corrected_vars = [], [], [], [], {}
    n_fallback = 0
    t_full = xr.DataArray(t_yr_full, dims=("time",), coords={"time": secondary["time"]})
    for comp in components:
        d = diff[comp]
        n_valid = d.notnull().sum("t_yr")
        with warnings.catch_warnings():
            # a pixel with a single valid epoch is rank-deficient for deg=1;
            # its fallback is handled explicitly below
            warnings.simplefilter("ignore", _RankWarning)
            pf = d.polyfit(dim="t_yr", deg=deg, skipna=True)["polyfit_coefficients"]
        # drop the scalar `degree` coord sel() leaves, or the concat below
        # sees two conflicting values of it
        offset = pf.sel(degree=0).drop_vars("degree")
        rate = (pf.sel(degree=1).drop_vars("degree") if deg == 1
                else xr.zeros_like(offset))

        if deg == 1:
            short = (n_valid < 2) & (n_valid > 0)
            n_fallback += int(short.sum())
            offset = offset.where(~short, d.mean("t_yr"))
            rate = rate.where(~short, 0.0)
        none = n_valid == 0
        offset = offset.where(~none)
        rate = rate.where(~none)

        coeffs = xr.concat([rate, offset], dim="degree").assign_coords(degree=[1, 0])
        model_ov = xr.polyval(d["t_yr"], coeffs)
        rms = np.sqrt(((d - model_ov) ** 2).mean("t_yr", skipna=True)).where(~none)

        correction = xr.polyval(t_full, coeffs)          # (time, y, x)
        corrected_vars[comp] = (secondary[comp] - correction).astype("float32")

        offsets.append(offset.astype("float32"))
        rates.append(rate.astype("float32"))
        rmss.append(rms.astype("float32"))
        n_valids.append(n_valid.astype("int16"))

    if n_fallback:
        logger.info(
            f"calibrate_to_reference: {n_fallback} pixel(s) had a single overlap "
            "epoch — rate set to 0, offset to that difference."
        )

    comp_coord = list(components)
    fit = xr.Dataset({
        "offset": xr.concat(offsets, dim="component"),
        "rate": xr.concat(rates, dim="component"),
        "rms": xr.concat(rmss, dim="component"),
        "n_valid": xr.concat(n_valids, dim="component"),
    }).assign_coords(component=comp_coord)
    fit = fit.drop_vars([c for c in ("degree", "t_yr", "time", "date") if c in fit.coords])

    attrs = {
        "model": model,
        "t0": t0.isoformat(),
        "overlap_dates": [pd.Timestamp(t).strftime("%Y%m%d") for t in t_ov],
        "n_overlap": n_overlap,
        "min_epochs": int(min_epochs),
        "reference_dates": _dates_of(reference),
        "secondary_dates": _dates_of(secondary),
        "n_fallback_pixels": int(n_fallback),
        "extrapolated": bool(model == "offset+rate"),
    }
    fit.attrs.update(attrs)
    corrected = xr.Dataset(corrected_vars, coords=secondary.coords, attrs=dict(secondary.attrs))
    corrected.attrs["calibration"] = dict(attrs)
    return corrected, fit


# ─────────────────────────────────────────────────────────────────────────────
# 2. fusion
# ─────────────────────────────────────────────────────────────────────────────

def fuse_series(
    reference: xr.Dataset,
    secondary_corrected: xr.Dataset,
    *,
    reference_label: str = "reference",
    secondary_label: str = "secondary",
    components: Sequence[str] = COMPONENTS,
) -> xr.Dataset:
    """Union of the two series' dates; the reference wins on a shared date.

    :returns: A dataset on the union of dates, sorted, with a ``source``
        coordinate (``reference_label`` / ``secondary_label``) per epoch, a
        ``date`` coordinate, and attrs ``fusion`` with the counts. A ``magn``
        variable is dropped with a warning — it is re-derived by the writer.
    """
    components = tuple(components)
    _check_series(reference, components, "reference")
    _check_series(secondary_corrected, components, "secondary")
    _require_same_grid(reference, secondary_corrected)
    if "magn" in reference.data_vars or "magn" in secondary_corrected.data_vars:
        logger.warning("fuse_series: 'magn' is dropped — magnitude is re-derived from the fused EW/NS.")

    ref = reference[list(components)]
    sec = secondary_corrected[list(components)]
    shared = np.isin(sec["time"].values, ref["time"].values)
    sec_only = sec.sel(time=~shared)

    ref = ref.assign_coords(source=("time", [reference_label] * ref.sizes["time"]))
    sec_only = sec_only.assign_coords(source=("time", [secondary_label] * sec_only.sizes["time"]))

    fused = xr.concat([ref.drop_vars("date", errors="ignore"),
                       sec_only.drop_vars("date", errors="ignore")],
                      dim="time", coords="minimal", compat="override").sortby("time")
    fused = fused.assign_coords(
        date=("time", list(pd.to_datetime(fused["time"].values).strftime("%Y%m%d"))))
    fused.attrs["fusion"] = {
        "reference": reference_label,
        "secondary": secondary_label,
        "n_reference": int(ref.sizes["time"]),
        "n_secondary_added": int(sec_only.sizes["time"]),
        "n_shared_dropped": int(shared.sum()),
    }
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# 3. closure diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def _default_pair_paths(pair) -> tuple[Path, Path]:
    """Corrected rasters when they exist, raw otherwise — the ``_load_and_filter`` rule."""
    ew = getattr(pair, "pa_ew_corr_path", None)
    ns = getattr(pair, "pa_ns_corr_path", None)
    if ew is None or not Path(ew).exists():
        ew = pair.pa_ew_path
    if ns is None or not Path(ns).exists():
        ns = pair.pa_ns_path
    return Path(ew), Path(ns)


def pair_closure_against_series(
    pairs,
    series: xr.Dataset,
    *,
    components: Sequence[str] = COMPONENTS,
    paths: Callable | None = None,
) -> xr.Dataset:
    """Residual of each pair map against a series it was **not** part of.

    For a pair ``(t1, t2)``, ``residual = pair_map − (series(t2) − series(t1))``
    with the series interpolated linearly in time — what the pair measured
    minus what the other sensor's series says happened over the same interval.
    Run with SPOT pairs against the PlanetScope series and vice versa, the two
    maps (and :func:`closure_summary`) say which sensor disagrees with the
    other, where, and whether it is a constant, a ramp or a rate.

    :param pairs: GMC ``Pair`` objects (``pa_key``, ``pa_left``/``pa_right``
        thumbs with ``th_date`` and ``th_sensor``, raster paths).
    :param series: A cumulative stack (``load_cumulative_stack``).
    :param components: ``"EW"``/``"NS"``.
    :param paths: ``pair → (ew_path, ns_path)`` override; default prefers the
        corrected rasters. Point it at the inversion's ``corrected/`` folder
        to use exactly what the solver consumed.
    :returns: ``xr.Dataset`` with the components on ``(pair, y, x)`` and
        coordinates ``pair`` (``pa_key``), ``date1``, ``date2``, ``dt_days``,
        ``sensor``; attrs ``skipped`` lists pairs outside the series span.
    """
    import rioxarray  # noqa: F401

    components = tuple(components)
    _check_series(series, components, "series")
    paths = paths or _default_pair_paths
    t_series = pd.to_datetime(series["time"].values)
    lo, hi = t_series.min(), t_series.max()
    res = abs(float(series["x"].values[1] - series["x"].values[0])) if series.sizes["x"] > 1 else 1.0
    template = series[components[0]].isel(time=0)

    stacks = {c: [] for c in components}
    keys, d1s, d2s, dts, sensors, skipped = [], [], [], [], [], []
    for pair in pairs:
        d1 = pd.Timestamp(str(pair.pa_left.th_date))
        d2 = pd.Timestamp(str(pair.pa_right.th_date))
        if not (lo <= d1 <= hi and lo <= d2 <= hi):
            skipped.append(pair.pa_key)
            continue
        ew_path, ns_path = paths(pair)
        by_comp = {"EW": ew_path, "NS": ns_path}
        sub = series[list(components)]
        predicted = (sub.interp(time=np.datetime64(d2)) - sub.interp(time=np.datetime64(d1)))
        for comp in components:
            da = rioxarray.open_rasterio(by_comp[comp], masked=True).squeeze("band", drop=True)
            da = da.reindex_like(template, method="nearest", tolerance=res / 2)
            resid = (da - predicted[comp]).astype("float32")
            stacks[comp].append(resid.drop_vars(("time", "date", "band"), errors="ignore"))
        keys.append(pair.pa_key)
        d1s.append(d1.strftime("%Y%m%d"))
        d2s.append(d2.strftime("%Y%m%d"))
        dts.append(float((d2 - d1).days))
        sensors.append(str(getattr(pair.pa_left, "th_sensor", "") or "").lower())

    if skipped:
        logger.warning(
            f"pair_closure_against_series: {len(skipped)} pair(s) outside the series span "
            f"{lo.date()} – {hi.date()} were skipped: {', '.join(skipped[:5])}"
            + (" …" if len(skipped) > 5 else "")
        )
    if not keys:
        raise ValueError("no pair falls within the series' time span")

    out = xr.Dataset({
        c: xr.concat(stacks[c], dim="pair", coords="minimal", compat="override")
        for c in components
    })
    out = out.assign_coords(pair=keys, date1=("pair", d1s), date2=("pair", d2s),
                            dt_days=("pair", dts), sensor=("pair", sensors))
    out.attrs["skipped"] = list(skipped)
    return out


def closure_summary(closure: xr.Dataset, mask: xr.DataArray | None = None) -> pd.DataFrame:
    """One row per (pair, component): where the pair disagrees with the series.

    :param closure: From :func:`pair_closure_against_series`.
    :param mask: Optional boolean ``(y, x)`` array (``True`` = use), e.g. the
        moving areas, so the disagreement is measured where it matters.
    :returns: Columns ``pair, sensor, date1, date2, dt_days, component, median,
        nmad, rms, n, rate_bias_m_yr`` (``median / dt_days × 365.25``).
    """
    from geomulticorr.stats import nmad as _nmad

    rows = []
    comps = [c for c in closure.data_vars]
    for k, key in enumerate(closure["pair"].values):
        dt_days = float(closure["dt_days"].values[k])
        for comp in comps:
            arr = np.asarray(closure[comp].isel(pair=k).values, dtype="float64")
            if mask is not None:
                arr = np.where(np.asarray(mask.values if hasattr(mask, "values") else mask, bool),
                               arr, np.nan)
            valid = arr[np.isfinite(arr)]
            med = float(np.median(valid)) if valid.size else float("nan")
            rows.append({
                "pair": str(key),
                "sensor": str(closure["sensor"].values[k]),
                "date1": str(closure["date1"].values[k]),
                "date2": str(closure["date2"].values[k]),
                "dt_days": dt_days,
                "component": comp,
                "median": med,
                "nmad": float(_nmad(valid)) if valid.size else float("nan"),
                "rms": float(np.sqrt(np.mean(valid ** 2))) if valid.size else float("nan"),
                "n": int(valid.size),
                "rate_bias_m_yr": (med / dt_days * _DAYS_PER_YEAR) if dt_days else float("nan"),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. orchestrator + trace
# ─────────────────────────────────────────────────────────────────────────────

def build_fusion_parameters(
    *,
    reference_dir, secondary_dir, reference_label: str, secondary_label: str,
    fit: xr.Dataset, fused: xr.Dataset, components: Sequence[str],
) -> dict:
    """Assemble the ``fusion_parameters.json`` document (pure; no filesystem).

    ``written_utc`` makes it not byte-stable by design — it is a log.
    """
    from geomulticorr import __version__

    def _summary(var: str, comp: str) -> dict:
        arr = np.asarray(fit[var].sel(component=comp).values, dtype="float64")
        valid = arr[np.isfinite(arr)]
        if not valid.size:
            return {"median": None, "p5": None, "p95": None}
        return {"median": float(np.median(valid)),
                "p5": float(np.percentile(valid, 5)),
                "p95": float(np.percentile(valid, 95))}

    per_component = {}
    for comp in components:
        n_valid = np.asarray(fit["n_valid"].sel(component=comp).values)
        per_component[comp] = {
            "offset": _summary("offset", comp),
            "rate_m_yr": _summary("rate", comp),
            "rms": _summary("rms", comp),
            "valid_fraction": float((n_valid > 0).mean()) if n_valid.size else None,
        }

    return {
        "gmc_version": __version__,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "fusion",
        "reference": {"inversion_dir": str(reference_dir), "label": reference_label,
                      "dates": list(fit.attrs.get("reference_dates", []))},
        "secondary": {"inversion_dir": str(secondary_dir), "label": secondary_label,
                      "dates": list(fit.attrs.get("secondary_dates", []))},
        "calibration": {
            "model": fit.attrs.get("model"),
            "t0": fit.attrs.get("t0"),
            "overlap_dates": list(fit.attrs.get("overlap_dates", [])),
            "n_overlap": fit.attrs.get("n_overlap"),
            "min_epochs": fit.attrs.get("min_epochs"),
            "n_fallback_pixels": fit.attrs.get("n_fallback_pixels"),
            "extrapolated_outside_overlap": fit.attrs.get("extrapolated"),
            "per_component": _jsonable(per_component),
        },
        "fused": {
            "n_dates": int(fused.sizes["time"]),
            "source_per_date": dict(zip(_dates_of(fused), [str(s) for s in fused["source"].values])),
            **_jsonable(fused.attrs.get("fusion", {})),
        },
        "components": list(components),
        "not_recorded_here": (
            "per-pixel offset/rate/rms/n_valid → calibration/*.tif; "
            "the fused series → inverse_*/TOT_*.tif"
        ),
    }


def fuse_inversions(
    reference,
    secondary,
    out_dir: str | Path,
    *,
    model: Literal["offset+rate", "offset"] = "offset+rate",
    overlap: tuple | None = None,
    min_epochs: int = 2,
    reference_label: str = "reference",
    secondary_label: str = "secondary",
    components: Sequence[str] = COMPONENTS,
    magnitude: bool = True,
    overwrite: bool = True,
    chunks="auto",
) -> tuple[xr.Dataset, xr.Dataset, dict]:
    """Load two inversions, calibrate the secondary to the reference, fuse, write.

    :param reference: A ``TIOInversion`` or an inversion directory.
    :param secondary: Likewise — the one that gets calibrated.
    :param out_dir: Where the product goes, typically
        ``resolve_inversion_dir(session, pzone, "<name>_fused")``.
    :returns: ``(fused, fit, written)`` — the fused dataset, the calibration
        dataset, and ``{component: {date: path}}`` for the rasters written.
    """
    ref_dir = resolve_inversion_dir(reference)
    sec_dir = resolve_inversion_dir(secondary)
    logger.info(
        f"fusion ── reference '{reference_label}' ({ref_dir.name}) ← secondary "
        f"'{secondary_label}' ({sec_dir.name}), model={model!r}"
    )
    ref = load_cumulative_stack(ref_dir, components, chunks=chunks)
    sec = load_cumulative_stack(sec_dir, components, chunks=chunks)

    corrected, fit = calibrate_to_reference(
        sec, ref, model=model, overlap=overlap, min_epochs=min_epochs, components=components)
    fused = fuse_series(ref, corrected, reference_label=reference_label,
                        secondary_label=secondary_label, components=components)
    fused = fused.compute() if hasattr(fused, "compute") else fused
    fit = fit.compute() if hasattr(fit, "compute") else fit

    for comp in components:
        summary = fit["rate"].sel(component=comp)
        logger.info(
            f"  {comp}: median offset {float(fit['offset'].sel(component=comp).median()):+.3f} m, "
            f"median rate bias {float(summary.median()):+.3f} m/yr, "
            f"overlap rms {float(fit['rms'].sel(component=comp).median()):.3f} m"
        )

    params = build_fusion_parameters(
        reference_dir=ref_dir, secondary_dir=sec_dir, reference_label=reference_label,
        secondary_label=secondary_label, fit=fit, fused=fused, components=components)
    written = write_cumulative_stack(
        fused, out_dir, components=components, magnitude=magnitude,
        overwrite=overwrite, fit=fit, parameters=params)
    logger.info(f"fusion ── {fused.sizes['time']} dates written to {out_dir}")
    return fused, fit, written
