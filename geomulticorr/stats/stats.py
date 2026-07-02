#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# stats.py
# creation date: 2026-05-11.
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
Raster statistics and pair-level stats JSON management.

Low-level API
-------------
    nmad(data)
    fraction_above(data, thresholds)
    count_above(data, thresholds)
    compute_raster_stats(raster_path, band, percentiles)

Pair-level API
--------------
    build_pair_metadata(pair)
    compute_pair_raw_stats(pair)
    init_pair_stats(pair)
    load_pair_stats(pair)
    update_pair_stats(pair, section, stats_data)
    save_raw_corr_stats(pair)
    save_corrected_stats(pair, xDisp_corr, yDisp_corr, correction_name)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import geoutils as gu
import numpy as np

from geomulticorr._logging import logger
from geomulticorr._typing import NDArrayNum

if TYPE_CHECKING:
    from geomulticorr.core.pair import Pair


# ─────────────────────────────────────────────────────────────────────────────
# Low-level statistics
# ─────────────────────────────────────────────────────────────────────────────

#: Default correlation thresholds for the CC-quality metric (see
#: :func:`fraction_above`). A CC map is bounded and left-skewed, so the fraction
#: of valid pixels above these thresholds is a more meaningful "average quality"
#: summary than mean/median/NMAD.
CC_QUALITY_THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 0.75)


def nmad(data: NDArrayNum, nfact: float = 1.4826) -> float:
    """Normalized Median Absolute Deviation — robust standard deviation estimator.

    With ``nfact=1.4826`` (default), NMAD is consistent with the standard
    deviation for normally distributed data.

    :param data: Input array (any shape; non-finite values are ignored).
    :type data: NDArrayNum
    :param nfact: Consistency factor.
    :type nfact: float
    :returns: NMAD value, or ``nan`` if no finite values are present.
    :rtype: float
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(nfact * np.median(np.abs(arr - np.median(arr))))


def fraction_above(
    data: NDArrayNum,
    thresholds: tuple[float, ...] = CC_QUALITY_THRESHOLDS,
) -> dict[float, float]:
    """Fraction of valid (finite) values greater than or equal to each threshold.

    This is the recommended "average quality" metric for a bounded, skewed
    correlation (CC/NCC) map: ``fraction_above(cc, τ)`` is the share of valid
    pixels whose correlation is at least ``τ``. The comparison is inclusive
    (``>= τ``). Non-finite values (NaN/inf, e.g. masked nodata) are excluded
    from both the numerator and the denominator.

    :param data: Input array (any shape; non-finite values are ignored).
    :type data: NDArrayNum
    :param thresholds: Thresholds to evaluate.
    :type thresholds: tuple[float, ...]
    :returns: Mapping ``{threshold: fraction}`` rounded to 6 decimals; each
        fraction is ``nan`` when no finite values are present.
    :rtype: dict[float, float]
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {float(t): float("nan") for t in thresholds}
    n = arr.size
    return {
        float(t): float(round(int(np.count_nonzero(arr >= t)) / n, 6))
        for t in thresholds
    }


def count_above(
    data: NDArrayNum,
    thresholds: tuple[float, ...] = CC_QUALITY_THRESHOLDS,
) -> dict[float, int]:
    """Number of valid (finite) values greater than or equal to each threshold.

    The absolute-count counterpart of :func:`fraction_above`: for a CC/NCC map,
    ``count_above(cc, τ)`` is the number of valid pixels whose correlation is at
    least ``τ`` (the numerator that :func:`fraction_above` divides by). The
    comparison is inclusive (``>= τ``). Non-finite values (NaN/inf, e.g. masked
    nodata) are excluded.

    :param data: Input array (any shape; non-finite values are ignored).
    :type data: NDArrayNum
    :param thresholds: Thresholds to evaluate.
    :type thresholds: tuple[float, ...]
    :returns: Mapping ``{threshold: count}``; each count is ``0`` when no finite
        values are present.
    :rtype: dict[float, int]
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return {float(t): int(np.count_nonzero(arr >= t)) for t in thresholds}


def compute_raster_stats(
    raster_path: str | Path,
    band: int = 1,
    percentiles: list[int] | None = None,
) -> dict | None:
    """Compute descriptive statistics for one band of a GeoTIFF.

    Opens the file with :class:`geoutils.Raster`. NaN and nodata values
    are excluded from all calculations.

    :param raster_path: Path to the GeoTIFF file.
    :type raster_path: str | Path
    :param band: 1-based band index to process.
    :type band: int
    :param percentiles: Percentile values to compute.
        Defaults to ``[5, 25, 50, 75, 95]``.
    :type percentiles: list[int] | None
    :returns: Dictionary with keys ``count_total``, ``count_valid``,
        ``valid_fraction``, ``mean``, ``median``, ``std``, ``nmad``,
        ``min``, ``max``, and one key per percentile (e.g. ``p5``).
        Returns ``None`` if the file does not exist.
    :rtype: dict | None
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    raster_path = Path(raster_path)
    if not raster_path.exists():
        return None

    r = gu.Raster(str(raster_path), load_data=True)

    # Extract the requested band as a float64 array
    if r.count == 1:
        raw = r.data.squeeze().astype(float)
    else:
        raw = np.array(r.data[band - 1], dtype=float)

    # Mask nodata / fill_value if the array is masked
    if np.ma.is_masked(raw):
        arr = raw.filled(np.nan)
    else:
        arr = raw

    count_total = int(arr.size)
    finite_mask = np.isfinite(arr)
    valid = arr[finite_mask]
    count_valid = int(valid.size)
    valid_fraction = round(count_valid / count_total, 6) if count_total > 0 else 0.0

    def _r(v: float) -> float:
        return float(round(v, 6))

    stats: dict = {
        "count_total": count_total,
        "count_valid": count_valid,
        "valid_fraction": valid_fraction,
    }

    if count_valid == 0:
        for key in ("mean", "median", "std", "nmad", "min", "max"):
            stats[key] = float("nan")
        for p in percentiles:
            stats[f"p{p}"] = float("nan")
        return stats

    stats["mean"] = _r(float(np.mean(valid)))
    stats["median"] = _r(float(np.median(valid)))
    stats["std"] = _r(float(np.std(valid)))
    stats["nmad"] = _r(nmad(valid))
    stats["min"] = _r(float(np.min(valid)))
    stats["max"] = _r(float(np.max(valid)))
    for p in percentiles:
        stats[f"p{p}"] = _r(float(np.percentile(valid, p)))

    return stats


def _compute_array_stats(
    arr_ma: np.ma.MaskedArray,
    percentiles: list[int] | None = None,
) -> dict:
    """Compute descriptive statistics from an in-memory masked array.

    Produces the same dict schema as :func:`compute_raster_stats`.

    :param arr_ma: Input masked array (any shape; flattened internally).
    :type arr_ma: numpy.ma.MaskedArray
    :param percentiles: Percentile values to compute.
        Defaults to ``[5, 25, 50, 75, 95]``.
    :type percentiles: list[int] | None
    :returns: Stats dict with keys ``count_total``, ``count_valid``,
        ``valid_fraction``, ``mean``, ``median``, ``std``, ``nmad``,
        ``min``, ``max``, and one key per percentile (e.g. ``p5``).
    :rtype: dict
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    arr = np.ma.filled(arr_ma, np.nan).astype(float).ravel()
    count_total = int(arr.size)
    valid = arr[np.isfinite(arr)]
    count_valid = int(valid.size)
    valid_fraction = round(count_valid / count_total, 6) if count_total > 0 else 0.0

    def _r(v: float) -> float:
        return float(round(v, 6))

    stats: dict = {
        "count_total": count_total,
        "count_valid": count_valid,
        "valid_fraction": valid_fraction,
    }
    if count_valid == 0:
        for key in ("mean", "median", "std", "nmad", "min", "max"):
            stats[key] = float("nan")
        for p in percentiles:
            stats[f"p{p}"] = float("nan")
        return stats

    stats.update(
        {
            "mean": _r(np.mean(valid)),
            "median": _r(np.median(valid)),
            "std": _r(np.std(valid)),
            "nmad": _r(nmad(valid)),
            "min": _r(np.min(valid)),
            "max": _r(np.max(valid)),
        }
    )
    for p in percentiles:
        stats[f"p{p}"] = _r(float(np.percentile(valid, p)))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Pair-level utilities
# ─────────────────────────────────────────────────────────────────────────────


def build_pair_metadata(pair: Pair) -> dict:
    """Extract identity and spatial metadata from a Pair.

    Opens ``pair.pa_disparity_f_path`` without loading pixel data to read
    spatial properties. All spatial fields fall back to ``None`` if the
    file does not exist.

    :param pair: Source pair instance.
    :type pair: Pair
    :returns: Dictionary with keys ``pair_key``, ``pair_name``, ``pzone``,
        ``left_date``, ``left_sensor``, ``right_date``, ``right_sensor``,
        ``dt_days``, ``dt_months``, ``dt_years``, ``direction``,
        ``ncols``, ``nrows``, ``resolution_x``, ``resolution_y``,
        ``crs``, and ``bounds``.
    :rtype: dict
    """
    spatial: dict = {
        "ncols": None,
        "nrows": None,
        "resolution_x": None,
        "resolution_y": None,
        "crs": None,
        "bounds": None,
    }

    if Path(pair.pa_disparity_f_path).exists():
        try:
            r = gu.Raster(str(pair.pa_disparity_f_path), load_data=False)
            spatial["ncols"] = int(r.shape[1])
            spatial["nrows"] = int(r.shape[0])
            spatial["resolution_x"] = float(r.res[0])
            spatial["resolution_y"] = float(r.res[1])
            spatial["crs"] = str(r.crs)
            b = r.bounds
            spatial["bounds"] = {
                "left": float(b.left),
                "bottom": float(b.bottom),
                "right": float(b.right),
                "top": float(b.top),
            }
        except Exception as exc:
            logger.warning(
                f"Could not read spatial metadata from disparity raster: {exc}"
            )

    return {
        "pair_key": pair.pa_key,
        "pair_name": pair.pa_key,
        "pzone": pair.pa_pz_name,
        "left_date": pair.pa_left.th_date,
        "left_sensor": pair.pa_left.th_sensor,
        "right_date": pair.pa_right.th_date,
        "right_sensor": pair.pa_right.th_sensor,
        "dt_days": pair.pa_dt_days,
        "dt_months": pair.pa_dt_months,
        "dt_years": pair.pa_dt_years,
        "direction": pair.pa_direction,
        **spatial,
    }


def compute_pair_raw_stats(pair: Pair) -> dict:
    """Compute raster statistics for all available raw correlation outputs.

    Rasters included if their path exists:

    * ``ew``   — ``pair.pa_ew_path``
    * ``ns``   — ``pair.pa_ns_path``
    * ``cc``   — ``pair.pa_cc_raw_path``
    * ``magn`` — ``pair.pa_magn_path``

    Missing paths are silently omitted from the returned dict. The ``cc`` entry
    is additionally enriched with two CC-quality families keyed by correlation
    threshold (0.25 / 0.5 / 0.75):

    * ``cc_quality_gte_025``/``050``/``075`` — fraction of valid pixels with
      correlation at or above the threshold (proportion, 0–1; see
      :func:`fraction_above`).
    * ``cc_fraction_gte_025``/``050``/``075`` — number of valid pixels at or
      above the threshold (absolute count; see :func:`count_above`).

    :param pair: Source pair instance.
    :type pair: Pair
    :returns: Dict mapping layer name to its stats dict
        (same schema as :func:`compute_raster_stats`; the ``cc`` entry carries
        the extra ``cc_quality_gte_*`` and ``cc_fraction_gte_*`` keys).
    :rtype: dict
    """
    single_band_targets = {
        "ew": pair.pa_ew_path,
        "ns": pair.pa_ns_path,
        "cc": pair.pa_cc_raw_path,
        "magn": pair.pa_magn_path,
    }
    raw_stats: dict = {}

    for key, path in single_band_targets.items():
        result = compute_raster_stats(path, band=1)
        if result is not None:
            raw_stats[key] = result

    # CC-quality metrics — the meaningful "average quality" summary for a
    # bounded, skewed CC map (mean/median/NMAD are not). Applied to the raw CC
    # only. Two complementary families per correlation threshold:
    #   cc_quality_gte_*  = fraction of valid pixels >= τ (proportion, 0–1)
    #   cc_fraction_gte_* = number  of valid pixels >= τ (absolute pixel count)
    # so cc_quality_gte_τ == cc_fraction_gte_τ / count_valid.
    if "cc" in raw_stats:
        cc_arr = gu.Raster(str(pair.pa_cc_raw_path), load_data=True).data
        fracs = fraction_above(cc_arr, CC_QUALITY_THRESHOLDS)
        counts = count_above(cc_arr, CC_QUALITY_THRESHOLDS)
        for thr in CC_QUALITY_THRESHOLDS:
            tag = f"{int(round(thr * 100)):03d}"
            raw_stats["cc"][f"cc_quality_gte_{tag}"] = fracs[float(thr)]
            raw_stats["cc"][f"cc_fraction_gte_{tag}"] = counts[float(thr)]

    return raw_stats


# ─────────────────────────────────────────────────────────────────────────────
# JSON I/O
# ─────────────────────────────────────────────────────────────────────────────


def init_pair_stats(pair: Pair) -> dict:
    """Create the skeleton stats JSON for a pair and write it to disk.

    The file is created at ``pair.pa_stats_path`` with the structure::

        {
            "metadata":        { ... },
            "raw_corr_stats":  {},
            "corrected_stats": {}
        }

    :param pair: Source pair instance.
    :type pair: Pair
    :returns: The created stats dict.
    :rtype: dict
    """
    stats_dict = {
        "metadata": build_pair_metadata(pair),
        "raw_corr_stats": {},
        "corrected_stats": {},
    }
    pair.pa_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pair.pa_stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    logger.file(f"Pair stats file initialised: {pair.pa_stats_path}")
    return stats_dict


def load_pair_stats(pair: Pair) -> dict:
    """Load the pair stats JSON from ``pair.pa_stats_path``.

    :param pair: Source pair instance.
    :type pair: Pair
    :returns: The full stats dict.
    :rtype: dict
    :raises FileNotFoundError: If the stats file does not exist yet.
        Call :func:`init_pair_stats` or :func:`save_raw_corr_stats` first.
    """
    if not pair.pa_stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found for pair '{pair.pa_key}'. "
            f"Call init_pair_stats(pair) or save_raw_corr_stats(pair) first."
        )
    with open(pair.pa_stats_path) as f:
        return json.load(f)


def update_pair_stats(pair: Pair, section: str, stats_data: dict) -> dict:
    """Update one section of the pair stats JSON and write it back to disk.

    Creates the file via :func:`init_pair_stats` if it does not yet exist.

    :param pair: Source pair instance.
    :type pair: Pair
    :param section: Top-level key to update (e.g. ``"raw_corr_stats"``).
    :type section: str
    :param stats_data: Dict to store under *section*.
    :type stats_data: dict
    :returns: The full updated stats dict.
    :rtype: dict
    """
    if pair.pa_stats_path.exists():
        stats_dict = load_pair_stats(pair)
    else:
        stats_dict = init_pair_stats(pair)

    stats_dict[section] = stats_data

    with open(pair.pa_stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    return stats_dict


def save_corrected_stats(
    pair: Pair,
    xDisp_corr: gu.Raster,
    yDisp_corr: gu.Raster,
    correction_name: str,
    percentiles: list[int] | None = None,
) -> dict:
    """Persist corrected displacement statistics under a named key.

    Loads the existing ``"corrected_stats"`` section, adds (or replaces) an
    entry keyed by *correction_name* with ``{"ew": {...}, "ns": {...}}``,
    then writes the merged result back to disk. Multiple calls with
    different *correction_name* values accumulate without overwriting.

    :param pair: Source pair instance (must have ``pa_stats_path``).
    :type pair: Pair
    :param xDisp_corr: Corrected EW displacement raster.
    :type xDisp_corr: geoutils.Raster
    :param yDisp_corr: Corrected NS displacement raster.
    :type yDisp_corr: geoutils.Raster
    :param correction_name: Key to use inside ``"corrected_stats"``
        (e.g. ``"MedianCentering"`` or ``str(pipeline)``).
    :type correction_name: str
    :param percentiles: Percentile values to compute.
        Defaults to ``[5, 25, 50, 75, 95]``.
    :type percentiles: list[int] | None
    :returns: The full updated stats dict (all sections).
    :rtype: dict

    Example::

        xc, yc = make_corrections(
            xDisp, yDisp, MedianCentering(),
            x_stable=x_mask, y_stable=y_mask,
        )
        save_corrected_stats(pair, xc, yc, "MedianCentering")
    """
    if pair.pa_stats_path.exists():
        corrected = load_pair_stats(pair).get("corrected_stats", {})
    else:
        corrected = {}

    corrected[correction_name] = {
        "ew": _compute_array_stats(xDisp_corr.data, percentiles),
        "ns": _compute_array_stats(yDisp_corr.data, percentiles),
    }

    result = update_pair_stats(pair, "corrected_stats", corrected)
    ew_valid = corrected[correction_name]["ew"]["count_valid"]
    ns_valid = corrected[correction_name]["ns"]["count_valid"]
    logger.statistics(
        f"Corrected stats saved for '{pair.pa_key}' [{correction_name}] "
        f"(ew valid={ew_valid}, ns valid={ns_valid})"
    )
    return result


def save_raw_corr_stats(pair: Pair) -> dict:
    """Compute and persist raw correlation statistics for a pair.

    Convenience wrapper around :func:`compute_pair_raw_stats` and
    :func:`update_pair_stats`.

    :param pair: Source pair instance.
    :type pair: Pair
    :returns: The full updated stats dict (all three sections).
    :rtype: dict
    """
    raw = compute_pair_raw_stats(pair)
    result = update_pair_stats(pair, "raw_corr_stats", raw)
    logger.statistics(
        f"Raw correlation stats saved for '{pair.pa_key}' "
        f"({len(raw)} raster(s) processed)"
    )
    return result
