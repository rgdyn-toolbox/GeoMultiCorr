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
    compute_raster_stats(raster_path, band, percentiles)

Pair-level API
--------------
    build_pair_metadata(pair)
    compute_pair_raw_stats(pair)
    init_pair_stats(pair)
    load_pair_stats(pair)
    update_pair_stats(pair, section, stats_data)
    save_raw_corr_stats(pair)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import geoutils as gu

from geomulticorr._typing import NDArrayNum
from geomulticorr._logging import logger


# ─────────────────────────────────────────────────────────────────────────────
# Low-level statistics
# ─────────────────────────────────────────────────────────────────────────────

def nmad(data: NDArrayNum, nfact: float = 1.4826) -> float:
    """Normalized Median Absolute Deviation — robust standard deviation estimator.

    nfact=1.4826 makes NMAD consistent with std for normally distributed data.
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(nfact * np.median(np.abs(arr - np.median(arr))))


def compute_raster_stats(
    raster_path: str | Path,
    band: int = 1,
    percentiles: list[int] | None = None,
) -> Optional[dict]:
    """Compute descriptive statistics for one band of a raster.

    Uses ``gu.Raster`` to open the file. NaN and nodata values are excluded
    from all calculations.

    Args:
        raster_path: Path to the raster file.
        band: 1-based band index.
        percentiles: Percentile values to compute (default: [5, 25, 50, 75, 95]).

    Returns:
        Dict with keys count_total, count_valid, valid_fraction, mean, median,
        std, nmad, min, max, and one key per percentile (e.g. ``p5``).
        Returns ``None`` if the file does not exist.
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
        "count_total":    count_total,
        "count_valid":    count_valid,
        "valid_fraction": valid_fraction,
    }

    if count_valid == 0:
        for key in ("mean", "median", "std", "nmad", "min", "max"):
            stats[key] = float("nan")
        for p in percentiles:
            stats[f"p{p}"] = float("nan")
        return stats

    stats["mean"]   = _r(float(np.mean(valid)))
    stats["median"] = _r(float(np.median(valid)))
    stats["std"]    = _r(float(np.std(valid)))
    stats["nmad"]   = _r(nmad(valid))
    stats["min"]    = _r(float(np.min(valid)))
    stats["max"]    = _r(float(np.max(valid)))
    for p in percentiles:
        stats[f"p{p}"] = _r(float(np.percentile(valid, p)))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Pair-level utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_pair_metadata(pair) -> dict:
    """Extract basic identity and spatial metadata from a Pair object.

    Opens ``pair.pa_disparity_f_path`` (without loading data) for spatial
    properties. Spatial fields fall back to ``None`` if the file doesn't exist.
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
            spatial["ncols"]        = int(r.shape[1])
            spatial["nrows"]        = int(r.shape[0])
            spatial["resolution_x"] = float(r.res[0])
            spatial["resolution_y"] = float(r.res[1])
            spatial["crs"]          = str(r.crs)
            b = r.bounds
            spatial["bounds"] = {
                "left":   float(b.left),
                "bottom": float(b.bottom),
                "right":  float(b.right),
                "top":    float(b.top),
            }
        except Exception as exc:
            logger.warning(f"Could not read spatial metadata from disparity raster: {exc}")

    return {
        "pair_key":      pair.pa_key,
        "pair_name":     pair.pa_key,
        "pzone":         pair.pa_pz_name,
        "left_date":     pair.pa_left.th_date,
        "left_sensor":   pair.pa_left.th_sensor,
        "right_date":    pair.pa_right.th_date,
        "right_sensor":  pair.pa_right.th_sensor,
        "dt_days":       pair.pa_dt_days,
        "dt_months":     pair.pa_dt_months,
        "dt_years":      pair.pa_dt_years,
        "direction":     pair.pa_direction,
        **spatial,
    }


def compute_pair_raw_stats(pair) -> dict:
    """Compute raster statistics for all available raw correlation outputs.

    Rasters included (if their path exists):
        - ``ew``               → ``pair.pa_ew_path``
        - ``ns``               → ``pair.pa_ns_path``
        - ``cc``               → ``pair.pa_cc_path``
        - ``magn``             → ``pair.pa_magn_path``
        - ``disparity_f_band1``→ ``pair.pa_disparity_f_path``, band 1
        - ``disparity_f_band2``→ ``pair.pa_disparity_f_path``, band 2

    Missing paths are silently omitted from the returned dict.
    """
    single_band_targets = {
        "ew":   pair.pa_ew_path,
        "ns":   pair.pa_ns_path,
        "cc":   pair.pa_cc_path,
        "magn": pair.pa_magn_path,
    }
    raw_stats: dict = {}

    for key, path in single_band_targets.items():
        result = compute_raster_stats(path, band=1)
        if result is not None:
            raw_stats[key] = result

    for band_idx, band_key in ((1, "disparity_f_band1"), (2, "disparity_f_band2")):
        result = compute_raster_stats(pair.pa_disparity_f_path, band=band_idx)
        if result is not None:
            raw_stats[band_key] = result

    return raw_stats


# ─────────────────────────────────────────────────────────────────────────────
# JSON I/O
# ─────────────────────────────────────────────────────────────────────────────

def init_pair_stats(pair) -> dict:
    """Create the skeleton stats JSON for a pair and write it to disk.

    Structure::

        {
            "metadata":       { ... },
            "raw_corr_stats": {},
            "corrected_stats": {}
        }

    Returns the created dict.
    """
    stats_dict = {
        "metadata":        build_pair_metadata(pair),
        "raw_corr_stats":  {},
        "corrected_stats": {},
    }
    pair.pa_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pair.pa_stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    logger.file(f"Pair stats file initialised: {pair.pa_stats_path}")
    return stats_dict


def load_pair_stats(pair) -> dict:
    """Load the pair stats JSON from ``pair.pa_stats_path``.

    Raises:
        FileNotFoundError: if the stats file does not exist yet.
    """
    if not pair.pa_stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found for pair '{pair.pa_key}'. "
            f"Call init_pair_stats(pair) or save_raw_corr_stats(pair) first."
        )
    with open(pair.pa_stats_path) as f:
        return json.load(f)


def update_pair_stats(pair, section: str, stats_data: dict) -> dict:
    """Update one section of the pair stats JSON and write it back to disk.

    Creates the file via ``init_pair_stats`` if it does not yet exist.

    Args:
        pair:       A ``geomulticorr.pair.Pair`` instance.
        section:    Key to update (e.g. ``"raw_corr_stats"``).
        stats_data: Dict to store under that key.

    Returns:
        The full updated stats dict.
    """
    if pair.pa_stats_path.exists():
        stats_dict = load_pair_stats(pair)
    else:
        stats_dict = init_pair_stats(pair)

    stats_dict[section] = stats_data

    with open(pair.pa_stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    return stats_dict


def save_raw_corr_stats(pair) -> dict:
    """Compute and persist raw correlation statistics for a pair.

    Convenience wrapper around ``compute_pair_raw_stats`` and
    ``update_pair_stats``.

    Returns:
        The full updated stats dict (all three sections).
    """
    raw = compute_pair_raw_stats(pair)
    result = update_pair_stats(pair, "raw_corr_stats", raw)
    logger.statistics(
        f"Raw correlation stats saved for '{pair.pa_key}' "
        f"({len(raw)} raster(s) processed)"
    )
    return result
