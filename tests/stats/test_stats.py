#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# tests/stats/test_stats.py
# creation date: 2026-07-02.
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
"""Unit tests for geomulticorr.stats.stats."""
from __future__ import annotations

import json
import math

import numpy as np
import numpy.ma as ma
import pytest

from geomulticorr.stats.stats import (
    CC_QUALITY_THRESHOLDS,
    _compute_array_stats,
    build_pair_metadata,
    compute_pair_raw_stats,
    compute_raster_stats,
    count_above,
    fraction_above,
    init_pair_stats,
    load_pair_stats,
    nmad,
    save_corrected_stats,
    save_raw_corr_stats,
    update_pair_stats,
)

from .conftest import MockRasterData, write_geotiff

# ──────────────────────────────────────────────────────────────────────────────


class TestNmad:
    """Tests for nmad() — robust standard deviation estimator."""

    def test_known_value(self):
        """[1, 2, 3]: MAD = 1 → NMAD = 1.4826."""
        assert nmad(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.4826, abs=1e-4)

    def test_constant_array_returns_zero(self):
        """All-same values: NMAD = 0."""
        assert nmad(np.full(10, 5.0)) == pytest.approx(0.0, abs=1e-10)

    def test_empty_array_returns_nan(self):
        """Empty array → nan."""
        assert math.isnan(nmad(np.array([])))

    def test_all_nan_returns_nan(self):
        """All-nan input → nan."""
        assert math.isnan(nmad(np.array([np.nan, np.nan, np.nan])))

    def test_ignores_non_finite(self):
        """inf and nan are excluded; finite subset governs the result."""
        data_full = np.array([1.0, 2.0, 3.0, np.nan, np.inf])
        data_finite = np.array([1.0, 2.0, 3.0])
        assert nmad(data_full) == pytest.approx(nmad(data_finite), abs=1e-10)

    def test_custom_nfact(self):
        """nfact=1.0 returns the plain median absolute deviation."""
        data = np.array([1.0, 2.0, 3.0])
        assert nmad(data, nfact=1.0) == pytest.approx(1.0, abs=1e-10)

    def test_2d_array_flattened(self):
        """A 2-D array produces the same result as the flattened 1-D version."""
        data = np.arange(1.0, 10.0)
        assert nmad(data.reshape(3, 3)) == pytest.approx(nmad(data), abs=1e-10)

    def test_gaussian_consistent_with_std(self):
        """Large Gaussian sample: NMAD ≈ std (within 5 %)."""
        rng = np.random.default_rng(42)
        data = rng.normal(0.0, 1.0, 10_000)
        assert nmad(data) == pytest.approx(1.0, rel=0.05)

    def test_returns_float(self):
        """Return type is always float."""
        assert isinstance(nmad(np.array([1.0, 2.0, 3.0])), float)
        assert isinstance(nmad(np.array([])), float)


# ──────────────────────────────────────────────────────────────────────────────


class TestFractionAbove:
    """Tests for fraction_above() — CC-quality metric (share of pixels >= τ)."""

    def test_known_fractions(self):
        """[0.1, 0.3, 0.6, 0.9]: 0.25→0.75, 0.5→0.5, 0.75→0.25."""
        result = fraction_above(np.array([0.1, 0.3, 0.6, 0.9]))
        assert result[0.25] == pytest.approx(0.75)
        assert result[0.5] == pytest.approx(0.5)
        assert result[0.75] == pytest.approx(0.25)

    def test_threshold_is_inclusive(self):
        """A value exactly at the threshold counts (>= is inclusive)."""
        result = fraction_above(np.array([0.5, 0.5, 0.1, 0.1]), thresholds=(0.5,))
        assert result[0.5] == pytest.approx(0.5)

    def test_all_above_returns_one(self):
        """All values above the threshold → fraction 1.0."""
        result = fraction_above(np.array([0.8, 0.9, 1.0]), thresholds=(0.25,))
        assert result[0.25] == pytest.approx(1.0)

    def test_all_below_returns_zero(self):
        """No values reach the threshold → fraction 0.0."""
        result = fraction_above(np.array([0.1, 0.2, 0.24]), thresholds=(0.25,))
        assert result[0.25] == pytest.approx(0.0)

    def test_empty_array_returns_nan_per_threshold(self):
        """Empty array → nan for every threshold, keys still present."""
        result = fraction_above(np.array([]))
        assert set(result.keys()) == {float(t) for t in CC_QUALITY_THRESHOLDS}
        assert all(math.isnan(v) for v in result.values())

    def test_all_nan_returns_nan_per_threshold(self):
        """All-nan input → nan for every threshold."""
        result = fraction_above(np.array([np.nan, np.nan]))
        assert all(math.isnan(v) for v in result.values())

    def test_non_finite_excluded_from_numerator_and_denominator(self):
        """inf/nan are dropped before computing the fraction."""
        # 2 finite values (0.6, 0.9), both >= 0.5 → 1.0 (not 2/4).
        result = fraction_above(
            np.array([0.6, 0.9, np.nan, np.inf]), thresholds=(0.5,)
        )
        assert result[0.5] == pytest.approx(1.0)

    def test_default_thresholds_keys_present(self):
        """Default call returns one key per CC_QUALITY_THRESHOLDS entry."""
        result = fraction_above(np.array([0.3, 0.6]))
        assert set(result.keys()) == {float(t) for t in CC_QUALITY_THRESHOLDS}

    def test_values_are_floats(self):
        """Returned fractions are plain floats."""
        result = fraction_above(np.array([0.3, 0.6]))
        assert all(isinstance(v, float) for v in result.values())

    def test_2d_array_flattened(self):
        """A 2-D array is flattened before computing the fraction."""
        data = np.array([[0.1, 0.6], [0.8, 0.9]])
        assert fraction_above(data, thresholds=(0.5,))[0.5] == pytest.approx(0.75)


# ──────────────────────────────────────────────────────────────────────────────


class TestCountAbove:
    """Tests for count_above() — number of valid pixels >= τ (CC-fraction)."""

    def test_known_counts(self):
        """[0.1, 0.3, 0.6, 0.9]: 0.25→3, 0.5→2, 0.75→1."""
        result = count_above(np.array([0.1, 0.3, 0.6, 0.9]))
        assert result[0.25] == 3
        assert result[0.5] == 2
        assert result[0.75] == 1

    def test_threshold_is_inclusive(self):
        """A value exactly at the threshold counts (>= is inclusive)."""
        result = count_above(np.array([0.5, 0.5, 0.1, 0.1]), thresholds=(0.5,))
        assert result[0.5] == 2

    def test_all_above_returns_n(self):
        """All values above the threshold → count == n."""
        result = count_above(np.array([0.8, 0.9, 1.0]), thresholds=(0.25,))
        assert result[0.25] == 3

    def test_all_below_returns_zero(self):
        """No values reach the threshold → count 0."""
        result = count_above(np.array([0.1, 0.2, 0.24]), thresholds=(0.25,))
        assert result[0.25] == 0

    def test_empty_array_returns_zero_per_threshold(self):
        """Empty array → 0 for every threshold, keys still present."""
        result = count_above(np.array([]))
        assert set(result.keys()) == {float(t) for t in CC_QUALITY_THRESHOLDS}
        assert all(v == 0 for v in result.values())

    def test_all_nan_returns_zero_per_threshold(self):
        """All-nan input → 0 for every threshold."""
        result = count_above(np.array([np.nan, np.nan]))
        assert all(v == 0 for v in result.values())

    def test_non_finite_excluded_from_count(self):
        """inf/nan are dropped before counting."""
        # 2 finite values (0.6, 0.9), both >= 0.5 → count 2.
        result = count_above(np.array([0.6, 0.9, np.nan, np.inf]), thresholds=(0.5,))
        assert result[0.5] == 2

    def test_default_thresholds_keys_present(self):
        """Default call returns one key per CC_QUALITY_THRESHOLDS entry."""
        result = count_above(np.array([0.3, 0.6]))
        assert set(result.keys()) == {float(t) for t in CC_QUALITY_THRESHOLDS}

    def test_values_are_ints(self):
        """Returned counts are plain ints."""
        result = count_above(np.array([0.3, 0.6]))
        assert all(isinstance(v, int) for v in result.values())

    def test_matches_fraction_numerator(self):
        """count_above == fraction_above * n_finite for the same data."""
        data = np.array([0.1, 0.3, 0.6, 0.9])
        counts = count_above(data)
        fracs = fraction_above(data)
        for t in CC_QUALITY_THRESHOLDS:
            assert counts[float(t)] == pytest.approx(fracs[float(t)] * 4)


# ──────────────────────────────────────────────────────────────────────────────


class TestComputeRasterStats:
    """Tests for compute_raster_stats() — GeoTIFF band statistics."""

    def test_returns_none_for_missing_file(self, tmp_path):
        """Non-existent path → None."""
        assert compute_raster_stats(tmp_path / "missing.tif") is None

    def test_returns_expected_keys(self, make_tif):
        """Default call returns all expected stat keys."""
        path = make_tif(np.ones((10, 10), dtype="float32"))
        result = compute_raster_stats(path)
        expected = {
            "count_total", "count_valid", "valid_fraction",
            "mean", "median", "std", "nmad", "min", "max",
            "p5", "p25", "p50", "p75", "p95",
        }
        assert expected.issubset(result.keys())

    def test_count_total_and_valid_all_good(self, make_tif):
        """All valid pixels: count_total == count_valid == number of pixels."""
        result = compute_raster_stats(make_tif(np.ones((5, 8), dtype="float32")))
        assert result["count_total"] == 40
        assert result["count_valid"] == 40

    def test_valid_fraction_with_nodata(self, make_tif):
        """Half pixels nodata → valid_fraction == 0.5, count_valid == 8."""
        data = np.ones((4, 4), dtype="float32")
        data[2:, :] = -9999.0
        result = compute_raster_stats(make_tif(data, nodata=-9999.0))
        assert result["count_total"] == 16
        assert result["count_valid"] == 8
        assert result["valid_fraction"] == pytest.approx(0.5, abs=1e-6)

    def test_mean_and_median_for_uniform_array(self, make_tif):
        """Uniform array: mean == median == the constant value."""
        data = np.full((4, 4), 3.0, dtype="float32")
        result = compute_raster_stats(make_tif(data))
        assert result["mean"] == pytest.approx(3.0, abs=1e-4)
        assert result["median"] == pytest.approx(3.0, abs=1e-4)

    def test_min_and_max_correct(self, make_tif):
        """min and max match numpy's own min/max on the same data."""
        data = np.arange(1, 17, dtype="float32").reshape(4, 4)
        result = compute_raster_stats(make_tif(data))
        assert result["min"] == pytest.approx(1.0, abs=1e-4)
        assert result["max"] == pytest.approx(16.0, abs=1e-4)

    def test_nmad_nonzero_for_non_constant_data(self, make_tif):
        """Non-constant raster: NMAD > 0."""
        data = np.arange(0, 100, dtype="float32").reshape(10, 10)
        result = compute_raster_stats(make_tif(data))
        assert result["nmad"] > 0.0

    def test_default_percentile_keys_present(self, make_tif):
        """Default percentiles produce p5, p25, p50, p75, p95 keys."""
        result = compute_raster_stats(make_tif(np.ones((5, 5), dtype="float32")))
        for p in ("p5", "p25", "p50", "p75", "p95"):
            assert p in result

    def test_custom_percentiles_returned(self, make_tif):
        """Custom percentiles=[10, 90] appear; defaults are absent."""
        result = compute_raster_stats(
            make_tif(np.ones((5, 5), dtype="float32")),
            percentiles=[10, 90],
        )
        assert "p10" in result
        assert "p90" in result
        assert "p5" not in result
        assert "p25" not in result

    def test_all_nodata_returns_nan_stats(self, make_tif):
        """All pixels masked → count_valid=0, all numeric stats are nan."""
        data = np.full((4, 4), -9999.0, dtype="float32")
        result = compute_raster_stats(make_tif(data, nodata=-9999.0))
        assert result["count_valid"] == 0
        assert result["valid_fraction"] == 0.0
        for key in ("mean", "median", "std", "nmad", "min", "max", "p5"):
            assert math.isnan(result[key]), f"Expected nan for '{key}'"

    def test_stats_rounded_to_six_decimal_places(self, make_tif):
        """Float stats are rounded to ≤ 6 decimal places (idempotent under round)."""
        rng = np.random.default_rng(0)
        data = rng.random((8, 8)).astype("float32")
        result = compute_raster_stats(make_tif(data))
        for key in ("mean", "median", "std", "nmad", "min", "max"):
            val = result[key]
            assert round(val, 6) == val, f"'{key}' = {val!r} has more than 6 dp"

    def test_accepts_string_path(self, make_tif):
        """A str path is accepted in addition to pathlib.Path."""
        result = compute_raster_stats(str(make_tif(np.ones((4, 4), dtype="float32"))))
        assert result is not None

    def test_p50_equals_median(self, make_tif):
        """The p50 percentile equals the median for any data."""
        data = np.arange(1, 26, dtype="float32").reshape(5, 5)
        result = compute_raster_stats(make_tif(data))
        assert result["p50"] == pytest.approx(result["median"], abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────


class TestComputeArrayStats:
    """Tests for _compute_array_stats() — in-memory masked-array statistics."""

    def test_known_mean(self):
        """All-2.0 array → mean == 2.0."""
        arr = ma.array(np.full((4, 4), 2.0), mask=False)
        assert _compute_array_stats(arr)["mean"] == pytest.approx(2.0, abs=1e-6)

    def test_fully_masked_returns_nan_stats(self):
        """Fully masked array → count_valid=0, all numeric stats nan."""
        arr = ma.array(np.ones((5, 5)), mask=True)
        result = _compute_array_stats(arr)
        assert result["count_valid"] == 0
        assert math.isnan(result["mean"])
        assert math.isnan(result["p5"])

    def test_partial_mask_counts(self):
        """Partially masked array: count_valid counts unmasked pixels."""
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :] = True  # 8 masked, 8 valid
        arr = ma.array(np.ones((4, 4)), mask=mask)
        result = _compute_array_stats(arr)
        assert result["count_valid"] == 8
        assert result["valid_fraction"] == pytest.approx(0.5, abs=1e-6)

    def test_custom_percentiles(self):
        """percentiles=[10, 50] → p10 and p50 present; p5 absent."""
        arr = ma.array(np.arange(100.0).reshape(10, 10), mask=False)
        result = _compute_array_stats(arr, percentiles=[10, 50])
        assert "p10" in result
        assert "p50" in result
        assert "p5" not in result

    def test_schema_matches_compute_raster_stats(self, make_tif):
        """Key set is identical to compute_raster_stats (default percentiles)."""
        arr = ma.array(np.ones((5, 5)), mask=False)
        from_array = _compute_array_stats(arr)
        from_raster = compute_raster_stats(make_tif(np.ones((5, 5), dtype="float32")))
        assert set(from_array.keys()) == set(from_raster.keys())

    def test_std_zero_for_constant(self):
        """Constant array → std == 0."""
        arr = ma.array(np.full((6, 6), 7.0), mask=False)
        assert _compute_array_stats(arr)["std"] == pytest.approx(0.0, abs=1e-10)


# ──────────────────────────────────────────────────────────────────────────────


class TestBuildPairMetadata:
    """Tests for build_pair_metadata() — identity + spatial metadata dict."""

    _EXPECTED_KEYS = {
        "pair_key", "pair_name", "pzone",
        "left_date", "left_sensor", "right_date", "right_sensor",
        "dt_days", "dt_months", "dt_years", "direction",
        "ncols", "nrows", "resolution_x", "resolution_y", "crs", "bounds",
    }

    def test_contains_all_expected_keys(self, mock_pair):
        """Returned dict contains all expected keys."""
        result = build_pair_metadata(mock_pair)
        assert self._EXPECTED_KEYS.issubset(result.keys())

    def test_identity_fields_match_pair(self, mock_pair):
        """pair_key, pzone, dates, sensors, and dt_* propagated correctly."""
        result = build_pair_metadata(mock_pair)
        assert result["pair_key"] == "testpz_20220601_20220901"
        assert result["pzone"] == "testpz"
        assert result["left_date"] == "2022-06-01"
        assert result["left_sensor"] == "planetscope"
        assert result["right_date"] == "2022-09-01"
        assert result["right_sensor"] == "planetscope"
        assert result["dt_days"] == pytest.approx(92.0)

    def test_spatial_fields_none_when_no_disparity(self, mock_pair):
        """Missing disparity file → all spatial fields are None."""
        result = build_pair_metadata(mock_pair)
        for key in ("ncols", "nrows", "resolution_x", "resolution_y", "crs", "bounds"):
            assert result[key] is None, f"Expected None for '{key}'"

    def test_spatial_fields_populated_when_disparity_exists(self, mock_pair):
        """Existing disparity file → spatial fields are non-None."""
        write_geotiff(mock_pair.pa_disparity_f_path, np.ones((5, 8), dtype="float32"))
        result = build_pair_metadata(mock_pair)
        assert result["ncols"] == 8
        assert result["nrows"] == 5
        assert result["resolution_x"] is not None
        assert result["crs"] is not None

    def test_bounds_is_dict_with_four_keys(self, mock_pair):
        """bounds dict has left, bottom, right, top keys."""
        write_geotiff(mock_pair.pa_disparity_f_path, np.ones((4, 4), dtype="float32"))
        result = build_pair_metadata(mock_pair)
        assert isinstance(result["bounds"], dict)
        assert set(result["bounds"].keys()) == {"left", "bottom", "right", "top"}


# ──────────────────────────────────────────────────────────────────────────────


class TestComputePairRawStats:
    """Tests for compute_pair_raw_stats() — per-raster stats for a pair."""

    def test_empty_when_no_rasters_exist(self, mock_pair):
        """No rasters on disk → empty dict."""
        assert compute_pair_raw_stats(mock_pair) == {}

    def test_includes_existing_rasters_only(self, mock_pair):
        """Only rasters that exist appear in the result."""
        data = np.ones((4, 4), dtype="float32")
        write_geotiff(mock_pair.pa_ew_path, data)
        write_geotiff(mock_pair.pa_ns_path, data)
        result = compute_pair_raw_stats(mock_pair)
        assert "ew" in result
        assert "ns" in result
        assert "cc" not in result
        assert "magn" not in result

    def test_all_four_rasters_present(self, mock_pair):
        """All four rasters on disk → four keys in result."""
        data = np.ones((4, 4), dtype="float32")
        for path in (
            mock_pair.pa_ew_path,
            mock_pair.pa_ns_path,
            mock_pair.pa_cc_raw_path,
            mock_pair.pa_magn_path,
        ):
            write_geotiff(path, data)
        result = compute_pair_raw_stats(mock_pair)
        assert set(result.keys()) == {"ew", "ns", "cc", "magn"}

    def test_values_are_stats_dicts(self, mock_pair):
        """Each value is a stats dict with the standard keys."""
        write_geotiff(mock_pair.pa_ew_path, np.ones((4, 4), dtype="float32"))
        result = compute_pair_raw_stats(mock_pair)
        assert "mean" in result["ew"]
        assert "count_valid" in result["ew"]
        assert "nmad" in result["ew"]

    #: 16 valid pixels: eight at 0.9, four at 0.6, four at 0.1.
    _CC_DATA = np.array(
        [[0.9, 0.9, 0.9, 0.9],
         [0.9, 0.9, 0.9, 0.9],
         [0.6, 0.6, 0.6, 0.6],
         [0.1, 0.1, 0.1, 0.1]],
        dtype="float32",
    )

    def test_cc_entry_has_quality_and_fraction_keys(self, mock_pair):
        """The cc entry carries both cc_quality_gte_* and cc_fraction_gte_* keys."""
        write_geotiff(mock_pair.pa_cc_raw_path, np.ones((4, 4), dtype="float32"))
        result = compute_pair_raw_stats(mock_pair)
        for tag in ("025", "050", "075"):
            assert f"cc_quality_gte_{tag}" in result["cc"]
            assert f"cc_fraction_gte_{tag}" in result["cc"]

    def test_cc_quality_values_match_hand_computed(self, mock_pair):
        """CC-quality fractions equal the hand-computed shares of valid pixels.

          >= 0.25 → 12/16 = 0.75 ; >= 0.5 → 12/16 = 0.75 ; >= 0.75 → 8/16 = 0.5
        """
        write_geotiff(mock_pair.pa_cc_raw_path, self._CC_DATA)
        cc = compute_pair_raw_stats(mock_pair)["cc"]
        assert cc["cc_quality_gte_025"] == pytest.approx(0.75)
        assert cc["cc_quality_gte_050"] == pytest.approx(0.75)
        assert cc["cc_quality_gte_075"] == pytest.approx(0.5)

    def test_cc_fraction_counts_match_hand_computed(self, mock_pair):
        """CC-fraction counts equal the hand-computed # pixels >= τ.

          >= 0.25 → 12 ; >= 0.5 → 12 ; >= 0.75 → 8
        """
        write_geotiff(mock_pair.pa_cc_raw_path, self._CC_DATA)
        cc = compute_pair_raw_stats(mock_pair)["cc"]
        assert cc["cc_fraction_gte_025"] == 12
        assert cc["cc_fraction_gte_050"] == 12
        assert cc["cc_fraction_gte_075"] == 8

    def test_cc_fraction_consistent_with_quality(self, mock_pair):
        """count == round(proportion * count_valid) for each threshold."""
        write_geotiff(mock_pair.pa_cc_raw_path, self._CC_DATA)
        cc = compute_pair_raw_stats(mock_pair)["cc"]
        n = cc["count_valid"]
        for tag in ("025", "050", "075"):
            expected = round(cc[f"cc_quality_gte_{tag}"] * n)
            assert cc[f"cc_fraction_gte_{tag}"] == expected

    def test_cc_keys_absent_from_non_cc_entries(self, mock_pair):
        """ew/ns/magn entries do not carry cc_quality_* or cc_fraction_* keys."""
        data = np.ones((4, 4), dtype="float32")
        for path in (mock_pair.pa_ew_path, mock_pair.pa_ns_path, mock_pair.pa_magn_path):
            write_geotiff(path, data)
        result = compute_pair_raw_stats(mock_pair)
        for entry in ("ew", "ns", "magn"):
            assert not any(
                k.startswith(("cc_quality_gte_", "cc_fraction_gte_"))
                for k in result[entry]
            )

    def test_no_cc_metrics_when_cc_absent(self, mock_pair):
        """Without a CC raster, there is no cc entry (hence no cc metrics)."""
        write_geotiff(mock_pair.pa_ew_path, np.ones((4, 4), dtype="float32"))
        result = compute_pair_raw_stats(mock_pair)
        assert "cc" not in result


# ──────────────────────────────────────────────────────────────────────────────


class TestInitPairStats:
    """Tests for init_pair_stats() — skeleton JSON creation."""

    def test_creates_file_on_disk(self, mock_pair):
        """Stats file is created at pa_stats_path."""
        init_pair_stats(mock_pair)
        assert mock_pair.pa_stats_path.exists()

    def test_skeleton_has_three_sections(self, mock_pair):
        """Returned dict has metadata, raw_corr_stats, corrected_stats."""
        result = init_pair_stats(mock_pair)
        assert set(result.keys()) >= {"metadata", "raw_corr_stats", "corrected_stats"}
        assert result["raw_corr_stats"] == {}
        assert result["corrected_stats"] == {}

    def test_file_content_is_valid_json(self, mock_pair):
        """Written file parses as valid JSON and has the skeleton keys."""
        init_pair_stats(mock_pair)
        with open(mock_pair.pa_stats_path) as f:
            content = json.load(f)
        assert "metadata" in content
        assert "raw_corr_stats" in content

    def test_metadata_contains_pair_key(self, mock_pair):
        """Metadata section contains the pair key."""
        result = init_pair_stats(mock_pair)
        assert result["metadata"]["pair_key"] == mock_pair.pa_key

    def test_returns_dict(self, mock_pair):
        """Return type is dict."""
        assert isinstance(init_pair_stats(mock_pair), dict)


# ──────────────────────────────────────────────────────────────────────────────


class TestLoadPairStats:
    """Tests for load_pair_stats() — reading stats JSON from disk."""

    def test_raises_when_no_file(self, mock_pair):
        """FileNotFoundError raised when stats file does not exist."""
        with pytest.raises(FileNotFoundError, match="Stats file not found"):
            load_pair_stats(mock_pair)

    def test_round_trips_with_init(self, mock_pair):
        """Data written by init_pair_stats is returned correctly."""
        init_pair_stats(mock_pair)
        loaded = load_pair_stats(mock_pair)
        assert loaded["metadata"]["pair_key"] == mock_pair.pa_key
        assert "raw_corr_stats" in loaded

    def test_returns_dict(self, mock_pair):
        """Return type is dict."""
        init_pair_stats(mock_pair)
        assert isinstance(load_pair_stats(mock_pair), dict)


# ──────────────────────────────────────────────────────────────────────────────


class TestUpdatePairStats:
    """Tests for update_pair_stats() — partial JSON update."""

    def test_creates_file_when_missing(self, mock_pair):
        """File is created on first call even without init."""
        update_pair_stats(mock_pair, "raw_corr_stats", {"ew": {}})
        assert mock_pair.pa_stats_path.exists()

    def test_updates_target_section(self, mock_pair):
        """Target section is replaced with supplied data."""
        init_pair_stats(mock_pair)
        update_pair_stats(mock_pair, "raw_corr_stats", {"ew": {"mean": 1.5}})
        assert load_pair_stats(mock_pair)["raw_corr_stats"]["ew"]["mean"] == pytest.approx(1.5)

    def test_preserves_other_sections(self, mock_pair):
        """Updating one section leaves the others intact."""
        init_pair_stats(mock_pair)
        update_pair_stats(mock_pair, "raw_corr_stats", {"ew": {}})
        loaded = load_pair_stats(mock_pair)
        assert "corrected_stats" in loaded
        assert "metadata" in loaded

    def test_returns_full_dict(self, mock_pair):
        """Return value contains all three top-level sections."""
        result = update_pair_stats(mock_pair, "raw_corr_stats", {})
        assert "metadata" in result
        assert "raw_corr_stats" in result
        assert "corrected_stats" in result

    def test_second_call_replaces_section(self, mock_pair):
        """Calling twice with the same section name overwrites it."""
        init_pair_stats(mock_pair)
        update_pair_stats(mock_pair, "raw_corr_stats", {"ew": {"mean": 1.0}})
        update_pair_stats(mock_pair, "raw_corr_stats", {"ew": {"mean": 2.0}})
        assert load_pair_stats(mock_pair)["raw_corr_stats"]["ew"]["mean"] == pytest.approx(2.0)


# ──────────────────────────────────────────────────────────────────────────────


class TestSaveCorrectedStats:
    """Tests for save_corrected_stats() — persist corrected displacement stats."""

    def _raster(self, value: float = 1.0, shape: tuple = (5, 5)) -> MockRasterData:
        return MockRasterData(data=ma.array(np.full(shape, value), mask=False))

    def test_saves_ew_and_ns_under_correction_name(self, mock_pair):
        """Stats saved under correction_name with 'ew' and 'ns' sub-keys."""
        save_corrected_stats(mock_pair, self._raster(1.0), self._raster(2.0), "MedianCentering")
        entry = load_pair_stats(mock_pair)["corrected_stats"]["MedianCentering"]
        assert "ew" in entry
        assert "ns" in entry

    def test_ew_and_ns_means_match_data(self, mock_pair):
        """Stored mean values match the constant raster values."""
        save_corrected_stats(mock_pair, self._raster(3.0), self._raster(7.0), "TestCorr")
        entry = load_pair_stats(mock_pair)["corrected_stats"]["TestCorr"]
        assert entry["ew"]["mean"] == pytest.approx(3.0, abs=1e-4)
        assert entry["ns"]["mean"] == pytest.approx(7.0, abs=1e-4)

    def test_multiple_corrections_accumulate(self, mock_pair):
        """Two calls with different names both appear in corrected_stats."""
        r = self._raster()
        save_corrected_stats(mock_pair, r, r, "MedianCentering")
        save_corrected_stats(mock_pair, r, r, "RampCorrection")
        corrected = load_pair_stats(mock_pair)["corrected_stats"]
        assert "MedianCentering" in corrected
        assert "RampCorrection" in corrected

    def test_second_call_with_same_name_overwrites(self, mock_pair):
        """Two calls with the same correction_name: second overwrites first."""
        save_corrected_stats(mock_pair, self._raster(1.0), self._raster(1.0), "TestCorr")
        save_corrected_stats(mock_pair, self._raster(5.0), self._raster(5.0), "TestCorr")
        entry = load_pair_stats(mock_pair)["corrected_stats"]["TestCorr"]
        assert entry["ew"]["mean"] == pytest.approx(5.0, abs=1e-4)

    def test_custom_percentiles_stored(self, mock_pair):
        """Custom percentiles=[10, 90] appear in the stored stats."""
        save_corrected_stats(
            mock_pair, self._raster(), self._raster(), "TestCorr", percentiles=[10, 90]
        )
        ew = load_pair_stats(mock_pair)["corrected_stats"]["TestCorr"]["ew"]
        assert "p10" in ew
        assert "p90" in ew
        assert "p5" not in ew

    def test_count_valid_nonzero_for_unmasked_data(self, mock_pair):
        """Unmasked raster → count_valid > 0."""
        save_corrected_stats(mock_pair, self._raster(), self._raster(), "TestCorr")
        ew = load_pair_stats(mock_pair)["corrected_stats"]["TestCorr"]["ew"]
        assert ew["count_valid"] == 25  # 5×5 raster, all valid


# ──────────────────────────────────────────────────────────────────────────────


class TestSaveRawCorrStats:
    """Tests for save_raw_corr_stats() — convenience wrapper."""

    def test_updates_raw_corr_stats_section(self, mock_pair):
        """raw_corr_stats section is populated after call."""
        write_geotiff(mock_pair.pa_ew_path, np.ones((4, 4), dtype="float32"))
        save_raw_corr_stats(mock_pair)
        assert "ew" in load_pair_stats(mock_pair)["raw_corr_stats"]

    def test_empty_section_when_no_rasters(self, mock_pair):
        """No rasters on disk → raw_corr_stats is an empty dict."""
        save_raw_corr_stats(mock_pair)
        assert load_pair_stats(mock_pair)["raw_corr_stats"] == {}

    def test_preserves_metadata_and_corrected_stats(self, mock_pair):
        """Updating raw stats does not erase the other two sections."""
        init_pair_stats(mock_pair)
        save_raw_corr_stats(mock_pair)
        loaded = load_pair_stats(mock_pair)
        assert "metadata" in loaded
        assert "corrected_stats" in loaded
