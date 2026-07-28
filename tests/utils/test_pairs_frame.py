#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the canonical pairs-frame contract (_pairs_frame)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomulticorr.core.pzone import _strategy_pair_indices
from geomulticorr.utils._pairs_frame import (
    PAIRS_FRAME_COLUMNS,
    STRATEGY_SEMANTICS,
    date_summary,
    empty_pairs_frame,
    format_pairs_summary,
    pairs_frame_from_indices,
    pairs_frame_from_overview,
    pairs_stats,
    unique_couples,
    unique_dates,
)


def _overview_from_indices(dates, index_pairs, sensors=None, pz="PZ1", with_sensors=True):
    """Build the geodatabase-shaped GeoDataFrame equivalent of the same pairs."""
    rows = []
    for i, j in index_pairs:
        left, right = pd.Timestamp(dates[i]), pd.Timestamp(dates[j])
        dt_days = abs((right - left).days)
        row = {
            "pa_pz_name": pz,
            "pa_left_date": dates[i],
            "pa_right_date": dates[j],
            "pa_dt_days": dt_days,
            "pa_dt_years": round(dt_days / 365.25, 4),
            "pa_direction": "forward" if left <= right else "backward",
        }
        if with_sensors and sensors is not None:
            row["pa_left_sensor"] = sensors[i]
            row["pa_right_sensor"] = sensors[j]
        rows.append(row)
    return pd.DataFrame(rows)


class TestEmptyFrame:
    def test_columns_and_dtypes(self):
        frame = empty_pairs_frame()
        assert tuple(frame.columns) == PAIRS_FRAME_COLUMNS
        assert len(frame) == 0
        assert frame["dt_days"].dtype == np.dtype("int64")
        assert frame["t_i"].dtype == np.dtype("datetime64[ns]")


class TestFromIndices:
    def test_exact_columns(self, dates_iso, index_pairs_consecutive):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_consecutive)
        assert tuple(frame.columns) == PAIRS_FRAME_COLUMNS
        assert len(frame) == len(index_pairs_consecutive)

    def test_dt_days_is_absolute_gap(self, dates_iso):
        frame = pairs_frame_from_indices(dates_iso, [(0, 1), (1, 0)])
        expected = abs((pd.Timestamp(dates_iso[1]) - pd.Timestamp(dates_iso[0])).days)
        assert list(frame["dt_days"]) == [expected, expected]

    def test_direction_from_index_order(self, dates_iso, index_pairs_redundancy):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_redundancy)
        n_fwd = sum(1 for i, j in index_pairs_redundancy if i < j)
        n_bwd = len(index_pairs_redundancy) - n_fwd
        assert (frame["direction"] == "forward").sum() == n_fwd
        assert (frame["direction"] == "backward").sum() == n_bwd
        assert n_fwd == n_bwd  # redundancy emits both directions

    def test_mid_dec_between_endpoints(self, dates_iso, index_pairs_consecutive):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_consecutive)
        lo = np.minimum(frame["dec_i"], frame["dec_j"])
        hi = np.maximum(frame["dec_i"], frame["dec_j"])
        assert np.all(frame["mid_dec"] > lo)
        assert np.all(frame["mid_dec"] < hi)

    def test_dt_years_matches_pair_convention(self, dates_iso, index_pairs_consecutive):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_consecutive)
        np.testing.assert_allclose(
            frame["dt_years"], np.round(frame["dt_days"] / 365.25, 4)
        )

    def test_sensors_default_to_blank(self, dates_iso, index_pairs_consecutive):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_consecutive)
        assert set(frame["sensor_i"]) == {""}

    def test_sensors_indexed_per_endpoint(self, dates_iso, sensors):
        frame = pairs_frame_from_indices(dates_iso, [(0, 1)], sensors=sensors)
        assert frame.loc[0, "sensor_i"] == sensors[0]
        assert frame.loc[0, "sensor_j"] == sensors[1]

    def test_pz_and_label(self, dates_iso):
        frame = pairs_frame_from_indices(dates_iso, [(0, 1)], pz="PZ_A")
        assert frame.loc[0, "pz"] == "PZ_A"
        assert frame.loc[0, "label"] == f"PZ_A_{dates_iso[0]}_{dates_iso[1]}"

    def test_empty_index_pairs(self, dates_iso):
        frame = pairs_frame_from_indices(dates_iso, [])
        assert len(frame) == 0
        assert tuple(frame.columns) == PAIRS_FRAME_COLUMNS

    def test_sensor_length_mismatch_raises(self, dates_iso):
        with pytest.raises(ValueError, match="sensors has"):
            pairs_frame_from_indices(dates_iso, [(0, 1)], sensors=["a", "b"])

    def test_out_of_range_index_raises(self, dates_iso):
        with pytest.raises(ValueError, match="beyond the end"):
            pairs_frame_from_indices(dates_iso, [(0, len(dates_iso))])


class TestFromOverview:
    def test_maps_columns(self, dates_iso, sensors, index_pairs_consecutive):
        gdf = _overview_from_indices(dates_iso, index_pairs_consecutive, sensors)
        frame = pairs_frame_from_overview(gdf)
        assert tuple(frame.columns) == PAIRS_FRAME_COLUMNS
        assert len(frame) == len(index_pairs_consecutive)
        assert frame.loc[0, "sensor_i"] == sensors[0]

    def test_empty_input(self):
        assert len(pairs_frame_from_overview(pd.DataFrame())) == 0
        assert len(pairs_frame_from_overview(None)) == 0

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError, match="missing required column"):
            pairs_frame_from_overview(pd.DataFrame({"pa_left_date": ["2020-01-01"]}))

    def test_legacy_layer_without_sensor_columns(self, dates_iso, index_pairs_consecutive):
        gdf = _overview_from_indices(dates_iso, index_pairs_consecutive, with_sensors=False)
        frame = pairs_frame_from_overview(gdf)
        assert set(frame["sensor_i"]) == {""}

    def test_legacy_layer_uses_date_sensor_fallback(
        self, dates_iso, sensors, index_pairs_consecutive
    ):
        gdf = _overview_from_indices(dates_iso, index_pairs_consecutive, with_sensors=False)
        frame = pairs_frame_from_overview(gdf, date_sensor=dict(zip(dates_iso, sensors)))
        assert frame.loc[0, "sensor_i"] == sensors[0]
        assert frame.loc[0, "sensor_j"] == sensors[1]

    def test_direction_recomputed_not_trusted(self, dates_iso):
        """direction is derived from the dates, so a stale column cannot mislead a plot."""
        gdf = _overview_from_indices(dates_iso, [(1, 0)], with_sensors=False)
        gdf.loc[0, "pa_direction"] = "forward"  # wrong on purpose
        assert pairs_frame_from_overview(gdf).loc[0, "direction"] == "backward"


class TestCandidateCommittedEquivalence:
    """The guarantee that candidate and committed pairs render identically."""

    def test_frames_match(self, dates_iso, sensors, index_pairs_redundancy):
        from_idx = pairs_frame_from_indices(
            dates_iso, index_pairs_redundancy, sensors=sensors, pz="PZ1"
        )
        from_gdf = pairs_frame_from_overview(
            _overview_from_indices(dates_iso, index_pairs_redundancy, sensors, pz="PZ1")
        )
        pd.testing.assert_frame_equal(from_idx, from_gdf)


class TestUniqueDates:
    def test_sorted_union(self, dates_iso, index_pairs_redundancy):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_redundancy)
        dates = unique_dates(frame)
        assert list(dates) == sorted(pd.Timestamp(d) for d in dates_iso)

    def test_empty(self):
        assert len(unique_dates(empty_pairs_frame())) == 0


class TestUniqueCouples:
    def test_halves_redundancy(self, dates_iso, index_pairs_redundancy):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_redundancy)
        assert len(unique_couples(frame)) == len(frame) // 2

    def test_leaves_consecutive_untouched(self, dates_iso, index_pairs_consecutive):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_consecutive)
        pd.testing.assert_frame_equal(unique_couples(frame), frame)

    def test_keeps_first_occurrence(self, dates_iso):
        frame = pairs_frame_from_indices(dates_iso, [(1, 0), (0, 1)])
        kept = unique_couples(frame)
        assert len(kept) == 1
        assert kept.loc[0, "direction"] == "backward"  # the first row survives

    def test_empty(self):
        assert len(unique_couples(empty_pairs_frame())) == 0


class TestPairsStats:
    def test_redundancy_is_bidirectional(self, dates_iso, sensors, index_pairs_redundancy):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_redundancy, sensors=sensors)
        s = pairs_stats(frame, strategy="redundancy")
        assert s["n_pairs"] == 2 * s["n_couples"]
        assert s["n_reversed"] == s["n_couples"]
        assert s["n_forward"] == s["n_backward"]
        assert s["bidirectional"] is True

    def test_consecutive_is_forward_only(self, dates_iso, index_pairs_consecutive):
        s = pairs_stats(pairs_frame_from_indices(dates_iso, index_pairs_consecutive),
                        strategy="consecutive")
        assert s["n_reversed"] == 0
        assert s["n_backward"] == 0
        assert s["bidirectional"] is False

    def test_sensor_dates_count_acquisitions_not_endpoints(
        self, dates_iso, sensors, index_pairs_redundancy
    ):
        """Regression test for the '250 pairs but it says 500' confusion.

        Counting pair endpoints gives 2 * n_pairs; counting acquisitions gives
        n_dates, which is the only number that can be read unambiguously.
        """
        frame = pairs_frame_from_indices(dates_iso, index_pairs_redundancy, sensors=sensors)
        s = pairs_stats(frame)
        assert sum(s["sensor_dates"].values()) == s["n_dates"] == len(dates_iso)
        assert sum(s["sensor_dates"].values()) != 2 * s["n_pairs"]

    def test_dt_stats(self, dates_iso, index_pairs_consecutive):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_consecutive)
        s = pairs_stats(frame)
        assert s["dt_min"] == frame["dt_days"].min()
        assert s["dt_max"] == frame["dt_days"].max()
        assert s["dt_min"] <= s["dt_median"] <= s["dt_max"]

    def test_pzone_count(self, dates_iso, index_pairs_consecutive):
        a = pairs_frame_from_indices(dates_iso, index_pairs_consecutive, pz="A")
        b = pairs_frame_from_indices(dates_iso, index_pairs_consecutive, pz="B")
        assert pairs_stats(pd.concat([a, b], ignore_index=True))["n_pzones"] == 2

    def test_empty_frame_all_zero(self):
        s = pairs_stats(empty_pairs_frame())
        assert s["n_pairs"] == s["n_couples"] == s["n_dates"] == 0
        assert s["sensor_dates"] == {}
        assert s["bidirectional"] is None


class TestSummaryString:
    def test_bidirectional_wording(self, dates_iso, sensors, index_pairs_redundancy):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_redundancy, sensors=sensors)
        s = pairs_stats(frame, strategy="redundancy")
        out = format_pairs_summary(s)
        assert f"{s['n_pairs']} pairs" in out
        assert f"{s['n_couples']} unique couples × 2 directions" in out

    def test_forward_only_wording(self, dates_iso, index_pairs_consecutive):
        out = format_pairs_summary(
            pairs_stats(pairs_frame_from_indices(dates_iso, index_pairs_consecutive),
                        strategy="consecutive")
        )
        assert "forward only" in out
        assert "× 2 directions" not in out

    def test_drawn_line_only_when_it_differs(self, dates_iso, index_pairs_redundancy):
        s = pairs_stats(pairs_frame_from_indices(dates_iso, index_pairs_redundancy))
        assert "chords drawn" not in format_pairs_summary(s, drawn=s["n_pairs"])
        shown = format_pairs_summary(s, drawn=s["n_couples"], drawn_label="chords drawn")
        assert f"{s['n_couples']} chords drawn" in shown

    def test_acquisitions_per_sensor_line(self, dates_iso, sensors, index_pairs_consecutive):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_consecutive, sensors=sensors)
        out = format_pairs_summary(pairs_stats(frame))
        assert f"{len(dates_iso)} acquisition dates" in out
        assert "acquisitions per sensor" in out

    def test_strategy_semantics_appended(self, dates_iso, index_pairs_redundancy):
        out = format_pairs_summary(
            pairs_stats(pairs_frame_from_indices(dates_iso, index_pairs_redundancy),
                        strategy="redundancy")
        )
        assert STRATEGY_SEMANTICS["redundancy"] in out

    def test_unknown_strategy_omits_semantics(self, dates_iso, index_pairs_consecutive):
        out = format_pairs_summary(
            pairs_stats(pairs_frame_from_indices(dates_iso, index_pairs_consecutive),
                        strategy="mystery")
        )
        assert "mystery —" not in out

    def test_plain_text_has_no_markup(self, dates_iso, sensors, index_pairs_redundancy):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_redundancy, sensors=sensors)
        out = format_pairs_summary(pairs_stats(frame, strategy="redundancy"), html=False)
        assert "<" not in out
        assert "\n" in out

    def test_empty_frame_message(self):
        out = format_pairs_summary(pairs_stats(empty_pairs_frame()))
        assert "0 pairs" in out
        assert "no pair matches" in out


class TestDateSummary:
    def test_counts_sum_to_twice_pairs(self, dates_iso, index_pairs_redundancy):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_redundancy)
        summary = date_summary(frame)
        assert summary["n_pairs"].sum() == 2 * len(frame)

    def test_one_row_per_date_sorted(self, dates_iso, index_pairs_consecutive, sensors):
        frame = pairs_frame_from_indices(dates_iso, index_pairs_consecutive, sensors=sensors)
        summary = date_summary(frame)
        assert list(summary["date"]) == sorted(dates_iso)
        assert list(summary["sensor"]) == sensors

    def test_empty(self):
        summary = date_summary(empty_pairs_frame())
        assert len(summary) == 0
        assert list(summary.columns) == ["date", "t", "sensor", "n_pairs"]
