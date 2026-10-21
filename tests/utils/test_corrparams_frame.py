#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the correlation-parameters frame contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomulticorr.correlation._correlation_plan import build_correlation_plan
from geomulticorr.correlation.corr_params import DAYS_PER_YEAR
from geomulticorr.utils._corrparams_frame import (
    CORR_MODE_KEYS,
    CORRPARAMS_FRAME_COLUMNS,
    SOURCE_ORDER,
    corrparams_frame_from_pairs,
    corrparams_frame_from_plan,
    corrparams_stats,
    empty_corrparams_frame,
    format_corrparams_summary,
    group_key,
    group_table,
    relevant_corr_keys,
)

KEYS = ["p_short", "p_mid", "p_long"]
DTS = [30.0, 400.0, 2000.0]
RES = [3.0, 3.0, 3.0]


def _frame(**kwargs):
    params = dict(
        pa_keys=KEYS, dt_days=DTS, resolution_m=RES,
        pz="PZ", sensor_i=["ps"] * 3, sensor_j=["ps"] * 3,
        velocity_m_yr=2.0,
    )
    params.update(kwargs)
    return corrparams_frame_from_pairs(**params)


class TestFrameContract:
    def test_columns_and_order(self):
        assert tuple(_frame().columns) == CORRPARAMS_FRAME_COLUMNS

    def test_empty_frame_has_the_same_shape(self):
        empty = empty_corrparams_frame()
        assert tuple(empty.columns) == CORRPARAMS_FRAME_COLUMNS
        assert len(empty) == 0

    def test_dtypes_are_pinned(self):
        frame = _frame()
        empty = empty_corrparams_frame()
        for column in CORRPARAMS_FRAME_COLUMNS:
            assert frame[column].dtype == empty[column].dtype, column

    def test_no_pairs_gives_the_empty_frame(self):
        out = corrparams_frame_from_pairs([], [], [])
        assert len(out) == 0
        assert tuple(out.columns) == CORRPARAMS_FRAME_COLUMNS

    def test_dt_years_matches_pair_pa_dt_years(self):
        """Derived here, never trusted from the caller — must agree with Pair."""
        frame = _frame()
        np.testing.assert_allclose(
            frame["dt_years"].to_numpy(),
            np.round(np.asarray(DTS) / DAYS_PER_YEAR, 4),
        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            corrparams_frame_from_pairs(KEYS, DTS[:2], RES)

    def test_sensor_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="sensor_i must have length"):
            corrparams_frame_from_pairs(KEYS, DTS, RES, sensor_i=["ps"])

    def test_pz_broadcasts_from_a_bare_string(self):
        assert set(_frame(pz="Chimborazo")["pz"]) == {"Chimborazo"}


class TestDerivation:
    def test_longer_baselines_need_a_larger_search(self):
        frame = _frame().set_index("pa_key")
        assert frame.loc["p_long", "search_px"] > frame.loc["p_short", "search_px"]

    def test_snr_grows_with_baseline(self):
        frame = _frame().set_index("pa_key")
        assert frame.loc["p_long", "snr"] > frame.loc["p_mid", "snr"] > frame.loc["p_short", "snr"]

    def test_footprint_is_kernel_times_resolution(self):
        frame = _frame()
        np.testing.assert_allclose(
            frame["footprint_m"].to_numpy(),
            frame["kernel_px"].to_numpy() * frame["resolution_m"].to_numpy(),
        )

    def test_short_baseline_is_flagged_below_the_floor(self):
        frame = _frame().set_index("pa_key")
        assert "detection floor" in frame.loc["p_short", "flags"]
        assert frame.loc["p_mid", "flags"] == ""

    def test_matched_footprint_gives_different_kernels_per_sensor(self):
        """The multi-sensor point: same metres, different pixel counts."""
        frame = corrparams_frame_from_pairs(
            ["spot", "planet"], [400.0, 400.0], [1.5, 3.0],
            sensor_i=["spot", "ps"], footprint_m=60.0, velocity_m_yr=2.0,
        ).set_index("pa_key")
        assert frame.loc["spot", "kernel_px"] == 41
        assert frame.loc["planet", "kernel_px"] == 21
        np.testing.assert_allclose(frame["footprint_m"].to_numpy(), [61.5, 63.0])

    def test_sensor_and_baseline_both_split_groups(self):
        frame = corrparams_frame_from_pairs(
            ["a", "b", "c"], [400.0, 400.0, 2000.0], [1.5, 3.0, 3.0],
            sensor_i=["spot", "ps", "ps"],
        )
        assert frame["group"].nunique() == 3

    def test_group_key_shape(self):
        assert group_key("ps", "dt30-90d") == "ps|dt30-90d"
        assert group_key("", "") == "unknown|all"


class TestPrecedence:
    def test_auto_is_the_default_source(self):
        assert set(_frame()["source"]) == {"auto"}

    def test_group_params_win_over_the_derivation(self):
        frame = _frame()
        group = frame.set_index("pa_key").loc["p_mid", "group"]
        patched = _frame(group_params={group: {"corr_kernel": (9, 9)}}).set_index("pa_key")
        assert patched.loc["p_mid", "kernel_px"] == 9
        assert patched.loc["p_mid", "source"] == "group"
        assert patched.loc["p_long", "source"] == "auto"

    def test_per_pair_override_wins_over_the_group(self):
        frame = _frame()
        group = frame.set_index("pa_key").loc["p_mid", "group"]
        patched = _frame(
            group_params={group: {"corr_kernel": (9, 9)}},
            params_by_pair={"p_mid": {"corr_kernel": (31, 31)}},
        ).set_index("pa_key")
        assert patched.loc["p_mid", "kernel_px"] == 31
        assert patched.loc["p_mid", "source"] == "override"

    def test_source_order_is_documented(self):
        assert SOURCE_ORDER == ("auto", "group", "override")

    def test_override_search_updates_the_half_width(self):
        patched = _frame(
            params_by_pair={"p_mid": {"corr_search": (-40, -4, 40, 4)}}
        ).set_index("pa_key")
        assert patched.loc["p_mid", "search_px"] == 40.0


class TestPlanRoundTrip:
    def test_derived_and_replayed_frames_are_identical(self):
        """The contract that makes a plan a trace, not a summary."""
        derived = _frame()
        plan = build_correlation_plan(
            name="t", pzone="PZ",
            assumptions={"velocity_m_yr": 2.0},
            pairs=[
                {"pa_key": k, "dt_days": d, "resolution_m": r,
                 "pz": "PZ", "sensor_i": "ps", "sensor_j": "ps"}
                for k, d, r in zip(KEYS, DTS, RES)
            ],
        )
        replayed = corrparams_frame_from_plan(plan)
        pd.testing.assert_frame_equal(derived, replayed)

    def test_group_edits_survive_the_round_trip(self):
        derived = _frame()
        group = derived.set_index("pa_key").loc["p_mid", "group"]
        derived = _frame(group_params={group: {"corr_kernel": (9, 9)}})
        plan = build_correlation_plan(
            name="t", pzone="PZ",
            assumptions={"velocity_m_yr": 2.0},
            groups={group: {"params": {"corr_kernel": (9, 9)}}},
            pairs=[
                {"pa_key": k, "dt_days": d, "resolution_m": r,
                 "pz": "PZ", "sensor_i": "ps", "sensor_j": "ps"}
                for k, d, r in zip(KEYS, DTS, RES)
            ],
        )
        pd.testing.assert_frame_equal(derived, corrparams_frame_from_plan(plan))

    def test_empty_plan_gives_the_empty_frame(self):
        assert len(corrparams_frame_from_plan({})) == 0

    def test_unknown_assumption_keys_are_ignored(self):
        """An old plan must not blow up inside suggest_parameters."""
        plan = build_correlation_plan(
            name="t",
            assumptions={"velocity_m_yr": 2.0, "some_future_knob": 7},
            pairs=[{"pa_key": "a", "dt_days": 400, "resolution_m": 3.0}],
        )
        assert len(corrparams_frame_from_plan(plan)) == 1


class TestGroupTable:
    def test_one_row_per_group(self):
        frame = _frame()
        table = group_table(frame)
        assert len(table) == frame["group"].nunique()
        assert table["n_pairs"].sum() == len(frame)

    def test_group_takes_the_widest_reaching_pair(self):
        """A box sized for the median would rail on the longest baseline."""
        frame = corrparams_frame_from_pairs(
            ["a", "b"], [400.0, 700.0], [3.0, 3.0],
            sensor_i=["ps", "ps"], velocity_m_yr=5.0, dt_bin_ratio=10.0,
        )
        assert frame["group"].nunique() == 1
        table = group_table(frame)
        assert table.loc[0, "search_px"] == frame["search_px"].max()

    def test_empty_frame_gives_an_empty_table_not_an_error(self):
        table = group_table(empty_corrparams_frame())
        assert len(table) == 0
        assert "group" in table.columns

    def test_counts_flagged_pairs(self):
        assert group_table(_frame())["n_flagged"].sum() == 1


class TestStatsAndSummary:
    def test_stats_are_json_safe_scalars(self):
        import json

        json.dumps(corrparams_stats(_frame()))

    def test_counts_pairs_below_the_floor(self):
        assert corrparams_stats(_frame(), k=3.0)["n_below_floor"] == 1

    def test_empty_stats_do_not_raise(self):
        stats = corrparams_stats(empty_corrparams_frame())
        assert stats["n_pairs"] == 0
        assert stats["dt_min_days"] is None

    def test_summary_mentions_the_detection_floor(self):
        text = format_corrparams_summary(corrparams_stats(_frame()), html=False)
        assert "below detection floor" in text
        assert "3 pairs" in text

    def test_summary_handles_the_empty_frame(self):
        assert "no pairs" in format_corrparams_summary(
            corrparams_stats(empty_corrparams_frame()), html=False
        )

    def test_html_and_plain_agree_on_content(self):
        stats = corrparams_stats(_frame())
        assert "<b>" in format_corrparams_summary(stats, html=True)
        assert "<b>" not in format_corrparams_summary(stats, html=False)


class TestRelevantKeys:
    def test_block_matching_lists_footprint(self):
        assert "footprint_m" in relevant_corr_keys("asp_bm")

    def test_sgm_omits_the_inert_footprint(self):
        """SGM forces 9x9, so a footprint number in the stem changed nothing."""
        assert "footprint_m" not in relevant_corr_keys("asp_mgm")

    def test_optional_refinements_dropped_when_unset(self):
        keys = relevant_corr_keys("asp_bm")
        assert "flow_azimuth_deg" not in keys
        assert "strain_rate_per_yr" not in keys

    def test_optional_refinements_kept_when_set(self):
        keys = relevant_corr_keys("asp_bm", flow_azimuth_deg=90.0, strain_rate_per_yr=0.02)
        assert {"flow_azimuth_deg", "strain_rate_per_yr"} <= keys

    def test_returns_a_new_set_callers_may_mutate(self):
        keys = relevant_corr_keys("asp_bm")
        keys.add("scratch")
        assert "scratch" not in CORR_MODE_KEYS["asp_bm"]

    def test_unknown_algorithm_falls_back_to_block_matching(self):
        assert relevant_corr_keys("something_else") == relevant_corr_keys("asp_bm")


class TestPatchedRowsStayConsistent:
    """A patch changes the kernel and the box, so everything derived from them
    has to follow — a stale cost is what someone reads before committing a
    night of compute."""

    def test_cost_follows_a_patched_kernel(self):
        base = _frame().set_index("pa_key").loc["p_mid", "cost_index"]
        patched = _frame(
            params_by_pair={"p_mid": {"corr_kernel": (61, 61)}}
        ).set_index("pa_key").loc["p_mid", "cost_index"]
        assert patched > base * 5

    def test_cost_follows_a_patched_search_box(self):
        base = _frame().set_index("pa_key").loc["p_mid", "cost_index"]
        patched = _frame(
            params_by_pair={"p_mid": {"corr_search": (-80, -80, 80, 80)}}
        ).set_index("pa_key").loc["p_mid", "cost_index"]
        assert patched > base * 5

    def test_cost_matches_the_reported_kernel_and_search(self):
        from geomulticorr.correlation.corr_params import cost_index

        row = _frame(
            params_by_pair={"p_mid": {"corr_kernel": (41, 41),
                                      "corr_search": (-20, -20, 20, 20)}}
        ).set_index("pa_key").loc["p_mid"]
        assert row["cost_index"] == pytest.approx(
            cost_index((41, 41), (-20, -20, 20, 20), "asp_bm")
        )

    def test_unpatched_rows_keep_the_derived_cost(self):
        frame = _frame(params_by_pair={"p_mid": {"corr_kernel": (61, 61)}})
        plain = _frame()
        assert frame.set_index("pa_key").loc["p_long", "cost_index"] == pytest.approx(
            plain.set_index("pa_key").loc["p_long", "cost_index"]
        )


class TestGroupExtremesGoOppositeWays:
    """The search box must REACH the farthest pair; the kernel is a CEILING the
    most constrained pair sets. Getting these the same way round writes a kernel
    past a pair's strain limit — the failure the ceiling exists to prevent."""

    def _mixed_group(self):
        # One bin, two baselines: the longer one has a tighter strain ceiling
        # and needs a longer reach.
        return corrparams_frame_from_pairs(
            ["short", "long"], [300.0, 900.0], [3.0, 3.0],
            sensor_i=["ps", "ps"], velocity_m_yr=5.0,
            strain_rate_per_yr=0.05, dt_bin_ratio=10.0,
        )

    def test_the_two_pairs_really_share_a_group(self):
        assert self._mixed_group()["group"].nunique() == 1

    def test_group_kernel_is_the_minimum_over_its_pairs(self):
        frame = self._mixed_group()
        assert group_table(frame).loc[0, "kernel_px"] == frame["kernel_px"].min()

    def test_group_kernel_never_exceeds_any_pairs_strain_ceiling(self):
        from geomulticorr.correlation.corr_params import strain_kernel_limit_px

        frame = self._mixed_group()
        chosen = group_table(frame).loc[0, "kernel_px"]
        for row in frame.itertuples():
            limit = strain_kernel_limit_px(0.05, row.dt_days)
            assert chosen <= max(limit, 7)  # 7 = the texture floor

    def test_group_search_is_the_maximum_over_its_pairs(self):
        frame = self._mixed_group()
        assert group_table(frame).loc[0, "search_px"] == frame["search_px"].max()

    def test_group_footprint_follows_the_group_kernel(self):
        frame = self._mixed_group()
        table = group_table(frame)
        assert table.loc[0, "footprint_m"] == pytest.approx(
            table.loc[0, "kernel_px"] * table.loc[0, "resolution_m"]
        )


class TestPatchShapes:
    """A plan nests group parameters under ``"params"``; a hand-written mapping
    may be flat. Reading only the flat shape marked rows as patched without
    changing any value — a reduction reported but not applied."""

    def test_nested_plan_shape_is_applied(self):
        frame = _frame()
        group = frame.set_index("pa_key").loc["p_mid", "group"]
        patched = _frame(
            group_params={group: {"params": {"corr_kernel": (9, 9)}, "n_pairs": 1}}
        ).set_index("pa_key")
        assert patched.loc["p_mid", "kernel_px"] == 9
        assert patched.loc["p_mid", "source"] == "group"

    def test_flat_shape_is_applied(self):
        frame = _frame()
        group = frame.set_index("pa_key").loc["p_mid", "group"]
        patched = _frame(group_params={group: {"corr_kernel": (9, 9)}}).set_index("pa_key")
        assert patched.loc["p_mid", "kernel_px"] == 9

    def test_both_shapes_agree(self):
        frame = _frame()
        group = frame.set_index("pa_key").loc["p_mid", "group"]
        nested = _frame(group_params={group: {"params": {"corr_kernel": (9, 9)}}})
        flat = _frame(group_params={group: {"corr_kernel": (9, 9)}})
        pd.testing.assert_frame_equal(nested, flat)

    def test_source_is_never_set_without_a_real_change(self):
        """The exact failure: 'group' reported, nothing applied."""
        frame = _frame()
        group = frame.set_index("pa_key").loc["p_mid", "group"]
        patched = _frame(
            group_params={group: {"params": {"corr_kernel": (9, 9)}}}
        ).set_index("pa_key")
        assert patched.loc["p_mid", "kernel_px"] != _frame().set_index(
            "pa_key"
        ).loc["p_mid", "kernel_px"]
