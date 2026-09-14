#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the correlation-plan JSON and the parameter precedence."""
from __future__ import annotations

import json

import pytest

from geomulticorr.correlation._correlation_plan import (
    PLAN_ASSUMPTION_KEYS,
    PLAN_PARAM_KEYS,
    PLAN_SCHEMA_VERSION,
    build_correlation_plan,
    params_differ_from_file,
    parse_params_file,
    plan_path,
    read_correlation_plan,
    resolve_pair_params,
    write_correlation_plan,
)

SCALARS = {
    "corr_algorithm": "asp_bm",
    "corr_kernel": (21, 21),
    "corr_search": None,
    "corr_xthreshold": 10,
    "corr_seed_mode": 1,
    "subpixel_mode": 2,
    "subpixel_kernel": (21, 21),
    "prefilter_mode": 2,
    "cost_mode": 2,
}


class TestBuildPlan:
    def test_carries_a_schema_version_and_timestamp(self):
        plan = build_correlation_plan()
        assert plan["schema_version"] == PLAN_SCHEMA_VERSION
        assert plan["written_utc"]

    def test_is_json_serialisable_without_a_custom_encoder(self):
        plan = build_correlation_plan(
            assumptions={"velocity_m_yr": 2.0, "corr_kernel": (21, 21)},
            groups={"g": {"params": {"corr_search": (-5, -5, 5, 5)}}},
        )
        json.dumps(plan)

    def test_tuples_become_lists(self):
        plan = build_correlation_plan(groups={"g": {"params": {"corr_kernel": (21, 21)}}})
        assert plan["groups"]["g"]["params"]["corr_kernel"] == [21, 21]

    def test_records_what_it_deliberately_omits(self):
        assert build_correlation_plan()["not_recorded_here"]

    def test_relevant_params_are_sorted(self):
        plan = build_correlation_plan(relevant_params=["k", "c_px", "footprint_m"])
        assert plan["relevant_params"] == sorted(plan["relevant_params"])

    def test_assumption_keys_are_documented(self):
        assert "velocity_m_yr" in PLAN_ASSUMPTION_KEYS
        assert "footprint_m" in PLAN_ASSUMPTION_KEYS


class TestWriteRead:
    def test_round_trip(self, tmp_path):
        plan = build_correlation_plan(name="run1", pzone="PZ",
                                      assumptions={"velocity_m_yr": 2.0})
        path = write_correlation_plan(plan_path(tmp_path, "run1"), plan)
        assert path is not None
        back = read_correlation_plan(path)
        assert back["name"] == "run1"
        assert back["assumptions"]["velocity_m_yr"] == 2.0

    def test_plan_path_shape_and_sanitising(self, tmp_path):
        assert plan_path(tmp_path, "run 1/x").name == "correlation_plan_run-1-x.json"
        assert plan_path(tmp_path).name == "correlation_plan_default.json"

    def test_write_creates_missing_directories(self, tmp_path):
        target = tmp_path / "deep" / "deeper" / "correlation_plan_a.json"
        assert write_correlation_plan(target, build_correlation_plan()) is not None
        assert target.exists()

    def test_write_failure_is_logged_never_raised(self, tmp_path):
        """Losing a trace must not cost an otherwise prepared correlation run."""
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        assert write_correlation_plan(blocker / "plan.json", build_correlation_plan()) is None

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_correlation_plan(tmp_path / "nope.json")

    def test_read_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            read_correlation_plan(bad)

    def test_read_rejects_a_non_plan(self, tmp_path):
        other = tmp_path / "other.json"
        other.write_text('{"hello": 1}')
        with pytest.raises(ValueError, match="does not look like a correlation plan"):
            read_correlation_plan(other)

    def test_file_is_not_byte_stable_by_design(self, tmp_path):
        """written_utc makes it a log; no test may assert byte-equality."""
        a = build_correlation_plan()
        b = build_correlation_plan()
        assert "written_utc" in a and "written_utc" in b


class TestResolvePrecedence:
    def test_scalars_pass_through_untouched(self):
        params, source = resolve_pair_params("p", "", SCALARS)
        assert source == "scalar"
        assert params["corr_kernel"] == (21, 21)

    def test_only_known_keys_survive(self):
        params, _ = resolve_pair_params("p", "", {**SCALARS, "nodes": 4, "walltime": "1:00"})
        assert set(params) <= set(PLAN_PARAM_KEYS)

    def test_group_beats_scalars(self):
        params, source = resolve_pair_params(
            "p", "g", SCALARS, groups={"g": {"params": {"corr_kernel": (9, 9)}}}
        )
        assert params["corr_kernel"] == (9, 9)
        assert source == "group"

    def test_override_beats_group(self):
        params, source = resolve_pair_params(
            "p", "g", SCALARS,
            groups={"g": {"params": {"corr_kernel": (9, 9)}}},
            overrides={"p": {"corr_kernel": (31, 31)}},
        )
        assert params["corr_kernel"] == (31, 31)
        assert source == "override"

    def test_group_only_patches_the_keys_it_names(self):
        params, _ = resolve_pair_params(
            "p", "g", SCALARS, groups={"g": {"params": {"corr_kernel": (9, 9)}}}
        )
        assert params["corr_xthreshold"] == 10  # untouched scalar survives

    def test_flat_group_mapping_is_accepted(self):
        """A hand-written plan should not need the extra 'params' nesting."""
        params, source = resolve_pair_params(
            "p", "g", SCALARS, groups={"g": {"corr_kernel": (9, 9)}}
        )
        assert params["corr_kernel"] == (9, 9)
        assert source == "group"

    def test_json_lists_become_tuples(self):
        """ASP writers index positionally; a replayed plan hands back lists."""
        params, _ = resolve_pair_params(
            "p", "g", SCALARS, groups={"g": {"params": {"corr_search": [-5, -5, 5, 5]}}}
        )
        assert params["corr_search"] == (-5, -5, 5, 5)

    def test_unknown_group_falls_back_to_scalars(self):
        params, source = resolve_pair_params("p", "missing", SCALARS, groups={"g": {}})
        assert source == "scalar"
        assert params["corr_kernel"] == (21, 21)


class TestParamsFileComparison:
    def _write(self, asp_helper, path, **kwargs):
        params = dict(corr_algorithm="asp_bm", corr_kernel=(21, 21), corr_search=None,
                      corr_xthreshold=10, corr_seed_mode=1,
                      subpixel_refinement_mode=2, subpixel_kernel=(21, 21),
                      prefilter_mode=2, cost_mode=2)
        params.update(kwargs)
        asp_helper.build_correlation_params(params_file_path=path, **params)
        return path

    def test_parses_what_build_correlation_params_wrote(self, asp_helper, tmp_path):
        path = self._write(asp_helper, tmp_path / "p.txt")
        parsed = parse_params_file(path)
        assert parsed["corr-kernel"] == "21 21"
        assert parsed["stereo-algorithm"] == "asp_bm"
        assert "save-left-right-disparity-difference" in parsed

    def test_missing_file_parses_to_empty(self, tmp_path):
        assert parse_params_file(tmp_path / "nope.txt") == {}

    def test_identical_parameters_do_not_differ(self, asp_helper, tmp_path):
        path = self._write(asp_helper, tmp_path / "p.txt")
        assert params_differ_from_file(path, SCALARS) is False

    def test_a_changed_kernel_is_detected(self, asp_helper, tmp_path):
        """The silent no-op this fix exists for."""
        path = self._write(asp_helper, tmp_path / "p.txt")
        assert params_differ_from_file(path, {**SCALARS, "corr_kernel": (31, 31)}) is True

    def test_a_changed_search_box_is_detected(self, asp_helper, tmp_path):
        path = self._write(asp_helper, tmp_path / "p.txt")
        assert params_differ_from_file(path, {**SCALARS, "corr_search": (-5, -5, 5, 5)}) is True

    def test_dropping_an_explicit_search_is_detected(self, asp_helper, tmp_path):
        path = self._write(asp_helper, tmp_path / "p.txt", corr_search=(-5, -5, 5, 5))
        assert params_differ_from_file(path, SCALARS) is True

    def test_a_missing_file_counts_as_different(self, tmp_path):
        assert params_differ_from_file(tmp_path / "nope.txt", SCALARS) is True

    def test_sgm_override_is_applied_before_comparing(self, asp_helper, tmp_path):
        """Otherwise every asp_mgm run would look changed forever."""
        path = self._write(asp_helper, tmp_path / "p.txt",
                           corr_algorithm="asp_mgm", corr_kernel=(21, 21))
        requested = {**SCALARS, "corr_algorithm": "asp_mgm", "corr_kernel": (21, 21)}
        assert params_differ_from_file(path, requested) is False

    def test_flag_mismatch_is_detected(self, asp_helper, tmp_path):
        path = self._write(asp_helper, tmp_path / "p.txt", save_disparity_difference=True)
        assert params_differ_from_file(path, SCALARS, save_disparity_difference=False) is True
