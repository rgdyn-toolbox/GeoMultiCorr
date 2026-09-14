#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-pair parameter resolution in ``Session.prepare_pairs_correlation``.

Driven with a stub session rather than a real project, following the house
pattern in ``tests/core/test_explore_step_range.py``: the method only needs
``get_pairs`` plus the pairs themselves, so a stub exercises the whole
resolution chain with no geodatabase.
"""
from __future__ import annotations

import pathlib
import types

from unittest.mock import patch

import pytest

from geomulticorr.core.session import Session, _asp_param_kwargs
from geomulticorr.correlation._correlation_plan import (
    build_correlation_plan,
    parse_params_file,
    plan_path,
    write_correlation_plan,
)


class _StubPair:
    """The surface ``prepare_pairs_correlation`` touches on a Pair."""

    def __init__(self, key: str, root: pathlib.Path):
        self.pa_key = key
        self.pa_path = root / key
        self.clipped = False

    def get_status(self):
        return "clipped"

    def check_corr_outputs(self):
        return {"disparity": False}

    def clip(self):
        self.clipped = True


def _stub_session(tmp_path, keys):
    """A minimal object carrying the real method and nothing else."""
    pairs = [_StubPair(k, tmp_path) for k in keys]
    session = types.SimpleNamespace(
        get_pairs=lambda criterias="": list(pairs),
        _corr_params_changed=lambda *a, **kw: Session._corr_params_changed(session, *a, **kw),
        pairs=pairs,
    )
    return session


def _prepare(session, **kwargs):
    """Call the real method against the stub, with ASP's script writer faked."""
    def _fake_write_bash_script(pair, **kw):
        path = pair.pa_path / f"{pair.pa_key}_CorrelationJob.sh"
        path.write_text(f"# {kw.get('corr_kernel')}\n")
        _fake_write_bash_script.calls.append((pair.pa_key, kw))
        return path

    _fake_write_bash_script.calls = []

    with patch("shutil.which", return_value="/fake/parallel_stereo"), patch(
        "geomulticorr.correlation.correlation.ASP.write_bash_script",
        side_effect=_fake_write_bash_script,
    ):
        scripts = Session.prepare_pairs_correlation(session, print_summary=False, **kwargs)
    return scripts, _fake_write_bash_script.calls


def _kernel_of(pair) -> str:
    return parse_params_file(pair.pa_path / f"{pair.pa_key}_CorrParameters.txt")["corr-kernel"]


class TestBackwardCompatibility:
    def test_no_plan_applies_the_scalars_uniformly(self, tmp_path):
        session = _stub_session(tmp_path, ["a", "b"])
        _prepare(session, corr_kernel=(21, 21))
        assert {_kernel_of(p) for p in session.pairs} == {"21 21"}

    def test_every_pair_gets_its_own_files(self, tmp_path):
        session = _stub_session(tmp_path, ["a", "b"])
        scripts, _ = _prepare(session)
        assert len(scripts) == 2
        for pair in session.pairs:
            assert (pair.pa_path / f"{pair.pa_key}_CorrParameters.txt").exists()

    def test_no_pairs_returns_empty(self, tmp_path):
        session = _stub_session(tmp_path, [])
        scripts, _ = _prepare(session)
        assert scripts == []


class TestPrecedence:
    def test_group_parameters_reach_the_params_file(self, tmp_path):
        session = _stub_session(tmp_path, ["short", "long"])
        plan = build_correlation_plan(
            groups={"g_long": {"params": {"corr_kernel": (15, 15)}}},
            pairs=[
                {"pa_key": "short", "group": "g_short"},
                {"pa_key": "long", "group": "g_long"},
            ],
        )
        _prepare(session, corr_kernel=(21, 21), plan=plan)
        by_key = {p.pa_key: p for p in session.pairs}
        assert _kernel_of(by_key["short"]) == "21 21"
        assert _kernel_of(by_key["long"]) == "15 15"

    def test_two_groups_really_differ_on_disk(self, tmp_path):
        """The whole point of the feature, asserted at the file level."""
        session = _stub_session(tmp_path, ["short", "long"])
        plan = build_correlation_plan(
            groups={
                "g_short": {"params": {"corr_search": (-4, -4, 4, 4)}},
                "g_long": {"params": {"corr_search": (-12, -12, 12, 12)}},
            },
            pairs=[
                {"pa_key": "short", "group": "g_short"},
                {"pa_key": "long", "group": "g_long"},
            ],
        )
        _prepare(session, plan=plan)
        by_key = {p.pa_key: p for p in session.pairs}
        searches = {
            k: parse_params_file(p.pa_path / f"{k}_CorrParameters.txt")["corr-search"]
            for k, p in by_key.items()
        }
        assert searches["short"] == "-4 -4 4 4"
        assert searches["long"] == "-12 -12 12 12"

    def test_per_pair_override_wins(self, tmp_path):
        session = _stub_session(tmp_path, ["a"])
        plan = build_correlation_plan(
            groups={"g": {"params": {"corr_kernel": (15, 15)}}},
            pairs=[{"pa_key": "a", "group": "g"}],
        )
        _prepare(session, corr_kernel=(21, 21), plan=plan,
                 params_by_pair={"a": {"corr_kernel": (31, 31)}})
        assert _kernel_of(session.pairs[0]) == "31 31"

    def test_override_alone_needs_no_plan(self, tmp_path):
        session = _stub_session(tmp_path, ["a", "b"])
        _prepare(session, corr_kernel=(21, 21), params_by_pair={"b": {"corr_kernel": (9, 9)}})
        by_key = {p.pa_key: p for p in session.pairs}
        assert _kernel_of(by_key["a"]) == "21 21"
        assert _kernel_of(by_key["b"]) == "9 9"

    def test_plan_from_a_json_file_on_disk(self, tmp_path):
        session = _stub_session(tmp_path, ["a"])
        plan = build_correlation_plan(
            groups={"g": {"params": {"corr_kernel": (15, 15)}}},
            pairs=[{"pa_key": "a", "group": "g"}],
        )
        path = write_correlation_plan(plan_path(tmp_path / "plans", "run1"), plan)
        _prepare(session, plan=path)
        assert _kernel_of(session.pairs[0]) == "15 15"

    def test_corr_eval_kernel_follows_the_resolved_value(self, tmp_path):
        """The script must be built from the same kernel as the params file."""
        session = _stub_session(tmp_path, ["a"])
        plan = build_correlation_plan(
            groups={"g": {"params": {"corr_kernel": (15, 15)}}},
            pairs=[{"pa_key": "a", "group": "g"}],
        )
        _, calls = _prepare(session, corr_kernel=(21, 21), plan=plan)
        assert calls[0][1]["corr_kernel"] == (15, 15)


class TestRegenerationOnChange:
    def test_unchanged_parameters_reuse_the_script(self, tmp_path):
        session = _stub_session(tmp_path, ["a"])
        _prepare(session, corr_kernel=(21, 21))
        _, calls = _prepare(session, corr_kernel=(21, 21))
        assert calls == []  # reused, nothing rewritten
        assert _kernel_of(session.pairs[0]) == "21 21"

    def test_changed_parameters_are_no_longer_a_silent_no_op(self, tmp_path):
        """Before the fix this discarded the new kernel and re-ran the old one."""
        session = _stub_session(tmp_path, ["a"])
        _prepare(session, corr_kernel=(21, 21))
        _, calls = _prepare(session, corr_kernel=(31, 31))
        assert calls  # regenerated
        assert _kernel_of(session.pairs[0]) == "31 31"

    def test_the_change_is_logged(self, tmp_path, caplog_gmc):
        session = _stub_session(tmp_path, ["a"])
        _prepare(session, corr_kernel=(21, 21))
        with caplog_gmc.at_level("INFO"):
            _prepare(session, corr_kernel=(31, 31))
        assert "correlation parameters changed" in caplog_gmc.text

    def test_overwrite_scripts_still_forces_a_rewrite(self, tmp_path):
        session = _stub_session(tmp_path, ["a"])
        _prepare(session, corr_kernel=(21, 21))
        _, calls = _prepare(session, corr_kernel=(21, 21), overwrite_scripts=True)
        assert calls

    def test_a_group_change_alone_triggers_regeneration(self, tmp_path):
        session = _stub_session(tmp_path, ["a"])
        pairs = [{"pa_key": "a", "group": "g"}]
        _prepare(session, plan=build_correlation_plan(
            groups={"g": {"params": {"corr_kernel": (21, 21)}}}, pairs=pairs))
        _, calls = _prepare(session, plan=build_correlation_plan(
            groups={"g": {"params": {"corr_kernel": (15, 15)}}}, pairs=pairs))
        assert calls
        assert _kernel_of(session.pairs[0]) == "15 15"


class TestAspParamKwargs:
    def test_renames_only_subpixel_mode(self):
        out = _asp_param_kwargs({"corr_kernel": (21, 21), "subpixel_mode": 2})
        assert out == {"corr_kernel": (21, 21), "subpixel_refinement_mode": 2}

    def test_passes_everything_else_through(self):
        out = _asp_param_kwargs({"corr_algorithm": "asp_bm", "cost_mode": 2})
        assert out == {"corr_algorithm": "asp_bm", "cost_mode": 2}

    def test_absent_subpixel_mode_is_not_invented(self):
        assert "subpixel_refinement_mode" not in _asp_param_kwargs({"cost_mode": 2})
