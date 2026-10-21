#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-module contracts of the correlation-parameter feature.

Every assertion here is an agreement between two files that nothing else would
catch: a view added to one dispatch table and not its twin, a stash key the
commit method does not accept, an assumption the frame builder cannot take.
Each has a failure mode that is silent at import time and only shows up as a
``TypeError`` or a missing figure much later.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("plotly")

from geomulticorr.correlation._correlation_plan import (  # noqa: E402
    PLAN_ASSUMPTION_KEYS,
    PLAN_PARAM_KEYS,
)
from geomulticorr.correlation.correlation import ASP  # noqa: E402
from geomulticorr.core.session import Session, _asp_param_kwargs  # noqa: E402
from geomulticorr.utils._corrparams_frame import (  # noqa: E402
    CORR_MODE_KEYS,
    corrparams_frame_from_pairs,
)
from geomulticorr.utils._corrparams_plotly import VIEW_BUILDERS, VIEW_LABELS  # noqa: E402
from geomulticorr.utils.gmc_functions import CORRPARAMS_MPL_BUILDERS  # noqa: E402


class TestViewTables:
    def test_all_five_view_tables_agree(self):
        """A view in one table and not another renders in one backend only."""
        assert (
            set(VIEW_BUILDERS)
            == set(VIEW_LABELS)
            == set(CORRPARAMS_MPL_BUILDERS)
            == set(Session._CORR_VIEW_CONTROLS)
            == set(Session._CORR_DRAWN_NOUN)
        )

    @pytest.mark.parametrize("view", list(VIEW_BUILDERS))
    def test_each_view_has_a_matplotlib_drawer(self, view):
        from geomulticorr.utils import gmc_functions as gmc_fn

        assert hasattr(gmc_fn, f"_draw_correlation_{view}_on_ax")


class TestStashContract:
    def test_stash_keys_are_accepted_by_prepare_pairs_correlation(self):
        """``prepare_pairs_correlation(**session._last_corr_params)`` must work."""
        accepted = set(inspect.signature(Session.prepare_pairs_correlation).parameters)
        assert set(Session._CORR_PARAM_KEYS) <= accepted

    def test_assumptions_are_not_in_the_splat(self):
        """prepare_pairs_correlation knows nothing about velocities."""
        assert not set(Session._CORR_ASSUMPTION_KEYS) & set(Session._CORR_PARAM_KEYS)

    def test_assumption_keys_match_what_the_plan_records(self):
        """The plan records both halves: the derivation inputs and the uniform
        overrides that replaced their result."""
        assert (
            set(Session._CORR_ASSUMPTION_KEYS) | set(Session._CORR_UNIFORM_KEYS)
            == set(PLAN_ASSUMPTION_KEYS)
        )

    def test_the_two_halves_do_not_overlap(self):
        """Uniform overrides are applied after the derivation, not fed into it."""
        assert not (
            set(Session._CORR_ASSUMPTION_KEYS) & set(Session._CORR_UNIFORM_KEYS)
        )

    def test_every_assumption_is_accepted_by_the_frame_builder(self):
        accepted = set(inspect.signature(corrparams_frame_from_pairs).parameters)
        assert set(Session._CORR_ASSUMPTION_KEYS) <= accepted

    def test_explorer_defaults_cover_every_assumption(self):
        assert set(Session._CORR_ASSUMPTION_KEYS) <= Session._CORRPARAMS_EXPLORER_DEFAULTS

    def test_explorer_defaults_cover_every_uniform_override(self):
        assert set(Session._CORR_UNIFORM_KEYS) <= Session._CORRPARAMS_EXPLORER_DEFAULTS


class TestPlanParamsContract:
    def test_plan_params_splat_into_build_correlation_params(self):
        """The resolved dict goes straight through ``_asp_param_kwargs``."""
        accepted = set(inspect.signature(ASP.build_correlation_params).parameters)
        mapped = set(_asp_param_kwargs({k: None for k in PLAN_PARAM_KEYS}))
        assert mapped <= accepted

    def test_the_only_rename_is_subpixel_mode(self):
        renamed = _asp_param_kwargs({k: None for k in PLAN_PARAM_KEYS})
        assert "subpixel_refinement_mode" in renamed
        assert "subpixel_mode" not in renamed
        assert set(PLAN_PARAM_KEYS) - {"subpixel_mode"} <= set(renamed)


class TestModeKeysContract:
    def test_every_mode_key_is_a_real_assumption(self):
        """A stem fragment naming something the run ignores is a lie."""
        known = set(Session._CORR_ASSUMPTION_KEYS) | set(Session._CORR_UNIFORM_KEYS)
        for algorithm, keys in CORR_MODE_KEYS.items():
            assert keys <= known, algorithm

    def test_every_asp_algorithm_has_an_entry(self):
        from geomulticorr.correlation.corr_params import SGM_ALGORITHMS

        assert set(SGM_ALGORITHMS) <= set(CORR_MODE_KEYS)
        assert "asp_bm" in CORR_MODE_KEYS


class TestSgmOverrideIsAppliedEverywhere:
    def test_correval_and_params_file_use_the_same_kernel(self, asp_helper, tmp_path):
        """A CC map computed over a different window than its disparity is wrong."""
        from geomulticorr.correlation._correlation_plan import parse_params_file

        path = tmp_path / "p.txt"
        asp_helper.build_correlation_params(
            corr_algorithm="asp_mgm", corr_kernel=(21, 21),
            subpixel_kernel=(21, 21), params_file_path=path,
        )
        written = parse_params_file(path)["corr-kernel"]

        cmd = asp_helper._build_correval_cmd(
            str(tmp_path / "pfx"), corr_algorithm="asp_mgm",
            corr_kernel=(21, 21), metric="ncc",
        )
        used = next(c for c in cmd if "kernel-size" in c).split(None, 1)[1]
        assert written == used == "9 9"

    def test_block_matching_keeps_the_requested_kernel(self, asp_helper, tmp_path):
        from geomulticorr.correlation._correlation_plan import parse_params_file

        path = tmp_path / "p.txt"
        asp_helper.build_correlation_params(
            corr_algorithm="asp_bm", corr_kernel=(31, 31), params_file_path=path
        )
        assert parse_params_file(path)["corr-kernel"] == "31 31"
        cmd = asp_helper._build_correval_cmd(
            str(tmp_path / "pfx"), corr_algorithm="asp_bm",
            corr_kernel=(31, 31), metric="ncc",
        )
        assert "--kernel-size 31 31" in cmd
