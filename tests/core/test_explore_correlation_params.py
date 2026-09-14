#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless and widget behaviour of ``Session.explore_correlation_params``.

Driven with a stub session, as ``test_explore_step_range.py`` does: the explorer
needs only ``get_pairs_overview()``, the Thumbs layer and the class tables, so
the whole widget tree is exercisable with no project, no geodatabase and no
display.

That it reads *tables* is the point, not an implementation detail: constructing
``Pair`` objects builds two ``Thumb``s per pair, and each ``Thumb.__init__``
computes a densified ~20 000-vertex footprint polygon costing ~150 ms. The stub's
``get_pairs()`` raises so that path cannot come back unnoticed.
"""
from __future__ import annotations

import contextlib
import io

import pandas as pd
import pytest

pytest.importorskip("plotly")
pytest.importorskip("ipywidgets")

from geomulticorr.core.session import Session  # noqa: E402
from geomulticorr.utils._corrparams_frame import (  # noqa: E402
    CORRPARAMS_FRAME_COLUMNS,
    corrparams_frame_from_plan,
)


#: Resolutions the stub archive's two sensors were sieved to.
_SENSOR_RES = {"planetscope": 3.0, "spot": 1.5}


class _StubSession:
    """Exposes just what the correlation explorer touches.

    The explorer reads **tables** — ``get_pairs_overview()`` and the Thumbs
    layer — and constructs no ``Pair`` and no ``Thumb``, so the stub supplies
    frames rather than objects. ``resolution_reads`` counts header reads, which
    must stay at zero once ``th_res`` is populated.
    """

    _CORR_VIEW_CONTROLS = Session._CORR_VIEW_CONTROLS
    _CORR_DRAWN_NOUN = Session._CORR_DRAWN_NOUN
    _CORRPARAMS_EXPLORER_DEFAULTS = Session._CORRPARAMS_EXPLORER_DEFAULTS
    _CORR_PARAM_KEYS = Session._CORR_PARAM_KEYS
    _CORR_ASSUMPTION_KEYS = Session._CORR_ASSUMPTION_KEYS
    _CORR_UNIFORM_KEYS = Session._CORR_UNIFORM_KEYS

    explore_correlation_params = Session.explore_correlation_params
    _corrparams_pair_facts = Session._corrparams_pair_facts
    _corrparams_frame = Session._corrparams_frame
    _thumb_resolution_lookup = Session._thumb_resolution_lookup
    # staticmethod, or assigning the plain function here would re-bind it and
    # pass `self` as the lookup dict
    _resolution_for = staticmethod(Session._resolution_for)

    def __init__(self, pairs_df, thumbs_df):
        self._pairs_df = pairs_df
        self._thumbs = thumbs_df
        self.resolution_reads = 0
        self.saved = []

    def get_pairs_overview(self, criterias=""):
        return self._pairs_df.copy()

    def get_pairs(self, criterias=""):
        raise AssertionError(
            "the explorer must not construct Pair objects — that is the 140 s path"
        )

    def _thumb_resolution(self, path):
        self.resolution_reads += 1
        return _SENSOR_RES[str(path).split("/")[-1].split("_")[0]]

    def pz_dir(self, pz_name, kind):
        raise ValueError("stub session has no project on disk")

    def save_correlation_figure(self, **kwargs):
        self.saved.append(kwargs)
        return {"png": __import__("pathlib").Path("/tmp/x.png")}


def _stub_frames(specs, pz="PZ", with_res=True):
    """Build the Pairs and Thumbs frames for *specs* = [(key, sensor, dt), ...]."""
    pairs, thumbs = [], {}
    for key, sensor, dt in specs:
        # One acquisition date per (sensor, dt) so the join has something to hit;
        # the derivation only reads dt_days, so the dates need only be distinct.
        left_date, right_date = "2020-01-01", f"2020-01-{(dt % 27) + 1:02d}"
        pairs.append({
            "pa_pz_name": pz,
            "pa_path": f"/proj/{pz}/image_correlation/{key}",
            "pa_left_date": left_date, "pa_left_sensor": sensor,
            "pa_right_date": right_date, "pa_right_sensor": sensor,
            "pa_dt_days": dt,
        })
        for date in (left_date, right_date):
            thumbs[(pz, date, sensor)] = {
                "th_pz_name": pz, "th_date": date, "th_sensor": sensor,
                "th_path": f"/data/{sensor}_{date}.tif",
                "th_res": _SENSOR_RES[sensor] if with_res else float("nan"),
            }
    return pd.DataFrame(pairs), pd.DataFrame(list(thumbs.values()))


def _session(n_planet=4, n_spot=3, with_res=True):
    specs = [(f"ps_{dt}", "planetscope", dt) for dt in [30, 200, 900, 2500][:n_planet]]
    specs += [(f"sp_{dt}", "spot", dt) for dt in [60, 400, 1800][:n_spot]]
    return _StubSession(*_stub_frames(specs, with_res=with_res))


def _headless(session, **kwargs):
    params = dict(interactive=False, savefig=False, write_plan=False)
    params.update(kwargs)
    return session.explore_correlation_params(**params)


def _widget(box, description):
    """Find a control by its description — positional indices break on relayout."""
    found = [
        w for row in box.children
        for w in getattr(row, "children", ())
        if getattr(w, "description", "").startswith(description)
    ]
    assert found, f"no control described {description!r}"
    return found[0]


class TestHeadless:
    def test_returns_a_frame_and_a_figure(self):
        frame, fig = _headless(_session())
        assert tuple(frame.columns) == CORRPARAMS_FRAME_COLUMNS
        assert len(frame) == 7
        assert fig is not None

    def test_imports_no_ipywidgets(self, monkeypatch):
        """The whole point of interactive=False: scripts and batch jobs."""
        import builtins

        real_import = builtins.__import__

        def _guard(name, *args, **kwargs):
            if name.startswith("ipywidgets"):
                raise AssertionError("headless path must not import ipywidgets")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _guard)
        _headless(_session())

    def test_reads_no_raster_headers_when_th_res_is_populated(self):
        """The whole load-time fix: resolution comes from the Thumbs table."""
        session = _session()
        _headless(session)
        assert session.resolution_reads == 0

    def test_constructs_no_pair_objects(self):
        """Pair -> Thumb -> ras.footprint is the ~150 ms/thumb path."""
        session = _session()
        _headless(session)  # the stub's get_pairs() raises if it is called

    def test_falls_back_to_headers_when_th_res_is_missing(self):
        """An older geodatabase must still work, just more slowly."""
        session = _session(with_res=False)
        frame, _ = _headless(session)
        assert len(frame) == 7
        # bounded by unique thumbs, not pairs
        assert 0 < session.resolution_reads <= 14
        assert set(frame["resolution_m"]) == {3.0, 1.5}

    def test_defaults_flow_into_the_derivation(self):
        frame, _ = _headless(_session(), velocity_m_yr=5.0, footprint_m=30.0)
        assert set(frame["velocity_m_yr"]) == {5.0}
        # 30 m footprint at 3 m/px is 11 px, at 1.5 m/px is 21 px
        assert set(frame[frame["sensor_i"] == "planetscope"]["kernel_px"]) <= {11}

    def test_unknown_view_raises(self):
        with pytest.raises(ValueError, match="Unknown view"):
            _headless(_session(), view="nope")

    def test_unknown_default_raises_rather_than_being_ignored(self):
        with pytest.raises(TypeError, match="Unknown keyword"):
            _headless(_session(), velocity=5.0)

    @pytest.mark.parametrize("view", list(Session._CORR_VIEW_CONTROLS))
    def test_every_view_builds_headlessly(self, view):
        _, fig = _headless(_session(), view=view)
        assert fig is not None

    def test_no_pairs_gives_an_empty_frame_not_an_error(self):
        frame, fig = _headless(_StubSession(*_stub_frames([])))
        assert len(frame) == 0
        assert fig is not None

    def test_savefig_routes_through_save_correlation_figure(self):
        session = _session()
        session.explore_correlation_params(
            interactive=False, savefig=True, write_plan=False
        )
        assert len(session.saved) == 1
        assert session.saved[0]["view"] == "design_map"

    def test_savefig_skipped_on_an_empty_frame(self):
        session = _StubSession(*_stub_frames([]))
        session.explore_correlation_params(
            interactive=False, savefig=True, write_plan=False
        )
        assert session.saved == []


class TestFilters:
    def test_sensor_filter_narrows_the_frame(self):
        frame, _ = _headless(_session(), sensor_filter="spot")
        assert set(frame["sensor_i"]) == {"spot"}

    def test_min_dt_prunes_below_the_detection_floor(self):
        """The operational punchline: pruning beats adding parameter sets."""
        frame, _ = _headless(_session(), min_dt_days=500)
        assert frame["dt_days"].min() >= 500

    def test_max_dt_prunes_the_long_tail(self):
        frame, _ = _headless(_session(), max_dt_days=500)
        assert frame["dt_days"].max() <= 500

    def test_bin_ratio_changes_the_grouping(self):
        coarse, _ = _headless(_session(), dt_bin_ratio=10.0)
        fine, _ = _headless(_session(), dt_bin_ratio=2.0)
        assert coarse["group"].nunique() < fine["group"].nunique()


class TestStash:
    def test_stash_carries_exactly_the_documented_keys(self):
        session = _session()
        _headless(session)
        assert set(session._last_corr_params) == set(Session._CORR_PARAM_KEYS)

    def test_stash_splats_into_prepare_pairs_correlation(self):
        """The documented commit workflow must not raise TypeError."""
        import inspect

        session = _session()
        _headless(session)
        accepted = set(
            inspect.signature(Session.prepare_pairs_correlation).parameters
        )
        assert set(session._last_corr_params) <= accepted

    def test_assumptions_are_stashed_separately(self):
        session = _session()
        _headless(session, velocity_m_yr=7.0)
        assert session._last_corr_assumptions["velocity_m_yr"] == 7.0
        # Both the derivation inputs and the uniform overrides: a trace that
        # recorded only the resolved numbers would not say WHY every group
        # shares them.
        assert set(session._last_corr_assumptions) == (
            set(Session._CORR_ASSUMPTION_KEYS) | set(Session._CORR_UNIFORM_KEYS)
        )

    def test_no_private_keys_leak_into_the_stash(self):
        """The parsed box is an internal detail, not part of the recipe."""
        session = _session()
        _headless(session)
        assert not any(k.startswith("_") for k in session._last_corr_assumptions)

    def test_velocity_is_not_in_the_splat(self):
        """prepare_pairs_correlation knows nothing about velocities."""
        session = _session()
        _headless(session)
        assert "velocity_m_yr" not in session._last_corr_params

    def test_plan_round_trips_back_to_the_exact_same_frame(self):
        """Every column, including the parameters — not a partial comparison.

        This only holds because the explorer derives twice: per-pair, then
        reduced to groups and re-derived.  Without the second pass the frame
        would show the per-pair ideal while the plan carried the group's
        reduction, and the figures would disagree with the files on disk.
        """
        session = _session()
        frame, _ = _headless(session, velocity_m_yr=2.0)
        pd.testing.assert_frame_equal(
            frame, corrparams_frame_from_plan(session._last_corr_plan)
        )

    def test_round_trip_holds_with_a_strain_ceiling_in_play(self):
        """The case that exposed it: strain caps some pairs and not others, so
        the group reduction genuinely differs from the per-pair derivation."""
        session = _session()
        frame, _ = _headless(session, velocity_m_yr=3.0, strain_rate_per_yr=0.02)
        pd.testing.assert_frame_equal(
            frame, corrparams_frame_from_plan(session._last_corr_plan)
        )

    def test_the_frame_shows_the_committed_parameters_not_the_ideal(self):
        """A pair whose group was reduced must report the group's kernel."""
        session = _session()
        frame, _ = _headless(session, velocity_m_yr=3.0, strain_rate_per_yr=0.02,
                             dt_bin_ratio=10.0)
        for group, block in session._last_corr_plan["groups"].items():
            committed = max(block["params"]["corr_kernel"])
            assert set(frame[frame["group"] == group]["kernel_px"]) == {committed}

    def test_every_row_is_sourced_from_its_group(self):
        session = _session()
        frame, _ = _headless(session)
        assert set(frame["source"]) == {"group"}

    def test_plan_groups_carry_asp_parameters(self):
        session = _session()
        _headless(session, explicit_search_above_days=0)
        for block in session._last_corr_plan["groups"].values():
            assert "corr_kernel" in block["params"]
            assert block["params"]["corr_search"] is not None
            # An explicit search leaves the low-res disparity nothing to do.
            assert block["params"]["corr_seed_mode"] == 0

    def test_seed_mode_moves_with_the_search_box(self):
        """0 and an explicit box, or 1 and none — never a box with seed 1, and
        never seed 0 with nothing for the correlator to work from."""
        session = _session()
        _headless(session, explicit_search_above_days=365)
        blocks = list(session._last_corr_plan["groups"].values())
        assert any(b["params"]["corr_search"] is None for b in blocks)
        assert any(b["params"]["corr_search"] is not None for b in blocks)
        for block in blocks:
            has_box = block["params"]["corr_search"] is not None
            assert block["params"]["corr_seed_mode"] == (0 if has_box else 1)

    def test_group_search_covers_its_longest_baseline(self):
        session = _session()
        frame, _ = _headless(session, velocity_m_yr=5.0, dt_bin_ratio=10.0,
                             explicit_search_above_days=0)
        for name, block in session._last_corr_plan["groups"].items():
            reach = max(abs(v) for v in block["params"]["corr_search"])
            worst = frame[frame["group"] == name]["disp_px"].max()
            assert reach >= worst


class TestWidgetMode:
    def _drive(self, session, **kwargs):
        # display(fig) prints a figure repr outside a kernel
        with contextlib.redirect_stdout(io.StringIO()):
            return session.explore_correlation_params(write_plan=False, **kwargs)

    def test_returns_a_widget_tree(self):
        import ipywidgets as widgets

        box = self._drive(_session())
        assert isinstance(box, widgets.VBox)

    def test_layout_ends_with_the_save_row(self):
        import ipywidgets as widgets

        box = self._drive(_session())
        assert isinstance(box.children[-1], widgets.HBox)

    def test_setup_reads_resolutions_once_not_per_redraw(self):
        """Expensive I/O at setup; every control change is pure arithmetic."""
        session = _session()
        box = self._drive(session)
        after_setup = session.resolution_reads
        view = box.children[0].children[0]
        view.value = "cost"
        view.value = "snr"
        assert session.resolution_reads == after_setup

    def test_a_view_change_does_not_re_derive(self):
        session = _session()
        box = self._drive(session)
        stash_before = session._last_corr_plan
        box.children[0].children[0].value = "cost"
        assert session._last_corr_plan is stash_before

    def test_a_derivation_change_does_re_derive(self):
        session = _session()
        box = self._drive(session)
        stash_before = session._last_corr_plan
        box.children[1].children[0].value = 9.0  # velocity
        assert session._last_corr_plan is not stash_before
        assert session._last_corr_assumptions["velocity_m_yr"] == 9.0

    def test_view_specific_controls_are_hidden_off_their_view(self):
        session = _session()
        box = self._drive(session, view="design_map")
        row1 = box.children[0]
        colorby = row1.children[1]
        assert colorby.layout.display is None  # design_map uses color_by
        row1.children[0].value = "cost"
        assert colorby.layout.display == "none"

    def test_initial_view_is_configured_too_not_only_on_switch(self):
        """Applying per-view defaults only on switch left the first view wrong."""
        session = _session()
        box = self._drive(session, view="cost")
        assert box.children[0].children[1].layout.display == "none"

    def test_save_button_never_raises(self):
        session = _session()
        box = self._drive(session)
        button = box.children[-1].children[1].children[0]
        session.save_correlation_figure = lambda **kw: (_ for _ in ()).throw(
            RuntimeError("disk full")
        )
        button.click()  # must not propagate
        assert not button.disabled

    def test_save_button_reports_success(self):
        session = _session()
        box = self._drive(session)
        box.children[-1].children[1].children[0].click()
        assert session.saved


class TestPlanFollowsLiveSettings:
    """The plan's boxes must reflect what the user actually set, not the
    values the explorer happened to open with."""

    def test_coreg_allowance_reaches_the_plan(self):
        tight = _session()
        _headless(tight, velocity_m_yr=5.0, coreg_px=0.0, margin_px=0.0,
                  explicit_search_above_days=0)
        loose = _session()
        _headless(loose, velocity_m_yr=5.0, coreg_px=20.0, margin_px=20.0,
                  explicit_search_above_days=0)

        def widest(session):
            return max(
                max(abs(v) for v in block["params"]["corr_search"])
                for block in session._last_corr_plan["groups"].values()
            )

        assert widest(loose) > widest(tight)

    def test_flow_azimuth_reaches_the_plan(self):
        session = _session()
        _headless(session, velocity_m_yr=5.0, flow_azimuth_deg=90.0,
                  explicit_search_above_days=0)
        for block in session._last_corr_plan["groups"].values():
            x_lo, y_lo, x_hi, y_hi = block["params"]["corr_search"]
            assert (x_hi - x_lo) >= (y_hi - y_lo)

    def test_plan_search_matches_the_frame_reach(self):
        session = _session()
        frame, _ = _headless(session, velocity_m_yr=5.0, coreg_px=7.0,
                             margin_px=3.0, explicit_search_above_days=0)
        for name, block in session._last_corr_plan["groups"].items():
            reach = max(abs(v) for v in block["params"]["corr_search"])
            assert reach >= frame[frame["group"] == name]["search_px"].max()


class TestMultiPzonePlan:
    """A plan naming one pzone when the frame holds several would file it under
    whichever pzone sorted first, and silently apply it there."""

    def _two_pzones(self):
        pairs, thumbs = [], []
        for pz in ("PZ_A", "PZ_B"):
            pa, th = _stub_frames([(f"{pz}_400", "planetscope", 400)], pz=pz)
            pairs.append(pa)
            thumbs.append(th)
        return _StubSession(pd.concat(pairs, ignore_index=True),
                            pd.concat(thumbs, ignore_index=True))

    def test_multi_pzone_frame_claims_no_pzone(self):
        session = self._two_pzones()
        _headless(session)
        assert session._last_corr_plan["pzone"] == ""

    def test_single_pzone_frame_still_claims_it(self):
        session = _session()
        _headless(session)
        assert session._last_corr_plan["pzone"] == "PZ"

    def test_explicit_pz_name_wins(self):
        session = self._two_pzones()
        # pz_name filters the facts, so only that pzone's pairs survive
        _headless(session, pz_name="PZ_A")
        assert session._last_corr_plan["pzone"] == "PZ_A"

    def test_multi_pzone_plan_is_not_written(self, tmp_path):
        session = self._two_pzones()
        session.explore_correlation_params(
            interactive=False, savefig=False, write_plan=True
        )
        # pz_dir would have raised had it been called with a real pzone name
        assert session._last_corr_plan["pzone"] == ""


class TestDerivationInputsStayVisible:
    """A hidden control still shapes the parameters written to disk — the user
    would be governed by a value they cannot see."""

    def _drive(self, session, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return session.explore_correlation_params(write_plan=False, **kwargs)

    def test_strain_and_tau_are_never_hidden(self):
        session = _session()
        box = self._drive(session, view="cost")
        row3 = box.children[2]
        for widget in row3.children:
            assert widget.layout.display is None

    def test_only_view_local_options_are_conditional(self):
        for view, controls in Session._CORR_VIEW_CONTROLS.items():
            assert controls <= {"color_by"}, view

    def test_no_derivation_input_is_listed_as_view_local(self):
        for controls in Session._CORR_VIEW_CONTROLS.values():
            assert not controls & set(Session._CORR_ASSUMPTION_KEYS)

    def test_strain_rate_still_reaches_the_derivation_from_any_view(self):
        session = _session()
        box = self._drive(session, view="cost")
        strain = box.children[2].children[3]
        strain.value = 0.05
        assert session._last_corr_assumptions["strain_rate_per_yr"] == 0.05


class TestOverridesOnTopOfTheStash:
    """``_last_corr_params`` already carries a ``params_by_pair`` key, so
    splatting it *and* passing overrides separately is a TypeError. The
    documented pattern is to merge into the stash — locked here because both
    the guide and the tutorial notebook tell users to do it this way."""

    def test_splatting_plus_a_second_override_collides(self):
        session = _session()
        _headless(session)
        with pytest.raises(TypeError, match="params_by_pair"):
            Session.prepare_pairs_correlation(
                session, **session._last_corr_params, params_by_pair={"x": {}}
            )

    def test_merging_into_the_stash_is_accepted(self):
        import inspect

        session = _session()
        _headless(session)
        merged = {**session._last_corr_params, "params_by_pair": {"ps_30": {}}}
        accepted = set(inspect.signature(Session.prepare_pairs_correlation).parameters)
        assert set(merged) <= accepted
        assert merged["params_by_pair"] == {"ps_30": {}}
        assert merged["plan"] is session._last_corr_params["plan"]


class TestDurationStrings:
    """Δt filters accept the same "1Y"/"6M"/"30D" forms update_pairs documents."""

    def test_duration_string_through_defaults(self):
        frame, _ = _headless(_session(), min_dt_days="6M")
        assert frame["dt_days"].min() >= 180

    def test_duration_string_matches_the_equivalent_integer(self):
        by_string, _ = _headless(_session(), max_dt_days="1Y")
        by_int, _ = _headless(_session(), max_dt_days=365)
        pd.testing.assert_frame_equal(by_string, by_int)

    def test_case_insensitive(self):
        lower, _ = _headless(_session(), min_dt_days="6m")
        upper, _ = _headless(_session(), min_dt_days="6M")
        pd.testing.assert_frame_equal(lower, upper)

    def test_bad_duration_raises_from_a_script(self):
        """Headless callers get the error; only the widget degrades."""
        with pytest.raises(ValueError, match="Cannot parse duration"):
            _headless(_session(), min_dt_days="nonsense")

    def test_both_bounds_together(self):
        frame, _ = _headless(_session(), min_dt_days="2M", max_dt_days="3Y")
        assert frame["dt_days"].min() >= 60
        assert frame["dt_days"].max() <= 1095

    def _drive(self, session, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return session.explore_correlation_params(write_plan=False, **kwargs)

    def test_widget_accepts_a_duration_string(self):
        session = _session()
        box = self._drive(session, min_dt_days="6M")
        mindt = _widget(box, "min Δt")
        assert mindt.value == "6M"
        mindt.value = "1Y"
        assert session._last_corr_plan["pairs"]
        assert min(p["dt_days"] for p in session._last_corr_plan["pairs"]) >= 365

    def test_widget_accepts_a_bare_number_as_days(self):
        session = _session()
        box = self._drive(session)
        _widget(box, "min Δt").value = "500"
        assert min(p["dt_days"] for p in session._last_corr_plan["pairs"]) >= 500

    def test_bad_widget_input_degrades_instead_of_raising(self):
        """An exception here would reach only the kernel log, which VSCode hides."""
        session = _session()
        box = self._drive(session)
        mindt = _widget(box, "min Δt")
        mindt.value = "6M"
        n_before = len(session._last_corr_plan["pairs"])
        mindt.value = "!!!"            # must not propagate
        assert len(session._last_corr_plan["pairs"]) == n_before

    def test_clearing_the_field_turns_the_filter_off(self):
        session = _session()
        box = self._drive(session, min_dt_days="1Y")
        mindt = _widget(box, "min Δt")
        mindt.value = ""
        assert len(session._last_corr_plan["pairs"]) == 7

    def test_stash_keeps_resolved_integer_days(self):
        """The stash records what was applied, not what was typed."""
        session = _session()
        _headless(session, min_dt_days="6M")
        assert session._last_corr_filters["min_dt_days"] == 180


class TestUniformOverrides:
    """Two independent checkboxes, so all four combinations are meaningful."""

    def test_neither_is_the_derived_behaviour(self):
        frame, _ = _headless(_session(), explicit_search_above_days=0)
        assert frame["kernel_px"].nunique() > 1   # per-group derivation

    def test_uniform_kernel_reaches_every_group(self):
        session = _session()
        frame, _ = _headless(session, uniform_kernel=True, uniform_kernel_px=21)
        assert set(frame["kernel_px"]) == {21}
        for block in session._last_corr_plan["groups"].values():
            assert block["params"]["corr_kernel"] == [21, 21]

    def test_uniform_kernel_is_coerced_odd(self):
        frame, _ = _headless(_session(), uniform_kernel=True, uniform_kernel_px=20)
        assert set(frame["kernel_px"]) == {21}

    def test_uniform_kernel_respects_the_texture_floor(self):
        frame, _ = _headless(_session(), uniform_kernel=True, uniform_kernel_px=3)
        assert set(frame["kernel_px"]) == {7}

    def test_uniform_search_reaches_every_explicit_group(self):
        session = _session()
        _headless(session, uniform_search=True, uniform_search_box="20",
                  explicit_search_above_days=0)
        for block in session._last_corr_plan["groups"].values():
            assert block["params"]["corr_search"] == [-20, -20, 20, 20]

    def test_uniform_search_accepts_an_asymmetric_box(self):
        session = _session()
        _headless(session, uniform_search=True, uniform_search_box="-80 -2 20 2",
                  explicit_search_above_days=0)
        for block in session._last_corr_plan["groups"].values():
            assert block["params"]["corr_search"] == [-80, -2, 20, 2]

    def test_both_together_give_one_parameter_set(self):
        session = _session()
        _headless(session, uniform_kernel=True, uniform_kernel_px=21,
                  uniform_search=True, uniform_search_box="20",
                  explicit_search_above_days=0)
        blocks = list(session._last_corr_plan["groups"].values())
        assert len({str(b["params"]["corr_kernel"]) for b in blocks}) == 1
        assert len({str(b["params"]["corr_search"]) for b in blocks}) == 1

    def test_kernel_uniform_leaves_search_derived(self):
        session = _session()
        _headless(session, uniform_kernel=True, explicit_search_above_days=0)
        boxes = {str(b["params"]["corr_search"])
                 for b in session._last_corr_plan["groups"].values()}
        assert len(boxes) > 1

    def test_the_threshold_still_gates_whether_a_box_is_used(self):
        """Uniform sets the VALUE; the Δt threshold decides WHETHER."""
        session = _session()
        _headless(session, uniform_search=True, uniform_search_box="20",
                  explicit_search_above_days=365)
        blocks = session._last_corr_plan["groups"]
        short = [b for name, b in blocks.items()
                 if b["dt_days"][1] < 365]
        assert short, "the stub archive must contain a short-baseline group"
        for block in short:
            assert block["params"]["corr_search"] is None
            assert block["params"]["corr_seed_mode"] == 1

    def test_group_keys_are_unchanged_by_the_checkboxes(self):
        """Toggling a checkbox changes the plan's values, never its shape."""
        plain = _session()
        _headless(plain)
        uniform = _session()
        _headless(uniform, uniform_kernel=True, uniform_search=True)
        assert set(plain._last_corr_plan["groups"]) == set(
            uniform._last_corr_plan["groups"]
        )

    def test_a_conflicting_uniform_kernel_is_flagged_not_clamped(self):
        """The whole point: you get the run you asked for, plus the warning."""
        frame, _ = _headless(_session(), uniform_kernel=True, uniform_kernel_px=41,
                             strain_rate_per_yr=0.05)
        assert set(frame["kernel_px"]) == {41}          # not clamped
        assert frame["flags"].str.contains("strain ceiling").any()

    def test_a_too_small_uniform_box_is_flagged(self):
        frame, _ = _headless(_session(), velocity_m_yr=20.0, uniform_search=True,
                             uniform_search_box="2", explicit_search_above_days=0)
        assert frame["flags"].str.contains("rail at the box edge").any()

    def test_a_compatible_uniform_value_flags_nothing_new(self):
        # +/-10 px covers 30 m on 3 m pixels, far beyond 1 m/yr over 2500 d,
        # and costs 4x the ASP default -- inside the budget.
        frame, _ = _headless(_session(), velocity_m_yr=1.0, uniform_kernel=True,
                             uniform_kernel_px=21, uniform_search=True,
                             uniform_search_box="10", explicit_search_above_days=0)
        assert not frame["flags"].str.contains("uniform").any()

    def test_an_expensive_uniform_box_is_flagged_for_every_pair(self):
        """An oversized box costs the whole archive, not one pair."""
        frame, _ = _headless(_session(), velocity_m_yr=1.0, uniform_search=True,
                             uniform_search_box="40", explicit_search_above_days=0)
        assert frame["flags"].str.contains("costs").all()

    def test_the_plan_records_why_the_groups_match(self):
        session = _session()
        _headless(session, uniform_kernel=True, uniform_kernel_px=21)
        assumptions = session._last_corr_plan["assumptions"]
        assert assumptions["uniform_kernel"] is True
        assert assumptions["uniform_kernel_px"] == 21

    def test_a_bad_box_raises_from_a_script(self):
        with pytest.raises(ValueError, match="Cannot parse search box"):
            _headless(_session(), uniform_search=True, uniform_search_box="nonsense")

    def _drive(self, session, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return session.explore_correlation_params(write_plan=False, **kwargs)

    def test_checkbox_toggles_live(self):
        session = _session()
        box = self._drive(session, explicit_search_above_days=0)
        before = {str(b["params"]["corr_kernel"])
                  for b in session._last_corr_plan["groups"].values()}
        _widget(box, "uniform kernel").value = True
        after = {str(b["params"]["corr_kernel"])
                 for b in session._last_corr_plan["groups"].values()}
        assert len(before) > 1 and len(after) == 1

    def test_a_bad_box_in_the_widget_degrades_instead_of_raising(self):
        session = _session()
        box = self._drive(session, explicit_search_above_days=0)
        _widget(box, "uniform search box").value = True
        good = _widget(box, "box:")
        good.value = "30"
        n_before = len(session._last_corr_plan["groups"])
        good.value = "not a box"          # must not propagate
        assert len(session._last_corr_plan["groups"]) == n_before
