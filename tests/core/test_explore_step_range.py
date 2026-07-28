#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The explorer's max_step ceiling must follow the number of acquisitions.

A hardcoded ceiling silently drops the longest-baseline pairs on any archive with
more images than the constant — the regression these tests lock out.
"""
from __future__ import annotations

import contextlib
import io
import pathlib

import pandas as pd
import pytest

pytest.importorskip("plotly")
pytest.importorskip("ipywidgets")

from geomulticorr.core.session import Session  # noqa: E402
from geomulticorr.utils._pairs_frame import unique_couples  # noqa: E402


class _Thumb:
    """Minimal stand-in — the explorer only reads these four attributes."""

    def __init__(self, date: str, sensor: str, pz: str):
        self.th_date = date
        self.th_date_datetime = pd.Timestamp(date)
        self.th_sensor = sensor
        self.th_path = f"/data/{pz}_{date}_{sensor}.tif"


class _Pzone:
    def __init__(self, name: str, thumbs: list):
        self.pz_name = name
        self._thumbs = thumbs

    def get_valid_thumbs(self):
        return self._thumbs


class _StubSession:
    """A Session-like object exposing just what explore_pairs_strategy touches.

    ``save_pairs_figure`` records its arguments instead of writing files, so the
    auto-save wiring is testable without a project on disk.
    """

    _STRATEGY_CONTROLS = Session._STRATEGY_CONTROLS
    _VIEW_CONTROLS = Session._VIEW_CONTROLS
    _VIEW_DEDUPE_DEFAULT = Session._VIEW_DEDUPE_DEFAULT
    _DRAWN_NOUN = Session._DRAWN_NOUN
    _EXPLORER_DEFAULTS = Session._EXPLORER_DEFAULTS
    explore_pairs_strategy = Session.explore_pairs_strategy

    def __init__(self, pzones: list):
        self._pzones = pzones
        self.saved: list[dict] = []

    def get_pzones(self, pz_name: str = ""):
        return self._pzones

    def save_pairs_figure(self, **kwargs):
        self.saved.append(kwargs)
        return {fmt: pathlib.Path(f"stub.{fmt}") for fmt in kwargs.get("formats", ())}


def _make_session(pzones: list):
    return _StubSession(pzones)


def _thumbs(n: int, sensor: str = "planetscope", pz: str = "PZ1", start: str = "2016-03-01"):
    dates = pd.date_range(start, periods=n, freq="73D").strftime("%Y-%m-%d")
    return [_Thumb(d, sensor, pz) for d in dates]


def _explore(session, **kwargs):
    """Build the widget without letting the figure repr reach stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        return session.explore_pairs_strategy(**kwargs)


class _Controls:
    """Named access to the explorer's widgets."""

    def __init__(self, box):
        self.box = box
        self.view, self.strategy, self.step, self.pz = box.children[0].children
        self.maxdt, self.mindt, self.sensor = box.children[1].children
        (self.colorby, self.bins, self.dedupe,
         self.arrows, self.mirror) = box.children[2].children
        self.summary = box.children[4]

    def set(self, widget, value):
        with contextlib.redirect_stdout(io.StringIO()):
            widget.value = value


@pytest.fixture
def ctrl_18():
    session = _make_session([_Pzone("PZ1", _thumbs(18))])
    return _Controls(_explore(session, strategy="redundancy"))


class TestCeilingFollowsImageCount:
    @pytest.mark.parametrize("n, expected", [(18, 17), (5, 4), (2, 1), (1, 1)])
    def test_ceiling_is_n_minus_one(self, n, expected):
        session = _make_session([_Pzone("PZ1", _thumbs(n))])
        ctrl = _Controls(_explore(session, strategy="step"))
        assert ctrl.step.max == expected

    def test_eighteen_images_reach_seventeen(self, ctrl_18):
        """The reported bug: the slider stopped at 10 with 18 images."""
        assert ctrl_18.step.max == 17
        assert ctrl_18.step.max > 10

    def test_single_thumb_does_not_crash(self):
        session = _make_session([_Pzone("PZ1", _thumbs(1))])
        ctrl = _Controls(_explore(session, strategy="consecutive"))
        assert ctrl.step.min == ctrl.step.max == 1

    def test_no_thumbs_does_not_crash(self):
        session = _make_session([_Pzone("PZ1", [])])
        ctrl = _Controls(_explore(session, strategy="consecutive"))
        assert ctrl.step.max == 1

    def test_tooltip_reports_the_ceiling(self, ctrl_18):
        assert "17" in (ctrl_18.step.tooltip or "")


class TestInitialValueClamped:
    def test_default_value_preserved_when_it_fits(self, ctrl_18):
        assert ctrl_18.step.value == 2

    def test_oversized_default_is_clamped(self):
        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        ctrl = _Controls(_explore(session, strategy="redundancy", max_step=25))
        assert ctrl.step.value == 17

    def test_zero_default_floored_to_one(self):
        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        ctrl = _Controls(_explore(session, strategy="step", max_step=0))
        assert ctrl.step.value == 1


class TestCeilingTracksFilters:
    def _mixed_session(self):
        # 12 planetscope + 6 spot in one pzone
        thumbs = _thumbs(12, "planetscope") + _thumbs(6, "spot", start="2020-01-01")
        return _make_session([_Pzone("PZ1", thumbs)])

    def test_sensor_filter_lowers_ceiling(self):
        ctrl = _Controls(_explore(self._mixed_session(), strategy="step"))
        assert ctrl.step.max == 17
        ctrl.set(ctrl.sensor, "spot")
        assert ctrl.step.max == 5

    def test_value_clamped_when_ceiling_drops(self):
        ctrl = _Controls(_explore(self._mixed_session(), strategy="step", max_step=15))
        assert ctrl.step.value == 15
        ctrl.set(ctrl.sensor, "spot")
        assert ctrl.step.value == 5
        assert ctrl.step.max == 5

    def test_widening_filter_restores_ceiling(self):
        ctrl = _Controls(_explore(self._mixed_session(), strategy="step"))
        ctrl.set(ctrl.sensor, "spot")
        assert ctrl.step.max == 5
        ctrl.set(ctrl.sensor, "")
        assert ctrl.step.max == 17

    def test_pzone_selection_changes_ceiling(self):
        session = _make_session(
            [_Pzone("BIG", _thumbs(18, pz="BIG")), _Pzone("SMALL", _thumbs(5, pz="SMALL"))]
        )
        ctrl = _Controls(_explore(session, strategy="step"))
        assert ctrl.step.max == 17  # <all> → bounded by the largest pzone
        ctrl.set(ctrl.pz, "SMALL")
        assert ctrl.step.max == 4
        ctrl.set(ctrl.pz, "BIG")
        assert ctrl.step.max == 17

    def test_unmatched_filter_gives_minimum_ceiling(self):
        ctrl = _Controls(_explore(self._mixed_session(), strategy="step"))
        ctrl.set(ctrl.sensor, "no_such_sensor")
        assert ctrl.step.max == 1


class TestNoReentrantRecompute:
    @staticmethod
    def _count_recomputes(monkeypatch) -> dict:
        """Count _compute() passes.

        The explorer resolves ``gmc_pzone._strategy_pair_indices`` as a module
        attribute at call time, so patching it here is actually seen — unlike the
        names it imports into its local scope at setup.  One selected pzone means
        one call per recompute.
        """
        import geomulticorr.core.pzone as pzone_mod

        calls = {"n": 0}
        original = pzone_mod._strategy_pair_indices

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(pzone_mod, "_strategy_pair_indices", counting)
        return calls

    def test_filter_change_that_clamps_recomputes_once(self, monkeypatch):
        """The clamp fires a value event; the guard must absorb it."""
        thumbs = _thumbs(12, "planetscope") + _thumbs(6, "spot", start="2020-01-01")
        session = _make_session([_Pzone("PZ1", thumbs)])
        ctrl = _Controls(_explore(session, strategy="step", max_step=15))

        calls = self._count_recomputes(monkeypatch)
        ctrl.set(ctrl.sensor, "spot")     # 17 -> 5, clamping value 15 -> 5
        assert ctrl.step.value == 5
        assert calls["n"] == 1

    def test_plain_filter_change_recomputes_once(self, monkeypatch):
        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        ctrl = _Controls(_explore(session, strategy="step", max_step=2))

        calls = self._count_recomputes(monkeypatch)
        ctrl.set(ctrl.mindt, 100)          # no ceiling change at all
        assert calls["n"] == 1

    def test_moving_the_slider_recomputes_once(self, monkeypatch):
        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        ctrl = _Controls(_explore(session, strategy="step", max_step=2))

        calls = self._count_recomputes(monkeypatch)
        ctrl.set(ctrl.step, 17)
        assert calls["n"] == 1


class TestMirrorCheckboxReflectsAvailability:
    """"mirror directions" must not look broken when there is nothing to mirror."""

    def _ctrl(self, strategy, **kwargs):
        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        ctrl = _Controls(_explore(session, strategy=strategy, view="network", **kwargs))
        return ctrl

    @pytest.mark.parametrize("strategy", ["consecutive", "step"])
    def test_disabled_for_forward_only_strategies(self, strategy):
        ctrl = self._ctrl(strategy, max_step=5)
        assert ctrl.mirror.disabled is True
        assert "forward-only" in ctrl.mirror.tooltip
        assert "redundancy" in ctrl.mirror.tooltip

    @pytest.mark.parametrize("strategy", ["redundancy", "forward-backward"])
    def test_enabled_for_bidirectional_strategies(self, strategy):
        ctrl = self._ctrl(strategy, max_step=5)
        assert ctrl.mirror.disabled is False
        assert "below the timeline" in ctrl.mirror.tooltip

    def test_follows_a_strategy_switch(self):
        ctrl = self._ctrl("redundancy", max_step=5)
        assert ctrl.mirror.disabled is False
        ctrl.set(ctrl.strategy, "step")
        assert ctrl.mirror.disabled is True
        ctrl.set(ctrl.strategy, "redundancy")
        assert ctrl.mirror.disabled is False

    def test_dedupe_also_removes_every_backward_pair(self):
        """The second trap: dedupe keeps the forward copy of each couple."""
        ctrl = self._ctrl("redundancy", max_step=5)
        assert ctrl.mirror.disabled is False
        ctrl.set(ctrl.dedupe, True)
        assert ctrl.mirror.disabled is True
        assert "dedupe" in ctrl.mirror.tooltip
        ctrl.set(ctrl.dedupe, False)
        assert ctrl.mirror.disabled is False

    def test_network_defaults_to_dedupe_off(self):
        """So a bidirectional strategy shows its backward arcs straight away."""
        ctrl = self._ctrl("redundancy", max_step=5)
        assert ctrl.dedupe.value is False


class TestForwardOnlyFigureIsLabelled:
    def test_annotation_explains_absent_backward_pairs(self, dates_iso=None):
        from geomulticorr.core.pzone import _strategy_pair_indices
        from geomulticorr.utils import _pairs_plotly as pp
        from geomulticorr.utils._pairs_frame import pairs_frame_from_indices

        dates = pd.date_range("2016-03-01", periods=18, freq="73D")
        forward_only = pairs_frame_from_indices(
            dates, _strategy_pair_indices(18, "step", 5)
        )
        texts = [a.text for a in pp.figure_network(forward_only).layout.annotations]
        assert len(texts) == 1
        assert "no backward pairs to mirror" in texts[0]

    def test_bidirectional_keeps_the_two_side_labels(self):
        from geomulticorr.core.pzone import _strategy_pair_indices
        from geomulticorr.utils import _pairs_plotly as pp
        from geomulticorr.utils._pairs_frame import pairs_frame_from_indices

        dates = pd.date_range("2016-03-01", periods=18, freq="73D")
        both = pairs_frame_from_indices(dates, _strategy_pair_indices(18, "redundancy", 5))
        texts = [a.text for a in pp.figure_network(both).layout.annotations]
        assert len(texts) == 2
        assert any("above: forward" in t for t in texts)
        assert any("below: backward" in t for t in texts)


class TestNonInteractiveMode:
    """``interactive=False`` returns (frame, fig) and touches no widgets."""

    def _session(self, n=18):
        return _make_session([_Pzone("PZ1", _thumbs(n))])

    def test_returns_frame_and_figure(self):
        import plotly.graph_objects as go
        from geomulticorr.utils._pairs_frame import PAIRS_FRAME_COLUMNS

        session = self._session()
        frame, fig = session.explore_pairs_strategy(
            strategy="redundancy", max_step=17, view="chord", interactive=False
        )
        assert tuple(frame.columns) == PAIRS_FRAME_COLUMNS
        assert len(frame) == 306
        assert isinstance(fig, go.Figure)

    def test_arguments_drive_the_result(self):
        session = self._session()
        few, _ = session.explore_pairs_strategy(
            strategy="step", max_step=2, interactive=False)
        many, _ = session.explore_pairs_strategy(
            strategy="step", max_step=17, interactive=False)
        assert len(few) == 33      # sum(18-k) for k=1,2
        assert len(many) == 153    # 18*17/2

    def test_view_selects_the_figure(self):
        session = self._session()
        _, chord = session.explore_pairs_strategy(view="chord", strategy="step",
                                                  max_step=3, interactive=False)
        _, hist = session.explore_pairs_strategy(view="dt_hist", strategy="step",
                                                 max_step=3, interactive=False)
        assert len(chord.data) == 5                 # ring, links, hover, nodes, ticks
        assert hist.data[0].type == "histogram"

    def test_dt_filters_applied(self):
        session = self._session()
        frame, _ = session.explore_pairs_strategy(
            strategy="step", max_step=17, max_dt_days=200, interactive=False)
        assert len(frame) > 0
        assert frame["dt_days"].max() <= 200

    def test_max_step_clamped_to_the_data(self):
        session = self._session()
        frame, _ = session.explore_pairs_strategy(
            strategy="redundancy", max_step=999, interactive=False)
        assert len(frame) == 306
        assert session._last_pairs_params["max_step"] == 17

    def test_stash_written_and_commit_ready(self):
        import inspect

        session = self._session()
        session.explore_pairs_strategy(strategy="redundancy", max_step=5,
                                       interactive=False)
        params = session._last_pairs_params
        assert params["strategy"] == "redundancy"
        assert params["max_step"] == 5
        inspect.signature(Session.update_pairs).bind(None, **params)

    def test_forward_only_strategy_stashes_no_max_step(self):
        session = self._session()
        session.explore_pairs_strategy(strategy="consecutive", interactive=False)
        assert session._last_pairs_params["max_step"] is None

    def test_sensor_filter_honoured(self):
        thumbs = _thumbs(12, "planetscope") + _thumbs(6, "spot", start="2020-01-01")
        session = _make_session([_Pzone("PZ1", thumbs)])
        frame, _ = session.explore_pairs_strategy(
            strategy="consecutive", sensor_filter="spot", interactive=False)
        assert set(frame["sensor_i"]) == {"spot"}
        assert len(frame) == 5

    def test_needs_no_ipywidgets(self, monkeypatch):
        """The headless path must not import the widget stack."""
        import builtins

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.startswith(("ipywidgets", "IPython")):
                raise AssertionError(f"headless path imported {name}")
            return real_import(name, *args, **kwargs)

        session = self._session()
        monkeypatch.setattr(builtins, "__import__", guard)
        frame, fig = session.explore_pairs_strategy(strategy="step", max_step=3,
                                                    interactive=False)
        assert len(frame) > 0

    def test_figure_title_uses_the_shared_builder(self):
        """Chord dedupes, so the title reports drawn curves *and* total pairs."""
        _, fig = self._session().explore_pairs_strategy(
            strategy="redundancy", max_step=17, view="chord", interactive=False)
        assert "153 chords drawn of 306 pairs [redundancy]" in fig.layout.title.text

    @pytest.mark.parametrize("strategy, max_step",
                             [("step", 4), ("redundancy", 7), ("consecutive", None)])
    def test_frame_matches_the_interactive_one(self, strategy, max_step):
        """Both modes must agree pair-for-pair — they share one _build_frame."""
        kwargs = {"max_step": max_step} if max_step else {}
        frame, _ = self._session().explore_pairs_strategy(
            strategy=strategy, interactive=False, **kwargs)

        ctrl = _Controls(_explore(self._session(), strategy=strategy))
        if max_step:
            ctrl.set(ctrl.step, max_step)
        # the widget caches its frame privately; the summary reports its size
        assert f"<b>{len(frame)} pairs</b>" in ctrl.summary.value

    def test_default_view_is_baseline(self):
        _, fig = self._session().explore_pairs_strategy(
            strategy="step", max_step=2, interactive=False)
        assert fig.layout.yaxis.title.text == "Temporal baseline Δt (days)"
        # forward-only data still yields both direction traces, the second empty
        assert [t.name for t in fig.data] == ["forward", "backward"]
        assert len(fig.data[1].x) == 0

    def test_empty_selection_returns_empty_frame_not_an_error(self):
        session = self._session()
        frame, fig = session.explore_pairs_strategy(
            strategy="step", max_step=3, sensor_filter="nope", interactive=False)
        assert len(frame) == 0
        assert "No pairs" in fig.layout.annotations[0].text


class TestHeadlessAutoSave:
    """``savefig`` defaults to True so a scripted run leaves files behind."""

    def _session(self):
        return _make_session([_Pzone("PZ1", _thumbs(18))])

    def test_saves_by_default(self):
        session = self._session()
        session.explore_pairs_strategy(strategy="redundancy", max_step=5,
                                       view="chord", interactive=False)
        assert len(session.saved) == 1
        call = session.saved[0]
        assert call["view"] == "chord"
        assert call["formats"] == ("html", "png")
        assert len(call["frame"]) == 150

    def test_savefig_false_writes_nothing(self):
        session = self._session()
        session.explore_pairs_strategy(strategy="redundancy", max_step=5,
                                       interactive=False, savefig=False)
        assert session.saved == []

    def test_formats_are_forwarded(self):
        session = self._session()
        session.explore_pairs_strategy(strategy="step", max_step=3,
                                       interactive=False,
                                       formats=("png", "jpg", "pdf"))
        assert session.saved[0]["formats"] == ("png", "jpg", "pdf")

    def test_view_options_forwarded_so_saved_matches_returned(self):
        session = self._session()
        session.explore_pairs_strategy(strategy="redundancy", max_step=5,
                                       view="network", interactive=False)
        call = session.saved[0]
        assert call["mirror_direction"] is True
        assert call["dedupe"] is False           # network default
        assert call["color_by"] == "dt"

    def test_title_matches_the_returned_figure(self):
        session = self._session()
        _, fig = session.explore_pairs_strategy(strategy="redundancy", max_step=17,
                                                view="chord", interactive=False)
        assert session.saved[0]["title"] == fig.layout.title.text

    def test_empty_frame_skips_saving(self):
        session = self._session()
        frame, _ = session.explore_pairs_strategy(
            strategy="step", max_step=3, sensor_filter="nope", interactive=False)
        assert len(frame) == 0
        assert session.saved == []

    def test_interactive_mode_ignores_savefig(self):
        session = self._session()
        _explore(session, strategy="redundancy", savefig=True)
        assert session.saved == []      # the button saves, not the constructor


class TestArgumentValidation:
    def test_unknown_kwarg_raises(self):
        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        with pytest.raises(TypeError, match="Unexpected argument"):
            session.explore_pairs_strategy(max_dt=400, interactive=False)

    def test_unknown_kwarg_raises_in_interactive_mode_too(self):
        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        with pytest.raises(TypeError, match="Unexpected argument"):
            _explore(session, strategy="step", maxstep=3)

    def test_unknown_view_raises(self):
        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        with pytest.raises(ValueError, match="Unknown view"):
            session.explore_pairs_strategy(view="scatter", interactive=False)


class TestLongestPairsRecovered:
    def test_max_ceiling_pairs_everything(self, ctrl_18):
        """At the ceiling, redundancy emits every ordered couple — nothing lost."""
        ctrl_18.set(ctrl_18.step, ctrl_18.step.max)
        summary = ctrl_18.summary.value
        n_dates = 18
        expected_couples = n_dates * (n_dates - 1) // 2
        assert f"{2 * expected_couples} pairs" in summary
        assert f"{expected_couples} unique couples" in summary

    def test_ceiling_beats_the_old_hardcoded_ten(self, ctrl_18):
        ctrl_18.set(ctrl_18.step, 10)
        at_ten = ctrl_18.summary.value
        ctrl_18.set(ctrl_18.step, 17)
        at_max = ctrl_18.summary.value
        assert "250 pairs" in at_ten     # what the old slider capped you to
        assert "306 pairs" in at_max     # what the data actually supports

    def test_stash_carries_the_new_ceiling_to_update_pairs(self):
        """The tuned max_step must survive the documented commit workflow."""
        import inspect

        session = _make_session([_Pzone("PZ1", _thumbs(18))])
        ctrl = _Controls(_explore(session, strategy="redundancy"))
        ctrl.set(ctrl.step, ctrl.step.max)

        assert session._last_pairs_params["max_step"] == 17
        inspect.signature(Session.update_pairs).bind(None, **session._last_pairs_params)
