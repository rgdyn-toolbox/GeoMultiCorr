#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_explore_weights.py
# creation date: 2026-08-31.
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
"""tests/inversion/test_explore_weights.py

``explore_weights`` is drivable headlessly (``interactive=False``) and its widget
tree needs only a stub session, so the whole thing is testable without a project,
a geodatabase or a display.

The contracts worth not re-breaking:

* ``_last_weights_params`` carries **exactly** the twelve keys both writers
  accept, so ``prepare_inversion(**params)`` is a legal splat;
* the weight vectors are computed **from that dict**, so the two cannot disagree;
* one recompute per weighting change, and **zero** on a view-only change.
"""
from __future__ import annotations

import contextlib
import inspect
import io
from types import SimpleNamespace

import pytest

import geomulticorr.inversion.tio_inversion as tio
from geomulticorr.inversion.tio_inversion import TIOInversion


def _stats(nmad: float = 0.3, cc: float = 0.7) -> dict:
    return {
        "final_corrected_stats": {"ew": {"nmad": nmad}, "ns": {"nmad": nmad + 0.05}},
        "raw_corr_stats": {"cc": {"cc_quality_gte_050": cc}},
    }


def _pairs(n: int = 4):
    out = []
    for i in range(n):
        out.append(
            SimpleNamespace(
                pa_key=f"PZ_p{i}",
                pa_dt_days=30 * (i + 1),
                pa_direction="forward",
                pa_left=SimpleNamespace(th_date="2022-01-01"),
                pa_right=SimpleNamespace(th_date=f"2022-0{i + 2}-01"),
            )
        )
    return out


@pytest.fixture
def inv(monkeypatch):
    """A TIOInversion with pairs and stats, built without touching a project."""
    monkeypatch.setattr(tio, "load_pair_stats", lambda p: _stats())
    obj = TIOInversion.__new__(TIOInversion)
    obj.pairs = _pairs()
    obj.inversion_name = "test_inv"
    obj.pzone_name = "PZ"
    obj.filter_pipeline = None
    obj._quality_metrics = None
    obj._last_weights = None
    obj._last_weights_params = None
    return obj


class TestSplatContracts:
    """The whole point of the dict: it can be splatted into both writers."""

    def test_stash_holds_exactly_the_declared_keys(self, inv):
        frame, _ = inv.explore_weights(interactive=False, savefig=False)
        assert set(inv._last_weights_params) == set(TIOInversion._WEIGHT_PARAM_KEYS)

    @pytest.mark.parametrize("method", ["write_liste_couple", "prepare_inversion"])
    def test_keys_are_a_subset_of_both_writers(self, method):
        params = set(inspect.signature(getattr(TIOInversion, method)).parameters)
        assert set(TIOInversion._WEIGHT_PARAM_KEYS) <= params, (
            f"{method} cannot accept the stash as **kwargs"
        )

    def test_direction_is_excluded(self):
        """Including it would make both splats raise TypeError — it is view-only."""
        assert "direction" not in TIOInversion._WEIGHT_PARAM_KEYS
        for method in ("write_liste_couple", "prepare_inversion"):
            params = inspect.signature(getattr(TIOInversion, method)).parameters
            assert "direction" not in params

    def test_dt_range_is_carried(self, inv):
        """Before this, the explorer could never reproduce a dt_range run."""
        inv.explore_weights("sigmoid", interactive=False, savefig=False,
                            dt_range=(300, 400))
        assert inv._last_weights_params["dt_range"] == (300.0, 400.0)

    def test_stash_is_json_safe(self, inv):
        import json

        inv.explore_weights(interactive=False, savefig=False)
        json.dumps(inv._last_weights_params)


class TestHeadless:
    """No ipywidgets, no display — usable from scripts and batch jobs."""

    def test_returns_a_frame_and_a_figure(self, inv):
        frame, fig = inv.explore_weights(interactive=False, savefig=False)
        assert len(frame) == len(inv.pairs)
        assert len(fig.data) == 2  # one series per direction

    def test_frame_carries_the_pair_keys(self, inv):
        frame, _ = inv.explore_weights(interactive=False, savefig=False)
        assert frame["pa_key"].tolist() == [p.pa_key for p in inv.pairs]

    def test_direction_selects_the_visible_series(self, inv):
        _, fig = inv.explore_weights(interactive=False, savefig=False,
                                     direction="EW")
        assert [t.visible for t in fig.data] == [True, False]

    def test_weights_are_stashed_too(self, inv):
        inv.explore_weights(interactive=False, savefig=False)
        assert set(inv._last_weights) == {"EW", "NS"}
        assert len(inv._last_weights["EW"]) == len(inv.pairs)

    def test_savefig_warns_and_skips_on_no_pairs(self, inv, caplog_gmc):
        inv.pairs = []
        inv._quality_metrics = None
        frame, _ = inv.explore_weights("uniform", interactive=False, savefig=True)
        assert len(frame) == 0
        assert "nothing saved" in caplog_gmc.text.lower()


class TestComputeReadsTheStash:
    """Params and vectors are one thing derived from the other, not two."""

    def test_vectors_match_a_direct_recompute_from_the_dict(self, inv):
        inv.explore_weights("quality", interactive=False, savefig=False,
                            combine="wmean", cc_gamma=1.4)
        params = dict(inv._last_weights_params)
        mode = params.pop("weight_mode")
        expected = inv.compute_pair_weights(mode, direction="EW", **params)
        assert inv._last_weights["EW"] == pytest.approx(expected)

    def test_the_stash_replays_through_write_liste_couple(self, inv, tmp_path,
                                                          monkeypatch):
        """The documented commit step must actually run."""
        inv.inversion_dir = tmp_path
        inv._DIRECTIONS = ("EW", "NS")
        (tmp_path / "inverse_EW").mkdir()
        (tmp_path / "inverse_NS").mkdir()
        monkeypatch.setattr(inv, "_sync_pair_weights", lambda *a, **k: None)

        inv.explore_weights("quality", interactive=False, savefig=False)
        written = inv.write_liste_couple(**inv._last_weights_params,
                                         sync_geodb=False)
        assert written["EW"] == pytest.approx(inv._last_weights["EW"])


class TestDefaultsValidation:
    """`sharpnes=8` used to be silently ignored."""

    def test_unknown_keyword_raises(self, inv):
        with pytest.raises(TypeError, match="sharpnes"):
            inv.explore_weights(interactive=False, savefig=False, sharpnes=8)

    def test_the_error_lists_what_is_accepted(self, inv):
        with pytest.raises(TypeError, match="combine"):
            inv.explore_weights(interactive=False, savefig=False, nope=1)

    @pytest.mark.parametrize("key", TIOInversion._WEIGHTS_EXPLORER_DEFAULTS)
    def test_every_declared_default_is_accepted(self, inv, key):
        value = {"direction": "EW", "combine": "wmean", "invert": True,
                 "dt_range": (10, 20)}.get(key, 1.0)
        inv.explore_weights(interactive=False, savefig=False, **{key: value})


class TestWidgetTree:
    """Driven headlessly — the explorer needs no project, only the pairs."""

    @pytest.fixture
    def widget_inv(self, inv):
        pytest.importorskip("ipywidgets")
        return inv

    def _build(self, obj, **kwargs):
        # display(fig) prints a figure repr outside a kernel
        with contextlib.redirect_stdout(io.StringIO()):
            return obj.explore_weights(**kwargs)

    def test_builds_a_widget_tree(self, widget_inv):
        import ipywidgets as widgets

        box = self._build(widget_inv)
        assert isinstance(box, widgets.VBox)

    def test_a_save_row_is_present(self, widget_inv):
        import ipywidgets as widgets

        box = self._build(widget_inv)
        buttons = [w for row in box.children
                   if isinstance(row, widgets.HBox)
                   for w in row.children if isinstance(w, widgets.Button)]
        assert [b.description for b in buttons] == ["Save figure"]

    def test_construction_stashes_and_computes(self, widget_inv):
        self._build(widget_inv)
        assert set(widget_inv._last_weights_params) == set(
            TIOInversion._WEIGHT_PARAM_KEYS)
        assert widget_inv._last_weights is not None

    def _controls(self, box):
        import ipywidgets as widgets

        found = {}
        for row in box.children:
            if not isinstance(row, widgets.HBox):
                continue
            for w in row.children:
                found[getattr(w, "description", "")] = w
        return found

    def test_one_recompute_per_weighting_change(self, widget_inv, monkeypatch):
        box = self._build(widget_inv)
        controls = self._controls(box)

        calls = []
        original = TIOInversion.compute_pair_weights
        monkeypatch.setattr(
            TIOInversion, "compute_pair_weights",
            lambda self, *a, **k: calls.append(1) or original(self, *a, **k),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            controls["cc_gamma"].value = 2.0
        # exactly one recompute pass — two calls, one per direction
        assert len(calls) == 2

    def test_a_view_only_change_does_not_recompute(self, widget_inv, monkeypatch):
        """Switching the visible series must reuse the cached weights."""
        box = self._build(widget_inv)
        controls = self._controls(box)

        calls = []
        original = TIOInversion.compute_pair_weights
        monkeypatch.setattr(
            TIOInversion, "compute_pair_weights",
            lambda self, *a, **k: calls.append(1) or original(self, *a, **k),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            controls["direction"].value = "EW"
        assert calls == []

    def test_mode_switch_reapplies_visibility(self, widget_inv):
        box = self._build(widget_inv, weight_mode="quality")
        controls = self._controls(box)
        # quality shows cc_gamma but not slope
        assert controls["cc_gamma"].layout.display is None
        assert controls["slope"].layout.display == "none"

        with contextlib.redirect_stdout(io.StringIO()):
            controls["mode"].value = "parametric"
        assert controls["slope"].layout.display is None
        assert controls["cc_gamma"].layout.display == "none"

    def test_abg_appear_only_under_wmean(self, widget_inv):
        box = self._build(widget_inv, weight_mode="quality")
        controls = self._controls(box)
        assert controls["α nmad"].layout.display == "none"

        with contextlib.redirect_stdout(io.StringIO()):
            controls["combine"].value = "wmean"
        assert controls["α nmad"].layout.display is None


class TestQualityMetricsAreHoisted:
    """One stats JSON per pair, read once — not 2 x len(pairs) per slider tick."""

    def test_repeated_calls_read_the_files_once(self, inv, monkeypatch):
        reads = []
        monkeypatch.setattr(tio, "load_pair_stats",
                            lambda p: reads.append(p.pa_key) or _stats())
        inv._quality_metrics = None

        inv.compute_pair_weights("quality", direction="EW")
        first = len(reads)
        inv.compute_pair_weights("quality", direction="NS")
        assert len(reads) == first == len(inv.pairs)

    def test_the_nmad_filter_invalidates_the_cache(self, inv, monkeypatch):
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: _stats())
        inv._image_dates = None
        inv.compute_pair_weights("quality")
        assert inv._quality_metrics is not None

        inv.filter_pairs_by_nmad(threshold=10.0)
        assert inv._quality_metrics is None

    def test_the_metrics_stay_aligned_after_filtering(self, inv, monkeypatch):
        """A stale cache would misalign every weight against its pair."""
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: _stats())
        inv._image_dates = None
        inv.compute_pair_weights("quality")

        inv.filter_pairs_by_nmad(threshold=0.1)  # drops everything
        assert inv.pairs == []
        assert len(inv.compute_pair_weights("quality")) == 0
