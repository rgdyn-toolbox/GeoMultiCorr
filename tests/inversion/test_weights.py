#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for TIO pair-weighting math and compute_pair_weights."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import geomulticorr.inversion.tio_inversion as tio
from geomulticorr.inversion.tio_inversion import (
    TIOInversion,
    combine_weights,
    normalise_dt,
    normalise_invert,
)

# ──────────────────────────────────────────────────────────────────────────────


class TestNormaliseInvert:
    """Tests for normalise_invert() — low value maps to 1.0."""

    def test_lowest_maps_to_one(self):
        out = normalise_invert([1.0, 2.0, 3.0])
        assert out[0] == pytest.approx(1.0)
        assert out[2] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.5)

    def test_all_equal_maps_to_one(self):
        assert normalise_invert([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]

    def test_empty_returns_empty(self):
        assert normalise_invert([]) == []

    def test_nan_is_neutral_one(self):
        out = normalise_invert([1.0, float("nan"), 3.0])
        assert out[1] == 1.0  # neutral, not dragged down
        assert out[0] == pytest.approx(1.0)
        assert out[2] == pytest.approx(0.0)

    def test_all_in_unit_range(self):
        out = normalise_invert([0.1, 0.5, 0.9, 0.3])
        assert all(0.0 <= v <= 1.0 for v in out)


class TestNormaliseDt:
    """Tests for normalise_dt() — short baseline maps to 1.0 by default."""

    def test_short_dt_high_weight(self):
        out = normalise_dt([10.0, 100.0])
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.0)

    def test_invert_reverses(self):
        out = normalise_dt([10.0, 100.0], invert=True)
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(1.0)

    def test_degenerate_span_all_one(self):
        assert normalise_dt([30.0, 30.0]) == [1.0, 1.0]

    def test_nan_is_neutral(self):
        out = normalise_dt([10.0, float("nan"), 100.0])
        assert out[1] == 1.0


class TestCombineWeights:
    """Tests for combine_weights() — three combination strategies."""

    def test_geomean(self):
        assert combine_weights(1.0, 0.5, 0.25, method="geomean") == pytest.approx(
            (1.0 * 0.5 * 0.25) ** (1 / 3)
        )

    def test_product(self):
        assert combine_weights(0.8, 0.5, 0.5, method="product") == pytest.approx(0.2)

    def test_wmean_equal_weights(self):
        assert combine_weights(0.3, 0.6, 0.9, method="wmean") == pytest.approx(0.6)

    def test_wmean_renormalises(self):
        # weights (2,0,0) -> only nmad term counts
        assert combine_weights(0.4, 0.9, 0.9, method="wmean",
                               alpha=2, beta=0, gamma=0) == pytest.approx(0.4)

    def test_nan_component_is_neutral(self):
        # missing cc -> treated as 1.0
        assert combine_weights(0.5, float("nan"), 0.5, method="product") == pytest.approx(0.25)

    def test_result_clamped_unit(self):
        for m in ("geomean", "wmean", "product"):
            w = combine_weights(1.0, 1.0, 1.0, method=m)
            assert 0.0 <= w <= 1.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown combine method"):
            combine_weights(1.0, 1.0, 1.0, method="nope")


# ──────────────────────────────────────────────────────────────────────────────


def _make_pair(key, dt_days):
    """Minimal stand-in for a Pair used by compute_pair_weights."""
    return SimpleNamespace(pa_key=key, pa_dt_days=dt_days)


def _stats(nmad, cc):
    return {
        "final_corrected_stats": {"ew": {"nmad": nmad}, "ns": {"nmad": nmad}},
        "raw_corr_stats": {"cc": {"cc_quality_gte_050": cc}},
    }


def _stats2(ew_nmad, ns_nmad, cc):
    return {
        "final_corrected_stats": {"ew": {"nmad": ew_nmad}, "ns": {"nmad": ns_nmad}},
        "raw_corr_stats": {"cc": {"cc_quality_gte_050": cc}},
    }


class TestComputePairWeights:
    """Tests for TIOInversion.compute_pair_weights (built via __new__, no binaries)."""

    def _inv(self, pairs):
        inv = TIOInversion.__new__(TIOInversion)  # bypass __init__ (no session/binaries)
        inv.pairs = pairs
        return inv

    def test_uniform_all_ones(self):
        inv = self._inv([_make_pair("a", 30), _make_pair("b", 60)])
        assert inv.compute_pair_weights("uniform") == [1.0, 1.0]

    def test_quality_orders_by_quality(self, monkeypatch):
        # good pair: low nmad, high cc; bad pair: high nmad, low cc; same dt.
        good = _make_pair("good", 30)
        bad = _make_pair("bad", 30)
        table = {"good": _stats(0.1, 0.9), "bad": _stats(0.9, 0.1)}
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: table[p.pa_key])

        inv = self._inv([good, bad])
        w = inv.compute_pair_weights("quality", combine="geomean")
        assert all(0.0 <= x <= 1.0 for x in w)
        assert w[0] > w[1]  # good pair weighted higher

    def test_quality_missing_stats_neutral(self, monkeypatch):
        p = _make_pair("p", 30)

        def _raise(_):
            raise FileNotFoundError

        monkeypatch.setattr(tio, "load_pair_stats", _raise)
        inv = self._inv([p])
        w = inv.compute_pair_weights("quality")
        # all components neutral (1.0) -> weight 1.0
        assert w == [pytest.approx(1.0)]

    def test_pair_quality_metrics_returns_four_lists(self, monkeypatch):
        pairs = [_make_pair("a", 30), _make_pair("b", 60)]
        table = {"a": _stats2(0.1, 0.4, 0.8), "b": _stats2(0.5, 0.2, 0.6)}
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: table[p.pa_key])
        inv = self._inv(pairs)
        ew, ns, cc, dt = inv._pair_quality_metrics()
        assert ew == [pytest.approx(0.1), pytest.approx(0.5)]
        assert ns == [pytest.approx(0.4), pytest.approx(0.2)]
        assert len(cc) == len(dt) == 2

    def test_direction_ew_vs_ns(self, monkeypatch):
        # p1: EW good (low nmad), NS bad; p2: EW bad, NS good. Same dt & cc.
        p1, p2 = _make_pair("p1", 40), _make_pair("p2", 40)
        table = {"p1": _stats2(0.1, 0.9, 0.7), "p2": _stats2(0.9, 0.1, 0.7)}
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: table[p.pa_key])
        inv = self._inv([p1, p2])

        w_ew = inv.compute_pair_weights("quality", direction="EW")
        w_ns = inv.compute_pair_weights("quality", direction="NS")
        # EW view favours p1 (its EW nmad is lowest); NS view favours p2.
        assert w_ew[0] > w_ew[1]
        assert w_ns[1] > w_ns[0]
        # the two directions genuinely differ
        assert w_ew != w_ns

    def test_recorded_mode_resolution(self, monkeypatch, tmp_path):
        """The label passed to the DB sync reflects what produced the weights."""
        pairs = [
            SimpleNamespace(pa_key="p1", pa_dt_days=30,
                            pa_left=SimpleNamespace(th_date="2022-01-01"),
                            pa_right=SimpleNamespace(th_date="2022-02-01")),
        ]
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: _stats(0.3, 0.6))
        inv = self._inv(pairs)
        inv.inversion_name = "t"
        inv.inversion_dir = tmp_path
        inv._DIRECTIONS = ("EW", "NS")
        (tmp_path / "inverse_EW").mkdir()
        (tmp_path / "inverse_NS").mkdir()

        captured = {}
        monkeypatch.setattr(
            inv, "_sync_pair_weights",
            lambda w_ew, w_ns, mode, combine: captured.update(mode=mode, combine=combine),
        )

        # (a) explicit weight_mode wins
        inv.write_liste_couple(weight_mode="quality", combine="geomean")
        assert captured == {"mode": "quality", "combine": "geomean"}

        # (b) weights from explore_weights → reuse its stashed mode/combine
        inv._last_weights = {"EW": [0.5], "NS": [0.5]}
        inv._last_weight_mode = "quality"
        inv._last_combine = "wmean"
        inv.write_liste_couple(weights=inv._last_weights)
        assert captured == {"mode": "quality", "combine": "wmean"}

        # (c) unknown external weights → honest "explicit" (not "uniform")
        inv.write_liste_couple(weights={"EW": [0.2], "NS": [0.8]})
        assert captured["mode"] == "explicit"

        # (d) no weights, no mode → uniform
        inv.write_liste_couple()
        assert captured["mode"] == "uniform"

    def test_all_modes_in_unit_range(self, monkeypatch):
        pairs = [
            SimpleNamespace(pa_key="p1", pa_dt_days=30,
                            pa_left=SimpleNamespace(th_date="2022-01-01"),
                            pa_right=SimpleNamespace(th_date="2022-02-01")),
            SimpleNamespace(pa_key="p2", pa_dt_days=200,
                            pa_left=SimpleNamespace(th_date="2022-01-01"),
                            pa_right=SimpleNamespace(th_date="2022-07-20")),
        ]
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: _stats(0.3, 0.6))
        inv = self._inv(pairs)
        for mode in ("uniform", "temporal", "relative_temporal", "sigmoid",
                     "parametric", "quality"):
            w = inv.compute_pair_weights(mode)
            assert len(w) == 2
            assert all(0.0 <= x <= 1.0 for x in w), f"{mode} produced out-of-range weight"
