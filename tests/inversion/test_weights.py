#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for TIO pair-weighting math and compute_pair_weights."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import geomulticorr.inversion.tio_inversion as tio
from geomulticorr.inversion.tio_inversion import (
    TIOInversion,
    combine_weights,
    combine_weights_spatial,
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


class TestCombineWeightsSpatial:
    """Tests for combine_weights_spatial() — two-term combination (NMAD + CC only)."""

    def test_geomean(self):
        assert combine_weights_spatial(0.8, 0.6, method="geomean") == pytest.approx(
            (0.8 * 0.6) ** 0.5
        )

    def test_product(self):
        assert combine_weights_spatial(0.8, 0.6, method="product") == pytest.approx(0.48)

    def test_wmean_equal_weights(self):
        assert combine_weights_spatial(0.8, 0.6, method="wmean") == pytest.approx(0.7)

    def test_wmean_renormalises(self):
        # weights (2,0) -> only nmad term counts
        assert combine_weights_spatial(0.4, 0.9, method="wmean",
                                       alpha=2, beta=0) == pytest.approx(0.4)

    def test_nan_component_is_neutral(self):
        # missing cc -> treated as 1.0
        assert combine_weights_spatial(0.5, float("nan"), method="product") == pytest.approx(0.5)

    def test_result_clamped_unit(self):
        for m in ("geomean", "wmean", "product"):
            w = combine_weights_spatial(1.0, 1.0, method=m)
            assert 0.0 <= w <= 1.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown combine method"):
            combine_weights_spatial(1.0, 1.0, method="nope")


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

    def test_quality_spatial_ignores_dt(self, monkeypatch):
        """Quality_spatial gives equal weight to pairs with identical NMAD/CC but different Δt."""
        # Two pairs: identical NMAD and CC, but very different temporal baselines.
        # quality mode should suppress the long-Δt pair; quality_spatial should treat them equally.
        short_dt = _make_pair("short", 30)     # short baseline
        long_dt = _make_pair("long", 300)      # long baseline
        table = {
            "short": _stats2(0.2, 0.2, 0.8),  # identical NMAD/CC
            "long": _stats2(0.2, 0.2, 0.8),
        }
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: table[p.pa_key])
        inv = self._inv([short_dt, long_dt])

        # quality mode: long Δt pair gets suppressed weight
        w_quality = inv.compute_pair_weights("quality", combine="geomean")
        assert w_quality[0] > w_quality[1], "quality should suppress long-Δt pair"

        # quality_spatial mode: identical NMAD/CC → equal weights
        w_spatial = inv.compute_pair_weights("quality_spatial", combine="geomean")
        assert w_spatial[0] == pytest.approx(w_spatial[1], rel=1e-6), \
            "quality_spatial should give equal weight despite different Δt"

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

        # (b) weights from explore_weights → reuse its stashed mode/combine.
        # The explorer now stashes one params dict; _last_weight_mode /
        # _last_combine are read-only deprecated views onto it, so setting them
        # would raise.
        inv._last_weights = {"EW": [0.5], "NS": [0.5]}
        inv._last_weights_params = {"weight_mode": "quality", "combine": "wmean"}
        inv.write_liste_couple(weights=inv._last_weights)
        assert captured == {"mode": "quality", "combine": "wmean"}

        # (b2) the same weights in a *different object* are still recognised —
        # identity was too strict, dict(inv._last_weights) used to fall through
        # to "explicit".
        inv.write_liste_couple(weights=dict(inv._last_weights))
        assert captured == {"mode": "quality", "combine": "wmean"}

        # (b3) numpy payloads must not raise. A bare `==` on a dict of ndarrays
        # returns an array, and `if <array>` raises "truth value of an array is
        # ambiguous" — aborting a write that works today.
        inv.write_liste_couple(
            weights={"EW": np.array([0.5]), "NS": np.array([0.5])}
        )
        assert captured == {"mode": "quality", "combine": "wmean"}

        # (c) unknown external weights → honest "explicit" (not "uniform")
        inv.write_liste_couple(weights={"EW": [0.2], "NS": [0.8]})
        assert captured["mode"] == "explicit"

        # (d) no weights, no mode → uniform
        inv.write_liste_couple()
        assert captured["mode"] == "uniform"

    def test_deprecated_weight_mode_views(self, monkeypatch):
        """The old loose attributes still read through, with a warning."""
        inv = TIOInversion.__new__(TIOInversion)
        inv._last_weights_params = {"weight_mode": "sigmoid", "combine": "geomean"}

        with pytest.warns(DeprecationWarning, match="_last_weights_params"):
            assert inv._last_weight_mode == "sigmoid"
        with pytest.warns(DeprecationWarning, match="_last_weights_params"):
            assert inv._last_combine == "geomean"

    def test_deprecated_views_are_read_only(self):
        """Assignment must fail loudly rather than create a second source of truth.

        A getter-only ``property`` is a data descriptor, so it shadows the
        instance ``__dict__`` — this is what the assignment actually hits.
        """
        inv = TIOInversion.__new__(TIOInversion)
        with pytest.raises(AttributeError):
            inv._last_weight_mode = "quality"
        with pytest.raises(AttributeError):
            inv._last_combine = "wmean"

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
                     "parametric", "quality", "quality_spatial"):
            w = inv.compute_pair_weights(mode)
            assert len(w) == 2
            assert all(0.0 <= x <= 1.0 for x in w), f"{mode} produced out-of-range weight"


# ──────────────────────────────────────────────────────────────────────────────
# sensor_weights — the thirteenth key
# ──────────────────────────────────────────────────────────────────────────────

from geomulticorr.inversion.tio_inversion import (  # noqa: E402
    _pair_sensor_label,
    _validate_sensor_weights,
    sensor_weight_factor,
)


def _sensor_pair(key, dt_days, left, right=None):
    from datetime import date, timedelta
    right = left if right is None else right
    d2 = (date(2022, 1, 1) + timedelta(days=int(dt_days))).isoformat()
    return SimpleNamespace(
        pa_key=key, pa_dt_days=dt_days,
        pa_left=SimpleNamespace(th_date="2022-01-01", th_sensor=left),
        pa_right=SimpleNamespace(th_date=d2, th_sensor=right),
    )


class TestSensorWeightFactor:
    def test_no_mapping_is_neutral(self):
        assert sensor_weight_factor(("spot6", "spot6"), None) == 1.0
        assert sensor_weight_factor(("spot6", "spot6"), {}) == 1.0

    def test_case_insensitive_substring_match(self):
        assert sensor_weight_factor(("SPOT6", "Spot7"), {"spot": 0.5}) == 0.5

    def test_longest_key_wins(self):
        sw = {"spot": 0.5, "spot7": 0.8}
        assert sensor_weight_factor(("spot7", "spot7"), sw) == 0.8
        assert sensor_weight_factor(("spot6", "spot6"), sw) == 0.5

    def test_mixed_pair_takes_the_min(self):
        sw = {"spot": 1.0, "planetscope": 0.3}
        assert sensor_weight_factor(("spot6", "planetscope"), sw) == 0.3

    def test_unmatched_sensor_keeps_one(self):
        assert sensor_weight_factor(("pleiades", "pleiades"), {"spot": 0.5}) == 1.0
        assert sensor_weight_factor(("", ""), {"spot": 0.5}) == 1.0

    def test_validation_normalises_and_rejects(self):
        assert _validate_sensor_weights(None) is None
        assert _validate_sensor_weights({}) is None
        assert _validate_sensor_weights({" SPOT ": "0.5"}) == {"spot": 0.5}
        with pytest.raises(TypeError, match="mapping"):
            _validate_sensor_weights(["spot"])
        with pytest.raises(ValueError, match="non-empty"):
            _validate_sensor_weights({"": 1.0})
        with pytest.raises(ValueError, match="number"):
            _validate_sensor_weights({"spot": "abc"})
        with pytest.raises(ValueError, match="finite"):
            _validate_sensor_weights({"spot": -1.0})

    def test_pair_sensor_label(self):
        assert _pair_sensor_label(_sensor_pair("a", 1, "SPOT6")) == "spot6"
        assert _pair_sensor_label(_sensor_pair("a", 1, "spot6", "planetscope")) == "spot6+planetscope"
        assert _pair_sensor_label(_make_pair("a", 1)) == ""


class TestSensorWeightsInComputePairWeights:
    def _inv(self, pairs):
        inv = TIOInversion.__new__(TIOInversion)
        inv.pairs = pairs
        inv._quality_metrics = None
        return inv

    def test_uniform_times_sensor_factor(self):
        inv = self._inv([_sensor_pair("a", 30, "spot6"), _sensor_pair("b", 30, "planetscope")])
        assert inv.compute_pair_weights("uniform", sensor_weights={"planetscope": 0.5}) == [1.0, 0.5]

    def test_applied_after_the_w_min_floor(self):
        """The floor is per mode; the sensor factor is a deliberate exception to it."""
        pairs = [_sensor_pair("a", 30, "spot6"), _sensor_pair("b", 400, "spot6"),
                 _sensor_pair("c", 30, "planetscope")]
        inv = self._inv(pairs)
        plain = inv.compute_pair_weights("sigmoid", w_min=0.2)
        with_sw = inv.compute_pair_weights("sigmoid", w_min=0.2, sensor_weights={"spot": 0.5})
        assert with_sw[0] == pytest.approx(0.5 * plain[0])
        assert with_sw[1] == pytest.approx(0.5 * plain[1])
        assert with_sw[2] == pytest.approx(plain[2])
        assert plain[1] >= 0.2                        # the floor held before the factor
        assert with_sw[1] < 0.2                       # and is undercut by it, by design

    def test_applies_to_quality_modes_too(self, monkeypatch):
        good = _sensor_pair("good", 30, "spot6")
        bad = _sensor_pair("bad", 30, "planetscope")
        table = {"good": _stats(0.1, 0.9), "bad": _stats(0.9, 0.1)}
        monkeypatch.setattr(tio, "load_pair_stats", lambda p: table[p.pa_key])
        inv = self._inv([good, bad])
        for mode in ("quality", "quality_spatial"):
            inv._quality_metrics = None
            plain = inv.compute_pair_weights(mode)
            with_sw = inv.compute_pair_weights(mode, sensor_weights={"spot": 0.25})
            assert with_sw[0] == pytest.approx(0.25 * plain[0])
            assert with_sw[1] == pytest.approx(plain[1])

    def test_pairs_without_sensor_attribute_are_fine_when_unused(self):
        inv = self._inv([_make_pair("a", 30), _make_pair("b", 60)])
        assert inv.compute_pair_weights("uniform", sensor_weights=None) == [1.0, 1.0]

    def test_pairs_without_sensor_attribute_are_unmatched_when_used(self):
        inv = self._inv([_make_pair("a", 30), _sensor_pair("b", 60, "spot6")])
        assert inv.compute_pair_weights("uniform", sensor_weights={"spot": 0.5}) == [1.0, 0.5]

    def test_write_liste_couple_threads_it_through(self, tmp_path):
        inv = self._inv([_sensor_pair("a", 30, "spot6"), _sensor_pair("b", 30, "planetscope")])
        inv.inversion_dir = tmp_path
        inv._DIRECTIONS = ("EW", "NS")
        inv.solver = tio.TIOConfig()
        for d in inv._DIRECTIONS:
            (tmp_path / f"inverse_{d}").mkdir()
        out = inv.write_liste_couple(sensor_weights={"planetscope": 0.5}, sync_geodb=False)
        assert out["EW"] == [1.0, 0.5]
        text = (tmp_path / "inverse_EW" / "liste_couple").read_text().splitlines()
        assert text[1].endswith(" 0.500000")

    def test_explicit_vectors_ignore_it_with_a_warning(self, tmp_path, caplog_gmc):
        inv = self._inv([_sensor_pair("a", 30, "spot6"), _sensor_pair("b", 30, "planetscope")])
        inv.inversion_dir = tmp_path
        inv._DIRECTIONS = ("EW", "NS")
        inv.solver = tio.TIOConfig()
        for d in inv._DIRECTIONS:
            (tmp_path / f"inverse_{d}").mkdir()
        out = inv.write_liste_couple(weights=[0.9, 0.9], sensor_weights={"planetscope": 0.5},
                                     sync_geodb=False)
        assert out["EW"] == [0.9, 0.9]
        assert "sensor_weights ignored" in caplog_gmc.text
