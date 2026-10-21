#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the pure correlation-parameter physics (corr_params).

The numeric expectations here are taken from the worked examples in
``docs/correlation_parameters.md``.  They are deliberately bit-for-bit: a silent
drift in these formulas moves every derived parameter set and every published
design map without failing anything else.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from geomulticorr.correlation.corr_params import (
    DAYS_PER_YEAR,
    DEFAULT_COST_BUDGET,
    SGM_ALGORITHMS,
    SGM_COST_MODE,
    SGM_KERNEL,
    STRAIN_TOLERANCE_PX,
    TEXTURE_FLOOR_PX,
    assign_dt_bin,
    coherence,
    cost_index,
    days_to_years,
    displacement_px,
    dynamic_range,
    expected_displacement,
    feasibility_bounds,
    footprint_m,
    format_dt_span,
    geometric_dt_bins,
    kernel_from_footprint,
    max_measurable_velocity,
    min_detectable_velocity,
    n_parameter_sets,
    noise_floor_m,
    optimal_baseline_days,
    required_search_px,
    search_half_width_px,
    sgm_kernel_override,
    snr,
    strain_kernel_limit_px,
    parse_search_box,
    suggest_parameters,
    to_odd,
    uniform_conflicts,
)

# The guide's worked site: PlanetScope at 3 m with a measured 0.5 m stable-ground
# NMAD, k = 3.  c follows as sigma / r.
GUIDE_RES = 3.0
GUIDE_SIGMA = 0.5
GUIDE_C = GUIDE_SIGMA / GUIDE_RES
GUIDE_K = 3.0


class TestSmallHelpers:
    def test_days_to_years_uses_julian_year(self):
        assert days_to_years(DAYS_PER_YEAR) == pytest.approx(1.0)
        assert DAYS_PER_YEAR == 365.25

    def test_days_to_years_vectorises(self):
        out = days_to_years(np.array([365.25, 730.5]))
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, [1.0, 2.0])

    @pytest.mark.parametrize(
        "value, expected",
        [(20.0, 21), (21.0, 21), (21.4, 21), (22.0, 23), (28.0, 29), (7.0, 7)],
    )
    def test_to_odd_rounds_up_to_odd(self, value, expected):
        """Ties and evens go *up*, so a footprint is never quietly smaller."""
        assert to_odd(value) == expected

    def test_to_odd_honours_minimum_and_makes_it_odd(self):
        assert to_odd(3.0, minimum=7) == 7
        assert to_odd(3.0, minimum=8) == 9

    def test_to_odd_survives_non_finite(self):
        assert to_odd(float("nan"), minimum=7) == 7


class TestExpectedDisplacement:
    def test_product_of_velocity_and_baseline(self):
        assert expected_displacement(5.0, 4 * DAYS_PER_YEAR) == pytest.approx(20.0)

    def test_same_product_gives_same_displacement(self):
        """Two pairs with equal v*dt pose the same problem (guide §1)."""
        assert expected_displacement(1.0, 4 * DAYS_PER_YEAR) == pytest.approx(
            expected_displacement(4.0, DAYS_PER_YEAR)
        )

    def test_displacement_px_divides_by_resolution(self):
        assert displacement_px(5.0, 4 * DAYS_PER_YEAR, 3.0) == pytest.approx(20.0 / 3.0)

    def test_displacement_px_rejects_bad_resolution(self):
        with pytest.raises(ValueError, match="resolution_m must be positive"):
            displacement_px(1.0, 100.0, 0.0)


class TestNoiseFloor:
    def test_noise_floor_is_c_times_r(self):
        assert noise_floor_m(GUIDE_RES, GUIDE_C) == pytest.approx(GUIDE_SIGMA)

    def test_finer_resolution_lowers_the_floor_linearly(self):
        """The dominant reason to want finer imagery (guide §4)."""
        assert noise_floor_m(1.5, 0.15) == pytest.approx(noise_floor_m(3.0, 0.15) / 2.0)

    def test_kernel_scaling_is_saturating_and_optional(self):
        base = noise_floor_m(3.0, 0.15)
        assert noise_floor_m(3.0, 0.15, kernel_px=21) == pytest.approx(base)
        bigger = noise_floor_m(3.0, 0.15, kernel_px=84)
        assert bigger == pytest.approx(base / 2.0)  # 4x kernel -> sqrt(4) = 2x better

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_inputs(self, bad):
        with pytest.raises(ValueError):
            noise_floor_m(bad, 0.15)
        with pytest.raises(ValueError):
            noise_floor_m(3.0, bad)


class TestDetectionFloor:
    @pytest.mark.parametrize(
        "velocity, expected_days",
        [(1.0, 1.5 * DAYS_PER_YEAR), (2.0, 0.75 * DAYS_PER_YEAR), (5.0, 0.3 * DAYS_PER_YEAR)],
    )
    def test_guide_table_of_minimum_useful_baselines(self, velocity, expected_days):
        """Guide §2.4: on 3 m imagery a 1 m/yr landform needs ~18 months."""
        assert snr(velocity, expected_days, GUIDE_RES, GUIDE_C) == pytest.approx(GUIDE_K)

    def test_min_detectable_velocity_is_inverse_in_baseline(self):
        v1 = min_detectable_velocity(365.25, GUIDE_RES, GUIDE_C, GUIDE_K)
        v2 = min_detectable_velocity(730.5, GUIDE_RES, GUIDE_C, GUIDE_K)
        assert v1 == pytest.approx(2.0 * v2)
        assert v1 == pytest.approx(GUIDE_K * GUIDE_SIGMA)

    def test_min_detectable_velocity_vectorises(self):
        out = min_detectable_velocity(np.array([365.25, 730.5]), GUIDE_RES, GUIDE_C, GUIDE_K)
        assert isinstance(out, np.ndarray)
        assert out[0] == pytest.approx(2.0 * out[1])

    def test_snr_below_k_is_the_noise_regime(self):
        """Guide §7 case 1: v=1 m/yr over one month is noise, not a noisy signal."""
        assert snr(1.0, 30.0, GUIDE_RES, GUIDE_C) == pytest.approx(0.164, abs=1e-3)

    def test_snr_rejects_zero_precision_rather_than_dividing_by_zero(self):
        """A zero noise floor is unreachable: noise_floor_m validates c_px."""
        with pytest.raises(ValueError, match="c_px must be positive"):
            snr(1.0, 365.25, 3.0, 0.0)

    def test_snr_grows_without_bound_as_precision_improves(self):
        assert snr(1.0, 365.25, 3.0, 1e-6) > snr(1.0, 365.25, 3.0, 1e-3)


class TestSearchRange:
    def test_symmetric_box_covers_displacement_plus_allowances(self):
        """Guide §2.4: 5 m/yr over 4 yr at 3 m is 6.7 px, +2 coreg +2 margin."""
        box = required_search_px(5.0, 4 * DAYS_PER_YEAR, GUIDE_RES, coreg_px=2, margin_px=2)
        assert box == (-11, -11, 11, 11)
        assert search_half_width_px(box) == 11.0

    def test_box_always_reaches_at_least_the_displacement(self):
        for dt in (30.0, 365.25, 1461.0):
            box = required_search_px(5.0, dt, GUIDE_RES, coreg_px=2, margin_px=2)
            assert search_half_width_px(box) >= displacement_px(5.0, dt, GUIDE_RES)

    def test_asymmetric_box_is_elongated_along_flow(self):
        """Guide §6.1 — the single largest cost saving available."""
        box = required_search_px(
            5.0, 4 * DAYS_PER_YEAR, GUIDE_RES, coreg_px=2, margin_px=2, flow_azimuth_deg=90.0
        )
        x_lo, y_lo, x_hi, y_hi = box
        assert x_hi > 6  # eastward reach covers the 6.7 px displacement
        assert (y_hi - y_lo) < (x_hi - x_lo)  # narrower across flow

    @pytest.mark.parametrize(
        "azimuth, axis, sign",
        [(0.0, "y", -1), (90.0, "x", +1), (180.0, "y", +1), (270.0, "x", -1)],
    )
    def test_azimuth_maps_compass_to_raster_axes(self, azimuth, axis, sign):
        """North is -y (rows grow southward); east is +x."""
        x_lo, y_lo, x_hi, y_hi = required_search_px(
            5.0, 4 * DAYS_PER_YEAR, GUIDE_RES, flow_azimuth_deg=azimuth
        )
        reach = {"x": (x_lo, x_hi), "y": (y_lo, y_hi)}[axis]
        assert (reach[1] if sign > 0 else -reach[0]) > 6

    def test_asymmetric_box_is_much_cheaper(self):
        sym = required_search_px(20.0, 4 * DAYS_PER_YEAR, GUIDE_RES)
        asym = required_search_px(20.0, 4 * DAYS_PER_YEAR, GUIDE_RES, flow_azimuth_deg=90.0)
        assert cost_index(21, asym) < cost_index(21, sym) / 5.0

    def test_search_half_width_of_auto_is_nan(self):
        assert math.isnan(search_half_width_px(None))

    def test_max_measurable_velocity_is_the_ceiling_line(self):
        v = max_measurable_velocity(4 * DAYS_PER_YEAR, GUIDE_RES, 11.0, coreg_px=2, margin_px=2)
        assert v == pytest.approx(7 * GUIDE_RES / 4.0)

    def test_max_measurable_velocity_never_negative(self):
        assert max_measurable_velocity(365.25, 3.0, 1.0, coreg_px=2, margin_px=2) == 0.0


class TestKernelSizing:
    @pytest.mark.parametrize(
        "resolution, expected", [(1.5, 41), (3.0, 21), (10.0, 7)]
    )
    def test_matched_ground_footprint_across_sensors(self, resolution, expected):
        """Guide §4.2: a 60 m template is a different pixel count per sensor."""
        assert kernel_from_footprint(60.0, resolution) == expected

    def test_fixed_pixel_kernel_is_a_different_physical_filter(self):
        """The correctness argument: 21 px is 31.5 m on SPOT, 63 m on Planet."""
        assert footprint_m(21, 1.5) == pytest.approx(31.5)
        assert footprint_m(21, 3.0) == pytest.approx(63.0)

    def test_texture_floor_clamps_coarse_sensors(self):
        assert kernel_from_footprint(60.0, 30.0) == TEXTURE_FLOOR_PX

    def test_k_max_never_exceeded(self):
        assert kernel_from_footprint(60.0, 1.5, k_max=12.5) <= 12.5
        assert kernel_from_footprint(60.0, 1.5, k_max=12.5) == 11

    def test_k_max_below_texture_floor_still_returns_the_floor(self):
        assert kernel_from_footprint(60.0, 1.5, k_max=2.5) == TEXTURE_FLOOR_PX

    @pytest.mark.parametrize("bad", [0.0, -3.0])
    def test_rejects_non_positive_inputs(self, bad):
        with pytest.raises(ValueError):
            kernel_from_footprint(60.0, bad)
        with pytest.raises(ValueError):
            kernel_from_footprint(bad, 3.0)


class TestStrainCeiling:
    @pytest.mark.parametrize(
        "gradient, dt_years, expected",
        [(0.02, 1.0, 50.0), (0.02, 4.0, 12.5), (0.1, 4.0, 2.5)],
    )
    def test_guide_worked_strain_limits(self, gradient, dt_years, expected):
        assert strain_kernel_limit_px(gradient, dt_years * DAYS_PER_YEAR) == pytest.approx(expected)

    def test_longer_baselines_want_smaller_kernels(self):
        """The counterintuitive result of guide §3.1, stated as a test."""
        short = strain_kernel_limit_px(0.02, DAYS_PER_YEAR)
        long = strain_kernel_limit_px(0.02, 4 * DAYS_PER_YEAR)
        assert long < short

    def test_zero_strain_is_unbounded(self):
        assert strain_kernel_limit_px(0.0, 1000.0) == math.inf

    def test_uses_the_documented_tolerance(self):
        assert STRAIN_TOLERANCE_PX == 1.0
        assert strain_kernel_limit_px(1.0, DAYS_PER_YEAR) == pytest.approx(1.0)


class TestDynamicRange:
    def test_identity_is_search_over_k_times_c(self):
        assert dynamic_range(20.0, 0.10, 3.0) == pytest.approx(20.0 / 0.30)
        assert dynamic_range(20.0, 0.10, 3.0) == pytest.approx(66.67, abs=0.01)

    def test_guide_worked_value(self):
        assert dynamic_range(12.0, GUIDE_C, GUIDE_K) == pytest.approx(24.0)

    def test_independent_of_resolution_and_baseline(self):
        """The whole point: the band width cancels r and dt (guide §2.2)."""
        a = dynamic_range(12.0, 0.15, 3.0)
        b = dynamic_range(12.0, 0.15, 3.0)
        assert a == b  # no resolution/baseline argument exists to vary

    def test_zero_precision_is_unbounded(self):
        assert dynamic_range(12.0, 0.0, 3.0) == math.inf

    def test_n_sets_for_the_full_archive(self):
        """Guide §2.4: 30 d to 10 yr needs two sets."""
        assert n_parameter_sets(30.0, 10 * DAYS_PER_YEAR, 12.0, GUIDE_C, GUIDE_K) == 2

    def test_pruning_below_the_floor_collapses_to_one_set(self):
        """The operational punchline of guide §2.4."""
        assert n_parameter_sets(550.0, 3650.0, 12.0, GUIDE_C, GUIDE_K) == 1

    def test_velocity_ratio_widens_the_span(self):
        narrow = n_parameter_sets(550.0, 3650.0, 12.0, GUIDE_C, GUIDE_K)
        wide = n_parameter_sets(550.0, 3650.0, 12.0, GUIDE_C, GUIDE_K, velocity_ratio=50.0)
        assert wide > narrow

    def test_never_below_one(self):
        assert n_parameter_sets(365.0, 365.0, 12.0, GUIDE_C, GUIDE_K) == 1

    def test_rejects_non_positive_baselines(self):
        with pytest.raises(ValueError, match="must be positive"):
            n_parameter_sets(0.0, 365.0, 12.0, 0.15, 3.0)


class TestDecorrelation:
    def test_coherence_decays_exponentially(self):
        assert coherence(0.0, 400.0) == pytest.approx(1.0)
        assert coherence(400.0, 400.0) == pytest.approx(math.exp(-1.0))

    def test_seasonal_term_dips_at_half_year_and_recovers_at_anniversary(self):
        """An anniversary pair beats a 6-month one despite being longer."""
        half = coherence(DAYS_PER_YEAR / 2, 4000.0, seasonal_amplitude=0.6)
        anniversary = coherence(DAYS_PER_YEAR, 4000.0, seasonal_amplitude=0.6)
        assert anniversary > half

    def test_coherence_vectorises_and_stays_non_negative(self):
        out = coherence(np.array([0.0, 400.0, 4000.0]), 400.0, seasonal_amplitude=1.0)
        assert isinstance(out, np.ndarray)
        assert (out >= 0).all()

    def test_optimal_baseline_is_one_decorrelation_time(self):
        assert optimal_baseline_days(400.0) == 400.0

    def test_optimal_baseline_maximises_modelled_snr(self):
        """Verify dt = tau really is the argmax of v*dt/exp(dt/tau)."""
        tau = 400.0
        grid = np.linspace(10.0, 2000.0, 4000)
        modelled = grid * np.exp(-grid / tau)
        assert grid[int(np.argmax(modelled))] == pytest.approx(tau, abs=1.0)

    @pytest.mark.parametrize("bad", [0.0, -5.0])
    def test_rejects_non_positive_tau(self, bad):
        with pytest.raises(ValueError, match="tau_days must be positive"):
            coherence(100.0, bad)
        with pytest.raises(ValueError, match="tau_days must be positive"):
            optimal_baseline_days(bad)


class TestCostIndex:
    def test_asp_defaults_are_unity(self):
        assert cost_index(21, 5.0) == pytest.approx(1.0)

    def test_doubling_search_quadruples_cost(self):
        """Guide §6.3, the warning that keeps search boxes honest."""
        assert cost_index(21, 10.0) == pytest.approx(4.0 * cost_index(21, 5.0))

    def test_doubling_kernel_quadruples_cost(self):
        assert cost_index(42, 5.0) == pytest.approx(4.0 * cost_index(21, 5.0))

    def test_guide_sixteen_fold_example(self):
        """K 21->41, S 12->24 is roughly a factor of 16."""
        ratio = cost_index(41, 24.0) / cost_index(21, 12.0)
        assert ratio == pytest.approx(15.25, abs=0.5)

    def test_accepts_tuple_kernel_and_box(self):
        assert cost_index((21, 21), (-5, -5, 5, 5)) == pytest.approx(1.0)

    def test_auto_search_charged_at_reference(self):
        assert cost_index(21, None) == pytest.approx(1.0)

    def test_sgm_ignores_requested_kernel(self):
        assert cost_index(41, 5.0, "asp_mgm") == cost_index(21, 5.0, "asp_mgm")


class TestGeometricBins:
    def test_edges_span_the_data_geometrically(self):
        edges, labels = geometric_dt_bins([30, 90, 400, 2000], ratio=3.0)
        assert edges[0] == 30.0
        assert edges[-1] >= 2000.0
        assert len(labels) == len(edges) - 1
        for lo, hi in zip(edges, edges[1:]):
            assert hi == pytest.approx(lo * 3.0)

    def test_every_baseline_lands_in_a_bin(self):
        values = [30, 90, 400, 2000]
        edges, labels = geometric_dt_bins(values, ratio=3.0)
        for v in values:
            assert assign_dt_bin(v, edges, labels) in labels

    def test_longest_baseline_is_not_orphaned(self):
        edges, labels = geometric_dt_bins([100.0, 300.0], ratio=3.0)
        assert assign_dt_bin(300.0, edges, labels) == labels[-1]

    def test_single_distinct_baseline_gets_one_bin(self):
        edges, labels = geometric_dt_bins([365.0, 365.0], ratio=3.0)
        assert len(labels) == 1
        assert assign_dt_bin(365.0, edges, labels) == labels[0]

    def test_empty_input_is_empty_not_an_error(self):
        assert geometric_dt_bins([]) == ([], [])
        assert assign_dt_bin(100.0, [], []) == ""

    def test_ignores_non_finite_and_non_positive(self):
        edges, labels = geometric_dt_bins([np.nan, -5.0, 100.0, 300.0], ratio=3.0)
        assert edges[0] == 100.0

    def test_labels_are_deterministic(self):
        a, _ = geometric_dt_bins([30, 2000], ratio=3.0)
        b, _ = geometric_dt_bins([30, 2000], ratio=3.0)
        assert a == b

    def test_rejects_ratio_at_or_below_one(self):
        with pytest.raises(ValueError, match="ratio must exceed 1"):
            geometric_dt_bins([30, 90], ratio=1.0)


class TestFeasibilityBounds:
    def test_both_curves_are_slope_minus_one(self):
        dt = np.array([100.0, 200.0, 400.0])
        b = feasibility_bounds(dt, GUIDE_RES, GUIDE_C, 12.0, k=GUIDE_K)
        np.testing.assert_allclose(b["v_min"] * dt, b["v_min"][0] * dt[0])
        np.testing.assert_allclose(b["v_max"] * dt, b["v_max"][0] * dt[0])

    def test_band_width_equals_dynamic_range(self):
        """The two boundaries are parallel; their ratio is S/(k*c)."""
        dt = np.array([100.0, 1000.0])
        b = feasibility_bounds(dt, GUIDE_RES, GUIDE_C, 12.0, k=GUIDE_K, coreg_px=0, margin_px=0)
        np.testing.assert_allclose(
            b["v_max"] / b["v_min"], dynamic_range(12.0, GUIDE_C, GUIDE_K)
        )

    def test_returns_float_arrays(self):
        b = feasibility_bounds([100.0, 200.0], GUIDE_RES, GUIDE_C, 12.0)
        for key in ("dt_days", "v_min", "v_max"):
            assert isinstance(b[key], np.ndarray)
            assert b[key].dtype == np.dtype("float64")


class TestSgmOverride:
    @pytest.mark.parametrize("algorithm", SGM_ALGORITHMS)
    def test_sgm_forces_kernel_and_cost_mode(self, algorithm):
        kernel, subpixel, cost, overridden = sgm_kernel_override(algorithm, (41, 41), (41, 41), 2)
        assert kernel == SGM_KERNEL
        assert subpixel == SGM_KERNEL
        assert cost == SGM_COST_MODE
        assert overridden is True

    def test_block_matching_passes_through_untouched(self):
        kernel, subpixel, cost, overridden = sgm_kernel_override("asp_bm", (41, 41), (35, 35), 2)
        assert (kernel, subpixel, cost, overridden) == ((41, 41), (35, 35), 2, False)


class TestSuggestParameters:
    def _good(self, **kwargs):
        """A pair comfortably inside the feasible band."""
        params = dict(
            resolution_m=GUIDE_RES,
            dt_days=3 * DAYS_PER_YEAR,
            velocity_m_yr=2.0,
            footprint_m=60.0,
            c_px=GUIDE_C,
            k=GUIDE_K,
        )
        params.update(kwargs)
        return suggest_parameters(**params)

    def test_returns_asp_ready_parameters(self):
        out = self._good()
        assert out["corr_kernel"] == (21, 21)
        assert len(out["corr_search"]) == 4
        assert out["subpixel_mode"] == 2
        assert out["corr_algorithm"] == "asp_bm"

    def test_explicit_search_sets_seed_mode_zero(self):
        """Guide §6.1 — nothing left for the low-res disparity stage to estimate."""
        assert self._good()["corr_seed_mode"] == 0

    def test_a_healthy_pair_warns_about_nothing(self):
        assert self._good()["warnings"] == []

    def test_diagnostics_are_self_consistent(self):
        out = self._good()
        assert out["disp_m"] == pytest.approx(expected_displacement(2.0, 3 * DAYS_PER_YEAR))
        assert out["disp_px"] == pytest.approx(out["disp_m"] / GUIDE_RES)
        assert out["snr"] == pytest.approx(out["disp_m"] / out["noise_floor_m"])
        assert out["footprint_m"] == pytest.approx(21 * GUIDE_RES)
        assert out["search_px"] == search_half_width_px(out["corr_search"])
        assert out["cost_index"] == pytest.approx(
            cost_index(out["corr_kernel"], out["corr_search"], "asp_bm")
        )

    # --- the warning paths, one each ------------------------------------- #
    def test_warns_below_the_texture_floor(self):
        out = self._good(resolution_m=30.0)
        assert any("texture floor" in w for w in out["warnings"])
        assert out["corr_kernel"] == (TEXTURE_FLOOR_PX, TEXTURE_FLOOR_PX)

    def test_warns_and_shrinks_at_the_strain_ceiling(self):
        out = self._good(dt_days=4 * DAYS_PER_YEAR, strain_rate_per_yr=0.02)
        assert any("strain ceiling" in w for w in out["warnings"])
        assert max(out["corr_kernel"]) <= 12.5

    def test_warns_when_strain_ceiling_is_under_the_texture_floor(self):
        """Guide §3.1: that margin cannot be correlated at that baseline at all."""
        out = self._good(dt_days=4 * DAYS_PER_YEAR, strain_rate_per_yr=0.1)
        assert any("below the texture floor" in w for w in out["warnings"])

    def test_warns_below_the_detection_floor(self):
        out = self._good(dt_days=30.0, velocity_m_yr=1.0)
        assert any("below the detection floor" in w or "detection floor" in w for w in out["warnings"])
        assert out["snr"] < GUIDE_K

    def test_warns_over_the_cost_budget(self):
        out = self._good(velocity_m_yr=200.0)
        assert any("exceeds the budget" in w for w in out["warnings"])

    def test_warns_on_the_sgm_override(self):
        out = self._good(corr_algorithm="asp_mgm")
        assert any("forces corr-kernel" in w for w in out["warnings"])
        assert out["corr_kernel"] == SGM_KERNEL
        assert out["cost_mode"] == SGM_COST_MODE

    def test_warns_beyond_twice_the_decorrelation_time(self):
        out = self._good(tau_days=200.0)
        assert any("decorrelation time" in w for w in out["warnings"])
        assert 0.0 <= out["coherence"] <= 1.0

    def test_coherence_is_nan_without_tau(self):
        assert math.isnan(self._good()["coherence"])

    def test_strain_limit_is_inf_without_a_strain_rate(self):
        assert self._good()["strain_limit_px"] == math.inf

    def test_default_cost_budget_is_exposed(self):
        assert DEFAULT_COST_BUDGET > 1.0

    def test_asymmetric_box_when_flow_is_known(self):
        out = self._good(flow_azimuth_deg=90.0)
        x_lo, y_lo, x_hi, y_hi = out["corr_search"]
        assert (x_hi - x_lo) > (y_hi - y_lo)

    def test_warnings_are_strings(self):
        out = self._good(resolution_m=30.0, dt_days=30.0, velocity_m_yr=0.1)
        assert out["warnings"]
        assert all(isinstance(w, str) for w in out["warnings"])


class TestNaNResolutionDegrades:
    """``_thumb_resolution`` returns nan for an unreadable header on purpose.

    One bad thumb must not abort the whole derivation with an opaque
    "cannot convert float NaN to integer".
    """

    def test_search_box_does_not_raise_on_nan_resolution(self):
        box = required_search_px(1.0, 30.0, float("nan"), coreg_px=2, margin_px=2)
        assert len(box) == 4
        assert all(isinstance(v, int) for v in box)

    def test_nan_box_still_covers_the_allowances(self):
        box = required_search_px(1.0, 30.0, float("nan"), coreg_px=3, margin_px=2)
        assert search_half_width_px(box) >= 5

    def test_nan_box_is_never_degenerate(self):
        box = required_search_px(1.0, 30.0, float("nan"), coreg_px=0, margin_px=0)
        assert search_half_width_px(box) >= 1

    def test_asymmetric_path_also_survives_nan(self):
        box = required_search_px(1.0, 30.0, float("nan"), flow_azimuth_deg=90.0)
        assert len(box) == 4

    def test_suggest_parameters_survives_nan_resolution(self):
        out = suggest_parameters(float("nan"), 400.0, 2.0)
        assert len(out["corr_search"]) == 4


class TestConditionalSearch:
    """An explicit box is not unconditionally better than ASP's automatic range.

    ASP derives that range from interest-point matches, which are most reliable
    exactly where displacement is small. The failure modes are asymmetric: auto
    failing costs runtime, an explicit box failing costs correctness, because a
    low velocity guess clips real motion.
    """

    def _at(self, dt_days, **kwargs):
        params = dict(resolution_m=3.0, dt_days=dt_days, velocity_m_yr=5.0)
        params.update(kwargs)
        return suggest_parameters(**params)

    def test_below_the_threshold_leaves_the_range_to_asp(self):
        out = self._at(30.0, explicit_search_above_days=365)
        assert out["corr_search"] is None

    def test_below_the_threshold_keeps_the_low_res_disparity_stage(self):
        """seed-mode 1, not 0: that stage IS what produces ASP's estimate."""
        assert self._at(30.0, explicit_search_above_days=365)["corr_seed_mode"] == 1

    def test_above_the_threshold_derives_a_box(self):
        out = self._at(800.0, explicit_search_above_days=365)
        assert out["corr_search"] is not None
        assert out["corr_seed_mode"] == 0

    def test_exactly_at_the_threshold_is_explicit(self):
        assert self._at(365.0, explicit_search_above_days=365)["corr_search"] is not None

    def test_zero_threshold_makes_every_pair_explicit(self):
        assert self._at(1.0, explicit_search_above_days=0)["corr_search"] is not None

    def test_the_checkbox_forces_auto_at_any_baseline(self):
        out = self._at(5000.0, explicit_search=False, explicit_search_above_days=0)
        assert out["corr_search"] is None
        assert out["corr_seed_mode"] == 1

    def test_seed_mode_and_box_always_move_together(self):
        for dt in (10.0, 200.0, 365.0, 2000.0):
            out = self._at(dt, explicit_search_above_days=365)
            has_box = out["corr_search"] is not None
            assert out["corr_seed_mode"] == (0 if has_box else 1)

    def test_search_px_is_nan_when_left_to_asp(self):
        assert math.isnan(self._at(30.0, explicit_search_above_days=365)["search_px"])

    def test_auto_is_charged_at_the_reference_cost(self):
        out = self._at(30.0, explicit_search_above_days=365)
        assert out["cost_index"] == pytest.approx(cost_index(out["corr_kernel"], None))

    def test_default_is_explicit_for_every_pair(self):
        """Unchanged behaviour for callers that pass nothing."""
        assert self._at(30.0)["corr_search"] is not None


class TestParseSearchBox:
    def test_one_number_is_symmetric(self):
        assert parse_search_box("20") == (-20, -20, 20, 20)

    def test_four_numbers_are_taken_verbatim(self):
        assert parse_search_box("-80 -2 20 2") == (-80, -2, 20, 2)

    def test_commas_separate_too(self):
        assert parse_search_box("-80, -2, 20, 2") == (-80, -2, 20, 2)

    def test_a_sequence_is_accepted(self):
        assert parse_search_box((-5, -5, 5, 5)) == (-5, -5, 5, 5)

    def test_the_asymmetric_form_is_far_cheaper(self):
        """Guide §6.1 — the largest cost saving available."""
        sym = parse_search_box("80")
        asym = parse_search_box("-80 -2 20 2")
        assert cost_index(21, asym) < cost_index(21, sym) / 20

    def test_negative_single_number_is_read_as_a_half_width(self):
        assert parse_search_box("-20") == (-20, -20, 20, 20)

    @pytest.mark.parametrize("bad", ["a b", "", "1 2", "1 2 3", "1 2 3 4 5", "x"])
    def test_rejects_wrong_shapes(self, bad):
        with pytest.raises(ValueError):
            parse_search_box(bad)

    def test_rejects_a_zero_width_box(self):
        with pytest.raises(ValueError, match="non-zero"):
            parse_search_box("0")

    def test_rejects_an_empty_box(self):
        with pytest.raises(ValueError, match="empty"):
            parse_search_box("10 10 10 10")


class TestUniformConflicts:
    """A uniform value is applied, and its conflicts are reported — never
    clamped. Silently adjusting it would reintroduce exactly the failures the
    per-group derivation exists to prevent."""

    def _strained(self):
        # 4-year baseline in a 0.02/yr strain field: ceiling ~12.5 px
        return suggest_parameters(3.0, 4 * DAYS_PER_YEAR, 5.0, strain_rate_per_yr=0.02)

    def test_kernel_past_the_strain_ceiling_is_flagged(self):
        out = uniform_conflicts(self._strained(), kernel_px=41)
        assert any("strain ceiling" in c for c in out)

    def test_kernel_below_the_texture_floor_is_flagged(self):
        out = uniform_conflicts(self._strained(), kernel_px=3)
        assert any("texture floor" in c for c in out)

    def test_a_box_shorter_than_the_motion_is_flagged(self):
        """The dangerous one: the field rails and reads as saturation."""
        out = uniform_conflicts(self._strained(), corr_search=(-2, -2, 2, 2))
        assert any("rail at the box edge" in c for c in out)

    def test_an_expensive_box_is_flagged(self):
        out = uniform_conflicts(self._strained(), corr_search=(-300, -300, 300, 300))
        assert any("costs" in c for c in out)

    def test_a_compatible_uniform_value_is_silent(self):
        assert uniform_conflicts(self._strained(), kernel_px=11,
                                 corr_search=(-30, -30, 30, 30)) == []

    def test_nothing_overridden_is_silent(self):
        assert uniform_conflicts(self._strained()) == []

    def test_conflicts_are_strings(self):
        out = uniform_conflicts(self._strained(), kernel_px=41,
                                corr_search=(-2, -2, 2, 2))
        assert out and all(isinstance(c, str) for c in out)

    def test_a_pair_with_no_strain_limit_never_trips_the_ceiling(self):
        clean = suggest_parameters(3.0, 400.0, 2.0)
        assert not any(
            "strain ceiling" in c for c in uniform_conflicts(clean, kernel_px=99)
        )


class TestFormatDtSpan:
    """Presentation only: the canonical bin label stays in days because it is
    also a plan-JSON group name and a figure-stem fragment."""

    @pytest.mark.parametrize(
        "lo, hi, unit",
        [(10, 60, "d"), (30, 90, "mo"), (90, 270, "mo"), (270, 810, "yr"),
         (810, 2430, "yr")],
    )
    def test_picks_a_readable_unit(self, lo, hi, unit):
        assert format_dt_span(lo, hi).endswith(unit)

    def test_an_anniversary_span_reads_as_years(self):
        assert format_dt_span(365, 1095) == "1.0–3.0 yr"

    def test_anything_reaching_a_year_reads_in_years(self):
        assert format_dt_span(365, 730) == "1.0–2.0 yr"

    def test_a_single_baseline_is_not_repeated(self):
        assert format_dt_span(2900, 2900) == "7.9 yr"
        assert format_dt_span(45, 45) == "45 d"

    def test_does_not_change_the_canonical_label(self):
        """The group key must stay stable, or plans stop replaying."""
        _edges, labels = geometric_dt_bins([30.0, 2500.0], ratio=3.0)
        assert all(label.startswith("dt") and label.endswith("d") for label in labels)
