#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for geomulticorr.corrections.fit — pure math functions."""

import numpy as np
import numpy.ma as ma
import pytest
from affine import Affine

from geomulticorr.corrections.fit import (
    compute_shift,
    compute_slope,
    fit_ramp_linear,
    fit_linear_with_predictor,
    fit_quadratic_with_predictor,
    rotate_coords,
    fit_directional_profile,
    estimate_directional_angle,
)

def _seam_field(angle_deg, nr=120, nc=180, res=3.0, noise=0.3, seed=0):
    """Synthetic displacement with a step seam whose normal is *angle_deg*.

    Returns (xx, yy, field) in a north-up map frame.
    """
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(
        np.arange(nc) * res,
        (np.arange(nr) * -res) + 6_477_000.0,
    )
    s = rotate_coords(xx, yy, angle_deg)
    field = np.where(s > np.median(s), 2.0, -2.0) + rng.normal(0, noise, (nr, nc))
    return xx, yy, field


class TestComputeShift:
    """Tests for compute_shift() — median/mean estimation."""

    def test_median_symmetric(self):
        """Median of symmetric values."""
        band = ma.array([[0.1, 0.2], [0.3, 0.4]])
        fit_mask = np.ones((2, 2), dtype=bool)
        result = compute_shift(band, fit_mask, "median")
        assert result == pytest.approx(0.25)

    def test_mean_symmetric(self):
        """Mean of symmetric values."""
        band = ma.array([[0.1, 0.2], [0.3, 0.4]])
        fit_mask = np.ones((2, 2), dtype=bool)
        result = compute_shift(band, fit_mask, "mean")
        assert result == pytest.approx(0.25)

    def test_median_robust_to_outlier(self):
        """Median robust: ignores outlier 100.0."""
        band = ma.array([[1.0, 2.0], [3.0, 100.0]])
        fit_mask = np.ones((2, 2), dtype=bool)
        result = compute_shift(band, fit_mask, "median")
        assert result == pytest.approx(2.5)

    def test_mean_sensitive_to_outlier(self):
        """Mean sensitive: pulled by outlier 100.0."""
        band = ma.array([[1.0, 2.0], [3.0, 100.0]])
        fit_mask = np.ones((2, 2), dtype=bool)
        result = compute_shift(band, fit_mask, "mean")
        assert result == pytest.approx(26.5)

    def test_partial_fit_mask(self):
        """Fit mask excludes some pixels: only [0.1, 0.3, 0.5] used."""
        band = ma.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        fit_mask = np.array([[True, False, True, False, True]])
        result = compute_shift(band, fit_mask, "median")
        assert result == pytest.approx(0.3)

    def test_band_mask_overrides_fit_mask(self):
        """Band mask ORed with ~fit_mask: masked pixels excluded."""
        band = ma.array([[1.0, 2.0, 3.0]], mask=[[True, False, False]])
        fit_mask = np.ones((1, 3), dtype=bool)
        result = compute_shift(band, fit_mask, "median")
        assert result == pytest.approx(2.5)

    def test_return_type_is_float(self):
        """Result is a Python float."""
        band = ma.array([[1.0, 2.0], [3.0, 4.0]])
        fit_mask = np.ones((2, 2), dtype=bool)
        result = compute_shift(band, fit_mask, "median")
        assert isinstance(result, float)


class TestComputeSlope:
    """Tests for compute_slope() — terrain slope in degrees."""

    def test_flat_dem_zero_slope(self):
        """Flat DEM: all zeros slope."""
        dem_arr = np.ones((10, 10)) * 500.0
        transform = Affine(10, 0, 0, 0, -10, 0)
        slope = compute_slope(dem_arr, transform)
        assert np.allclose(slope, 0.0, atol=1e-6)
        assert slope.shape == (10, 10)

    def test_row_ramp_45_degrees(self):
        """Row ramp dem[i,j]=10.0·i with 10m pixels → interior slope≈45°."""
        dem_arr = np.zeros((10, 10))
        for i in range(10):
            dem_arr[i, :] = 10.0 * i
        transform = Affine(10, 0, 0, 0, -10, 0)
        slope = compute_slope(dem_arr, transform)
        assert slope[5, 5] == pytest.approx(45.0, abs=0.5)
        assert slope.shape == (10, 10)

    def test_col_ramp_arctan_2(self):
        """Col ramp dem[i,j]=20.0·j with 10m pixels → interior slope≈arctan(2)°."""
        dem_arr = np.zeros((10, 10))
        for j in range(10):
            dem_arr[:, j] = 20.0 * j
        transform = Affine(10, 0, 0, 0, -10, 0)
        slope = compute_slope(dem_arr, transform)
        expected_slope = np.degrees(np.arctan(2.0))
        assert slope[5, 5] == pytest.approx(expected_slope, abs=0.5)

    def test_diagonal_ramp(self):
        """Diagonal ramp: dem[i,j]=10.0·(i+j) → interior slope≈arctan(√2)°."""
        i, j = np.mgrid[:10, :10]
        dem_arr = 10.0 * (i + j)
        transform = Affine(10, 0, 0, 0, -10, 0)
        slope = compute_slope(dem_arr, transform)
        expected_slope = np.degrees(np.arctan(np.sqrt(2)))
        assert slope[5, 5] == pytest.approx(expected_slope, abs=0.5)

    def test_output_non_negative(self):
        """Slope magnitude is always non-negative."""
        dem_arr = np.random.default_rng(42).uniform(100, 500, (10, 10))
        transform = Affine(10, 0, 0, 0, -10, 0)
        slope = compute_slope(dem_arr, transform)
        assert slope.min() >= 0.0

    def test_pixel_size_scaling(self):
        """Larger pixel size → shallower slope gradient."""
        # Same DEM: dem[i,j] = 10.0·i
        dem_arr = np.zeros((10, 10))
        for i in range(10):
            dem_arr[i, :] = 10.0 * i

        # 10m pixels
        transform_10m = Affine(10, 0, 0, 0, -10, 0)
        slope_10m = compute_slope(dem_arr, transform_10m)

        # 100m pixels
        transform_100m = Affine(100, 0, 0, 0, -100, 0)
        slope_100m = compute_slope(dem_arr, transform_100m)

        assert slope_100m[5, 5] < slope_10m[5, 5]


class TestFitRampLinear:
    """Tests for fit_ramp_linear() — pure spatial ramp fitting."""

    def test_perfect_ramp_recovery(self, ramp_raster):
        """Known ramp data[i,j]=0.1·i+0.2·j+1.0 → a≈0.1, b≈0.2, d≈1.0."""
        fit_mask = np.ones((10, 10), dtype=bool)
        a, b, d, r2 = fit_ramp_linear(ramp_raster.data, fit_mask, 1e-6)
        assert a == pytest.approx(0.1, abs=1e-4)
        assert b == pytest.approx(0.2, abs=1e-4)
        assert d == pytest.approx(1.0, abs=1e-4)
        assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_flat_signal_r2_zero(self, flat_raster):
        """All-zeros raster → r2=0 (ss_tot=0 branch)."""
        fit_mask = np.ones((10, 10), dtype=bool)
        a, b, d, r2 = fit_ramp_linear(flat_raster.data, fit_mask, 1e-6)
        assert r2 == 0.0

    def test_return_type(self, ramp_raster):
        """Result is a 4-tuple of floats."""
        fit_mask = np.ones((10, 10), dtype=bool)
        result = fit_ramp_linear(ramp_raster.data, fit_mask, 1e-6)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)

    def test_regularization_shrinks_coefficients(self, ramp_raster):
        """Higher lambda_reg pulls coefficients toward zero."""
        fit_mask = np.ones((10, 10), dtype=bool)
        a_low, _, _, _ = fit_ramp_linear(ramp_raster.data, fit_mask, 1e-6)
        a_high, _, _, _ = fit_ramp_linear(ramp_raster.data, fit_mask, 1.0)
        assert abs(a_high) < abs(a_low)

    def test_partial_mask_recovers_ramp(self, ramp_raster):
        """Only top-left 5×5 pixels stable → still recovers ramp (global signal)."""
        fit_mask = np.zeros((10, 10), dtype=bool)
        fit_mask[:5, :5] = True
        a, b, d, r2 = fit_ramp_linear(ramp_raster.data, fit_mask, 1e-6)
        assert r2 > 0.99


class TestFitLinearWithPredictor:
    """Tests for fit_linear_with_predictor() — ramp + external predictor."""

    def test_pure_predictor_recovery(self, dem_raster):
        """Pure predictor signal: data=3.0·dem + 2.0."""
        dem_arr = ma.filled(dem_raster.data, np.nan)
        band = ma.array(3.0 * dem_arr + 2.0, mask=False)
        fit_mask = np.ones((10, 10), dtype=bool)
        a_row, b_col, c_pred, d, surface, r2 = fit_linear_with_predictor(
            band, dem_arr, fit_mask, 1e-6
        )
        assert a_row == pytest.approx(0.0, abs=1e-6)
        assert b_col == pytest.approx(0.0, abs=1e-6)
        assert c_pred == pytest.approx(3.0, abs=1e-6)
        assert d == pytest.approx(2.0, abs=1e-5)
        assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_combined_recovery(self, mixed_raster, dem_raster):
        """Combined ramp + predictor: data=0.1·R + 0.2·C + 3.0·dem + 2.0."""
        dem_arr = ma.filled(dem_raster.data, np.nan)
        fit_mask = np.ones((10, 10), dtype=bool)
        a_row, b_col, c_pred, d, surface, r2 = fit_linear_with_predictor(
            mixed_raster.data, dem_arr, fit_mask, 1e-6
        )
        assert a_row == pytest.approx(0.1, abs=1e-5)
        assert b_col == pytest.approx(0.2, abs=1e-5)
        assert c_pred == pytest.approx(3.0, abs=1e-5)
        assert d == pytest.approx(2.0, abs=1e-5)
        assert r2 == pytest.approx(1.0, abs=1e-5)

    def test_surface_shape(self, mixed_raster, dem_raster):
        """Surface shape matches input band shape."""
        dem_arr = ma.filled(dem_raster.data, np.nan)
        fit_mask = np.ones((10, 10), dtype=bool)
        _, _, _, _, surface, _ = fit_linear_with_predictor(
            mixed_raster.data, dem_arr, fit_mask, 1e-6
        )
        assert surface.shape == (10, 10)

    def test_constant_predictor_c_pred_zero(self, flat_raster):
        """Constant predictor (pred_std=0) → c_pred=0."""
        predictor = np.ones((10, 10)) * 300.0
        fit_mask = np.ones((10, 10), dtype=bool)
        _, _, c_pred, _, _, _ = fit_linear_with_predictor(
            flat_raster.data, predictor, fit_mask, 1e-6
        )
        assert c_pred == 0.0

    def test_return_6tuple(self, ramp_raster, dem_raster):
        """Returns exactly 6 values: 4 floats, 1 ndarray, 1 float."""
        dem_arr = ma.filled(dem_raster.data, np.nan)
        fit_mask = np.ones((10, 10), dtype=bool)
        result = fit_linear_with_predictor(
            ramp_raster.data, dem_arr, fit_mask, 1e-6
        )
        assert isinstance(result, tuple)
        assert len(result) == 6
        assert isinstance(result[4], np.ndarray)

    def test_partial_mask(self, ramp_raster, dem_raster):
        """Checkerboard stable mask (50%) still recovers signal."""
        dem_arr = ma.filled(dem_raster.data, np.nan)
        fit_mask = np.zeros((10, 10), dtype=bool)
        fit_mask[::2, ::2] = True
        _, _, _, _, _, r2 = fit_linear_with_predictor(
            ramp_raster.data, dem_arr, fit_mask, 1e-6
        )
        assert r2 > 0.90


class TestFitQuadraticWithPredictor:
    """Tests for fit_quadratic_with_predictor() — quadratic + predictor."""

    def test_pure_predictor_r2_one(self, dem_raster):
        """Pure predictor signal: data=2.0·dem + 1.5 → r2≈1."""
        dem_arr = ma.filled(dem_raster.data, np.nan)
        band = ma.array(2.0 * dem_arr + 1.5, mask=False)
        fit_mask = np.ones((10, 10), dtype=bool)
        g, a2, b2, cxy, ax, by, fp, surface, r2 = fit_quadratic_with_predictor(
            band, dem_arr, fit_mask, 1e-6
        )
        assert r2 == pytest.approx(1.0, abs=1e-8)
        assert fp != 0  # predictor coefficient is non-zero
        assert surface.shape == (10, 10)

    def test_quadratic_spatial_r2_one(self):
        """Quadratic signal with predictor: data=2.0·R² - R·C + 5.0·dem + 3.0."""
        rng = np.random.default_rng(77)
        dem_arr = rng.uniform(100, 500, (20, 20))
        i, j = np.mgrid[:20, :20]
        band_data = (
            2.0 * i**2 - i * j + 5.0 * dem_arr + 3.0
        )
        band = ma.array(band_data, mask=False)
        fit_mask = np.ones((20, 20), dtype=bool)
        g, a2, b2, cxy, ax, by, fp, surface, r2 = fit_quadratic_with_predictor(
            band, dem_arr, fit_mask, 1e-6
        )
        assert r2 == pytest.approx(1.0, abs=1e-7)
        assert surface.shape == (20, 20)

    def test_surface_shape(self):
        """Surface shape matches input band."""
        rng = np.random.default_rng(99)
        dem_arr = rng.uniform(100, 500, (20, 20))
        band = ma.array(np.random.default_rng(100).standard_normal((20, 20)))
        fit_mask = np.ones((20, 20), dtype=bool)
        _, _, _, _, _, _, _, surface, _ = fit_quadratic_with_predictor(
            band, dem_arr, fit_mask, 1e-6
        )
        assert surface.shape == (20, 20)

    def test_return_9tuple(self, ramp_raster, dem_raster):
        """Returns exactly 9 values."""
        dem_arr = ma.filled(dem_raster.data, np.nan)
        fit_mask = np.ones((10, 10), dtype=bool)
        result = fit_quadratic_with_predictor(
            ramp_raster.data, dem_arr, fit_mask, 1e-6
        )
        assert isinstance(result, tuple)
        assert len(result) == 9
        assert isinstance(result[7], np.ndarray)

    def test_constant_predictor_no_crash(self, flat_raster):
        """Constant predictor (pred_std=0) → no exception."""
        predictor = np.ones((10, 10)) * 300.0
        fit_mask = np.ones((10, 10), dtype=bool)
        result = fit_quadratic_with_predictor(
            flat_raster.data, predictor, fit_mask, 1e-6
        )
        assert len(result) == 9
        assert isinstance(result[7], np.ndarray)

class TestRotateCoords:
    """Tests for rotate_coords() — along-track projection."""

    def test_angle_zero_is_easting(self):
        """angle=0 → s varies with x (East) only, constant down columns."""
        xx, yy = np.meshgrid(np.arange(5) * 3.0, np.arange(4) * -3.0 + 100.0)
        s = rotate_coords(xx, yy, 0.0)
        # every row identical (no dependence on y)
        assert np.allclose(s - s[0, :], 0.0)

    def test_angle_90_is_northing(self):
        """angle=90 → s varies with y only, constant along rows."""
        xx, yy = np.meshgrid(np.arange(5) * 3.0, np.arange(4) * -3.0 + 100.0)
        s = rotate_coords(xx, yy, 90.0)
        assert np.allclose(s - s[:, [0]], 0.0)

    def test_180_periodicity_flips_sign(self):
        """s(θ+180) == -s(θ): the projection axis reverses direction."""
        xx, yy, _ = _seam_field(0.0)
        s = rotate_coords(xx, yy, 30.0)
        s180 = rotate_coords(xx, yy, 210.0)
        assert np.allclose(s180, -s, atol=1e-6)


class TestFitDirectionalProfile:
    """Tests for fit_directional_profile()."""

    def test_reduces_residual_std(self):
        """Subtracting the fitted surface cuts the seam std toward the noise floor."""
        xx, yy, field = _seam_field(-12.0)
        mask = np.ones_like(field, dtype=bool)
        s = rotate_coords(xx, yy, -12.0)
        surface, centers, median, fitted = fit_directional_profile(
            s, field, mask, n_bins=100, profile="bin"
        )
        std_before = np.nanstd(field[mask])
        std_after = np.nanstd((field - surface)[mask])
        assert std_after < 0.5 * std_before
        assert surface.shape == field.shape

    def test_invalid_profile_raises(self):
        xx, yy, field = _seam_field(0.0)
        s = rotate_coords(xx, yy, 0.0)
        with pytest.raises(ValueError, match="profile must be"):
            fit_directional_profile(
                s, field, np.ones_like(field, bool), profile="bogus"
            )

    def test_too_few_bins_raises(self):
        """A near-constant s with high min_bin_count leaves too few bins."""
        xx, yy, field = _seam_field(0.0)
        s = rotate_coords(xx, yy, 0.0)
        with pytest.raises(ValueError, match="too few populated bins"):
            fit_directional_profile(
                s, field, np.ones_like(field, bool),
                n_bins=4, min_bin_count=10_000_000,
            )

    def test_profile_models_run(self):
        """spline / poly / bin all return a full-grid surface."""
        xx, yy, field = _seam_field(-8.0)
        s = rotate_coords(xx, yy, -8.0)
        mask = np.ones_like(field, dtype=bool)
        for model in ("spline", "poly", "bin"):
            surface, *_ = fit_directional_profile(s, field, mask, profile=model)
            assert surface.shape == field.shape


class TestEstimateDirectionalAngle:
    """Tests for estimate_directional_angle()."""

    @pytest.mark.parametrize("true_angle", [-12.0, 0.0, 25.0, -40.0])
    def test_recovers_angle(self, true_angle):
        """Auto-estimate recovers the seam angle within the refine resolution."""
        xx, yy, field = _seam_field(true_angle)
        mask = np.ones_like(field, dtype=bool)
        est = estimate_directional_angle(xx, yy, field, mask, n_bins=100)
        assert est == pytest.approx(true_angle, abs=2.0)

    def test_returns_normalised_range(self):
        """A seam at 170° is reported as its (-90, 90] equivalent (≈-10°)."""
        xx, yy, field = _seam_field(170.0)
        mask = np.ones_like(field, dtype=bool)
        est = estimate_directional_angle(xx, yy, field, mask, n_bins=100)
        assert -90.0 < est <= 90.0
        assert est == pytest.approx(-10.0, abs=2.0)