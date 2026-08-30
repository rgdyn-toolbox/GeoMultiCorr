#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for geomulticorr.corrections.corrections — OOP correction classes."""

import copy

import numpy as np
import numpy.ma as ma
import pytest

from geomulticorr.corrections.corrections import (
    BaseCorrection,
    CorrectionPipeline,
    MedianCentering,
    RampCorrection,
    TopoCorrection,
    TopoRampCorrection,
    DirectionalBiasCorrection,
    SlopeRampCorrection,
    AlongTrackDestriping,
    AcrossTrackDestriping,
    make_corrections,
)
# Import geomulticorr utilities for testing
from .conftest import MockRaster
from geomulticorr.corrections.fit import rotate_coords
from geomulticorr.corrections.masks import SlopeMask

def _seam_mock_raster(angle_deg=-12.0, nr=80, nc=100, noise=0.3, seed=0):
    """MockRaster with a step seam whose normal is *angle_deg* (map space)."""
    rng = np.random.default_rng(seed)
    rast = MockRaster(data=ma.array(np.zeros((nr, nc)), mask=False))
    xx, yy = rast.coords(grid=True, force_offset="ll")
    s = rotate_coords(xx, yy, angle_deg)
    field = np.where(s > np.median(s), 2.0, -2.0) + rng.normal(0, noise, (nr, nc))
    return MockRaster(data=ma.array(field, mask=False))

class TestBaseCorrection:
    """Tests for BaseCorrection abstract base and shared helpers."""

    def test_fit_raises_not_implemented(self):
        """BaseCorrection.fit() raises NotImplementedError."""
        class _NoImpl(BaseCorrection):
            pass
        corr = _NoImpl()
        from .conftest import MockRaster
        raster = MockRaster(data=ma.array(np.zeros((5, 5))))
        with pytest.raises(NotImplementedError, match="does not implement fit"):
            corr.fit(raster)

    def test_apply_before_fit_raises(self, ramp_raster):
        """apply() before fit() raises RuntimeError."""
        corr = RampCorrection()
        with pytest.raises(RuntimeError, match="Call fit\\(\\) before apply"):
            corr.apply(ramp_raster)

    def test_apply_corrects_valid_pixels(self, offset_raster):
        """Apply MedianCentering: all-5.0 raster → all-0.0."""
        corr = MedianCentering()
        corr.fit(offset_raster)
        result = corr.apply(offset_raster)
        assert np.allclose(
            result.data.data[~result.data.mask], 0.0, atol=1e-10
        )

    def test_apply_preserves_existing_mask(self, partial_mask_raster):
        """Masked pixels stay masked after apply()."""
        corr = MedianCentering()
        corr.fit(partial_mask_raster)
        result = corr.apply(partial_mask_raster)
        assert result.data.mask[0, 0] == True
        assert result.data.mask[9, 9] == True

    def test_apply_returns_new_raster(self, partial_mask_raster):
        """apply() returns a new MockRaster, not the same object."""
        corr = MedianCentering()
        corr.fit(partial_mask_raster)
        result = corr.apply(partial_mask_raster)
        assert result is not partial_mask_raster

    def test_add_two_corrections_creates_pipeline(self):
        """Using + operator: correction + correction → CorrectionPipeline."""
        pipeline = MedianCentering() + RampCorrection()
        assert isinstance(pipeline, CorrectionPipeline)
        assert len(pipeline.steps) == 2

    def test_add_mask_to_correction_raises_type_error(self):
        """Using + operator: correction + mask → TypeError."""
        corr = RampCorrection()
        mask = SlopeMask()
        with pytest.raises(TypeError, match="Cannot chain correction"):
            corr + mask

    def test_resolve_stable_mask_none(self, partial_mask_raster):
        """_resolve_stable_mask(None, ...) uses all valid pixels."""
        result = BaseCorrection._resolve_stable_mask(
            None, partial_mask_raster.data, (10, 10), partial_mask_raster.transform
        )
        assert result.sum() == 98  # 100 - 2 masked

    def test_resolve_stable_mask_numpy_array(self, partial_mask_raster):
        """_resolve_stable_mask(bool_array, ...) ANDs with valid pixels."""
        stable_mask = np.zeros((10, 10), dtype=bool)
        stable_mask[:5, :5] = True
        result = BaseCorrection._resolve_stable_mask(
            stable_mask, partial_mask_raster.data, (10, 10), partial_mask_raster.transform
        )
        assert result.sum() <= 25

    def test_resolve_stable_mask_dem_nan_excluded(self, partial_mask_raster, dem_raster):
        """DEM NaN pixels additionally excluded from fit."""
        dem_arr = ma.filled(dem_raster.data, np.nan).copy()
        dem_arr[2, 2] = np.nan
        dem_arr[3, 3] = np.nan
        dem_arr[4, 4] = np.nan
        result = BaseCorrection._resolve_stable_mask(
            None, partial_mask_raster.data, (10, 10), partial_mask_raster.transform, dem_arr
        )
        assert result.sum() == 95  # 100 - 2 masked - 3 NaN


class TestCorrectionPipeline:
    """Tests for CorrectionPipeline — sequential fitting and application."""

    def test_construct_with_plus(self):
        """Using + to chain 3 corrections."""
        p = MedianCentering() + RampCorrection() + MedianCentering()
        assert len(p.steps) == 3
        assert isinstance(p.steps[0], MedianCentering)
        assert isinstance(p.steps[1], RampCorrection)
        assert isinstance(p.steps[2], MedianCentering)

    def test_extend_pipeline_with_plus(self):
        """(pipeline + step) merges steps, not nests."""
        p1 = MedianCentering() + RampCorrection()
        p2 = p1 + MedianCentering()
        assert len(p2.steps) == 3

    def test_fit_on_residuals(self, ramp_raster):
        """Each step fits on residual of previous step."""
        p = MedianCentering() + RampCorrection()
        p.fit(ramp_raster)

        # First step shifts by median ≈ 2.35
        shift = p.steps[0].meta["shift"]
        assert shift == pytest.approx(2.35, abs=0.01)

        # Second step fits ramp on residual (after shift)
        # Residual ramp: data[i,j] - shift ≈ 0.1·i + 0.2·j + 1.0 - 2.35
        a = p.steps[1].meta["a"]
        assert a == pytest.approx(0.1, abs=1e-4)

    def test_apply_before_fit_raises(self):
        """apply() without fit() raises RuntimeError."""
        p = MedianCentering() + RampCorrection()
        from .conftest import MockRaster
        raster = MockRaster(data=ma.array(np.zeros((5, 5))))
        with pytest.raises(RuntimeError, match="Call fit\\(\\) before apply"):
            p.apply(raster)

    def test_fit_and_apply_removes_all_signal(self, ramp_raster):
        """Pipeline fit_and_apply: ramp raster → residuals near 0."""
        p = MedianCentering() + RampCorrection()
        result = p.fit_and_apply(ramp_raster)
        assert np.allclose(result.data.data, 0.0, atol=1e-4)

    def test_meta_structure(self, ramp_raster):
        """Pipeline.meta is dict keyed by step index."""
        p = MedianCentering() + RampCorrection()
        p.fit(ramp_raster)
        meta = p.meta
        assert 0 in meta
        assert 1 in meta
        assert meta[0]["step"] == "MedianCentering"
        assert "shift" in meta[0]["params"]
        assert meta[1]["step"] == "RampCorrection"
        assert "r2" in meta[1]["params"]

    def test_repr(self):
        """repr shows step names in order."""
        p = MedianCentering() + RampCorrection()
        assert "MedianCentering" in repr(p)
        assert "RampCorrection" in repr(p)

    def test_add_mask_raises_type_error(self):
        """pipeline + mask → TypeError."""
        p = MedianCentering() + RampCorrection()
        mask = SlopeMask()
        with pytest.raises(TypeError, match="Cannot chain correction with mask"):
            p + mask


class TestMedianCentering:
    """Tests for MedianCentering — scalar offset removal."""

    def test_default_stat_is_median(self):
        """Default stat='median'."""
        corr = MedianCentering()
        assert corr.stat == "median"

    def test_invalid_stat_raises(self):
        """Invalid stat raises ValueError."""
        with pytest.raises(ValueError, match="stat must be"):
            MedianCentering(stat="mode")

    def test_fit_sets_surface_to_median(self, offset_raster):
        """Fit on all-5.0 raster: shift=5.0."""
        corr = MedianCentering()
        corr.fit(offset_raster)
        assert corr._surface == pytest.approx(5.0)
        assert corr.meta == {"shift": pytest.approx(5.0)}

    def test_fit_returns_self(self, ramp_raster):
        """fit() returns self for chaining."""
        corr = MedianCentering()
        result = corr.fit(ramp_raster)
        assert result is corr

    def test_apply_centers_to_zero(self, offset_raster):
        """Apply to all-5.0 raster: result all-0.0."""
        corr = MedianCentering()
        corr.fit(offset_raster)
        result = corr.apply(offset_raster)
        assert np.allclose(result.data.data, 0.0, atol=1e-10)

    def test_apply_with_ramp(self, ramp_raster):
        """Apply to ramp raster: result[5,5] ≈ data[5,5] - median."""
        corr = MedianCentering()
        corr.fit(ramp_raster)
        shift = corr._surface
        result = corr.apply(ramp_raster)
        expected = ramp_raster.data[5, 5] - shift
        assert result.data[5, 5] == pytest.approx(expected, abs=1e-6)

    def test_stable_mask_restricts_fit(self):
        """Stable mask excludes outlier: shift = median(1,2,3) = 2.0."""
        from .conftest import MockRaster
        band_data = np.array([[1.0, 2.0], [3.0, 100.0]])
        band = ma.array(band_data, mask=False)
        raster = MockRaster(data=band)
        stable_mask = np.array([[True, True], [True, False]])
        corr = MedianCentering()
        corr.fit(raster, stable_mask=stable_mask)
        assert corr._surface == pytest.approx(2.0)

    def test_mean_stat(self, ramp_raster):
        """stat='mean': shift ≈ mean of all values."""
        corr = MedianCentering(stat="mean")
        corr.fit(ramp_raster)
        expected_mean = ma.mean(ramp_raster.data)
        assert corr._surface == pytest.approx(expected_mean, abs=1e-6)

    def test_fit_and_apply(self, offset_raster):
        """fit_and_apply in one call."""
        corr = MedianCentering()
        result = corr.fit_and_apply(offset_raster)
        assert np.allclose(result.data.data, 0.0, atol=1e-10)


class TestRampCorrection:
    """Tests for RampCorrection — linear spatial ramp removal."""

    def test_fit_recovers_coefficients(self, ramp_raster):
        """Fit on ramp_raster: a≈0.1, b≈0.2, d≈1.0, r2≈1.0."""
        corr = RampCorrection()
        corr.fit(ramp_raster)
        assert corr.meta["a"] == pytest.approx(0.1, abs=1e-4)
        assert corr.meta["b"] == pytest.approx(0.2, abs=1e-4)
        assert corr.meta["d"] == pytest.approx(1.0, abs=1e-4)
        assert corr.meta["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_apply_removes_ramp(self, ramp_raster):
        """After fit_and_apply: residuals ≈ 0."""
        corr = RampCorrection()
        result = corr.fit_and_apply(ramp_raster)
        assert np.allclose(result.data.data, 0.0, atol=1e-5)

    def test_surface_shape(self, ramp_raster):
        """_surface has same shape as input raster."""
        corr = RampCorrection()
        corr.fit(ramp_raster)
        assert corr._surface.shape == (10, 10)

    def test_fit_returns_self(self, ramp_raster):
        """fit() returns self, _fit_called=True."""
        corr = RampCorrection()
        result = corr.fit(ramp_raster)
        assert result is corr
        assert corr._fit_called == True

    def test_too_few_pixels_raises(self, ramp_raster):
        """Fewer than 4 stable pixels → ValueError."""
        stable_mask = np.zeros((10, 10), dtype=bool)
        stable_mask[0, 0] = stable_mask[0, 1] = stable_mask[0, 2] = True
        corr = RampCorrection()
        with pytest.raises(ValueError, match="Too few valid pixels for RampCorrection"):
            corr.fit(ramp_raster, stable_mask=stable_mask)

    def test_flat_signal_r2_zero(self, flat_raster):
        """All-zeros raster: r2=0."""
        corr = RampCorrection()
        corr.fit(flat_raster)
        assert corr.meta["r2"] == 0.0

    def test_meta_keys(self, ramp_raster):
        """meta has exactly {'a', 'b', 'd', 'r2'}."""
        corr = RampCorrection()
        corr.fit(ramp_raster)
        assert set(corr.meta.keys()) == {"a", "b", "d", "r2"}

    def test_apply_to_different_raster(self, ramp_raster, flat_raster):
        """Fit on ramp, apply to flat: result[5,5] ≈ -(0.1·5+0.2·5+1.0)."""
        corr = RampCorrection()
        corr.fit(ramp_raster)
        result = corr.apply(flat_raster)
        expected = -(0.1 * 5 + 0.2 * 5 + 1.0)
        assert result.data[5, 5] == pytest.approx(expected, abs=1e-4)

    def test_custom_lambda_reg(self, ramp_raster):
        """Higher lambda_reg shrinks coefficients."""
        corr_low = RampCorrection(lambda_reg=1e-6)
        corr_high = RampCorrection(lambda_reg=1.0)
        corr_low.fit(ramp_raster)
        corr_high.fit(ramp_raster)
        assert abs(corr_high.meta["a"]) < abs(corr_low.meta["a"])


class TestTopoCorrection:
    """Tests for TopoCorrection — ramp + DEM elevation correction."""

    def test_missing_dem_raises(self, ramp_raster):
        """fit() without dem= kwarg → ValueError."""
        corr = TopoCorrection()
        with pytest.raises(ValueError, match="requires dem="):
            corr.fit(ramp_raster)

    def test_dem_shape_mismatch_reprojected(self, ramp_raster, dem_raster):
        """DEM on a different grid is reprojected onto the displacement grid.

        Unlike SlopeMask (which asserts on a shape mismatch), TopoCorrection
        regrids the DEM via ``dem.reproject(ref=raster)`` and carries on.
        """
        dem_off_grid = type(dem_raster)(
            data=ma.array(np.random.default_rng(1).uniform(100, 500, (15, 15)))
        )
        corr = TopoCorrection(order="linear")
        corr.fit(ramp_raster, dem=dem_off_grid)
        assert corr._fit_called
        assert set(corr.meta) >= {"row_slope", "col_slope", "dem_slope", "intercept", "r2"}

    def test_same_shape_misaligned_dem_is_regridded_and_warned(
        self, ramp_raster, dem_raster, caplog_gmc
    ):
        """A DEM with matching shape but a shifted origin must NOT pass through.

        This is the case the old shape-only guard accepted silently: same
        shape, different ground. It produced a wrong topographic correction
        with no error and no warning. Now it regrids and says so.
        """
        from affine import Affine

        shifted = type(dem_raster)(
            data=dem_raster.data.copy(),
            transform=dem_raster.transform * Affine.translation(0.5, 0.5),
        )
        assert shifted.data.shape == ramp_raster.data.shape, "precondition"

        corr = TopoCorrection(order="linear")
        with caplog_gmc.at_level("WARNING"):
            corr.fit(ramp_raster, dem=shifted)

        assert corr._fit_called
        assert "shape matches" in caplog_gmc.text
        assert "georeferencing does not" in caplog_gmc.text
        # Both grids must be reported, and they must differ — describing the
        # DEM after the regrid would print the target grid twice.
        dem_line = next(l for l in caplog_gmc.text.splitlines() if "DEM  :" in l)
        pair_line = next(l for l in caplog_gmc.text.splitlines() if "pair :" in l)
        assert dem_line.split("DEM  :")[1] != pair_line.split("pair :")[1]

    def test_aligned_dem_is_not_regridded(self, ramp_raster, dem_raster, caplog_gmc):
        """The common case stays silent and does no work."""
        corr = TopoCorrection(order="linear")
        with caplog_gmc.at_level("INFO"):
            corr.fit(ramp_raster, dem=dem_raster)

        assert corr._fit_called
        assert "regrid" not in caplog_gmc.text.lower()

    def test_declares_dem_required(self):
        """apply_pairs_corrections validates _REQUIRED_KWARGS before looping."""
        assert "dem" in TopoCorrection._REQUIRED_KWARGS
        assert "dem" in TopoRampCorrection._REQUIRED_KWARGS
        assert "dem" in SlopeRampCorrection._REQUIRED_KWARGS

    def test_invalid_order_raises(self):
        """order not in ('linear', 'quadratic') → ValueError."""
        with pytest.raises(ValueError, match="order must be"):
            TopoCorrection(order="cubic")

    def test_linear_recovers_dem_coeff(self, mixed_raster, dem_raster):
        """Linear fit on mixed_raster: dem_slope≈3.0, r2≈1.0."""
        corr = TopoCorrection(order="linear")
        corr.fit(mixed_raster, dem=dem_raster)
        assert corr.meta["dem_slope"] == pytest.approx(3.0, abs=1e-4)
        assert corr.meta["r2"] == pytest.approx(1.0, abs=1e-5)

    def test_linear_meta_keys(self, mixed_raster, dem_raster):
        """Linear meta has specific keys."""
        corr = TopoCorrection(order="linear")
        corr.fit(mixed_raster, dem=dem_raster)
        expected_keys = {"row_slope", "col_slope", "dem_slope", "intercept", "r2"}
        assert set(corr.meta.keys()) == expected_keys

    def test_quadratic_high_r2(self, mixed_raster, dem_raster):
        """Quadratic fit: r2 > 0.999."""
        corr = TopoCorrection(order="quadratic")
        corr.fit(mixed_raster, dem=dem_raster)
        assert corr.meta["r2"] > 0.999

    def test_quadratic_meta_keys(self, mixed_raster, dem_raster):
        """Quadratic meta has 8 keys."""
        corr = TopoCorrection(order="quadratic")
        corr.fit(mixed_raster, dem=dem_raster)
        expected_keys = {"c_bias", "c_row2", "c_col2", "c_rowcol", "c_row", "c_col", "c_dem", "r2"}
        assert set(corr.meta.keys()) == expected_keys

    def test_too_few_pixels_linear(self, ramp_raster, dem_raster):
        """4 pixels, order='linear' (needs≥5) → ValueError."""
        stable_mask = np.zeros((10, 10), dtype=bool)
        stable_mask[:2, :2] = True
        corr = TopoCorrection(order="linear")
        with pytest.raises(ValueError, match="Too few valid pixels for TopoCorrection"):
            corr.fit(ramp_raster, stable_mask=stable_mask, dem=dem_raster)

    def test_too_few_pixels_quadratic(self, ramp_raster, dem_raster):
        """7 pixels, order='quadratic' (needs≥8) → ValueError."""
        stable_mask = np.zeros((10, 10), dtype=bool)
        stable_mask.flat[:7] = True
        corr = TopoCorrection(order="quadratic")
        with pytest.raises(ValueError, match="Too few valid pixels for TopoCorrection"):
            corr.fit(ramp_raster, stable_mask=stable_mask, dem=dem_raster)

    def test_apply_removes_topo_signal(self, mixed_raster, dem_raster):
        """fit_and_apply: residuals ≈ 0."""
        corr = TopoCorrection(order="linear")
        result = corr.fit_and_apply(mixed_raster, dem=dem_raster)
        assert np.allclose(result.data.data, 0.0, atol=1e-4)

    def test_dem_nan_pixels_excluded(self, ramp_raster):
        """DEM NaN pixels excluded from fit."""
        dem_arr = ma.array(np.ones((10, 10)) * 250.0, mask=False)
        dem_data = ma.filled(dem_arr, np.nan).copy()
        dem_data[0, 0] = np.nan
        from .conftest import MockRaster
        dem_raster = MockRaster(data=ma.array(dem_data, mask=False))
        corr = TopoCorrection(order="linear")
        corr.fit(ramp_raster, dem=dem_raster)
        assert corr._fit_called == True


class TestTopoRampCorrection:
    """Tests for TopoRampCorrection — same fitting, different keys."""

    def test_linear_meta_keys(self, mixed_raster, dem_raster):
        """Linear meta: {'a', 'b', 'c', 'd', 'r2'}."""
        corr = TopoRampCorrection(order="linear")
        corr.fit(mixed_raster, dem=dem_raster)
        expected_keys = {"a", "b", "c", "d", "r2"}
        assert set(corr.meta.keys()) == expected_keys

    def test_quadratic_fz_key(self, mixed_raster, dem_raster):
        """Quadratic meta: key is 'fz' not 'c_dem'."""
        corr = TopoRampCorrection(order="quadratic")
        corr.fit(mixed_raster, dem=dem_raster)
        expected_keys = {"g", "a2", "b2", "cxy", "ax", "by", "fz", "r2"}
        assert set(corr.meta.keys()) == expected_keys

    def test_linear_min_4_pixels_ok(self, dem_raster):
        """Exactly 4 pixels for linear order (min=4) → no error."""
        from .conftest import MockRaster
        band_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        band = ma.array(band_data, mask=False)
        raster = MockRaster(data=band)
        dem_data = np.array([[100.0, 110.0], [120.0, 130.0]])
        dem_raster_local = MockRaster(data=ma.array(dem_data, mask=False))
        corr = TopoRampCorrection(order="linear")
        corr.fit(raster, dem=dem_raster_local)
        assert corr._fit_called == True

    def test_too_few_pixels_linear(self, ramp_raster, dem_raster):
        """3 pixels (needs≥4) → ValueError."""
        stable_mask = np.zeros((10, 10), dtype=bool)
        stable_mask[0, 0] = stable_mask[0, 1] = stable_mask[0, 2] = True
        corr = TopoRampCorrection(order="linear")
        with pytest.raises(ValueError, match="Too few valid pixels for TopoRampCorrection"):
            corr.fit(ramp_raster, stable_mask=stable_mask, dem=dem_raster)


class TestSlopeRampCorrection:
    """Tests for SlopeRampCorrection — uses slope as predictor."""

    def test_missing_dem_raises(self, ramp_raster):
        """fit() without dem= → ValueError."""
        corr = SlopeRampCorrection()
        with pytest.raises(ValueError, match="requires dem="):
            corr.fit(ramp_raster)

    def test_linear_meta_keys(self, ramp_raster, slope_dem_raster):
        """Linear meta: {'row_slope', 'col_slope', 'slope_coeff', 'intercept', 'r2'}."""
        corr = SlopeRampCorrection(order="linear")
        corr.fit(ramp_raster, dem=slope_dem_raster)
        expected_keys = {"row_slope", "col_slope", "slope_coeff", "intercept", "r2"}
        assert set(corr.meta.keys()) == expected_keys

    def test_quadratic_meta_keys(self, ramp_raster, slope_dem_raster):
        """Quadratic meta: 'c_slope' instead of 'c_dem'."""
        corr = SlopeRampCorrection(order="quadratic")
        corr.fit(ramp_raster, dem=slope_dem_raster)
        expected_keys = {"c_bias", "c_row2", "c_col2", "c_rowcol", "c_row", "c_col", "c_slope", "r2"}
        assert set(corr.meta.keys()) == expected_keys

    def test_uses_slope_as_predictor(self):
        """Signal correlated with slope (not elevation)."""
        from .conftest import MockRaster
        from geomulticorr.corrections.fit import compute_slope
        # DEM with VARYING slopes: dem[i,j] = 0.1·i² + 0.05·j² (quadratic → varying slope)
        i, j = np.mgrid[:10, :10]
        dem_data = 0.1 * i**2 + 0.05 * j**2
        dem = MockRaster(data=ma.array(dem_data, mask=False))
        slope_arr = compute_slope(dem.data, dem.transform)
        # Signal: pure noise + 0.5·slope (tests that slope predictor is used)
        band = ma.array(0.5 * slope_arr, mask=False)
        raster = MockRaster(data=band)
        corr = SlopeRampCorrection(order="linear")
        corr.fit(raster, dem=dem)
        # Slope coefficient should recover the 0.5 scaling
        assert abs(corr.meta["slope_coeff"] - 0.5) < 0.2

    def test_surface_shape(self, ramp_raster, slope_dem_raster):
        """_surface.shape == raster.shape."""
        corr = SlopeRampCorrection()
        corr.fit(ramp_raster, dem=slope_dem_raster)
        assert corr._surface.shape == (10, 10)

    def test_too_few_pixels_linear(self, ramp_raster, dem_raster):
        """4 pixels, order='linear' (needs≥5) → ValueError."""
        stable_mask = np.zeros((10, 10), dtype=bool)
        stable_mask[0, 0] = stable_mask[0, 1] = stable_mask[0, 2] = stable_mask[0, 3] = True
        corr = SlopeRampCorrection(order="linear")
        with pytest.raises(ValueError, match="Too few valid pixels for SlopeRampCorrection"):
            corr.fit(ramp_raster, stable_mask=stable_mask, dem=dem_raster)

class TestDirectionalBiasCorrection:
    """Tests for DirectionalBiasCorrection."""

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="profile must be"):
            DirectionalBiasCorrection(profile="bogus")

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            DirectionalBiasCorrection(n_bins=2)

    def test_apply_before_fit_raises(self):
        corr = DirectionalBiasCorrection(angle=0.0)
        raster = _seam_mock_raster()
        with pytest.raises(RuntimeError):
            corr.apply(raster)

    def test_too_few_pixels_raises(self):
        from .conftest import MockRaster
        small = MockRaster(data=ma.array(np.zeros((5, 5)), mask=False))
        with pytest.raises(ValueError, match="Too few stable pixels"):
            DirectionalBiasCorrection(n_bins=50).fit(small)

    def test_auto_angle_recovered(self):
        """Auto mode recovers the seam angle and logs it normalised."""
        raster = _seam_mock_raster(angle_deg=-12.0)
        corr = DirectionalBiasCorrection(angle=None, profile="bin", n_bins=80)
        corr.fit(raster)
        assert corr.meta["angle_deg"] == pytest.approx(-12.0, abs=1.5)
        assert -90.0 < corr.meta["angle_deg"] <= 90.0

    def test_reduces_residual_std(self):
        raster = _seam_mock_raster(angle_deg=-12.0)
        corr = DirectionalBiasCorrection(angle=None, profile="bin", n_bins=80)
        out = corr.fit_and_apply(raster)
        assert corr.meta["std_after"] < corr.meta["std_before"]
        assert out.data.shape == raster.data.shape

    def test_meta_contains_profile(self):
        raster = _seam_mock_raster()
        corr = DirectionalBiasCorrection(angle=0.0, n_bins=60).fit(raster)
        for key in ("profile_centers", "profile_median", "profile_fitted",
                    "std_before", "std_after", "angle_deg"):
            assert key in corr.meta

    def test_fixed_angle_used_verbatim(self):
        raster = _seam_mock_raster()
        corr = DirectionalBiasCorrection(angle=33.0, n_bins=60).fit(raster)
        assert corr.meta["angle_deg"] == 33.0

    def test_composable_into_pipeline(self):
        """+ chains into a CorrectionPipeline."""
        pipeline = MedianCentering() + DirectionalBiasCorrection(angle=0.0, n_bins=60)
        assert isinstance(pipeline, CorrectionPipeline)
        assert len(pipeline.steps) == 2

class TestDestripingConstruction:
    """Construction-time invariants for the destriping corrections."""

    def test_empty_meta_before_fit(self):
        """meta is empty until fit() is called."""
        assert AlongTrackDestriping().meta == {}
        assert AcrossTrackDestriping().meta == {}


class TestMakeCorrections:
    """Tests for make_corrections() — applies pipeline to x/y independently."""

    def test_returns_two_rasters(self, ramp_raster, flat_raster):
        """Returns tuple of 2 MockRasters."""
        pipeline = MedianCentering()
        xc, yc = make_corrections(ramp_raster, flat_raster, pipeline)
        assert hasattr(xc, "data")
        assert hasattr(yc, "data")
        assert hasattr(xc, "copy")
        assert hasattr(yc, "copy")

    def test_original_pipeline_not_mutated(self, ramp_raster, flat_raster):
        """Original pipeline._fit_called remains False (deepcopy proof)."""
        pipeline = MedianCentering()
        make_corrections(ramp_raster, flat_raster, pipeline)
        assert pipeline._fit_called == False

    def test_both_components_corrected(self, offset_raster):
        """Two offset_rasters → both outputs ≈ 0.0."""
        pipeline = MedianCentering()
        xc, yc = make_corrections(offset_raster, offset_raster, pipeline)
        assert np.allclose(xc.data.data, 0.0, atol=1e-10)
        assert np.allclose(yc.data.data, 0.0, atol=1e-10)

    def test_different_stable_masks(self, offset_raster, ramp_raster):
        """Different x_stable, y_stable → different corrections."""
        pipeline = MedianCentering()
        x_stable = np.ones((10, 10), dtype=bool)
        y_stable = np.zeros((10, 10), dtype=bool)
        y_stable[:5, :5] = True
        xc, yc = make_corrections(offset_raster, ramp_raster, pipeline, x_stable, y_stable)
        # x should be centered to 0, y centered to median of top-left 5×5 of ramp
        assert np.allclose(xc.data.data, 0.0, atol=1e-10)

    def test_dem_kwarg_forwarded(self, mixed_raster, dem_raster):
        """TopoCorrection with dem kwarg → no error."""
        pipeline = TopoCorrection(order="linear")
        xc, yc = make_corrections(mixed_raster, mixed_raster, pipeline, dem=dem_raster)
        assert hasattr(xc, "data")
        assert hasattr(yc, "data")


def _stripe_mock_raster(axis=0, wavelength_px=40, nr=120, nc=80, amp=0.8, noise=0.08, seed=0):
    """MockRaster with a sinusoidal stripe varying along *axis* (0=rows, 1=cols)."""
    from .conftest import MockRaster
    rng = np.random.default_rng(seed)
    row, col = np.mgrid[0:nr, 0:nc]
    coord = row if axis == 0 else col
    field = amp * np.sin(2 * np.pi * coord / wavelength_px) + rng.normal(0, noise, (nr, nc))
    return MockRaster(data=ma.array(field, mask=False))


class TestDestriping:
    """Tests for AlongTrackDestriping / AcrossTrackDestriping (Fourier)."""

    def test_invalid_taper_raises(self):
        with pytest.raises(ValueError, match="taper"):
            AlongTrackDestriping(taper=1.0)

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError, match="wavelength_max"):
            AlongTrackDestriping(wavelength_min=400.0, wavelength_max=100.0)

    def test_apply_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            AlongTrackDestriping(wavelength_min=100.0).apply(_stripe_mock_raster())

    def test_along_track_reduces_std(self):
        """Row-varying stripe → AlongTrackDestriping cuts the residual std."""
        raster = _stripe_mock_raster(axis=0, wavelength_px=40)  # res=10 → wl=400 m
        corr = AlongTrackDestriping(wavelength_min=None)
        out = corr.fit_and_apply(raster)
        assert corr.meta["std_after"] < 0.5 * corr.meta["std_before"]
        assert out.data.shape == raster.data.shape

    def test_across_track_reduces_std(self):
        """Column-varying stripe → AcrossTrackDestriping cuts the residual std."""
        raster = _stripe_mock_raster(axis=1, wavelength_px=40)
        corr = AcrossTrackDestriping(wavelength_min=None)
        corr.fit(raster)
        assert corr.meta["std_after"] < 0.5 * corr.meta["std_before"]

    def test_profile_axis_differs(self):
        """The two classes collapse orthogonal axes."""
        assert AlongTrackDestriping._profile_axis == 0
        assert AcrossTrackDestriping._profile_axis == 1

    def test_fit_uses_stable_mask(self):
        """Moving-area pixels are excluded from the fitted profile."""
        raster = _stripe_mock_raster(axis=0)
        stable = np.ones(raster.data.shape, dtype=bool)
        stable[:, :10] = False  # exclude a strip of columns
        corr = AlongTrackDestriping(wavelength_min=None).fit(raster, stable_mask=stable)
        assert corr.meta["n_stable_px"] == int(stable.sum())

    def test_meta_contains_profile(self):
        raster = _stripe_mock_raster(axis=0)
        corr = AlongTrackDestriping(wavelength_min=200.0).fit(raster)
        for key in ("profile_raw", "profile_model", "band", "std_before",
                    "std_after", "freqs", "amp_raw", "amp_kept"):
            assert key in corr.meta

    def test_composable_into_pipeline(self):
        pipeline = MedianCentering() + AlongTrackDestriping() + AcrossTrackDestriping()
        assert isinstance(pipeline, CorrectionPipeline)
        assert len(pipeline.steps) == 3
