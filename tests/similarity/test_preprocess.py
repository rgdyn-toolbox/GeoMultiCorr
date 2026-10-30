"""Preprocessing for similarity (similarity/preprocess.py): NaN-aware filters."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from geomulticorr.similarity import preprocess as pp


def _texture(shape=(128, 128), seed=0):
    return ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=shape), 2.0)


class TestAsFloat:
    def test_nodata_and_inf_to_nan(self):
        out = pp.as_float(np.array([[0, 5], [65535, 7]], dtype=np.uint16), nodata=65535)
        assert out.dtype == np.float64 and np.isnan(out[1, 0]) and out[0, 0] == 0

    def test_masked_array(self):
        m = np.ma.array([[1.0, 2.0]], mask=[[False, True]])
        assert np.isnan(pp.as_float(m)[0, 1])

    def test_does_not_mutate(self):
        a = np.array([[1.0, -9999.0]])
        pp.as_float(a, nodata=-9999.0)
        assert a[0, 1] == -9999.0


class TestNanFilters:
    def test_gaussian_matches_scipy_without_nan(self):
        a = _texture()
        np.testing.assert_allclose(
            pp.nan_gaussian(a, 1.5), ndimage.gaussian_filter(a, 1.5, mode="nearest")
        )

    def test_hole_stays_nan_and_does_not_bleed(self):
        a = np.ones((20, 20))
        a[10, 10] = np.nan
        out = pp.nan_gaussian(a, 2.0)
        assert np.isnan(out[10, 10])
        np.testing.assert_allclose(out[np.isfinite(out)], 1.0)

    def test_uniform_constant(self):
        a = np.full((10, 10), 3.0)
        a[0, :] = np.nan
        np.testing.assert_allclose(pp.nan_uniform(a, 3)[1:], 3.0)

    def test_sigma_zero_is_copy(self):
        a = _texture()
        out = pp.nan_gaussian(a, 0)
        assert out is not a and np.array_equal(out, a)


class TestAntialias:
    @pytest.mark.parametrize(
        "ri, ro, s",
        [(1.5, 3.0, 0.5), (0.5, 3.0, 2.5), (3.0, 3.0, 0.0), (3.0, 1.5, 0.0)],
    )
    def test_sigma(self, ri, ro, s):
        assert pp.antialias_sigma(ri, ro) == pytest.approx(s)


class TestPrefilter:
    def test_mode0_identity(self):
        a = _texture()
        assert np.array_equal(pp.asp_prefilter(a, 0), a)

    def test_modes_remove_constant_offset(self):
        a = _texture()
        for mode in (1, 2):
            np.testing.assert_allclose(
                pp.asp_prefilter(a + 1000.0, mode), pp.asp_prefilter(a, mode), atol=1e-8
            )

    def test_log_matches_scipy(self):
        a = _texture()
        np.testing.assert_allclose(
            pp.asp_prefilter(a, 2, 1.4),
            ndimage.gaussian_laplace(a - a.mean(), 1.4, mode="nearest"),
        )

    def test_nan_preserved(self):
        a = _texture()
        a[5:10, 5:10] = np.nan
        for mode in (1, 2):
            out = pp.asp_prefilter(a, mode)
            assert np.isnan(out[5:10, 5:10]).all() and np.isfinite(out[50:, 50:]).all()

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            pp.asp_prefilter(_texture(), 3)


class TestBandPassWallis:
    def test_band_pass_removes_constant(self):
        a = _texture()
        np.testing.assert_allclose(
            pp.band_pass(a + 50, 1, 8), pp.band_pass(a, 1, 8), atol=1e-8
        )

    def test_band_pass_order(self):
        with pytest.raises(ValueError):
            pp.band_pass(_texture(), 5, 2)

    def test_wallis_standardises_locally(self):
        """Two halves with different gain/offset come out on one scale."""
        a = _texture((256, 256))
        a = (a - a.mean()) / a.std()
        b = a.copy()
        b[:, :128] = 0.2 * a[:, :128] + 10
        b[:, 128:] = 8.0 * a[:, 128:] + 500
        out = pp.wallis(b, window=31)
        for half in (out[20:-20, 20:100], out[20:-20, 156:-20]):
            assert abs(np.nanmean(half)) < 0.2
            assert np.nanstd(half) == pytest.approx(1.0, abs=0.25)

    def test_wallis_window(self):
        with pytest.raises(ValueError):
            pp.wallis(_texture(), window=1)


class TestMatchHistogram:
    def test_maps_distribution_and_keeps_order(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(100, 100))
        ref = rng.exponential(size=(100, 100)) * 50 + 10
        out = pp.match_histogram(src, ref)
        assert np.percentile(out, 50) == pytest.approx(np.percentile(ref, 50), rel=0.02)
        order = np.argsort(src.ravel())
        assert np.all(np.diff(out.ravel()[order]) >= -1e-9)

    def test_fit_on_mask_only(self):
        src = np.arange(100.0).reshape(10, 10)
        ref = src * 2
        ref[5:] = 1e6  # "moving" half must not influence the mapping
        mask = np.zeros_like(src, bool)
        mask[:5] = True
        out = pp.match_histogram(src, ref, mask)
        assert out[:5].max() <= 2 * 49 + 1e-6

    def test_quantised_source(self):
        src = np.repeat([0.0, 1.0], 50).reshape(10, 10)
        out = pp.match_histogram(src, np.linspace(0, 1, 100).reshape(10, 10))
        assert np.isfinite(out).all()

    def test_empty_mask(self):
        out = pp.match_histogram(_texture(), _texture(), np.zeros((128, 128), bool))
        assert np.isnan(out).all()
