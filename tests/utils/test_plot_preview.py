#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for display decimation in the correction control figures.

The plotting layer draws 10+ Mpx correlation output into panels that render at
well under 1000 px.  ``_preview`` decimates for display; these tests lock down
that it preserves georeferencing, costs nothing on small rasters, and that the
public ``plot_*`` functions still produce complete, leak-free figures with it.
"""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import geoutils as gu  # noqa: E402
import rasterio  # noqa: E402
from affine import Affine  # noqa: E402

from geomulticorr.utils.gmc_functions import (  # noqa: E402
    PLOT_PREVIEW_PX,
    _preview,
    _preview_mask,
    _preview_stride,
    plot_median_centering,
    plot_ramp_correction,
    plot_show_corrected_results,
)

CRS = rasterio.crs.CRS.from_epsg(32717)
TRANSFORM = Affine(3.0, 0.0, 740000.0, 0.0, -3.0, 9840000.0)


def _raster(shape, fill=1.0):
    rng = np.random.default_rng(0)
    data = (rng.normal(fill, 0.5, shape)).astype("float32")
    return gu.Raster.from_array(data, transform=TRANSFORM, crs=CRS, nodata=-9999)


@pytest.fixture
def big():
    """Larger than PLOT_PREVIEW_PX on both axes, so decimation always fires."""
    return _raster((1800, 2100))


@pytest.fixture
def small():
    """Well under PLOT_PREVIEW_PX, so decimation must be a no-op."""
    return _raster((150, 200))


class TestPreviewStride:
    def test_no_decimation_when_already_small(self):
        assert _preview_stride((150, 200), 1400) == 1

    def test_stride_brings_long_axis_under_target(self):
        k = _preview_stride((3158, 3708), 1400)
        assert k == 3
        assert 3708 / k <= 1400

    def test_none_target_disables(self):
        assert _preview_stride((10000, 10000), None) == 1

    @pytest.mark.parametrize("shape", [(3158, 3708), (12000, 900), (1401, 100)])
    def test_stride_always_reaches_target(self, shape):
        k = _preview_stride(shape, 1400)
        assert max(shape) / k <= 1400


class TestPreview:
    def test_small_raster_is_returned_unchanged(self, small):
        # Same object, not a copy — small rasters must pay nothing.
        assert _preview(small) is small

    def test_preview_px_none_returns_unchanged(self, big):
        assert _preview(big, None) is big

    def test_shape_is_reduced(self, big):
        p = _preview(big)
        assert max(p.shape) <= PLOT_PREVIEW_PX
        assert p.shape != big.shape

    def test_origin_and_crs_preserved(self, big):
        p = _preview(big)
        assert p.bounds.left == big.bounds.left
        assert p.bounds.top == big.bounds.top
        assert p.crs == big.crs

    def test_pixel_size_scales_with_stride(self, big):
        k = _preview_stride(big.shape, PLOT_PREVIEW_PX)
        p = _preview(big)
        assert p.res == pytest.approx((big.res[0] * k, big.res[1] * k))

    def test_values_are_sampled_not_interpolated(self, big):
        k = _preview_stride(big.shape, PLOT_PREVIEW_PX)
        p = _preview(big)
        assert np.asarray(p.data)[0, 0] == np.asarray(big.data)[0, 0]
        assert np.asarray(p.data)[1, 1] == np.asarray(big.data)[k, k]

    def test_none_raster_passes_through(self):
        # plot_show_* accept ncc=None; the helper must tolerate it.
        assert _preview(None) is None


class TestPreviewMask:
    def test_mask_stride_matches_raster_stride(self, big):
        keep = np.ones(big.shape, dtype=bool)
        assert _preview_mask(keep).shape == _preview(big).shape

    def test_none_passes_through(self):
        assert _preview_mask(None) is None

    def test_small_mask_unchanged(self):
        keep = np.ones((150, 200), dtype=bool)
        assert _preview_mask(keep).shape == (150, 200)


class TestFiguresWithPreview:
    """Every public plot_* must still build a complete figure and leak nothing."""

    def test_median_centering_renders_and_closes(self, big):
        fig = plot_median_centering(big, big, big, big, fig_name="pair")
        assert len(fig.axes) >= 6
        plt.close(fig)
        assert plt.get_fignums() == []

    def test_ramp_correction_renders(self, big):
        fig = plot_ramp_correction(big, big, big, big, fig_name="pair")
        assert len(fig.axes) >= 8
        plt.close(fig)
        assert plt.get_fignums() == []

    def test_corrected_dashboard_accepts_full_grid_stable_masks(self, big):
        # The stable masks arrive full-grid and must be decimated in step with
        # the rasters — a mismatch would raise inside _apply_keep_mask.
        keep = np.random.default_rng(0).random(big.shape) > 0.5
        fig = plot_show_corrected_results(
            big, big, None, x_stable=keep, y_stable=keep, fig_name="pair"
        )
        assert len(fig.axes) >= 5
        plt.close(fig)
        assert plt.get_fignums() == []

    def test_full_resolution_path_still_works(self, small):
        fig = plot_median_centering(small, small, small, small, preview_px=None)
        plt.close(fig)
        assert plt.get_fignums() == []
