#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The per-pixel solver through its xarray driver, on synthetic pair stacks.

Everything is built from a known cumulative series ``u(t)`` per pixel, so the
assertions are against the truth rather than against a stored answer. Outputs
are float32, hence ``atol=1e-5``.

Two conventions inherited from the Fortran are load-bearing for the fixtures:
an exact ``0.0`` in a pair map is **no-data** (``EPS_ZERO``), so every synthetic
velocity is non-zero; and the ``ε`` link rows exist even with ``gamma=0``, so a
"no smoothing" solve is a tiny ridge toward zero displacement, not a pure
least squares — recovery is exact to ~1e-6, not to machine precision.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geomulticorr.inversion.pytio import (TIOConfig, invert_stack,
                                          variance_weights)

DATES5 = ["20200101", "20201201", "20220201", "20221201", "20240101"]
T5 = np.array([0.0, 0.9, 2.1, 2.9, 4.0])
FULL5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]   # 10 pairs


def make_stack(t, pairs, dates, *, v=0.5, ny=3, nx=4, block_offset=None, uniform=False):
    """(stack, truth) for ``u = v_pixel · (t − t0)`` plus an optional block offset.

    ``block_offset=(indices, value)`` shifts the truth of a subset of dates by a
    constant — the pair data of a *disconnected* subset is unchanged by it.
    ``uniform=True`` gives every pixel the same velocity (for tests of the
    whole-map variance weighting, where a velocity gradient would count as noise).
    """
    t = np.asarray(t, float)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    vel = v * (1 + (0.0 if uniform else 0.1) * (yy * nx + xx))   # never 0 → no false no-data
    u = vel[None] * (t - t[0])[:, None, None]
    if block_offset is not None:
        idx, off = block_offset
        u[list(idx)] += off
    data = np.stack([u[j] - u[i] for i, j in pairs])
    stack = xr.DataArray(
        data, dims=("pair", "y", "x"),
        coords={"date1": ("pair", [dates[i] for i, _ in pairs]),
                "date2": ("pair", [dates[j] for _, j in pairs])},
        name="displacement")
    return stack, u - u[0]


class TestExactRecovery:
    @pytest.mark.parametrize("gamma", [0.0, 0.003])
    def test_full_network_linear_series(self, gamma):
        stack, truth = make_stack(T5, FULL5, DATES5)
        ds = invert_stack(stack, DATES5, T5, TIOConfig(gamma=gamma, weight_mode="none")).compute()
        assert ds.cum_disp.dims == ("date", "y", "x")
        np.testing.assert_allclose(ds.cum_disp.values, truth, atol=1e-5)
        assert float(ds.rms.max()) < 1e-5
        assert (ds.n_pairs == 10).all()
        assert (ds.n_images == 5).all()
        assert (ds.rank_defect == 0).all()

    def test_smoothed_series_equals_the_series_for_linear_truth(self):
        stack, truth = make_stack(T5, FULL5, DATES5)
        ds = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="none")).compute()
        np.testing.assert_allclose(ds.cum_disp_smooth.values, truth, atol=1e-5)

    def test_backward_pairs_are_consistent(self):
        pairs = FULL5 + [(j, i) for i, j in FULL5]      # redundancy strategy
        stack, truth = make_stack(T5, pairs, DATES5)
        ds = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="none")).compute()
        np.testing.assert_allclose(ds.cum_disp.values, truth, atol=1e-5)
        assert (ds.n_pairs == 20).all()

    def test_pair_out_and_residual_shapes(self):
        stack, _ = make_stack(T5, FULL5, DATES5)
        ds = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="none")).compute()
        assert ds.residual.dims == ("pair_out", "y", "x")
        assert list(ds.pair_out.values)[0] == "20200101-20201201"
        assert np.abs(ds.residual.values).max() < 1e-5
        assert ds.attrs["gamma"] == 0.003


class TestNoDataHandling:
    def test_missing_pairs_on_some_pixels_are_grouped_and_solved(self):
        stack, truth = make_stack(T5, FULL5, DATES5)
        data = stack.values.copy()
        data[[0, 4, 7], 1, 2] = np.nan                  # one pixel loses three pairs
        data[[2], 0, 0] = 0.0                           # exact zero == no-data
        stack = stack.copy(data=data)
        ds = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="none")).compute()
        np.testing.assert_allclose(ds.cum_disp.values, truth, atol=1e-5)
        assert ds.n_pairs.values[1, 2] == 7
        assert ds.n_pairs.values[0, 0] == 9
        assert ds.n_pairs.values[2, 3] == 10
        assert np.isnan(ds.residual.values[0, 1, 2])

    def test_frac_discard_masks_pixels_with_too_few_pairs(self):
        stack, _ = make_stack(T5, FULL5, DATES5)
        data = stack.values.copy()
        data[:8, 0, 0] = np.nan                         # 8/10 invalid > 0.6
        stack = stack.copy(data=data)
        ds = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="none", frac_discard=0.6)).compute()
        assert np.isnan(ds.cum_disp.values[:, 0, 0]).all()
        assert np.isfinite(ds.cum_disp.values[:, 1, 1]).all()

    def test_image_without_pairs_is_dropped_for_that_pixel(self):
        stack, truth = make_stack(T5, FULL5, DATES5)
        data = stack.values.copy()
        touching_last = [k for k, (i, j) in enumerate(FULL5) if 4 in (i, j)]
        data[touching_last, 0, 0] = np.nan              # date 5 unconstrained there
        stack = stack.copy(data=data)
        ds = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="none")).compute()
        assert ds.n_images.values[0, 0] == 4
        assert np.isnan(ds.cum_disp.values[4, 0, 0])
        np.testing.assert_allclose(ds.cum_disp.values[:4, 0, 0], truth[:4, 0, 0], atol=1e-5)


class TestDisconnectedNetwork:
    """The fact behind ``TIOInversion.network_components()``.

    Two blocks of dates with no pair between them: the pair data are identical
    whatever the true offset between the blocks, so the solver returns the same
    series for both truths, with a closure RMS of zero — and ``rank_defect``
    stays 0 because the ε link rows keep every column non-zero. Nothing in the
    solver's own outputs flags the situation; only the graph does.
    """
    T4 = np.array([0.0, 1.0, 2.0, 3.0])
    DATES4 = ["20200101", "20210101", "20220101", "20230101"]
    PAIRS = [(0, 1), (2, 3)]

    @pytest.mark.parametrize("gamma", [0.0, 0.003])
    def test_solution_is_independent_of_the_true_offset(self, gamma):
        cfg = TIOConfig(gamma=gamma, weight_mode="none")
        runs = []
        for off in (0.0, 5.0):
            stack, _ = make_stack(self.T4, self.PAIRS, self.DATES4,
                                  block_offset=([2, 3], off))
            runs.append(invert_stack(stack, self.DATES4, self.T4, cfg).compute())
        np.testing.assert_allclose(runs[0].cum_disp.values, runs[1].cum_disp.values, atol=1e-6)
        for ds in runs:
            assert float(ds.rms.max()) < 1e-5
            assert (ds.rank_defect == 0).all()

    def test_smoothing_prior_decides_the_bridge(self):
        stack, truth = make_stack(self.T4, self.PAIRS, self.DATES4, block_offset=([2, 3], 5.0))
        smooth = invert_stack(stack, self.DATES4, self.T4,
                              TIOConfig(gamma=0.003, weight_mode="none")).compute()
        ridge = invert_stack(stack, self.DATES4, self.T4,
                             TIOConfig(gamma=0.0, weight_mode="none")).compute()
        # within each block the increments are the data's …
        for ds in (smooth, ridge):
            np.testing.assert_allclose(ds.cum_disp.values[1] - ds.cum_disp.values[0],
                                       truth[1] - truth[0], atol=1e-5)
            np.testing.assert_allclose(ds.cum_disp.values[3] - ds.cum_disp.values[2],
                                       truth[3] - truth[2], atol=1e-5)
        # … but the bridge is whatever the prior prefers, and the two priors disagree
        bridge_smooth = smooth.cum_disp.values[2] - smooth.cum_disp.values[1]
        bridge_ridge = ridge.cum_disp.values[2] - ridge.cum_disp.values[1]
        assert np.abs(bridge_smooth - bridge_ridge).min() > 0.1
        assert np.abs(bridge_smooth - (truth[2] - truth[1])).min() > 1.0


class TestWeighting:
    @staticmethod
    def _corrupted(bias=5.0, k=3):
        stack, truth = make_stack(T5, FULL5, DATES5)
        data = stack.values.copy()
        data[k] += bias                                  # one bad pair, all pixels
        return stack.copy(data=data), truth, k

    def test_robust_reweighting_shrinks_an_outlier(self):
        stack, truth, _ = self._corrupted()
        plain = invert_stack(stack, DATES5, T5,
                             TIOConfig(weight_mode="none", reweight_iterations=0)).compute()
        robust = invert_stack(stack, DATES5, T5,
                              TIOConfig(weight_mode="none", reweight_iterations=3,
                                        reweight_scale=0.2)).compute()
        err_plain = np.abs(plain.cum_disp.values - truth).max()
        err_robust = np.abs(robust.cum_disp.values - truth).max()
        assert err_plain > 0.5
        assert err_robust < err_plain / 3

    def test_file_weight_zero_removes_the_pair(self):
        stack, truth, k = self._corrupted()
        w = np.ones(len(FULL5))
        w[k] = 0.0
        ds = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="file"),
                          pair_weights=w).compute()
        np.testing.assert_allclose(ds.cum_disp.values, truth, atol=1e-5)
        np.testing.assert_allclose(ds.pair_weight.values, w)

    def test_file_mode_requires_the_weights(self):
        stack, _ = make_stack(T5, FULL5, DATES5)
        with pytest.raises(ValueError, match="pair_weights"):
            invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="file"))

    def test_file_weights_only_matter_as_ratios(self):
        stack, truth, k = self._corrupted(bias=2.0)
        w = np.ones(len(FULL5)); w[k] = 0.1
        a = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="file"), pair_weights=w).compute()
        b = invert_stack(stack, DATES5, T5, TIOConfig(weight_mode="file"), pair_weights=w * 7).compute()
        np.testing.assert_allclose(a.cum_disp.values, b.cum_disp.values, atol=1e-5)
        assert np.abs(a.cum_disp.values - truth).max() < 0.2

    def test_variance_mode_computes_its_own_weights(self):
        # variance_weights samples 33 px in from every edge → needs a real grid.
        # σ is taken over the whole map, so the truth must be spatially uniform
        # for σ to measure the injected noise rather than a velocity gradient —
        # the very reason 1/σ penalises long-baseline pairs on a moving target.
        stack, _ = make_stack(T5, FULL5, DATES5, ny=80, nx=80, uniform=True)
        rng = np.random.default_rng(0)
        noisy = stack.copy(data=stack.values + rng.normal(scale=[[[0.05]], [[0.5]]] * 5, size=stack.shape))
        ds = invert_stack(noisy, DATES5, T5, TIOConfig(weight_mode="variance")).compute()
        np.testing.assert_allclose(ds.pair_weight.values,
                                   variance_weights(noisy, 0.6), rtol=1e-6)
        # the noisier half of the pairs weighs ~10× less
        ratio = ds.pair_weight.values[0::2].mean() / ds.pair_weight.values[1::2].mean()
        assert 5 < ratio < 20
