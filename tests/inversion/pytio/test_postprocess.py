#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``fit_velocity`` — the Python twin of ``lect_depl_cumule_lin``.

``TOT_<date>`` is the series re-referenced through the fitted line's value at
``ref_index`` (Fortran ``iref``), which is why a fused product must always fit
an offset between two runs: each carries its own per-pixel constant.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from geomulticorr.inversion.pytio import fit_velocity

T = np.array([0.0, 0.9, 2.1, 2.9, 4.0])
DATES = ["20200101", "20201201", "20220201", "20221201", "20240101"]


def _cube(fn):
    yy, xx = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    arr = np.stack([np.broadcast_to(fn(t, yy, xx), yy.shape) for t in T]).astype(np.float32)
    return xr.DataArray(arr, dims=("date", "y", "x"),
                        coords={"date": DATES, "time": ("date", T)})


class TestLinearFit:
    def test_exact_slope_and_zero_tot_at_the_reference(self):
        vel = lambda yy, xx: 0.5 + 0.1 * (yy * 4 + xx)
        cum = _cube(lambda t, yy, xx: vel(yy, xx) * t + 2.0)
        out = fit_velocity(cum, t=T, n_iter=1, ref_index=0)
        yy, xx = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
        np.testing.assert_allclose(out.velocity.values, vel(yy, xx), atol=1e-5)
        np.testing.assert_allclose(out.intercept.values, 2.0, atol=1e-5)
        np.testing.assert_allclose(out.rho.values, 1.0, atol=1e-5)
        np.testing.assert_allclose(out.TOT.values[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(out.TOT.values, cum.values - cum.values[0], atol=1e-5)
        np.testing.assert_allclose(out.APS.values, 0.0, atol=1e-5)
        assert out.epoch_weight.shape == (5,)
        assert out.TOT.dims == ("date", "y", "x")

    def test_reference_index_moves_the_zero(self):
        cum = _cube(lambda t, yy, xx: 1.0 * t + 3.0)
        out = fit_velocity(cum, t=T, ref_index=2)
        np.testing.assert_allclose(out.TOT.values[2], 0.0, atol=1e-5)
        np.testing.assert_allclose(out.TOT.values[0], -T[2], atol=1e-5)

    def test_missing_epochs_are_skipped(self):
        cum = _cube(lambda t, yy, xx: 1.0 * t + 3.0)
        arr = cum.values.copy()
        arr[3, 1, 1] = np.nan
        cum = cum.copy(data=arr)
        out = fit_velocity(cum, t=T)
        np.testing.assert_allclose(out.velocity.values, 1.0, atol=1e-5)
        assert np.isnan(out.TOT.values[3, 1, 1])

    def test_smoothed_cube_gives_def_maps(self):
        cum = _cube(lambda t, yy, xx: 1.0 * t + 3.0)
        out = fit_velocity(cum, t=T, cum_smooth=cum)
        assert "DEF" in out and "velocity_smooth" in out
        np.testing.assert_allclose(out.DEF.values, out.TOT.values, atol=1e-5)

    def test_time_defaults_to_the_coordinate(self):
        cum = _cube(lambda t, yy, xx: 2.0 * t + 1.0)
        np.testing.assert_allclose(fit_velocity(cum).velocity.values, 2.0, atol=1e-5)
