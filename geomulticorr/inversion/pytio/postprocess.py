#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# postprocess.py
# creation date: 2026-09-14.
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
"""Python equivalent of `lect_depl_cumule_lin.f` (vendored verbatim from pytio).

Weighted per-pixel linear regression through the inverted time series, with
iterative down-weighting of noisy epochs (APS-style), and referenced
per-date maps (TOT / DEF / APS).

The weighting scheme replicates the Fortran `linear_fit` exactly: means use
weights 1/q, covariances 1/q**2, where q is the per-image weight (initially
1, then aps_k + min(aps) after each iteration).
"""

import numpy as np
import xarray as xr


def _linear_fit(cum, t, q):
    """Vectorized transcription of subroutine linear_fit.

    cum: (S, ...) with NaN at missing epochs; t: (S,); q: (S,) weights.
    Returns slope, cst, rho, err_slope (maps ...).
    """
    S = cum.shape[0]
    shape = (S,) + (1,) * (cum.ndim - 1)
    w1 = (1.0 / q).reshape(shape)
    w2 = w1**2
    tt = t.reshape(shape)
    ok = np.isfinite(cum)
    n = ok.sum(axis=0)

    def s(x, w):
        return np.nansum(np.where(ok, x * w, 0.0), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        wei = s(np.ones_like(cum), w1)
        wei2 = s(np.ones_like(cum), w2)
        mx = s(tt * np.ones_like(cum), w1) / wei
        my = s(cum, w1) / wei
        cov = s(tt * cum, w2) / wei2 - mx * my
        varx = np.sqrt(s(tt**2 * np.ones_like(cum), w2) / wei2 - mx**2)
        vary = np.sqrt(s(cum**2, w2) / wei2 - my**2)
        slope = cov / varx**2
        rho = cov / (varx * vary)
        cst = my - slope * mx
        res2 = s((cum - slope[None] * tt - cst[None])**2, w2)
        resm = np.sqrt(res2 / wei2)
        err = resm / varx / np.sqrt(n - 2.0)
    bad = n <= 2
    for a in (slope, rho, cst, err):
        a[bad] = 0.0
    return slope, cst, rho, err


def fit_velocity(cum, t=None, n_iter=1, ref_index=0, cum_smooth=None,
                 aps_rows=None):
    """Fit u = slope*t + cst per pixel; iterate epoch re-weighting n_iter times.

    Parameters mirror `lect_depl_cumule_lin ncol nlign S niter iref`:
    cum : (date, y, x) DataArray (NaN = missing) — the `depl_cumule` cube
    t   : time axis (default: `time` coord; Fortran uses elapsed years)
    ref_index : epoch whose *model value* references the output maps (iref)
    cum_smooth : optional smoothed cube for DEF maps / slope_liss
    aps_rows : optional (jstart, jend) row slice restricting the APS stats
    """
    arr = np.asarray(cum.data, dtype=np.float64)
    t = np.asarray(cum["time"].values if t is None else t, np.float64)
    S = arr.shape[0]
    q = np.ones(S)

    for _ in range(max(n_iter, 1)):
        slope, cst, rho, err = _linear_fit(arr, t, q)
        resid = arr - (slope[None] * t.reshape(-1, 1, 1) + cst[None])
        sl = resid if aps_rows is None else resid[:, aps_rows[0]:aps_rows[1]]
        ok = np.isfinite(sl)
        aps = np.sqrt(np.nansum(np.where(ok, sl**2, 0), axis=(1, 2))
                      / ok.sum(axis=(1, 2)))
        q = aps + aps.min()

    ref = cst + slope * t[ref_index]
    tot = arr - ref[None]
    aps_maps = arr - (slope[None] * t.reshape(-1, 1, 1) + cst[None])

    coords = {"y": cum["y"], "x": cum["x"]} if "y" in cum.coords else {}
    dsdict = {
        "velocity": (("y", "x"), slope.astype(np.float32)),
        "intercept": (("y", "x"), cst.astype(np.float32)),
        "rho": (("y", "x"), rho.astype(np.float32)),
        "velocity_error": (("y", "x"), err.astype(np.float32)),
        "TOT": (("date", "y", "x"), tot.astype(np.float32)),
        "APS": (("date", "y", "x"), aps_maps.astype(np.float32)),
        "epoch_weight": (("date",), q.astype(np.float32)),
    }
    if cum_smooth is not None:
        smooth = np.asarray(cum_smooth.data, dtype=np.float64)
        sl_l, cst_l, rho_l, _ = _linear_fit(smooth, t, q)
        dsdict["velocity_smooth"] = (("y", "x"), sl_l.astype(np.float32))
        dsdict["rho_smooth"] = (("y", "x"), rho_l.astype(np.float32))
        dsdict["DEF"] = (("date", "y", "x"), (smooth - ref[None]).astype(np.float32))
    ds = xr.Dataset(dsdict, coords={**coords, "date": cum["date"].values,
                                    "time": ("date", t)})
    return ds
