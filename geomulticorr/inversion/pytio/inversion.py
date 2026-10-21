#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# inversion.py
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
"""High-level xarray/dask driver of the TIO inversion.

Vendored verbatim from the standalone pytio port of `invers_pixel_brit_omp.f90`
(M.-P. Doin, CNRS/ISTerre). `invert_stack` is what
`TIOInversion.launch(mode="python")` calls per direction.
"""

import numpy as np
import xarray as xr

from .config import EPS_ZERO, NODATA_IN, TIOConfig
from .kernel import Operators, invert_block
from .operators import incidence_matrix


def _valid(data):
    return np.isfinite(data) & (np.abs(data) > EPS_ZERO) & (data > NODATA_IN)


def pair_image_indices(date1, date2, dates):
    """(N, 2) indices of each pair's listed dates in the image list."""
    lut = {d: i for i, d in enumerate(dates)}
    return np.array([[lut[a], lut[b]] for a, b in zip(date1, date2)])


def common_mask(stack, frac_discard):
    """Fortran `masque_commun`: pixel kept if #invalid <= int(N*frac)."""
    n = stack.sizes["pair"]
    invalid = (~_valid(stack)).sum("pair")
    return (invalid <= int(n * frac_discard)).compute()


def variance_weights(stack, frac_discard=0.6, inset=33, step=2):
    """Per-pair weights 1/sigma, exactly like `ponderation_variance`.

    sigma_k is the standard deviation of pair map k sampled every `step`
    pixels inside the common mask, `inset` pixels away from the first/last
    valid column of each row and from the top/bottom rows.
    """
    mask = common_mask(stack, frac_discard).values
    ny, nx = mask.shape
    if ny <= 2 * inset + 1 or nx <= 2 * inset + 1:
        raise ValueError(
            "weight_mode='variance' samples each pair map "
            f"{inset} px in from every edge (the Fortran ponderation_variance "
            f"region), so it needs a grid larger than {2 * inset + 1} px on both "
            f"axes — got {ny}x{nx}. Use weight_mode='file' or 'none'."
        )
    row0 = inset                                   # irec1 (0-based)
    row1 = ny - 1 - inset                          # irec2
    rows = np.arange(row0, row1 + 1, step)

    first = mask.argmax(axis=1)
    last = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    has = mask.any(axis=1)

    sig = np.zeros(stack.sizes["pair"])
    data = np.asarray(stack.transpose("pair", "y", "x").data)
    for k in range(stack.sizes["pair"]):
        vals = []
        for r in rows:
            if not has[r]:
                continue
            lo, hi = first[r] + inset, last[r] - inset
            if hi < lo:
                continue
            row = data[k, r, lo:hi + 1:step]
            m = mask[r, lo:hi + 1:step] & _valid(row)
            vals.append(row[m])
        v = np.concatenate(vals).astype(np.float64)
        sig[k] = np.sqrt(np.mean(v**2) - np.mean(v)**2)
    w = 1.0 / sig
    return w / w.mean()


def image_equalization_weights(pair_img, n_dates, rcond=1e-6):
    """Fortran `ponder_image`: per-pair factors giving each image weight ~1."""
    n_pairs = len(pair_img)
    counts = np.zeros(n_dates)
    np.add.at(counts, pair_img.ravel(), 1)
    A = np.zeros((n_dates + n_pairs, n_pairs))
    b = np.zeros(n_dates + n_pairs)
    for k, (i, j) in enumerate(pair_img):
        A[i, k] = 1.0
        A[j, k] = 1.0
        A[n_dates + k, k] = 2.0
        b[n_dates + k] = 4.0 / (counts[i] + counts[j])
    b[:n_dates] = 1.0
    w, *_ = np.linalg.lstsq(A, b, rcond=rcond)
    return w


def invert_stack(stack, dates, t, cfg: TIOConfig = None,
                 pair_weights=None) -> xr.Dataset:
    """Run the TIO inversion of a (pair, y, x) stack.

    Parameters
    ----------
    stack : xr.DataArray with dims (pair, y, x) and coords date1/date2 (str)
    dates : sequence of date labels (chronological), length S
    t     : array of dates in decimal years, length S
    cfg   : TIOConfig
    pair_weights : optional (N,) array used when cfg.weight_mode == 'file'
    """
    cfg = cfg or TIOConfig()
    dates = list(dates)
    t = np.asarray(t, np.float64)
    S, N = len(dates), stack.sizes["pair"]

    pair_img = pair_image_indices(stack["date1"].values,
                                  stack["date2"].values, dates)
    J = incidence_matrix(pair_img, S)

    w_fixed = np.ones(N)
    w_norm = np.ones(N)
    if cfg.weight_mode == "variance":
        # pair_weights may hold precomputed full-domain variance weights
        # (useful when inverting a spatial subset)
        w_fixed = (np.asarray(pair_weights, np.float64) if pair_weights
                   is not None else variance_weights(stack, cfg.frac_discard))
    elif cfg.weight_mode == "file":
        if pair_weights is None:
            raise ValueError("weight_mode='file' requires pair_weights")
        w_norm = np.asarray(pair_weights, np.float64)
    if cfg.equalize_image_weights:
        w_norm = w_norm * image_equalization_weights(pair_img, S)

    ops = Operators(
        J=J, pair_img=pair_img, t=t, w_fixed=w_fixed, w_norm=w_norm,
        quality=(np.ones(S) if cfg.image_quality is None
                 else np.asarray(cfg.image_quality, np.float64)),
        shift=(np.zeros(N) if cfg.shift is None
               else np.asarray(cfg.shift, np.float64)),
        cfg=cfg)

    stack_t = stack.transpose("y", "x", "pair")
    if stack_t.chunks is not None:
        stack_t = stack_t.chunk({"pair": -1})
    out = xr.apply_ufunc(
        invert_block, stack_t,
        kwargs={"ops": ops},
        input_core_dims=[["pair"]],
        output_core_dims=[["date"], ["date"], ["pair_out"], [], ["date"],
                          ["date"], [], [], [], []],
        dask="parallelized",
        output_dtypes=[np.float32] * 10,
        dask_gufunc_kwargs={"output_sizes": {"date": S, "pair_out": N}},
    )
    names = ["cum_disp", "cum_disp_smooth", "residual", "rms",
             "rms_per_date", "n_pairs_per_date", "n_images", "n_pairs",
             "rank_defect", "var_ini"]
    ds = xr.Dataset(dict(zip(names, out)))
    ds = ds.assign_coords(
        date=dates, time=("date", t),
        pair_out=[f"{a}-{b}" for a, b in zip(stack["date1"].values,
                                             stack["date2"].values)])
    ds["pair_weight"] = ("pair_out", w_fixed * w_norm)
    ds["cum_disp"] = ds.cum_disp.where(ds.cum_disp < 9990.0)
    ds["cum_disp_smooth"] = ds.cum_disp_smooth.where(ds.cum_disp_smooth < 9990.0)
    ds = ds.transpose("date", "pair_out", "y", "x", missing_dims="ignore")
    ds.attrs["gamma"] = cfg.gamma
    return ds
