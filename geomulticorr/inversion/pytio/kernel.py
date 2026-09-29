#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# kernel.py
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
"""Per-pixel inversion kernel (numpy only, dask-agnostic).

Transcribes the pixel loop of `iter_px` + `matrice_px` + `select_image` of
`invers_pixel_brit_omp.f90` (M.-P. Doin, CNRS/ISTerre). Vendored verbatim from
the standalone pytio port — its agreement with the Fortran is the asset, so
edit with a validation run at hand.
The kernel receives a block of pixels with the pair axis last, groups the
pixels by identical no-data pattern, builds/caches one system per pattern
and batch-solves all pixels of the pattern with a single pseudo-inverse
matmul. The optional robust iterations run per pixel on top.
"""

from dataclasses import dataclass

import numpy as np

from .config import (EPS_ZERO, LINK_WEIGHT, NODATA_IN, NODATA_OUT, TIOConfig)
from .operators import laplacian


@dataclass
class Operators:
    """Static, per-run quantities shared by all pixels."""
    J: np.ndarray            # (N, S-1) signed incidence matrix
    pair_img: np.ndarray     # (N, 2) image indices of each pair
    t: np.ndarray            # (S,) dates (decimal years)
    w_fixed: np.ndarray      # (N,) globally-normalized weights (variance mode)
    w_norm: np.ndarray       # (N,) weights renormalized per pixel subset
    quality: np.ndarray      # (S,) link-row quality weights (iqual)
    shift: np.ndarray        # (N,) referencing shift per pair
    cfg: TIOConfig

    @property
    def n_pairs(self):
        return self.J.shape[0]

    @property
    def n_dates(self):
        return self.J.shape[1] + 1


def _select_images(valid, pair_img, n_dates, min_pairs):
    """`select_image`: drop pairs of under-linked images, then flag images."""
    keep_pairs = valid.copy()
    counts = np.zeros(n_dates, np.int64)
    for _ in range(10):
        counts[:] = 0
        np.add.at(counts, pair_img[:, 0][keep_pairs], 1)
        np.add.at(counts, pair_img[:, 1][keep_pairs], 1)
        if min_pairs <= 1:
            break
        under = counts < min_pairs
        drop = keep_pairs & (under[pair_img[:, 0]] | under[pair_img[:, 1]])
        if not drop.any():
            break
        keep_pairs &= ~drop
    keep_img = counts >= max(min_pairs, 1)
    return keep_pairs, keep_img


class _System:
    """Assembled least-squares system for one no-data pattern."""

    __slots__ = ("keep_pairs", "keep_img", "img_pos", "col_idx", "Gd",
                 "A", "pinv", "rank_defect", "w_row", "touch", "S_inv", "N_inv")

    def __init__(self, valid, ops: Operators):
        cfg = ops.cfg
        S = len(ops.t)
        self.keep_pairs, self.keep_img = _select_images(
            valid, ops.pair_img, S, cfg.min_pairs_per_image)
        self.img_pos = np.flatnonzero(self.keep_img)
        self.S_inv = len(self.img_pos)
        self.N_inv = int(self.keep_pairs.sum())
        if self.S_inv < 3 or self.N_inv == 0:
            self.A = None      # not invertible -> pixel masked
            return

        S_inv, N_inv = self.S_inv, self.N_inv
        # increment columns: retained images among 0..S-2, first S_inv-1 of them
        self.col_idx = np.flatnonzero(self.keep_img[:S - 1])[:S_inv - 1]
        pk = np.flatnonzero(self.keep_pairs)
        self.Gd = ops.J[np.ix_(pk, self.col_idx)]

        # per-pair row weights: fixed part (no subset renorm, like sigma_int)
        # times subset-renormalized part (image equalization / user weights)
        w = ops.w_fixed[pk].copy()
        wn = ops.w_norm[pk]
        w *= wn / wn.mean()
        self.w_row = w

        if cfg.smoothing:
            ncol = (S_inv - 1) + S_inv
            nrow = N_inv + 2 * S_inv
        else:
            ncol = S_inv - 1
            nrow = N_inv + S_inv
        A = np.zeros((nrow, ncol))
        A[:N_inv, :S_inv - 1] = self.Gd * w[:, None]

        # link rows: eps*quality * (cumsum of increments - phi_s) = 0
        for s in range(S_inv):
            q = LINK_WEIGHT * ops.quality[self.img_pos[s]]
            A[N_inv + s, :S_inv - 1] = q * (self.col_idx < self.img_pos[s])
            if cfg.smoothing:
                A[N_inv + s, S_inv - 1 + s] = -q

        if cfg.smoothing:
            D = laplacian(ops.t[self.keep_img], cfg.scheme, cfg.pond_liss,
                          cfg.first_deriv_zero)
            A[N_inv + S_inv:, S_inv - 1:] = cfg.gamma * D

        self.A = A
        U, sv, Vt = np.linalg.svd(A, full_matrices=False)
        rank = int(np.count_nonzero(sv > cfg.rcond * sv[0]))
        self.rank_defect = ncol - rank
        self.pinv = (Vt[:rank].T / sv[:rank]) @ U[:, :rank].T

        # (S_inv, N_inv) bool: pair k touches image s (for per-date RMS)
        pi = ops.pair_img[pk]
        self.touch = (pi[:, 0][None, :] == self.img_pos[:, None]) | \
                     (pi[:, 1][None, :] == self.img_pos[:, None])

    def solve(self, d):
        """Batch solve; d is (N_inv, npix). Returns (m, phi, resid)."""
        b = np.zeros((self.A.shape[0], d.shape[1]))
        b[:self.N_inv] = d * self.w_row[:, None]
        x = self.pinv @ b
        m = x[:self.S_inv - 1]
        phi = x[self.S_inv - 1:2 * self.S_inv - 1] if self.A.shape[1] > self.S_inv - 1 else None
        resid = self.Gd @ m - d
        return x, m, phi, resid


def _robust_resolve(sys_: _System, d, x0, resid0, cfg: TIOConfig):
    """Optional per-pixel iterations: +-2pi unwrapping and residual reweighting.

    Mirrors the goto-33 loop of iter_px for a single pixel; d is (N_inv,).
    """
    d = d.copy()
    x, resid = x0, resid0
    for _ in range(cfg.unwrap_iterations):
        k = np.argmax(np.abs(resid))
        if resid[k] > 4.5:
            d[k] += 2.0 * 3.1416
        elif resid[k] < -4.5:
            d[k] -= 2.0 * 3.1416
        else:
            break
        x, _, _, resid = sys_.solve(d[:, None])
        x, resid = x[:, 0], resid[:, 0]
    for _ in range(cfg.n_robust_iterations):
        w = np.ones_like(resid)
        if cfg.reweight_iterations > 0:
            w = 1.0 / (cfg.reweight_scale**2 + resid**2)
        if cfg.mask_iterations > 0:
            w[np.abs(resid) > cfg.mask_residual_threshold] /= 1000.0
        w /= w.mean()
        A = sys_.A.copy()
        A[:sys_.N_inv] *= w[:, None]
        b = np.zeros(A.shape[0])
        b[:sys_.N_inv] = d * sys_.w_row * w
        x, *_ = np.linalg.lstsq(A, b, rcond=cfg.rcond)
        m = x[:sys_.S_inv - 1]
        resid = sys_.Gd @ m - d
    return x, resid


def invert_block(data: np.ndarray, ops: Operators):
    """Invert a block of pixels. data: (..., n_pairs) float.

    Returns a tuple of arrays matching the Fortran outputs:
    cum (..., S), cum_smooth (..., S), resid (..., N), rms (...),
    rms_per_date (..., S), n_pairs_per_date (..., S),
    n_images (...), n_pairs (...), rank_defect (...), var_ini (...).
    """
    cfg = ops.cfg
    N = ops.n_pairs
    S = len(ops.t)
    shp = data.shape[:-1]
    flat = data.reshape(-1, N).astype(np.float64)
    npix = flat.shape[0]

    cum = np.full((npix, S), NODATA_OUT, np.float32)
    cum_s = np.full((npix, S), NODATA_OUT, np.float32)
    resid_out = np.full((npix, N), np.nan, np.float32)
    rms = np.full(npix, np.nan, np.float32)
    rms_date = np.zeros((npix, S), np.float32)
    n_date = np.zeros((npix, S), np.float32)
    n_img = np.zeros(npix, np.float32)
    n_pair = np.zeros(npix, np.float32)
    rankd = np.full(npix, 15.0, np.float32)
    var_ini = np.zeros(npix, np.float32)

    valid = np.isfinite(flat) & (np.abs(flat) > EPS_ZERO) & (flat > NODATA_IN)
    flat = flat - ops.shift[None, :]

    # common mask: pixel kept if #invalid <= int(N * frac_discard)
    pix_ok = (N - valid.sum(axis=1)) <= int(N * cfg.frac_discard)
    if not pix_ok.any():
        outs = (cum, cum_s, resid_out, rms, rms_date, n_date, n_img, n_pair,
                rankd, var_ini)
        return tuple(o.reshape(shp + o.shape[1:]) for o in outs)

    idx_ok = np.flatnonzero(pix_ok)
    patterns, inverse = np.unique(valid[idx_ok], axis=0, return_inverse=True)

    iterate = cfg.unwrap_iterations > 0 or cfg.n_robust_iterations > 0

    for p in range(patterns.shape[0]):
        pix = idx_ok[inverse == p]
        sys_ = _System(patterns[p], ops)
        if sys_.A is None:
            continue
        pk = np.flatnonzero(sys_.keep_pairs)
        d = flat[np.ix_(pix, pk)].T                      # (N_inv, npix_p)
        x, m, phi, resid = sys_.solve(d)

        if iterate:
            for c in range(len(pix)):
                x[:, c], resid[:, c] = _robust_resolve(
                    sys_, d[:, c], x[:, c], resid[:, c], cfg)
            m = x[:sys_.S_inv - 1]
            phi = x[sys_.S_inv - 1:2 * sys_.S_inv - 1] if cfg.smoothing else None

        # solver failure guard (Fortran: NaN or |x| > 10000 -> nodata)
        bad = ~np.isfinite(x).all(axis=0) | (np.abs(x) > 1e4).any(axis=0)
        good = ~bad

        rmsout = np.sqrt((resid**2).mean(axis=0))
        keep = good.copy()
        if cfg.mask_high_rms:
            keep &= rmsout <= cfg.rms_threshold

        gpix = pix[good]
        n_img[gpix] = sys_.S_inv
        n_pair[gpix] = sys_.N_inv
        rankd[gpix] = sys_.rank_defect
        var_ini[gpix] = np.sqrt((d[:, good]**2).mean(axis=0))
        rms[gpix] = rmsout[good]
        resid_out[np.ix_(gpix, pk)] = resid[:, good].T.astype(np.float32)

        counts = sys_.touch.sum(axis=1).astype(np.float64)   # pairs per image
        rd = np.sqrt((sys_.touch @ resid[:, good]**2) / counts[:, None])
        rms_date[np.ix_(gpix, sys_.img_pos)] = rd.T.astype(np.float32)
        n_date[np.ix_(gpix, sys_.img_pos)] = counts.astype(np.float32)

        kpix = pix[keep]
        cum_sub = np.zeros((sys_.S_inv, keep.sum()))
        cum_sub[1:] = np.cumsum(m[:, keep], axis=0)
        cum[np.ix_(kpix, sys_.img_pos)] = cum_sub.T.astype(np.float32)
        if phi is not None:
            cum_s[np.ix_(kpix, sys_.img_pos)] = phi[:, keep].T.astype(np.float32)

    outs = (cum, cum_s, resid_out, rms, rms_date, n_date, n_img, n_pair,
            rankd, var_ini)
    return tuple(o.reshape(shp + o.shape[1:]) for o in outs)
