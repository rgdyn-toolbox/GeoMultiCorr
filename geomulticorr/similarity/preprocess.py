#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# preprocess.py
# creation date: 2026-10-30.
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
"""
Image preparation before a similarity measurement.

Pure NumPy/SciPy, no GMC imports and no georeferencing — same layering as
``corrections/fit.py``. The caller puts both images on one grid first; this
module only changes pixel values.

Conventions
-----------
- Arrays are 2-D floats; no-data is ``NaN`` in and ``NaN`` out.
- Every filter is **NaN-aware** (normalised convolution): a hole does not
  bleed a fabricated value into its neighbours, and its own pixels stay NaN.
- Sizes (``sigma``, ``window``) are in **pixels of the array passed in**.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

#: Below this weight a normalised-convolution output is treated as unsupported.
_MIN_WEIGHT = 1e-6


def as_float(arr, nodata=None) -> np.ndarray:
    """Return *arr* as float64 with *nodata* (and non-finite values) as NaN.

    Masked arrays are honoured: masked pixels become NaN.
    """
    if np.ma.isMaskedArray(arr):
        out = np.ma.filled(arr.astype(np.float64), np.nan)
    else:
        out = np.array(arr, dtype=np.float64, copy=True)
    if nodata is not None and np.isfinite(nodata):
        out[out == nodata] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _normalised(a: np.ndarray, smoother) -> np.ndarray:
    valid = np.isfinite(a)
    num = smoother(np.where(valid, a, 0.0))
    den = smoother(valid.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < _MIN_WEIGHT] = np.nan
    out[~valid] = np.nan
    return out


def nan_gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores NaN pixels (normalised convolution)."""
    a = np.asarray(a, dtype=np.float64)
    if sigma <= 0:
        return a.copy()
    return _normalised(a, lambda x: ndimage.gaussian_filter(x, sigma, mode="nearest"))


def nan_uniform(a: np.ndarray, window: int) -> np.ndarray:
    """Box mean over a ``window × window`` neighbourhood, ignoring NaN pixels."""
    a = np.asarray(a, dtype=np.float64)
    return _normalised(a, lambda x: ndimage.uniform_filter(x, window, mode="nearest"))


def _fill_nan(a: np.ndarray, sigma: float) -> np.ndarray:
    """Fill holes with a local mean, then the global median, so a linear filter
    can run; the caller re-imposes the NaN afterwards."""
    filled = a.copy()
    hole = ~np.isfinite(filled)
    if not hole.any():
        return filled
    # Not _normalised: that blanks the holes, which are exactly what we fill.
    valid = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(valid, a, 0.0), sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(np.float64), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        local = num / den
    filled[hole] = local[hole]
    still = ~np.isfinite(filled)
    if still.any():
        filled[still] = np.nanmedian(a) if valid.any() else 0.0
    return filled


def antialias_sigma(res_in: float, res_out: float) -> float:
    """Gaussian σ (in *input* pixels) to apply before resampling ``res_in → res_out``.

    Uses ``(f − 1) / 2`` with ``f = res_out / res_in``, the scikit-image
    ``rescale`` convention. Returns 0 when not downsampling. Skipping this step
    aliases the finer image's high frequencies into the coarse grid, which reads
    as *dissimilarity* that no radiometric fix can remove.
    """
    f = float(res_out) / float(res_in)
    return max(0.0, (f - 1.0) / 2.0)


def asp_prefilter(a: np.ndarray, mode: int = 2, width: float = 1.4) -> np.ndarray:
    """Mirror ASP's ``prefilter-mode`` so similarity is measured on what ASP correlates.

    - ``0``: none.
    - ``1``: subtracted mean, ``a − G_width(a)`` (a high-pass).
    - ``2``: Laplacian of Gaussian of σ = *width* (ASP's default, band-pass).

    ASP calls *width* the prefilter kernel width; it is used here as the
    Gaussian σ in pixels. LoG is linear, so its sign is irrelevant to NCC.
    """
    a = np.asarray(a, dtype=np.float64)
    if mode == 0:
        return a.copy()
    if mode == 1:
        return a - nan_gaussian(a, width)
    if mode == 2:
        hole = ~np.isfinite(a)
        filled = _fill_nan(a, 3 * width)
        # SciPy's truncated LoG kernel does not sum exactly to zero and leaks
        # ~5e-5 of the DC level; centring first makes the output offset-free.
        filled -= filled.mean()
        out = ndimage.gaussian_laplace(filled, width, mode="nearest")
        out[hole] = np.nan
        return out
    raise ValueError(f"prefilter mode must be 0, 1 or 2, got {mode!r}")


def band_pass(a: np.ndarray, sigma_low: float, sigma_high: float) -> np.ndarray:
    """Difference of Gaussians: keeps wavelengths between ~2πσ_low and ~2πσ_high.

    ``sigma_low`` removes pixel noise, ``sigma_high`` removes illumination
    gradients. Requires ``sigma_low < sigma_high``.
    """
    if not 0 <= sigma_low < sigma_high:
        raise ValueError("band_pass needs 0 <= sigma_low < sigma_high")
    return nan_gaussian(a, sigma_low) - nan_gaussian(a, sigma_high)


def wallis(
    a: np.ndarray,
    window: int,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Wallis local-contrast filter (Baltsavias 1991 form).

    ``out = (a − m)·(c·s_t) / (c·s + (1 − c)·s_t) + b·m_t + (1 − b)·m``

    with ``m, s`` the local mean/std over *window*. With the defaults
    (``b = c = 1``) this is local standardisation: each neighbourhood is brought
    to mean ``target_mean`` and std ``target_std``, which removes spatially
    varying gain/offset (haze, illumination) that a global normalisation —
    including ASP's ``individually-normalize`` — leaves in place.
    """
    if window < 3:
        raise ValueError("wallis window must be >= 3 pixels")
    a = np.asarray(a, dtype=np.float64)
    m = nan_uniform(a, window)
    var = nan_uniform(a * a, window) - m * m
    s = np.sqrt(np.clip(var, 0.0, None))
    denom = contrast * s + (1.0 - contrast) * target_std
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - m) * (contrast * target_std) / denom
    out[denom <= 0] = np.nan
    return out + brightness * target_mean + (1.0 - brightness) * m


def match_histogram(
    src: np.ndarray, ref: np.ndarray, mask=None, n_quantiles: int = 1001
) -> np.ndarray:
    """Map *src* onto *ref*'s distribution by quantile mapping.

    The mapping is **learnt on** ``mask`` (typically stable ground, where the
    two images should agree) and **applied to** every finite *src* pixel.
    Fitting it on moving terrain would bake the displacement into radiometry.
    The mapping is monotonic, so rank-based metrics do not change.
    """
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    sel_s = np.isfinite(src)
    sel_r = np.isfinite(ref)
    if mask is not None:
        sel_s &= mask
        sel_r &= mask
    if not sel_s.any() or not sel_r.any():
        return np.full_like(src, np.nan)
    q = np.linspace(0.0, 1.0, n_quantiles)
    qs = np.quantile(src[sel_s], q)
    qr = np.quantile(ref[sel_r], q)
    # np.interp needs increasing x; collapse flat runs of a quantised image.
    qs, keep = np.unique(qs, return_index=True)
    qr = qr[keep]
    out = np.full_like(src, np.nan)
    fin = np.isfinite(src)
    out[fin] = np.interp(src[fin], qs, qr)
    return out
