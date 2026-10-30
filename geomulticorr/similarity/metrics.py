#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# metrics.py
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
Similarity between two co-registered images, as a predictor of correlability.

Pure NumPy/SciPy/scikit-image, no GMC imports and no georeferencing: two
arrays on the same grid in, numbers out. Works for any pair of images —
optical, cross-sensor, Landsat/Sentinel-2, hillshades.

The guiding rule: **measure what the correlator measures, at the scale it
measures it.** ASP's block matcher scores zero-mean normalised cross-
correlation (NCC) over a ``corr_kernel`` window on prefiltered images, so it is
blind to gain/offset and sees only kernel-scale texture. A metric that is
global (Pearson), radiometry-sensitive (MSE, full SSIM) or blind to spatial
arrangement (histograms) answers a different question. Those are kept here as
**diagnostics** — they explain *why* a pair is dissimilar — not as predictors.

Conventions
-----------
- ``a`` is the reference (left) image, ``b`` the secondary (right) one.
- NaN is no-data. ``mask`` (optional, boolean) selects the pixels a metric may
  use — normally **stable ground**: on moving terrain the images *should*
  differ, and that difference is the signal, not dissimilarity.
- Offsets follow image axes: a feature at ``(row, col)`` in ``a`` is found at
  ``(row + dy, col + dx)`` in ``b``. Units are pixels.
- Functions never raise on degenerate data (no valid pixel, flat image); they
  return NaN, so a batch over hundreds of pairs is not stopped by one bad tile.
  Shape mismatches and invalid parameters do raise.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, stats
from skimage.feature import match_template

from geomulticorr.similarity.preprocess import asp_prefilter, nan_gaussian

#: Column order of :func:`similarity_report` — the future similarity-frame contract.
SIMILARITY_KEYS = (
    "n_pixels",
    # global, raw intensities
    "pearson",
    "spearman",
    "mi_nats",
    "nmi",
    "linfoot",
    "nonlinear_gap",
    # radiometric diagnostics (z-scored histograms, linear fit)
    "hist_intersection",
    "hist_chi2",
    "gain",
    "offset",
    "std_ratio",
    # local structure at the kernel scale
    "zncc_local_median",
    "zncc_local_p25",
    "ssim_luminance",
    "ssim_contrast",
    "ssim_structure",
    "oc",
    "oc_axial",
    # spectral content
    "coh_mean",
    "coh_cutoff_m",
    "coh_n_tiles",
    # trial correlation (the headline)
    "trial_n_tiles",
    "trial_valid_frac",
    "trial_railed_frac",
    "trial_peak_median",
    "trial_pkr_median",
    "trial_dx_median_px",
    "trial_dy_median_px",
    "trial_dx_nmad_px",
    "trial_dy_nmad_px",
    "trial_sigma_px",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def nmad(x) -> float:
    """Normalised median absolute deviation (σ-equivalent for a Gaussian)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _check_pair(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            f"images must be 2-D and on the same grid, got {a.shape} and {b.shape}"
        )
    return a, b


def joint_valid(a, b, mask=None) -> np.ndarray:
    """Pixels finite in both images and selected by *mask*."""
    a, b = _check_pair(a, b)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(f"mask shape {mask.shape} != image shape {a.shape}")
        valid &= mask
    return valid


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _samples(a, b, mask, max_samples, rng):
    valid = joint_valid(a, b, mask)
    x = np.asarray(a, dtype=np.float64)[valid]
    y = np.asarray(b, dtype=np.float64)[valid]
    if max_samples is not None and x.size > max_samples:
        idx = _rng(rng).choice(x.size, size=max_samples, replace=False)
        x, y = x[idx], y[idx]
    return x, y


# --------------------------------------------------------------------------- #
# global metrics
# --------------------------------------------------------------------------- #
def pearson(a, b, mask=None) -> float:
    """Global Pearson correlation (= global ZNCC).

    Invariant to a linear gain/offset, but global: low frequencies
    (illumination, snow cover, one shadowed valley) dominate it, and they say
    little about kernel-scale texture. Kept as a cheap baseline.
    Note ``MSE(z(a), z(b)) = 2·(1 − r)`` for z-scored images, so MSE adds
    nothing beyond this number, and raw MSE only measures radiometry.
    """
    x, y = _samples(a, b, mask, None, None)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a, b, mask=None, max_samples=200_000, rng=0) -> float:
    """Rank correlation: invariant to any **monotonic** radiometric relation."""
    x, y = _samples(a, b, mask, max_samples, rng)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def mutual_information(a, b, mask=None, bins=64, max_samples=200_000, rng=0) -> dict:
    """Mutual information on **rank-binned** intensities.

    Equal-frequency (rank) bins make the estimate invariant to monotonic
    transforms of either image, as MI is in theory. The plug-in estimate is
    biased upwards by ``(Bx − 1)(By − 1) / 2N`` nats; that bias is subtracted
    (otherwise two independent images would score ~0.1 in ``linfoot``).

    Returns ``mi_nats``, ``nmi`` (Studholme, in [1, 2]) and ``linfoot``, the
    informational coefficient of correlation ``√(1 − e^(−2·MI))``. For a
    bivariate Gaussian ``linfoot == |ρ|``, which puts MI on Pearson's scale:
    ``linfoot − |spearman|`` is the part of the relation that is **not
    monotonic** (e.g. a contrast reversal between bands).
    """
    nan = {"mi_nats": np.nan, "nmi": np.nan, "linfoot": np.nan}
    x, y = _samples(a, b, mask, max_samples, rng)
    n = x.size
    if n < 10 * bins or x.std() == 0 or y.std() == 0:
        return nan
    bx = np.minimum((stats.rankdata(x) - 0.5) / n * bins, bins - 1).astype(np.int64)
    by = np.minimum((stats.rankdata(y) - 0.5) / n * bins, bins - 1).astype(np.int64)
    joint = np.bincount(bx * bins + by, minlength=bins * bins).reshape(bins, bins) / n
    px, py = joint.sum(axis=1), joint.sum(axis=0)

    def _h(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    hx, hy, hxy = _h(px), _h(py), _h(joint.ravel())
    bias = (np.count_nonzero(px) - 1) * (np.count_nonzero(py) - 1) / (2.0 * n)
    mi = float(max(hx + hy - hxy - bias, 0.0))
    return {
        "mi_nats": mi,
        "nmi": (hx + hy) / hxy if hxy > 0 else np.nan,
        "linfoot": float(np.sqrt(1.0 - np.exp(-2.0 * mi))),
    }


def histogram_comparison(a, b, mask=None, bins=64, max_samples=200_000, rng=0) -> dict:
    """Radiometric **diagnostic**, not a similarity predictor.

    Histograms ignore where pixels are: shuffle an image and nothing changes.
    What they do tell is how far apart the two distributions are, i.e. whether
    a histogram match is worth trying. Intensities are z-scored first (sensor
    units differ) so the comparison is of distribution *shape*.

    Returns ``hist_intersection`` (1 = identical shapes), ``hist_chi2``
    (symmetric χ², 0 = identical), and the linear fit ``b ≈ gain·a + offset``
    with ``std_ratio = σ_b / σ_a``.
    """
    out = dict.fromkeys(
        ("hist_intersection", "hist_chi2", "gain", "offset", "std_ratio"), np.nan
    )
    x, y = _samples(a, b, mask, max_samples, rng)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return out
    edges = np.linspace(-4.0, 4.0, bins + 1)
    p, _ = np.histogram((x - x.mean()) / x.std(), bins=edges)
    q, _ = np.histogram((y - y.mean()) / y.std(), bins=edges)
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    s = p + q
    nz = s > 0
    gain, offset = np.polyfit(x, y, 1)
    out.update(
        hist_intersection=float(np.minimum(p, q).sum()),
        hist_chi2=float(0.5 * ((p[nz] - q[nz]) ** 2 / s[nz]).sum()),
        gain=float(gain),
        offset=float(offset),
        std_ratio=float(y.std() / x.std()),
    )
    return out


# --------------------------------------------------------------------------- #
# local metrics
# --------------------------------------------------------------------------- #
def _local_moments(a, b, window, valid):
    w = valid.astype(np.float64)
    A = np.where(valid, a, 0.0)
    B = np.where(valid, b, 0.0)

    def f(x):
        return ndimage.uniform_filter(x, window, mode="constant")

    n = f(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = f(A) / n, f(B) / n
        va = f(A * A) / n - ma * ma
        vb = f(B * B) / n - mb * mb
        cov = f(A * B) / n - ma * mb
    return n, ma, mb, np.clip(va, 0, None), np.clip(vb, 0, None), cov


def local_zncc(a, b, window, mask=None, min_fraction=0.5) -> np.ndarray:
    """Zero-shift windowed ZNCC map — the correlator's own score, per pixel.

    Each output pixel is the NCC of the two ``window × window`` neighbourhoods
    centred on it, using only valid (masked) pixels; NaN where fewer than
    *min_fraction* of the window is valid or either side is flat. Use the
    ``corr_kernel`` as *window*. Measures at zero shift, so evaluate it on
    stable ground (or accept that motion lowers it).
    """
    a, b = _check_pair(a, b)
    valid = joint_valid(a, b, mask)
    n, _, _, va, vb, cov = _local_moments(a, b, window, valid)
    eps_a = 1e-10 * (np.nanvar(a[valid]) if valid.any() else 1.0)
    eps_b = 1e-10 * (np.nanvar(b[valid]) if valid.any() else 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    bad = (n < min_fraction) | (va <= eps_a) | (vb <= eps_b) | ~valid
    r[bad] = np.nan
    return np.clip(r, -1.0, 1.0)


def ssim_components(a, b, window, mask=None, min_fraction=0.5) -> dict:
    """Medians of SSIM's three terms over the valid pixels.

    ``l`` (luminance) and ``c`` (contrast) penalise exactly the radiometric
    differences NCC normalises away; ``s`` (structure, with ``C3 = C2/2``) is
    a regularised windowed ZNCC. Reported separately so the reason for a low
    SSIM is visible: cross-sensor pairs typically have low ``l``/``c`` and a
    perfectly usable ``s``. The full SSIM is deliberately not returned.
    """
    nan = {"ssim_luminance": np.nan, "ssim_contrast": np.nan, "ssim_structure": np.nan}
    a, b = _check_pair(a, b)
    valid = joint_valid(a, b, mask)
    if not valid.any():
        return nan
    lo = min(np.percentile(a[valid], 0.5), np.percentile(b[valid], 0.5))
    hi = max(np.percentile(a[valid], 99.5), np.percentile(b[valid], 99.5))
    L = hi - lo
    if L <= 0:
        return nan
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    c3 = c2 / 2
    n, ma, mb, va, vb, cov = _local_moments(a, b, window, valid)
    sa, sb = np.sqrt(va), np.sqrt(vb)
    lum = (2 * ma * mb + c1) / (ma**2 + mb**2 + c1)
    con = (2 * sa * sb + c2) / (va + vb + c2)
    struc = (cov + c3) / (sa * sb + c3)
    keep = valid & (n >= min_fraction)
    if not keep.any():
        return nan
    return {
        "ssim_luminance": float(np.nanmedian(lum[keep])),
        "ssim_contrast": float(np.nanmedian(con[keep])),
        "ssim_structure": float(np.nanmedian(struc[keep])),
    }


def orientation_correlation(a, b, mask=None, sigma=1.0, min_grad_quantile=0.25) -> dict:
    """Agreement of gradient **directions** (Fitch et al. 2002; Heid & Kääb 2012).

    Each pixel's gradient becomes a unit complex number, so gain, offset and
    any monotonic contrast change disappear. ``oc`` is ``mean cos(Δθ)``;
    ``oc_axial`` is ``mean cos(2Δθ)``, which also ignores a **polarity flip**
    (an edge bright-to-dark in one band and dark-to-bright in the other):
    ``oc ≈ −1, oc_axial ≈ 1`` is the signature of a contrast reversal.
    The weakest *min_grad_quantile* of gradients on either side are dropped:
    their direction is noise. *sigma* is a light pre-smoothing in pixels.
    """
    nan = {"oc": np.nan, "oc_axial": np.nan}
    a, b = _check_pair(a, b)
    valid = joint_valid(a, b, mask)
    if valid.sum() < 10:
        return nan
    sa = nan_gaussian(np.where(valid, a, np.nan), sigma)
    sb = nan_gaussian(np.where(valid, b, np.nan), sigma)
    gya, gxa = np.gradient(sa)
    gyb, gxb = np.gradient(sb)
    za, zb = gxa + 1j * gya, gxb + 1j * gyb
    ma, mb = np.abs(za), np.abs(zb)
    sel = valid & np.isfinite(ma) & np.isfinite(mb) & (ma > 0) & (mb > 0)
    if sel.sum() < 10:
        return nan
    qa = np.quantile(ma[sel], min_grad_quantile)
    qb = np.quantile(mb[sel], min_grad_quantile)
    sel &= (ma > qa) & (mb > qb)
    if sel.sum() < 10:
        return nan
    prod = (za[sel] / ma[sel]) * np.conj(zb[sel] / mb[sel])
    return {"oc": float(prod.real.mean()), "oc_axial": float((prod**2).real.mean())}


# --------------------------------------------------------------------------- #
# spectral coherence
# --------------------------------------------------------------------------- #
def spectral_coherence(
    a, b, mask=None, tile=128, res=1.0, threshold=0.5, max_tiles=400, rng=0
) -> dict:
    """Magnitude-squared coherence vs spatial frequency, from fully valid tiles.

    ``γ²(k) = |⟨A·B*⟩|² / (⟨|A|²⟩·⟨|B|²⟩)``, averaged over tiles per 2-D
    frequency, then over annuli of ``|k|``. It answers: **up to which spatial
    frequency do the two images share signal?** Coherence is unchanged by any
    linear filter applied to either image (blur, MTF, ASP's LoG), so a softer
    sensor does not lower it — only noise and genuinely different content do.
    Averaging per 2-D frequency before the annulus keeps a constant
    misregistration from cancelling.

    Returns ``k`` (cycles/m), ``wavelength_m``, ``coherence``, ``n_tiles``,
    ``noise_floor`` (≈ 1/n_tiles, the expected γ² of unrelated images),
    ``mean_coherence`` and ``cutoff_wavelength_m``: the wavelength at which
    γ² first falls below *threshold*, interpolated; ``2·res`` (Nyquist) if it
    never does, NaN if even the largest scale is incoherent. Below 4 tiles
    everything is NaN — with one tile γ² is identically 1.
    """
    a, b = _check_pair(a, b)
    valid = joint_valid(a, b, mask)
    H, W = a.shape
    corners = [
        (r, c)
        for r in range(0, H - tile + 1, tile)
        for c in range(0, W - tile + 1, tile)
        if valid[r : r + tile, c : c + tile].all()
    ]
    nbin = tile // 2 - 1
    out = {
        "k": np.arange(1, nbin + 1) / (tile * res) + 0.5 / (tile * res),
        "coherence": np.full(nbin, np.nan),
        "n_tiles": len(corners),
        "noise_floor": np.nan,
        "mean_coherence": np.nan,
        "cutoff_wavelength_m": np.nan,
    }
    out["wavelength_m"] = 1.0 / out["k"]
    if len(corners) < 4:
        return out
    if len(corners) > max_tiles:
        pick = _rng(rng).choice(len(corners), size=max_tiles, replace=False)
        corners = [corners[i] for i in sorted(pick)]
    win = np.outer(np.hanning(tile), np.hanning(tile))
    saa = np.zeros((tile, tile))
    sbb = np.zeros((tile, tile))
    sab = np.zeros((tile, tile), dtype=np.complex128)
    for r, c in corners:
        ta = a[r : r + tile, c : c + tile]
        tb = b[r : r + tile, c : c + tile]
        fa = np.fft.fft2((ta - ta.mean()) * win)
        fb = np.fft.fft2((tb - tb.mean()) * win)
        saa += np.abs(fa) ** 2
        sbb += np.abs(fb) ** 2
        sab += fa * np.conj(fb)
    with np.errstate(invalid="ignore", divide="ignore"):
        g2 = np.abs(sab) ** 2 / (saa * sbb)
    fy = np.fft.fftfreq(tile)[:, None]
    fx = np.fft.fftfreq(tile)[None, :]
    kpx = np.sqrt(fx**2 + fy**2)
    # annulus i covers [ (i+1)/tile, (i+2)/tile ) cycles/px, i = 0..nbin-1
    idx = np.floor(kpx * tile).astype(int) - 1
    coh = np.full(nbin, np.nan)
    for i in range(nbin):
        sel = (idx == i) & np.isfinite(g2)
        if sel.any():
            coh[i] = g2[sel].mean()
    out["coherence"] = coh
    out["n_tiles"] = len(corners)
    out["noise_floor"] = 1.0 / len(corners)
    out["mean_coherence"] = float(np.nanmean(coh))
    below = np.flatnonzero(coh < threshold)
    if below.size == 0:
        out["cutoff_wavelength_m"] = 2.0 * res
    elif below[0] == 0:
        out["cutoff_wavelength_m"] = np.nan
    else:
        i = below[0]
        k0, k1 = out["k"][i - 1], out["k"][i]
        c0, c1 = coh[i - 1], coh[i]
        kc = k0 + (c0 - threshold) / (c0 - c1) * (k1 - k0)
        out["cutoff_wavelength_m"] = float(1.0 / kc)
    return out


# --------------------------------------------------------------------------- #
# trial correlation
# --------------------------------------------------------------------------- #
def _parabolic(fm, f0, fp) -> float:
    d = fm - 2 * f0 + fp
    return 0.0 if d >= 0 else 0.5 * (fm - fp) / d


def trial_correlation(
    a,
    b,
    mask=None,
    kernel=21,
    search=8,
    n_tiles=200,
    min_peak=0.3,
    min_valid=0.9,
    rng=0,
    return_tiles=False,
) -> dict:
    """A miniature block-matching correlation on sampled tiles.

    The most direct predictor there is: it does what ASP's block matcher does
    (NCC of a ``kernel`` template over a ``±search`` box) on up to *n_tiles*
    non-overlapping templates sampled from the valid/stable area, with a
    parabolic sub-pixel peak. Pass prefiltered images (see
    :func:`~geomulticorr.similarity.preprocess.asp_prefilter`) so the score is
    the one ASP will compute.

    A tile **fails** when its peak is below *min_peak* or lies on the edge of
    the search box (``railed`` — a clipped or false match). On stable ground the
    true offset is the co-registration error, so ``trial_dx/dy_median_px`` is
    that error and ``trial_dx/dy_nmad_px`` is the **matching noise floor** —
    the same quantity as the precision ``c`` of ``corr_params``.

    Returns the summary keys of :data:`SIMILARITY_KEYS` starting with
    ``trial_``; with ``return_tiles=True`` also a ``"tiles"`` dict of per-tile
    arrays (``row, col, dx, dy, peak, pkr, ok``).
    """
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError(f"kernel must be an odd integer >= 3, got {kernel}")
    if search < 1:
        raise ValueError(f"search must be >= 1 pixel, got {search}")
    a, b = _check_pair(a, b)
    keys = [k for k in SIMILARITY_KEYS if k.startswith("trial_")]
    out = dict.fromkeys(keys, np.nan)
    out["trial_n_tiles"] = 0
    valid = joint_valid(a, b, mask)
    h, s = kernel // 2, int(search)
    m = h + s
    H, W = a.shape
    rows = np.arange(m, H - m, kernel)
    cols = np.arange(m, W - m, kernel)
    if rows.size == 0 or cols.size == 0:
        return out
    frac_a = ndimage.uniform_filter(valid.astype(np.float64), kernel, mode="constant")
    frac_b = ndimage.uniform_filter(
        np.isfinite(b).astype(np.float64), kernel + 2 * s, mode="constant"
    )
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    ok_c = (frac_a[rr, cc] >= min_valid) & (frac_b[rr, cc] >= min_valid)
    rr, cc = rr[ok_c], cc[ok_c]
    if rr.size == 0:
        return out
    if rr.size > n_tiles:
        pick = np.sort(_rng(rng).choice(rr.size, size=n_tiles, replace=False))
        rr, cc = rr[pick], cc[pick]

    n = rr.size
    dx, dy = np.full(n, np.nan), np.full(n, np.nan)
    peak, pkr = np.full(n, np.nan), np.full(n, np.nan)
    railed = np.zeros(n, dtype=bool)
    yy, xx = np.mgrid[-s : s + 1, -s : s + 1]
    for t in range(n):
        r, c = rr[t], cc[t]
        tpl = a[r - h : r + h + 1, c - h : c + h + 1]
        win = b[r - m : r + m + 1, c - m : c + m + 1]
        tpl = np.where(np.isfinite(tpl), tpl, np.nanmean(tpl))
        win = np.where(np.isfinite(win), win, np.nanmean(win))
        if tpl.std() == 0 or win.std() == 0:
            continue
        ncc = match_template(win, tpl)
        i, j = np.unravel_index(np.argmax(ncc), ncc.shape)
        peak[t] = ncc[i, j]
        railed[t] = i in (0, 2 * s) or j in (0, 2 * s)
        oy = ox = 0.0
        if not railed[t]:
            oy = _parabolic(ncc[i - 1, j], ncc[i, j], ncc[i + 1, j])
            ox = _parabolic(ncc[i, j - 1], ncc[i, j], ncc[i, j + 1])
        dy[t], dx[t] = i - s + oy, j - s + ox
        far = (yy - (i - s)) ** 2 + (xx - (j - s)) ** 2 > 4
        second = ncc[far].max() if far.any() else np.nan
        pkr[t] = peak[t] / max(second, 1e-3)

    scored = np.isfinite(peak)
    ok = scored & ~railed & (peak >= min_peak)
    out["trial_n_tiles"] = int(n)
    out["trial_valid_frac"] = float(ok.sum() / n)
    out["trial_railed_frac"] = float((railed & scored).sum() / n)
    if scored.any():
        out["trial_peak_median"] = float(np.median(peak[scored]))
        out["trial_pkr_median"] = float(np.median(pkr[scored]))
    if ok.sum() >= 3:
        out["trial_dx_median_px"] = float(np.median(dx[ok]))
        out["trial_dy_median_px"] = float(np.median(dy[ok]))
        out["trial_dx_nmad_px"] = nmad(dx[ok])
        out["trial_dy_nmad_px"] = nmad(dy[ok])
        out["trial_sigma_px"] = float(
            np.hypot(out["trial_dx_nmad_px"], out["trial_dy_nmad_px"])
        )
    if return_tiles:
        out["tiles"] = {
            "row": rr,
            "col": cc,
            "dx": dx,
            "dy": dy,
            "peak": peak,
            "pkr": pkr,
            "ok": ok,
        }
    return out


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def similarity_report(
    a,
    b,
    *,
    mask=None,
    res=1.0,
    kernel=21,
    search=8,
    prefilter_mode=2,
    prefilter_width=1.4,
    bins=64,
    tile=128,
    n_tiles=200,
    min_peak=0.3,
    rng=0,
) -> dict:
    """Every metric of this module for one image pair, as one flat row.

    Keys and order are :data:`SIMILARITY_KEYS`. Global, radiometric, SSIM,
    orientation and coherence metrics run on the **raw** intensities (they are
    either invariant to the prefilter or are meant to describe the raw
    radiometry); the local ZNCC and the trial correlation run on the
    **ASP-prefiltered** images, because that is what ASP scores. *kernel* is
    used both as the local-ZNCC window and the trial template; *res* (m/px)
    only converts the coherence cut-off to metres.
    """
    a, b = _check_pair(a, b)
    valid = joint_valid(a, b, mask)
    row = dict.fromkeys(SIMILARITY_KEYS, np.nan)
    row["n_pixels"] = int(valid.sum())
    row["coh_n_tiles"] = 0
    row["trial_n_tiles"] = 0
    if row["n_pixels"] == 0:
        return row

    row["pearson"] = pearson(a, b, valid)
    row["spearman"] = spearman(a, b, valid, rng=rng)
    row.update(mutual_information(a, b, valid, bins=bins, rng=rng))
    row["nonlinear_gap"] = row["linfoot"] - abs(row["spearman"])
    row.update(histogram_comparison(a, b, valid, bins=bins, rng=rng))
    row.update(ssim_components(a, b, kernel, valid))
    row.update(orientation_correlation(a, b, valid))
    coh = spectral_coherence(a, b, valid, tile=tile, res=res, rng=rng)
    row["coh_mean"] = coh["mean_coherence"]
    row["coh_cutoff_m"] = coh["cutoff_wavelength_m"]
    row["coh_n_tiles"] = coh["n_tiles"]

    pa = asp_prefilter(a, prefilter_mode, prefilter_width)
    pb = asp_prefilter(b, prefilter_mode, prefilter_width)
    z = local_zncc(pa, pb, kernel, valid)
    zv = z[np.isfinite(z)]
    if zv.size:
        row["zncc_local_median"] = float(np.median(zv))
        row["zncc_local_p25"] = float(np.percentile(zv, 25))
    row.update(
        trial_correlation(
            pa,
            pb,
            valid,
            kernel=kernel,
            search=search,
            n_tiles=n_tiles,
            min_peak=min_peak,
            rng=rng,
        )
    )
    return {k: row[k] for k in SIMILARITY_KEYS}
