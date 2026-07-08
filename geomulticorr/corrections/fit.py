#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# fit.py
# creation date: 2026-06-30.
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
"""Pure-math fitting routines for displacement corrections.

All functions operate on plain NumPy arrays — no :class:`geoutils.Raster`,
no geospatial objects.  They are shared across multiple correction classes in
:mod:`geomulticorr.corrections.corrections` and can be called independently
on raw arrays for diagnostics or testing.

Design notes:

- Predictors are **z-score normalised** before solving for numerical stability.
- Coefficients are returned in their **original (denormalised) scale** so they
  carry physical units (m/pixel, m/m, etc.).
- The quadratic functions keep the surface in **normalised space** because
  back-converting 7-parameter quadratic coefficients to original scale is
  non-trivial; the correction surface itself is what callers use.
- All functions return ``r2`` as the last scalar so callers can log fit quality
  without repeating the residual computation.

Public API:

- :func:`compute_shift` — scalar median/mean shift, used by
  :class:`~geomulticorr.corrections.corrections.MedianCentering`.
- :func:`fit_ramp_linear` — linear spatial ramp, used by
  :class:`~geomulticorr.corrections.corrections.RampCorrection`.
- :func:`fit_linear_with_predictor` — linear ramp + external predictor, shared
  by :class:`~geomulticorr.corrections.corrections.TopoCorrection`,
  :class:`~geomulticorr.corrections.corrections.TopoRampCorrection`, and
  :class:`~geomulticorr.corrections.corrections.SlopeRampCorrection`.
- :func:`rotate_coords` — along-track projection of map coordinates for a given
  angle, used by
  :class:`~geomulticorr.corrections.corrections.DirectionalBiasCorrection`.
- :func:`fit_directional_profile` — robust 1-D bias profile along a rotated axis,
  used by
  :class:`~geomulticorr.corrections.corrections.DirectionalBiasCorrection`.
- :func:`estimate_directional_angle` — auto-estimate the seam-normal angle, used by
  :class:`~geomulticorr.corrections.corrections.DirectionalBiasCorrection`.
- :func:`fit_fourier_stripe_profile` — 1-D FFT band-pass model of a stripe/jitter
  undulation, used by
  :class:`~geomulticorr.corrections.corrections.AlongTrackDestriping` and
  :class:`~geomulticorr.corrections.corrections.AcrossTrackDestriping`.
- :func:`fit_quadratic_with_predictor` — quadratic ramp + predictor, shared by
  the same three classes.
- :func:`compute_slope` — terrain slope magnitude in degrees, used by
  :class:`~geomulticorr.corrections.corrections.SlopeRampCorrection`.
"""
from __future__ import annotations

import numpy as np
import numpy.ma as ma
from scipy.interpolate import UnivariateSpline
from scipy.signal.windows import tukey
from scipy.stats import binned_statistic


# --------------------------------------------------------------------------- #
# Centering
# --------------------------------------------------------------------------- #

def compute_shift(
    band_ma: np.ma.MaskedArray,
    fit_mask: np.ndarray,
    stat: str,
) -> float:
    """Return the scalar median or mean shift of stable pixels.

    Used by :class:`~geomulticorr.corrections.corrections.MedianCentering` to
    estimate the displacement offset over stable ground.

    :param band_ma: 2-D masked displacement array.
    :type band_ma: np.ma.MaskedArray
    :param fit_mask: Boolean array — ``True`` = stable valid pixel to include
        in the estimate.  Must have the same shape as *band_ma*.
    :type fit_mask: np.ndarray
    :param stat: Statistic to compute — ``'median'`` or ``'mean'``.
    :type stat: str
    :returns: Scalar shift in the same units as *band_ma*.
    :rtype: float

    Example::

        >>> import numpy as np, numpy.ma as ma
        >>> from geomulticorr.corrections.fit import compute_shift
        >>> band = ma.array([[0.1, 0.2], [0.3, 0.4]])
        >>> mask = np.ones((2, 2), dtype=bool)
        >>> compute_shift(band, mask, "median")
        0.25
    """
    stable = ma.array(
        band_ma.data,
        mask=np.ma.getmaskarray(band_ma) | ~fit_mask,
    )
    return float(ma.median(stable) if stat == "median" else ma.mean(stable))


# --------------------------------------------------------------------------- #
# Terrain helpers
# --------------------------------------------------------------------------- #

def compute_slope(dem_arr: np.ndarray, transform) -> np.ndarray:
    """Compute terrain slope magnitude in degrees from a DEM.

    Slope is derived from the spatial gradient of *dem_arr* using the pixel
    spacing from the affine *transform*.  Used by
    :class:`~geomulticorr.corrections.corrections.SlopeRampCorrection` to
    build the slope predictor for orthorectification parallax correction.

    :param dem_arr: 2-D DEM values on the same grid as the displacement raster.
        No-data values should be ``NaN``.
    :type dem_arr: np.ndarray
    :param transform: Affine geotransform of the displacement raster
        (``raster.transform``).  Pixel width is read from ``transform.a``
        and row height from ``transform.e`` (negative for north-up grids).
    :type transform: affine.Affine
    :returns: Slope magnitude in degrees, same shape as *dem_arr*.
    :rtype: np.ndarray
    """
    pixel_size_row = abs(transform.e)
    pixel_size_col = abs(transform.a)
    dz_drow, dz_dcol = np.gradient(dem_arr, pixel_size_row, pixel_size_col)
    return np.degrees(np.arctan(np.sqrt(dz_drow**2 + dz_dcol**2)))


# --------------------------------------------------------------------------- #
# Linear spatial ramp (no external predictor)
# --------------------------------------------------------------------------- #

def fit_ramp_linear(
    band_ma: np.ma.MaskedArray,
    fit_mask: np.ndarray,
    lambda_reg: float,
) -> tuple[float, float, float, float]:
    """Fit a linear spatial ramp on stable pixels.

    Solves for coefficients :math:`a`, :math:`b`, :math:`d` in the model:

    .. math::

        \\text{displacement} \\approx a \\cdot \\text{row} + b \\cdot \\text{col} + d

    using a normalised least-squares system with Tikhonov (L2) regularisation.
    Pixel coordinates are z-score normalised before solving; returned
    coefficients are de-normalised to original pixel-index units.

    Used by :class:`~geomulticorr.corrections.corrections.RampCorrection`.

    :param band_ma: 2-D masked displacement array.
    :type band_ma: np.ma.MaskedArray
    :param fit_mask: Boolean fit mask — ``True`` = include pixel.
        Same shape as *band_ma*.  At least 4 ``True`` pixels required.
    :type fit_mask: np.ndarray
    :param lambda_reg: L2 regularisation strength.  Typical value ``1e-6``.
    :type lambda_reg: float
    :returns: Tuple ``(a, b, d, r2)`` —

        - **a** — row slope (displacement units per pixel).
        - **b** — col slope (displacement units per pixel).
        - **d** — intercept.
        - **r2** — coefficient of determination on the fit pixels.
    :rtype: tuple[float, float, float, float]

    Example::

        >>> import numpy as np, numpy.ma as ma
        >>> from geomulticorr.corrections.fit import fit_ramp_linear
        >>> band = ma.array(np.random.default_rng(0).standard_normal((50, 50)))
        >>> mask = np.ones((50, 50), dtype=bool)
        >>> a, b, d, r2 = fit_ramp_linear(band, mask, 1e-6)
        >>> print(f"row slope={a:.4f}, col slope={b:.4f}, R²={r2:.3f}")
    """
    m1, m2 = band_ma.shape
    X1, X2 = np.mgrid[:m1, :m2]

    X1v, X2v = X1[fit_mask], X2[fit_mask]
    Yv = ma.filled(band_ma, np.nan)[fit_mask]

    X1_mean, X1_std = X1v.mean(), X1v.std()
    X2_mean, X2_std = X2v.mean(), X2v.std()
    X1n = (X1v - X1_mean) / X1_std
    X2n = (X2v - X2_mean) / X2_std

    A   = np.column_stack([np.ones(X1n.shape), X1n, X2n])
    ATA = A.T @ A + lambda_reg * np.eye(3)
    ATY = A.T @ Yv
    theta = np.linalg.solve(ATA, ATY)

    d_norm, a_norm, b_norm = theta
    a = a_norm / X1_std
    b = b_norm / X2_std
    d = d_norm - a * X1_mean - b * X2_mean

    y_pred = a_norm * X1n + b_norm * X2n + d_norm
    ss_res = np.sum((Yv - y_pred) ** 2)
    ss_tot = np.sum((Yv - Yv.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return float(a), float(b), float(d), r2


# --------------------------------------------------------------------------- #
# Linear ramp + external predictor (DEM elevation or terrain slope)
# --------------------------------------------------------------------------- #

def fit_linear_with_predictor(
    band_ma: np.ma.MaskedArray,
    predictor_arr: np.ndarray,
    fit_mask: np.ndarray,
    lambda_reg: float,
) -> tuple[float, float, float, float, np.ndarray, float]:
    """Fit a linear spatial ramp plus one external predictor on stable pixels.

    Solves for coefficients in the model:

    .. math::

        \\text{displacement} \\approx a_{row} \\cdot \\text{row}
        + b_{col} \\cdot \\text{col}
        + c_{pred} \\cdot \\text{predictor} + d

    Design matrix: ``[1, row_n, col_n, pred_n]`` (all z-score normalised).
    Returned coefficients are de-normalised to original units.  The full-grid
    correction surface is built with the de-normalised coefficients and can be
    subtracted directly from the displacement array.

    Shared by :class:`~geomulticorr.corrections.corrections.TopoCorrection`,
    :class:`~geomulticorr.corrections.corrections.TopoRampCorrection`, and
    :class:`~geomulticorr.corrections.corrections.SlopeRampCorrection`.
    The *predictor_arr* can be DEM elevation, terrain slope, or any other
    spatially co-registered scalar field.

    :param band_ma: 2-D masked displacement array.
    :type band_ma: np.ma.MaskedArray
    :param predictor_arr: Full-grid predictor values, same shape as *band_ma*.
        Typical inputs: DEM elevation (m) or terrain slope (degrees).
    :type predictor_arr: np.ndarray
    :param fit_mask: Boolean fit mask — ``True`` = include pixel.  At least
        5 ``True`` pixels required.
    :type fit_mask: np.ndarray
    :param lambda_reg: L2 regularisation strength (typical ``1e-6``).
    :type lambda_reg: float
    :returns: Tuple ``(a_row, b_col, c_pred, d, surface, r2)`` —

        - **a_row**   — row coefficient (denormalised, displacement/pixel).
        - **b_col**   — col coefficient (denormalised, displacement/pixel).
        - **c_pred**  — predictor coefficient (denormalised, displacement/unit).
        - **d**       — intercept (denormalised).
        - **surface** — 2-D correction surface, same shape as *band_ma*.
        - **r2**      — coefficient of determination.
    :rtype: tuple[float, float, float, float, np.ndarray, float]

    Example::

        >>> from geomulticorr.corrections.fit import fit_linear_with_predictor
        >>> a_row, b_col, c_dem, d, surf, r2 = fit_linear_with_predictor(
        ...     xDisp.data, dem_arr, stable_mask, lambda_reg=1e-6
        ... )
        >>> print(f"DEM coeff: {c_dem:.4f} m/m,  R²={r2:.3f}")
    """
    n_rows, n_cols = band_ma.shape
    row_grid, col_grid = np.mgrid[:n_rows, :n_cols]

    row_v  = row_grid[fit_mask]
    col_v  = col_grid[fit_mask]
    pred_v = predictor_arr[fit_mask]
    y_v    = ma.filled(band_ma, np.nan)[fit_mask]

    row_mean,  row_std  = row_v.mean(),  row_v.std()
    col_mean,  col_std  = col_v.mean(),  col_v.std()
    pred_mean, pred_std = pred_v.mean(), pred_v.std()

    row_n  = (row_v  - row_mean)  / row_std
    col_n  = (col_v  - col_mean)  / col_std
    pred_n = ((pred_v - pred_mean) / pred_std
              if pred_std > 0 else np.zeros_like(pred_v))

    A   = np.column_stack([np.ones(row_n.shape), row_n, col_n, pred_n])
    ATA = A.T @ A + lambda_reg * np.eye(4)
    ATY = A.T @ y_v
    theta = np.linalg.solve(ATA, ATY)
    d_n, a_n, b_n, c_n = theta

    a_row  = a_n / row_std
    b_col  = b_n / col_std
    c_pred = c_n / pred_std if pred_std > 0 else 0.0
    d      = d_n - a_row * row_mean - b_col * col_mean - c_pred * pred_mean

    surface = a_row * row_grid + b_col * col_grid + c_pred * predictor_arr + d

    y_pred = A @ theta
    ss_res = np.sum((y_v - y_pred) ** 2)
    ss_tot = np.sum((y_v - y_v.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return float(a_row), float(b_col), float(c_pred), float(d), surface, r2


# --------------------------------------------------------------------------- #
# Quadratic ramp + external predictor
# --------------------------------------------------------------------------- #

def fit_quadratic_with_predictor(
    band_ma: np.ma.MaskedArray,
    predictor_arr: np.ndarray,
    fit_mask: np.ndarray,
    lambda_reg: float,
) -> tuple[float, float, float, float, float, float, float, np.ndarray, float]:
    """Fit a second-order spatial ramp plus one external predictor on stable pixels.

    Solves for 7 coefficients in normalised space:

    .. math::

        \\text{displacement} \\approx g + a_2 r_n^2 + b_2 c_n^2
        + c_{xy} r_n c_n + a_x r_n + b_y c_n + f_p p_n

    where :math:`r_n`, :math:`c_n`, :math:`p_n` are z-score normalised row,
    col, and predictor values respectively.

    .. note::
       The correction *surface* is computed in normalised space and returned
       directly subtractable — no further de-normalisation is needed.
       The returned scalar coefficients are also in normalised units.

    Shared by :class:`~geomulticorr.corrections.corrections.TopoCorrection`,
    :class:`~geomulticorr.corrections.corrections.TopoRampCorrection`, and
    :class:`~geomulticorr.corrections.corrections.SlopeRampCorrection`.

    :param band_ma: 2-D masked displacement array.
    :type band_ma: np.ma.MaskedArray
    :param predictor_arr: Full-grid predictor values (DEM elevation or terrain
        slope), same shape as *band_ma*.
    :type predictor_arr: np.ndarray
    :param fit_mask: Boolean fit mask — ``True`` = include.  At least 8 pixels
        required (7 parameters + 1 degree of freedom).
    :type fit_mask: np.ndarray
    :param lambda_reg: L2 regularisation strength (typical ``1e-6``).
    :type lambda_reg: float
    :returns: Tuple ``(g, a2, b2, cxy, ax, by, fp, surface, r2)`` —

        - **g**       — bias / normalised intercept.
        - **a2**      — row² coefficient (normalised).
        - **b2**      — col² coefficient (normalised).
        - **cxy**     — row·col cross-term coefficient (normalised).
        - **ax**      — row coefficient (normalised).
        - **by**      — col coefficient (normalised).
        - **fp**      — predictor coefficient (normalised).
        - **surface** — 2-D correction surface, same shape as *band_ma*.
        - **r2**      — coefficient of determination.
    :rtype: tuple[float, float, float, float, float, float, float, np.ndarray, float]

    Example::

        >>> from geomulticorr.corrections.fit import fit_quadratic_with_predictor
        >>> g, a2, b2, cxy, ax, by, fp, surf, r2 = fit_quadratic_with_predictor(
        ...     xDisp.data, dem_arr, stable_mask, lambda_reg=1e-6
        ... )
        >>> print(f"DEM coeff (normalised): {fp:.4f},  R²={r2:.3f}")
    """
    n_rows, n_cols = band_ma.shape
    row_grid, col_grid = np.mgrid[:n_rows, :n_cols]

    row_v  = row_grid[fit_mask]
    col_v  = col_grid[fit_mask]
    pred_v = predictor_arr[fit_mask]
    y_v    = ma.filled(band_ma, np.nan)[fit_mask]

    row_mean,  row_std  = row_v.mean(),  row_v.std()
    col_mean,  col_std  = col_v.mean(),  col_v.std()
    pred_mean, pred_std = pred_v.mean(), pred_v.std()

    row_n  = (row_v  - row_mean)  / row_std
    col_n  = (col_v  - col_mean)  / col_std
    pred_n = ((pred_v - pred_mean) / pred_std
              if pred_std > 0 else np.zeros_like(pred_v))

    A = np.column_stack([
        np.ones(row_n.shape),
        row_n**2, col_n**2, row_n * col_n,
        row_n, col_n, pred_n,
    ])
    ATA = A.T @ A + lambda_reg * np.eye(7)
    ATY = A.T @ y_v
    theta = np.linalg.solve(ATA, ATY)
    g, a2, b2, cxy, ax, by, fp = theta

    row_full  = (row_grid      - row_mean)  / row_std
    col_full  = (col_grid      - col_mean)  / col_std
    pred_full = ((predictor_arr - pred_mean) / pred_std
                 if pred_std > 0 else np.zeros_like(predictor_arr))

    surface = (
        a2  * row_full**2
        + b2  * col_full**2
        + cxy * row_full * col_full
        + ax  * row_full
        + by  * col_full
        + fp  * pred_full
        + g
    )

    y_pred = A @ theta
    ss_res = np.sum((y_v - y_pred) ** 2)
    ss_tot = np.sum((y_v - y_v.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return (
        float(g), float(a2), float(b2), float(cxy),
        float(ax), float(by), float(fp),
        surface, r2,
    )

# --------------------------------------------------------------------------- #
# Directional (mosaic-seam) bias
# --------------------------------------------------------------------------- #

def rotate_coords(
    xx: np.ndarray,
    yy: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Project map coordinates onto the along-track (seam-normal) axis.

    Returns the rotated *x* coordinate

    .. math::

        s = (x - x_{min}) \\cos\\theta - (y - y_{min}) \\sin\\theta

    along which a directional bias varies.  The formula is copied from
    :func:`geoutils.raster.get_xy_rotated` so the angle convention is identical
    to xDEM's :class:`~xdem.coreg.biascorr.DirectionalBias`: *angle_deg* is
    measured from the **X (East) geographic axis, increasing clockwise**, with
    ``angle_deg=0`` → bias varies along East (vertical N-S seam) and
    ``angle_deg=90`` → bias varies along North (horizontal E-W seam).

    Used by
    :class:`~geomulticorr.corrections.corrections.DirectionalBiasCorrection`.

    :param xx: Full-grid easting coordinates (e.g. from ``raster.coords``).
    :type xx: np.ndarray
    :param yy: Full-grid northing coordinates, same shape as *xx*.
    :type yy: np.ndarray
    :param angle_deg: Seam-normal direction in degrees (see above).
    :type angle_deg: float
    :returns: Along-track coordinate ``s``, same shape as *xx*.
    :rtype: np.ndarray
    """
    ang = np.deg2rad(angle_deg)
    return (xx - np.min(xx)) * np.cos(ang) - (yy - np.min(yy)) * np.sin(ang)


def fit_directional_profile(
    s: np.ndarray,
    values: np.ndarray,
    fit_mask: np.ndarray,
    n_bins: int = 120,
    profile: str = "spline",
    smoothing: float | None = None,
    poly_order: int = 3,
    min_bin_count: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a robust 1-D bias profile along the rotated axis *s*.

    Bins ``values`` against the along-track coordinate ``s`` over stable
    pixels, takes the median per bin (robust to outliers and residual motion),
    smooths the binned profile, and evaluates it at every pixel to build a
    full 2-D correction surface.

    Used by
    :class:`~geomulticorr.corrections.corrections.DirectionalBiasCorrection`.

    :param s: Along-track coordinate (from :func:`rotate_coords`), full grid.
    :type s: np.ndarray
    :param values: Full-grid displacement values (``NaN`` where invalid).
    :type values: np.ndarray
    :param fit_mask: Boolean fit mask — ``True`` = stable valid pixel to
        include.  Same shape as *s* and *values*.
    :type fit_mask: np.ndarray
    :param n_bins: Number of bins along *s* for the median profile.
    :type n_bins: int
    :param profile: Smoothing model — ``'spline'`` (smoothing spline),
        ``'poly'`` (polynomial of order *poly_order*), or ``'bin'``
        (piecewise-linear between bin medians).
    :type profile: str
    :param smoothing: Smoothing-spline ``s`` parameter (``None`` uses the SciPy
        default).  Larger values give a smoother profile.
    :type smoothing: float or None
    :param poly_order: Polynomial order when ``profile='poly'``.
    :type poly_order: int
    :param min_bin_count: Bins with fewer stable pixels than this are dropped.
    :type min_bin_count: int
    :returns: Tuple ``(surface, centers, median, fitted)`` —

        - **surface** — 2-D correction surface, same shape as *s*.
        - **centers** — populated bin centres (1-D).
        - **median**  — binned median bias at *centers* (1-D).
        - **fitted**  — smoothed profile evaluated at *centers* (1-D).
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    :raises ValueError: If fewer than 4 populated bins remain, or *profile*
        is not one of the accepted values.
    """
    s_fit = s[fit_mask]
    v_fit = values[fit_mask]

    median, edges, _ = binned_statistic(
        s_fit, v_fit, statistic="median", bins=n_bins
    )
    count, _, _ = binned_statistic(s_fit, v_fit, statistic="count", bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    good = np.isfinite(median) & (count >= min_bin_count)
    centers, median = centers[good], median[good]
    if centers.size < 4:
        raise ValueError(
            "Directional profile has too few populated bins "
            f"({centers.size}); lower min_bin_count or n_bins."
        )

    if profile == "spline":
        spl = UnivariateSpline(centers, median, k=3, s=smoothing, ext=3)
        surface = spl(s)
        fitted = spl(centers)
    elif profile == "poly":
        coeffs = np.polyfit(centers, median, poly_order)
        surface = np.polyval(coeffs, s)
        fitted = np.polyval(coeffs, centers)
    elif profile == "bin":
        surface = np.interp(s, centers, median)
        fitted = median
    else:
        raise ValueError("profile must be 'spline', 'poly', or 'bin'")

    return surface, centers, median, fitted


def estimate_directional_angle(
    xx: np.ndarray,
    yy: np.ndarray,
    values: np.ndarray,
    fit_mask: np.ndarray,
    n_bins: int = 120,
    angle_grid: np.ndarray | None = None,
    min_bin_count: int = 20,
) -> float:
    """Auto-estimate the seam-normal angle that best explains the bias.

    Scans candidate angles, projects stable pixels onto the rotated axis
    (:func:`rotate_coords`), and selects the angle whose binned-median profile
    has the largest variance — i.e. the orientation along which the field has
    the strongest 1-D structure (the seam).  The coarse argmax is refined
    ±2° at 0.25° steps.

    Searching ``0–180°`` is sufficient because :math:`\\theta \\equiv
    \\theta + 180°` (the projection flips sign, giving the identical surface).
    The returned angle is normalised into ``(-90, 90]`` so it reads as the
    intuitive signed value (e.g. ``-10`` instead of ``170``).

    Used by
    :class:`~geomulticorr.corrections.corrections.DirectionalBiasCorrection`.

    :param xx: Full-grid easting coordinates.
    :type xx: np.ndarray
    :param yy: Full-grid northing coordinates, same shape as *xx*.
    :type yy: np.ndarray
    :param values: Full-grid displacement values (``NaN`` where invalid).
    :type values: np.ndarray
    :param fit_mask: Boolean fit mask — ``True`` = stable valid pixel.
    :type fit_mask: np.ndarray
    :param n_bins: Number of bins for the profile variance metric.
    :type n_bins: int
    :param angle_grid: Candidate angles in degrees.  Default
        ``np.arange(0, 180, 2)``.
    :type angle_grid: np.ndarray or None
    :param min_bin_count: Bins with fewer stable pixels than this are ignored.
    :type min_bin_count: int
    :returns: Estimated seam-normal angle in degrees, normalised to ``(-90, 90]``.
    :rtype: float
    """
    grid = (
        np.arange(0, 180, 2.0)
        if angle_grid is None
        else np.asarray(angle_grid, dtype=float)
    )

    def _strength(angle_deg: float) -> float:
        s = rotate_coords(xx, yy, angle_deg)
        median, edges, _ = binned_statistic(
            s[fit_mask], values[fit_mask], statistic="median", bins=n_bins
        )
        count, _, _ = binned_statistic(
            s[fit_mask], values[fit_mask], statistic="count", bins=n_bins
        )
        good = np.isfinite(median) & (count >= min_bin_count)
        return float(np.var(median[good])) if good.sum() >= 3 else -np.inf

    coarse = float(grid[int(np.argmax([_strength(a) for a in grid]))])
    fine = np.arange(coarse - 2.0, coarse + 2.001, 0.25)
    angle = float(fine[int(np.argmax([_strength(a) for a in fine]))])

    # Normalise into (-90, 90] — theta and theta+180 give the identical surface.
    if angle > 90.0:
        angle -= 180.0
    return angle


# --------------------------------------------------------------------------- #
# Fourier stripe / jitter destriping
# --------------------------------------------------------------------------- #

def fit_fourier_stripe_profile(
    profile: np.ndarray,
    spacing: float,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    detrend: bool = True,
    taper: float = 0.0,
    min_valid: int = 8,
) -> tuple[np.ndarray, dict]:
    """Model a stripe/jitter undulation in a 1-D profile via FFT band-pass.

    Given a 1-D *profile* (the median displacement across the swath, as a
    function of the along- or across-track coordinate), isolates the coherent
    undulation in a wavelength band and returns it as a full-length 1-D model
    to subtract.  Hardened against the classic failure modes of naive FFT
    destriping:

    - **NaN handling** — interior gaps are linearly interpolated and the ends
      nearest-filled (never zero-filled, which injects step discontinuities and
      spectral ringing).
    - **Detrend** — a linear trend is removed before the transform for spectral
      stability and *not* re-added (a global tilt is not jitter; it is handled
      upstream by ramp/topo corrections).
    - **Band-pass** — only wavelengths in ``[wavelength_min, wavelength_max]``
      are kept in the model, so real long-wavelength signal can be preserved.
    - **Edge taper** — an optional Tukey taper rolls the model off at the ends
      (under-correcting rather than ringing).

    Used by :class:`~geomulticorr.corrections.corrections.AlongTrackDestriping`
    and :class:`~geomulticorr.corrections.corrections.AcrossTrackDestriping`.

    :param profile: 1-D profile (may contain ``NaN`` where a whole
        row/column was masked out).
    :type profile: np.ndarray
    :param spacing: Sample spacing along the profile, in metres (the pixel size
        in the profile direction).
    :type spacing: float
    :param wavelength_min: Shortest wavelength (m) kept in the removed
        undulation.  ``None`` auto-detects the dominant spectral peak and builds
        a one-octave band around it.
    :type wavelength_min: float or None
    :param wavelength_max: Longest wavelength (m) kept in the removed undulation.
        ``None`` means no upper bound (remove everything at/above
        *wavelength_min*), matching a plain high-pass on the data.
    :type wavelength_max: float or None
    :param detrend: Remove a linear trend before the FFT.  Default ``True``.
    :type detrend: bool
    :param taper: Tukey taper fraction in ``[0, 1)`` applied to the reconstructed
        model to roll off the ends.  ``0.0`` disables it.
    :type taper: float
    :param min_valid: Minimum number of finite samples required.  Default ``8``.
    :type min_valid: int
    :returns: Tuple ``(model, meta)`` —

        - **model** — 1-D undulation to subtract, same length as *profile*.
        - **meta** — dict with ``band`` ``(wavelength_min, wavelength_max)``,
          ``wavelength_peak`` (auto mode), ``freqs``, ``amp_raw``, ``amp_kept``,
          and ``profile_filled``.
    :rtype: tuple[np.ndarray, dict]
    :raises ValueError: If fewer than *min_valid* finite samples are available,
        or *taper* is outside ``[0, 1)``.
    """
    if not 0.0 <= taper < 1.0:
        raise ValueError("taper must be in [0, 1)")

    profile = np.asarray(profile, dtype=float)
    n = profile.size
    valid = np.isfinite(profile)
    if valid.sum() < min_valid:
        raise ValueError(
            f"Too few finite samples in profile ({int(valid.sum())} < {min_valid})."
        )

    # 1. Fill NaNs — linear interpolation inside, nearest at the ends.
    idx = np.arange(n)
    filled = np.interp(idx, idx[valid], profile[valid])

    # 2. Detrend (discarded — a linear trend is not jitter).
    if detrend:
        slope, intercept = np.polyfit(idx, filled, 1)
        detrended = filled - (slope * idx + intercept)
    else:
        detrended = filled - filled.mean()

    # 3. FFT and per-component wavelengths.
    spectrum = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(n, d=spacing)
    with np.errstate(divide="ignore"):
        wavelength = np.where(freqs != 0.0, 1.0 / np.abs(freqs), np.inf)

    # 4. Resolve the band (auto-detect the dominant peak if wavelength_min is None).
    wavelength_peak = None
    if wavelength_min is None:
        # Consider only resolvable, non-DC wavelengths up to half the profile span.
        span = n * spacing
        candidate = (freqs > 0.0) & (wavelength <= span / 2.0)
        if not candidate.any():
            raise ValueError("Cannot auto-detect a jitter wavelength; set wavelength_min.")
        peak = np.argmax(np.where(candidate, np.abs(spectrum), -np.inf))
        wavelength_peak = float(wavelength[peak])
        wl_min = wavelength_peak / np.sqrt(2.0)
        wl_max = wavelength_peak * np.sqrt(2.0)
    else:
        if wavelength_min <= 0:
            raise ValueError("wavelength_min must be > 0")
        wl_min = float(wavelength_min)
        wl_max = np.inf if wavelength_max is None else float(wavelength_max)
        if wl_max <= wl_min:
            raise ValueError("wavelength_max must be > wavelength_min")

    # 5. Keep only the undulation band (never the DC component).
    keep = (wavelength >= wl_min) & (wavelength <= wl_max)
    keep[0] = False  # drop DC — centring is handled by MedianCentering
    model = np.real(np.fft.ifft(spectrum * keep))

    # 6. Optional edge taper — roll the model off at the ends.
    if taper > 0.0:
        model = model * tukey(n, alpha=taper)

    meta = {
        "band": (wl_min, None if np.isinf(wl_max) else wl_max),
        "wavelength_peak": wavelength_peak,
        "freqs": freqs,
        "amp_raw": np.abs(spectrum),
        "amp_kept": np.abs(spectrum * keep),
        "profile_filled": filled,
    }
    return model, meta