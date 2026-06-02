#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# fit.py
# creation date: 2026-05-28.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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
- :func:`fit_quadratic_with_predictor` — quadratic ramp + predictor, shared by
  the same three classes.
- :func:`compute_slope` — terrain slope magnitude in degrees, used by
  :class:`~geomulticorr.corrections.corrections.SlopeRampCorrection`.
"""
from __future__ import annotations

import numpy as np
import numpy.ma as ma


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
