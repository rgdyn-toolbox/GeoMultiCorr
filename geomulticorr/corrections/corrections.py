#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# corrections.py
# creation date: 2026-05-13.
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
"""Composable correction steps for displacement fields.

Each correction follows the xDEM-inspired ``fit`` / ``apply`` pattern:

1. :meth:`~BaseCorrection.fit` ``(raster, stable_mask=None, **kwargs)``
   Estimate correction parameters from *raster* using only stable pixels.
   Stores the fitted correction surface in ``self._surface`` and diagnostics
   in ``self.meta``.  Returns ``self`` for chaining.

2. :meth:`~BaseCorrection.apply` ``(raster)``
   Subtract the pre-computed ``self._surface`` from *raster*.
   Raises :exc:`RuntimeError` if called before :meth:`~BaseCorrection.fit`.

3. :meth:`~BaseCorrection.fit_and_apply` ``(raster, **kwargs)``
   Convenience: :meth:`~BaseCorrection.fit` then
   :meth:`~BaseCorrection.apply` on the same raster.

Build stable-area masks with
:class:`~geomulticorr.corrections.masks.FilterPipeline`, then call
:func:`make_corrections` to apply a pipeline independently per displacement
component with per-component stable masks.

Example::

    >>> from geomulticorr.corrections import (
    ...     CCFilter, SlopeMask, StableAreaMask, OutlierFilter,
    ...     MedianCentering, RampCorrection, TopoCorrection, make_corrections,
    ... )
    >>> common = CCFilter(0.6) + SlopeMask(60) + StableAreaMask(rg_gdf)
    >>> x_stable = common.compute(xDisp, cc=cc, dem=dem) \\
    ...            & OutlierFilter((-10, 10)).generate_mask(xDisp)
    >>> y_stable = common.compute(yDisp, cc=cc, dem=dem) \\
    ...            & OutlierFilter((-5, 5)).generate_mask(yDisp)
    >>> pipeline = MedianCentering() + RampCorrection() + TopoCorrection()
    >>> xc, yc = make_corrections(xDisp, yDisp, pipeline,
    ...                            x_stable=x_stable, y_stable=y_stable, dem=dem)
"""
from __future__ import annotations

import copy
from abc import ABC
from pathlib import Path

import geopandas as gpd
import geoutils as gu
import numpy as np
import numpy.ma as ma
from rasterio.features import geometry_mask

from .fit import (
    compute_shift,
    compute_slope,
    fit_ramp_linear,
    fit_linear_with_predictor,
    fit_quadratic_with_predictor,
    rotate_coords,
    fit_directional_profile,
    estimate_directional_angle,
    fit_fourier_stripe_profile,
)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaseCorrection(ABC):
    """Abstract base class for all correction steps.

    Subclasses implement :meth:`fit` to estimate parameters and store the
    correction surface in ``self._surface``.  :meth:`apply` is a shared
    concrete implementation that subtracts ``self._surface`` from any raster.

    Shared static helpers for stable-mask resolution and raster I/O live here
    so every subclass can call them without duplication.

    Chaining corrections with ``+`` returns a :class:`CorrectionPipeline`::

        >>> pipeline = MedianCentering() + RampCorrection() + TopoCorrection()
    """

    _fit_called: bool = False
    _is_mask: bool = False
    _surface: np.ndarray | float | None = None

    # ── Shared static helpers ───────────────────────────────────────────────

    @staticmethod
    def _copy_raster(reference: gu.Raster, new_ma: np.ma.MaskedArray) -> gu.Raster:
        """Return a new :class:`geoutils.Raster` with updated data.

        The new raster inherits all georeference metadata (CRS, transform,
        no-data value) from *reference* while carrying *new_ma* as its data.

        :param reference: Source raster whose georeferencing is copied.
        :type reference: geoutils.Raster
        :param new_ma: New masked array to use as the raster data.
        :type new_ma: np.ma.MaskedArray
        :returns: A new raster with the same georeferencing as *reference*.
        :rtype: geoutils.Raster
        """
        return reference.copy(new_array=new_ma)
    # END def

    @staticmethod
    def _load_gdf(stable_mask) -> gpd.GeoDataFrame:
        """Normalise *stable_mask* to a :class:`geopandas.GeoDataFrame`.

        Accepts a GeoDataFrame directly, a :class:`geoutils.Vector`, or a
        file path pointing to any vector format supported by Fiona.

        :param stable_mask: Stable-area vector data in one of the accepted
            forms: :class:`geopandas.GeoDataFrame`, :class:`geoutils.Vector`,
            ``str``, or :class:`pathlib.Path`.
        :returns: GeoDataFrame of stable-area polygons.
        :rtype: geopandas.GeoDataFrame
        """
        if isinstance(stable_mask, gpd.GeoDataFrame):
            return stable_mask
        if isinstance(stable_mask, gu.Vector):
            return stable_mask.ds
        return gpd.read_file(stable_mask)
    # END def

    @staticmethod
    def _rasterize_moving_areas(
        gdf: gpd.GeoDataFrame,
        shape: tuple[int, int],
        transform,
    ) -> np.ndarray:
        """Rasterize moving-area polygons to a boolean exclusion mask.

        Pixels that fall *inside* any polygon in *gdf* are marked ``True``
        (they belong to a moving feature and are excluded from fitting).
        Pixels outside all polygons are ``False``.

        :param gdf: GeoDataFrame of moving-area polygons (e.g. rock-glacier
            or landslide outlines).
        :type gdf: geopandas.GeoDataFrame
        :param shape: Output raster shape ``(n_rows, n_cols)``.
        :type shape: tuple[int, int]
        :param transform: Affine geotransform of the output raster.
        :type transform: affine.Affine
        :returns: Boolean array — ``True`` = inside a moving polygon
            (exclude from fit).
        :rtype: np.ndarray
        """
        geometries = [g for g in gdf.geometry if g is not None and g.is_valid]
        if not geometries:
            return np.zeros(shape, dtype=bool)
        return geometry_mask(
            geometries=geometries,
            out_shape=shape,
            transform=transform,
            all_touched=False,
            invert=True,  # True = inside polygon = moving area
        )
    # END def

    @staticmethod
    def _resolve_stable_mask(
        stable_mask,
        band_ma: np.ma.MaskedArray,
        shape: tuple[int, int],
        transform,
        dem_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a boolean fit mask from various stable-mask input types.

        Returns a boolean array where ``True`` = pixel is valid **and** stable
        (i.e. should be included in least-squares fitting).

        *stable_mask* may be any of:

        - ``None`` — all valid (unmasked) pixels are used.
        - A **numpy boolean array** — ``True`` = stable pixel to include.
        - A **GeoDataFrame**, **str/Path**, or **geoutils.Vector** of polygons
          marking the *moving* area; pixels *outside* those polygons are stable.

        :param stable_mask: Stable-area specification in one of the accepted
            forms described above.
        :param band_ma: Displacement masked array used to derive the valid
            pixel map from its mask.
        :type band_ma: np.ma.MaskedArray
        :param shape: Raster shape ``(n_rows, n_cols)``.
        :type shape: tuple[int, int]
        :param transform: Affine geotransform of the displacement raster.
        :type transform: affine.Affine
        :param dem_arr: Optional DEM array.  When provided, pixels with
            non-finite DEM values are additionally excluded from the fit.
        :type dem_arr: np.ndarray or None
        :returns: Boolean fit mask — ``True`` = include in fit.
        :rtype: np.ndarray
        """
        valid = ~np.ma.getmaskarray(band_ma)

        if stable_mask is None:
            fit = valid
        elif isinstance(stable_mask, np.ndarray):
            fit = valid & stable_mask.astype(bool)
        else:
            gdf = BaseCorrection._load_gdf(stable_mask)
            moving = BaseCorrection._rasterize_moving_areas(gdf, shape, transform)
            fit = valid & ~moving

        if dem_arr is not None:
            fit = fit & np.isfinite(dem_arr)
        return fit
    # END def

    # --------------------------------------------------------------------------- #
    # Public interface
    # --------------------------------------------------------------------------- #

    def fit(self, raster: gu.Raster, stable_mask=None, **kwargs) -> BaseCorrection:
        """Estimate correction parameters from *raster* on stable pixels.

        Must be overridden by concrete correction subclasses.  A correct
        implementation must:

        1. Call :meth:`_resolve_stable_mask` to build the fit mask.
        2. Solve for correction parameters using functions from
           :mod:`geomulticorr.corrections.fit`.
        3. Set ``self._surface`` (array or scalar) with the correction surface.
        4. Populate ``self.meta`` with diagnostic key/value pairs.
        5. Set ``self._fit_called = True``.
        6. Return ``self`` for chaining.

        :param raster: Displacement raster to fit against.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask — see
            :meth:`_resolve_stable_mask` for accepted types.  Defaults to
            ``None`` (all valid pixels used).
        :param kwargs: Correction-specific keyword arguments forwarded to the
            concrete implementation (e.g. ``dem=`` for topographic corrections).
        :returns: ``self`` for method chaining.
        :rtype: BaseCorrection
        :raises NotImplementedError: Always, from this abstract base.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fit(). "
            "If this is a mask, call generate_mask() or apply() instead."
        )
    # END def

    def apply(self, raster: gu.Raster) -> gu.Raster:
        """Subtract ``self._surface`` from *raster*.

        The correction is applied only to valid (unmasked) pixels.  Pixels
        that become non-finite after subtraction are added to the output mask.

        :param raster: Displacement raster to correct.  Must have the same
            shape and grid as the raster used during :meth:`fit`.
        :type raster: geoutils.Raster
        :returns: Corrected displacement raster with the same georeferencing
            as *raster*.
        :rtype: geoutils.Raster
        :raises RuntimeError: If :meth:`fit` has not been called yet.
        """
        if not self._fit_called:
            raise RuntimeError(
                f"Call fit() before apply() on {type(self).__name__}."
            )
        valid     = ~np.ma.getmaskarray(raster.data)
        filled    = ma.filled(raster.data, 0.0).astype(float)
        corrected = np.where(valid, filled - self._surface, filled)
        new_mask  = np.ma.getmaskarray(raster.data) | ~np.isfinite(corrected)
        return self._copy_raster(raster, ma.array(corrected, mask=new_mask))
    # END def

    def fit_and_apply(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> gu.Raster:
        """Estimate parameters and apply the correction in a single call.

        Equivalent to::

            corr.fit(raster, stable_mask=stable_mask, **kwargs).apply(raster)

        :param raster: Displacement raster to fit against and correct.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask (see :meth:`fit`).
        :param kwargs: Forwarded to :meth:`fit`.
        :returns: Corrected displacement raster.
        :rtype: geoutils.Raster
        """
        return self.fit(raster, stable_mask=stable_mask, **kwargs).apply(raster)
    # END def

    def __add__(self, other: BaseCorrection) -> CorrectionPipeline:
        """Chain this correction with *other* into a :class:`CorrectionPipeline`.

        :param other: Next correction step.  Must not be a mask class.
        :type other: BaseCorrection
        :returns: Pipeline containing ``[self, other]`` (or merged steps if
            *other* is already a pipeline).
        :rtype: CorrectionPipeline
        :raises TypeError: If *other* is a mask class (``_is_mask=True``).
        """
        if getattr(other, "_is_mask", False):
            raise TypeError(
                f"Cannot chain correction {type(self).__name__} with mask "
                f"{type(other).__name__} using +. "
                "Build a FilterPipeline for masks and a CorrectionPipeline for "
                "corrections — they cannot be mixed."
            )
        if isinstance(other, CorrectionPipeline):
            return CorrectionPipeline([self] + other.steps)
        return CorrectionPipeline([self, other])
    # END def

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
    # END def


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class CorrectionPipeline(BaseCorrection):
    """Ordered sequence of correction steps chained via ``fit`` / ``apply``.

    Each step is fitted on the **residual** of the previous step, so
    corrections do not interfere with each other numerically.

    Build a pipeline with the ``+`` operator::

        >>> pipeline = MedianCentering() + RampCorrection() + TopoCorrection()
        >>> pipeline.fit(xDisp, stable_mask=stable, dem=dem)
        >>> xc = pipeline.apply(xDisp)

    :param steps: Ordered list of :class:`BaseCorrection` instances.
    :type steps: list[BaseCorrection]
    """

    def __init__(self, steps: list[BaseCorrection]) -> None:
        self.steps = list(steps)
    # END def

    def __add__(self, other: BaseCorrection) -> CorrectionPipeline:
        if getattr(other, "_is_mask", False):
            raise TypeError(
                f"Cannot chain correction with mask {type(other).__name__} using +. "
                "Build a FilterPipeline for masks and a CorrectionPipeline for "
                "corrections — they cannot be mixed."
            )
        if isinstance(other, CorrectionPipeline):
            return CorrectionPipeline(self.steps + other.steps)
        return CorrectionPipeline(self.steps + [other])
    # END def

    def fit(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> CorrectionPipeline:
        """Fit each step sequentially on the residual from the previous step.

        After fitting, the pipeline is ready to :meth:`apply` to any raster
        with the same grid.

        :param raster: Input displacement raster.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask forwarded to each step's
            :meth:`fit` call (see :meth:`~BaseCorrection._resolve_stable_mask`).
        :param kwargs: Extra keyword arguments forwarded to each step's
            :meth:`fit` (e.g. ``dem=dem_raster``).
        :returns: ``self`` for chaining.
        :rtype: CorrectionPipeline
        """
        residual = raster
        for step in self.steps:
            step.fit(residual, stable_mask=stable_mask, **kwargs)
            residual = step.apply(residual)
        self._fit_called = True
        return self
    # END def

    def apply(self, raster: gu.Raster) -> gu.Raster:
        """Apply all fitted steps in order to *raster*.

        :param raster: Displacement raster to correct.
        :type raster: geoutils.Raster
        :returns: Corrected displacement raster.
        :rtype: geoutils.Raster
        :raises RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fit_called:
            raise RuntimeError("Call fit() before apply() on CorrectionPipeline.")
        for step in self.steps:
            raster = step.apply(raster)
        return raster
    # END def

    def fit_and_apply(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> gu.Raster:
        """Fit and apply the pipeline in a single call.

        :param raster: Displacement raster to fit against and correct.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask (see :meth:`fit`).
        :param kwargs: Forwarded to :meth:`fit`.
        :returns: Corrected displacement raster.
        :rtype: geoutils.Raster
        """
        return self.fit(raster, stable_mask=stable_mask, **kwargs).apply(raster)
    # END def

    @property
    def meta(self) -> dict:
        """Per-step diagnostics keyed by step index.

        :returns: Dictionary ``{i: {"step": class_name, "params": meta_dict}}``
            for each step in the pipeline.
        :rtype: dict
        """
        return {
            i: {"step": type(s).__name__, "params": getattr(s, "meta", {})}
            for i, s in enumerate(self.steps)
        }
    # END def

    def __repr__(self) -> str:
        names = " -> ".join(type(s).__name__ for s in self.steps)
        return f"CorrectionPipeline([{names}])"
    # END def


# --------------------------------------------------------------------------- #
# MedianCentering
# --------------------------------------------------------------------------- #
class MedianCentering(BaseCorrection):
    """Subtract the median (or mean) displacement offset from the field.

    The scalar shift is estimated only from stable pixels when *stable_mask*
    is provided; if omitted, all valid pixels are used.  The shift is then
    subtracted from the full field (including moving-area pixels), bringing
    the stable-ground median to zero.

    :param stat: Statistic used to estimate the shift.
        ``'median'`` (default) is robust to outliers; ``'mean'`` is faster
        but sensitive to skewed distributions.
    :type stat: str

    Example::

        >>> corr = MedianCentering()
        >>> corr.fit(xDisp, stable_mask=stable)
        >>> print(corr.meta)    # {"shift": -0.042}
        >>> xc = corr.apply(xDisp)
    """

    def __init__(self, stat: str = "median") -> None:
        if stat not in ("median", "mean"):
            raise ValueError("stat must be 'median' or 'mean'")
        self.stat = stat
        self.meta: dict = {}
    # END def

    def fit(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> MedianCentering:
        """Compute the scalar shift from stable pixels.

        Sets ``self._surface`` to the estimated scalar (broadcastable over
        the full array in :meth:`~BaseCorrection.apply`).

        :param raster: Displacement raster to centre.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask — see
            :meth:`~BaseCorrection._resolve_stable_mask` for accepted types.
        :returns: ``self`` for chaining.
        :rtype: MedianCentering
        """
        fit_mask = self._resolve_stable_mask(
            stable_mask, raster.data, raster.data.shape, raster.transform
        )
        self._surface = compute_shift(raster.data, fit_mask, self.stat)
        self.meta = {"shift": self._surface}
        self._fit_called = True
        return self
    # END def
    # apply() inherited from BaseCorrection — scalar broadcasts over the array


# --------------------------------------------------------------------------- #
# RampCorrection
# --------------------------------------------------------------------------- #
class RampCorrection(BaseCorrection):
    """Remove a linear spatial ramp from the displacement field.

    Fits ``a·row + b·col + d`` on stable pixels using normalised least squares
    with Tikhonov (L2) regularisation, then subtracts the ramp from the full
    field.  No DEM is required — use :class:`TopoCorrection` when the ramp
    is correlated with topography.

    :param lambda_reg: L2 regularisation strength.  Default ``1e-6``.
    :type lambda_reg: float

    Example::

        >>> corr = RampCorrection()
        >>> corr.fit(xDisp, stable_mask=stable)
        >>> print(corr.meta)    # {"a": ..., "b": ..., "d": ..., "r2": ...}
        >>> xc = corr.apply(xDisp)
        >>> yc = corr.apply(yDisp)   # same fitted ramp applied to NS component
    """

    def __init__(self, lambda_reg: float = 1e-6) -> None:
        self.lambda_reg = lambda_reg
        self.meta: dict = {}
    # END def

    def fit(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> RampCorrection:
        """Fit the linear ramp on stable pixels.

        :param raster: Displacement raster from which to estimate the ramp.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask (see
            :meth:`~BaseCorrection._resolve_stable_mask`).
        :returns: ``self`` for chaining.
        :rtype: RampCorrection
        :raises ValueError: If fewer than 4 valid stable pixels are available.
        """
        fit_mask = self._resolve_stable_mask(
            stable_mask, raster.data, raster.data.shape, raster.transform
        )
        if fit_mask.sum() < 4:
            raise ValueError("Too few valid pixels for RampCorrection (need ≥ 4).")

        a, b, d, r2 = fit_ramp_linear(raster.data, fit_mask, self.lambda_reg)
        n_rows, n_cols = raster.data.shape
        row_g, col_g = np.mgrid[:n_rows, :n_cols]
        self._surface = a * row_g + b * col_g + d
        self.meta = {"a": a, "b": b, "d": d, "r2": r2}
        self._fit_called = True
        return self
    # END def


# --------------------------------------------------------------------------- #
# TopoCorrection
# --------------------------------------------------------------------------- #
class TopoCorrection(BaseCorrection):
    """Remove a ramp correlated with DEM elevation from the displacement field.

    Fits spatial coordinates (row, col) **and** DEM elevation jointly in a
    single least-squares step, then subtracts the trend.  Two polynomial
    orders are available:

    - ``'linear'`` — fits ``a·row + b·col + c·DEM + d``  (4 parameters).
    - ``'quadratic'`` — fits a 7-parameter model with second-order spatial
      terms plus the DEM predictor (default).

    The DEM must be pre-resampled to the same grid as the displacement field.

    :param order: Polynomial order — ``'linear'`` or ``'quadratic'`` (default).
    :type order: str
    :param lambda_reg: L2 regularisation strength.  Default ``1e-6``.
    :type lambda_reg: float

    Example::

        >>> corr = TopoCorrection(order="linear")
        >>> corr.fit(xDisp, stable_mask=stable, dem=dem)
        >>> print(corr.meta)    # {"row_slope": ..., "dem_slope": ..., "r2": ...}
        >>> xc = corr.apply(xDisp)
    """

    def __init__(self, order: str = "quadratic", lambda_reg: float = 1e-6) -> None:
        if order not in ("linear", "quadratic"):
            raise ValueError("order must be 'linear' or 'quadratic'")
        self.order = order
        self.lambda_reg = lambda_reg
        self.meta: dict = {}
    # END def

    def fit(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> TopoCorrection:
        """Fit the ramp + DEM elevation trend on stable pixels.

        :param raster: Displacement raster to correct.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask (see
            :meth:`~BaseCorrection._resolve_stable_mask`).
        :param kwargs: Must include ``dem`` — a :class:`geoutils.Raster` DEM
            pre-resampled to the same grid as *raster*.
        :returns: ``self`` for chaining.
        :rtype: TopoCorrection
        :raises ValueError: If ``dem`` is not provided, or if too few valid
            stable pixels are available for the chosen *order*.
        """
        dem: gu.Raster = kwargs.get("dem")
        if dem is None:
            raise ValueError("TopoCorrection.fit() requires dem= kwarg.")
        if dem.data.shape != raster.data.shape:
            dem = dem.reproject(ref=raster)
        dem_arr  = ma.filled(dem.data, np.nan).astype(float)
        fit_mask = self._resolve_stable_mask(
            stable_mask, raster.data, raster.data.shape, raster.transform, dem_arr
        )
        min_px = 5 if self.order == "linear" else 8
        if fit_mask.sum() < min_px:
            raise ValueError(
                f"Too few valid pixels for TopoCorrection(order='{self.order}') "
                f"(need ≥ {min_px})."
            )
        if self.order == "linear":
            a_row, b_col, c_pred, d, self._surface, r2 = fit_linear_with_predictor(
                raster.data, dem_arr, fit_mask, self.lambda_reg
            )
            self.meta = {
                "row_slope": a_row, "col_slope": b_col,
                "dem_slope": c_pred, "intercept": d, "r2": r2,
            }
        else:
            g, a2, b2, cxy, ax, by, fp, self._surface, r2 = fit_quadratic_with_predictor(
                raster.data, dem_arr, fit_mask, self.lambda_reg
            )
            self.meta = {
                "c_bias":   g,   "c_row2":   a2,
                "c_col2":   b2,  "c_rowcol": cxy,
                "c_row":    ax,  "c_col":    by,
                "c_dem":    fp,  "r2":       r2,
            }
        self._fit_called = True
        return self
    # END def


# --------------------------------------------------------------------------- #
# TopoRampCorrection
# --------------------------------------------------------------------------- #
class TopoRampCorrection(BaseCorrection):
    """Remove a joint spatial ramp + DEM-correlated trend from the displacement field.

    Fits row/col pixel coordinates **and** DEM elevation jointly in a single
    least-squares step.  This is the OOP equivalent of the classic two-function
    ``apply_ramp_topo_correction_1 / _2`` pattern.

    Two polynomial orders:

    - ``'linear'`` — fits ``a·row + b·col + c·DEM + d``  (4 parameters, default).
    - ``'quadratic'`` — fits a 7-parameter model with second-order spatial
      terms plus the DEM predictor.

    All predictors are normalised before solving; coefficients in ``self.meta``
    are in their original (denormalised) scale.

    .. note::
       When the DEM is the dominant predictor of displacement errors (e.g.
       orthorectification parallax on steep terrain), prefer
       :class:`SlopeRampCorrection`.  Use this class when residual errors
       correlate more with **elevation** than with slope.

    :param order: Polynomial order — ``'linear'`` (default) or ``'quadratic'``.
    :type order: str
    :param lambda_reg: L2 regularisation strength.  Default ``1e-6``.
    :type lambda_reg: float

    Example::

        >>> corr = TopoRampCorrection(order="linear")
        >>> corr.fit(xDisp, stable_mask=stable, dem=dem)
        >>> print(corr.meta)   # {"a": ..., "b": ..., "c": ..., "d": ..., "r2": ...}
        >>> xc = corr.apply(xDisp)
        >>> yc = corr.apply(yDisp)   # reuse fitted surface on NS component
    """

    def __init__(self, order: str = "linear", lambda_reg: float = 1e-6) -> None:
        if order not in ("linear", "quadratic"):
            raise ValueError("order must be 'linear' or 'quadratic'")
        self.order = order
        self.lambda_reg = lambda_reg
        self.meta: dict = {}
    # END def

    def fit(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> TopoRampCorrection:
        """Fit the ramp + DEM elevation trend on stable pixels.

        :param raster: Displacement raster to correct.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask (see
            :meth:`~BaseCorrection._resolve_stable_mask`).
        :param kwargs: Must include ``dem`` — a :class:`geoutils.Raster` DEM
            pre-resampled to the same grid as *raster*.
        :returns: ``self`` for chaining.
        :rtype: TopoRampCorrection
        :raises ValueError: If ``dem`` is not provided, or if too few valid
            stable pixels are available for the chosen *order*.
        """
        dem: gu.Raster = kwargs.get("dem")
        if dem is None:
            raise ValueError("TopoRampCorrection.fit() requires dem= kwarg.")
        if dem.data.shape != raster.data.shape:
            dem = dem.reproject(ref=raster)
        dem_arr  = ma.filled(dem.data, np.nan).astype(float)
        fit_mask = self._resolve_stable_mask(
            stable_mask, raster.data, raster.data.shape, raster.transform, dem_arr
        )
        min_px = 4 if self.order == "linear" else 8
        if fit_mask.sum() < min_px:
            raise ValueError(
                f"Too few valid pixels for TopoRampCorrection(order='{self.order}') "
                f"(need ≥ {min_px})."
            )
        if self.order == "linear":
            a_row, b_col, c_pred, d, self._surface, r2 = fit_linear_with_predictor(
                raster.data, dem_arr, fit_mask, self.lambda_reg
            )
            self.meta = {"a": a_row, "b": b_col, "c": c_pred, "d": d, "r2": r2}
        else:
            g, a2, b2, cxy, ax, by, fp, self._surface, r2 = fit_quadratic_with_predictor(
                raster.data, dem_arr, fit_mask, self.lambda_reg
            )
            self.meta = {
                "g": g, "a2": a2, "b2": b2, "cxy": cxy,
                "ax": ax, "by": by, "fz": fp, "r2": r2,
            }
        self._fit_called = True
        return self
    # END def

# --------------------------------------------------------------------------- #
# DirectionalBiasCorrection
# --------------------------------------------------------------------------- #
class DirectionalBiasCorrection(BaseCorrection):
    """Remove a directional (mosaic-seam) bias from the displacement field.

    Models a 1-D bias that varies *across* an oblique seam and is ~constant
    *along* it — typical of optical mosaics where two image tiles are joined.
    The bias is fitted as a robust median profile along the seam-normal axis

    .. math::

        s = (x - x_{min}) \\cos\\theta - (y - y_{min}) \\sin\\theta

    on stable pixels only, then broadcast back to 2-D and subtracted.

    The angle convention is identical to xDEM's
    :class:`~xdem.coreg.biascorr.DirectionalBias`: *angle* is measured from the
    **X (East) geographic axis, increasing clockwise**, in **map space**
    (``angle=0`` → vertical N-S seam; ``angle=90`` → horizontal E-W seam).
    When *angle* is ``None`` the seam-normal direction is estimated
    automatically and reported in ``meta`` normalised to ``(-90, 90]``.

    :param angle: Seam-normal direction in degrees, or ``None`` to auto-estimate.
    :type angle: float or None
    :param n_bins: Number of bins along *s* for the median profile.  Default ``120``.
    :type n_bins: int
    :param profile: Profile-smoothing model — ``'spline'`` (default, smoothing
        spline; good for a broad seam band), ``'poly'`` (polynomial ramp), or
        ``'bin'`` (piecewise-linear; most faithful to a sharp step).
    :type profile: str
    :param smoothing: Smoothing-spline ``s`` parameter (``None`` = SciPy default).
        Larger = smoother.  Only used when ``profile='spline'``.
    :type smoothing: float or None
    :param poly_order: Polynomial order when ``profile='poly'``.  Default ``3``.
    :type poly_order: int
    :param min_bin_count: Bins with fewer stable pixels than this are dropped.
        Default ``20``.
    :type min_bin_count: int
    :param angle_grid: Candidate angles (degrees) for auto-estimation.  ``None``
        uses ``np.arange(0, 180, 2)``.
    :type angle_grid: np.ndarray or None
    :raises ValueError: If *profile* is invalid, or numeric parameters are out
        of range.

    Example::

        >>> corr = DirectionalBiasCorrection(angle=None, profile="bin")
        >>> corr.fit(xDisp, stable_mask=stable)
        >>> print(corr.meta["angle_deg"], corr.meta["std_before"], corr.meta["std_after"])
        >>> xc = corr.apply(xDisp)
        >>> # fix the estimated angle, then apply to the NS component too
        >>> yc = DirectionalBiasCorrection(angle=corr.meta["angle_deg"]) \\
        ...          .fit_and_apply(yDisp, stable_mask=stable_ns)
    """

    def __init__(
        self,
        angle: float | None = None,
        n_bins: int = 120,
        profile: str = "spline",
        smoothing: float | None = None,
        poly_order: int = 3,
        min_bin_count: int = 20,
        angle_grid=None,
    ) -> None:
        if profile not in ("spline", "poly", "bin"):
            raise ValueError("profile must be 'spline', 'poly', or 'bin'")
        if n_bins < 4:
            raise ValueError("n_bins must be >= 4")
        if poly_order < 1:
            raise ValueError("poly_order must be >= 1")
        if min_bin_count < 1:
            raise ValueError("min_bin_count must be >= 1")
        if angle is not None and not isinstance(angle, (int, float)):
            raise ValueError("angle must be a number or None")
        self.angle = float(angle) if angle is not None else None
        self.n_bins = n_bins
        self.profile = profile
        self.smoothing = smoothing
        self.poly_order = poly_order
        self.min_bin_count = min_bin_count
        self.angle_grid = angle_grid
        self.meta: dict = {}
    # END def

    def fit(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> DirectionalBiasCorrection:
        """Estimate the directional bias profile on stable pixels.

        Sets ``self._surface`` to the fitted 2-D bias surface and stores the
        chosen angle, before/after residual std, and the 1-D profile arrays in
        ``self.meta``.

        :param raster: Displacement raster from which to estimate the bias.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask (see
            :meth:`~BaseCorrection._resolve_stable_mask`).
        :returns: ``self`` for chaining.
        :rtype: DirectionalBiasCorrection
        :raises ValueError: If fewer than ``max(n_bins, 50)`` stable pixels are
            available.
        """
        fit_mask = self._resolve_stable_mask(
            stable_mask, raster.data, raster.data.shape, raster.transform
        )
        if fit_mask.sum() < max(self.n_bins, 50):
            raise ValueError(
                "Too few stable pixels for DirectionalBiasCorrection "
                f"(need >= {max(self.n_bins, 50)}, got {int(fit_mask.sum())})."
            )

        xx, yy = raster.coords(grid=True, force_offset="ll")
        values = ma.filled(raster.data, np.nan).astype(float)

        if self.angle is not None:
            angle = self.angle
        else:
            angle = estimate_directional_angle(
                xx, yy, values, fit_mask,
                n_bins=self.n_bins, angle_grid=self.angle_grid,
                min_bin_count=self.min_bin_count,
            )

        s = rotate_coords(xx, yy, angle)
        surface, centers, median, fitted = fit_directional_profile(
            s, values, fit_mask,
            n_bins=self.n_bins, profile=self.profile,
            smoothing=self.smoothing, poly_order=self.poly_order,
            min_bin_count=self.min_bin_count,
        )

        self._surface = surface
        std_before = float(np.nanstd(values[fit_mask]))
        std_after = float(np.nanstd((values - surface)[fit_mask]))
        self.meta = {
            "angle_deg": angle,
            "n_bins": self.n_bins,
            "profile": self.profile,
            "n_stable_px": int(fit_mask.sum()),
            "std_before": std_before,
            "std_after": std_after,
            "profile_centers": centers,
            "profile_median": median,
            "profile_fitted": fitted,
        }
        self._fit_called = True
        return self
    # END def
    # apply() / fit_and_apply() inherited — self._surface is a full 2-D array

# --------------------------------------------------------------------------- #
# SlopeRampCorrection
# --------------------------------------------------------------------------- #
class SlopeRampCorrection(BaseCorrection):
    """Remove a ramp correlated with terrain **slope** from the displacement field.

    Unlike :class:`TopoCorrection` (which uses DEM elevation as predictor),
    this class uses terrain slope magnitude in degrees (``|∇DEM|``) computed
    from the DEM gradient.  Slope is the physically correct predictor for
    orthorectification-induced parallax errors: a sensor with a fixed look angle
    displaces steep terrain more than flat terrain regardless of altitude.

    Two polynomial orders:

    - ``'linear'`` — fits ``a·row + b·col + c·slope + d``  (4 parameters, default).
    - ``'quadratic'`` — fits a 7-parameter model with second-order spatial
      terms plus the slope predictor.

    :param order: Polynomial order — ``'linear'`` (default) or ``'quadratic'``.
    :type order: str
    :param lambda_reg: L2 regularisation strength.  Default ``1e-6``.
    :type lambda_reg: float

    Example::

        >>> corr = SlopeRampCorrection(order="linear")
        >>> corr.fit(xDisp, stable_mask=stable, dem=dem)
        >>> print(corr.meta)    # {"row_slope": ..., "slope_coeff": ..., "r2": ...}
        >>> xc = corr.apply(xDisp)
    """

    def __init__(self, order: str = "linear", lambda_reg: float = 1e-6) -> None:
        if order not in ("linear", "quadratic"):
            raise ValueError("order must be 'linear' or 'quadratic'")
        self.order = order
        self.lambda_reg = lambda_reg
        self.meta: dict = {}
    # END def

    def fit(
        self, raster: gu.Raster, stable_mask=None, **kwargs
    ) -> SlopeRampCorrection:
        """Fit the ramp + terrain slope trend on stable pixels.

        The terrain slope is derived internally from the DEM via
        :func:`~geomulticorr.corrections.fit.compute_slope`.

        :param raster: Displacement raster to correct.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask (see
            :meth:`~BaseCorrection._resolve_stable_mask`).
        :param kwargs: Must include ``dem`` — a :class:`geoutils.Raster` DEM
            pre-resampled to the same grid as *raster*.
        :returns: ``self`` for chaining.
        :rtype: SlopeRampCorrection
        :raises ValueError: If ``dem`` is not provided, or if too few valid
            stable pixels are available for the chosen *order*.
        """
        dem: gu.Raster = kwargs.get("dem")
        if dem is None:
            raise ValueError("SlopeRampCorrection.fit() requires dem= kwarg.")
        if dem.data.shape != raster.data.shape:
            dem = dem.reproject(ref=raster)
        dem_arr   = ma.filled(dem.data, np.nan).astype(float)
        slope_arr = compute_slope(dem_arr, raster.transform)
        fit_mask  = self._resolve_stable_mask(
            stable_mask, raster.data, raster.data.shape, raster.transform, dem_arr
        )
        min_px = 5 if self.order == "linear" else 8
        if fit_mask.sum() < min_px:
            raise ValueError(
                f"Too few valid pixels for SlopeRampCorrection(order='{self.order}') "
                f"(need ≥ {min_px})."
            )
        if self.order == "linear":
            a_row, b_col, c_pred, d, self._surface, r2 = fit_linear_with_predictor(
                raster.data, slope_arr, fit_mask, self.lambda_reg
            )
            self.meta = {
                "row_slope":   a_row, "col_slope": b_col,
                "slope_coeff": c_pred, "intercept": d,
                "r2": r2,
            }
        else:
            g, a2, b2, cxy, ax, by, fp, self._surface, r2 = fit_quadratic_with_predictor(
                raster.data, slope_arr, fit_mask, self.lambda_reg
            )
            self.meta = {
                "c_bias":   g,   "c_row2":   a2,
                "c_col2":   b2,  "c_rowcol": cxy,
                "c_row":    ax,  "c_col":    by,
                "c_slope":  fp,  "r2":       r2,
            }
        self._fit_called = True
        return self
    # END def


# --------------------------------------------------------------------------- #
# Fourier destriping (along- / across-track jitter)
# --------------------------------------------------------------------------- #
class _FourierDestriping(BaseCorrection):
    """Shared machinery for FFT-based stripe / jitter removal.

    Collapses the displacement field to a 1-D profile (median across the
    perpendicular axis, **stable pixels only**), models the coherent undulation
    in a wavelength band with :func:`~geomulticorr.corrections.fit.fit_fourier_stripe_profile`,
    broadcasts the 1-D model back to 2-D, and subtracts it.

    Subclasses set :attr:`_profile_axis` — the array axis the profile varies
    *over* (``0`` = rows, ``1`` = columns).  Not instantiated directly; use
    :class:`AlongTrackDestriping` or :class:`AcrossTrackDestriping`.

    :param wavelength_min: Shortest wavelength (m) removed; ``None`` auto-detects
        the dominant spectral peak and removes a one-octave band around it.
    :type wavelength_min: float or None
    :param wavelength_max: Longest wavelength (m) removed; ``None`` = no upper
        bound (remove everything at/above *wavelength_min*).
    :type wavelength_max: float or None
    :param detrend: Remove a linear trend before the FFT (default ``True``).
    :type detrend: bool
    :param taper: Tukey taper fraction in ``[0, 1)`` on the reconstructed model
        to roll off the scene edges (default ``0.0``).
    :type taper: float
    """

    _profile_axis: int = 0

    def __init__(
        self,
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
        detrend: bool = True,
        taper: float = 0.0,
    ) -> None:
        if not 0.0 <= taper < 1.0:
            raise ValueError("taper must be in [0, 1)")
        if wavelength_min is not None and wavelength_min <= 0:
            raise ValueError("wavelength_min must be > 0")
        if (
            wavelength_min is not None
            and wavelength_max is not None
            and wavelength_max <= wavelength_min
        ):
            raise ValueError("wavelength_max must be > wavelength_min")
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.detrend = detrend
        self.taper = taper
        self.meta: dict = {}
    # END def

    def fit(self, raster: gu.Raster, stable_mask=None, **kwargs) -> _FourierDestriping:
        """Estimate the stripe/jitter undulation on stable pixels.

        Sets ``self._surface`` to the fitted 2-D undulation and stores the band,
        before/after residual std, and the 1-D profile/spectrum in ``self.meta``.

        :param raster: Displacement raster from which to estimate the undulation.
        :type raster: geoutils.Raster
        :param stable_mask: Stable-area mask (see
            :meth:`~BaseCorrection._resolve_stable_mask`).
        :returns: ``self`` for chaining.
        :rtype: _FourierDestriping
        """
        fit_mask = self._resolve_stable_mask(
            stable_mask, raster.data, raster.data.shape, raster.transform
        )
        values = ma.filled(raster.data, np.nan).astype(float)
        stable_values = np.where(fit_mask, values, np.nan)

        collapse_axis = 1 - self._profile_axis
        profile = np.nanmedian(stable_values, axis=collapse_axis)
        spacing = raster.res[1] if self._profile_axis == 0 else raster.res[0]

        model_1d, fmeta = fit_fourier_stripe_profile(
            profile, spacing,
            wavelength_min=self.wavelength_min,
            wavelength_max=self.wavelength_max,
            detrend=self.detrend,
            taper=self.taper,
        )

        # Broadcast the 1-D model back to 2-D along the collapsed axis.
        if self._profile_axis == 0:
            surface = np.broadcast_to(model_1d[:, None], values.shape)
        else:
            surface = np.broadcast_to(model_1d[None, :], values.shape)
        self._surface = np.array(surface, dtype=float)

        std_before = float(np.nanstd(values[fit_mask]))
        std_after = float(np.nanstd((values - self._surface)[fit_mask]))
        self.meta = {
            "profile_axis": self._profile_axis,
            "spacing": spacing,
            "band": fmeta["band"],
            "wavelength_peak": fmeta["wavelength_peak"],
            "detrend": self.detrend,
            "taper": self.taper,
            "n_stable_px": int(fit_mask.sum()),
            "std_before": std_before,
            "std_after": std_after,
            "profile_raw": profile,
            "profile_model": model_1d,
            "freqs": fmeta["freqs"],
            "amp_raw": fmeta["amp_raw"],
            "amp_kept": fmeta["amp_kept"],
        }
        self._fit_called = True
        return self
    # END def
    # apply() / fit_and_apply() inherited — self._surface is a full 2-D array


class AlongTrackDestriping(_FourierDestriping):
    """Remove along-track jitter — a quasi-periodic undulation that varies down
    the flight direction and is coherent across the swath.

    The 1-D profile runs over **rows** (``f(row)``); it is built by taking the
    median across columns over stable pixels, band-pass modelled, and subtracted.
    Ideal for smooth, periodic platform-vibration undulations (see
    :class:`_FourierDestriping` for parameters).

    Example::

        >>> corr = AlongTrackDestriping(wavelength_min=None)   # auto-detect the jitter band
        >>> xc = corr.fit_and_apply(xDisp, stable_mask=stable)
        >>> print(corr.meta["band"], corr.meta["std_before"], corr.meta["std_after"])
    """

    _profile_axis = 0


class AcrossTrackDestriping(_FourierDestriping):
    """Remove across-track striping — a banding that varies across the swath.

    The 1-D profile runs over **columns** (``f(col)``); it is built by taking the
    median down rows over stable pixels, band-pass modelled, and subtracted.

    .. note:: This is the unified Fourier band-pass approach.  Sharp,
       high-frequency *per-detector* striping (near the pixel scale) overlaps
       real terrain frequencies and may be better served by a future wavelet
       variant; choose a narrow, short-wavelength band accordingly.
    """

    _profile_axis = 1


# --------------------------------------------------------------------------- #
# Top-level convenience
# --------------------------------------------------------------------------- #
def make_corrections(
    xDisp: gu.Raster,
    yDisp: gu.Raster,
    pipeline: CorrectionPipeline,
    x_stable: np.ndarray | None = None,
    y_stable: np.ndarray | None = None,
    **kwargs,
) -> tuple[gu.Raster, gu.Raster]:
    """Fit and apply a pipeline independently to each displacement component.

    The pipeline is deep-copied so the EW and NS fits are fully independent —
    each component gets its own fitted surfaces and can use a different
    stable-area mask.

    :param xDisp: Raw EW (east–west) displacement raster.
    :type xDisp: geoutils.Raster
    :param yDisp: Raw NS (north–south) displacement raster.
    :type yDisp: geoutils.Raster
    :param pipeline: Correction pipeline built with ``+``, e.g.
        ``MedianCentering() + RampCorrection() + TopoCorrection()``.
    :type pipeline: CorrectionPipeline
    :param x_stable: Boolean stable-area mask for the EW component
        (``True`` = inlier pixel).  ``None`` uses all valid pixels.
    :type x_stable: np.ndarray or None
    :param y_stable: Boolean stable-area mask for the NS component.
        ``None`` uses all valid pixels.
    :type y_stable: np.ndarray or None
    :param kwargs: Extra keyword arguments forwarded to each step's
        :meth:`~BaseCorrection.fit` (e.g. ``dem=dem_raster``).
    :returns: Tuple ``(xc, yc)`` — corrected EW and NS displacement rasters
        with the same shape, CRS, and transform as the inputs.
    :rtype: tuple[geoutils.Raster, geoutils.Raster]

    Example::

        >>> xc, yc = make_corrections(
        ...     xDisp, yDisp,
        ...     MedianCentering() + RampCorrection() + TopoCorrection(),
        ...     x_stable=stable_ew, y_stable=stable_ns, dem=dem,
        ... )
    """
    xc = copy.deepcopy(pipeline).fit(xDisp, stable_mask=x_stable, **kwargs).apply(xDisp)
    yc = copy.deepcopy(pipeline).fit(yDisp, stable_mask=y_stable, **kwargs).apply(yDisp)
    return xc, yc
