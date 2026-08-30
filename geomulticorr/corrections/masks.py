#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# masks.py
# creation date: 2026-05-19.
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
"""Mask steps for EW/NS displacement fields.

Each mask computes a boolean valid-pixel array for a **single** raster
(``True`` = keep, ``False`` = mask out).  Use :class:`FilterPipeline` to
combine masks with the ``+`` operator and produce a single stable-area boolean
mask per component — which is then passed to the correction pipeline via
:func:`~geomulticorr.corrections.corrections.make_corrections`.

Masks and corrections are kept in **separate pipelines** on purpose:
the stable-area mask used for regression fitting should incorporate all
filter decisions (outlier rejection, CC threshold, slope limit, moving-area
exclusion), and different thresholds can be used for each displacement
component.

Typical workflow
----------------
>>> # Step 1 — build a common spatial filter (same for EW and NS)
>>> common = CCFilter(0.6) + SlopeMask(max_slope=60) + StableAreaMask(rg_gdf)
>>>
>>> # Step 2 — per-component stable masks (OutlierFilter can differ)
>>> x_stable = common.compute(xDisp, cc=cc, dem=dem) \\
...            & OutlierFilter((-10, 10)).generate_mask(xDisp)
>>> y_stable = common.compute(yDisp, cc=cc, dem=dem) \\
...            & OutlierFilter((-5, 5)).generate_mask(yDisp)
>>>
>>> # Step 3 — apply corrections (stats fitted on stable pixels only)
>>> from geomulticorr.corrections import MedianCentering, RampCorrection, make_corrections
>>> corrections = MedianCentering() + RampCorrection()
>>> xc, yc = make_corrections(xDisp, yDisp, corrections, x_stable, y_stable)
"""
from __future__ import annotations

from abc import abstractmethod

import geoutils as gu
import numpy as np
import numpy.ma as ma

from geomulticorr.corrections.corrections import BaseCorrection


# --------------------------------------------------------------------------- #
# Private array helpers (no I/O, no gu.Raster)
# --------------------------------------------------------------------------- #

def _compute_slope_degrees(
    dem_ma: np.ma.MaskedArray,
    res_x: float,
    res_y: float,
) -> np.ma.MaskedArray:
    """Slope in degrees derived from a DEM masked array.

    Uses ``numpy.gradient`` with the raster pixel spacing.  Pixels where the
    DEM is masked propagate as masked in the output.
    """
    filled = ma.filled(dem_ma, np.nan)
    dz_dy, dz_dx = np.gradient(filled, res_y, res_x)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    return ma.array(slope, mask=np.ma.getmaskarray(dem_ma) | ~np.isfinite(slope))


def _compute_hillshade(
    dem_ma: np.ma.MaskedArray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> np.ma.MaskedArray:
    """Hillshade (0 = fully shadowed, 1 = fully lit) using ``matplotlib.colors.LightSource``.

    Nodata holes in the DEM are filled with the DEM mean before shading to
    avoid edge artefacts, but those pixels are masked in the output.
    """
    from matplotlib.colors import LightSource

    fill_val = float(ma.mean(dem_ma)) if dem_ma.count() > 0 else 0.0
    filled = ma.filled(dem_ma, fill_val)
    ls = LightSource(azdeg=sun_azimuth_deg, altdeg=sun_elevation_deg)
    hs = ls.hillshade(filled)
    return ma.array(hs, mask=np.ma.getmaskarray(dem_ma))


# --------------------------------------------------------------------------- #
# FilterPipeline
# --------------------------------------------------------------------------- #

class FilterPipeline:
    """Ordered combination of :class:`BaseMask` steps into a single boolean mask.

    Build with the ``+`` operator between any two :class:`BaseMask` objects:

    >>> filters = CCFilter(0.6) + SlopeMask(max_slope=60) + StableAreaMask(rg_gdf)

    Then compute per-component stable-area masks and apply corrections:

    >>> x_stable = filters.compute(xDisp, cc=cc, dem=dem) \\
    ...            & OutlierFilter((-10, 10)).generate_mask(xDisp)
    >>> y_stable = filters.compute(yDisp, cc=cc, dem=dem) \\
    ...            & OutlierFilter((-5, 5)).generate_mask(yDisp)

    Parameters
    ----------
    masks:
        Ordered list of :class:`BaseMask` instances.
    """

    def __init__(self, masks: list[BaseMask]) -> None:
        self.masks = list(masks)

    def __add__(self, other: BaseMask | FilterPipeline) -> FilterPipeline:
        if isinstance(other, FilterPipeline):
            return FilterPipeline(self.masks + other.masks)
        if isinstance(other, BaseMask):
            return FilterPipeline(self.masks + [other])
        raise TypeError(
            f"Cannot add {type(other).__name__} to a FilterPipeline. "
            "Only BaseMask instances are accepted."
        )

    def compute(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        """Return the combined boolean mask for *raster*.

        Each constituent mask's :meth:`~BaseMask.generate_mask` is
        AND-combined in order.  ``True`` = pixel is valid and should be kept.
        """
        result = np.ones(raster.data.shape, dtype=bool)
        for m in self.masks:
            result = result & m.generate_mask(raster, **kwargs)
        return result

    def apply(
        self,
        xDisp: gu.Raster,
        yDisp: gu.Raster,
        **kwargs,
    ) -> tuple[gu.Raster, gu.Raster]:
        """Apply the combined mask independently to both displacement rasters."""
        x_keep = self.compute(xDisp, **kwargs)
        y_keep = self.compute(yDisp, **kwargs)

        x_ma = xDisp.data
        y_ma = yDisp.data
        new_x = ma.array(x_ma.data, mask=np.ma.getmaskarray(x_ma) | ~x_keep)
        new_y = ma.array(y_ma.data, mask=np.ma.getmaskarray(y_ma) | ~y_keep)

        return BaseCorrection._copy_raster(xDisp, new_x), BaseCorrection._copy_raster(yDisp, new_y)

    def __repr__(self) -> str:
        names = " + ".join(type(m).__name__ for m in self.masks)
        return f"FilterPipeline([{names}])"


# --------------------------------------------------------------------------- #
# BaseMask
# --------------------------------------------------------------------------- #

class BaseMask(BaseCorrection):
    """Abstract base for all masking steps.

    Subclasses implement :meth:`generate_mask` which accepts a **single**
    raster and returns a boolean array (``True`` = keep the pixel).
    :meth:`apply` calls it independently for *xDisp* and *yDisp* so that
    per-component behaviour (e.g. different :class:`OutlierFilter` thresholds)
    is naturally supported.

    Use ``+`` to combine masks into a :class:`FilterPipeline`:

    >>> filters = SlopeMask(60) + CCFilter(0.6)   # → FilterPipeline
    """

    _is_mask: bool = True

    @abstractmethod
    def generate_mask(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        """Return a boolean array with the same spatial shape as *raster*.

        ``True`` means the pixel is **valid and should be kept**.
        ``False`` means the pixel should be masked out.
        """

    def __add__(self, other: BaseMask | FilterPipeline) -> FilterPipeline:
        if isinstance(other, FilterPipeline):
            return FilterPipeline([self] + other.masks)
        if isinstance(other, BaseMask):
            return FilterPipeline([self, other])
        raise TypeError(
            f"Cannot chain {type(self).__name__} with {type(other).__name__} using +. "
            "Masks form a FilterPipeline; corrections form a CorrectionPipeline — "
            "they cannot be mixed in the same pipeline."
        )

    def apply(
        self,
        xDisp: gu.Raster,
        yDisp: gu.Raster,
        **kwargs,
    ) -> tuple[gu.Raster, gu.Raster]:
        x_keep = self.generate_mask(xDisp, **kwargs)
        y_keep = self.generate_mask(yDisp, **kwargs)

        x_ma = xDisp.data
        y_ma = yDisp.data

        new_x = ma.array(x_ma.data, mask=np.ma.getmaskarray(x_ma) | ~x_keep)
        new_y = ma.array(y_ma.data, mask=np.ma.getmaskarray(y_ma) | ~y_keep)

        n_new_x = int((~x_keep & ~np.ma.getmaskarray(x_ma)).sum())
        n_new_y = int((~y_keep & ~np.ma.getmaskarray(y_ma)).sum())
        self.meta = {
            "newly_masked_x": n_new_x,
            "newly_masked_y": n_new_y,
            "masked_fraction_x": float(n_new_x / x_keep.size),
            "masked_fraction_y": float(n_new_y / y_keep.size),
        }

        return BaseCorrection._copy_raster(xDisp, new_x), BaseCorrection._copy_raster(yDisp, new_y)

    def apply_single(self, raster: gu.Raster, **kwargs) -> gu.Raster:
        """Apply this mask to a single displacement raster."""
        keep = self.generate_mask(raster, **kwargs)
        old_ma = raster.data
        new_ma = ma.array(old_ma.data, mask=np.ma.getmaskarray(old_ma) | ~keep)
        self.meta = {
            "newly_masked": int((~keep & ~np.ma.getmaskarray(old_ma)).sum()),
            "masked_fraction": float((~keep & ~np.ma.getmaskarray(old_ma)).sum() / keep.size),
        }
        return BaseCorrection._copy_raster(raster, new_ma)


# --------------------------------------------------------------------------- #
# Implemented masks
# --------------------------------------------------------------------------- #

class SlopeMask(BaseMask):
    """Mask pixels located on terrain slopes outside *[min_slope, max_slope]*.

    Slope is computed from the DEM using ``numpy.gradient``.

    Parameters
    ----------
    min_slope:
        Lower slope angle bound in degrees (default 0°). Pixels with slope
        **below** this value are masked.
    max_slope:
        Upper slope angle bound in degrees (default 80°). Pixels with slope
        **above** this value are masked.

    Required kwargs in ``generate_mask()`` / ``apply()``
    -----------------------------------------------------
    dem : gu.Raster
        DEM resampled to the same grid as the displacement field.
    """

    _REQUIRED_KWARGS = ("dem",)

    def __init__(self, min_slope: float = 0.0, max_slope: float = 80.0) -> None:
        self.min_slope = min_slope
        self.max_slope = max_slope
        self.meta: dict = {}

    def generate_mask(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        dem: gu.Raster = kwargs["dem"]
        assert dem.data.shape == raster.data.shape, (
            f"DEM shape {dem.data.shape} must match displacement field {raster.data.shape}. "
            "Pre-resample the DEM to the same grid."
        )
        slope = _compute_slope_degrees(dem.data, *dem.res)
        valid = ~np.ma.getmaskarray(slope)
        return valid & (slope.data >= self.min_slope) & (slope.data <= self.max_slope)


class ShadowMask(BaseMask):
    """Mask pixels in topographic shadow using a DEM-based hillshade simulation.

    Sun position is specified via azimuth and elevation angles passed as
    keyword arguments at ``generate_mask()`` / ``apply()`` time (they are
    acquisition-specific).

    Parameters
    ----------
    shadow_threshold:
        Hillshade values **below** this threshold are considered shadowed and
        masked out (default 0.1; range 0–1).

    Required kwargs in ``generate_mask()`` / ``apply()``
    -----------------------------------------------------
    dem : gu.Raster
        DEM resampled to the same grid as the displacement field.
    sun_azimuth_deg : float
        Sun azimuth angle in degrees (0° = North, clockwise).
    sun_elevation_deg : float
        Sun elevation angle above the horizon in degrees.
    """

    _REQUIRED_KWARGS = ("dem", "sun_azimuth_deg", "sun_elevation_deg")

    def __init__(self, shadow_threshold: float = 0.1) -> None:
        self.shadow_threshold = shadow_threshold
        self.meta: dict = {}

    def generate_mask(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        dem: gu.Raster = kwargs["dem"]
        assert dem.data.shape == raster.data.shape, (
            f"DEM shape {dem.data.shape} must match displacement field {raster.data.shape}. "
            "Pre-resample the DEM to the same grid."
        )
        hs = _compute_hillshade(
            dem.data,
            float(kwargs["sun_azimuth_deg"]),
            float(kwargs["sun_elevation_deg"]),
        )
        valid = ~np.ma.getmaskarray(hs)
        return valid & (hs.data > self.shadow_threshold)


class OutlierFilter(BaseMask):
    """Mask pixels whose displacement magnitude exceeds *threshold*.

    A pixel is masked when the displacement value for **this component** falls
    outside ``(threshold[0], threshold[1])``.  Because ``generate_mask``
    operates on a single raster, EW and NS components can be filtered with
    different thresholds by using two separate instances:

    >>> x_of = OutlierFilter((-10, 10)).generate_mask(xDisp)
    >>> y_of = OutlierFilter((-5, 5)).generate_mask(yDisp)

    Parameters
    ----------
    threshold:
        ``(min, max)`` exclusive bounds; pixels at or beyond these values
        are masked (default ``(-10.0, 10.0)``).
    """

    _REQUIRED_KWARGS: tuple[str, ...] = ()

    def __init__(self, threshold: tuple[float, float] = (-10.0, 10.0)) -> None:
        self.threshold = threshold
        self.meta: dict = {}

    def generate_mask(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        lo, hi = self.threshold
        arr = ma.filled(raster.data, np.nan)
        return np.isfinite(arr) & (arr > lo) & (arr < hi)


class CCFilter(BaseMask):
    """Mask pixels where the cross-correlation score is below *cc_threshold*.

    Parameters
    ----------
    cc_threshold:
        Minimum acceptable CC value (default 0.5).

    Required kwargs in ``generate_mask()`` / ``apply()``
    -----------------------------------------------------
    cc : gu.Raster
        Cross-correlation raster resampled to the same grid as the
        displacement field.
    """

    _REQUIRED_KWARGS = ("cc",)

    def __init__(self, cc_threshold: float = 0.5) -> None:
        self.cc_threshold = cc_threshold
        self.meta: dict = {}

    def generate_mask(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        cc: gu.Raster = kwargs["cc"]
        cc_arr = ma.filled(cc.data, 0.0)
        return cc_arr >= self.cc_threshold


class StableAreaMask(BaseMask):
    """Mask pixels inside known unstable areas (rock glaciers, landslides, …).

    Pixels **inside** the provided geometries are masked out; pixels outside
    are kept.

    Parameters
    ----------
    stable_mask:
        One of:

        - ``str`` or ``pathlib.Path`` — path to a vector file (GeoJSON,
          Shapefile, GeoPackage …) whose polygons outline **unstable** areas.
        - ``geopandas.GeoDataFrame`` — same, already loaded.
        - ``numpy.ndarray`` (boolean, same shape as the displacement field) —
          pre-computed pixel mask where ``True`` means the pixel is **stable**
          (kept) and ``False`` means unstable (masked out).
    """

    _REQUIRED_KWARGS: tuple[str, ...] = ()

    def __init__(self, stable_mask) -> None:
        self.stable_mask = stable_mask
        self.meta: dict = {}

    def generate_mask(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        if isinstance(self.stable_mask, np.ndarray):
            return self.stable_mask.astype(bool)

        if isinstance(self.stable_mask, gu.Vector):
            # create_mask returns True inside polygon (moving area) — invert for stable
            moving_mask = self.stable_mask.create_mask(raster)
            return ~np.asarray(moving_mask.data, dtype=bool)

        # Fallback for str, Path, or gpd.GeoDataFrame inputs.
        # The grid CRS is passed so polygons in a different CRS are reprojected
        # instead of landing outside the grid and producing an empty mask.
        gdf = BaseCorrection._load_gdf(self.stable_mask)
        moving = BaseCorrection._rasterize_moving_areas(
            gdf, raster.data.shape, raster.transform, getattr(raster, "crs", None)
        )
        return ~moving  # True = stable = keep


class SnowMask(BaseMask):
    """Mask pixels covered by snow.

    Two usage modes:

    1. **Pre-computed mask** (ready to use): pass ``snow_mask=`` as a
       ``gu.Raster`` or boolean ``numpy.ndarray`` where ``True`` = snow.
    2. **Automatic detection** (not yet implemented): will use NDSI
       (Normalised Difference Snow Index) computed from optical image bands.

    Parameters
    ----------
    enabled:
        If ``False``, this step is a no-op (default ``True``).
    ndsi_threshold:
        Future NDSI threshold for automatic detection (default 0.4).
    """

    _REQUIRED_KWARGS: tuple[str, ...] = ()

    def __init__(self, enabled: bool = True, ndsi_threshold: float = 0.4) -> None:
        self.enabled = enabled
        self.ndsi_threshold = ndsi_threshold
        self.meta: dict = {}

    def generate_mask(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        if not self.enabled:
            return np.ones(raster.data.shape, dtype=bool)

        if "snow_mask" in kwargs:
            m = kwargs["snow_mask"]
            snow = np.asarray(
                m.data if isinstance(m, gu.Raster) else m,
                dtype=bool,
            )
            return ~snow  # True = no snow = keep

        raise NotImplementedError(
            "Automatic snow detection is not yet implemented (NDSI-based). "
            "Pass a pre-computed boolean mask as snow_mask= "
            "(True = snow-covered pixel)."
        )


class CloudMask(BaseMask):
    """Mask pixels covered by clouds.

    Two usage modes:

    1. **Pre-computed mask** (ready to use): pass ``cloud_mask=`` as a
       ``gu.Raster`` or boolean ``numpy.ndarray`` where ``True`` = cloud.
    2. **Automatic detection** (not yet implemented): will use sensor-specific
       QA bands or a reflectance threshold.

    Parameters
    ----------
    enabled:
        If ``False``, this step is a no-op (default ``True``).
    """

    _REQUIRED_KWARGS: tuple[str, ...] = ()

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.meta: dict = {}

    def generate_mask(self, raster: gu.Raster, **kwargs) -> np.ndarray:
        if not self.enabled:
            return np.ones(raster.data.shape, dtype=bool)

        if "cloud_mask" in kwargs:
            m = kwargs["cloud_mask"]
            cloud = np.asarray(
                m.data if isinstance(m, gu.Raster) else m,
                dtype=bool,
            )
            return ~cloud  # True = no cloud = keep

        raise NotImplementedError(
            "Automatic cloud detection is not yet implemented. "
            "Pass a pre-computed boolean mask as cloud_mask= "
            "(True = cloud-covered pixel)."
        )
