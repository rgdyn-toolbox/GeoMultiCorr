#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared fixtures for the inversion tests: synthetic ``TOT_*.tif`` series.

``write_tot_stack`` lays down an inversion folder in the exact layout a solver
leaves (``inverse_<comp>/TOT_<date>_<comp>.tif``, float32, NaN nodata) from a
function of ``(t_years, yy, xx)``, so the loader, the fusion and the extractor
can be tested against a known truth without a project.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

DEFAULT_CRS = "EPSG:32632"
DEFAULT_ORIGIN = (500000.0, 4649000.0)


def _profile(shape, origin=DEFAULT_ORIGIN, res=3.0, crs=DEFAULT_CRS) -> dict:
    return {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "width": shape[1], "height": shape[0], "crs": crs,
        "transform": from_origin(origin[0], origin[1], res, res),
        "nodata": np.nan, "compress": "lzw",
    }


def years_since(dates, t0=None) -> np.ndarray:
    """Decimal years elapsed since ``t0`` (default: the first date), from ``YYYYMMDD``."""
    import pandas as pd
    t = pd.to_datetime(list(dates), format="%Y%m%d")
    t0 = t[0] if t0 is None else pd.Timestamp(t0)
    return ((t - t0) / pd.Timedelta(days=365.25)).to_numpy()


def write_tot_stack(root, dates, comp, fn, *, shape=(6, 5), origin=DEFAULT_ORIGIN,
                    res=3.0, crs=DEFAULT_CRS, t0=None) -> dict[str, Path]:
    """Write ``root/inverse_<comp>/TOT_<date>_<comp>.tif`` for every date.

    *fn(t_years, yy, xx)* returns the value at each pixel; it may return a
    scalar or an array broadcastable to *shape*. ``t_years`` is measured from
    *t0* (default: the first date). Returns ``{date: path}``.
    """
    root = Path(root)
    folder = root / f"inverse_{comp}"
    folder.mkdir(parents=True, exist_ok=True)
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    out = {}
    for date, t in zip(dates, years_since(dates, t0)):
        data = np.broadcast_to(np.asarray(fn(t, yy, xx), dtype=np.float32), shape)
        path = folder / f"TOT_{date}_{comp}.tif"
        with rasterio.open(str(path), "w", **_profile(shape, origin, res, crs)) as dst:
            dst.write(np.ascontiguousarray(data), 1)
        out[date] = path
    return out


@pytest.fixture
def tot_writer():
    return write_tot_stack
