#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _stack.py
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
"""Cumulative-displacement stacks as ``xarray`` — the loader and the writer.

An inversion leaves one GeoTIFF per date and component
(``inverse_EW/TOT_<YYYYMMDD>_EW.tif`` …). Everything downstream of that —
comparing two sensors' series, calibrating one to the other, sampling a point
through time — wants the whole series as one ``(time, y, x)`` array, which is
what :func:`load_cumulative_stack` returns and :func:`write_cumulative_stack`
takes back to the same on-disk layout, so
:class:`~geomulticorr.stats.inversion_extractor.InversionExtractor` and
``rasterstats`` read a fused product exactly as they read a solver's.

Public API:

- :data:`COMPONENTS` — the two components a stack carries; magnitude is
  always **derived** from them by :func:`magnitude`, never stored as data.
- :func:`load_cumulative_stack` — ``TIOInversion`` or directory → ``xr.Dataset``.
- :func:`write_cumulative_stack` — ``xr.Dataset`` → ``TOT_*.tif`` layout
  (+ magnitude, + calibration maps, + a JSON trace).
- :func:`tot_tif_name` — the one place the file name is spelled.

Two conventions inherited from the solvers, worth knowing before "fixing":

- A Fortran-run series has ``height − 1`` rows (``lect_depl_cumule_lin`` drops
  the last row); a Python-run series has the full height. The two do not
  align exactly — :func:`xr.align(..., join="exact")` refuses them — and
  fusion requires the same backend for both runs.
- ``_tot_to_geotiff`` treats an exact ``0.0`` as no-data (the Fortran writes
  zeros where it has nothing), so a Fortran-run pixel that genuinely did not
  move between two dates reads back as NaN there. Nothing here changes that.

No ``geomulticorr.core`` imports; the TOT discovery helpers come from
:mod:`geomulticorr.stats.inversion_extractor` (``stats`` initialises before
``inversion``, the reverse import would be a cycle).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from geomulticorr._logging import logger
from geomulticorr.stats.inversion_extractor import (
    COMP_DIR,
    discover_tot_rasters,
    resolve_inversion_dir,
)

#: The components a cumulative stack carries. ``"magn"`` is derived.
COMPONENTS: tuple[str, ...] = ("EW", "NS")


def tot_tif_name(directory: str | Path, date: str, component: str) -> Path:
    """``<directory>/TOT_<date>_<component>.tif`` — the one spelling of the name.

    ``TIOInversion._tot_tif_name`` delegates here, so the solver backends, the
    fusion writer and the extractor cannot drift apart on a file name.
    """
    return Path(directory) / f"TOT_{date}_{component}.tif"


def _dask_available() -> bool:
    try:
        import dask  # noqa: F401
    except ImportError:
        return False
    return True


def load_cumulative_stack(
    source,
    components: Sequence[str] = COMPONENTS,
    *,
    chunks="auto",
) -> xr.Dataset:
    """Read an inversion's ``TOT_*.tif`` series into one ``(time, y, x)`` dataset.

    :param source: A ``TIOInversion`` (anything with ``.inversion_dir``) or the
        inversion directory itself.
    :param components: Which components to load; each becomes a data variable.
    :param chunks: ``"auto"`` chunks lazily through dask when it is installed
        (nothing is read until used) and loads eagerly otherwise; ``None``
        forces eager loading; a dict is passed to
        :func:`rioxarray.open_rasterio` as is.
    :returns: ``xr.Dataset`` with data variables per component on
        ``(time, y, x)``, a ``time`` coordinate (``datetime64[ns]``), a
        ``date`` coordinate (the ``YYYYMMDD`` strings the files carry), the
        ``spatial_ref`` coordinate rioxarray uses for the CRS, and attrs
        ``inversion_dir`` / ``components``.
    :raises FileNotFoundError: When a requested component has no rasters.
    :raises ValueError: When the rasters do not share one grid.
    """
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    inv_dir = resolve_inversion_dir(source)
    found = discover_tot_rasters(inv_dir, components)
    missing = [c for c in components if c not in found]
    if missing:
        raise FileNotFoundError(
            f"No TOT_*_{'/'.join(missing)}.tif under {inv_dir} — run "
            "TIOInversion.post_process() or launch(mode='python') first."
        )

    date_sets = [set(found[c]) for c in components]
    dates = sorted(set.intersection(*date_sets))
    dropped = sorted(set.union(*date_sets) - set(dates))
    if dropped:
        logger.warning(
            f"load_cumulative_stack: {len(dropped)} date(s) present for only some "
            f"components are dropped: {', '.join(dropped)}"
        )
    if not dates:
        raise FileNotFoundError(f"No date has every requested component under {inv_dir}.")

    if chunks == "auto":
        chunks_arg = {} if _dask_available() else None
        if chunks_arg is None:
            logger.info("dask not available — loading the cumulative stack eagerly.")
    else:
        chunks_arg = chunks

    def _open(path: Path) -> xr.DataArray:
        da = rioxarray.open_rasterio(path, masked=True, chunks=chunks_arg)
        return da.squeeze("band", drop=True)

    data_vars = {}
    template: xr.DataArray | None = None
    for comp in components:
        arrays = []
        for date in dates:
            da = _open(found[comp][date])
            if template is None:
                template = da
            else:
                try:
                    xr.align(template, da, join="exact")
                except ValueError as exc:
                    raise ValueError(
                        f"{found[comp][date].name} is not on the grid of "
                        f"{found[components[0]][dates[0]].name}: {exc}"
                    ) from None
            arrays.append(da)
        stacked = xr.concat(arrays, dim="time", coords="minimal", compat="override")
        data_vars[comp] = stacked.astype("float32")

    time = pd.to_datetime(dates, format="%Y%m%d")
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(time=("time", time.values), date=("time", list(dates)))
    ds.attrs.update({"inversion_dir": str(inv_dir), "components": list(components)})
    return ds


def magnitude(ds: xr.Dataset) -> xr.DataArray:
    """``hypot(EW, NS)`` — NaN wherever either component is NaN."""
    if "EW" not in ds or "NS" not in ds:
        raise KeyError("magnitude needs both 'EW' and 'NS' data variables")
    out = np.hypot(ds["EW"], ds["NS"])
    out.name = "magn"
    return out


def write_cumulative_stack(
    ds: xr.Dataset,
    out_dir: str | Path,
    *,
    components: Sequence[str] | None = None,
    magnitude: bool = True,
    overwrite: bool = True,
    fit: xr.Dataset | None = None,
    parameters: dict | None = None,
) -> dict[str, dict[str, Path]]:
    """Write a cumulative stack back into the ``inverse_<comp>/TOT_<date>_<comp>.tif`` layout.

    The files are indistinguishable from a solver's output to
    :class:`~geomulticorr.stats.inversion_extractor.InversionExtractor`:
    float32, NaN nodata, LZW, the dataset's CRS and transform.

    :param ds: A dataset from :func:`load_cumulative_stack` or
        :mod:`geomulticorr.inversion.fusion`. A ``magn`` variable, if present,
        is **ignored with a warning** — magnitude is always re-derived from
        the written EW/NS so the three can never disagree.
    :param out_dir: The product's folder (``<pzone>/inversion/<name>/``).
    :param components: Which variables to write; default every one of
        :data:`COMPONENTS` present in *ds*.
    :param magnitude: Also write ``inverse_magn/TOT_<date>_magn.tif`` from the
        written EW/NS (only when both are written).
    :param overwrite: ``False`` keeps existing files.
    :param fit: A calibration dataset (``offset``/``rate``/``rms``/``n_valid`` on
        ``(component, y, x)``) written under ``calibration/``.
    :param parameters: A JSON-able trace written as ``fusion_parameters.json``;
        a failure to write it is logged, never raised.
    :returns: ``{component: {date: path}}`` for the rasters written, including
        ``"magn"`` when derived.
    """
    import rioxarray  # noqa: F401

    out_dir = Path(out_dir)
    comps = [c for c in (components or COMPONENTS) if c in ds.data_vars]
    if "magn" in ds.data_vars and "magn" in (components or ()):
        logger.warning("write_cumulative_stack: 'magn' is derived from EW/NS, never written as given.")
    comps = [c for c in comps if c != "magn"]
    if not comps:
        raise ValueError("nothing to write: the dataset carries none of EW/NS")

    dates = [str(d) for d in ds["date"].values] if "date" in ds.coords else \
            list(pd.to_datetime(ds["time"].values).strftime("%Y%m%d"))

    written: dict[str, dict[str, Path]] = {}
    for comp in comps:
        da = ds[comp]
        folder = out_dir / COMP_DIR[comp]
        folder.mkdir(parents=True, exist_ok=True)
        written[comp] = {}
        for k, date in enumerate(dates):
            path = tot_tif_name(folder, date, comp)
            if path.exists() and not overwrite:
                written[comp][date] = path
                continue
            _to_geotiff(da.isel(time=k), path)
            written[comp][date] = path
        logger.file(f"  {COMP_DIR[comp]}: {len(dates)} TOT_*_{comp}.tif")

    if magnitude and {"EW", "NS"} <= set(comps):
        folder = out_dir / COMP_DIR["magn"]
        folder.mkdir(parents=True, exist_ok=True)
        magn = np.hypot(ds["EW"], ds["NS"])
        written["magn"] = {}
        for k, date in enumerate(dates):
            path = tot_tif_name(folder, date, "magn")
            if not (path.exists() and not overwrite):
                _to_geotiff(magn.isel(time=k), path)
            written["magn"][date] = path
        logger.file(f"  {COMP_DIR['magn']}: {len(dates)} TOT_*_magn.tif")

    if fit is not None:
        folder = out_dir / "calibration"
        folder.mkdir(parents=True, exist_ok=True)
        for var in ("offset", "rate", "rms", "n_valid"):
            if var not in fit:
                continue
            for comp in [str(c) for c in fit["component"].values]:
                _to_geotiff(fit[var].sel(component=comp), folder / f"{comp}_{var}.tif")
        logger.file("  calibration/: offset, rate, rms, n_valid per component")

    if parameters is not None:
        from geomulticorr.inversion._run_parameters import write_run_parameters

        try:
            write_run_parameters(out_dir / "fusion_parameters.json", parameters)
        except Exception as exc:  # a trace must never cost the product
            logger.warning(f"[fusion parameters] could not write the trace: {exc}")

    return written


def _to_geotiff(da: xr.DataArray, path: Path) -> Path:
    """One 2-D ``DataArray`` → float32 / NaN-nodata / LZW GeoTIFF via rioxarray."""
    out = da.astype("float32")
    if out.rio.crs is None and "spatial_ref" in da.coords:
        out = out.rio.write_crs(da.rio.crs)
    out = out.rio.write_nodata(np.nan, encoded=False)
    # drop coords that are not the two spatial dims (time/date/component…),
    # which rioxarray would otherwise try to encode as extra dimensions
    out = out.drop_vars([c for c in out.coords if c not in ("x", "y", "spatial_ref")])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.rio.to_raster(path, dtype="float32", compress="LZW", driver="GTiff")
    return path
