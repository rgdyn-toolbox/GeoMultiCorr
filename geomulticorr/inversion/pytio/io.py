#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# io.py
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
"""Readers/writers for the TIO flat-binary formats.

The `depl_cumule` contract (consumed by `lect_depl_cumule_lin`):
float32 little-endian, no headers/record markers, pixel-interleaved (BIP) —
for each pixel in row-major (y, x) order, the S dates follow contiguously.
No-data = 9999.
"""

import os
import re

import numpy as np
import xarray as xr

from .config import NODATA_OUT


# ---------------------------------------------------------------- list files

def read_image_list(path):
    """`liste_image_inv`: date YYYYMMDD, decimal year, elapsed, bperp [, qual].

    Returns dict of arrays: dates (str), t, elapsed, bperp, quality.
    """
    dates, t, elapsed, bperp, qual = [], [], [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            dates.append(parts[0])
            t.append(float(parts[1]))
            elapsed.append(float(parts[2]) if len(parts) > 2 else np.nan)
            bperp.append(float(parts[3]) if len(parts) > 3 else 0.0)
            qual.append(float(parts[4]) if len(parts) > 4 else 1.0)
    return dict(dates=np.array(dates), t=np.array(t),
                elapsed=np.array(elapsed), bperp=np.array(bperp),
                quality=np.array(qual))


def read_pair_list(path):
    """`liste_couple`: date1, date2 [, weight]. Returns (d1, d2, weight)."""
    d1, d2, w = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            d1.append(parts[0])
            d2.append(parts[1])
            w.append(float(parts[2]) if len(parts) > 2 else 1.0)
    return np.array(d1), np.array(d2), np.array(w)


# ------------------------------------------------------------- binary rasters

def read_rsc(path):
    """Parse a ROI_PAC-style .rsc file into a dict of strings."""
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def guess_shape(binary_path):
    """(length, width) from a sibling .rsc or ENVI .hdr file."""
    rsc = binary_path + ".rsc"
    if os.path.exists(rsc):
        r = read_rsc(rsc)
        return int(r["FILE_LENGTH"]), int(r["WIDTH"])
    hdr = binary_path + ".hdr"
    if os.path.exists(hdr):
        txt = open(hdr).read()
        lines = int(re.search(r"lines\s*=\s*(\d+)", txt).group(1))
        samples = int(re.search(r"samples\s*=\s*(\d+)", txt).group(1))
        return lines, samples
    raise FileNotFoundError(f"no .rsc or .hdr next to {binary_path}")


def load_r4(path, shape=None):
    """Memory-map one float32 flat binary as (y, x)."""
    if shape is None:
        shape = guess_shape(path)
    return np.memmap(path, dtype="<f4", mode="r", shape=tuple(shape))


def load_pair_stack(directory, date1, date2, template="{d1}-{d2}.r4",
                    shape=None, chunks={"y": 128, "x": -1}):
    """Lazy (pair, y, x) DataArray from flat float32 binaries.

    directory/template.format(d1=..., d2=...) must exist for every pair
    (e.g. the LN_DATA folder, or the GMC `binary` folder with
    template='{d1}_{d2}_EW').

    With dask installed the stack is chunked in space and nothing is read
    until the inversion runs; without it the memmaps are stacked eagerly,
    which is fine for small scenes but loads every pair map into memory.
    Chunk only in ``y``/``x`` — the kernel needs the whole pair axis per pixel.
    """
    paths = [os.path.join(directory, template.format(d1=a, d2=b))
             for a, b in zip(date1, date2)]
    if shape is None:
        shape = guess_shape(paths[0])
    try:
        import dask.array as da
    except ImportError:  # pragma: no cover - exercised only without dask
        from geomulticorr._logging import logger
        logger.warning(
            "dask is not installed — loading the pair stack eagerly "
            f"({len(paths)} maps of {shape[0]}x{shape[1]} px)."
        )
        stack = np.stack([np.asarray(load_r4(p, shape)) for p in paths], axis=0)
    else:
        arrs = [da.from_array(load_r4(p, shape), chunks=(chunks.get("y", 128),
                                                         chunks.get("x", -1)))
                for p in paths]
        stack = da.stack(arrs, axis=0)
    return xr.DataArray(
        stack, dims=("pair", "y", "x"),
        coords={"date1": ("pair", list(date1)), "date2": ("pair", list(date2))},
        name="displacement")


# ------------------------------------------------------------------- writers

def write_envi_hdr(path, samples, lines, bands, band_names=None,
                   nodata=NODATA_OUT, interleave="bip"):
    with open(path + ".hdr", "w") as f:
        f.write("ENVI\n")
        f.write(f"samples = {samples}\nlines = {lines}\nbands = {bands}\n")
        f.write("header offset = 0\ndata type = 4\n")
        f.write(f"interleave = {interleave}\nbyte order = 0\n")
        if bands > 1:
            f.write(f"data ignore value = {nodata:9.2f}\n")
        if band_names is not None:
            f.write("band names = {" + ",".join(band_names) + "}")


def write_depl_cumule(cum, path, nodata_in=None):
    """Write a (date, y, x) DataArray/array in the exact Fortran format."""
    arr = np.asarray(cum.transpose("y", "x", "date").data
                     if isinstance(cum, xr.DataArray) else np.moveaxis(cum, 0, -1))
    arr = np.ascontiguousarray(arr, dtype="<f4")
    arr[~np.isfinite(arr)] = NODATA_OUT
    if nodata_in is not None:
        arr[arr == nodata_in] = NODATA_OUT
    arr.tofile(path)
    names = ([str(d) for d in cum["date"].values]
             if isinstance(cum, xr.DataArray) and "date" in cum.coords else None)
    write_envi_hdr(path, arr.shape[1], arr.shape[0], arr.shape[2], names)


def read_depl_cumule(path, width, length, n_dates, dates=None,
                     mask_nodata=True):
    """Read a Fortran `depl_cumule` file into a (date, y, x) DataArray."""
    arr = np.fromfile(path, dtype="<f4")
    length_eff = arr.size // (width * n_dates)   # Fortran may skip last row
    arr = arr[:length_eff * width * n_dates].reshape(length_eff, width, n_dates)
    arr = np.moveaxis(arr, -1, 0).astype(np.float32)
    if mask_nodata:
        arr = np.where(arr < 9990.0, arr, np.nan)
    coords = {"date": list(dates)} if dates is not None else {}
    return xr.DataArray(arr, dims=("date", "y", "x"), coords=coords,
                        name="cumulative_displacement")


def write_r4(arr2d, path, hdr=True):
    """Write one 2-D float32 map (NaN kept as NaN, like the Fortran zeros)."""
    a = np.ascontiguousarray(np.asarray(arr2d), dtype="<f4")
    a.tofile(path)
    if hdr:
        write_envi_hdr(path, a.shape[1], a.shape[0], 1)
