#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# __init__.py
# creation date: 2026-05-12.
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
import os
import sys
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import pathlib
import subprocess
import geoutils as gu
import rasterstats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpl_patches
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress

from geomulticorr.stats import nmad
from geomulticorr.corrections.corrections import (
    RampCorrection,
    TopoCorrection,
    TopoRampCorrection,
    SlopeRampCorrection,
    DirectionalBiasCorrection,
    AlongTrackDestriping,
    AcrossTrackDestriping,
)
from geomulticorr.corrections.fit import compute_slope
from geomulticorr.utils._pairs_frame import date_summary, unique_couples, unique_dates
from geomulticorr.utils._pairs_geometry import (
    CircularTimeScale,
    arc_heights,
    arc_ylim,
    chord_polylines,
    color_ring_mesh,
    polar_to_xy,
    polyline_point_and_tangent,
    timeline_arc_polylines,
    year_label_placement,
    year_tick_segments,
)

def outlier_filter(disp_array: np.array,
                   disp_threshold: tuple[int, float] = (-10, 10)) -> np.array:
    """Filter array value based on min & max threshold.

    Args:
        disp_array (np.array): 2D displacements array.
        disp_threshold (tuple[int, float], optional): Min & max threshold values for filter. Defaults to (-10, 10).

    Returns:
        np.array: Filtered array.
    """
    #TODO: add condition for open since we provide a text or pathlib object.
    diap_array_out = np.where(np.logical_or(disp_array <= disp_threshold[0],
                                        disp_array >= disp_threshold[1]),
                                        np.nan, disp_array)
    return diap_array_out
#END def

def cc_threshold_fiter(disp_array: np.array,
                       cc_array: np.array,
                       cc_threshold: float = 0.5) -> np.array:
    """Filter array based on cross correlation (CC) threshold.

    Args:
        disp_array (np.array): 2D displacements array.
        cc_array (np.array): 2D cross-correlation array.
        cc_threshold (float, optional): min value for cross-correlation threshold. Defaults to 0.5.

    Returns:
        np.array: Filtered array.
    """
    # Getting ncc map as boolean following threshold
    cc_arr_bool = np.where(cc_array >= cc_threshold, 1, 0)

    #Filter displacements using CC values
    disp_array[cc_arr_bool == 0] = np.nan
    return disp_array
#END def
# TO look at the end: this function is double.
# def remove_outliers(disp_array: np.array,
#                     cc_array: np.array,
#                     ncc_threshold: float = 0.5,
#                     disp_treshold: tuple[int, float] = (-10, 10)):

#     # Getting ncc map as boolean following threshold
#     cc_arr_bool = np.where(cc_array >= ncc_threshold, 1, 0)

#     # Filtering outliers values grater than threshold. disp_treshold parameter
#     disp_array_out = np.where(np.logical_or(disp_array <= disp_treshold[0],
#                                         disp_array >= disp_treshold[1]),
#                                         np.nan,
#                                         disp_array)
#     #Filter displacements using CC values
#     disp_array_out[cc_arr_bool == 0] = np.nan
#     return disp_array_out
# #END def

def center_disp(disp_array: np.array,
                stat: str = 'median') -> np.array:
    # Center values using the median value
    #TODO: add check function to check stat method.
    if stat == 'median':
        center_disp_array = disp_array - np.nanmedian(disp_array)
    if stat == 'mean':
        center_disp_array = disp_array - np.mean(disp_array)
    return center_disp_array

def remove_deramping():
    # Very similar to ondulations?
    return
#END def

def remove_destriping():
    # TODO:
    # Main concempt
    # 1. rotate timage to put it vertical.
    # 2. get average/median of rows and columns
    # 3. where average/median value is far from 0, substract average/median vaue.
    # 4. Ask again to pascal to be sure. 
    return
#END def

################################################################################
#TODO: Functions to visualize and explore data
################################################################################

def quick_view(array: np.array,
               ax: str = None,
               **plt_kwargs: any) -> None:
    """Quick view for 2D array. This function includes colorbar within the plot.

    Args:
        array (np.array): 2D np.array.
        ax (str, optional): axis for subplots. Defaults to None.
        **plt_kwargs (any): most common used parameters for figure plotting.
    Return:
        None
    """
    if ax is None:
        ax = plt.gca()
    #END if
    im = ax.imshow(array, **plt_kwargs)
    plt.colorbar(im, ax=ax, fraction=0.049, pad=0.04)
#END def
    
def quick_disp_plot(xDisp_array, yDisp_array, arr_freq=None, ax=None, **plt_kwargs):#arr_scale=50,
    if ax is None:
        ax = plt.gca()
    #END if
    magn_disp = np.sqrt((xDisp_array*xDisp_array) + (yDisp_array*yDisp_array))
    im = plt.imshow(magn_disp, **plt_kwargs)
    plt.colorbar(im, ax=ax, fraction=0.049, pad=0.04)

    nrows, ncols = xDisp_array.shape
    x, y = np.arange(0, ncols, 1), np.arange(0, nrows, 1)
    xi, yi = np.meshgrid(x, y, indexing='xy')
    if arr_freq is None:
        qver = plt.quiver(xi, yi, xDisp_array, yDisp_array,
                       color='k', pivot='mid', units='inches', scale_units='inches')#, scale=arr_scale)
    else:
        arr_scale = arr_freq/2
        qver = plt.quiver(xi[::arr_freq, ::arr_freq], yi[::arr_freq, ::arr_freq],
                          xDisp_array[::arr_freq, ::arr_freq], yDisp_array[::arr_freq, ::arr_freq],
                          color='k', pivot='mid', units='inches', scale_units='inches', scale=arr_scale)
    #END if
    qver_key = plt.quiverkey(qver, 0.79, 0.82, arr_scale, f'{arr_scale} m', coordinates='figure', labelpos='W')
#END def
    
def _as_nan_ndarray(arr) -> np.ndarray:
    """Convert array-like (ndarray, masked array) to 1-D float64 with NaN for masked values."""
    if np.ma.is_masked(arr):
        return arr.filled(np.nan).astype(float).ravel()
    return np.asarray(arr, dtype=float).ravel()

def compute_stats(arr: np.ndarray) -> dict:
    """Compute NMAD, median, mean, std on finite values of a 1-D float array."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {'nmad': np.nan, 'median': np.nan, 'mean': np.nan, 'std': np.nan}
    med = float(np.median(finite))
    return {
        'nmad':   float(1.4826 * np.median(np.abs(finite - med))),
        'median': med,
        'mean':   float(np.mean(finite)),
        'std':    float(np.std(finite)),
    }

def _draw_histogram_on_ax(
    ax,
    arrays,
    array_keys,
    error_metrics=None,
    xlim=None,
    ylim=None,
    alpha: float = 0.5,
    bins: int = 100,
    symmetric_xlim: bool = True,
    cmap: str | None = None,
    cmaps: list[str] | None = None,
    colors: list | None = None,
) -> list:
    """Draw histograms with stat annotations on an existing axis.

    Returns stats_list (one dict per input array).
    """
    if error_metrics is None:
        error_metrics = ['nmad', 'median']
    supported = ['nmad', 'median', 'mean', 'std']
    em_valid = [m for m in error_metrics if m in supported] or ['nmad', 'median']

    clean = [_as_nan_ndarray(a) for a in arrays]
    stats_list = [compute_stats(a) for a in clean]

    cmap_fn = plt.get_cmap("brg_r")
    color_cycle = [cmap_fn(v) for v in np.linspace(0.15, 1.0, max(len(clean), 1))]

    for i, (arr, key) in enumerate(zip(clean, array_keys)):
        finite = arr[np.isfinite(arr)]
        ax.hist(finite, bins=bins, edgecolor='none', alpha=alpha,
                label=key, color=color_cycle[i])

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        vals = np.concatenate([a[np.isfinite(a)] for a in clean])
        if vals.size > 0:
            low, up = np.percentile(vals, [1, 99])
            if symmetric_xlim:
                lim = max(abs(low), abs(up))
                ax.set_xlim(-lim, lim)
            else:
                ax.set_xlim(low, up)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        all_counts = []
        for arr in clean:
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                counts, _ = np.histogram(finite, bins=bins)
                all_counts.append(counts)
        if all_counts:
            up = np.percentile(np.concatenate(all_counts), 99)
            ax.set_ylim(0, up)

    # Annotation positioning: top-left for index 0, top-right for index 1
    ANCHORS = [
        (0.02, 0.98, 'left',  'top'),
        (0.98, 0.98, 'right', 'top'),
    ]
    for i, (key, stats) in enumerate(zip(array_keys, stats_list)):
        lines = [key] + [f"{m}: {stats[m]:.1f}" for m in em_valid]
        x, y, ha, va = ANCHORS[i]
        ax.text(x, y, "\n".join(lines), transform=ax.transAxes,
                va=va, ha=ha, fontsize=9, color=color_cycle[i],
                bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.6,
                          ec=color_cycle[i], lw=1))

    ax.axvline(0, color='black', alpha=0.5, linestyle=':')
    return stats_list


def plot_diff_histogram(
    diff_arrays,
    array_keys,
    error_metrics=None,
    xlim=None,
    ylim=None,
    figsize: tuple = (5, 5),
    alpha: float = 0.5,
    bins: int = 256,
    cmap: str | None = None,
    cmaps: list[str] | None = None,
    colors: list | None = None,
):
    """Plot histograms of one or more displacement/difference arrays with metric annotations.

    Parameters
    ----------
    diff_arrays : list of array-like
        Arrays of displacement or difference values (NaNs and masked values allowed).
    array_keys : list of str
        Labels corresponding to each array.
    error_metrics : list of {"nmad", "median", "mean", "std"}, optional
        Metrics to annotate on the plot. Defaults to ['nmad', 'median'].
    xlim, ylim : tuple or None
        Explicit axis limits. Auto-computed when None.
    figsize : tuple
        Figure size in inches.
    alpha : float
        Histogram bar transparency.
    bins : int
        Number of histogram bins.
    cmap : str, optional
        Colormap to use for histogram bars (color sampled at array median).
    cmaps : list of str, optional
        Per-array colormaps (one per array). Overrides cmap.
    colors : list, optional
        Explicit list of colors for histogram bars (overrides cmaps and cmap).

    Returns
    -------
    fig, ax, stats_list
        Figure, axis, and list of per-array stats dicts.
    """
    if len(diff_arrays) != len(array_keys):
        raise ValueError("diff_arrays and array_keys must have the same length.")
    if error_metrics is None:
        error_metrics = ['nmad', 'median']
    fig, ax = plt.subplots(1, figsize=figsize)
    stats_list = _draw_histogram_on_ax(
        ax, diff_arrays, array_keys, error_metrics, xlim, ylim, alpha, bins,
        symmetric_xlim=True, cmap=cmap, cmaps=cmaps, colors=colors,
    )
    ax.set_xlabel('Displacement / difference (m)')
    ax.set_ylabel('Count')
    fig.tight_layout()
    return fig, ax, stats_list


def get_rounded_limits(
        raster,
        round_to: float | None = None,
        symmetric: bool = True) -> float:
    """Return a display limit for a raster based on the 95th percentile of absolute values.

    Args:
        round_to: If given, round the limit *up* to the nearest multiple of this value.
                  If ``None`` (default), return the raw 95th-percentile value.
    """
    arr = np.asarray(raster.data)
    if np.ma.is_masked(arr):
        arr = arr.compressed()
    else:
        arr = arr.ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(round_to) if round_to is not None else 0.0
    lim = float(np.percentile(np.abs(arr), 95))
    if round_to is None:
        return lim
    return float(np.ceil(lim / round_to) * round_to)


# Hue convention shared by every correction diagnostic panel:
# EW = orange family (matches PuOr), NS = green family (matches PiYG);
# "before" is lighter/dashed, "after" is darker/solid.
_EW_BEFORE = "#f0a860"
_EW_AFTER  = "#b2560a"
_NS_BEFORE = "#8fce6a"
_NS_AFTER  = "#1a7a3a"


def _bin_disp_by_predictor(
    disp_vals: np.ndarray,
    pred_vals: np.ndarray,
    edges: np.ndarray,
    min_pixels: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median + NMAD of ``disp_vals`` within predictor bands defined by ``edges``.

    Bands with fewer than ``min_pixels`` valid pixels are dropped.

    Returns
    -------
    (bin_centers, medians, nmads) as 1-D float arrays.
    """
    finite = np.isfinite(disp_vals) & np.isfinite(pred_vals)
    disp_vals, pred_vals = disp_vals[finite], pred_vals[finite]
    centers, medians, nmads = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pred_vals >= lo) & (pred_vals < hi)
        if mask.sum() < min_pixels:
            continue
        v = disp_vals[mask]
        centers.append((lo + hi) / 2)
        medians.append(float(np.median(v)))
        nmads.append(nmad(v))
    return np.asarray(centers), np.asarray(medians), np.asarray(nmads)


def _draw_disp_vs_predictor_component_on_ax(
    ax,
    before: np.ndarray,
    after: np.ndarray,
    predictor: np.ndarray,
    *,
    label_prefix: str,
    c_before: str,
    c_after: str,
    bin_width: float,
    predictor_label: str = "Elevation (m)",
    min_pixels: int = 10,
    show_nmad: bool = True,
) -> None:
    """Binned median displacement vs predictor for one component (before & after only).

    Plots a single component's before/after curves on the given axis, with a title
    set to "{label_prefix} displacement vs {predictor_label}".
    """
    pred_finite = predictor[np.isfinite(predictor)]
    if pred_finite.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No predictor data available",
                ha="center", va="center", transform=ax.transAxes)
        return

    lo, hi = np.percentile(pred_finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(pred_finite.min()), float(pred_finite.max())
    n_bins = max(int(np.ceil((hi - lo) / bin_width)), 1)
    edges = lo + np.arange(n_bins + 1) * bin_width

    series = [
        (before, f"{label_prefix} before", c_before, "--"),
        (after,  f"{label_prefix} after",  c_after,  "-"),
    ]
    for disp, label, color, ls in series:
        centers, medians, nmads = _bin_disp_by_predictor(
            disp, predictor, edges, min_pixels=min_pixels
        )
        if centers.size == 0:
            continue
        ax.plot(medians, centers, marker="o", ms=3, lw=1.8, ls=ls,
                color=color, label=label)
        if show_nmad:
            ax.fill_betweenx(centers, medians - nmads, medians + nmads,
                             color=color, alpha=0.12)

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Displacement (m)")
    ax.set_ylabel(predictor_label)
    ax.set_title(f"{label_prefix} displacement vs {predictor_label.split(' (')[0].lower()}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _draw_disp_vs_predictor_on_ax(
    ax,
    ew_before: np.ndarray,
    ew_after: np.ndarray,
    ns_before: np.ndarray,
    ns_after: np.ndarray,
    predictor: np.ndarray,
    *,
    bin_width: float,
    predictor_label: str = "Elevation (m)",
    min_pixels: int = 10,
    show_nmad: bool = True,
) -> None:
    """Binned median displacement vs a topographic predictor for EW & NS, before & after.

    All five arrays are 1-D and pixel-aligned (same shape, NaNs allowed).  The
    ``predictor`` (elevation or terrain slope) is binned into ``bin_width``-wide
    bands; the median (± NMAD envelope) of each displacement series is plotted
    with displacement on the x-axis and the predictor on the y-axis.
    """
    pred_finite = predictor[np.isfinite(predictor)]
    if pred_finite.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No predictor data available",
                ha="center", va="center", transform=ax.transAxes)
        return

    lo, hi = np.percentile(pred_finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(pred_finite.min()), float(pred_finite.max())
    n_bins = max(int(np.ceil((hi - lo) / bin_width)), 1)
    edges = lo + np.arange(n_bins + 1) * bin_width

    series = [
        (ew_before, "EW before", _EW_BEFORE, "--"),
        (ew_after,  "EW after",  _EW_AFTER,  "-"),
        (ns_before, "NS before", _NS_BEFORE, "--"),
        (ns_after,  "NS after",  _NS_AFTER,  "-"),
    ]
    for disp, label, color, ls in series:
        centers, medians, nmads = _bin_disp_by_predictor(
            disp, predictor, edges, min_pixels=min_pixels
        )
        if centers.size == 0:
            continue
        ax.plot(medians, centers, marker="o", ms=3, lw=1.8, ls=ls,
                color=color, label=label)
        if show_nmad:
            ax.fill_betweenx(centers, medians - nmads, medians + nmads,
                             color=color, alpha=0.12)

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Displacement (m)")
    ax.set_ylabel(predictor_label)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)


def plot_show_raw_results(
    xDisp, yDisp, ncc,
    figsize: tuple[float, float] = (12, 6),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_cc: str = 'binary_r',
    fig_name=None,
    symmetric_limits: bool = True,
    hexbin_gridsize: int = 1000,
    hexbin_cmap: str = 'viridis',
    hexbin_nmad_multiplier: float = 1.0,
    hexbin_log_scale: bool = True,
) -> plt.Figure:
    """Compact displacement control figure with hexbin scatter and histograms.

    Layout: 2 rows × 3 cols with hexbin spanning both rows on the right.
    - xDisp (raster) | yDisp (raster)     | xyDisp_comp (hexbin, spans both)
    - ncc (raster)   | xyDisp_hist (hist) |
    """
    fig, axs = plt.subplot_mosaic(
        [['xDisp',       'yDisp',        'xyDisp_comp'],
         ['ncc',        'xyDisp_hist',  'xyDisp_comp']],
        width_ratios=[1, 1, 1.5],
        figsize=figsize,
    )
    if fig_name:
        fig.suptitle(fig_name)

    # Compute displacement limits
    vlim_x = get_rounded_limits(xDisp, round_to=10, symmetric=True)
    vlim_y = get_rounded_limits(yDisp, round_to=10, symmetric=True)

    if symmetric_limits:
        vmax = max(vlim_x, vlim_y)
        vlim_x = vlim_y = vmax

    # Displacement rasters
    xDisp.plot(ax=axs['xDisp'], cmap=cmap_ew, vmin=-vlim_x, vmax=vlim_x, title="Raw EW-Disp", cbar_title="Surf. disp (m)")
    yDisp.plot(ax=axs['yDisp'], cmap=cmap_ns, vmin=-vlim_y, vmax=vlim_y, title="Raw NS-Disp", cbar_title="Surf. disp (m)")

    # NCC raster
    if ncc is not None:
        ncc.plot(ax=axs['ncc'], cmap=cmap_cc, vmin=0, vmax=1, title="NCC", cbar_title="NCC")
    else:
        axs['ncc'].set_title("NCC (not available)")
        axs['ncc'].axis('off')

    # Combined EW/NS histogram
    _draw_histogram_on_ax(
        axs['xyDisp_hist'],
        [xDisp.data, yDisp.data],
        ["xDisp", "yDisp"],
        cmaps=[cmap_ew, cmap_ns],
    )
    axs['xyDisp_hist'].set_title("Displacement histograms")
    axs['xyDisp_hist'].set_xlabel("Displacement (m)")
    axs['xyDisp_hist'].set_ylabel("Pixel count")

    # Hexbin scatter plot of EW vs NS displacement
    x_flat = np.ma.filled(xDisp.data, np.nan).ravel()
    y_flat = np.ma.filled(yDisp.data, np.nan).ravel()
    valid = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_flat, y_flat = x_flat[valid], y_flat[valid]

    nmad_x = nmad(x_flat)
    nmad_y = nmad(y_flat)

    hb = axs['xyDisp_comp'].hexbin(x_flat, y_flat, gridsize=hexbin_gridsize, cmap=hexbin_cmap,
                                   mincnt=1, bins='log' if hexbin_log_scale else None)
    ellipse = Ellipse((0, 0), width=hexbin_nmad_multiplier*nmad_x, height=hexbin_nmad_multiplier*nmad_y,
                      edgecolor='white', facecolor='none', lw=1.5, label=f'± {hexbin_nmad_multiplier} NMAD')
    axs['xyDisp_comp'].add_patch(ellipse)
    axs['xyDisp_comp'].axhline(0, color='k', lw=0.5)
    axs['xyDisp_comp'].axvline(0, color='k', lw=0.5)
    axs['xyDisp_comp'].set_xlim(-vlim_x, vlim_x)
    axs['xyDisp_comp'].set_ylim(-vlim_y, vlim_y)
    axs['xyDisp_comp'].set_aspect('equal')
    axs['xyDisp_comp'].set_xlabel("EW displacement (m)")
    axs['xyDisp_comp'].set_ylabel("NS displacement (m)")
    axs['xyDisp_comp'].set_title("Pixel-wise displacement density")
    axs['xyDisp_comp'].legend(fontsize=8)

    # Colorbar for hexbin
    divider = make_axes_locatable(axs['xyDisp_comp'])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(hb, cax=cax, label='pixel count')

    fig.tight_layout(pad=1.0, w_pad=0.3, h_pad=0.4)
    return fig


def _apply_keep_mask(raster: gu.Raster, keep) -> gu.Raster:
    """Return a copy of *raster* with non-kept (``keep == False``) pixels masked out."""
    if keep is None:
        return raster
    import numpy.ma as ma
    base_mask = np.ma.getmaskarray(raster.data)
    new_mask = base_mask | ~np.asarray(keep, dtype=bool)
    return raster.copy(new_array=ma.array(raster.data.data, mask=new_mask))


def plot_show_corrected_results(
    xDisp, yDisp, ncc,
    x_stable=None,
    y_stable=None,
    figsize: tuple[float, float] = (14, 12),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_cc: str = 'binary_r',
    fig_name=None,
    symmetric_limits: bool = True,
    hexbin_gridsize: int = 1000,
    hexbin_cmap: str = 'viridis',
    hexbin_nmad_multiplier: float = 1.0,
    hexbin_log_scale: bool = True,
) -> plt.Figure:
    """Corrected-displacement dashboard (3×3) with stable-area panels.

    Layout
    ------
    EW disp   | NS disp   | hexbin (EW-vs-NS density, ±NMAD ellipse; spans rows 1–2)
    EW stable | NS stable | hexbin
    NCC       | EW/NS hist | .

    The *stable* panels show the corrected displacement with non-stable pixels
    masked out (the pixels the corrections were fitted on).  ``x_stable`` /
    ``y_stable`` are boolean *keep* masks; if ``None`` the full field is shown.
    """
    fig, axs = plt.subplot_mosaic(
        [['xDisp',    'yDisp',    'xyDisp_comp'],
         ['xStable',  'yStable',  'xyDisp_comp'],
         ['ncc',      'xyDisp_hist', '.']],
        figsize=figsize,
    )
    if fig_name:
        fig.suptitle(fig_name)

    vlim_x = get_rounded_limits(xDisp, round_to=1, symmetric=True)
    vlim_y = get_rounded_limits(yDisp, round_to=1, symmetric=True)
    if symmetric_limits:
        vlim_x = vlim_y = max(vlim_x, vlim_y)

    # Corrected displacement rasters
    xDisp.plot(ax=axs['xDisp'], cmap=cmap_ew, vmin=-vlim_x, vmax=vlim_x,
               title="Corrected EW-Disp", cbar_title="Surf. disp (m)")
    yDisp.plot(ax=axs['yDisp'], cmap=cmap_ns, vmin=-vlim_y, vmax=vlim_y,
               title="Corrected NS-Disp", cbar_title="Surf. disp (m)")

    # Stable-area-only displacement (non-stable pixels greyed out)
    _apply_keep_mask(xDisp, x_stable).plot(
        ax=axs['xStable'], cmap=cmap_ew, vmin=-vlim_x, vmax=vlim_x,
        title="EW (stable area)", cbar_title="Surf. disp (m)")
    _apply_keep_mask(yDisp, y_stable).plot(
        ax=axs['yStable'], cmap=cmap_ns, vmin=-vlim_y, vmax=vlim_y,
        title="NS (stable area)", cbar_title="Surf. disp (m)")

    # NCC raster
    if ncc is not None:
        ncc.plot(ax=axs['ncc'], cmap=cmap_cc, vmin=0, vmax=1, title="NCC", cbar_title="NCC")
    else:
        axs['ncc'].set_title("NCC (not available)")
        axs['ncc'].axis('off')

    # Combined EW/NS histogram
    _draw_histogram_on_ax(
        axs['xyDisp_hist'],
        [xDisp.data, yDisp.data],
        ["xDisp", "yDisp"],
        cmaps=[cmap_ew, cmap_ns],
    )
    axs['xyDisp_hist'].set_title("Displacement histograms")
    axs['xyDisp_hist'].set_xlabel("Displacement (m)")
    axs['xyDisp_hist'].set_ylabel("Pixel count")

    # Hexbin scatter of EW vs NS displacement (spans rows 1–2)
    x_flat = np.ma.filled(xDisp.data, np.nan).ravel()
    y_flat = np.ma.filled(yDisp.data, np.nan).ravel()
    valid = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_flat, y_flat = x_flat[valid], y_flat[valid]

    nmad_x = nmad(x_flat)
    nmad_y = nmad(y_flat)

    hb = axs['xyDisp_comp'].hexbin(x_flat, y_flat, gridsize=hexbin_gridsize, cmap=hexbin_cmap,
                                   mincnt=1, bins='log' if hexbin_log_scale else None)
    ellipse = Ellipse((0, 0), width=hexbin_nmad_multiplier*nmad_x, height=hexbin_nmad_multiplier*nmad_y,
                      edgecolor='white', facecolor='none', lw=1.5, label=f'± {hexbin_nmad_multiplier} NMAD')
    axs['xyDisp_comp'].add_patch(ellipse)
    axs['xyDisp_comp'].axhline(0, color='k', lw=0.5)
    axs['xyDisp_comp'].axvline(0, color='k', lw=0.5)
    axs['xyDisp_comp'].set_xlim(-vlim_x, vlim_x)
    axs['xyDisp_comp'].set_ylim(-vlim_y, vlim_y)
    axs['xyDisp_comp'].set_aspect('equal')
    axs['xyDisp_comp'].set_xlabel("EW displacement (m)")
    axs['xyDisp_comp'].set_ylabel("NS displacement (m)")
    axs['xyDisp_comp'].set_title("Pixel-wise displacement density")
    axs['xyDisp_comp'].legend(fontsize=8)

    divider = make_axes_locatable(axs['xyDisp_comp'])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(hb, cax=cax, label='pixel count')

    fig.tight_layout(pad=1.0, w_pad=0.3, h_pad=0.4)
    return fig


def plot_median_centering(
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    fig_name: str | None = None,
    figsize: tuple[float, float] = (12, 7),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
) -> plt.Figure:
    """3-col × 2-row control figure for a MedianCentering correction step.

    Layout
    ------
    a) EW before  |  b) EW after  |  c) EW histograms before / after
    d) NS before  |  e) NS after  |  f) NS histograms before / after
    """
    fig, axs = plt.subplot_mosaic(
        [['a)', 'b)', 'c)'],
         ['d)', 'e)', 'f)']],
        figsize=figsize,
    )
    if fig_name:
        fig.suptitle(f"{fig_name} — Median centering")

    vlim_x = get_rounded_limits(xDisp_before, round_to=1, symmetric=True)
    vlim_y = get_rounded_limits(yDisp_before, round_to=1, symmetric=True)

    xDisp_before.plot(ax=axs['a)'], cmap=cmap_ew, vmin=-vlim_x, vmax=vlim_x, title="EW before")
    xDisp_after.plot( ax=axs['b)'], cmap=cmap_ew, vmin=-vlim_x, vmax=vlim_x, title="EW after")
    _draw_histogram_on_ax(axs['c)'], [xDisp_before.data, xDisp_after.data],
                          ["EW before", "EW after"], symmetric_xlim=True, cmap=cmap_ew)
    axs['c)'].set_title("EW histogram")
    axs['c)'].set_xlabel("Displacement (m)")
    axs['c)'].set_ylabel("Count")

    yDisp_before.plot(ax=axs['d)'], cmap=cmap_ns, vmin=-vlim_y, vmax=vlim_y, title="NS before")
    yDisp_after.plot( ax=axs['e)'], cmap=cmap_ns, vmin=-vlim_y, vmax=vlim_y, title="NS after")
    _draw_histogram_on_ax(axs['f)'], [yDisp_before.data, yDisp_after.data],
                          ["NS before", "NS after"], symmetric_xlim=True, cmap=cmap_ns)
    axs['f)'].set_title("NS histogram")
    axs['f)'].set_xlabel("Displacement (m)")
    axs['f)'].set_ylabel("Count")

    fig.tight_layout()
    return fig

def _finalize_diagnostic_panel(
    ax,
    drew: bool,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    hline_ls: str = "--",
) -> None:
    """Format and annotate a diagnostic panel (seam profile, stripe profile, etc.)

    If ``drew`` is True, adds axis labels, title, legend, grid, and horizontal
    reference line. If False, turns off the axis and displays a "no data" message.
    """
    if drew:
        ax.axhline(0, color="k", ls=hline_ls, lw=0.6, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No profile metadata available",
                ha="center", va="center", transform=ax.transAxes)


def _draw_seam_profile_on_ax(ax, step, *, label_prefix, c_before, c_after) -> bool:
    """Draw one component's fitted seam profile (binned median vs residual after).

    Reads ``profile_centers`` / ``profile_median`` / ``profile_fitted`` from
    ``step.meta``.  ``before`` is the binned stable-ground median; ``after`` is
    ``median − fitted`` (the residual once the fitted profile is subtracted).
    Returns ``True`` if anything was drawn.
    """
    meta = getattr(step, "meta", {}) or {}
    centers = meta.get("profile_centers")
    median  = meta.get("profile_median")
    fitted  = meta.get("profile_fitted")
    if centers is None or median is None:
        return False
    centers = np.asarray(centers, dtype=float)
    median  = np.asarray(median, dtype=float)
    ax.plot(centers, median, ls="--", lw=1.5, color=c_before, alpha=0.9,
            label=f"{label_prefix} before")
    if fitted is not None:
        fitted = np.asarray(fitted, dtype=float)
        ax.plot(centers, median - fitted, ls="-", lw=1.8, color=c_after,
                label=f"{label_prefix} after")
    return True


def plot_directional_bias_correction(
    step_ew,
    step_ns,
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    fig_name: str | None = None,
    figsize: tuple[float, float] = (16, 13),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_bias: str = 'RdBu_r',
) -> plt.Figure:
    """Control figure for a DirectionalBiasCorrection step.

    The removed bias is the surface that was subtracted (before − after).
    The bottom panel shows the fitted 1-D seam profiles of **both** components
    (EW & NS, side-by-side) from ``step_ew.meta`` / ``step_ns.meta``
    (``profile_centers`` / ``profile_median`` / ``profile_fitted``), with
    before (dashed) and residual after (solid) on each.

    Layout
    ------
    a) EW before  |  b) EW bias  |  c) EW after  |  d) EW histogram
    e) NS before  |  f) NS bias  |  g) NS after  |  h) NS histogram
    i) EW seam profile  |  i) EW seam profile  |  j) NS seam profile  |  j) NS seam profile
    """
    fig, axs = plt.subplot_mosaic(
        [['a)', 'b)', 'c)', 'd)'],
         ['e)', 'f)', 'g)', 'h)'],
         ['i)', 'i)', 'j)', 'j)']],
        figsize=figsize,
    )
    angle_ew = step_ew.meta.get("angle_deg") if getattr(step_ew, "meta", None) else None
    angle_ns = step_ns.meta.get("angle_deg") if getattr(step_ns, "meta", None) else None
    title = "Directional bias correction"
    angles = [f"EW {angle_ew:.2f}°" if angle_ew is not None else None,
              f"NS {angle_ns:.2f}°" if angle_ns is not None else None]
    angles = [a for a in angles if a]
    if angles:
        title += " — angle: " + ", ".join(angles)
    if fig_name:
        title = f"{fig_name} — {title}"
    fig.suptitle(title)

    _draw_removed_surface_rows(
        axs, _ROW_KEYS, xDisp_before, xDisp_after, yDisp_before, yDisp_after,
        cmap_ew=cmap_ew, cmap_ns=cmap_ns, cmap_removed=cmap_bias, removed_label="bias",
    )

    # EW seam profile
    drew_ew = _draw_seam_profile_on_ax(axs['i)'], step_ew, label_prefix="EW",
                                       c_before=_EW_BEFORE, c_after=_EW_AFTER)
    m_ew = getattr(step_ew, "meta", {}) or {}
    sub_ew = []
    if m_ew.get("std_before") is not None and m_ew.get("std_after") is not None:
        sub_ew.append(f"std {m_ew['std_before']:.3f}→{m_ew['std_after']:.3f} m")
    title_ew = "EW seam profile" + (f"  ({'; '.join(sub_ew)})" if sub_ew else "")
    _finalize_diagnostic_panel(
        axs['i)'], drew_ew,
        title=title_ew,
        xlabel="Seam-normal coordinate s (m)",
        ylabel="Bias (m)",
        hline_ls=":",
    )

    # NS seam profile
    drew_ns = _draw_seam_profile_on_ax(axs['j)'], step_ns, label_prefix="NS",
                                       c_before=_NS_BEFORE, c_after=_NS_AFTER)
    m_ns = getattr(step_ns, "meta", {}) or {}
    sub_ns = []
    if m_ns.get("std_before") is not None and m_ns.get("std_after") is not None:
        sub_ns.append(f"std {m_ns['std_before']:.3f}→{m_ns['std_after']:.3f} m")
    title_ns = "NS seam profile" + (f"  ({'; '.join(sub_ns)})" if sub_ns else "")
    _finalize_diagnostic_panel(
        axs['j)'], drew_ns,
        title=title_ns,
        xlabel="Seam-normal coordinate s (m)",
        ylabel="Bias (m)",
        hline_ls=":",
    )

    fig.tight_layout()
    return fig
#END def

def _flat_disp(raster: gu.Raster) -> np.ndarray:
    """Displacement raster → 1-D float array with NaN for masked pixels."""
    return np.ma.filled(raster.data, np.nan).astype(float).ravel()


def _dem_on_grid(dem: gu.Raster, ref: gu.Raster) -> np.ndarray:
    """Reproject *dem* onto *ref*'s grid (if needed) → 2-D float array with NaNs."""
    d = dem
    if d.data.shape != ref.data.shape:
        d = d.reproject(ref=ref)
    return np.ma.filled(d.data, np.nan).astype(float)


def _draw_removed_surface_rows(
    axs: dict,
    keys: dict,
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    *,
    cmap_ew: str,
    cmap_ns: str,
    cmap_removed: str,
    removed_label: str,
) -> tuple[gu.Raster, gu.Raster]:
    """Draw the standard *before | removed | after | hist* rows for EW and NS.

    The removed surface is what was subtracted (``before − after``).  ``keys``
    maps the 8 logical panels to mosaic axis labels: ``ew_before``,
    ``ew_removed``, ``ew_after``, ``ew_hist`` and the ``ns_*`` equivalents.

    Returns the ``(x_removed, y_removed)`` rasters for optional reuse.
    """
    import numpy.ma as ma

    x_removed_data = ma.array(
        xDisp_before.data.data - xDisp_after.data.data,
        mask=np.ma.getmaskarray(xDisp_before.data),
    )
    y_removed_data = ma.array(
        yDisp_before.data.data - yDisp_after.data.data,
        mask=np.ma.getmaskarray(yDisp_before.data),
    )
    x_removed = xDisp_before.copy(new_array=x_removed_data)
    y_removed = yDisp_before.copy(new_array=y_removed_data)

    vlim_x  = get_rounded_limits(xDisp_before, round_to=1, symmetric=True)
    vlim_y  = get_rounded_limits(yDisp_before, round_to=1, symmetric=True)
    vlim_rx = get_rounded_limits(x_removed,    round_to=1, symmetric=True)
    vlim_ry = get_rounded_limits(y_removed,    round_to=1, symmetric=True)

    xDisp_before.plot(ax=axs[keys['ew_before']],  cmap=cmap_ew,      vmin=-vlim_x,  vmax=vlim_x,  title="EW before")
    x_removed.plot(   ax=axs[keys['ew_removed']], cmap=cmap_removed, vmin=-vlim_rx, vmax=vlim_rx, title=f"EW {removed_label}")
    xDisp_after.plot( ax=axs[keys['ew_after']],   cmap=cmap_ew,      vmin=-vlim_x,  vmax=vlim_x,  title="EW after")
    _draw_histogram_on_ax(axs[keys['ew_hist']], [xDisp_before.data, xDisp_after.data],
                          ["EW before", "EW after"], symmetric_xlim=True, cmap=cmap_ew)
    axs[keys['ew_hist']].set_title("EW histogram")
    axs[keys['ew_hist']].set_xlabel("Displacement (m)")
    axs[keys['ew_hist']].set_ylabel("Count")

    yDisp_before.plot(ax=axs[keys['ns_before']],  cmap=cmap_ns,      vmin=-vlim_y,  vmax=vlim_y,  title="NS before")
    y_removed.plot(   ax=axs[keys['ns_removed']], cmap=cmap_removed, vmin=-vlim_ry, vmax=vlim_ry, title=f"NS {removed_label}")
    yDisp_after.plot( ax=axs[keys['ns_after']],   cmap=cmap_ns,      vmin=-vlim_y,  vmax=vlim_y,  title="NS after")
    _draw_histogram_on_ax(axs[keys['ns_hist']], [yDisp_before.data, yDisp_after.data],
                          ["NS before", "NS after"], symmetric_xlim=True, cmap=cmap_ns)
    axs[keys['ns_hist']].set_title("NS histogram")
    axs[keys['ns_hist']].set_xlabel("Displacement (m)")
    axs[keys['ns_hist']].set_ylabel("Count")

    return x_removed, y_removed


# The 8-panel (before | removed | after | hist) × (EW, NS) key map used by the
# ramp/topo/slope templates.
_ROW_KEYS = dict(
    ew_before='a)', ew_removed='b)', ew_after='c)', ew_hist='d)',
    ns_before='e)', ns_removed='f)', ns_after='g)', ns_hist='h)',
)


def plot_ramp_correction(
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    fig_name: str | None = None,
    figsize: tuple[float, float] = (16, 7),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_ramp: str = 'RdBu_r',
) -> plt.Figure:
    """4-col × 2-row control figure for a RampCorrection step.

    The ramp shown is the surface that was subtracted (before − after).

    Layout
    ------
    a) EW before  |  b) EW ramp  |  c) EW after  |  d) EW histograms before / after
    e) NS before  |  f) NS ramp  |  g) NS after  |  h) NS histograms before / after
    """
    fig, axs = plt.subplot_mosaic(
        [['a)', 'b)', 'c)', 'd)'],
         ['e)', 'f)', 'g)', 'h)']],
        figsize=figsize,
    )
    if fig_name:
        fig.suptitle(f"{fig_name} — Ramp correction")

    _draw_removed_surface_rows(
        axs, _ROW_KEYS, xDisp_before, xDisp_after, yDisp_before, yDisp_after,
        cmap_ew=cmap_ew, cmap_ns=cmap_ns, cmap_removed=cmap_ramp, removed_label="ramp",
    )

    fig.tight_layout()
    return fig


def _plot_topo_like_correction(
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    *,
    predictor: np.ndarray | None,
    predictor_label: str,
    bin_width: float,
    suptitle: str,
    removed_label: str,
    fig_name: str | None,
    figsize: tuple[float, float],
    cmap_ew: str,
    cmap_ns: str,
    cmap_removed: str,
) -> plt.Figure:
    """Shared 4×3 template: removed-surface rows + disp-vs-predictor bottom panels (EW & NS)."""
    fig, axs = plt.subplot_mosaic(
        [['a)', 'b)', 'c)', 'd)'],
         ['e)', 'f)', 'g)', 'h)'],
         ['i)', 'i)', 'j)', 'j)']],
        figsize=figsize,
    )
    if fig_name:
        fig.suptitle(f"{fig_name} — {suptitle}")

    _draw_removed_surface_rows(
        axs, _ROW_KEYS, xDisp_before, xDisp_after, yDisp_before, yDisp_after,
        cmap_ew=cmap_ew, cmap_ns=cmap_ns, cmap_removed=cmap_removed, removed_label=removed_label,
    )

    if predictor is None:
        for ax_key in ['i)', 'j)']:
            axs[ax_key].axis("off")
            axs[ax_key].text(0.5, 0.5, "No DEM provided — predictor panel unavailable",
                           ha="center", va="center", transform=axs[ax_key].transAxes)
    else:
        _draw_disp_vs_predictor_component_on_ax(
            axs['i)'],
            _flat_disp(xDisp_before), _flat_disp(xDisp_after),
            predictor.ravel(),
            label_prefix="EW", c_before=_EW_BEFORE, c_after=_EW_AFTER,
            bin_width=bin_width, predictor_label=predictor_label,
        )
        _draw_disp_vs_predictor_component_on_ax(
            axs['j)'],
            _flat_disp(yDisp_before), _flat_disp(yDisp_after),
            predictor.ravel(),
            label_prefix="NS", c_before=_NS_BEFORE, c_after=_NS_AFTER,
            bin_width=bin_width, predictor_label=predictor_label,
        )

    fig.tight_layout()
    return fig


def plot_topo_correction(
    step_ew,
    step_ns,
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    *,
    dem: gu.Raster | None = None,
    fig_name: str | None = None,
    figsize: tuple[float, float] = (16, 13),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_removed: str = 'RdBu_r',
    elev_bin_width: float = 100.0,
) -> plt.Figure:
    """4-col × 3-row control figure for a TopoCorrection / TopoRampCorrection step.

    Rows 1–2 show *before | trend | after | histogram* for EW and NS.  Row 3
    (full width) shows binned-median displacement vs **elevation** (EW & NS,
    before & after) so the elevation-correlated bias and its removal are visible.
    Requires ``dem=`` (any grid; reprojected internally).
    """
    predictor = _dem_on_grid(dem, xDisp_before) if dem is not None else None
    return _plot_topo_like_correction(
        xDisp_before, xDisp_after, yDisp_before, yDisp_after,
        predictor=predictor, predictor_label="Elevation (m)", bin_width=elev_bin_width,
        suptitle="Topographic correction", removed_label="trend",
        fig_name=fig_name, figsize=figsize,
        cmap_ew=cmap_ew, cmap_ns=cmap_ns, cmap_removed=cmap_removed,
    )


def plot_slope_correction(
    step_ew,
    step_ns,
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    *,
    dem: gu.Raster | None = None,
    fig_name: str | None = None,
    figsize: tuple[float, float] = (16, 13),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_removed: str = 'RdBu_r',
    slope_bin_width: float = 2.0,
) -> plt.Figure:
    """4-col × 3-row control figure for a SlopeRampCorrection step.

    Same layout as :func:`plot_topo_correction`, but the row-3 panel shows
    binned-median displacement vs **terrain slope** (degrees, derived from the
    DEM via :func:`~geomulticorr.corrections.fit.compute_slope`).
    Requires ``dem=``.
    """
    predictor = None
    if dem is not None:
        dem_arr = _dem_on_grid(dem, xDisp_before)
        predictor = compute_slope(dem_arr, xDisp_before.transform)
    return _plot_topo_like_correction(
        xDisp_before, xDisp_after, yDisp_before, yDisp_after,
        predictor=predictor, predictor_label="Terrain slope (°)", bin_width=slope_bin_width,
        suptitle="Slope-ramp correction", removed_label="trend",
        fig_name=fig_name, figsize=figsize,
        cmap_ew=cmap_ew, cmap_ns=cmap_ns, cmap_removed=cmap_removed,
    )


def _draw_stripe_profile_on_ax(ax, step, *, label_prefix, c_before, c_after) -> bool:
    """Draw one component's 1-D stripe profile (raw vs model vs residual).

    Reads ``profile_raw`` / ``profile_model`` / ``spacing`` from ``step.meta``.
    ``before`` = raw stable-ground profile; the fitted ``model`` is drawn thin;
    ``after`` = ``raw − model``.  Returns ``True`` if anything was drawn.
    """
    meta = getattr(step, "meta", {}) or {}
    profile_raw = meta.get("profile_raw")
    if profile_raw is None:
        return False
    profile_raw = np.asarray(profile_raw, dtype=float)
    profile_model = meta.get("profile_model")
    spacing = meta.get("spacing", 1.0)
    coord = np.arange(len(profile_raw)) * spacing
    ax.plot(coord, profile_raw, ls="--", lw=1.0, color=c_before, alpha=0.9,
            label=f"{label_prefix} before")
    if profile_model is not None:
        profile_model = np.asarray(profile_model, dtype=float)
        ax.plot(coord, profile_model, ls=":", lw=1.4, color=c_after, alpha=0.7,
                label=f"{label_prefix} model")
        ax.plot(coord, profile_raw - profile_model, ls="-", lw=1.6, color=c_after,
                label=f"{label_prefix} after")
    return True


def _fmt_band(band) -> str | None:
    """Format a (lo, hi) wavelength band in metres as 'lo–hi m'."""
    if band is None:
        return None
    lo, hi = band
    return f"{lo:.0f}–{'∞' if hi is None else f'{hi:.0f}'} m"


def plot_destriping_correction(
    step_ew,
    step_ns,
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    fig_name: str | None = None,
    figsize: tuple[float, float] = (16, 13),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_bias: str = 'RdBu_r',
) -> plt.Figure:
    """Control figure for an Along-/Across-track destriping step.

    The removed undulation is the surface that was subtracted (before − after).
    The bottom panel shows the 1-D stripe profiles of **both** components
    (EW & NS, side-by-side) from ``step_ew.meta`` / ``step_ns.meta``
    (``profile_raw`` vs fitted ``profile_model`` vs residual).

    Layout
    ------
    a) EW before  |  b) EW undulation  |  c) EW after  |  d) EW histogram
    e) NS before  |  f) NS undulation  |  g) NS after  |  h) NS histogram
    i) EW stripe profile  |  i) EW stripe profile  |  j) NS stripe profile  |  j) NS stripe profile
    """
    fig, axs = plt.subplot_mosaic(
        [['a)', 'b)', 'c)', 'd)'],
         ['e)', 'f)', 'g)', 'h)'],
         ['i)', 'i)', 'j)', 'j)']],
        figsize=figsize,
    )
    band_ew = _fmt_band((getattr(step_ew, "meta", {}) or {}).get("band"))
    band_ns = _fmt_band((getattr(step_ns, "meta", {}) or {}).get("band"))
    title = "Destriping correction"
    bands = [f"EW {band_ew}" if band_ew else None, f"NS {band_ns}" if band_ns else None]
    bands = [b for b in bands if b]
    if bands:
        title += " — band: " + ", ".join(bands)
    if fig_name:
        title = f"{fig_name} — {title}"
    fig.suptitle(title)

    _draw_removed_surface_rows(
        axs, _ROW_KEYS, xDisp_before, xDisp_after, yDisp_before, yDisp_after,
        cmap_ew=cmap_ew, cmap_ns=cmap_ns, cmap_removed=cmap_bias, removed_label="undulation",
    )

    # EW stripe profile
    drew_ew = _draw_stripe_profile_on_ax(axs['i)'], step_ew, label_prefix="EW",
                                         c_before=_EW_BEFORE, c_after=_EW_AFTER)
    m_ew = getattr(step_ew, "meta", {}) or {}
    sub_ew = []
    if m_ew.get("std_before") is not None and m_ew.get("std_after") is not None:
        sub_ew.append(f"std {m_ew['std_before']:.3f}→{m_ew['std_after']:.3f} m")
    title_ew = "EW stripe profile" + (f"  ({'; '.join(sub_ew)})" if sub_ew else "")
    _finalize_diagnostic_panel(
        axs['i)'], drew_ew,
        title=title_ew,
        xlabel="Profile-axis coordinate (m)",
        ylabel="Displacement (m)",
        hline_ls="--",
    )

    # NS stripe profile
    drew_ns = _draw_stripe_profile_on_ax(axs['j)'], step_ns, label_prefix="NS",
                                         c_before=_NS_BEFORE, c_after=_NS_AFTER)
    m_ns = getattr(step_ns, "meta", {}) or {}
    sub_ns = []
    if m_ns.get("std_before") is not None and m_ns.get("std_after") is not None:
        sub_ns.append(f"std {m_ns['std_before']:.3f}→{m_ns['std_after']:.3f} m")
    title_ns = "NS stripe profile" + (f"  ({'; '.join(sub_ns)})" if sub_ns else "")
    _finalize_diagnostic_panel(
        axs['j)'], drew_ns,
        title=title_ns,
        xlabel="Profile-axis coordinate (m)",
        ylabel="Displacement (m)",
        hline_ls="--",
    )

    fig.tight_layout()
    return fig


def plot_correction_result(
    step_ew,
    step_ns,
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    fig_name: str | None = None,
    *,
    dem: gu.Raster | None = None,
    **_,
) -> plt.Figure:
    """Dispatch to the appropriate before/after figure for a correction step.

    ``step_ew`` / ``step_ns`` are the fitted EW and NS steps (their ``.meta`` feeds
    the diagnostic bottom panels); routing is decided on ``type(step_ew)``:

    - ``DirectionalBiasCorrection`` → :func:`plot_directional_bias_correction`
    - ``AlongTrackDestriping`` / ``AcrossTrackDestriping`` → :func:`plot_destriping_correction`
    - ``SlopeRampCorrection`` → :func:`plot_slope_correction` (needs ``dem=``)
    - ``TopoCorrection`` / ``TopoRampCorrection`` → :func:`plot_topo_correction` (needs ``dem=``)
    - ``RampCorrection`` → :func:`plot_ramp_correction`
    - All other steps → :func:`plot_median_centering`
    """
    if isinstance(step_ew, DirectionalBiasCorrection):
        return plot_directional_bias_correction(
            step_ew, step_ns, xDisp_before, xDisp_after, yDisp_before, yDisp_after,
            fig_name=fig_name,
        )
    if isinstance(step_ew, (AlongTrackDestriping, AcrossTrackDestriping)):
        return plot_destriping_correction(
            step_ew, step_ns, xDisp_before, xDisp_after, yDisp_before, yDisp_after,
            fig_name=fig_name,
        )
    if isinstance(step_ew, SlopeRampCorrection):
        return plot_slope_correction(
            step_ew, step_ns, xDisp_before, xDisp_after, yDisp_before, yDisp_after,
            dem=dem, fig_name=fig_name,
        )
    if isinstance(step_ew, (TopoCorrection, TopoRampCorrection)):
        return plot_topo_correction(
            step_ew, step_ns, xDisp_before, xDisp_after, yDisp_before, yDisp_after,
            dem=dem, fig_name=fig_name,
        )
    if isinstance(step_ew, RampCorrection):
        return plot_ramp_correction(xDisp_before, xDisp_after, yDisp_before, yDisp_after,
                                    fig_name=fig_name)
    return plot_median_centering(xDisp_before, xDisp_after, yDisp_before, yDisp_after,
                                 fig_name=fig_name)


def plot_disp_vs_elev_raw(
    disp_vals: np.ndarray,
    elev_vals: np.ndarray,
    xlabel: str = "Displacement (m)",
    ylabel: str = "Elevation (m)",
    fig_name: str | None = None,
    figsize: tuple[float, float] = (7, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of displacement vs elevation with a linear regression overlay.

    Parameters
    ----------
    disp_vals, elev_vals : np.ndarray
        1-D arrays of valid (finite) displacement and elevation values.
    xlabel, ylabel : str
        Axis labels.
    fig_name : str or None
        Optional figure title.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, ax
    """
    slope, intercept, r, p, _ = linregress(disp_vals, elev_vals)

    fig, ax = plt.subplots(figsize=figsize)
    if fig_name:
        ax.set_title(fig_name)

    ax.scatter(disp_vals, elev_vals, s=0.5, alpha=0.3, color="steelblue", label="pixels")
    x_line = np.linspace(disp_vals.min(), disp_vals.max(), 200)
    ax.plot(
        x_line, slope * x_line + intercept,
        color="crimson", lw=2,
        label=f"slope={slope:.4f}, R²={r**2:.3f}, p={p:.2e}",
    )
    ax.axhline(np.median(elev_vals), color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_disp_vs_elev_bins(
    disp_vals: np.ndarray,
    elev_vals: np.ndarray,
    n_bins: int = 30,
    min_pixels: int = 10,
    xlabel: str = "Displacement (m)",
    ylabel: str = "Elevation (m)",
    fig_name: str | None = None,
    figsize: tuple[float, float] = (7, 5),
    color: str = "steelblue",
) -> tuple[plt.Figure, plt.Axes]:
    """Median displacement per elevation band with ± NMAD uncertainty envelope.

    Bins *elev_vals* into *n_bins* equal-width bands and computes the median
    and NMAD of *disp_vals* within each band.  Bins with fewer than
    *min_pixels* pixels are silently dropped.

    Parameters
    ----------
    disp_vals, elev_vals : np.ndarray
        1-D arrays of valid (finite) displacement and elevation values.
    n_bins : int
        Number of equal-width elevation bins.
    min_pixels : int
        Minimum pixel count required to include a bin.
    xlabel, ylabel : str
        Axis labels.
    fig_name : str or None
        Optional figure title.
    figsize : tuple
        Figure size in inches.
    color : str
        Line and fill colour.

    Returns
    -------
    fig, ax
    """
    edges = np.linspace(elev_vals.min(), elev_vals.max(), n_bins + 1)
    bin_centers, medians, nmads = _bin_disp_by_predictor(
        disp_vals, elev_vals, edges, min_pixels=min_pixels
    )

    fig, ax = plt.subplots(figsize=figsize)
    if fig_name:
        ax.set_title(fig_name)

    ax.plot(medians, bin_centers, marker="o", ms=4, color=color, label="Median per bin")
    ax.fill_betweenx(
        bin_centers,
        medians - nmads,
        medians + nmads,
        alpha=0.25, color=color, label="± NMAD",
    )
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_disp_vs_elev_hexbin(
    disp_vals: np.ndarray,
    elev_vals: np.ndarray,
    gridsize: int = 60,
    mincnt: int = 5,
    cmap: str = "YlOrRd",
    log_scale: bool = True,
    xlabel: str = "Displacement (m)",
    ylabel: str = "Elevation (m)",
    fig_name: str | None = None,
    figsize: tuple[float, float] = (7, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """2-D density hexbin plot of displacement vs elevation.

    Parameters
    ----------
    disp_vals, elev_vals : np.ndarray
        1-D arrays of valid (finite) displacement and elevation values.
    gridsize : int
        Number of hexagons along the x-axis.
    mincnt : int
        Minimum count per hexagon to display.
    cmap : str
        Matplotlib colormap name.
    log_scale : bool
        If ``True``, use logarithmic colour scaling (``bins='log'``).
    xlabel, ylabel : str
        Axis labels.
    fig_name : str or None
        Optional figure title.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    if fig_name:
        ax.set_title(fig_name)

    hb = ax.hexbin(
        disp_vals, elev_vals,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=mincnt,
        bins="log" if log_scale else None,
    )
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)

    cb_label = "log count" if log_scale else "count"
    plt.colorbar(hb, ax=ax, label=cb_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax


def sample_raster(points_fn: gpd.GeoDataFrame | str | pathlib.Path,
                  raster_fn: rasterio.io.DatasetReader | str | pathlib.Path,
                  with_buffer: int | float = None) -> gpd.GeoDataFrame:
    """Tool for sampling raster file.

    Args:
        points_fn (gpd.GeoDataFrame | str | pathlib.Path): GeoDataFrame of XYZ points.
        raster_fn (rasterio.io.DatasetReader | str | pathlib.Path): Raster dataset.
        with_buffer (int | float, optional): Buffer radius in raster units. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Return the same GeoDataFrame of XYZ points with a new column containing sampled values based on point or buffer strategy.
    """
    # TODO: This function suppose that each GeoDataFrame contain a column named point_id.
    # Check input format
    if isinstance(points_fn, gpd.GeoDataFrame) == False:
        points_df = gpd.open(points_fn)
    else:
        points_df = points_fn
    #END if
    if isinstance(raster_fn, rasterio.io.DatasetReader) == False:
        raster_ds = rasterio.open(raster_fn)
    else:
        raster_ds = raster_fn
    #END if
    # Check if epsg codes are coherent
    if raster_ds.crs.to_epsg() != points_df.crs.to_epsg():
        print(f"Different spatial reference found. Projecting {points_fn} into raster_ds.crs.to_epsg()")
        #TODO: project
    else:
        pass
    #END if
    if with_buffer is not None:
        buffer = points_df.buffer(with_buffer)
        gdf_buffer = gpd.GeoDataFrame({'index':buffer.index.tolist(),
                                       'geometry':buffer.geometry}).set_index('index')
        # Reading raster
        raster_data = raster_ds.read(1)
        # Get column name from filename and filling with nan
        col_name = raster_fn.name.split('.')[0]
        # Creating new column based on new `col_name`
        points_df[col_name] = np.nan
        # Iterate though each column (i.e. feature)
        ras_stats = rasterstats.zonal_stats(gdf_buffer, raster_data,
                                   affine=raster_ds.transform, stats=['mean'],
                                   geojson_out = True)
        for i in range(len(ras_stats)):
            points_df[col_name].loc[int(ras_stats[i]['id'])] = ras_stats[i]['properties']['mean']
    else:
        # Reading raster
        raster_data = raster_ds.read(1)
        # Get column name from filename and filling with nan
        col_name = raster_fn.name.split('.')[0]
        # Creating new column based on new `col_name`
        points_df[col_name] = np.nan
        # Iterate though each column (i.e. feature)
        for index, row in points_df.iterrows():
            point_name = row['point_id']
            x = row['geometry'].x
            y = row['geometry'].y
            # getting row and col index to sample value
            rowIndex, colIndex = raster_ds.index(x,y)
            # Filling dataframe with sampled values based on index values
            points_df[col_name].loc[index] = raster_data[rowIndex, colIndex]
        #END for
        # Closing raster to free memory
        raster_ds.close()
        #END for
    #END if
    return points_df
#END def

# ══════════════════════════════════════════════════════════════════════════════ #
#                          Pair-network figures                                  #
# ══════════════════════════════════════════════════════════════════════════════ #
# Both drawers consume a *pairs frame* (see geomulticorr.utils._pairs_frame) and
# compute their coordinates through geomulticorr.utils._pairs_geometry, which the
# interactive plotly views share — so the static and interactive figures draw
# provably identical curves.
#
# Performance note: every repeated element is batched into a SINGLE artist
# (one LineCollection for all links, one pcolormesh for the colour ring).  The
# naive per-element version costs ~1 s per redraw at a few hundred pairs; batched
# it is ~20 ms, which is what makes these usable inside an interactive loop.

#: Colours for the forward/backward split, shared with the plotly views.
_DIRECTION_COLORS: dict[str, str] = {"forward": "#1f77b4", "backward": "#ff7f0e"}


def _pairs_color_groups(frame: pd.DataFrame, color_by: str) -> tuple[dict, dict, str]:
    """Resolve *color_by* into per-pair colours, per-date node colours and a legend.

    Returns ``(groups, node_color, legend_title)`` where ``groups`` maps a label to
    a boolean row mask + colour.  ``"dt"`` returns an empty ``groups`` dict — that
    mode is drawn as one continuous-valued LineCollection instead.
    """
    if color_by == "sensor":
        sensors = sorted(set(frame["sensor_i"]) | set(frame["sensor_j"]))
        palette = matplotlib.colormaps["tab10"].resampled(max(len(sensors), 1))
        colors = {s: palette(i) for i, s in enumerate(sensors)}
        groups = {s: (frame["sensor_i"] == s, colors[s]) for s in sensors}
        node_color = {"__by_sensor__": colors}
        return groups, node_color, "Sensor"

    if color_by == "pzone":
        pzones = sorted(frame["pz"].unique())
        palette = matplotlib.colormaps["Set2"].resampled(max(len(pzones), 1))
        colors = {pz: palette(i) for i, pz in enumerate(pzones)}
        return {pz: (frame["pz"] == pz, colors[pz]) for pz in pzones}, {}, "Pzone"

    if color_by == "direction":
        return (
            {
                d: (frame["direction"] == d, c)
                for d, c in _DIRECTION_COLORS.items()
                if (frame["direction"] == d).any()
            },
            {},
            "Direction",
        )

    if color_by == "dt":
        return {}, {}, "Time delta (years)"

    raise ValueError(
        f"Unknown color_by '{color_by}'. Valid options: 'sensor', 'pzone', 'direction', 'dt'."
    )


def _draw_pairs_network_on_ax(
    ax,
    frame: pd.DataFrame,
    *,
    color_by: str = "sensor",
    cmap_dt: str = "plasma_r",
    linewidth: float = 1.5,
    alpha: float = 0.75,
    n_bezier_points: int = 24,
    node_size: float = 60,
    max_xticks: int = 25,
    mirror_direction: bool = True,
    dedupe: bool = False,
) -> dict:
    """Draw a temporal pair network (timeline + Bézier arcs) into *ax*.

    Each acquisition date is a node on a horizontal timeline; each pair is an arc
    whose height grows with the temporal baseline, so short pairs sit low and
    long pairs arch higher.  With *mirror_direction* the plot is two-sided:
    forward pairs arch above the timeline and backward pairs below it, so each
    pair's direction is readable from geometry alone.

    :param ax: Target matplotlib Axes.
    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param color_by: ``"sensor"`` | ``"pzone"`` | ``"direction"`` | ``"dt"``.
    :param cmap_dt: Colormap used when ``color_by="dt"``.
    :param linewidth: Arc line width.
    :param alpha: Arc opacity.
    :param n_bezier_points: Samples per arc; raise for print-quality curves.
    :param node_size: Marker size of the date nodes.
    :param max_xticks: Label at most this many dates on the x axis, thinning at a
        regular stride beyond it (every date still gets a node).  Long archives
        otherwise produce an unreadable smear of overlapping labels.
    :param mirror_direction: Draw backward pairs below the axis.  ``False``
        restores the one-sided layout exactly.
    :param dedupe: Drop duplicate unordered couples (``redundancy`` /
        ``forward-backward`` emit each couple twice).  Defaults to ``False`` so
        the arc density matches the number of pairs actually built.
    :returns: ``{"n_pairs", "n_dates", "mappable", "legend_handles", "legend_title"}``.
    """
    if dedupe:
        frame = unique_couples(frame)

    dates = unique_dates(frame)
    if len(frame) == 0 or len(dates) == 0:
        raise ValueError("Cannot draw a pair network from an empty pairs frame.")

    x1 = frame["dec_i"].to_numpy()
    x2 = frame["dec_j"].to_numpy()
    backward = (frame["direction"] == "backward").to_numpy()
    height = arc_heights(frame["dt_years"].to_numpy(), backward=backward,
                         mirror=mirror_direction)
    curves = timeline_arc_polylines(x1, x2, height, n_points=n_bezier_points)

    mappable = None
    legend_handles: list = []
    groups, node_color, legend_title = _pairs_color_groups(frame, color_by)

    if color_by == "dt":
        # One artist, continuous colour — matplotlib can do here what plotly cannot.
        dt_vals = frame["dt_years"].to_numpy()
        norm = mcolors.Normalize(vmin=dt_vals.min(), vmax=dt_vals.max())
        lc = LineCollection(curves, cmap=cmap_dt, norm=norm, linewidths=linewidth, alpha=alpha)
        lc.set_array(dt_vals)
        ax.add_collection(lc)
        mappable = lc
    else:
        for label, (mask, color) in groups.items():
            sel = curves[mask.to_numpy()]
            if len(sel) == 0:
                continue
            ax.add_collection(
                LineCollection(sel, colors=[color], linewidths=linewidth, alpha=alpha)
            )
            legend_handles.append(mpl_patches.Patch(color=color, label=label))

    # --- timeline and date nodes -------------------------------------------------
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    node_dec = np.array(
        [d.year + (d - pd.Timestamp(d.year, 1, 1)) / pd.Timedelta(days=365.25) for d in dates]
    )
    if color_by == "sensor" and node_color:
        summary = date_summary(frame)
        sensor_colors = node_color["__by_sensor__"]
        node_c = [sensor_colors.get(s, "grey") for s in summary["sensor"]]
    else:
        node_c = "black"
    ax.scatter(node_dec, np.zeros_like(node_dec), c=node_c, s=node_size, zorder=3,
               edgecolors="black", linewidths=0.5)

    # --- axes formatting ---------------------------------------------------------
    x_pad = (node_dec.max() - node_dec.min()) * 0.03 if len(node_dec) > 1 else 0.1
    ax.set_xlim(node_dec.min() - x_pad, node_dec.max() + x_pad)
    ax.set_ylim(*arc_ylim(height))
    stride = max(1, int(np.ceil(len(node_dec) / max(max_xticks, 1))))
    ax.set_xticks(node_dec[::stride])
    ax.set_xticklabels(dates.strftime("%Y-%m-%d")[::stride], rotation=45, ha="right",
                       fontsize=8)
    # No y ticks: arc height is a non-linear proxy for Δt (a cubic with both
    # control points at h peaks at 0.75 h), so a numeric axis would misinform.
    ax.set_yticks([])
    ax.set_ylabel("")
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(False)

    if mirror_direction:
        if (~backward).any():
            ax.text(0.004, 0.985, "▲ above: forward (i → j)", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8,
                    color=_DIRECTION_COLORS["forward"])
        if backward.any():
            ax.text(0.004, 0.015, "▼ below: backward (j → i)", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8,
                    color=_DIRECTION_COLORS["backward"])

    return {
        "n_pairs": len(frame),
        "n_dates": len(dates),
        "mappable": mappable,
        "legend_handles": legend_handles,
        "legend_title": legend_title,
        "mirrored": bool(mirror_direction and backward.any()),
    }


def plot_pairs_network(
    frame: pd.DataFrame,
    figsize: tuple[float, float] = (14, 5),
    color_by: str = "sensor",
    fig_name: str | None = None,
    ax=None,
    **style,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot image pairs as a temporal network diagram.

    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param figsize: Figure size in inches (ignored when *ax* is given).
    :param color_by: ``"sensor"`` | ``"pzone"`` | ``"direction"`` | ``"dt"``.
    :param fig_name: Optional title; a pair/date count title is used when omitted.
    :param ax: Draw into an existing Axes instead of creating a figure.
    :param style: Forwarded to :func:`_draw_pairs_network_on_ax` (``linewidth``,
        ``alpha``, ``n_bezier_points``, ``node_size``, ``cmap_dt``, ``dedupe``).
    :returns: ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    info = _draw_pairs_network_on_ax(ax, frame, color_by=color_by, **style)

    n_pairs, n_dates = info["n_pairs"], info["n_dates"]
    ax.set_title(
        fig_name
        or f"Pair network — {n_pairs} pair{'s' if n_pairs != 1 else ''} "
           f"({n_dates} acquisition dates)",
        fontsize=10,
    )

    if info["mappable"] is not None:
        fig.colorbar(info["mappable"], ax=ax, orientation="vertical",
                     label=info["legend_title"], fraction=0.03, pad=0.02)
    elif info["legend_handles"]:
        ax.legend(handles=info["legend_handles"], title=info["legend_title"],
                  loc="upper left", fontsize=8, title_fontsize=9)

    fig.tight_layout()
    return fig, ax


def _draw_pairs_chord_on_ax(
    ax,
    frame: pd.DataFrame,
    *,
    cmap: str = "jet",
    n_color_segments: int = 720,
    outer_radius: float = 1.0,
    ring_width: float = 0.045,
    link_radius: float | None = None,
    label_radius: float = 1.02,
    show_year_ticks: bool = True,
    year_tick_length: float = 0.075,
    year_tick_linewidth: float = 1.4,
    year_tick_color: str = "black",
    show_year_labels: bool = True,
    year_label_fontsize: float = 9,
    year_label_color: str = "black",
    link_color: str = "black",
    link_linewidth: float = 0.6,
    link_alpha: float = 0.35,
    link_curvature: float = 0.85,
    border_color: str = "black",
    border_linewidth: float = 1.0,
    show_arrows: bool = False,
    arrow_position: float = 0.5,
    arrow_size: float = 0.05,
    arrow_width: float = 0.005,
    arrow_color: str | None = None,
    arrow_alpha: float | None = None,
    show_date_nodes: bool = False,
    node_size: float = 18,
    xylim: float = 1.30,
    n_bezier_points: int = 24,
    dedupe: bool = True,
) -> dict:
    """Draw a circular chord diagram of the pair network into *ax*.

    Dates are placed on the circle **proportionally to time** (Jan 1 at the top,
    clockwise), so seasonal clustering and gaps in the archive are visible at a
    glance — unlike an equal-angle layout, which hides irregular sampling.  The
    colour ring encodes position within the covered period.

    :param ax: Target matplotlib Axes (should be ``aspect="equal"``).
    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param cmap: Colormap of the time ring.
    :param n_color_segments: Ring resolution — drawn as one ``pcolormesh``, so
        raising it costs almost nothing.
    :param outer_radius, ring_width: Ring geometry.
    :param link_radius: Radius of the chord endpoints; defaults to the ring
        mid-radius.
    :param label_radius: Radius of the year labels.
    :param show_year_ticks, year_tick_*: The ``|`` marks at Jan 1 of each year.
    :param show_year_labels, year_label_*: Year text around the ring.
    :param link_color, link_linewidth, link_alpha: Chord styling.
    :param link_curvature: 0 → chords hug the rim, 1 → antipodal chords pass
        through the centre.
    :param border_color, border_linewidth: The two ring outlines.
    :param show_arrows: Draw an arrowhead per chord pointing from the left image
        to the right image of the pair.  Off by default — past roughly 150 chords
        the heads crowd each other.
    :param arrow_position: Where along each chord the head sits, ``0``–``1``.
        Mid-chord by default: chord *endpoints* coincide with the acquisition
        angles, so end-placed heads from many pairs would pile up on a few points.
    :param arrow_size: Arrow length in data units (the ring has radius 1).
    :param arrow_width, arrow_color, arrow_alpha: Arrow styling; the colour and
        opacity default to a stronger version of the chord's own.
    :param show_date_nodes, node_size: Optional markers at each acquisition date.
    :param xylim: Axis half-extent, leaving room for the labels.
    :param n_bezier_points: Samples per chord.
    :param dedupe: Drop duplicate unordered couples — ``True`` by default because
        overplotting the same chord twice doubles its effective opacity.
    :returns: ``{"n_pairs", "n_dates", "scale", "mappable", "legend_handles",
        "legend_title"}``.
    """
    if dedupe:
        frame = unique_couples(frame)

    dates = unique_dates(frame)
    if len(frame) == 0 or len(dates) < 2:
        raise ValueError("A chord diagram needs at least two distinct acquisition dates.")

    scale = CircularTimeScale.from_dates(dates)
    inner_radius = outer_radius - ring_width
    if link_radius is None:
        link_radius = inner_radius + ring_width / 2

    ax.set_aspect("equal")
    ax.axis("off")

    # --- colour ring: ONE artist (vs n_color_segments wedges) --------------------
    ring_x, ring_y, ring_frac = color_ring_mesh(
        n_color_segments, outer_radius=outer_radius, ring_width=ring_width
    )
    mesh = ax.pcolormesh(ring_x, ring_y, ring_frac[None, :], cmap=cmap,
                         shading="flat", zorder=0, rasterized=True)

    for radius in (outer_radius, inner_radius):
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor=border_color,
                            linewidth=border_linewidth, zorder=4))

    # --- chords: ONE artist ------------------------------------------------------
    theta_i = scale.angle(frame["t_i"])
    theta_j = scale.angle(frame["t_j"])
    curves = chord_polylines(theta_i, theta_j, link_radius=link_radius,
                             curvature=link_curvature, n_points=n_bezier_points)
    ax.add_collection(
        LineCollection(curves, colors=[link_color], linewidths=link_linewidth,
                       alpha=link_alpha, zorder=1)
    )

    # --- arrowheads: ONE artist (quiver takes per-point directions natively) ----
    if show_arrows:
        tips, tangents = polyline_point_and_tangent(curves, position=arrow_position)
        ax.quiver(
            tips[:, 0], tips[:, 1], tangents[:, 0], tangents[:, 1],
            angles="xy", scale_units="xy", scale=1.0 / arrow_size,
            width=arrow_width, headwidth=4.0, headlength=5.0, headaxislength=5.0,
            pivot="mid", color=arrow_color or link_color,
            alpha=arrow_alpha if arrow_alpha is not None else min(link_alpha * 2, 1.0),
            zorder=2,
        )

    if show_date_nodes:
        node_x, node_y = polar_to_xy(link_radius, scale.angle(dates))
        ax.scatter(node_x, node_y, s=node_size, c=scale.fraction(dates), cmap=cmap,
                   vmin=0, vmax=1, edgecolors="black", linewidths=0.4, zorder=5)

    # --- year ticks: ONE artist --------------------------------------------------
    if show_year_ticks:
        ax.add_collection(
            LineCollection(
                year_tick_segments(scale, outer_radius=outer_radius,
                                   ring_width=ring_width, tick_length=year_tick_length),
                colors=[year_tick_color], linewidths=year_tick_linewidth,
                capstyle="butt", zorder=5,
            )
        )

    if show_year_labels:
        lx, ly, rot, ha = year_label_placement(scale, label_radius=label_radius)
        for x, y, r, a, year in zip(lx, ly, rot, ha, scale.years):
            ax.text(x, y, str(int(year)), rotation=r, rotation_mode="anchor",
                    ha=a, va="center", fontsize=year_label_fontsize,
                    color=year_label_color)

    ax.set_xlim(-xylim, xylim)
    ax.set_ylim(-xylim, xylim)

    return {
        "n_pairs": len(frame),
        "n_dates": len(dates),
        "scale": scale,
        "mappable": mesh,
        "legend_handles": [],
        "legend_title": "Position in period",
    }


def plot_pairs_chord(
    frame: pd.DataFrame,
    figsize: tuple[float, float] = (10, 10),
    fig_name: str | None = None,
    ax=None,
    **style,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot image pairs as a time-proportional chord diagram.

    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param figsize: Figure size in inches (ignored when *ax* is given).
    :param fig_name: Optional title; a pair/date count title is used when omitted.
    :param ax: Draw into an existing Axes instead of creating a figure.
    :param style: Forwarded to :func:`_draw_pairs_chord_on_ax` — see there for the
        full list of geometry and styling parameters.
    :returns: ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})
    else:
        fig = ax.get_figure()

    info = _draw_pairs_chord_on_ax(ax, frame, **style)

    n_pairs, n_dates = info["n_pairs"], info["n_dates"]
    ax.set_title(
        fig_name
        or f"Pair network (chord diagram) — {n_pairs} pair{'s' if n_pairs != 1 else ''} "
           f"({n_dates} acquisition dates)",
        fontsize=11, pad=20,
    )

    fig.tight_layout()
    return fig, ax


def _draw_pairs_baseline_on_ax(
    ax,
    frame: pd.DataFrame,
    *,
    color_by: str = "direction",
    marker_size: float = 45,
    alpha: float = 0.75,
    cmap_dt: str = "plasma_r",
    dedupe: bool = False,
) -> dict:
    """Draw temporal baseline vs pair midpoint date into *ax*.

    The publication twin of
    :func:`geomulticorr.utils._pairs_plotly.figure_baseline` — same split into
    forward circles and backward diamonds, same colours.

    :param ax: Target matplotlib Axes.
    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param color_by: ``"direction"`` | ``"sensor"`` | ``"pzone"`` | ``"dt"``.
    :param marker_size: Scatter marker area.
    :param alpha: Marker opacity.
    :param cmap_dt: Colormap used when ``color_by="dt"``.
    :param dedupe: Drop duplicate unordered couples before plotting.
    :returns: ``{"n_pairs", "n_dates", "mappable", "legend_handles", "legend_title"}``.
    """
    if dedupe:
        frame = unique_couples(frame)
    if len(frame) == 0:
        raise ValueError("Cannot draw a baseline plot from an empty pairs frame.")

    mappable = None
    legend_handles: list = []
    legend_title = color_by.capitalize()

    if color_by == "dt":
        sc = ax.scatter(frame["mid_dec"], frame["dt_days"], c=frame["dt_years"],
                        cmap=cmap_dt, s=marker_size, alpha=alpha,
                        edgecolors="black", linewidths=0.3)
        mappable = sc
        legend_title = "Time delta (years)"
    elif color_by == "direction":
        # Backward first and larger: a bidirectional couple puts both markers on
        # exactly the same (midpoint, Δt), so the smaller forward circle must land
        # on top or one direction would be invisible.
        for name, marker, scale in (("backward", "D", 1.6), ("forward", "o", 1.0)):
            sub = frame[frame["direction"] == name]
            if len(sub) == 0:
                continue
            ax.scatter(sub["mid_dec"], sub["dt_days"], marker=marker,
                       color=_DIRECTION_COLORS[name], s=marker_size * scale,
                       alpha=alpha, edgecolors="black", linewidths=0.3, label=name)
            legend_handles.append(
                mpl_patches.Patch(color=_DIRECTION_COLORS[name], label=name)
            )
        legend_handles.reverse()
    else:
        groups, _, legend_title = _pairs_color_groups(frame, color_by)
        for label, (mask, color) in groups.items():
            sub = frame.loc[mask]
            if len(sub) == 0:
                continue
            ax.scatter(sub["mid_dec"], sub["dt_days"], color=color, s=marker_size,
                       alpha=alpha, edgecolors="black", linewidths=0.3, label=label)
            legend_handles.append(mpl_patches.Patch(color=color, label=label))

    ax.set_xlabel("Pair midpoint date (decimal year)")
    ax.set_ylabel("Temporal baseline Δt (days)")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    return {
        "n_pairs": len(frame),
        "n_dates": len(unique_dates(frame)),
        "mappable": mappable,
        "legend_handles": legend_handles,
        "legend_title": legend_title,
    }


def plot_pairs_baseline(
    frame: pd.DataFrame,
    figsize: tuple[float, float] = (10, 5),
    color_by: str = "direction",
    fig_name: str | None = None,
    ax=None,
    **style,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot temporal baseline against pair midpoint date.

    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param figsize: Figure size in inches (ignored when *ax* is given).
    :param color_by: ``"direction"`` | ``"sensor"`` | ``"pzone"`` | ``"dt"``.
    :param fig_name: Optional title; a pair/date count title is used when omitted.
    :param ax: Draw into an existing Axes instead of creating a figure.
    :param style: Forwarded to :func:`_draw_pairs_baseline_on_ax`.
    :returns: ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    info = _draw_pairs_baseline_on_ax(ax, frame, color_by=color_by, **style)

    n_pairs, n_dates = info["n_pairs"], info["n_dates"]
    ax.set_title(
        fig_name
        or f"Temporal baselines — {n_pairs} pair{'s' if n_pairs != 1 else ''} "
           f"({n_dates} acquisition dates)",
        fontsize=10,
    )
    if info["mappable"] is not None:
        fig.colorbar(info["mappable"], ax=ax, orientation="vertical",
                     label=info["legend_title"], fraction=0.03, pad=0.02)
    elif info["legend_handles"]:
        ax.legend(handles=info["legend_handles"], title=info["legend_title"],
                  loc="upper left", fontsize=8, title_fontsize=9)

    fig.tight_layout()
    return fig, ax


def _draw_pairs_dt_hist_on_ax(
    ax,
    frame: pd.DataFrame,
    *,
    nbins: int = 40,
    split_direction: bool = True,
    show_stats: bool = True,
    alpha: float = 0.7,
    dedupe: bool = False,
) -> dict:
    """Draw the distribution of temporal baselines into *ax*.

    The publication twin of
    :func:`geomulticorr.utils._pairs_plotly.figure_dt_hist`.

    :param ax: Target matplotlib Axes.
    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param nbins: Number of histogram bins.
    :param split_direction: Overlay forward and backward separately.
    :param show_stats: Draw dashed median and dotted mean guides.
    :param alpha: Bar opacity.
    :param dedupe: Drop duplicate unordered couples before binning.
    :returns: ``{"n_pairs", "n_dates", "mappable", "legend_handles", "legend_title"}``.
    """
    if dedupe:
        frame = unique_couples(frame)
    if len(frame) == 0:
        raise ValueError("Cannot draw a Δt histogram from an empty pairs frame.")

    dt = frame["dt_days"].to_numpy()
    bins = np.histogram_bin_edges(dt, bins=nbins)
    legend_handles: list = []

    present = [
        name for name in ("forward", "backward")
        if (frame["direction"] == name).any()
    ] if split_direction else []

    if len(present) > 1:
        # Grouped, not overlaid: matplotlib draws successive hist() calls opaquely
        # over each other, and for bidirectional strategies the two distributions
        # are identical by construction — so an overlay would hide one entirely.
        ax.hist(
            [dt[(frame["direction"] == name).to_numpy()] for name in present],
            bins=bins, color=[_DIRECTION_COLORS[n] for n in present],
            alpha=alpha, label=present,
        )
        legend_handles = [
            mpl_patches.Patch(color=_DIRECTION_COLORS[n], label=n) for n in present
        ]
    elif present:
        ax.hist(dt, bins=bins, color=_DIRECTION_COLORS[present[0]], alpha=alpha,
                label=present[0])
        legend_handles = [
            mpl_patches.Patch(color=_DIRECTION_COLORS[present[0]], label=present[0])
        ]
    else:
        ax.hist(dt, bins=bins, color="grey", alpha=alpha, label="all pairs")

    if show_stats:
        stats = ((np.median(dt), "median", "--", 0.99), (dt.mean(), "mean", ":", 0.91))
        for value, label, ls, y in stats:
            ax.axvline(value, color="black", lw=1.2, linestyle=ls)
            ax.annotate(f"{label} {value:.0f} d", xy=(value, y),
                        xycoords=("data", "axes fraction"), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="top", fontsize=8)

    ax.set_xlabel("Temporal baseline Δt (days)")
    ax.set_ylabel("Number of pairs")
    ax.grid(True, alpha=0.25, linestyle=":", axis="y")
    ax.set_axisbelow(True)

    return {
        "n_pairs": len(frame),
        "n_dates": len(unique_dates(frame)),
        "mappable": None,
        "legend_handles": legend_handles,
        "legend_title": "Direction",
    }


def plot_pairs_dt_hist(
    frame: pd.DataFrame,
    figsize: tuple[float, float] = (8, 5),
    fig_name: str | None = None,
    ax=None,
    **style,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the distribution of temporal baselines.

    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param figsize: Figure size in inches (ignored when *ax* is given).
    :param fig_name: Optional title; a pair/date count title is used when omitted.
    :param ax: Draw into an existing Axes instead of creating a figure.
    :param style: Forwarded to :func:`_draw_pairs_dt_hist_on_ax` (``nbins``,
        ``split_direction``, ``show_stats``, ``alpha``, ``dedupe``).
    :returns: ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    info = _draw_pairs_dt_hist_on_ax(ax, frame, **style)

    n_pairs, n_dates = info["n_pairs"], info["n_dates"]
    ax.set_title(
        fig_name
        or f"Temporal baseline distribution — {n_pairs} pair{'s' if n_pairs != 1 else ''} "
           f"({n_dates} acquisition dates)",
        fontsize=10,
    )
    if info["legend_handles"]:
        ax.legend(handles=info["legend_handles"], title=info["legend_title"],
                  loc="upper right", fontsize=8, title_fontsize=9)

    fig.tight_layout()
    return fig, ax


#: View key → publication-quality matplotlib builder.  Mirrors
#: :data:`geomulticorr.utils._pairs_plotly.VIEW_BUILDERS` key for key, so a
#: caller can render the same view through either backend.
PAIRS_MPL_BUILDERS: dict = {
    "baseline": plot_pairs_baseline,
    "chord": plot_pairs_chord,
    "network": plot_pairs_network,
    "dt_hist": plot_pairs_dt_hist,
}
