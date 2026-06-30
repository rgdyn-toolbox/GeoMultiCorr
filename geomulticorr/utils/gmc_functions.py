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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress

from geomulticorr.stats import nmad
from geomulticorr.corrections.corrections import (
    RampCorrection,
    TopoCorrection,
    DirectionalBiasCorrection,
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
    axs['c)'].legend()

    yDisp_before.plot(ax=axs['d)'], cmap=cmap_ns, vmin=-vlim_y, vmax=vlim_y, title="NS before")
    yDisp_after.plot( ax=axs['e)'], cmap=cmap_ns, vmin=-vlim_y, vmax=vlim_y, title="NS after")
    _draw_histogram_on_ax(axs['f)'], [yDisp_before.data, yDisp_after.data],
                          ["NS before", "NS after"], symmetric_xlim=True, cmap=cmap_ns)
    axs['f)'].set_title("NS histogram")
    axs['f)'].set_xlabel("Displacement (m)")
    axs['f)'].set_ylabel("Count")
    axs['f)'].legend()

    fig.tight_layout()
    return fig

def plot_directional_bias_correction(
    step,
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    fig_name: str | None = None,
    figsize: tuple[float, float] = (16, 11),
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_bias: str = 'RdBu_r',
) -> plt.Figure:
    """Control figure for a DirectionalBiasCorrection step.

    The removed bias is the surface that was subtracted (before − after).  A
    full-width bottom panel shows the fitted 1-D bias profile stored in
    ``step.meta`` (``profile_centers`` / ``profile_median`` / ``profile_fitted``),
    annotated with the seam angle and the before/after residual std.

    Layout
    ------
    a) EW before  |  b) EW bias  |  c) EW after
    d) NS before  |  e) NS bias  |  f) NS after
    g) 1-D bias profile along the seam-normal axis (spans all columns)
    """
    import numpy.ma as ma

    fig, axs = plt.subplot_mosaic(
        [['a)', 'b)', 'c)'],
         ['d)', 'e)', 'f)'],
         ['g)', 'g)', 'g)']],
        figsize=figsize,
    )
    angle = step.meta.get("angle_deg") if getattr(step, "meta", None) else None
    title = "Directional bias correction"
    if angle is not None:
        title += f" — angle = {angle:.2f}°"
    if fig_name:
        title = f"{fig_name} — {title}"
    fig.suptitle(title)

    # Removed bias = what was subtracted (before − after)
    x_bias_data = ma.array(
        xDisp_before.data.data - xDisp_after.data.data,
        mask=np.ma.getmaskarray(xDisp_before.data),
    )
    y_bias_data = ma.array(
        yDisp_before.data.data - yDisp_after.data.data,
        mask=np.ma.getmaskarray(yDisp_before.data),
    )
    x_bias = xDisp_before.copy(new_array=x_bias_data)
    y_bias = yDisp_before.copy(new_array=y_bias_data)

    vlim_x  = get_rounded_limits(xDisp_before, round_to=1, symmetric=True)
    vlim_y  = get_rounded_limits(yDisp_before, round_to=1, symmetric=True)
    vlim_bx = get_rounded_limits(x_bias,       round_to=1, symmetric=True)
    vlim_by = get_rounded_limits(y_bias,       round_to=1, symmetric=True)

    xDisp_before.plot(ax=axs['a)'], cmap=cmap_ew,   vmin=-vlim_x,  vmax=vlim_x,  title="EW before")
    x_bias.plot(      ax=axs['b)'], cmap=cmap_bias,  vmin=-vlim_bx, vmax=vlim_bx, title="EW bias")
    xDisp_after.plot( ax=axs['c)'], cmap=cmap_ew,   vmin=-vlim_x,  vmax=vlim_x,  title="EW after")

    yDisp_before.plot(ax=axs['d)'], cmap=cmap_ns,   vmin=-vlim_y,  vmax=vlim_y,  title="NS before")
    y_bias.plot(      ax=axs['e)'], cmap=cmap_bias,  vmin=-vlim_by, vmax=vlim_by, title="NS bias")
    yDisp_after.plot( ax=axs['f)'], cmap=cmap_ns,   vmin=-vlim_y,  vmax=vlim_y,  title="NS after")

    # 1-D bias profile from the fitted step metadata
    meta = getattr(step, "meta", {}) or {}
    centers = meta.get("profile_centers")
    median  = meta.get("profile_median")
    fitted  = meta.get("profile_fitted")
    if centers is not None and median is not None:
        axs['g)'].plot(centers, median, ".", ms=4, alpha=0.5, label="binned median (stable)")
        if fitted is not None:
            axs['g)'].plot(centers, fitted, "-", lw=2, label="fitted profile")
        axs['g)'].set_xlabel("Seam-normal coordinate s (m)")
        axs['g)'].set_ylabel("Bias (m)")
        sub = []
        if meta.get("std_before") is not None and meta.get("std_after") is not None:
            sub.append(f"std {meta['std_before']:.3f} → {meta['std_after']:.3f} m")
        axs['g)'].set_title("1-D bias profile" + (f"  ({'; '.join(sub)})" if sub else ""))
        axs['g)'].legend()
        axs['g)'].grid(alpha=0.3)
    else:
        axs['g)'].axis("off")
        axs['g)'].text(0.5, 0.5, "No profile metadata available",
                       ha="center", va="center", transform=axs['g)'].transAxes)

    fig.tight_layout()
    return fig
#END def

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
    import numpy.ma as ma

    fig, axs = plt.subplot_mosaic(
        [['a)', 'b)', 'c)', 'd)'],
         ['e)', 'f)', 'g)', 'h)']],
        figsize=figsize,
    )
    if fig_name:
        fig.suptitle(f"{fig_name} — Ramp correction")

    # Ramp = what was subtracted
    x_ramp_data = ma.array(
        xDisp_before.data.data - xDisp_after.data.data,
        mask=np.ma.getmaskarray(xDisp_before.data),
    )
    y_ramp_data = ma.array(
        yDisp_before.data.data - yDisp_after.data.data,
        mask=np.ma.getmaskarray(yDisp_before.data),
    )
    x_ramp = xDisp_before.copy(new_array=x_ramp_data)
    y_ramp = yDisp_before.copy(new_array=y_ramp_data)

    vlim_x    = get_rounded_limits(xDisp_before, round_to=1, symmetric=True)
    vlim_y    = get_rounded_limits(yDisp_before, round_to=1, symmetric=True)
    vlim_rx   = get_rounded_limits(x_ramp,       round_to=1, symmetric=True)
    vlim_ry   = get_rounded_limits(y_ramp,       round_to=1, symmetric=True)

    xDisp_before.plot(ax=axs['a)'], cmap=cmap_ew,   vmin=-vlim_x,  vmax=vlim_x,  title="EW before")
    x_ramp.plot(      ax=axs['b)'], cmap=cmap_ramp,  vmin=-vlim_rx, vmax=vlim_rx, title="EW ramp")
    xDisp_after.plot( ax=axs['c)'], cmap=cmap_ew,   vmin=-vlim_x,  vmax=vlim_x,  title="EW after")
    _draw_histogram_on_ax(axs['d)'], [xDisp_before.data, xDisp_after.data],
                          ["EW before", "EW after"], symmetric_xlim=True, cmap=cmap_ew)
    axs['d)'].set_title("EW histogram")
    axs['d)'].set_xlabel("Displacement (m)")
    axs['d)'].set_ylabel("Count")
    axs['d)'].legend()

    yDisp_before.plot(ax=axs['e)'], cmap=cmap_ns,   vmin=-vlim_y,  vmax=vlim_y,  title="NS before")
    y_ramp.plot(      ax=axs['f)'], cmap=cmap_ramp,  vmin=-vlim_ry, vmax=vlim_ry, title="NS ramp")
    yDisp_after.plot( ax=axs['g)'], cmap=cmap_ns,   vmin=-vlim_y,  vmax=vlim_y,  title="NS after")
    _draw_histogram_on_ax(axs['h)'], [yDisp_before.data, yDisp_after.data],
                          ["NS before", "NS after"], symmetric_xlim=True, cmap=cmap_ns)
    axs['h)'].set_title("NS histogram")
    axs['h)'].set_xlabel("Displacement (m)")
    axs['h)'].set_ylabel("Count")
    axs['h)'].legend()

    fig.tight_layout()
    return fig


def plot_correction_result(
    step,
    xDisp_before: gu.Raster,
    xDisp_after: gu.Raster,
    yDisp_before: gu.Raster,
    yDisp_after: gu.Raster,
    fig_name: str | None = None,
    **_,
) -> plt.Figure:
    """Dispatch to the appropriate plot for a single correction step.

    - ``RampCorrection`` / ``TopoCorrection`` → :func:`plot_ramp_correction`
    - ``DirectionalBiasCorrection`` → :func:`plot_directional_bias_correction`
    - All other steps → :func:`plot_median_centering`
    """
    if isinstance(step, DirectionalBiasCorrection):
        return plot_directional_bias_correction(
            step, xDisp_before, xDisp_after, yDisp_before, yDisp_after, fig_name=fig_name
        )
    if isinstance(step, (RampCorrection, TopoCorrection)):
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
    bin_centers, medians, nmads = [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (elev_vals >= lo) & (elev_vals < hi)
        if mask.sum() < min_pixels:
            continue
        v = disp_vals[mask]
        bin_centers.append((lo + hi) / 2)
        medians.append(float(np.median(v)))
        nmads.append(nmad(v))

    bin_centers = np.asarray(bin_centers)
    medians     = np.asarray(medians)
    nmads       = np.asarray(nmads)

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