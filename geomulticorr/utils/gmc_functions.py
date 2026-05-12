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
import matplotlib.pyplot as plt
import rasterstats

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
    bins: int = 10,
    symmetric_xlim: bool = True,
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

    xs = np.linspace(0.02, 0.68, max(len(clean), 1))
    for i, (key, stats) in enumerate(zip(array_keys, stats_list)):
        lines = [key] + [f"{m}: {stats[m]:.1f}" for m in em_valid]
        ax.text(xs[i], 0.98, "\n".join(lines), transform=ax.transAxes,
                va='top', ha='right', fontsize=9, color=color_cycle[i],
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
        symmetric_xlim=True,
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
    left, right, xDisp, yDisp, GoodPixMap, ncc,
    figsize: tuple[float, ...] = None,
    cmap_ew: str = 'PuOr',
    cmap_ns: str = 'PiYG',
    cmap_cc: str = 'binary_r',
    fig_name=None,
) -> plt.Figure:
    if figsize is None:
        figsize = (10, 8)
    fig, axs = plt.subplot_mosaic(
        [['a)', 'b)', 'c)'],
         ['d)', 'e)', 'f)'],
         ['g)', 'h)', 'i)']],
        layout='tight', figsize=figsize, num=1, clear=True)
    fig.suptitle(fig_name)
    for label, ax in axs.items():
        ax.annotate(label, xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
    left.plot(ax=axs['a)'], cmap='gray', title="Left image")
    right.plot(ax=axs['b)'], cmap='gray', title="Right image")
    GoodPixMap.plot(ax=axs['c)'], cmap='viridis', title="GoodPixelMap", vmin=0, vmax=1)
    vlim_x = get_rounded_limits(xDisp, round_to=10, symmetric=True)
    xDisp.plot(ax=axs['d)'], cmap=cmap_ew, title="Raw EW-Disp", vmin=-vlim_x, vmax=vlim_x)
    vlim_y = get_rounded_limits(yDisp, round_to=10, symmetric=True)
    yDisp.plot(ax=axs['e)'], cmap=cmap_ns, title="Raw NS-Disp", vmin=-vlim_y, vmax=vlim_y)
    if ncc is not None:
        ncc.plot(ax=axs['f)'], cmap=cmap_cc, title="NCC")
    else:
        axs['f)'].set_title("NCC (not available)")
        axs['f)'].axis('off')
    _draw_histogram_on_ax(axs['g)'], [xDisp.data], ["EW-Disp"], symmetric_xlim=True)
    axs['g)'].set_title("EW-Disp histogram")
    axs['g)'].set_xlabel("Displacement (m)")
    axs['g)'].set_ylabel("Count")
    _draw_histogram_on_ax(axs['h)'], [yDisp.data], ["NS-Disp"], symmetric_xlim=True)
    axs['h)'].set_title("NS-Disp histogram")
    axs['h)'].set_xlabel("Displacement (m)")
    axs['h)'].set_ylabel("Count")
    _draw_histogram_on_ax(axs['i)'], [left.data, right.data], ["Left", "Right"],
                          error_metrics=['mean', 'std'], symmetric_xlim=False)
    axs['i)'].set_title("Image normalized histogram")
    axs['i)'].set_xlabel("Pixel value")
    axs['i)'].set_ylabel("Count")
    return fig


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