#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------#
# @Author metadata
# name: Diego Cusicanqui
# email: diego.cusicanqui@univ-grenoble-alpes.fr
# affiliation: Centre National d'Etudes Spatiales (CNES)
#              Institut des Sciences de la Terre (ISTerre)
#------------------------------------------------------------------------------#
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