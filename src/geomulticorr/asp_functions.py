#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------- #
# Author metadata
# name: Diego Cusicanqui
# email: diego.cusicanqui@univ-grenoble-alpes.fr
# affiliation: Centre National d'Etudes Spatiales (CNES)
#              Institut des Sciences de la Terre (ISTerre)
# --------------------------------------------------------- #
#%%
import os
import sys
import glob
import subprocess
import shutil
import pathlib
from distutils.spawn import find_executable
#%%
def run_asp_cmd(bin, args, **kw):
    bin_path = find_executable(bin)
    if bin_path is None:
        msg = (f"Unable to find executable {bin_path}\n" 
        f"Install ASP and ensure it is in your PATH env variable\n" 
        "https://stereopipeline.readthedocs.io/en/latest/installation.html" % bin)
        sys.exit(msg)
    call = [bin_path,]
    call.extend(args)
    # print(' '.join(call))
    # subprocess.run(' '.join(call), shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        result = subprocess.run(' '.join(call), shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as e:
        raise Exception('%s: %s' % (bin_path, e))
    if result.stderr != "":
        print(result.stderr)
#%%
def asp_corr_params(primary: str | pathlib.PurePath,
                    secondary: str | pathlib.PurePath,
                    out_dirname: str | pathlib.PurePath = "run",
                    prefix: str = "run_corr",
                    corr_algorithm: list[str] = ['asp_bm', 'asp_sgm', 'asp_mgm', 'asp_final_mgm'],
                    corr_kernel: tuple = (35, 35),
                    xcorr_th: int = 10,
                    sp_mode: int = 2,
                    nodata_value: str = None,
                    cpu_threads: int = 16,
                    img_align: str = None,
                    individual_normalization: bool = False,
                    ) -> str:

    imgcorr_parameters = []
    # Set feature tracking image correlation function
    imgcorr_parameters.extend(['--correlator-mode'])
    # Set parameters based on correlation algorithm 
    if corr_algorithm == 'asp_bm':
        # Set correlation algorithm
        imgcorr_parameters.extend(['--stereo-algorithm', f"{corr_algorithm}"])
        imgcorr_parameters.extend(['--corr-kernel', f"{corr_kernel[0]}", f"{corr_kernel[1]}"])
        imgcorr_parameters.extend(['--subpixel-kernel', f"{corr_kernel[0]}", f"{corr_kernel[1]}"])
        imgcorr_parameters.extend(['--subpixel-mode', f"{sp_mode}"])
    if corr_algorithm == 'asp_sgm' or corr_algorithm == 'asp_mgm' or corr_algorithm == 'asp_final_mgm':
        print(f"With 'asp_sgm' or 'asp_mgm' or 'asp_final_mgm', '--corr-kernel' values can not be grater than 9 9. Forcing to '--corr-kernel' = 9 9 by default")
        corr_kernel = (9, 9)
        imgcorr_parameters.extend(['--stereo-algorithm', f"{corr_algorithm}"])
        imgcorr_parameters.extend(['--corr-kernel', f"{corr_kernel[0]}", f"{corr_kernel[1]}"])
        imgcorr_parameters.extend(['--subpixel-kernel', f"{corr_kernel[0]+10}", f"{corr_kernel[1]+10}"])
        imgcorr_parameters.extend(['--subpixel-mode', f"{sp_mode}"])
    #END ifimg_align
    imgcorr_parameters.extend(['--xcorr-threshold', f"{xcorr_th}"])
    # Set number of threads/cores to use for correlation (will use all CPUs if possible).
    imgcorr_parameters.extend(['--threads-multiprocess', f"{cpu_threads}"])
    imgcorr_parameters.extend(['--processes', '1']) 
    #This assumes input images has already a good georeference.
    #TODO: can image_align come into this parameter? This should be explored further.
    imgcorr_parameters.extend(['--alignment-method', f"{img_align}"])
    # individual normalization is only useful with bigger time interval between acquisitions.
    if individual_normalization:
        # Normalize image histograms individually. Very useful when time interval between images is important.
        imgcorr_parameters.extend('--individually-normalize')
    #END if
    imgcorr_parameters.extend(['--ip-per-tile', '200'])
    imgcorr_parameters.extend(['--min-num-ip', '5'])
    if nodata_value is not None:
        imgcorr_parameters.extend(['--nodata-value', f"{nodata_value}"])
    #END if
    # Finally, add images and prefix to the command line
    imgcorr_parameters.extend([primary, secondary, f"{out_dirname}/{prefix}"])
    return imgcorr_parameters
#END def
#%%
def asp_img_align_params(primary: str | pathlib.PurePath,
                    secondary: str | pathlib.PurePath,
                    prefix: str,
                    robust_align: bool) -> str:
    """
    Run ASP 'image_align' function to align the secondary image to the reference (primary) image.

    Args:
        primary (str): Path to the reference image.
        secondary (str): Path to the secondary image.
        prefix (str): Output file prefix (default: None)
        robust_align (bool | False): If option selected, ASP perform classic image correlation to estimate surface displacements.
    """
    
    # cmd f""
    # folder = primary.replace(primary.split("/")[-1], "")
    # print(f"Data will be saved under {folder}image_align/")
    # if prefix: 
    #     cmd = f"{os.path.join('image_align')} {primary} {secondary} -o {secondary[:-4]}_aligned.tif --output-prefix {folder}image_align/{prefix} --alignment-transform affine --disparity-params '{folder}disparity_maps/{prefix}-F.tif 10000' --inlier-threshold 100" 
    # else:
    #     cmd = f"{os.path.join('image_align')} {primary} {secondary} -o {secondary[:-4]}_aligned.tif --output-prefix {folder}image_align/{prefix} --alignment-transform affine  --inlier-threshold 100" 
    # #END if
    # if robust_align:

    # os.system(cmd)
#END def
#%%
def asp_ortho_params(dem_fn: str | pathlib.PurePath,
                    image_fn: str | pathlib.PurePath,
                    rpc_fn: str | pathlib.PurePath = None,
                    epsg_code: str = None,
                    resolution: str = None,
                    out_fn: str | pathlib.PurePath = None) -> str:
    # Check if image contain RPC file

    # Check image parameters

    # Handel projection codes
    proj = "+proj=lcc +lat_1=49 +lat_2=44 +lat_0=46.5 +lon_0=3 +x_0=700000 +y_0=6600000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs" #ESPG: 2154
    proj = "+proj=utm +zone=32 +north +units=m +datum=WGS84" #EPSG: UTMZ32N

    # Get parameters
    ortho_params = []
    ortho_params.extend(['-t', 'rpc'])
    ortho_params.extend(['--t_srs', proj])
    ortho_params.extend(['--threads', '8'])
    ortho_params.extend(['--mpp', '3.0'])
    if rpc_fn is not None:
        ortho_params.extend([f'{dem_fn}', f'{image_fn}', f'{rpc_fn}'])
    else:
        ortho_params.extend([f'{dem_fn}', f'{image_fn}'])
    #END if
    if out_fn is None:
        out_fn = f"{image_fn.name.split('.')[0]}_mapproj_{image_fn.name.split('.')[1]}"
        ortho_params.extend([f"{image_fn.parent.joinpath(out_fn)}"])
    #END if
#END def

def sensors(sensors_names=['spot5', 'spot6', 'spot7', 'aerial', 'planet', 'planetscope', 'sentinel2', 'landsat5', 'landsat7']):
    """
    List of sensors accepted on GeoMultiCorr.

    Args:
        sensors_names (list, optional): List of sensors accepted on GeoMultiCorr.
        Defaults to ['spot5', 'spot6', 'spot7', 'aerial', 'planet', 'planetscope', 'sentinel2', 'landsat5', 'landsat7'].

    Returns:
        str: Concatenated string with all sensors using lowercase and UPPERCASE.
    """
