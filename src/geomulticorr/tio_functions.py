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
import pathlib
import subprocess

def tiff2bin(image_path: str | pathlib.Path, outbin_path: str | pathlib.Path) -> None:
    cmd = ["gdal_translate", "-of", "envi", "-ot", "Float32",
           "--config", "GDAL_PAM_ENABLED", "NO", f"{image_path}", f"{outbin_path}"]
    cmd_run = subprocess.run(' '.join(cmd),
                             shell = True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    # print(cmd_run.stderr)
    # print(cmd_run.stdout)
    del(cmd)
    return None
#END def

def bin2tiff(imagebin_path: str | pathlib.Path,
            out_path: str | pathlib.Path) -> None:
    hdr_tiff_fn = list(imagebin_path.parent.parent.glob(f"{imagebin_path.name.split('.')[0]}.hdr"))
    if hdr_tiff_fn not in bin_dir:
        image_dir = imagebin_path.parent.parent
        hdr_tiff_fn = imagebin_path.parent.parent.glob(f"{imagebin_path.name.split('.')[0]}.hdr")
        shutil.copy(hdr_tiff_fn, out_path.joinpath(hdr_tiff_fn.name))
    else:
        pass
    cmd = ["gdal_translate", "-if", "envi", "-of", "GTiff", f"{imagebin_path}", f"{out_path}"]
    cmd_run = subprocess.run(' '.join(cmd),
                             shell = True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    del(cmd)
    return

def make_TIOouts_proper(inverse_dir: str | pathlib.Path, list_images: list = None):
    if list_images == None:
        with open(inverse_dir.joinpath("liste_image")) as file:
            liste_images = [line.rstrip() for line in file]
        #END with
    else:
        liste_images == list_images
    #END if
    keep_files=["depl_cumule", "depl_cumule_liss"]
    inv_files = keep_files + liste_images

