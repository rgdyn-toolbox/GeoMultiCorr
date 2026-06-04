#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# pair.py
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
import shutil
from pathlib import Path
from rasterio.features import shapes
import geopandas as gpd
import rasterio

# import cv2 as cv
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn import cluster

import geoutils as gu
import matplotlib.pyplot as plt

# from telenvi import raster_tools as rt

import geomulticorr.core.thumb as gmc_thumb
from geomulticorr.correlation.correlation import ASP
from geomulticorr._logging import logger

# ASP cannot write his outputs everywhere
# In the temp (for temporary) directory, we know it will be ok
# Then, we move the data (def save_corr_data)
ROOT_OUTPUTS = Path(__file__).parent.parent / "temp"
if not ROOT_OUTPUTS.exists():
    ROOT_OUTPUTS.mkdir()
# END if


class Pair:
    """
    Represents an image pair for optical correlation analysis.
    
    A Pair is created from two Thumb objects (optical images) from the same processing zone.
    It manages correlation workflows, displacement computation, and result storage.
    
    Pairs can be constructed in two ways:
    
    1. From two Thumb objects: ``pair = thumb_a + thumb_b``
    2. From a session and pair directory path (for loading existing pairs)
    
    The pair orchestrates the full workflow: clipping, ASP correlation,
    displacement extraction, magnitude computation, and vectorization.
    """

    def __init__(self, session=None, target_path=None, left=None, right=None):
        """
        Initialize a Pair object from either two Thumbs or a pair directory path.
        
        **Construction mode 1: From two Thumbs**
        
        Provide ``left`` and ``right`` Thumb objects. They must belong to the same
        processing zone. This is the typical mode when creating a new pair.
        
        **Construction mode 2: From existing pair directory**
        
        Provide ``session`` and ``target_path``. The path must point to a directory
        under ``image_correlation/`` with the naming pattern:
        ``<pzone>_<date1>-<sensor1>_<date2>-<sensor2>``
        
        Thumbs are reconstructed from the session's raster archive.
        
        :param session: GMC session object (required if loading from path).
        :type session: Session or None
        
        :param target_path: Path to an existing pair directory.
        :type target_path: str, Path, or None
        
        :param left: First Thumb (required if creating from thumbs).
        :type left: Thumb or None
        
        :param right: Second Thumb (required if creating from thumbs).
        :type right: Thumb or None
        
        :raises AssertionError: If left and right are from different pzones, or if they are the same thumb.
        
        :ivar pa_pz_name: Processing zone name.
        :ivar pa_key: Unique pair identifier.
        :ivar pa_left: Left (earlier) Thumb.
        :ivar pa_right: Right (later) Thumb.
        :ivar pa_dt_days: Temporal baseline in days.
        :ivar pa_dt_months: Temporal baseline in months.
        :ivar pa_dt_years: Temporal baseline in years (decimal).
        :ivar pa_direction: "forward" or "backward" based on date order.
        :ivar pa_path: Directory path for pair outputs.
        :ivar pa_status: Correlation status ("empty", "clipped", "complete", "corrupt").
        """
        self.session = session
        # Construction from a pair path
        if type(target_path) in (str, Path):
            target_path = Path(target_path)
            assert (
                Path(target_path).parent.exists()
                and Path(target_path).parent.name.lower() == "image_correlation"
            ), "this path is not leading to a well-formed pair"

            # Get metadata about the thumbs component of the pair
            left_year, left_month, left_day, left_sensor = target_path.name.split("_")[
                1
            ].split("-")
            right_year, right_month, right_day, right_sensor = target_path.name.split(
                "_"
            )[2].split("-")
            left_date = f"{left_year}-{left_month}-{left_day}"
            right_date = f"{right_year}-{right_month}-{right_day}"
            pz_name = target_path.name.split("_")[0]
            left_key = f"{pz_name}_{left_date}_{left_sensor}.tif"
            right_key = f"{pz_name}_{right_date}_{right_sensor}.tif"

            # Reconstruct their path
            thumbs_pzone_path = Path(session.path_raster_data, pz_name, "opticals")
            left_path = Path(thumbs_pzone_path, left_key)
            right_path = Path(thumbs_pzone_path, right_key)

            # Make Thumbs from them
            left = gmc_thumb.Thumb(left_path)
            right = gmc_thumb.Thumb(right_path)

        # Construction from two thumbs
        assert (
            left.th_pz_name == right.th_pz_name
        ), "left and right have not the same pzone"
        assert left.th_path != right.th_path, "left thumb is right thumb"

        # Metadata
        self.pa_pz_name = left.th_pz_name
        self.pa_key = f"{self.pa_pz_name}_{left.th_date}-{left.th_sensor}_{right.th_date}-{right.th_sensor}"
        self.pa_left = left
        self.pa_right = right

        # Temporal metadata
        _dt = right.th_date_datetime - left.th_date_datetime
        self.pa_dt_days = abs(_dt.days)
        self.pa_dt_months = round(self.pa_dt_days / 30.4375, 2)
        self.pa_dt_years = round(self.pa_dt_days / 365.25, 4)
        self.pa_direction = (
            "forward" if left.th_date_datetime <= right.th_date_datetime else "backward"
        )

        # Base path
        self.pa_path = Path(
            Path(left.th_path).parent.parent, "image_correlation", f"{self.pa_key}"
        ).absolute()

        # Symlink paths — point to the sieved thumb images (no clipping, no copy)
        self.pa_left_link_path = Path(self.pa_path, f"{self.pa_left.th_key}.tif")
        self.pa_right_link_path = Path(self.pa_path, f"{self.pa_right.th_key}.tif")

        # ASP correlation outputs (written directly into pa_path)
        self.pa_raw_left_path = Path(self.pa_path, f"{self.pa_key}-L.tif")
        self.pa_raw_right_path = Path(self.pa_path, f"{self.pa_key}-R.tif")
        self.pa_disparity_rd_path = Path(self.pa_path, f"{self.pa_key}-RD.tif")
        self.pa_disparity_f_path = Path(self.pa_path, f"{self.pa_key}-F.tif")
        self.pa_cc_raw_path = Path(self.pa_path, f"{self.pa_key}-ncc.tif")

        # Derived outputs
        self.pa_ew_path = Path(self.pa_path, f"{self.pa_key}-F_EW.tif")
        self.pa_ns_path = Path(self.pa_path, f"{self.pa_key}-F_NS.tif")
        self.pa_cc_path = Path(self.pa_path, f"{self.pa_key}-F_CC.tif")
        self.pa_plot_raw_path = Path(self.pa_path, f"{self.pa_key}_raw_disp.jpg")
        self.pa_magn_path = Path(self.pa_path, f"{self.pa_key}_magn.tif")
        self.pa_vect_path = Path(self.pa_path, f"{self.pa_key}_vect.gpkg")
        self.pa_stats_path = Path(self.pa_path, f"{self.pa_key}.json")

        # Current state of the pair
        self.pa_status = self.get_status()

        # Pair spatial extent
        self.geometry = left.geometry

    def __repr__(self):
        """
        Return a formatted string representation of the Pair.
        
        Displays key metadata: processing zone, left/right dates and sensors,
        temporal direction, temporal baseline, and correlation status.
        
        :return: Formatted string representation.
        :rtype: str
        """
        return f"""---------
type      : GMC_Pair
pzone     : {self.pa_pz_name}
left      : {self.pa_left.th_date}-{self.pa_left.th_sensor}
right     : {self.pa_right.th_date}-{self.pa_right.th_sensor}
direction : {self.pa_direction}
dt        : {self.pa_dt_days} days / {self.pa_dt_months} months / {self.pa_dt_years} years
status    : {self.pa_status}
---------
"""

    def get_status(self):
        """
        Determine the correlation processing status of this pair.
        
        Status levels:
        
        - **empty**: Pair directory does not exist yet.
        - **clipped**: Symlinks to input thumbs created; awaiting correlation.
        - **complete**: ASP correlation finished; all outputs present.
        - **corrupt**: Pair directory exists but outputs are incomplete or malformed.
        
        :return: Current pair status.
        :rtype: str ("empty", "clipped", "complete", or "corrupt")
        """
        if self.pa_path.exists():
            if self.pa_disparity_f_path.exists():
                pa_status = "complete"
            elif self.pa_left_link_path.exists() and self.pa_right_link_path.exists():
                pa_status = "clipped"
            else:
                pa_status = "corrupt"
        else:
            pa_status = "empty"
        return pa_status

    def check_corr_outputs(self) -> dict[str, bool]:
        """
        Check which ASP correlation output files exist for this pair.
        
        Verifies the presence of all expected ASP output suffixes defined in the
        ASP class (e.g., ``-F.tif``, ``-ncc.tif``, ``-RD.tif``).
        
        :return: Dictionary mapping each expected suffix to True (file exists) or False (missing).
        :rtype: dict[str, bool]
        """
        return {
            suffix: (self.pa_path / f"{self.pa_key}{suffix}").exists()
            for suffix in ASP._KEEP_SUFFIXES
        }

    def to_pdserie(self):
        """
        Convert Pair metadata to a pandas Series.
        
        Serializes all pair attributes and file paths into a pandas Series for
        integration with GeoDataFrames and tabular workflows.
        
        :return: Series containing pair metadata, dates, sensors, paths, and geometry.
        :rtype: pd.Series
        """
        return pd.Series(
            {
                "pa_pz_name": self.pa_pz_name,
                "pa_path": str(self.pa_path),
                "pa_left_date": self.pa_left.th_date,
                "pa_left_sensor": self.pa_left.th_sensor,
                "pa_right_date": self.pa_right.th_date,
                "pa_right_sensor": self.pa_right.th_sensor,
                "pa_dt_days": self.pa_dt_days,
                "pa_dt_months": self.pa_dt_months,
                "pa_dt_years": self.pa_dt_years,
                "pa_direction": self.pa_direction,
                "pa_magn_path": str(self.pa_magn_path),
                "pa_disparity_rd_path": str(self.pa_disparity_rd_path),
                "pa_disparity_f_path": str(self.pa_disparity_f_path),
                "pa_status": self.pa_status,
                "pa_ew_path": str(self.pa_ew_path),
                "pa_ns_path": str(self.pa_ns_path),
                "pa_cc_path": str(self.pa_cc_path),
                "geometry": self.geometry,
            }
        )

    def get_magn_raster(self):
        """
        Load and cache the displacement magnitude raster.
        
        Reads the magnitude raster file (``pa_magn_path``) from disk on first call,
        then caches it as ``pa_magn`` for subsequent calls.
        
        :return: Magnitude raster in metres.
        :rtype: geoutils.Raster
        """
        try:
            return self.pa_magn
        except AttributeError:
            self.pa_magn = gu.Raster(str(self.pa_magn_path), load_pixels=True)
            return self.pa_magn

    # END def

    def get_cc_raster(self):
        """
        Load and cache the correlation coefficient raster.
        
        Reads the raw NCC (normalized cross-correlation) raster from disk on first call,
        then caches it as ``pa_cc`` for subsequent calls.
        
        :return: Correlation coefficient raster (range 0–1).
        :rtype: geoutils.Raster
        """
        try:
            return self.pa_cc
        except AttributeError:
            self.pa_cc = gu.Raster(str(self.pa_cc_raw_path), load_pixels=True)
            return self.pa_cc

    def get_disp_corr_raster(self):
        """
        Load the full 3-band disparity raster (xDisp, yDisp, GoodPixMap).
        
        Reads the final disparity output from ASP (``pa_disparity_f_path``)
        containing horizontal displacement, vertical displacement, and quality map.
        
        :return: 3-band raster with all disparity components.
        :rtype: geoutils.Raster
        """
        try:
            return self.pa_disparity_f
        except AttributeError:
            self.pa_disparity_f = gu.Raster(
                str(self.pa_disparity_f_path), load_pixels=True, nBands=3
            )
            return self.pa_disparity_f

    def get_xDisp_raster(self):
        """
        Load and cache the horizontal (EW) displacement raster in pixels.
        
        Extracts band 1 from the final disparity raster. Values are in pixel coordinates
        before unit conversion (see ``pa_ew_path`` for metre-converted version).
        
        :return: Horizontal displacement raster (pixels).
        :rtype: geoutils.Raster
        """
        try:
            return self.pa_xDisp
        except AttributeError:
            self.pa_xDisp = gu.Raster(
                str(self.pa_disparity_f_path), load_pixels=True, nBands=1
            )
            # Get Metadata
            pxSizeX, _ = self.pa_xDisp.res
            return self.pa_xDisp * pxSizeX

    def get_yDisp_raster(self):
        """
        Load and cache the vertical (NS) displacement raster in pixels.
        
        Extracts band 2 from the final disparity raster. Values are in pixel coordinates
        before unit conversion (see ``pa_ns_path`` for metre-converted version).
        
        :return: Vertical displacement raster (pixels).
        :rtype: geoutils.Raster
        """
        try:
            return self.pa_yDisp
        except AttributeError:
            self.pa_yDisp = gu.Raster(
                str(self.pa_disparity_f_path), load_pixels=True, nBands=2
            )
            # Get Metadata
            _, pxSizeY = self.pa_xDisp.res
            return self.pa_yDisp * (pxSizeY * -1.0)  # Flip NS sign (ASP uses upper-left corner convention)

    def get_xVel(self):
        """
        Compute annual horizontal (EW) velocity.
        
        Divides horizontal displacement by the temporal baseline (``pa_dt_years``).
        
        :return: Horizontal velocity raster in pixels/year.
        :rtype: geoutils.Raster
        """
        return self.get_xDisp_raster() / self.pa_dt_years

    def get_yVel(self):
        """
        Compute annual vertical (NS) velocity.
        
        Divides vertical displacement by the temporal baseline (``pa_dt_years``).
        
        :return: Vertical velocity raster in pixels/year.
        :rtype: geoutils.Raster
        """
        return self.get_yDisp_raster() / self.pa_dt_years

    def get_vMagn_raster(self):
        """
        Compute annual velocity magnitude.
        
        Divides the displacement magnitude by the temporal baseline (``pa_dt_years``).
        
        :return: Velocity magnitude raster in metres/year.
        :rtype: geoutils.Raster
        """
        return self.compute_magnitude() / self.pa_dt_years

    ### Creation of the displacement fields

    def clip(self, numBand=1):
        """
        Prepare the pair for correlation by creating symlinks to input thumbs.
        
        Creates the pair directory and symbolic links to the left and right Thumb files.
        This is the first step before launching ASP correlation. If the pair already
        has a "clipped" or "complete" status, returns early.
        
        :param numBand: Unused parameter (for API compatibility).
        :type numBand: int
        
        :return: True if clipping succeeded.
        :rtype: bool
        """
        if self.pa_status in ["complete", "clipped"]:
            return True

        self.pa_path.mkdir(parents=True, exist_ok=True)

        if not self.pa_left_link_path.exists():
            self.pa_left_link_path.symlink_to(Path(self.pa_left.th_path).resolve())
        if not self.pa_right_link_path.exists():
            self.pa_right_link_path.symlink_to(Path(self.pa_right.th_path).resolve())

        self.pa_status = "clipped"
        return True

    # def define_parameters(self):

    # ! This function is useless because the new implementation on correlation.py replace Should be removed
    def launch_correlation(
        self,
        corr_algorithm="asp_mgm",
        corr_kernel_size=7,
        corr_xthreshold=10,
        corr_tile_size=2048,
        dry_run=False,
    ):
        """
        Launch ASP correlation via the ASP correlation handler.
        
        .. warning::
            This method is deprecated. Use the new correlation pipeline in
            ``geomulticorr.correlation.correlation`` instead.
        
        Triggers the full ASP correlation workflow if the pair is not already complete.
        Passes parameters to the ASP handler for execution.
        
        :param corr_algorithm: ASP correlation algorithm (default "asp_mgm").
        :type corr_algorithm: str
        
        :param corr_kernel_size: Correlation kernel size in pixels.
        :type corr_kernel_size: int
        
        :param corr_xthreshold: Cross-correlation threshold for outlier rejection.
        :type corr_xthreshold: float
        
        :param corr_tile_size: Tile size for processing.
        :type corr_tile_size: int
        
        :param dry_run: If True, prepare but do not execute correlation.
        :type dry_run: bool
        
        :return: True if correlation succeeded or was already complete.
        :rtype: bool
        """

        # Check if pair already exists
        if self.get_status() == "complete":
            return True
        asp = ASP(session=self.session)

        params_file_path = self.pa_path / "corr_params.txt"

        return asp.run_correlation(
            left=self.pa_left.th_path,
            right=self.pa_right.th_path,
            params_file_path=params_file_path,
        )

    # ! This function is useless because the new implementation on correlation.py replace Should be removed
    def corr(
        self,
        corr_algorithm=2,
        corr_kernel_size=7,
        corr_xthreshold=10,
        corr_tile_size=2048,
    ):
        """
        Execute ASP correlation directly (legacy method).
        
        .. warning::
            This method is deprecated and uses outdated parameters.
            Use the new correlation pipeline instead.
        
        Constructs and executes an ``parallel_stereo`` command with the specified
        parameters. Outputs are written directly to the pair directory.
        
        :param corr_algorithm: Algorithm code for stereo (default 2 = semi-global).
        :type corr_algorithm: int
        
        :param corr_kernel_size: Correlation window size.
        :type corr_kernel_size: int
        
        :param corr_xthreshold: Cross-correlation threshold.
        :type corr_xthreshold: float
        
        :param corr_tile_size: Tile size for processing.
        :type corr_tile_size: int
        
        :return: True if execution completed (regardless of success).
        :rtype: bool
        """

        # Check clip
        # self.clip()

        # Check existing correlation
        if self.get_status() == "complete":
            return True

        # Create a directory in a place where ASP can write their outputs : the temp directory, inside GeoMultiCorr app
        temp = Path(ROOT_OUTPUTS, self.pa_key)
        if not temp.exists():
            temp.mkdir()

        # Write correlation command
        # Diego remove run, and replace temp by self.pa_path
        # corr_command = f'parallel_stereo {self.pa_left.th_path} {self.pa_right.th_path} {temp}/{self.pa_key} \
        corr_command = f"parallel_stereo {self.pa_left.th_path} {self.pa_right.th_path} {self.pa_path}/{self.pa_key} \
            --correlator-mode \
            --threads-multiprocess 8 \
            --processes 1 \
            --stereo-algorithm {corr_algorithm} \
            --corr-kernel {corr_kernel_size} {corr_kernel_size} \
            --xcorr-threshold {corr_xthreshold} \
            --corr-tile-size {corr_tile_size} \
            --corr-memory-limit-mb 8000 \
            --save-left-right-disparity-difference\
            --ip-per-tile 200 \
            --min-num-ip 2"

        # Launch
        os.system(corr_command)
        # self.pa_status='complete'
        return True

    # ! This function is useless because the new implementation on correlation.py replace Should be removed
    def corr_eval(self, corr_kernel_size=7, metric="ncc"):
        """
        Compute correlation quality metrics using ASP's ``corr_eval`` tool (legacy).
        
        .. warning::
            This method is deprecated. Use the modern corrections pipeline for
            quality assessment.
        
        Calls the ASP ``corr_eval`` command to generate a normalized cross-correlation
        (NCC) quality raster. Renames the output to follow GMC naming conventions and
        applies a default QGIS style.
        
        :param corr_kernel_size: Kernel size for correlation evaluation.
        :type corr_kernel_size: int
        
        :param metric: Quality metric ("ncc" by default).
        :type metric: str
        
        :return: True if evaluation succeeded.
        :rtype: bool
        """
        """
        Appel de la fonction d'ASP pour générer un raster SNR
        """

        # Check if the snr is already existing for this pair
        if self.pa_cc_raw_path.exists():
            return self.get_snr_geoim()

        # Get Left and Right normalized thumbs
        left_p = self.pa_raw_left_path
        right_p = self.pa_raw_right_path

        # Get Run-F path
        disp_p = self.pa_disparity_f_path

        # Build output suffix. Add by default "-ncc"
        suffix = str(self.pa_path / self.pa_key)

        # Build command
        corr_eval_command = f"corr_eval {left_p} {right_p} {disp_p} {suffix}\
        --kernel-size {corr_kernel_size} {corr_kernel_size}\
        --metric {metric}\
        --prefilter-mode 2"  # Used by Amaury in his command but... I don't know why

        # Launch
        os.system(corr_eval_command)

        # Rename corr_eval to fit with GMC structure
        self.pa_cc_path = self.pa_cc_raw_path.rename(
            self.pa_path / f"{self.pa_key}_CC.tif"
        )

        # Copy .qml file for set a default style in Qgis
        template_style = Path(
            Path(__file__).parent, "resources", "map_styles", "corr-eval_style.qml"
        )
        target_style = str(self.pa_cc_raw_path)[:-4]
        cp_command = f"cp {template_style} {target_style}.qml"
        os.system(cp_command)

        return True

    # ! This function is useless. Should be removed
    def save_corrdata(self, verbose=False):
        """
        Move ASP correlation outputs from temporary storage to the pair directory (legacy).
        
        .. warning::
            This method is deprecated. Modern workflows write directly to the pair directory.
        
        Relocates correlation results from GeoMultiCorr's temporary staging area
        (``geomulticorr/temp/``) to the final project pair directory.
        
        :param verbose: If True, print shell command output; otherwise suppress it.
        :type verbose: bool
        
        :return: True if transfer succeeded; False if the directory does not exist afterward.
        :rtype: bool
        """
        # ! This function should be evaluated

        # Hide ugly warnings about symlinks
        verbose_mode = {False: " > /dev/null 2>&1", True: ""}

        # get the GeoMultiCorr temp storage location
        departure = Path(ROOT_OUTPUTS, self.pa_key)

        # get the displacements folder path in the current session
        destination = self.pa_path

        # send the command to bring back the temporal data in the current session
        os.system(f"mv {departure} {destination} {verbose_mode[verbose]}")

        # If the transfer have worked, we delete the data in the temp dir
        if self.pa_path.exists():
            os.system(f"rm -rf {departure}")
            return True

        return False

    def compute_magnitude(self):
        """
        Compute and save the displacement magnitude raster in metres.
        
        Extracts EW and NS displacements from the final disparity raster (``pa_disparity_f_path``),
        converts from pixel to metre coordinates, computes the vector magnitude,
        and saves to ``pa_magn_path``. Returns cached result on subsequent calls.
        
        :return: Magnitude raster in metres.
        :rtype: geoutils.Raster
        
        :raises AssertionError: If the disparity raster does not exist.
        """

        assert self.pa_disparity_f_path.exists(), "this pair is not yet correlate"

        # Check if the file is ever existing
        if self.pa_magn_path.exists():
            return self.get_magn()

        # Open the stack with horizontal and vertical displacements
        xDisp, yDisp = gu.Raster(
            str(self.pa_disparity_f_path), nBands=[1, 2], load_pixels=True
        ).splitBands()

        # We switch values using negative values because ASP gives displacements
        # in pixel coordinates using as reference upper-left corner
        yDisp *= -1.0

        # Get Metadata
        pxSizeX, _ = xDisp.getPixelSize()

        # Magnitude raster creation
        magn = (xDisp**2 + yDisp**2) ** 0.5

        # Magnitude in meters
        magn_in_meters = magn * pxSizeX
        magn_in_meters.save(str(self.pa_magn_path))

        return magn_in_meters

    def extract_raw_displacements(self, save_plot: bool = True) -> dict:
        """
        Extract and convert EW/NS displacements from ASP disparity raster to metres.
        
        Reads the 3-band final disparity raster (xDisp, yDisp, GoodPixMap in pixels),
        converts pixel values to metres using the raster resolution, flips the NS sign
        (ASP uses upper-left corner convention), saves individual EW/NS rasters,
        renames the NCC quality raster, computes displacement statistics, and optionally
        generates a 9-panel control figure.
        
        **Output files created:**
        
        - ``pa_ew_path``: EW displacement in metres
        - ``pa_ns_path``: NS displacement in metres
        - ``pa_cc_path``: Renamed quality raster (NCC)
        - ``pa_plot_raw_path``: Control plot (if ``save_plot=True``)
        - ``pa_stats_path``: JSON file with raw correlation statistics
        
        :param save_plot: If True, generate and save a 9-panel control figure.
        :type save_plot: bool
        
        :return: Full pair stats dictionary (persisted to JSON).
        :rtype: dict
        
        :raises AssertionError: If the disparity raster does not exist.
        """
        assert (
            self.pa_disparity_f_path.exists()
        ), f"Disparity raster not found: {self.pa_disparity_f_path}"

        # 1. Split bands from -F.tif
        r = gu.Raster(str(self.pa_disparity_f_path), nodata=0.0)
        xDisp_px, yDisp_px, goodPixMap = r.split_bands(copy=True)

        # 2. Convert pixels → metres; NS sign convention (ASP uses UL-corner reference)
        res_x, res_y = r.res
        xDisp_m = xDisp_px * res_x
        yDisp_m = yDisp_px * res_y * -1.0

        # 3. Save EW and NS rasters
        xDisp_m.save(str(self.pa_ew_path))
        yDisp_m.save(str(self.pa_ns_path))
        logger.file(f"Saved EW displacement: {self.pa_ew_path.name}")
        logger.file(f"Saved NS displacement: {self.pa_ns_path.name}")

        # 4. Compute stats and persist to JSON under "raw_corr_stats"
        import geomulticorr.stats.stats as gmc_stats

        stats_dict = gmc_stats.save_raw_corr_stats(self)

        # 5. Control plot
        if save_plot:
            from geomulticorr.utils.gmc_functions import plot_show_raw_results

            ncc = (
                gu.Raster(str(self.pa_cc_raw_path))
                if self.pa_cc_raw_path.exists()
                else None
            )
            fig = plot_show_raw_results(
                xDisp=xDisp_m,
                yDisp=yDisp_m,
                ncc=ncc,
                fig_name=self.pa_key,
            )
            fig.savefig(
                str(self.pa_plot_raw_path), dpi=150, format="JPEG", bbox_inches="tight"
            )
            plt.close(fig)
            logger.save(f"Control plot saved: {self.pa_plot_raw_path.name}")

        # 6. Rename -ncc.tif → _CC.tif (after plot so pa_cc_raw_path is still valid above)
        if self.pa_cc_raw_path.exists():
            cc_renamed = self.pa_cc_raw_path.rename(
                self.pa_path / f"{self.pa_key}_CC.tif"
            )
            logger.file(f"Renamed {self.pa_cc_raw_path.name} → {cc_renamed.name}")

        return stats_dict

    # def vectorize(self, epsg, output_pixel_size=None, method='average', write=True, crop='', cropFeatureNum=0):
    #     """
    #     create a points vector layer, mappable as a vector field in Qgis

    #     output_pixel_size : space between each point. By default, it is the pixel size of the pair disparity map
    #     method  : algorithm to use to resample the data (is a output_pixel_size value is given)
    #     """

    #     # Get vector components - unit = pixels and referential = matrix (Y top is Y min)
    #     if crop == '':
    #         initial_dx = self.get_dispX_geoim()
    #         initial_dy = self.get_dispY_geoim()
    #     else:
    #         disp = rt.Open(self.pa_disparity_f_path, geoExtent=crop, featureNum=cropFeatureNum, load_pixels=True, nBands=[1,2])
    #         initial_dx, initial_dy = disp.splitBands()

    #     initial_pixel_size_x, initial_pixel_size_y = initial_dx.getPixelSize()

    #     # Modify the original displacement rasters by resampling them with output resolution asked
    #     if output_pixel_size != None:
    #         dx_in_pixels = initial_dx.resize(output_pixel_size, method=method)
    #         dy_in_pixels = initial_dy.resize(output_pixel_size, method=method)

    #     # Elsewhere, we will work on a copy of the original displacement rasters
    #     else:
    #         dx_in_pixels = initial_dx.copy()
    #         dy_in_pixels = initial_dy.copy()
    #         output_pixel_size = initial_pixel_size_x

    #     # Get metadata
    #     _, nRows, nCols = dx_in_pixels.getShape()

    #     # Convert displacements in meters
    #     dx_in_meters = dx_in_pixels * initial_pixel_size_x
    #     dy_in_meters = dy_in_pixels * initial_pixel_size_y

    #     # Switch the displacements if the pair is reversed
    #     if self.pa_left.th_year > self.pa_right.th_year :
    #         dx_in_meters *= -1
    #         dy_in_meters *= -1

    #     # Compute vector norm (magnitude) in meters
    #     d_in_meters = (dx_in_meters ** 2 + dy_in_meters ** 2) **0.5

    #     # Convert into annual velocity using decimal dt for precision
    #     dx_in_meters_per_year = dx_in_meters / self.pa_dt_years
    #     dy_in_meters_per_year = dy_in_meters / self.pa_dt_years
    #     d_in_meters_per_year  = d_in_meters  / self.pa_dt_years

    #     # Compute displacement geographic direction (vector orientation)
    #     array_complex_dx_dy = np.apply_along_axis(lambda args: [complex(*args)], 0, [dx_in_meters.array,dy_in_meters.array]).reshape(nCols, nRows)
    #     array_direction = np.angle(array_complex_dx_dy, deg=True)
    #     direction = d_in_meters.copy()
    #     direction.array = array_direction

    #     # Make a big GeoIm with all this lovely dataset
    #     to_vectorize = rt.stack([
    #         dx_in_pixels,
    #         dy_in_pixels,
    #         dx_in_meters,
    #         dy_in_meters,
    #         d_in_meters,
    #         dx_in_meters_per_year,
    #         dy_in_meters_per_year,
    #         d_in_meters_per_year,
    #         direction
    #     ])

    #     # Use the vectorize raster_tools function to make a vector point on each pixel
    #     # with an attribute column with the pixel value for each band (here we got 9 band)
    #     vectors = rt.vectorize(to_vectorize).set_crs(epsg=epsg)
    #     vectors.columns = [
    #             'dx_in_pixels',
    #             'dy_in_pixels',
    #             'dx_in_meters',
    #             'dy_in_meters',
    #             'd_in_meters',
    #             'dx_in_meters_per_year',
    #             'dy_in_meters_per_year',
    #             'd_in_meters_per_year',
    #             'direction',
    #             'geometry']

    #     # Write vector layer in geopackage
    #     if write == True:
    #         current_vector_layer_name = f"{output_pixel_size}_{self.pa_key}"
    #         vectors.to_file(self.pa_vect_path, layer=current_vector_layer_name+'shift')

    #         # Copy .qml file
    #         template_style = Path(Path(__file__).parent, 'resources', 'map_styles', 'vector-field_style_1.qml')
    #         target_style = str(self.pa_vect_path)[:-5]
    #         cp_command = f"cp {template_style} {target_style}.qml"
    #         os.system(cp_command)

    #     return vectors

    def pa_full(
        self,
        epsg,
        corr_algorithm=2,
        corr_kernel_size=7,
        corr_xthreshold=10,
        vector_res=20,
        method="average",
        metric_eval="ncc",
    ):
        """
        Execute the full pair processing pipeline (legacy method).
        
        .. warning::
            This method is deprecated. Use individual workflow methods or the
            modern corrections pipeline instead.
        
        Runs all processing steps in sequence: clipping, correlation, data migration,
        quality evaluation, magnitude computation, and vectorization.
        
        :param epsg: EPSG code for vector output CRS.
        :type epsg: int
        
        :param corr_algorithm: ASP algorithm code.
        :type corr_algorithm: int
        
        :param corr_kernel_size: Correlation kernel size.
        :type corr_kernel_size: int
        
        :param corr_xthreshold: Cross-correlation threshold.
        :type corr_xthreshold: float
        
        :param vector_res: Output vector point spacing in metres.
        :type vector_res: int
        
        :param method: Resampling method for vectorization ("average", etc.).
        :type method: str
        
        :param metric_eval: Quality metric ("ncc", etc.).
        :type metric_eval: str
        
        :return: True if all steps completed successfully.
        :rtype: bool
        """

        # Clip
        self.clip()

        # Corr
        self.corr(corr_algorithm, corr_kernel_size, corr_xthreshold)

        # Save
        self.save_corrdata()

        # Eval
        self.corr_eval(corr_kernel_size=corr_kernel_size, metric=metric_eval)

        # Magn
        self.compute_magnitude()

        # Vectors
        self.vectorize(epsg, output_pixel_size=vector_res, method=method)

        return True

    ### Analyze of the displacement fields

    # def get_moving_areas(self, n_clusters=2, mode='m', save=True):

    #     """
    #     Make a segmentation of a displacement
    #     raster with K-Means clustering
    #     """

    #     outpath = Path(self.pa_path, f"KMe_N{n_clusters}_{self.pa_key}.tif")

    #     # Check if the raster is already existing
    #     if outpath.exists():
    #         return rt.geoim.Geoim(outpath)

    #     # Get the interesting GeoIm
    #     disp = self.get_interesting_geoim(mode)

    #     # Extract his array
    #     disp_ar = disp.array

    #     # Reshape for the clustering
    #     disp_arX = disp_ar.reshape(-1,1)

    #     # Create the classifier
    #     k_means_classifier = cluster.KMeans(n_clusters=n_clusters, n_init=10)

    #     # Fit to the data
    #     k_means_classifier.fit(disp_arX)

    #     # Get the labels
    #     clusters_labels = k_means_classifier.labels_

    #     # re-switch the classified vector as image (2D array)
    #     cluster_disp_ar = clusters_labels.reshape(disp_ar.shape)

    #     # Assign this array to a new geoim
    #     cluster_disp = disp.copy()
    #     cluster_disp.array = cluster_disp_ar

    #     # Save it
    #     if save:
    #         cluster_disp.save(str(outpath))

    #     return cluster_disp

    # def denoise_moving_areas(self, operator_size=30, n_clusters=2, mode='m', save=True):
    #     """
    #     Create new raster of moving areas, normally with less noise
    #     """

    #     # Build output filepath
    #     outpath = Path(self.pa_path, f"KMe_N{n_clusters}_Un-{operator_size}_{self.pa_key}.tif")

    #     # Build a morphological operator
    #     operator = np.ones((operator_size, operator_size))

    #     # Get the moving areas from displacement field by k-means clustering
    #     mas = self.get_moving_areas(n_clusters, mode)

    #     # Extract the array and convert it compatible with the operator
    #     mas_ar = mas.array.astype('uint8')

    #     # Denoise
    #     mas_denoised_ar = cv.morphologyEx(mas_ar, cv.MORPH_OPEN, operator)

    #     # Build a new geoim and change his array
    #     mas_denoised = self.get_moving_areas().copy()
    #     mas_denoised.array = mas_denoised_ar

    #     # Save
    #     if save:
    #         mas_denoised.save(str(outpath))

    #     return mas_denoised

    def vectorize_moving_areas(self, epsg=21781, min_surf="", n_clusters=4, mode="m"):
        """
        Create vector polygon layer from moving areas (K-Means clustering result).
        
        Vectorizes the classified raster (from K-Means clustering; see ``get_moving_areas``)
        into polygon geometries, filters by minimum surface area, and saves to a GeoPackage.
        
        **Prerequisites:**
        
        - A classified raster must exist: ``KMe_N<n_clusters>_<pa_key>.tif``
          (generated by ``get_moving_areas`` method)
        
        **Output files:**
        
        - GeoPackage with polygon geometries
        - Optional minimum surface filter applied if ``min_surf`` is provided
        
        :param epsg: Target EPSG code for output CRS (default: 21781 = CH1903+).
        :type epsg: int
        
        :param min_surf: Minimum polygon surface area in km² to retain (empty string = no filter).
        :type min_surf: str or float
        
        :param n_clusters: Number of clusters used in K-Means classification.
        :type n_clusters: int
        
        :param mode: Displacement mode ("m", "x", "y", etc.; for future use).
        :type mode: str
        
        :return: None
        
        :raises FileNotFoundError: If the classified raster does not exist.
        """
        mask = None
        with rasterio.Env():
            target_path = str(
                Path(self.pa_path, f"KMe_N{n_clusters}_{self.pa_key}.tif")
            )
            with rasterio.open(target_path) as src:
                image = src.read(1)  # first band
                results = (
                    {"properties": {"raster_val": v}, "geometry": s}
                    for i, (s, v) in enumerate(
                        shapes(image, mask=mask, transform=src.transform)
                    )
                )
        geoms = list(results)
        gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms).set_crs(
            epsg=epsg
        )
        if min_surf != "":
            gpd_polygonized_raster = gpd_polygonized_raster[
                gpd_polygonized_raster.area / 1000 > min_surf
            ]
            output_path = str(
                Path(
                    self.pa_path,
                    f"{self.pa_key}_classif-{n_clusters}_min-surf-{min_surf}.gpkg",
                )
            )
        else:
            output_path = str(
                Path(self.pa_path, f"{self.pa_key}_classif-{n_clusters}.gpkg")
            )
        gpd_polygonized_raster.to_file(output_path)

    def get_interesting_geoim(self, mode):
        """
        Retrieve the raster corresponding to a requested displacement or velocity mode.
        
        Provides a unified interface for accessing different raster types (displacement,
        velocity, or component-specific). Useful for generalized analysis workflows.
        
        **Supported modes:**
        
        - ``m``: Displacement magnitude (metres)
        - ``x``: EW displacement (metres)
        - ``y``: NS displacement (metres)
        - ``vm``: Velocity magnitude (metres/year)
        - ``vx``: EW velocity (pixels/year)
        - ``vy``: NS velocity (pixels/year)
        
        :param mode: Mode identifier (case-insensitive).
        :type mode: str
        
        :return: Raster object for the requested mode.
        :rtype: geoutils.Raster
        
        :raises ValueError: If the mode is not recognized.
        """
        match mode.lower():
            case "m":
                target = self.get_magn_geoim()
            case "x":
                target = self.get_dispX_geoim()
            case "y":
                target = self.get_dispY_geoim()
            case "vm":
                target = self.get_vmagn_geoim()
            case "vx":
                target = self.get_vx_geoim()
            case "vy":
                target = self.get_vy_geoim()
        return target

    def get_slice(self, geoLine, mode="m"):
        """
        Extract displacement values along a linear geomultifeature.
        
        Samples a cross-section of displacement or velocity values at regular intervals
        along a provided line geometry. Useful for profile analysis and transect studies.
        
        :param geoLine: Linear geometry; can be a GeoSeries, GeoDataFrame,
                        shapely LineString, file path, or coordinate pair (A, B).
        :type geoLine: gpd.GeoSeries, gpd.GeoDataFrame, shapely.geometry.LineString, str, Path, or tuple
        
        :param mode: Displacement/velocity mode ("m", "x", "y", "vm", "vx", "vy").
        :type mode: str
        
        :return: Cross-section values along the line.
        :rtype: array-like
        """
        return self.get_interesting_geoim(mode).inspectGeoLine(geoLine)

    def get_slices(self, geoLine, ribLength, ribStep, ribOrientation="v", mode="m"):
        """
        Extract displacement profiles perpendicular to a linear feature.
        
        Creates perpendicular "ribs" (transects) at regular intervals along a provided
        line, sampling displacement or velocity values. Useful for geomorphic feature
        analysis (e.g., glacier flow across a centerline).
        
        **Profile parameters:**
        
        - ``ribLength``: Total rib length (metres)
        - ``ribStep``: Spacing between ribs along the feature (metres)
        - ``ribOrientation``: Direction perpendicular to the feature ("v" = vertical, "h" = horizontal)
        
        :param geoLine: Linear geometry (GeoSeries, GeoDataFrame, LineString, path, or tuple).
        :type geoLine: gpd.GeoSeries, gpd.GeoDataFrame, shapely.geometry.LineString, str, Path, or tuple
        
        :param ribLength: Total length of each perpendicular rib (metres).
        :type ribLength: float
        
        :param ribStep: Spacing between ribs along the centerline (metres).
        :type ribStep: float
        
        :param ribOrientation: Rib orientation ("v" or "h"; default "v").
        :type ribOrientation: str
        
        :param mode: Displacement/velocity mode ("m", "x", "y", "vm", "vx", "vy").
        :type mode: str
        
        :return: Transect profiles with displacement values.
        :rtype: array-like
        """
        return self.get_interesting_geoim(mode).inspectRibsAlongthumb(
            geoLine, ribLength, ribStep, ribOrientation
        )


# %%
