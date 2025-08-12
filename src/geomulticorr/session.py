#!/usr/bin/env python
# coding=utf-8
# ----------------------------------------------------------------------------- #
# GeoMultiCorr (GMC) project
# Copyright (C) GeoMultiCorr developer team, 2024.
# session.py
# creation date: 2024-06-10.
#
# Author(s) metadata
# -> author: Diego CUSICANQUI
#   -> affiliation: CNES | ISTerre | Univ. Grenoble Alpes
#   -> email(s): diego.cusicanqui@univ-grenoble-alpes.fr | diego.cusicanqui.vg@gmail.com
# -> author: Thibault DUVANEL
#   -> affiliation: UNIL | Univ. Lausanne
#   -> email(s): thibaut.duvanel@unil.ch
# ----------------------------------------------------------------------------- #

import os
import re
import tqdm
import shutil
from pathlib import Path

import geoutils as gu
import pandas as pd
import geopandas as gpd

from osgeo import gdal
from telenvi import raster_tools as rt

import geomulticorr.geomorph as gmc_geomorph
import geomulticorr.pzone as gmc_pzone
import geomulticorr.pair as gmc_pair
import geomulticorr.spine as gmc_spine
import geomulticorr.thumb as gmc_thumb
import geomulticorr.xzone as gmc_xzone

file_location = Path(__file__)

project_template_location = Path(file_location.parent, "resources", "project_template")

def print_infoBM(text: str,
                bold: bool = False) -> None:
    """Prints a formatted message to the console.

    Args:
        text (str): The string to be printed.
        bold (bool, optional): Whether to print the text in bold style. Defaults to False.
    """
    GMC_TEXT = "[ GMC-info ] :"
    if bold:
        print(f"\033[1m{GMC_TEXT} {text}\033[1m")
    else:
        print(f"{GMC_TEXT} {text}")
#END def

def is_conform_to_gmc_template(target_root_path: str | Path) -> bool:
    """Check GMC structure.
    Checks the 3 conditions to ensure that the 'target_root_path' leads to a folder conforming to the GeoMultiCorr project data architecture.

    Args:
        target_root_path (str | Path): Path where GMC project data will be stored.

    Returns:
        bool: True if the folder conforms to the GMC project data architecture. False otherwise.
    """
    target_root_path = Path(target_root_path)
    target_name = target_root_path.name

    # Firstly, there is a folder named raster-data_parent-folder-name
    target_raster_data_expected_path = Path(target_root_path, f"raster-data_{target_name}")
    if not target_raster_data_expected_path.is_dir():
        return False

    # Secondly, there is a file map_parent-folder-name.qgz
    target_map_expected_path = Path(target_root_path, f"GMC_mapset_{target_name}.qgz")
    if not target_map_expected_path.exists():
        return False

    # Thirdly, there is a file geodatabase_parent-folder-name.gpkg
    target_map_expected_path = Path(target_root_path, f"GMC_geodatabase_{target_name}.gpkg")
    if not target_map_expected_path.exists():
        return False

    return True

def re_searcher(string: str, pattern: str) -> str:
    """Search for a pattern in a string and return the first match.

    Args:
        string (str): The string to search in.
        pattern (str): The pattern to search for.

    Returns:
        str: The first match within the string.
    """
    try:
        return re.search(re.compile(pattern), string).group()
    except AttributeError:
        return "unknown"
    #END try
#END def

def search_date_in_filename(filename: str | Path) -> list:
    """
    Search for date strings in a filename using regular expressions.
    Supports formats like YYYYmmdd, YYYY-mm-dd, YYYY/mm/dd, DD-MM-YYYY, and DD/MM/YYYY.

    Args:
        filename (str | Path): The filename to search.

    Returns:
        list: A list of matched date strings.
    """
    date_patterns = [
        r'(19[0-9]{2}|20[0-9]{2}|21[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])',  # Matches YYYYmmdd
        r'(19[0-9]{2}|20[0-9]{2}|21[0-9]{2})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])',  # Matches YYYY-mm-dd
        r'(19[0-9]{2}|20[0-9]{2}|21[0-9]{2})/(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])',  # Matches YYYY/mm/dd
        r'(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(19[0-9]{2}|20[0-9]{2}|21[0-9]{2})',  # Matches DD-MM-YYYY
        r'(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(19[0-9]{2}|20[0-9]{2}|21[0-9]{2})'   # Matches DD/MM/YYYY
    ]
    
    matches = []
    for date_pattern in date_patterns:
        found = re.findall(date_pattern, filename)
        matches.extend(found)
    
    # Flatten matches to return only the full date strings
    dates = [''.join(match) for match in matches]
    return dates
#END def

def sensors(sensors_names: str | list[str] = ["landsat5", "landsat7", "landsat8", "landsat", "sentinel2", "sentinel",
                                              "spot4", "spot6", "spot7",
                                              "planetscope", "pscope", "planet", 
                                              "aerial", "swissimage", "uav",
                                              "dem"]) -> str:
    """List of sensors supported within GeoMultiCorr.

    Args:
        sensors_names (str | list[str], optional): Sensor names. Defaults to ["landsat5", "landsat7", "landsat8", "landsat", "sentinel2", "sentinel", "spot4", "spot6", "spot7", "planetscope", "pscope", "planet", "aerial", "swissimage", "dem"].

    Returns:
        str: List of sensor names in a regex pattern.
        Example: "landsat5|LANDSAT5|landsat7|LANDSAT7|landsat8|LANDSAT8|landsat|LANDSAT|sentinel2|SENTINEL2|sentinel|SENTINEL|spot4|SPOT4|spot6|SPOT6|spot7|SPOT7|planetscope|PSCOPE|planet|PLANET|aerial|AERIAL|swissimage|SWISSIMAGE|dem|DEM".
    """
    s = ""
    for string in sensors_names:
        s += string.lower() + "|"
        s += string.upper() + "|"
    s = s[:-1]
    return s

def open_gmc_session(target_directory_path: str | Path,
                     epsg: str | int | None = None) -> "Session":
    """Create a new GMC session.

    Args:
        location (str | Path): Path to the directory where the project data will be stored.
        epsg (str | int): EPSG code of the project. If the EPSG code is not provided, it defaults to 4326.

    Returns:
        Session: Returns a Session object.
    """
    if epsg is None:
        epsg = 4326
    #END if

    if not isinstance(target_directory_path, Path):
        target_directory_path = Path(target_directory_path)
    #END if
    return Session(target_directory_path, epsg)

class Session:
    """
    Manipulate data specific to a sample of sites related to an earth surface displacement study.
    """

    def __init__(self, target_root_path: str | Path,
                 epsg: str | int | None = None):

        # pathlib.Path conversion of the string target_root_path
        target_root_path = Path(target_root_path).resolve()

        # If EPSG code is not provided, EPSG:4326 is set by default.
        if epsg is None:
            epsg = 4326

        # Check the adress validity
        assert target_root_path.parent.exists(), print_infoBM("invalid address")

        # The address exist but it's something different than a geomulticorr project
        if target_root_path.exists() and not is_conform_to_gmc_template(target_root_path):
            raise ValueError(print_infoBM("something else exist at this address"))

        # The adress exist and it's a geomulticorr project
        elif target_root_path.exists() and is_conform_to_gmc_template(target_root_path):
            pass

        # The adress is valid but don't exist : we create a new geomulticorr project
        else:
            project_name = Path(target_root_path).name
            print_infoBM(f"Creating a new GMC project named '{project_name}'")
                        
            target_root_path.mkdir(parents=True)
            print_infoBM(f"Working on {target_root_path}")

            # Here we copy the template of the GMC project
            print_infoBM(f"Making a copy of GMC-geodatabase template into {target_root_path}")
            raw_gdb_path = project_template_location / "GMC_geodatabase_project.gpkg"
            shutil.copy(raw_gdb_path, target_root_path)
            new_gdb = target_root_path / f"GMC_geodatabase_project.gpkg"
            new_gdb.rename(target_root_path / f"GMC_geodatabase_{project_name}.gpkg")
            
            print_infoBM(f"Making a copy of GMC-Mapset template into {target_root_path}")
            raw_mapset_path = project_template_location / "GMC_mapset_project.qgz"
            shutil.copy(raw_mapset_path, target_root_path)
            new_mapset = target_root_path / "GMC_mapset_project.qgz"
            new_mapset.rename(target_root_path / f"GMC_mapset_{project_name}.qgz")

            # ! Delete this line once is validated
            # os.system(f"cp -r {project_template_location} {target_root_path}")

            # Here we change the name of the initial data
            # target_root_path.joinpath("geodatabase_template-project.gpkg").rename(target_root_path.joinpath(f"geodatabase_{project_name}.gpkg"))
            # os.rename(src=f"{os.path.join(target_root_path, 'geodatabase_template-project.gpkg')}",
            #     dst=f"{os.path.join(target_root_path, 'geodatabase_' + project_name + '.gpkg')}",)

            # target_root_path.joinpath("map_template-project.qgz").rename(target_root_path.joinpath(f"map_{project_name}.qgz"))
            # os.rename(src=f"{os.path.join(target_root_path, 'map_template-project.qgz')}",
            #     dst=f"{os.path.join(target_root_path, 'map_' + project_name + '.qgz')}",)

            # And we create en empty folder raster-data_project-name
            if not (target_root_path / f"raster-data_{project_name}").exists():
                print_infoBM(f"Creating a new folder raster-data_{project_name}")
                (target_root_path / f"raster-data_{project_name}").mkdir(parents=True)
            # Path(target_root_path, "raster-data_" + project_name).mkdir()

        # Load project data into the current session
        self.path_root = target_root_path
        self.project_name = target_root_path.name
        self.path_raster_data = target_root_path / f"raster-data_{self.project_name}"
        self.path_geodb = target_root_path / f"GMC_geodatabase_{self.project_name}.gpkg"
        # os.path.join(target_root_path, f"geodatabase_{self.project_name}.gpkg")
        self.epsg = epsg

        # There is underscore before the attribute name because user have to access to this data from the getters to ensure that the data is up to date
        self._pzones = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Pzones").to_crs(epsg=epsg)
        self._thumbs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Thumbs").to_crs(epsg=epsg)
        self._geomorphs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Geomorphs").to_crs(epsg=epsg)
        self._pairs = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Pairs").to_crs(epsg=epsg)
        self._xzones = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Xzones").to_crs(epsg=epsg)
        self._spines = gpd.read_file(self.path_geodb, engine="pyogrio", layer="Spines").to_crs(epsg=epsg)

        # List the processing zones names
        self.pz_names = list(self._pzones.pz_name.unique())
    #END def
    ###################################################
    #-------------------- GETTERS --------------------#
    def get_thumbs_overview(self, criterias: str | list[str] = "") -> pd.DataFrame:
        """Get thumbnails overview informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a dataframe with informations about each thumb raster file meeting the criteria. If there's no criteria, send informations about all the thumbs of the project.
        """
        return self._search_engine("Thumbs", criterias)

    def get_thumbs(self, criterias: str | list[str] = "") -> list[gmc_thumb.Thumb]:
        """Get thumbnails

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list: Send a list of thumb.Thumb objects meeting the criterias.
        """
        selected_thumbs = self.get_thumbs_overview(criterias)
        return [gmc_thumb.Thumb(x.th_path) for x in selected_thumbs.iloc]

    def get_pairs_overview(self, criterias: str | list[str] = "") -> pd.DataFrame:
        """Get pairs overview informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a dataframe with each possible Pair according to the Thumbs
        """
        return self._search_engine("Pairs", criterias)

    def get_pairs(self, criterias: str | list[str] = "") -> list[gmc_pair.Pair]:
        """Get pairs informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a list of pair.Pairs objects meeting the criterias
        """
        # TODO: solve issue regarding order of return pairs. When session.get_pairs(['ribon', '2014', '2018']), pair 2018-2014 is returned first.   
        selected_pairs = self.get_pairs_overview(criterias)
        return [gmc_pair.Pair(self, target_path=x.pa_path) for x in selected_pairs.iloc]

    def get_pzones_overview(self, pz_name: str | list[str] = "") -> pd.DataFrame:
        """Get pzones overview informations

        Args:
            pz_name (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a dataframe with each pzone
        """
        return self._search_engine("Pzones", pz_name)

    def get_pzones(self, pz_name: str = "") -> list[gmc_pzone.Pzone]:
        """Get pzones informations

        Args:
            pz_name (str, optional): pzone name. Defaults to "".

        Returns:
            list: Send a list of pzone.Pzones objects.
        """
        selected_pzones = self.get_pzones_overview(pz_name)
        return [gmc_pzone.Pzone(x.pz_name, self) for x in selected_pzones.iloc]

    def get_geomorphs_overview(self, criterias: str | list[str] = "") -> pd.DataFrame:
        """Get geomorphos overview informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Send a dataframe with each geomorph.
        """
        return self._search_engine("Geomorphs", criterias)

    def get_geomorphs(self, criterias: str | list[str] = "") -> list[gmc_geomorph.Geomorph]:
        """Get geomorphs informations

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list: Send a list of geomorph.Geomorphs objects.
        """
        selected_geomorphs = self.get_geomorphs_overview(criterias)
        return [gmc_geomorph.Geomorph(self, x.ge_frogi_id) for x in selected_geomorphs.iloc]

    def get_xzone(self, xz_id: str | int | float) -> gmc_xzone.Xzone:
        """Get a xzone

        Args:
            xz_id (str | int | float): xzone id.

        Returns:
            gmc_xzone.Xzone: Send a xzones.Xzones object
        """
        return gmc_xzone.Xzone(self, xz_id)

    def get_pairs_overview_on_period(self,
                                     ymin: str | pd.Timestamp,
                                     ymax: str | pd.Timestamp,
                                     criterias: str | list[str] = "") -> pd.DataFrame:
        """Get pairs overview informations on a given period

        Args:
            ymin (str | pd.Timestamp): minimun year of the given period.
            ymax (str | pd.Timestamp): maximum year of the given period.
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: Return a dataframe with each pair completely included in the period [yMin;yMax]
        """
        # def is_valid_timestamp(date_str):
        #     try:
        #         pd.to_datetime(date_str)
        #         return True
        #     except ValueError:
        #         return False
        # # Convert ymin and ymax to pd.Timestamp if they are strings
        # if isinstance(ymin, str) and is_valid_timestamp(ymin):
        #     ymin = pd.to_datetime(ymin)
        # if isinstance(ymax, str) and is_valid_timestamp(ymax):
        #     ymax = pd.to_datetime(ymax)
            
        pairs = self.get_pairs_overview(criterias)
        pairs["chrono_min"] = pairs.apply(
            lambda row: min(int(row.pa_left_date.split("-")[0]),
                            int(row.pa_right_date.split("-")[0]),),axis=1,)
        pairs["chrono_max"] = pairs.apply(
            lambda row: max(int(row.pa_left_date.split("-")[0]),
                            int(row.pa_right_date.split("-")[0]),),axis=1,)
        pairs = pairs[(pairs.chrono_min >= ymin) & (pairs.chrono_max <= ymax)]
        return pairs

    def get_pairs_on_period(self,
                            ymin: str | pd.Timestamp,
                            ymax: str | pd.Timestamp,
                            criterias: str | list[str] = "") -> list[gmc_pair.Pair]:
        """Get pairs on a given period

        Args:
            ymin (str | pd.Timestamp): minimun year of the given period.
            ymax (str | pd.Timestamp): maximun year of the given period.
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list[gmc_pair.Pair]: Returns a list of gcm_pair.Pair objects with the pairs completely included in the period [yMin;yMax]
        """
        board = self.get_pairs_overview_on_period(ymin, ymax, criterias)
        pairs = [gmc_pair.Pair(self, p.pa_path) for p in board.iloc]
        return pairs

    def get_spine(self, sp_id: str | int | float) -> gmc_spine.Spine:
        """Get spine

        Args:
            sp_id (str | int | float): Spine ID.

        Returns:
            gmc_spine.Spine: Send a spine.Spine object
        """
        return gmc_spine.Spine(self, sp_id)

    def get_dems(self, criterias: str | list[str] = "") -> list:
        """Get a list of DEMs

        Args:
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            list: Return a list of DEMs
        """
        return [pz.get_dem() for pz in self.get_pzones(criterias) if pz.get_dem() != False]

    def get_georasters_map(self,
                           image_type: str | list[str] = "opt",
                           suffix: str = "") -> gpd.GeoDataFrame:
        """Get a georasters map

        Args:
            image_type (str, optional): Data type to be mepped. In this current version, only optical data is supported. Defaults to "opt".
            suffix (str, optional): images extension ('*.tif' or '*.jp2'). Defaults to "".

        Returns:
            gpd.GeoDataFrame: Return a GeoPandas DataFrame with the georasters map extents
        """
        layername = f"Georaster_bank_{image_type}_{suffix}"
        try:
            return gpd.read_file(self.path_geodb, layer=layername, engine="pyogrio")
        except ValueError:
            print(f"No georasters map named {layername} in the session geodatabase")
            return None

    def map_georasters_bank(self,
                            georasters_bank_path: str | Path,
                            extensions: str | list[str] = ["tif", "jp2"],
                            image_type: str = "opt",
                            epsg: str | int | None = None,
                            suffix: str = ""):

        # By default, write the output vector layer in the epsg of the session / project
        if epsg is None:
            epsg = self.epsg

        # check the validity of the georasters_bank_path
        georasters_bank_path = Path(georasters_bank_path)
        assert (georasters_bank_path.exists()), f"{georasters_bank_path} is not an existing path"

        # Get a list of all the .tif or .jp2 under georasters_bank_path
        targets = []
        for x in extensions:
            targets += georasters_bank_path.glob(f"**/*.{x.lower()}")
            targets += georasters_bank_path.glob(f"**/*.{x.upper()}")

        # Build a vector layer, with metadata and extent of each image
        features = []
        print_infoBM(f"Mapping the images in {georasters_bank_path}...")
        for ta in tqdm.tqdm(targets):

            # Create empty serie
            ft = gpd.GeoSeries()

            # Write all kind of metadata
            ft["filename"] = (ta.name).split('/')[-1]
            if image_type == "opt":
                ft["sensor"] = re_searcher(str(ta), sensors())
            elif "dem" in image_type.lower():
                ft["sensor"] = "dem"
            #END if
            # ft["xRes"], ft["yRes"] = rt.getPixelSize(str(ta)) #! to delete after testing
            ft["xRes"], ft["yRes"] = gu.Raster(str(ta)).res
            # ft["bands"], ft["rows"], ft["cols"] = rt.getShape(str(ta)) #! to delete after testing
            ft["bands"] = gu.Raster(str(ta)).count
            ft["rows"], ft["cols"] = gu.Raster(str(ta)).shape

            ###### ###### ###### ######
            ###### Modulable code - you have to adapt this block according to the way of naming of your rasters in the rasters_bank
            ###### ###### ###### ######

            # Get acquisition date : Diego PhD thesis harddisk naming way
            if "aerial" in ft["sensor"].lower():
                # We split the filename to get the date -- according to UNILgis naming way
                acq_date = ft["filename"].split("-")[1]
                acq_date = f"{acq_date[:4]}-{acq_date[4:6]}-{acq_date[6:]}"
                ft["acq_date"] = acq_date
            #END if

            # Get acquisition date : SPOT naming way
            if "spot" in ft["sensor"].lower():
                # Here we get the path to the metadata.xml file associated to the image
                md_file_path = gdal.Info(gdal.Open(str(ta))).split("\n")[2]

                # Oppening the metadata file
                try:
                    with open(md_file_path.strip(), "r") as mdf:
                        mds = mdf.read()

                    # Here we track the 'imaging_date' tag and extract his value
                    imaging_date_line = re_searcher(mds, "<IMAGING_DATE>.+</IMAGING_DATE>")
                    ft["acq_date"] = re_searcher(imaging_date_line, "[0-9]+-[0-9]+-[0-9]+")

                except FileNotFoundError:
                    print_infoBM("Metadata file no found. Getting metadata from the image name by default")
                    ft['acq_date'] = search_date_in_filename(ft["filename"])
                    continue
                #END try
            #END if

            # # Get acquisition date : PlanetScope naming way
            #TODO: check if th metadata path is correct & check if the date is correct.
            if 'planet' in ft['sensor'].lower():
                # Here we get the path to the metadata.xml file associated to the image
                md_planet_file_path = gdal.Info(gdal.Open(str(ta))).split("\n")[2]

                # Oppening the metadata file
                try:
                    with open(md_planet_file_path.strip(), "r") as mdf:
                        mds = mdf.read()
                    
                    # Here we track the 'imaging_date' tag and extract his value
                    imaging_date_line = re_searcher(mds, "<eop:acquisitionDate>.+</eop:acquisitionDate>")
                    ft['acq_date'] = re_searcher(imaging_date_line, "[0-9]+-[0-9]+-[0-9]+")
                
                except FileNotFoundError:
                    print_infoBM("Metadata file no found. Getting metadata from the image name by default")
                    ft['acq_date'] = search_date_in_filename(ft["filename"])
                    continue

            # Get acquisition date : swiss alti 3D DEM
            if ft["sensor"] == "dem":
                ft["acq_date"] = ft["filename"].split("_")[1]

            # Get acquisition date : swissimage format
            if "swissimage" in ta.name:
                ft["acq_date"] = f"{ft['filename'].split('_')[1]}-01-01"
                ft["sensor"] = "aerial"

            ###### ###### ###### #######
            ###### End of modulable code ------------------------------
            ###### ###### ###### #######

            # Get the image geographic extent
            # ft["geometry"] = rt.drawGeomExtent(str(ta), geomType="shly") #! to delete after testing
            ft["geometry"] = gu.Raster(str(ta)).image.get_footprint_projected(ta.crs)
            ft["filepath"] = str(ta)

            # Add the feature to the future layer (just a list for now)
            features.append(ft)

        # Transform the list of GeoSeries features into a GeoDataFrame, and set his CRS
        layer = gpd.GeoDataFrame(features).set_crs(epsg)

        # Save the map in the geodatabase with dynamic name
        # self.copy_geodb()
        # ! To be tested with the new naming way
        # layer.to_file(self.path_geodb, layer=f"extents-map_{image_type}_{suffix}")
        layer.to_file(self.path_geodb, layer=f"Georaster_bank_{image_type}_{suffix}")
        return layer
    #------------------ END GETTERS ------------------#
    ###################################################

    ###################################################
    #-------------------- SETTERS --------------------#
    def update_vector_data_session(self):
        """
        Update the instance session attributes from geodatabase layers. Useful when user modify the data from Qgis
        """
        self._spines = gpd.read_file(self.path_geodb, layer="Spines", engine="pyogrio")
        self._pzones = gpd.read_file(self.path_geodb, layer="Pzones", engine="pyogrio")
        self._thumbs = gpd.read_file(self.path_geodb, layer="Thumbs", engine="pyogrio")
        self._geomorphs = gpd.read_file(self.path_geodb, layer="Geomorphs", engine="pyogrio")
        self._xzones = gpd.read_file(self.path_geodb, layer="Xzones", engine="pyogrio")

    def update_thumbs(self):
        """Add or remove rows in Thumbs layer, according to the raster thumbs stored in the GMC project
        """

        # Copy the geodatabase before the transaction
        # assert self.copy_geodb()

        # Get 2 version of the Thumbs layer
        opt_root = Path(self.path_raster_data)
        old = self._thumbs
        new = gpd.GeoDataFrame([gmc_thumb.Thumb(target_path).to_pdserie()
                                for target_path in filter(lambda x:
                                                          gmc_thumb.THUMBNAME_PATTERN.match(x.name),
                                                          list(opt_root.glob(pattern="**/opticals/*.tif")),)])

        # Comparison
        common = new.merge(old, on=["th_path"])
        stables = old[old.th_path.isin(common.th_path)]
        addeds = new[~new.th_path.isin(common.th_path)]

        # Push it into the geodatabase
        updated = pd.concat([stables, addeds])
        updated.to_file(self.path_geodb, layer="Thumbs")

        # Update instance
        self._thumbs = updated

        # Update pairs
        self.update_pairs()

        return updated

    def update_pairs(self):
        """Add or remove rows in Pairs layer, according to the thumbs stored in the project
        """

        # Copy the geodatabase before the transaction
        # assert self.copy_geodb()

        # For each processing zone we
        updated = []
        for pz in self.get_pzones():
            pairs = pz.get_pairs_overview()
            [updated.append(pa) for pa in pairs.iloc()]

        # Push it into the geodatabase
        updated = gpd.GeoDataFrame(updated).set_crs(epsg=self.epsg)
        updated.to_file(self.path_geodb, layer="Pairs")

        # Update instance
        self._pairs = updated
        return updated

    # def copy_geodb(self):
    #     """Quickly create a copy of the project geopackage named backupath_geodb.gpkg
    #     """
    #     backup_path = Path(self.path_root, "GMC_geod_backup.gpkg")
    #     shutil.copytree(self.path_geodb, backup_path)
    #     # os.system(f"cp -r {self.path_geodb} {backup_path}")
    #     return Path(self.path_root, "GMC_geodb_backup.gpkg").exists()

    ###############################

    def sieve(self, image_type: str | list[str] = "opt",
              suffix: str | list[str] = "",
              res: int | float | str = 1,
              alg: str | list[str] = "nearest",
              band: int = 1):
        """Intersect a georasters map layer and the project pzones layer to create thumbs"""
        # TODO: paralellize the process. URGENT!

        # Check if user have drawn some pzones
        assert (len(self.get_pzones_overview()) > 0), "no pzone registered for this project"

        # Open the corresponding georasters map
        georasters_map = self.get_georasters_map(image_type, suffix)

        # Inform user about CRS differences between georasters map and pzones layer
        print(f"Map epsg : {georasters_map.crs}\nPzone  epsg : {self.get_pzones_overview().crs}")

        # For each processing zone
        for pz in self.get_pzones_overview().iloc:
            print(f"\n---\n{pz['pz_name']}")

            # Create a directory to store the raster_data of the pzone
            pz_rasterdata = Path(self.path_root, f"raster-data_{self.project_name}", pz["pz_name"])

            # Create subdirectories for the thumbs and the displacements measurements associated
            if image_type == "opt":
                pz_opticals = Path(pz_rasterdata, "opticals")
                pz_disps = Path(pz_rasterdata, "displacements")
                for p in [pz_rasterdata, pz_opticals, pz_disps]:
                    if not p.exists():
                        p.mkdir(parents=True)

            # Get the images intersecting the zone
            selection = georasters_map[georasters_map["geometry"].intersects(pz["geometry"])]

            """
            Here we have to regroup the lines of 'selection' where the acquisition date and the sensor are identicals
            Then we loop on thoses groups
            And we merge the part of each group
            """

            # Group by acquisition date and sensor
            selection_merged_by_date_and_sensor = selection.groupby(["acq_date", "sensor"])["filepath"].apply(list)

            # For each group
            for group_id, group in enumerate(selection_merged_by_date_and_sensor):

                acq_date = selection_merged_by_date_and_sensor.index[group_id][0]
                sensor = selection_merged_by_date_and_sensor.index[group_id][1]

                # Define output filename and full path (ex: raster-data_vanoise/sachette/opticals/sachette_2021-03-08_AERIAL.tif)
                if image_type == "opt":
                    thumb_name = f"{pz['pz_name']}_{acq_date}_{sensor}.tif"
                    thumb_path = str(Path(pz_opticals, thumb_name))

                elif image_type == "dem":
                    thumb_name = f"{pz['pz_name']}_dem.tif"
                    thumb_path = str(Path(pz_rasterdata, thumb_name))

                if Path(thumb_path).exists():
                    continue

                # We check if the file is already existing
                # if not os.path.exists(thumb_path):

                # We crop each image of the group on the p_zone
                # Here, each image receive a number considered as nodata value by gdal
                # where the pixels are inside the p_zone but outside the image
                fully = []
                print("Cropping")
                for mosaic_path in tqdm.tqdm(group):

                    # Difference between opt and dem because of bugs with merge, when the input raster have 1 or many bands.
                    # But if you work with an optical raster containing only one band, bugs can happened... ?
                    if image_type == "opt":
                        # If we don't have to make a resampling
                        if rt.getPixelSize(mosaic_path)[0] == res:
                            mosaic = rt.Open(
                                target=mosaic_path,
                                geoExtent=pz.geometry.bounds,
                                nBands=1,)

                        else:
                            mosaic = rt.Open(
                                target=mosaic_path,
                                geoExtent=pz.geometry.bounds,
                                nRes=res,
                                resMethod=alg,
                                nBands=1,
                            )

                    elif image_type == "dem":
                        if rt.getPixelSize(mosaic_path)[0] == res:
                            mosaic = rt.Open(
                                target=mosaic_path, geoExtent=pz.geometry.bounds
                            )

                        else:
                            mosaic = rt.Open(
                                target=mosaic_path,
                                geoExtent=pz.geometry.bounds,
                                nRes=res,
                                resMethod=alg,
                            )

                    fully.append(mosaic)

                # If there is more than one image acquired at the same date and from the same sensor
                # we accept that it is the initial same image and we merged them together
                # because gdal considered as nodata the places outside each part, the merged work easily
                """
                TELENVI ISSUE 1
                ############### but it's not working if we don't extract a band with rt.Open()
                """

                if len(fully) > 1:
                    print("assemblage")
                    thumb = rt.merge(fully)
                else:
                    thumb = fully[0]

                # Write the thumb
                rt.write(thumb, thumb_path)
    # TODO: to be completed with the other parameters
    def pr_full(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10):
        """Launch all the possible correlations accross all the valid images among the project pzones"""
        logs = []
        for pzone in self.get_pzones():
            logs.append(
                pzone.pz_full(self.epsg, corr_algorithm, corr_kernel_size, corr_xthreshold))
        return pd.DataFrame(logs)

    def _search_engine(self, layername: str, criterias: str | list[str] = "") -> pd.DataFrame:
        """A search engine among project layers

        Args:
            layername (str): layer name.
            criterias (str | list[str], optional): criterias to filter data. Defaults to "".

        Returns:
            pd.DataFrame: return a dataframe with the filtered data.
        """

        # if there is only one criteria we store it in a list
        if type(criterias) in (str, int):
            criterias = [criterias]

        # add an "and" statement between each criteria
        pattern = ""
        for c in criterias:
            if type(c) == int:
                c = str(c)
            pattern += f"(?=.*{c.lower()})"
        pattern = re.compile(pattern)

        match layername:
            case "Thumbs":
                normal_th = self._thumbs
                lower_th = normal_th.apply(lambda x: x.str.lower(), 1)
                return normal_th[lower_th.th_path.str.contains(pattern)]

            case "Pairs":
                normal_pa = self._pairs
                lower_pa = normal_pa.apply(lambda x: x.str.lower(), 1)
                return normal_pa[lower_pa.pa_path.str.contains(pattern)]

            case "Pzones":
                pz_layer = self._pzones
                if criterias != [""]:
                    requested_pz_name = criterias[0].lower()
                    return pz_layer[pz_layer.pz_name == requested_pz_name]
                else:
                    return pz_layer

            case "Geomorphs":
                # We suppose than the criteria can only be a Geomorph ID or a pz_name
                selection = self._geomorphs
                if criterias != [""]:
                    for criteria in criterias:
                        if criteria in self.pz_names:
                            selection = selection[selection.ge_pz_name == criteria]
                        else:
                            selection = selection[selection.ge_frogi_id == criteria]
                    return selection
                else:
                    return selection

    def __repr__(self):
        return f"""
            ------------
            This is a GeoMultiCorr Session open on the project named {self.project_name}
            Their processing zones are : {self.pz_names}
            ------------
            """
