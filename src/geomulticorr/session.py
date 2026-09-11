import os
import re
import tqdm
from pathlib import Path

import pandas as pd

from osgeo import gdal
import geopandas as gpd
from telenvi import raster_tools as rt

import geomulticorr.pzone    as gmc_pzone
import geomulticorr.pair     as gmc_pair
import geomulticorr.thumb    as gmc_thumb

file_location = Path(__file__)

project_template_location = Path(file_location.parent, 'resources', 'project_template')

def is_conform_to_gmc_template(target_root_path):
    """
    check the 3 conditions to be sure than the target root path is leading to a folder conform to the geomulticorr project data architecture
    """

    target_root_path = Path(target_root_path)
    target_name = target_root_path.name

    # Firstly, there is a folder named raster-data_parent-folder-name
    target_raster_data_expected_path = Path(target_root_path, f"raster-data_{target_name}")
    if not target_raster_data_expected_path.is_dir():
        return False

    # Secondly, there is a file map_parent-folder-name.qgz
    target_map_expected_path = Path(target_root_path, f"map_{target_name}.qgz")
    if not target_map_expected_path.exists():
        return False

    # Thirdly, there is a file geodatabase_parent-folder-name.gpkg
    target_map_expected_path = Path(target_root_path, f"geodatabase_{target_name}.gpkg")
    if not target_map_expected_path.exists():
        return False

    return True

def re_searcher(string, pattern):
    """search a pattern in a string and return the first match, or 'unknown' if no match"""
    try : 
        return re.search(re.compile(pattern), string).group()
    except AttributeError:
        return "unknown"

def sensors(sensors_names=['spot6', 'spot7', 'aerial']):
    s = ""
    for string in sensors_names:
        s += string.lower() + "|"
        s += string.upper() + "|"
    s = s[:-1]
    return s

def Open(location, epsg=2056):
    """
    Open a geomulticorr session on the project located at 'location'
    If the project don't exist yet, create it
    
    epsg : the epsg code of the project (default = 2056)
    """

    return Session(location, epsg)

class Session:

    """
    Manipulate data specific to a sample of sites relative to an earth surface displacement study
    A session is linked to a project, which is a folder with a specific architecture
    A project contains :
        - a geopackage named geodatabase_project-name.gpkg
        - a qgis project named map_project-name.qgz
        - a folder named raster-data_project-name, which contains subfolders for each processing zone

    A session contains :
        - the path to the project root folder
        - the path to the raster-data folder
        - the path to the geodatabase
        - the project name
        - the epsg code of the project
        - the processing zones (as a geopandas dataframe)
        - the thumbnails (as a geopandas dataframe)
        - the pairs (as a geopandas dataframe)
        - the processing zones names (as a list of strings)
        
    User can access to the data through getters and setters methods
    User can also launch the main functions of geomulticorr through the session methods
    User can also access to the dataframes directly, but it's not recommended because they can be not up to date
    User can also access to the project files directly, but it's not recommended because they can be not up to date

    Example of use :
    >>> import geomulticorr as gmc
    >>> session = gmc.session.Open('/path/to/project/folder', epsg=2056)
    # here you need to create some pzones from Qgis or directly in the geodatabase
    >>> session.update_pzones()
    >>> session.update_thumbs()
    >>> session.update_pairs()
    >>> session.sieve(data_type='opt', suffix='', res=1, alg='nearest', band=1, delete_existant_thumbs=False, criterias='') 
    >>> pa = session.get_pairs(criterias='cliosses', allow_reversed=False, only_complete=False)[0]
    >>> pa.pr_full(corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10)
    """

    def __init__(self, target_root_path : str, epsg):

        # pathlib.Path conversion of the string target_root_path
        target_root_path = Path(target_root_path)

        # Check the adress validity
        assert target_root_path.parent.exists(), 'adresse invalide'
 
        # The adress exist but it's something different than a geomulticorr project
        if target_root_path.exists() and not is_conform_to_gmc_template(target_root_path):
            raise ValueError("autre chose existe à cette adresse")

        # The adress exist and it's a geomulticorr project
        elif target_root_path.exists() and is_conform_to_gmc_template(target_root_path):
            pass

        # The adress is valid but don't exist : we create a new geomulticorr project
        else:
            project_name = Path(target_root_path).name

            # Here we copy the template
            os.system(f"cp -r {project_template_location} {target_root_path}")

            # Here we change the name of the initial data
            os.rename(
                src = f"{os.path.join(target_root_path, 'geodatabase_template-project.gpkg')}",
                dst = f"{os.path.join(target_root_path, 'geodatabase_' + project_name + '.gpkg')}")

            os.rename(
                src = f"{os.path.join(target_root_path, 'map_template-project.qgz')}",
                dst = f"{os.path.join(target_root_path, 'map_' + project_name + '.qgz')}")

            # And we create en empty folder raster-data_project-name
            Path(target_root_path, 'raster-data_' + project_name).mkdir()

        # Load project data into the current session
        self.p_root = str(target_root_path)
        self.project_name = Path(self.p_root).name
        self.p_raster_data = str(Path(self.p_root, f'raster-data_{self.project_name}'))
        self.p_geodb = os.path.join(target_root_path, f'geodatabase_{self.project_name}.gpkg')
        self.epsg = epsg

        # There is underscore before the attribute name because 
        # user have to access to this data from the getters
        # to ensure that the data is up to date
        self._pzones = gpd.read_file(self.p_geodb, layer='Pzones').to_crs(epsg=epsg)
        self._thumbs = gpd.read_file(self.p_geodb, layer='Thumbs').to_crs(epsg=epsg)
        self._pairs  = gpd.read_file(self.p_geodb, layer='Pairs').to_crs(epsg=epsg)

        # List the processing zones names
        self.pz_names = list(self._pzones.pz_name.unique())

    ########### GETTERS ###########

    def get_thumbs_overview(self, criterias=''):
        """Send a dataframe with informations about each thumb raster file meeting the criterias.
           If there's no criterias, send informations about all the thumbs of the project """
        return self._search_engine('Thumbs', criterias)

    def get_thumbs(self, criterias=''):
        """Send a list of thumb.Thumb objects meeting the criterias"""
        selected_thumbs = self.get_thumbs_overview(criterias)
        return [gmc_thumb.Thumb(x.th_path) for x in selected_thumbs.iloc]

    def create_custom_pair(self, pz_name, left_year, right_year, pa_px_size=None, pa_asp_alg=None, pa_corr_kernel=None, pa_user_notes=None, pa_sub_pixel_mode=None):
        return gmc_pair.Pair(
            or_left=self.get_thumbs([pz_name, left_year])[0],
            or_right=self.get_thumbs([pz_name, right_year])[0],
            pa_px_size=pa_px_size,
            pa_asp_alg=pa_asp_alg,
            pa_corr_kernel=pa_corr_kernel,
            pa_sub_pixel_mode=pa_sub_pixel_mode,
            pa_user_notes=pa_user_notes)

    def get_pairs_overview(self, criterias='', allow_reversed=True, only_complete=False):
        """
        Send a dataframe with each possible Pair according to the Thumbs
        """

        # Recherche standard
        selected_pairs = self._search_engine('Pairs', criterias)

        # Get only the complete ones
        if only_complete :
            selected_pairs = selected_pairs[selected_pairs.pa_status == 'complete']

        # Get only the "left / right" chronological ordered paired
        if not allow_reversed:
            selected_pairs = selected_pairs[selected_pairs.pa_left_date < selected_pairs.pa_right_date]

        # Here we are
        return selected_pairs

    def get_pairs(self, criterias='', allow_reversed=True, only_complete=False):
        """Send a list of pair.Pairs objects meeting the criterias"""
        
        # Get all the pairs matching criterias
        selected_pairs = self.get_pairs_overview(criterias, allow_reversed=allow_reversed, only_complete=only_complete)
        return [gmc_pair.Pair(self, target_path = x.pa_path) for x in selected_pairs.iloc]

    def get_pzones_overview(self, pz_name=''):
        """Send a dataframe with each pzone"""
        return self._search_engine('Pzones', pz_name)

    def get_pzones(self, pz_name=''):
        """Send a list of pzone.Pzones objects"""
        selected_pzones = self.get_pzones_overview(pz_name)
        return [gmc_pzone.Pzone(x.pz_name, self) for x in selected_pzones.iloc]

    def get_pairs_overview_on_period(self, ymin, ymax, criterias=''):
        """Retourne un tableau des paires completement incluses dans la période [yMin;yMax]"""
        pairs = self.get_pairs_overview(criterias)
        pairs['chrono_min'] = pairs.apply(lambda row: min(int(row.pa_left_date.split('-')[0]), int(row.pa_right_date.split('-')[0])), axis=1)
        pairs['chrono_max'] = pairs.apply(lambda row: max(int(row.pa_left_date.split('-')[0]), int(row.pa_right_date.split('-')[0])), axis=1)
        pairs = pairs[(pairs.chrono_min>=ymin)&(pairs.chrono_max<=ymax)]
        return pairs

    def get_pairs_on_period(self, ymin, ymax, criterias=''):
        """Retourne les paires completement incluses dans la période [yMin;yMax]"""
        board = self.get_pairs_overview_on_period(ymin, ymax, criterias)
        pairs = [gmc_pair.Pair(self, p.pa_path) for p in board.iloc]
        return pairs

    def get_dems(self, criterias):
        return [pz.get_dem() for pz in self.get_pzones(criterias) if pz.get_dem() != False]

    def get_georasters_map(self, data_type='opt', suffix=''):
        layername = f"extents-map_{data_type}_{suffix}"
        try:
            return gpd.read_file(self.p_geodb, layer=layername)
        except ValueError:
            print(f'No georasters map named {layername} in the session geodatabase')
            return None

    def map_georasters_bank(self, georasters_bank_path, extensions=['tif','jp2'], data_type='opt', epsg='', suffix=''):

        # By default, write the output vector layer in the epsg of the session / project
        if epsg == '':
            epsg=self.epsg

        # check the validity of the georasters_bank_path
        georasters_bank_path = Path(georasters_bank_path)
        assert georasters_bank_path.exists(), f"{georasters_bank_path} is not an existing path"

        # Get a list of all the .tif or .jp2 under georasters_bank_path
        targets = []
        for x in extensions:
            targets += georasters_bank_path.glob(f'**/*.{x.lower()}')
            targets += georasters_bank_path.glob(f'**/*.{x.upper()}')

        # Build a vector layer, with metadata and extent of each image
        features = []
        print('map the images...')
        for ta in tqdm.tqdm(targets):

            # Create empty serie
            ft = gpd.GeoSeries()

            # Write all kind of metadata
            ft["filename"] = ta.name
            if data_type == 'opt':
                ft["sensor"] = re_searcher(str(ta), sensors())
            elif 'dem' in data_type.lower():
                ft["sensor"] = 'dem'
            ft["xRes"],  ft["yRes"] = rt.getPixelSize(str(ta))
            ft["bands"], ft["rows"], ft["cols"] = rt.getShape(str(ta))

            # Get acquisition date : swissimage format as I define on my computer for my phd thesis
            if 'swissimage' in ta.name:
                ft["acq_date"] = ft['filename'].split('_')[0]
                ft["sensor"] = 'aerial'
                ft["acq_year"] = f"{ft.acq_date.split('-')[0]}-01-01"

            # Get the image geographic extent
            ft["geometry"] = rt.drawGeomExtent(str(ta), geomType='shly')
            ft["filepath"] = str(ta)

            # Add the feature to the future layer (just a list for now)
            features.append(ft)

        # Transform the list of GeoSeries features into a GeoDataFrame, and set his CRS
        layer = gpd.GeoDataFrame(features).set_crs(epsg)

        # Save the map in the geodatabase with dynamic name
        self.copy_geodb()
        layer.to_file(self.p_geodb, layer=f"extents-map_{data_type}_{suffix}")
        return layer

    ########### SETTERS ###########

    def update_pzones(self):
        """
        Update the instance session.pzones with the content of the geodatabase Pzones layer
        Useful when user modify the data from Qgis
        """
        self._pzones = gpd.read_file(self.p_geodb, layer='Pzones')

    def update_thumbs(self):
        """add or remove rows in Thumbs layer, according to the thumbs stored in the project"""

        # Copy the geodatabase before the transaction
        assert self.copy_geodb()

        # Get 2 version of the Thumbs layer
        opt_root = Path(self.p_raster_data)
        old = self._thumbs
        new = gpd.GeoDataFrame([gmc_thumb.Thumb(target_path).to_pdserie() for target_path in filter(
            lambda x: gmc_thumb.THUMBNAME_PATTERN.match(x.name), list(opt_root.glob(pattern='**/opticals/*.tif')))
            ]).set_crs(epsg=self.epsg)

        # Comparison
        common = new.merge(old, on=['th_path'])
        stables = old[old.th_path.isin(common.th_path)]
        addeds = new[~new.th_path.isin(common.th_path)]

        # Push it into the geodatabase
        updated = pd.concat([stables, addeds])
        updated.to_file(self.p_geodb, layer='Thumbs')

        # Update instance
        self._thumbs = updated

        return updated

    def update_pairs(self):
        """add or remove rows in Pairs layer, according to the thumbs stored in the project"""

        # Copy the geodatabase before the transaction
        assert self.copy_geodb()

        # For each processing zone 
        # (because the directories are based on the pzones)
        # We update the pairs from what is possibly existing in the displacement directory
        updated = []
        for pz in tqdm.tqdm(self.get_pzones()):
            pairs = pz.get_pairs_overview(session=self)
            for pa in pairs.iloc():
                updated.append(pa)
    
        # Push it into the geodatabase        
        updated = gpd.GeoDataFrame(updated).set_crs(epsg=self.epsg)

        # Fix dtypes problems (ValueError: Invalid field type <class 'numpy.int64'>)
        updated.to_file(self.p_geodb, layer='Pairs')

        # Update instance
        self._pairs = updated
        return updated

    def copy_geodb(self):
        """quickly create a copy of the project geopackage named backup_geodb.gpkg"""
        backup_path = Path(self.p_root, 'backup_geodb.gpkg')
        os.system(f"cp -r {self.p_geodb} {backup_path}")
        return Path(self.p_root, 'backup_geodb.gpkg').exists()

    def sieve(self, data_type='opt', suffix='', res=1, alg='nearest', band=1, delete_existant_thumbs=False, criterias=''):
        """intersect a georasters map layer and the project pzones layer to create thumbs"""

        # TODO : Manage the images  without the same date, but the same year
        # Solution : I use the acq_year attribute of the georasterbanks, which annualize all the dates 
        #   --> 2017-09-05 et 2019-09-06 will be changed into 2019-01-01

        # Check if user have drawn some pzones
        assert len(self.get_pzones_overview()) > 0, 'no pzone registered for this project'

        # Open the corresponding georasters map
        georasters_map = self.get_georasters_map(data_type, suffix)

        # Inform user about CRS differences between georasters map and pzones layer
        print(f"map epsg : {georasters_map.crs}\npz  epsg : {self.get_pzones_overview().crs}")

        # For each processing zone
        for pz in self.get_pzones_overview(criterias).iloc:
            print(f"\n---\n{pz['pz_name']}")

            # Create a directory to store the raster_data of the pzone
            pz_rasterdata = Path(self.p_root, f'raster-data_{self.project_name}', pz['pz_name'])

            # Create subdirectories for the thumbs and the displacements measurements associated
            if data_type == 'opt':
                pz_opticals = Path(pz_rasterdata, 'opticals')
                pz_disps = Path(pz_rasterdata, 'displacements')
                for p in [pz_rasterdata, pz_opticals, pz_disps]:
                    if not p.exists():
                        p.mkdir()

            # Get the images intersecting the zone
            selection = georasters_map[georasters_map['geometry'].intersects(pz['geometry'])]

            # Here we have to regroup the lines of 'selection' where the acquisition date and the sensor are identicals
            # Then we loop on thoses groups
            # And we merge the part of each group

            # Group by acquisition date and sensor
            selection_merged_by_date_and_sensor = selection.groupby(['acq_year', 'sensor'])['filepath'].apply(list)
            
            # For each group
            for group_id, group in enumerate(selection_merged_by_date_and_sensor) :

                acq_year = selection_merged_by_date_and_sensor.index[group_id][0]
                sensor = selection_merged_by_date_and_sensor.index[group_id][1]

                # Define output filename and full path (ex: raster-data_vanoise/sachette/opticals/sachette_2021-03-08_AERIAL.tif)
                if data_type == 'opt':
                    thumb_name = f"{pz['pz_name']}_{acq_year}_{sensor}.tif"
                    thumb_path = str(Path(pz_opticals, thumb_name))

                elif data_type == 'dem':
                    thumb_name = f"{pz['pz_name']}_dem.tif"
                    thumb_path = str(Path(pz_rasterdata, thumb_name))

                # If we don't have the authorisation to delete existing files we go to the next thumb
                if not delete_existant_thumbs and Path(thumb_path).exists():
                    continue

                # For each image in the group, crop it to the processing zone's extent (p_zone).
                # Here, each image receives a nodata value (typically 0 or -9999) set via the 'nodata' parameter in GDAL functions, which GDAL interprets as missing data.
                # This ensures that only the intersecting area between the image and the p_zone is retained, with proper masking.
                fully = []
                print('cropping')
                for mosaic_path in tqdm.tqdm(group):

                    # If we don't have to make a resampling
                    if rt.getPixelSize(mosaic_path)[0] == res:
                        mosaic = rt.Open(
                            target     = mosaic_path,
                            geoExtent  = pz.geometry.bounds,
                            nBands=1)

                    else :
                        mosaic = rt.Open(
                            target     = mosaic_path,
                            geoExtent  = pz.geometry.bounds,
                            nRes       = res,
                            resMethod  = alg,
                            nBands=1)

                    fully.append(mosaic)

                print('merging')
                if len(fully) > 1:
                    thumb = rt.merge(fully)
                else:
                    thumb = fully[0]

                # Write the thumb
                rt.write(thumb, thumb_path)
                # break

    def pr_full(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10):
        """Launch all the possible correlations accross all the valid images among the project pzones"""
        logs = []
        for pzone in self.get_pzones():
            logs.append(pzone.pz_full(self.epsg, corr_algorithm, corr_kernel_size, corr_xthreshold))
        return pd.DataFrame(logs)

    def _search_engine(self, layername, criterias=''):
        """A search engine among project layers"""

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
            case 'Thumbs':
                normal_th = self._thumbs
                lower_th  = normal_th.apply(lambda x: x.str.lower(), 1)
                return normal_th[lower_th.th_path.str.contains(pattern)]
            
            case 'Pairs':
                normal_pa = self._pairs
                lower_pa = normal_pa.apply(lambda x: x.str.lower(), 1)
                return normal_pa[lower_pa.pa_path.str.contains(pattern)]
            
            case 'Pzones':
                pz_layer = self._pzones
                if criterias!=['']:
                    requested_pz_name = criterias[0].lower()
                    return pz_layer[pz_layer.pz_name == requested_pz_name]
                else:
                    return pz_layer
    
    def __repr__(self):
        return f"""
------------
this is a GeoMultiCorr Session open on the project named {self.project_name}
their processing zones are : {self.pz_names}
------------
"""
