import os
import re
from pathlib import Path

import tqdm
import pandas as pd

from osgeo import gdal
import geopandas as gpd
from telenvi import raster_tools as rt

import src.geomulticorr.geomorph
import src.geomulticorr.pzone
import src.geomulticorr.pair
import src.geomulticorr.spine
import src.geomulticorr.thumb
import src.geomulticorr.xzone

project_template_location = Path(Path(__file__).parent, 'resources', 'project_template')

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

def open(location):
    return Session(location)

class Session:

    """
    Manipulate data specific to a sample of sites relative to an earth surface displacement study
    """

    def __init__(self, target_root_path : str):
        
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
                src = f"{os.path.join(target_root_path, 'raster-data_template-project')}",
                dst = f"{os.path.join(target_root_path, 'raster-data_' + project_name)}")

            os.rename(
                src = f"{os.path.join(target_root_path, 'geodatabase_template-project.gpkg')}",
                dst = f"{os.path.join(target_root_path, 'geodatabase_' + project_name + '.gpkg')}")

            os.rename(
                src = f"{os.path.join(target_root_path, 'map_template-project.qgz')}",
                dst = f"{os.path.join(target_root_path, 'map_' + project_name + '.qgz')}")

        # Load project data into the current session
        self.p_root = str(target_root_path)
        self.project_name = Path(self.p_root).name
        self.p_raster_data = str(Path(self.p_root, f'raster-data_{self.project_name}'))
        self.p_geodb = os.path.join(target_root_path, f'geodatabase_{self.project_name}.gpkg')

        # There is underscore before the attribute name because 
        # user have to access to this data from the getters
        # to ensure that the data is up to date
        self._pzones = gpd.read_file(self.p_geodb, layer='Pzones')
        self._thumbs = gpd.read_file(self.p_geodb, layer='Thumbs')
        self._geomorphs = gpd.read_file(self.p_geodb, layer='Geomorphs')
        self._pairs  = gpd.read_file(self.p_geodb, layer='Pairs')
        self._xzones = gpd.read_file(self.p_geodb, layer='Xzones')
        self._spines = gpd.read_file(self.p_geodb, layer='Spines')

        # List the processing zones names
        self.pz_names = list(self._pzones.pz_name.unique())

    ########### GETTERS ###########

    def get_thumbs_overview(self, criterias=''):
        """Send a dataframe with informations about each thumb raster file meeting the criterias.
           If there's no criterias, send informations about all the thumbs of the project """
        return self._search_engine('Thumbs', criterias)

    def get_thumbs(self, criterias=''):
        """Send a list of src.geomulticorr.thumb.Thumb objects meeting the criterias"""
        selected_thumbs = self.get_thumbs_overview(criterias)
        return [src.geomulticorr.thumb.Thumb(x.th_path) for x in selected_thumbs.iloc]

    def get_pairs_overview(self, criterias=''):
        """Send a dataframe with each possible Pair according to the Thumbs"""
        return self._search_engine('Pairs', criterias)
    
    def get_pairs(self, criterias=''):
        """Send a list of src.geomulticorr.pair.Pairs objects meeting the criterias"""
        selected_pairs = self.get_pairs_overview(criterias)
        return [src.geomulticorr.pair.Pair(self, target_path = x.pa_path) for x in selected_pairs.iloc]

    def get_pzones_overview(self, pz_name=''):
        """Send a dataframe with each pzone"""
        return self._search_engine('Pzones', pz_name)

    def get_pzones(self, pz_name=''):
        """Send a list of src.geomulticorr.pzone.Pzones objects"""
        selected_pzones = self.get_pzones_overview(pz_name)
        return [src.geomulticorr.pzone.Pzone(x.pz_name, self) for x in selected_pzones.iloc]

    def get_geomorphs_overview(self, criterias=''):
        """Send a dataframe with each geomorph"""
        return self._search_engine('Geomorphs', criterias)

    def get_geomorphs(self, criterias=''):
        """Send a list of src.geomulticorr.geomorph.Geomorphs objects"""
        selected_geomorphs = self.get_geomorphs_overview(criterias)
        return [src.geomulticorr.geomorph.Geomorph(self, x.ge_frogi_id) for x in selected_geomorphs.iloc]

    def get_xzone(self, xz_id):
        """Send a src.geomulticorr.xzones.Xzones object"""
        return src.geomulticorr.xzones.Xzones(self, xz_id)

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
        pairs = [src.geomulticorr.pair.Pair(self, p.pa_path) for p in board.iloc]
        return pairs

    def get_spine(self, sp_id):
        """Send a src.geomulticorr.spine.Spine object"""
        return src.geomulticorr.spine.Spine(self, sp_id)

    def get_protomap(self, rawpath, extensions=['tif', 'jp2']):
        """make a vector layer with the extents of all the rasters stored under the rawpath
           and write metadata in hte layer attribute table"""

        # check the validity of the rawpath
        rawpath = Path(rawpath)
        assert rawpath.exists(), f"{rawpath} is not an existing path"

        # Get a list of all the .tif or .jp2 under rawpath
        targets = []
        for x in extensions:
            targets += rawpath.glob(f'**/*.{x.lower()}')
            targets += rawpath.glob(f'**/*.{x.upper()}')

        # Build a vector layer, with metadata and extent of each image
        features = []
        print('map the images...')
        for ta in tqdm.tqdm(targets):

            # Create empty serie
            ft = gpd.GeoSeries()

            # Write all kind of metadata
            ft["filename"] = ta.name
            ft["sensor"] = re_searcher(str(ta), sensors())
            ft["xRes"],  ft["yRes"] = rt.getPixelSize(str(ta))
            ft["bands"], ft["rows"], ft["cols"] = rt.getShape(str(ta))

            # Get acquisition date : SPOT way
            if 'spot' in ft['sensor'].lower():

                # Here we get the path to the metadata.xml file associated to the image
                md_file_path = gdal.Info(gdal.Open(str(ta))).split('\n')[2]

                # Here we open it
                try:
                    with open(md_file_path.strip(), 'r') as mdf:
                        mds = mdf.read()

                    # Here we track the 'imaging_date' tag and extract his value
                    imaging_date_line = re_searcher(mds, '<IMAGING_DATE>.+</IMAGING_DATE>')
                    ft['acq_date'] = re_searcher(imaging_date_line, '[0-9]+-[0-9]+-[0-9]+')

                except FileNotFoundError:
                    print('metadata file unfound')
                    # ft['acq_date'] = '0000-00-00'
                    continue

            # Get acquisition date : aerial way
            elif 'aerial' in ft['sensor'].lower():
                """
                GMC ISSUE 2
                ############### /!\ 
                IMPLIQUE QUE LES NOMS DES IMAGES AERIENNES SOIENT FORMATEES DE LA MEME MANIERE QUE LES NOTRES 
                ############### /!\ 
                """

                # We split the filename to get the date
                acq_date = ft["filename"].split('-')[1]
                acq_date = f'{acq_date[:4]}-{acq_date[4:6]}-{acq_date[6:]}'
                ft['acq_date'] = acq_date

            # Get the image geographic extent
            ft["geometry"] = rt.drawGeomExtent(str(ta), geomType='shly')
            ft["filepath"] = str(ta)

            # Add the feature to the future layer (just a list for now)
            features.append(ft)

        # Transform the list of GeoSeries features into a GeoDataFrame, and set his CRS
        layer = gpd.GeoDataFrame(features).set_crs(2154)
        return layer

    ########### SETTERS ###########

    def update_vector_data(self):
        """
        Update the instance session attributes from geodatabase layers
        Useful when user modify the data from Qgis
        """
        self._spines = gpd.read_file(self.p_geodb, layer='Spines')
        self._pzones = gpd.read_file(self.p_geodb, layer='Pzones')
        self._geomorphs = gpd.read_file(self.p_geodb, layer='Geomorphs')
        self._xzones = gpd.read_file(self.p_geodb, layer='Xzones')

    def update_thumbs(self):
        """add or remove rows in Thumbs layer, according to the thumbs stored in the project"""

        # Copy the geodatabase before the transaction
        assert self.copy_geodb()

        # Get 2 version of the Thumbs layer
        opt_root = Path(self.p_raster_data)
        old = self._thumbs
        new = gpd.GeoDataFrame([src.geomulticorr.thumb.Thumb(target_path).to_pdserie() for target_path in filter(lambda x: src.geomulticorr.thumb.THUMBNAME_PATTERN.match(x.name), list(opt_root.glob(pattern='**/opticals/*.tif')))])

        # Comparison
        common = new.merge(old, on=['th_path'])
        stables = old[old.th_path.isin(common.th_path)]
        addeds = new[~new.th_path.isin(common.th_path)]

        # Push it into the geodatabase
        updated = pd.concat([stables, addeds])
        updated.to_file(self.p_geodb, layer='Thumbs')

        # Update instance
        self._thumbs = updated

        # Update pairs
        self.update_pairs()

        return updated

    def update_pairs(self):
        """add or remove rows in Pairs layer, according to the thumbs stored in the project"""

        # Copy the geodatabase before the transaction
        assert self.copy_geodb()

        # For each processing zone we
        updated = []
        for pz in self.get_pzones():
            pairs = pz.get_pairs_overview()
            [updated.append(pa) for pa in pairs.iloc()]
    
        # Push it into the geodatabase        
        updated = gpd.GeoDataFrame(updated).set_crs(epsg=2154)
        updated.to_file(self.p_geodb, layer='Pairs')

        # Update instance
        self._pairs = updated
        return updated

    def update_protomap(self, rawpath, protomap=None, extensions=['tif', 'jp2']):
        if protomap == None:
            protomap = self.get_protomap(rawpath, extensions)
        protomap.to_file(self.p_geodb, layer='Protomap')

    def copy_geodb(self):
        """quickly create a copy of the project geopackage named backup_geodb.gpkg"""
        backup_path = Path(self.p_root, 'backup_geodb.gpkg')
        os.system(f"cp -r {self.p_geodb} {backup_path}")
        return Path(self.p_root, 'backup_geodb.gpkg').exists()

    ###############################

    def sieve(self, rawpath='', res=1, alg='nearest', band=1):
        """intersect a protomap layer and the project pzones layer to create thumbs"""

        # Check if user have drawn some pzones
        assert len(self.get_pzones_overview()) > 0, 'no pzone registered for this project'

        # Check if a protomap is registered in the project geodatabase
        try:
            protomap = gpd.read_file(self.p_geodb, layer='Protomap')

        # If not, we build it
        except ValueError:
            protomap = self.get_protomap(rawpath=rawpath)

        # For each processing zone
        for pz in self.get_pzones_overview().iloc:
            print(f"\n---\n{pz['pz_name']}")

            # Create a directory to store the raster_data of the pzone
            pz_rasterdata = Path(self.p_root, f'raster-data_{self.project_name}', pz['pz_name'])
            pz_opticals = Path(pz_rasterdata, 'opticals')
            pz_disps = Path(pz_rasterdata, 'displacements')
            for p in [pz_rasterdata, pz_opticals, pz_disps]:
                if not p.exists():
                    p.mkdir()

            # Get the images intersecting the zone
            selection = protomap[protomap['geometry'].intersects(pz['geometry'])]

            """
            Here we have to regroup the lines of 'selection' where the acquisition date and the sensor are identicals
            Then we loop on thoses groups
            And we merge the part of each group
            """

            # Group by acquisition date and sensor
            selection_merged_by_date_and_sensor = selection.groupby(['acq_date', 'sensor'])['filepath'].apply(list)

            # For each group
            for group_id, group in enumerate(selection_merged_by_date_and_sensor) :

                acq_date = selection_merged_by_date_and_sensor.index[group_id][0]
                sensor = selection_merged_by_date_and_sensor.index[group_id][1]

                # Define output filename and full path (ex: raster-data_vanoise/sachette/opticals/sachette_2021-03-08_AERIAL.tif)
                thumb_name = f"{pz['pz_name']}_{acq_date}_{sensor}.tif"
                thumb_path = str(Path(pz_opticals, thumb_name))

                # We check if the file is already existing
                if not os.path.exists(thumb_path):

                    # We crop each image of the group on the p_zone 
                    # Here, each image receive a number considered as nodata value by gdal
                    # where the pixels are inside the p_zone but outside the image
                    fully = []
                    for mosaic_path in group:

                        # If we don't have to make a resampling
                        if rt.getPixelSize(mosaic_path)[0] == res:
                            mosaic = rt.pre_process(
                                target     = mosaic_path,
                                geoExtent  = pz.geometry.bounds,
                                nBands     = 1)

                        else :
                            mosaic = rt.pre_process(
                                target     = mosaic_path,
                                geoExtent  = pz.geometry.bounds,
                                nBands     = 1,
                                nRes       = res,
                                resMethod  = alg)

                        fully.append(mosaic)

                    # If there is more than one image acquired at the same date and from the same sensor
                    # we accept that it is the initial same image and we merged them together
                    # because gdal considered as nodata the places outside each part, the merged work easily 
                    """
                    TELENVI ISSUE 1
                    ############### but it's not working if we don't extract a band with rt.pre_process()
                    """
                    if len(fully) > 1:
                        thumb = rt.merge(fully)
                    else:
                        thumb = fully[0]

                    # Write the thumb
                    rt.write(thumb, thumb_path)

    def pr_full(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10):
        """Launch all the possible correlations accross all the valid images among the project pzones"""
        logs = []
        for pzone in self.get_pzones():
            logs.append(pzone.pz_full(corr_algorithm, corr_kernel_size, corr_xthreshold))
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

            case 'Geomorphs':
                # We suppose than the criteria can only be a Geomorph ID or a pz_name
                selection = self._geomorphs
                if criterias!=['']:
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
this is a GeoMultiCorr Session open on the project named {self.project_name}
their processing zones are : {self.pz_names}
------------
"""
