import os
import re
from pathlib import Path
import tqdm
import geopandas as gpd
import pandas as pd
from osgeo import gdal
from telenvi import raster_tools as rt
import gmc_thumb as gmc_th
import gmc_pzone as gmc_pz
import gmc_pair as gmc_pa
import gmc_geomorph as gmc_ge
import gmc_xzone as gmc_xz
import gmc_spine as gmc_sp

VERSION = '0.0.0'
print(f"""---------
GeoMultiCorr {VERSION}
---------""")

ROOT_TEMPLATE = Path(__file__).parent.with_name('template')

def is_conform_to_gmc_template(target_root_path):
    for thing in os.listdir(ROOT_TEMPLATE):
        if not thing in os.listdir(target_root_path):
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

def create_new(location):
    return GMC_Project(location)

class GMC_Project:

    def __init__(self, target_root_path : str):
        target_root_path = Path(target_root_path)

        # Vérification de la validité de l'adresse
        assert target_root_path.parent.exists(), 'adresse invalide'
 
        # Si elle existe et que c'est pas un projet geomulticorr
        if target_root_path.exists() and not is_conform_to_gmc_template(target_root_path):
            raise ValueError("autre chose existe à cette adresse")

        # Si elle existe et que c'est un projett geomulticorr
        elif target_root_path.exists() and is_conform_to_gmc_template(target_root_path):
            pass

        # Si elle n'existe pas : on la créée
        else:
            os.system(f"cp -r {ROOT_TEMPLATE} {target_root_path}")

        self.p_root = str(target_root_path)
        self.p_raster_data = str(Path(self.p_root, 'raster_data'))
        self.p_geodb = os.path.join(target_root_path, 'database.gpkg')
        self._pzones = gpd.read_file(self.p_geodb, layer='Pzones')
        self._thumbs = gpd.read_file(self.p_geodb, layer='Thumbs')
        self._geomorphs = gpd.read_file(self.p_geodb, layer='Geomorphs')
        self._pairs  = gpd.read_file(self.p_geodb, layer='Pairs')
        self._xzones = gpd.read_file(self.p_geodb, layer='Xzones')
        self._spines = gpd.read_file(self.p_geodb, layer='Spines')
        self._ribs = gpd.read_file(self.p_geodb, layer='Ribs')
        self.pz_names = list(self._pzones.pz_name.unique())

    def update_vector_data(self):
        self._pzones = gpd.read_file(self.p_geodb, layer='Pzones')
        self._geomorphs = gpd.read_file(self.p_geodb, layer='Geomorphs')
        self._xzones = gpd.read_file(self.p_geodb, layer='Xzones')

    ########### GETTERS ###########

    def get_thumbs_overview(self, criterias=''):
        """Send a dataframe with informations about each thumb raster file meeting the criterias.
           If there's no criterias, send informations about all the thumbs of the project """
        return self._search_engine('Thumbs', criterias)

    def get_thumbs(self, criterias=''):
        """Send a list of GMC_Thumb objects meeting the criterias"""
        selected_thumbs = self.get_thumbs_overview(criterias)
        return [gmc_th.GMC_Thumb(x.th_path) for x in selected_thumbs.iloc]

    def get_pairs_overview(self, criterias=''):
        """Send a dataframe with each possible Pair according to the Thumbs"""
        return self._search_engine('Pairs', criterias)
    
    def get_pairs(self, criterias=''):
        """Send a list of GMC_Pairs objects meeting the criterias"""
        selected_pairs = self.get_pairs_overview(criterias)
        return [gmc_pa.GMC_Pair(self, target_path = x.pa_path) for x in selected_pairs.iloc]

    def get_pzones_overview(self, pz_name=''):
        """Send a dataframe with each pzone"""
        return self._search_engine('Pzones', pz_name)

    def get_pzones(self, pz_name=''):
        """Send a list of GMC_Pzones objects"""
        selected_pzones = self.get_pzones_overview(pz_name)
        return [gmc_pz.GMC_Pzone(x.pz_name, self) for x in selected_pzones.iloc]

    def get_geomorphs_overview(self, criterias=''):
        """Send a dataframe with each geomorph"""
        return self._search_engine('Geomorphs', criterias)

    def get_geomorphs(self, criterias=''):
        """Send a list of GMC_Geomorphs objects"""
        selected_geomorphs = self.get_geomorphs_overview(criterias)
        return [gmc_ge.GMC_Geomorph(self, x.ge_frogi_id) for x in selected_geomorphs.iloc]

    def get_xzone(self, xz_id):
        """Send a GMC_Xzone object"""
        return gmc_xz.GMC_Xzones(self, xz_id)

    def get_spine(self, sp_id):
        """Send a GMC_Spine object"""
        return gmc_sp.GMC_Spine(self, sp_id)

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
                ############### /!\ 
                IMPLIQUE QUE LES IMAGES AERIENNES SOIENT NOMMEES DE LA MEME MANIERE QUE LES NOTRES 
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

    ###############################

    def copy_geodb(self):
        """quickly create a copy of the project geopackage named backup_geodb.gpkg"""
        backup_path = Path(self.p_root, 'backup_geodb.gpkg')
        os.system(f"cp -r {self.p_geodb} {backup_path}")
        return Path(self.p_root, 'backup_geodb.gpkg').exists()

    ########### SETTERS ###########

    def update_thumbs(self):
        """add or remove rows in Thumbs layer, according to the thumbs stored in the project"""

        # Copy the database before the transaction
        assert self.copy_geodb()

        # Get 2 version of the Thumbs layer
        opt_root = Path(self.p_raster_data)
        old = self._thumbs
        new = gpd.GeoDataFrame([gmc_th.GMC_Thumb(target_path).to_pdserie() for target_path in filter(lambda x: gmc_th.THUMBNAME_PATTERN.match(x.name), list(opt_root.glob(pattern='**/opticals/*.tif')))])

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

        # Copy the database before the transaction
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
            pz_rasterdata = Path(self.p_root, 'raster_data', pz['pz_name'])
            pz_opticals = Path(pz_rasterdata, 'opticals')
            pz_disps = Path(pz_rasterdata, 'displacements')
            for p in [pz_rasterdata, pz_opticals, pz_disps]:
                if not p.exists():
                    p.mkdir()

            # Get the images intersecting the zone
            selection = protomap[protomap['geometry'].intersects(pz['geometry'])]

            # We have to regroup the lines of 'selection' where the acquisition date and the sensor are identicals
            # Then we loop on thoses groups
            # And we merge the part of each group

            # Group by acquisition date and sensor
            selection_merged_by_date_and_sensor = selection.groupby(['acq_date', 'sensor'])['filepath'].apply(list)

            # For each group
            for group_id, group in enumerate(selection_merged_by_date_and_sensor) :

                acq_date = selection_merged_by_date_and_sensor.index[group_id][0]
                sensor = selection_merged_by_date_and_sensor.index[group_id][1]

                # Define output filename and full path (ex: raster_data/sachette/opticals/sachette_2021-03-08_AERIAL.tif)
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
                    ############### but it's not working if we don't extract a band with rt.pre_process
                    """
                    if len(fully) > 1:
                        thumb = rt.merge(fully)
                    else:
                        thumb = fully[0]

                    # Write the thumb
                    rt.write(thumb, thumb_path)

    def pr_full(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10):
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
                """ on part du principe qu'on ne peut donner comme critere qu'un identifiant de rg ou un pz_name"""
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
this is a GeoMultiCorr Project named {Path(self.p_root).name}
the processing zones registered on it are : {self.pz_names}
------------
"""
