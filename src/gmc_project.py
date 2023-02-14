#%%
import os
import re
from pathlib import Path
import geopandas as gpd
import gmc_thumb as gmc_th
import gmc_pzone as gmc_pz
import gmc_pair as gmc_pa
import pandas as pd

ROOT_TEMPLATE = Path(__file__).parent.with_name('template')

def is_conform_to_gmc_template(target_root_path):
    for thing in os.listdir(ROOT_TEMPLATE):
        if not thing in os.listdir(target_root_path):
            return False
    return True

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
        self._pairs  = gpd.read_file(self.p_geodb, layer='Pairs')
        self.pz_names = list(self._pzones.pz_name.unique())

    ########### GETTERS ###########

    def get_thumbs_overview(self, criterias=''):
        """Send a dataframe with informations about each thumb raster file meeting the criterias.
           If there's no criterias, send informations about all the thumbs of the project """
        return self.search_engine('Thumbs', criterias)

    def get_thumbs(self, criterias=''):
        """Send a list of GMC_Thumb objects meeting the criterias"""
        selected_thumbs = self.get_thumbs_overview(criterias)
        return [gmc_th.GMC_Thumb(x.th_path) for x in selected_thumbs.iloc]

    def get_pairs_overview(self, criterias=''):
        """Send a dataframe with each possible Pair according to the Thumbs"""
        return self.search_engine('Pairs', criterias)
    
    def get_pairs(self, criterias=''):
        selected_pairs = self.get_pairs_overview(criterias)
        return [gmc_pa.GMC_Pair(self, target_path = x.pa_path) for x in selected_pairs.iloc]

    def get_pzones_overview(self, pz_name=''):
        return self.search_engine('Pzones', pz_name)

    def get_pzones(self, pz_name=''):
        selected_pzones = self.get_pzones_overview(pz_name)
        return [gmc_pz.GMC_Pzone(x.pz_name, self) for x in selected_pzones.iloc]

    ###############################

    def copy_geodb(self):
        """quickly create a copy of the project geopackage named backup.gpkg"""
        backup_path = Path(self.p_root, 'backup_geodb.gpkg')
        os.system(f"cp -r {self.p_geodb} {backup_path}")
        return len(gpd.read_file(self.p_geodb)) == len(gpd.read_file(backup_path))

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
        return updated

    ###############################

    def search_engine(self, layername, criterias=''):
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

p = GMC_Project('/media/duvanelt/TD002/sandbox_gmc/vanoise')
p.get_pairs_overview()
p.update_pairs()
# %%
