#%%
import os
import re
from pathlib import Path
import geopandas as gpd
import gmc_thumb
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

        # Chargement des métadonnées dans des attributs
        self.root_path = Path(target_root_path)
        self.geodb_path = os.path.join(target_root_path, 'database.gpkg')
        self.pz_overview = gpd.read_file(self.geodb_path, layer='Pzones').apply(lambda x: x.str.lower(), 1)
        self.th_overview = gpd.read_file(self.geodb_path, layer='Thumbs')
        self.pa_overview = gpd.read_file(self.geodb_path, layer='Pairs').apply(lambda x: x.str.lower(), 1)
        self.ge_overview = gpd.read_file(self.geodb_path, layer='Geomorphs').apply(lambda x: x.str.lower(), 1)

    def backup_geodb(self):
        backup_path = Path(self.root_path, 'backup_geodb.gpkg')
        os.system(f"cp -r {self.geodb_path} {backup_path}")
        return len(gpd.read_file(self.geodb_path)) == len(gpd.read_file(backup_path))

    def get_thumbs_overview(self, criterias=''):
        return self.search_engine(criterias, 'Thumbs')

    def get_pairs_overview(self, criterias=''):
        return self.search_engine(criterias, 'Pairs')

    def get_thumbs_objects(self, criterias=''):
        local_overview = self.search_engine('Thumbs', criterias)
        return [gmc_thumb.GMC_Thumb(x.th_path) for x in local_overview]

    def _get_pzone_pairs(self, pz_name):
        """Create a list of GMC_Pairs with all possibles pairs for a pzone"""

        pz_pairs = []
        pz_thumbs = [gmc_thumb.GMC_Thumb(th_row.th_path) for th_row in self.get_thumbs(pz_name).iloc]

        # For each thumbs on the pzone
        for left in pz_thumbs:

            # We make a Pair with all others
            for right in pz_thumbs:
                try:
                    pair = left + right
                    pz_pairs.append(pair)
                except AssertionError:
                    continue
        return pz_pairs

    def upd_thumbs(self):
        """add or remove rows in Thumbs layer, according to the thumbs stored in project raster_data directory"""

        # Copy the database before the transaction
        assert self.backup_geodb()

        # Get 2 version of the Thumbs layer
        opt_root = Path(self.root_path, 'raster_data')
        new = gpd.GeoDataFrame([gmc_thumb.GMC_Thumb(target_path).to_pdserie() for target_path in filter(lambda x: gmc_thumb.THUMBNAME_PATTERN.match(x.name), list(opt_root.glob(pattern='**/opticals/*.tif')))])
        old = self.th_overview

        # Comparison
        common = new.merge(old, on=['th_path'])
        stables = old[old.th_path.isin(common.th_path)]
        addeds = new[~new.th_path.isin(common.th_path)]
        
        # Update Thumbs layer instance
        self.th_overview = pd.concat([stables, addeds])

        # Push it in the geodatabase
        self.th_overview.to_file(self.geodb_path, layer='Thumbs')        

        return self.th_overview

    def upd_pairs(self):
        """Build a dataframe with each possible Pair"""

        # Copy the database before the transaction
        assert self.backup_geodb()

        # For each processing zone
        all_pairs = []
        for pz_name in self.th_overview.th_pz_name.unique():
            pz_pairs = [pair.to_pdserie() for pair in self._get_pzone_pairs(pz_name)]
            all_pairs += pz_pairs

        # Update Pairs layer instance
        self.pa = gpd.GeoDataFrame(all_pairs).set_crs(epsg=2154)

        # Push it in the geodatabase
        self.pa.to_file(self.geodb_path, layer='Pairs')
        return self.pa

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
                lower_th = self.th_overview.apply(lambda x: x.str.lower(), 1)
                return self.th_overview[lower_th.th_path.str.contains(pattern)]
            case 'Pairs':
                lower_pa = self.pa_overview.apply(lambda x: x.str.lower(), 1)
                return self.pa_overview[lower_pa.pa_path.str.contains(pattern)]

p = GMC_Project('/media/duvanelt/TD002/sandbox_gmc/vanoise')
d = p.get_pairs('fournache')
d
# %%
