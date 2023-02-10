#%%
import os
import re
from pathlib import Path
import geopandas as gpd
import gmc_thumb
import pandas as pd
from telenvi import raster_tools as rt

ROOT_TEMPLATE = Path(__file__).parent.with_name('template')
THUMBNAME_PATTERN = '^([a-z]|[A-Z]|-)+_[0-9]{4}(-[0-9]{2}){2}_.*.(tif|TIF)$'

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
            print('copie')
            print(target_root_path)
            os.system(f"cp -r {ROOT_TEMPLATE} {target_root_path}")

        self.root_path = Path(target_root_path)
        self.geodb_path = os.path.join(target_root_path, 'database.gpkg')
        self.pz = gpd.read_file(self.geodb_path, layer='Pzones').apply(lambda x: x.str.lower(), 1)
        self.th = gpd.read_file(self.geodb_path, layer='Thumbs')
        self.th['GMC_Thumb'] = self.th.apply(lambda x: gmc_thumb.GMC_Thumb(x), axis=1)
        self.pa = gpd.read_file(self.geodb_path, layer='Pairs').apply(lambda x: x.str.lower(), 1)
        self.ge = gpd.read_file(self.geodb_path, layer='Geomorphs').apply(lambda x: x.str.lower(), 1)

    def backup_db(self):
        backup_path = Path(self.root_path, 'backup_db.gpkg')
        os.system(f"cp -r {self.geodb_path} {backup_path}")
        return len(gpd.read_file(self.geodb_path)) == len(gpd.read_file(backup_path))

    def get_thumbs_infos(self, criterias):
        if type(criterias) in (str, int):
            criterias = [criterias]
        pattern = ""
        for c in criterias:
            if type(c) == int:
                c = str(c)
            pattern += f"(?=.*{c.lower()})"
        lower_th = self.th.apply(lambda x: x.str.lower(), 1)
        return self.th[lower_th.th_path.str.contains(re.compile(pattern))]

    def get_gmc_thumbs(self, criterias):
        thumbs_infos = self.get_thumbs_infos(criterias)
        gmc_thumbs = list(thumbs_infos.GMC_Thumb)
        return gmc_thumbs

    def get_gdf_from_physical_thumbs(self):

        """
        Retourne un GeoDataFrame à partir des fichiers
        rasters de vignettes tels qu'enregistrés sur le disque dur
        """

        opt_root = Path(self.root_path, 'raster_data')
        thumbs = []

        # Pour tous les fichiers tifs du dossier raster_data qui correspondent au pattern THUMBNAME_PATTERN
        for targetpath in filter(lambda x: re.compile(THUMBNAME_PATTERN).match(x.name), list(opt_root.glob(pattern='**/opticals/*.tif'))):

            # On récupère des métadonnées sur cette vignette
            th_path = str(targetpath)
            th_key = targetpath.name.split('.')[0]
            th_pz, th_date, th_sensor = th_key.split('_')
            th_year = int(th_date.split('-')[0])
            th_geom = rt.drawGeomExtent(th_path, geomType='shly')
            th_valid=0
            
            # On écrit tout ça dans une pd_serie
            th_serie = pd.Series({
                'th_path'   : th_path,
                'th_sensor' : th_sensor,
                'th_date'   : th_date,
                'th_year'   : th_year,
                'th_valid'  : th_valid,
                'pz_name'   : th_pz,
                'geometry'  : th_geom
            })

            thumbs.append(th_serie)

        return gpd.GeoDataFrame(thumbs)
    
    def update_thumbs_in_geodb_from_physical(self):

        """
        Met à jour la table Thumbs en fonction
        de l'état des fichiers rasters sur le disque
        """

        # Copy the database before the transaction
        assert self.backup_db()

        # Get 2 version of Thumbs
        ths_phy = self.get_gdf_from_physical_thumbs()
        ths_gdb = self.th.drop(labels='GMC_Thumb', axis=1)

        # Comparison
        common = ths_phy.merge(ths_gdb, on=['th_path'])
        ths_stables = ths_gdb[ths_gdb.th_path.isin(common.th_path)]
        ths_addeds = ths_phy[~ths_phy.th_path.isin(common.th_path)]
        ths_stables_and_news = pd.concat([ths_stables, ths_addeds])

        # Write new Thumbs table in the geotabase
        ths_stables_and_news.to_file(self.geodb_path,layer='Thumbs')

        # Update layer instance
        self.th = ths_stables_and_news

        return ths_stables_and_news        

    def update_pairs_in_geodb_from_thumbs(self):

        """
        Met à jour la table Pairs en fonction
        de l'état de la table Thumbs
        -
        si on ajoute des vignettes il faut d'abord mettre à jour
        la table Thumb ave update_thumbs_in_geodb_from_physical
        """

        # Copy the database before the transaction
        assert self.backup_db()

        # Create an empty list
        pairs_infos = []

        # For each processing zone
        for pz in self.th.pz_name.unique():

            # We get all the thumbs on this pz
            pz_thumbs = self.get_gmc_thumbs(pz)

            # For all images
            for left in pz_thumbs:

                # We make a Pair with all others
                # and we ad it to the list pairs_infos
                for right in pz_thumbs:
                    try:
                        pair = left + right
                        pairs_infos.append(pair.to_pdserie())
        
                    # It's the same image so we skip them
                    except AssertionError:
                        continue

        # Here we convert our list into a geoDataFrame
        new_pairs = gpd.GeoDataFrame(pairs_infos).set_crs(epsg=2154)

        # And we update the geodatabae layer Pairs
        new_pairs.to_file(self.geodb_path, layer='Pairs')
        return new_pairs

p = GMC_Project('/media/duvanelt/TD002/sandbox_gmc/vanoise')
# %%
