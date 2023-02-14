from pathlib import Path
import geopandas as gpd

class GMC_Pzone:

    def __init__(self, target_pz_name, project):
        
        # Vérification de la validité du nom de pzone par rapport au projet
        assert target_pz_name in project.pz_names, f'{target_pz_name} not existing in the Pzones layer'
        assert Path(project.p_root, 'raster_data', target_pz_name).exists(), f'no raster data folder for {target_pz_name}'

        # Ecriture attributs
        self.proj = project
        self.pz_name = target_pz_name

    def get_thumbs_overview(self, criterias=''):
        criterias = [criterias] + [self.pz_name]
        return self.proj.get_thumbs_overview(criterias)

    def get_thumbs(self, criterias=''):
        criterias = [criterias] + [self.pz_name]
        return self.proj.get_thumbs(criterias)

    def get_pairs_overview(self, criterias=''):
        pairs = gpd.GeoDataFrame([pa.to_pdserie() for pa in self.get_pairs()])
        return pairs

    def get_pairs(self):
        """
        Ne va pas chercher l'info dans les paires existantes mais la reconstruit à partir
        de l'état courant du layer thumbs. Sinon, pas de mise à jour possible de la base
        des des vignettes !
        """
        pairs = []
        thumbs = self.get_thumbs()
        for left in thumbs:
            for right in thumbs:
                try:
                    pairs.append(left + right)
                except AssertionError:
                    continue
        return pairs