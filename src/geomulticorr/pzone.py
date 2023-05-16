from pathlib import Path

import geopandas as gpd

try:
    import geomulticorr.thumb as gmc_thumb
except ModuleNotFoundError:
    import src.geomulticorr.thumb as gmc_thumb

class Pzone:

    def __init__(self, target_pz_name, session):
        
        # Vérification de la validité du nom de pzone par rapport à la session
        assert target_pz_name in session.pz_names, f'{target_pz_name} not existing in the Pzones layer'
        assert Path(session.p_root, session.p_raster_data, target_pz_name).exists(), f'no raster data folder for {target_pz_name}'

        # Ecriture attributs
        self.session = session
        self.pz_name = target_pz_name

    def get_thumbs_overview(self, criterias=''):
        criterias = [criterias] + [self.pz_name]
        return self.session.get_thumbs_overview(criterias)

    def get_thumbs(self, criterias=''):
        criterias = [criterias] + [self.pz_name]
        return self.session.get_thumbs(criterias)

    def get_pairs_overview(self, criterias=''):
        pairs = gpd.GeoDataFrame([pa.to_pdserie() for pa in self.get_pairs()])
        return pairs

    def get_pairs(self):
        """
        Ne va pas chercher l'info dans les paires existantes mais la reconstruit à partir
        de l'état courant du layer thumbs. Sinon, pas de mise à jour possible de la base
        des vignettes !
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

    def get_valid_thumbs(self):
        """
        Renvoie les vignettes selectionnées par l'user dans qgis, en modifiant la valeur attributaire "th_valid" dans la table Thumbs
        """
        ths = self.session.get_thumbs_overview(self.pz_name)
        ths_valid = ths[ths.th_valid=='1']
        gmc_ths_valid = [gmc_thumb.Thumb(th.th_path) for th in ths_valid.iloc]
        return gmc_ths_valid

    def get_valid_pairs(self):
        ps = []
        for left in self.get_valid_thumbs():
            for right in self.get_valid_thumbs():
                try:
                    ps.append(left+right)
                except AssertionError:
                    continue
        return ps
    
    def get_complete_pairs(self):
        session = session.update_pairs()
        return [p for p in self.get_pairs() if p.get_status() == 'complete']

    def pz_full(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10, vector_res=20, method='average'):
        logs = {}
        logs['COMPLETE'] = []
        logs['ABORT'] = []
        for p in self.get_valid_pairs():
            try:
                p.pa_full(corr_algorithm, corr_kernel_size, corr_xthreshold, vector_res, method)
                logs['COMPLETE'].append(p.pa_key)
            except ValueError:
                logs['ABORT'].append(p.pa_key)
                continue
            except AssertionError:
                logs['ABORT'].append(p.pa_key)
                continue
        return logs