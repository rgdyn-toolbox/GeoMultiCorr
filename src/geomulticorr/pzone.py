import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm 

import rasterio
import geopandas as gpd
from telenvi import raster_tools as rt
from rasterio.features import shapes

import cv2 as cv
import numpy as np
from sklearn import cluster

import geomulticorr.thumb as gmc_thumb
import geomulticorr.pair as gmc_pair

class Pzone:

    def __init__(self, target_pz_name, session):
        
        # Vérification de la validité du nom de pzone par rapport à la session
        assert target_pz_name in session.pz_names, f'{target_pz_name} not existing in the Pzones layer'
        # assert Path(session.p_raster_data, target_pz_name).absolute().exists(), f'no raster data folder for {target_pz_name}'

        # Ecriture attributs
        self.session = session
        self.pz_name = target_pz_name

        # Pzone directory
        self.pz_dir_path  = Path(self.session.p_raster_data, self.pz_name)

        # Directory with opticals thumbs
        self.pz_thumbs_path = Path(self.pz_dir_path, 'opticals')

        # Directory with displacements fields
        self.pz_disp_dir_path = Path(self.pz_dir_path, 'displacements')

        # Correlation reports
        self.pz_corr_report_path = Path(self.pz_disp_dir_path, f'corr_report_{self.pz_name}.csv')

        # Optionnal : dem
        self.pz_dem_path = Path(self.pz_dir_path, f"{self.pz_name}_dem.tif")

    def get_thumbs_overview(self, criterias=''):
        criterias = [criterias] + [self.pz_name+'_']
        return self.session.get_thumbs_overview(criterias)

    def get_thumbs(self, criterias=''):
        criterias = [criterias] + [self.pz_name+'_']
        return self.session.get_thumbs(criterias)

    def get_pairs_overview(self, criterias='', session=None):
        pairs = gpd.GeoDataFrame([pa.to_pdserie() for pa in self.get_pairs(criterias, session=session)])
        return pairs

    def get_custom_pairs(self):
        existing_pz_pairs_dir = [target_path for target_path in list(self.pz_disp_dir_path.glob('*')) if target_path.is_dir()]
        pairs = [gmc_pair.Pair(session=self.session, target_path=pa_path) for pa_path in existing_pz_pairs_dir]
        return pairs

    def get_pairs(self, criterias='', allow_reversed=True, only_complete=False, pa_px_size=None, pa_asp_alg=None, pa_corr_kernel=None, pa_user_notes=None, session=None):
        """
        Ne va pas chercher l'info dans les paires existantes mais la reconstruit à partir
        de l'état courant du dossier displacements. Sinon, pas de mise à jour possible de la base
        des paires créées entre temps, avec de nouveaux paramètres de corrélation ou de taille d'images.

        Le comportement par défaut donne toutes les paires possibles avec les paramètres par défaut.
        Et si on a créé des paires indépendamment, en changeant les paramètres par défaut (taille de pixels et paramètres de corrélation),
        alors ces paires sont détectées à partir des dossiers existants. Ca signifie qu'il n'est pas possible de récupérer une paire
        avec des paramètres customisés qui n'aurait pas été corrélé ou en tout cas dont la corrélation n'aurait pas été tentée, donc sans dossier existant.
        """

        # D'abord, si une session est renseignée,
        # on construit toutes les paires qui n'ont pas été créées à partir 
        # des paramètres de corrélation par défaut. 
        pairs = self.get_custom_pairs()

        # TODO : remove and clean the old system, creating the theorical pairs with all 
        # the same default corrparameters

        # # Puis on ajoute toutes celles avec les paramètres par défaut
        # thumbs = self.get_thumbs()
        # for left in thumbs:
        #     for right in thumbs:
        #         try:
        #             pairs.append(left + right)
        #         except AssertionError:
        #             continue

        # # Filters based on the meta criterias
        # if criterias != '':
        #     if type(criterias) != list:
        #         criterias=[criterias]

        #     for p in pairs:

        #         for criteria in criterias:
        #             criteria=str(criteria)

        #             if criteria in p.pa_name:
        #                 if p.pa_path not in [x.pa_path for x in pairs]:
        #                     pairs.append(p)
        
        # Only the correlated pairs
        if only_complete == True:
            pairs = [p for p in pairs if p.get_status() == 'complete']

        # Get only the "left / right" chronological ordered paired
        if allow_reversed == False:
            pairs = [p for p in pairs if p.pa_left.th_year < p.pa_right.th_year]

        # Filters based on the correlation parameters
        if pa_px_size is not None:
            pairs = [p for p in pairs if p.pa_px_size == pa_px_size]
        if pa_asp_alg is not None:
            pairs = [p for p in pairs if p.pa_asp_alg == pa_asp_alg]
        if pa_corr_kernel is not None:
            pairs = [p for p in pairs if p.pa_corr_kernel == pa_corr_kernel]
        if pa_user_notes is not None:
            pairs = [p for p in pairs if p.pa_user_notes == pa_user_notes]

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

    def get_dem(self):
        if self.pz_dem_path.exists():
            return rt.Open(str(self.pz_dem_path), load_data=True)
        else:
            print(f'No dem for pzone {self.pz_name}')
            return False

    def get_complete_pairs(self):
        return [p for p in self.get_pairs() if p.get_status() == 'complete']

    # Analyze the displacement fields on the pzone

    def get_moving_areas(self, n_clusters=2, mode='m'):
        """
        Build a clustered map for each completed pair of the pzone
        """
        mas = []
        for pa in tqdm(self.get_complete_pairs()):
            mas.append(pa.get_moving_areas(n_clusters, mode))
        return mas

    def add_moving_areas(self, n_clusters=2, mode='m'):
        """
        Make a global addition of the moving areas on the pzone
        """
        
        # Get all the moving areas
        mas = self.get_moving_areas(n_clusters, mode)

        # Extract the first - it will be the base
        basic = mas[0]

        # For each other moving area geoim
        for ma in mas[3:]:

            # We check if the basic geoim and the current
            # have exactly the same shape
            if ma.getShape() != basic.getShape():

                # We clip them together
                basic = basic.cropFromRaster(ma)
                ma = ma.cropFromRaster(basic)

            # Check the numeric type
            ma.array = ma.array.astype('uint8')
            basic += ma

        return basic

    def cluster_addition(self):
        def cluster_geoim(target, n_clusters=2):

            # Extract his array
            target_ar = target.array

            # Reshape for the clustering
            target_arX = target_ar.reshape(-1,1)

            # Create the classifier
            k_means_classifier = cluster.KMeans(n_clusters=n_clusters, n_init=10)

            # Fit to the data
            k_means_classifier.fit(target_arX)

            # Get the labels
            clusters_labels = k_means_classifier.labels_

            # re-switch the classified vector as image (2D array)
            cluster_target_ar = clusters_labels.reshape(target_ar.shape)

            # Assign this array to a new geoim
            cluster_target = target.copy()
            cluster_target.array = cluster_target_ar

            return cluster_target
        x = cluster_geoim(self.add_moving_areas())
        return x

    def denoise_moving_areas(self, operator_size=30, n_clusters=2, mode='m', save=True):
        """
        Create new raster of the cumul of the moving areas, normally with less noise
        """

        # Build output filepath
        outpath = Path(self.session.p_raster_data, self.pz_name, f"{self.pz_name}_moving-areas_denoised-{operator_size}_round-0.tif")

        # Build a morphological operator
        operator = np.ones((operator_size, operator_size))

        # Get the moving areas from displacement field by k-means clustering
        mas = self.cluster_addition()

        # Extract the array and convert it compatible with the operator
        mas_ar = mas.array.astype('uint8')

        # Denoise
        mas_denoised_ar = cv.morphologyEx(mas_ar, cv.MORPH_CLOSE, operator)

        # Build a new geoim and change his array
        mas_denoised = self.get_thumbs()[0].get_geoim().copy()
        mas_denoised.array = mas_denoised_ar

        # Save
        if save:
            mas_denoised.save(str(outpath))

        return mas_denoised

    def vectorize_multitemporal_moving_areas(self, epsg, min_surf = '', operator_size=30, n_clusters=2, mode='m'):
        mask = None
        with rasterio.Env():
            with rasterio.open(str(Path(self.session.p_raster_data, self.pz_name, f"{self.pz_name}_moving-areas_round-0.tif"))) as src:
                image = src.read(1) # first band
                results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    shapes(image, mask=mask, transform=src.transform)))
        geoms = list(results)
        gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms).set_crs(epsg=epsg)
        gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster.raster_val == 1]
        if min_surf != '':
            gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster.area / 1000 > min_surf]
            gpd_polygonized_raster.to_file(str(Path(self.session.p_raster_data, self.pz_name, f"{self.pz_name}_moving-areas_round-0.gpkg")), layer=f"{self.pz_name}_moving-areas_round-0_features-sup-{min_surf}")
        else:
            gpd_polygonized_raster.to_file(str(Path(self.session.p_raster_data, self.pz_name, f"{self.pz_name}_moving-areas_round-0.gpkg")), layer=f"{self.pz_name}_moving-areas_round-0")

    def pz_multi_corr(self, left_year, right_year, pa_pixel_sizes, corr_kernel_sizes, asp_algs):
        """
        create several correlations, with different pixel sizes, kernel sizes, and correlation algorithms
        """

        report = []

        # For each configuration
        for pxs in pa_pixel_sizes:
            for ks in corr_kernel_sizes:
                for alg in asp_algs:
                    
                    # Create a custom Pair instance
                    target_pa = self.session.create_custom_pair(
                        self.pz_name,
                        left_year=left_year,
                        right_year=right_year,
                        pa_px_size=pxs,
                        pa_asp_alg=alg,
                        pa_corr_kernel=ks)

                    # Pre-processings
                    target_pa.pre_process()

                    # Processings
                    try:

                        # Correlation and post-processings
                        target_pa.pa_full()

                    except (AttributeError, AssertionError):
                        continue
                    
                    # Get Metadata
                    report_row = target_pa.to_pdserie()
                    report.append(report_row)

        # Make the report
        try:
            updated_report = self.get_corr_report()
            updated_report = updated_report.drop_duplicates(subset=['pa_key'])
            return updated_report
        except AttributeError:
            return None

    def get_corr_report(self):

        # Récupérer les rapports des paires créées
        complete_pairs = self.get_complete_pairs()
        
        # S'il y a une ou plusieurs paires
        if len(complete_pairs) > 0:
            updated_report = pd.DataFrame([pa.to_pdserie() for pa in complete_pairs])
            updated_report.to_csv(self.pz_corr_report_path)
            return updated_report
        else:
            return None
        
    def show_boxplot_corr_times(self):
        ax=sns.boxplot(
            data=self.get_corr_report(),
            y='pa_corr_t',
            x='pa_px_size',
            hue='pa_asp_alg'
        )
        return ax

    def show_barplot_corr_times(self):
        ax=sns.barplot(
            data=self.get_corr_report(),
            y='pa_corr_t',
            x='pa_px_size',
            hue='pa_asp_alg'
        )
        return ax