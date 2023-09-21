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

class Pzone:

    def __init__(self, target_pz_name, session):
        
        # Vérification de la validité du nom de pzone par rapport à la session
        assert target_pz_name in session.pz_names, f'{target_pz_name} not existing in the Pzones layer'
        assert Path(session.p_raster_data, target_pz_name).absolute().exists(), f'no raster data folder for {target_pz_name}'

        # Ecriture attributs
        self.session = session
        self.pz_name = target_pz_name
        self.pz_dem_path = Path(self.session.p_raster_data, self.pz_name, f"{self.pz_name}_dem.tif")

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

    def get_dem(self):
        if self.pz_dem_path.exists():
            return rt.Open(str(self.pz_dem_path), load_data=True)
        else:
            print(f'No dem for pzone {self.pz_name}')
            return False

    def get_complete_pairs(self):
        return [p for p in self.get_pairs() if p.get_status() == 'complete']

    def pz_full(self, epsg, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10, vector_res=20, method='average'):
        logs = {}
        logs['COMPLETE'] = []
        logs['ABORT'] = []
        for p in self.get_valid_pairs():
            try:
                p.pa_full(epsg, corr_algorithm, corr_kernel_size, corr_xthreshold, vector_res, method)
                logs['COMPLETE'].append(p.pa_key)
            except ValueError:
                logs['ABORT'].append(p.pa_key)
                continue
            except AssertionError:
                logs['ABORT'].append(p.pa_key)
                continue
        return logs

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