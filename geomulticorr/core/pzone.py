from pathlib import Path
from tqdm import tqdm 

import itertools
import rasterio
import geoutils as gu
import geopandas as gpd
from telenvi import raster_tools as rt
from rasterio.features import shapes

import cv2 as cv
import numpy as np
from sklearn import cluster

import geomulticorr.core.thumb as gmc_thumb

# * Works properly
def print_infoBM(text: str,
                bold: bool = False) -> None:
    """Prints a formatted message to the console.

    Args:
        text (str): The string to be printed.
        bold (bool, optional): Whether to print the text in bold style. Defaults to False.
    """
    GMC_TEXT = "[ GMC-info ] :"
    if bold:
        print(f"\033[1m{GMC_TEXT} {text}\033[1m")
    else:
        print(f"{GMC_TEXT} {text}")
#END def

class Pzone:

    def __init__(self, target_pz_name, session):
        # Verifying the validity of the pzone name in relation to the session
        assert target_pz_name in session.pz_names, f'{target_pz_name} not existing in the Pzones layer'
        assert Path(session.path_raster_data, target_pz_name).absolute().exists(), f'no raster data folder for {target_pz_name}'
        # Writing attributes
        self.session = session
        self.pz_name = target_pz_name
        self.pz_dem_path = Path(self.session.path_raster_data, self.pz_name, f"{self.pz_name}_dem.tif")

    def get_thumbs_overview(self, criterias=''):
        criterias = [criterias] + [self.pz_name]
        return self.session.get_thumbs_overview(criterias)

    def get_thumbs(self, criterias=''):
        criterias = [criterias] + [self.pz_name]
        return self.session.get_thumbs(criterias)

    def get_pairs_overview(self, criterias=''):
        #! Conflict with name in session class.
        pairs = gpd.GeoDataFrame([pa.to_pdserie() for pa in self.get_pairs()])
        return pairs

    def get_valid_pairs_overview(self):
        pairs = gpd.GeoDataFrame([pa.to_pdserie() for pa in self.get_valid_pairs()])
        return pairs

    def get_pairs(self):
        """
        Builds all ordered pairs (A,B) and (B,A) from all thumbs.
        Uses permutations to include both directions.
        """
        thumbs = self.get_thumbs()
        pairs = []
        for left, right in itertools.permutations(thumbs, 2):
            try:
                pairs.append(left + right)
            except AssertionError:
                continue
        return pairs
    
    def get_pairs_with_strategy(
        self,
        strategy: str = "consecutive",
        max_step: int | None = None,
        max_dt_days: int | None = None,
    ) -> list:
        """Return Pair objects for valid thumbs using the specified pairing strategy.

        Args:
            strategy: One of ``"consecutive"``, ``"step"``, ``"redundancy"``.
            max_step: Required for ``"step"`` and ``"redundancy"``. Maximum index
                distance between paired thumbs.
            max_dt_days: Optional upper bound on the absolute date difference
                (in days). Pairs exceeding this are dropped.

        Returns:
            List of Pair objects.
        """
        thumbs = sorted(self.get_valid_thumbs(), key=lambda t: t.th_date_datetime)
        n = len(thumbs)
        pairs_idx = []

        if strategy == "consecutive":
            pairs_idx = [(i, i + 1) for i in range(n - 1)]

        elif strategy == "step":
            if max_step is None:
                raise ValueError("'step' strategy requires max_step")
            pairs_idx = [
                (i, j)
                for i in range(n)
                for j in range(i + 1, min(i + max_step + 1, n))
            ]

        elif strategy == "redundancy":
            if max_step is None:
                raise ValueError("'redundancy' strategy requires max_step")
            seen = set()
            for i in range(n):
                for offset in range(1, max_step + 1):
                    for a, b in [(i, i + offset), (i + offset, i)]:
                        if 0 <= a < n and 0 <= b < n and (a, b) not in seen:
                            pairs_idx.append((a, b))
                            seen.add((a, b))

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Valid options: 'consecutive', 'step', 'redundancy'."
            )

        pairs = []
        for i, j in pairs_idx:
            try:
                pair = thumbs[i] + thumbs[j]
            except AssertionError:
                continue
            if max_dt_days is not None and pair.pa_dt_days > max_dt_days:
                continue
            pairs.append(pair)
        return pairs

    def get_valid_pairs_with_strategy_overview(
        self,
        strategy: str = "consecutive",
        max_step: int | None = None,
        max_dt_days: int | None = None,
    ) -> gpd.GeoDataFrame:
        pairs = self.get_pairs_with_strategy(strategy, max_step, max_dt_days)
        return gpd.GeoDataFrame([p.to_pdserie() for p in pairs])


    def get_valid_thumbs(self):
        """
        Renvoie les vignettes selectionnées par l'user dans qgis, en modifiant la valeur attributaire "th_valid" dans la table Thumbs
        """
        ths = self.session.get_thumbs_overview(self.pz_name)
        ths_valid = ths[ths.th_valid == '1']
        gmc_ths_valid = [gmc_thumb.Thumb(th.th_path) for th in ths_valid.iloc]
        return gmc_ths_valid

    def get_valid_pairs(self):
        valid_thumbs = self.get_valid_thumbs()
        pairs = []
        for left, right in itertools.permutations(valid_thumbs, 2):
            try:
                pairs.append(left + right)
            except AssertionError:
                continue
        return pairs

    def get_dem(self):
        if self.pz_dem_path.exists():
            return gu.Raster(str(self.pz_dem_path), load_data=True)
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
        outpath = Path(self.session.path_raster_data, self.pz_name, f"{self.pz_name}_moving-areas_denoised-{operator_size}_round-0.tif")

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
            with rasterio.open(str(Path(self.session.path_raster_data, self.pz_name, f"{self.pz_name}_moving-areas_round-0.tif"))) as src:
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
            gpd_polygonized_raster.to_file(str(Path(self.session.path_raster_data, self.pz_name, f"{self.pz_name}_moving-areas_round-0.gpkg")), layer=f"{self.pz_name}_moving-areas_round-0_features-sup-{min_surf}")
        else:
            gpd_polygonized_raster.to_file(str(Path(self.session.path_raster_data, self.pz_name, f"{self.pz_name}_moving-areas_round-0.gpkg")), layer=f"{self.pz_name}_moving-areas_round-0")
    
    # ! To be implemented in pzone.
    def update_pairs_with_strategy(self,
                                   strategy: str = "consecutive",
                                   fixed_step: int = 3) -> gpd.GeoDataFrame:
        updated_frames = []
        for pz in self.get_pzones():
            print_infoBM(f"Updating pairs for PZone: '{pz.pz_name}'")
            pairs = pz.get_pairs_with_strategy(strategy, fixed_step)         # GeoDataFrame
            if not pairs.empty:
                updated_frames.append(pairs)
        
        if updated_frames:
            updated = pd.concat(updated_frames, ignore_index=True)
            if "geometry" not in updated.columns:
                raise ValueError("No 'geometry' column found in pairs DataFrame.")
            updated = gpd.GeoDataFrame(updated, geometry="geometry", crs=self.epsg)
        else:
            updated = gpd.GeoDataFrame(geometry=[], crs=self.epsg)
        
        # updated = (gpd.GeoDataFrame(pd.concat(updated_frames, ignore_index=True))
        #    .set_crs(epsg=self.epsg)) if updated_frames else gpd.GeoDataFrame(crs=self.epsg)
        updated.to_file(self.path_geodb, layer="Pairs")
        self._pairs = updated
        return updated