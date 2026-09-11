import time
import os
from pathlib import Path
from rasterio.features import shapes
import geopandas as gpd
import rasterio
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn import cluster
import math

import telenvi
from telenvi import vector_tools as vt
from telenvi import raster_tools as rt

import geomulticorr.thumb as gmc_thumb

# ASP cannot write his outputs everywhere
# In the temp (for temporary) directory, we know it will be ok
# Then, we move the data (def save_corr_data)
ROOT_OUTPUTS  = Path(__file__).with_name('temp')
if not ROOT_OUTPUTS.exists():
    ROOT_OUTPUTS.mkdir()

def control_print(to_print):
    print(f"CONTROL : {to_print}")

class Pair:

    def __init__(self, session=None, target_path=None, or_left=None, or_right=None, pa_px_size=None, pa_asp_alg=None, pa_corr_kernel=None, pa_sub_pixel_mode=None, pa_user_notes=''):

        """
        Represent a pair of 2 rasters (left and right), having a given pixel size (pa_px_size), and a set of correlation settings. 
        1 Pair = 1 pzone, 1 pa_px_size, and 1 combinaison of correlation settings

        Pair instances can be created as follow :
            Either session and target_path are defined (from the session we identify the path of the pair directory)
            Either left and right (2 Thumbs instances)
        """

        # Default pixel size - the one of the original images
        if pa_px_size is None:
            pa_px_size = 0.5

        # Default and fastest algorithm available in ASP to compute disparity map. 
        # Other choices are : asp_sgm, asp_mgm, opencv_sgbm, libelas, msmw, msmw2, opencv_bm          
        if pa_asp_alg is None:
            pa_asp_alg = 'asp_bm'

        # Correlation kernel size : default or to correct (ASP must receive an odd number)
        if pa_corr_kernel is None:
            pa_corr_kernel=5
        elif pa_corr_kernel % 2 == 0:
            pa_corr_kernel += 1
            control_print(f"Kernel Size must be odd number. 1 has been added to the user-argument. pa_corr_kernel = {pa_corr_kernel}")

        # Subpixel mode - default 1, can be 0 to be deactivated
        # Attention, ne figure pas dans la clé, est implicite si il n'est pas ajouté dans la note
        if pa_sub_pixel_mode is None:
            pa_sub_pixel_mode = 1
            
        if pa_user_notes is None:
            pa_user_notes = ''

        # Construction from a pair path
        # Here we need to reconstruct the attributes from the path of the pair
        if target_path is not None:
            target_path = Path(target_path)
            assert Path(target_path).parent.exists() and Path(target_path).parent.name.lower() == 'displacements', 'this path is not leading to a well-formed pair'

            # Get metadata about the thumbs component of the pair
            left_year, left_month, left_day, left_sensor = target_path.name.split('_')[1].split('-')
            right_year, right_month, right_day, right_sensor = target_path.name.split('_')[2].split('-')
            left_date = f"{left_year}-{left_month}-{left_day}"
            right_date = f"{right_year}-{right_month}-{right_day}"
            pz_name = target_path.name.split('_')[0]
            left_key = f"{pz_name}_{left_date}_{left_sensor}.tif"
            right_key = f"{pz_name}_{right_date}_{right_sensor}.tif"
            pa_px_size, pa_asp_alg, pa_corr_kernel = target_path.name.split('_')[3:6]
            try:
                pa_sup_notes = target_path.name.split('_')[3:6]
            except IndexError:
                pa_sup_notes=None

            # Reconstruct their path
            thumbs_pzone_path = Path(session.p_raster_data, pz_name, 'opticals')
            left_path = Path (thumbs_pzone_path, left_key)
            right_path = Path(thumbs_pzone_path, right_key)

            # Make original Thumbs from them
            or_left  = gmc_thumb.Thumb(left_path)
            or_right = gmc_thumb.Thumb(right_path)

        # Construction from two thumbs
        assert or_left.th_pz_name == or_right.th_pz_name, 'left and right have not the same pzone'
        assert or_left.th_path != or_right.th_path, 'left thumb is right thumb'

        # Metadata
        self.pa_pz_name = or_left.th_pz_name

        # Build the Primary Key of the pair
        self.pa_key = f"{self.pa_pz_name}_{or_left.th_date}-{or_left.th_sensor}_{or_right.th_date}-{or_right.th_sensor}_{pa_px_size}_{pa_asp_alg.replace('_', '-')}_{pa_corr_kernel}"
        if pa_user_notes != '':
            self.pa_key = f"{self.pa_key}_{pa_user_notes}"

        # Store the individuals attributes in instance variables
        self.pa_or_left = or_left
        self.pa_or_right = or_right

        # Correlation metadata
        self.pa_px_size = pa_px_size
        self.pa_asp_alg = pa_asp_alg.replace('_', '-')
        self.pa_corr_kernel = pa_corr_kernel
        self.pa_sub_pixel_mode = pa_sub_pixel_mode
        self.pa_user_notes = pa_user_notes

        # Base path
        self.pa_path = Path(Path(or_left.th_path).parent.parent, 'displacements',f"{self.pa_key}").absolute()
        self.pa_name = self.pa_path.name

        # Path for the clip outputs and correlation inputs
        self.pa_inputs_path = Path(self.pa_path, 'inputs').absolute()

        # Outputs 1
        self.pa_asp_path    = Path(self.pa_path, 'asp_outputs')
        self.pa_dispf_path  = Path(self.pa_asp_path, f"{self.pa_key}_run-F.tif")
        self.pa_snr_path    = Path(self.pa_asp_path, f"{self.pa_key}_corr-eval-ncc.tif")
        self.pa_perf_file_path = Path(self.pa_path, f"{self.pa_key}_perfs.log")

        # Outputs 2
        self.pa_magn_path   = Path(self.pa_path, f"{self.pa_key}_magn.tif")
        self.pa_vect_path   = Path(self.pa_path, f"{self.pa_key}_vect.gpkg")

        # Current state of the pair
        self.pa_status = self.get_status()

        # Pair spatial extent
        self.geometry = or_left.geometry

        # Check if the real thumbs exists and if the pre-processed has been made
        if self.get_status() in ['complete', 'pre-processed']:
            pa_left_path, pa_right_path = self.get_pp_thumb_paths()
            self.pa_left = gmc_thumb.Thumb(pa_left_path)
            self.pa_right = gmc_thumb.Thumb(pa_right_path)

        else:
            self.pa_left = None
            self.pa_right= None

        # Decorate some processing functions
        self.corr = telenvi.measure_time(self.pa_perf_file_path)(self.corr)
        self.corr_eval = telenvi.measure_time(self.pa_perf_file_path)(self.corr_eval)

    def rename_pa_perf_files(self):
        previous_path = Path(self.pa_path, f"{self.pa_key}_time-perf.txt")
        new_path = self.pa_perf_file_path
        os.system(f'mv {previous_path} {new_path}')

    def __repr__(self):
        return f"""---------
type          : GMC_Pair
pzone         : {self.pa_pz_name}
pixel size (m): {self.pa_px_size}
asp algorithm : {self.pa_asp_alg}
corr kernel   : {self.pa_corr_kernel}
left          : {self.pa_or_left.th_date}-{self.pa_or_left.th_sensor}
right         : {self.pa_or_right.th_date}-{self.pa_or_right.th_sensor}
status        : {self.pa_status}
---------
"""

    def get_pp_thumb_paths(self):
        thumbs_paths = [
            os.path.join(self.pa_inputs_path, Path(self.pa_or_left.th_path).name  + f'_{self.pa_px_size}_pp.tif'),
            os.path.join(self.pa_inputs_path, Path(self.pa_or_right.th_path).name  + f'_{self.pa_px_size}_pp.tif')]
        return thumbs_paths

    def get_status(self):
        """
        Check if the pair has been correlated or not
        """
        if self.pa_path.exists():
            if self.pa_dispf_path.exists():
                pa_status = 'complete'

            elif self.pa_inputs_path.exists():
                if len(list(self.pa_inputs_path.glob('*.tif'))) == 2:
                    pa_status = 'pre-processed'
                else:
                    pa_status='empty'
            else:
                pa_status = 'corrupt'
        else :
            pa_status = 'empty'

        return pa_status

    def to_pdserie(self):

        # Update correlation scores
        self.pa_corr_t, self.pa_corr_eval_t = self.get_pa_perfs()
        
        return pd.Series({
                    'pa_key':self.pa_key,
                    'pa_pz_name':self.pa_pz_name,
                    'pa_path':str(self.pa_path),
                    'pa_left_date':self.pa_or_left.th_date,
                    'pa_left_sensor':self.pa_or_left.th_sensor,
                    'pa_right_date':self.pa_or_right.th_date,
                    'pa_right_sensor':self.pa_or_right.th_sensor,
                    'pa_magn_path':str(self.pa_magn_path),
                    'pa_dispf_path':str(self.pa_dispf_path),
                    'pa_snr_path':str(self.pa_snr_path),
                    'pa_status':self.get_status(),
                    'pa_vect_path':str(self.pa_vect_path),
                    'pa_left_year':str(self.pa_or_left.th_year),
                    'pa_right_year':str(self.pa_or_right.th_year),
                    'pa_corr_kernel':int(self.pa_corr_kernel),
                    'pa_asp_alg':self.pa_asp_alg,
                    'pa_user_notes':self.pa_user_notes,
                    'pa_corr_t':float(self.pa_corr_t),
                    'pa_corr_eval_t':float(self.pa_corr_eval_t),
                    'pa_px_size':float(self.pa_px_size),
                    'geometry':self.geometry})

    def get_magn_geoim(self):
        try:
            return self.pa_magn_geoim
        except AttributeError:
            self.pa_magn_geoim = rt.Open(str(self.pa_magn_path), load_pixels=True)
            return self.pa_magn_geoim

    def get_snr_geoim(self):
        try:
            return self.pa_snr_geoim
        except AttributeError:
            self.pa_snr_geoim = rt.Open(str(self.pa_snr_path), load_pixels=True)
            return self.pa_snr_geoim

    def get_disp_corr_geoim(self):
            self.pa_dispf_geoim = rt.Open(str(self.pa_dispf_path), load_pixels=True, nBands=3)
            return self.pa_dispf_geoim
        
    def get_dispX_geoim(self):
        try:
            return self.pa_dispX_geoim
        except AttributeError:
            self.pa_dispX_geoim = rt.Open(str(self.pa_dispf_path), load_pixels=True, nBands=1)
            return self.pa_dispX_geoim

    def get_dispY_geoim(self):
        try:
            return self.pa_dispY_geoim
        except AttributeError:
            self.pa_dispY_geoim = rt.Open(str(self.pa_dispf_path), load_pixels=True, nBands=2)
            return self.pa_dispY_geoim
    
    def get_vx_geoim(self):
        """
        TODO : Change the time referential into decimal time
        """
        return self.get_dispX_geoim() / abs(self.pa_left.th_year - self.pa_right.th_year)

    def get_vy_geoim(self):
        return self.get_dispY_geoim() / abs(self.pa_left.th_year - self.pa_right.th_year)

    def get_vmagn_geoim(self):
        return self.get_magn_geoim() / abs(self.pa_left.th_year - self.pa_right.th_year)

    def get_disp_vectors(self, point_density=10):
        """
        send the disp field in the form of a geodataframe points layer
        """
        layername = f"{point_density}_{self.pa_key}"
        return gpd.read_file(self.pa_vect_path, layer=layername)

    ### Creation of the displacement fields

    def pre_process(self):

        # Check if pre processings have already be done
        if self.pa_status in ['complete', 'pre-processed']:
            return True

        # Create the Pair directories
        if not self.pa_path.exists():
            try:
                self.pa_path.mkdir()
            except FileNotFoundError:
                self.pa_path.parent.mkdir()
                self.pa_path.mkdir()
        if not self.pa_inputs_path.exists():
            self.pa_inputs_path.mkdir()

        # Resample the images to the pair pixel size as datasets
        print(f'resampling thumbs from {self.pa_or_left.th_px_size} to {self.pa_px_size}')
        pa_left_ds = rt.Open(self.pa_or_left.th_path, nRes=self.pa_px_size)
        pa_right_ds = rt.Open(self.pa_or_right.th_path, nRes=self.pa_px_size)

        # Define the Path of the pre_processed thumbs
        pa_left_path, pa_right_path = self.get_pp_thumb_paths()

        # Write the images in the pair directory
        print(f"Write the new thumbs to {pa_left_path} and {pa_right_path}")
        rt.write(pa_left_ds, pa_left_path)
        rt.write(pa_right_ds, pa_right_path)

        # Update the instance attributes by adding pre_processed thumbs in pa_left and pa_right (the real ones)
        print("pa_left and pa_right available")
        self.pa_left = gmc_thumb.Thumb(pa_left_path)
        self.pa_right = gmc_thumb.Thumb(pa_right_path)

        # Update the pair instance status
        self.pa_status='pre-processed'

        return True
    
    def corr(self, blank_shot=True):

        # Check pa status
        if self.get_status() == 'empty':
            raise ValueError(f'{self.pa_key} not materialized.\n Processing methods are locked.')
        
        # Check clip
        # self.clip()

        # Check existing correlation
        control_print(self.get_status())

        # Check if already correlated
        if self.get_status() == 'complete':
            return True 

        # Create a directory in a place where ASP can write their outputs : the temp directory, inside GeoMultiCorr app
        temp = Path(ROOT_OUTPUTS, self.pa_key)
        control_print(f'temp dir = {temp}')
        if not temp.exists():
            temp.mkdir()
            control_print(f'{temp} created')

        # Write correlation command
        corr_command = f'parallel_stereo {self.pa_left.th_path} {self.pa_right.th_path} {temp}/{self.pa_key}_run \
            --correlator-mode \
            --stereo-algorithm {self.pa_asp_alg.replace("-", "_")} \
            --corr-kernel {self.pa_corr_kernel} {self.pa_corr_kernel} \
            --xcorr-threshold 10 \
            --corr-memory-limit-mb 8000 \
            --save-left-right-disparity-difference\
            --ip-per-tile 2000 \
            --min-num-ip 10 \
            --subpixel-mode {self.pa_sub_pixel_mode} \
            --threads-multiprocess 3 \
            --alignment-method none' # Désactive parce que nos images sont déjà géoréférencées

        # Total CPU = num-threads (threads par tile) * threads-multiprocess (nb de tiles)
        # J'en ai 12 donc soit 12 * 1 pour petites images soit 4 * 3 pour grandes images
        # OMP_NUM_THREADS est défini dans mon bashrc car mon ASP ne supporte pas --num-threads
        # Sur la même zone avec les paarmètres de corr identiques je suis passé de 160 à 120 secondes de corrélation

        # Print the command
        control_print(corr_command)

        # Launch
        if not blank_shot:
            os.system(corr_command)
        
        return corr_command

    def corr_eval(self, eval_corr_kernel=None, metric='ncc'):
        """
        Appel de la fonction d'ASP pour générer un raster SNR
        Par défaut, la taille du noyau d'évaluation est la même que celle de la corrélation
        """

        if eval_corr_kernel is None:
            eval_corr_kernel = self.pa_corr_kernel

        # Check if the snr is already existing for this pair
        if self.pa_snr_path.exists():
            return self.get_snr_geoim()

        # Get Left and Right normalized thumbs
        left_p  = Path(self.pa_asp_path, f"{self.pa_key}_run-L.tif")
        right_p = Path(self.pa_asp_path, f"{self.pa_key}_run-R.tif")

        # Get Run-F path
        disp_p = self.pa_dispf_path

        # Build output suffix
        suffix = str(Path(self.pa_asp_path, f"{self.pa_key}_corr-eval"))

        # Build command
        corr_eval_command = f"corr_eval {left_p} {right_p} {disp_p} {suffix}\
        --kernel-size {eval_corr_kernel} {eval_corr_kernel}\
        --metric {metric}\
        --prefilter-mode 2" # Used by Amaury in his command but... I don't know why

        # Launch
        os.system(corr_eval_command)

        # Copy .qml file for set a default style in Qgis
        template_style = Path(Path(__file__).parent, 'resources', 'map_styles', 'corr-eval_style.qml')
        target_style = str(self.pa_snr_path)[:-4]
        cp_command = f"cp {template_style} {target_style}.qml"
        os.system(cp_command)

        return corr_eval_command

    def save_corrdata(self, verbose=False):
        """
        Move the ASP outputs from GeoMultiCorr temporal storage location
        """

        # Hide ugly warnings about symlinks
        verbose_mode = {False:' > /dev/null 2>&1', True:''}

        # get the GeoMultiCorr temp storage location
        departure = Path(ROOT_OUTPUTS, self.pa_key)

        # get the displacements folder path in the current session
        destination = self.pa_asp_path

        # send the command to bring back the temporal data in the current session
        os.system(f"mv {departure} {destination} {verbose_mode[verbose]}")

        # If the transfer have worked, we delete the data in the temp dir
        if self.pa_asp_path.exists():
            os.system(f"rm -rf {departure}")
            return True

        return False

    def del_useless_data(self):
        """
        Delete a lot of files within the ASP_outputs directory
        """
        n_files_before = len(list(self.pa_asp_path.glob('**/*')))

        # For each element within asp_directory
        for asp_element in self.pa_asp_path.glob('**/*'):

            # We only want to keep run-F and corr-eval (snr) files
            if not 'run-f' in str(asp_element).lower() and not 'corr-eval' in str(asp_element).lower() :
                os.system(f"rm -r {asp_element}")

        # Count number of files deleted
        n_files_after = len(list(self.pa_asp_path.glob('**/*')))
        n_del_files = n_files_before-n_files_after
        print(f"{n_del_files} files deleted for {self.pa_key}")

        if n_del_files > 0:
            return True
        else:
            return False

    def compute_magnitude(self):

        assert self.pa_dispf_path.exists(), 'this pair is not yet correlate'

        # Check if the file is ever existing
        if self.pa_magn_path.exists():
            return self.get_magn_geoim()

        # Open the stack with horizontal and vertical displacements
        xDisp, yDisp = rt.Open(str(self.pa_dispf_path), nBands=[1,2], load_pixels=True).splitBands()

        # We switch values using negative values because ASP gives displacements 
        # in pixel coordinates using as reference upper-left corner
        yDisp *= -1.0

        # Get Metadata
        pxSizeX,_ = xDisp.getPixelSize()

        # Magnitude raster creation
        magn = ((xDisp ** 2 + yDisp ** 2) ** 0.5)

        # Magnitude in meters
        magn_in_meters = magn * pxSizeX
        magn_in_meters.save(str(self.pa_magn_path))

        return magn_in_meters

    def vectorize(self, epsg, output_pixel_size=10, method='average', write=True, crop='', cropFeatureNum=0, integrate_snr=True):
        """
        create a points vector layer, mappable as a vector field in Qgis

        output_pixel_size : space between each point. By default, it is the pixel size of the pair disparity map
        method  : algorithm to use to resample the data (is a output_pixel_size value is given)
        """

        # Get vector components - unit = pixels and referential = matrix (Y top is Y min)
        if crop == '':
            initial_dx = self.get_dispX_geoim()
            initial_dy = self.get_dispY_geoim()
        else:
            disp = rt.Open(self.pa_dispf_path, geoExtent=crop, featureNum=cropFeatureNum, load_pixels=True, nBands=[1,2])
            initial_dx, initial_dy = disp.splitBands()

        initial_pixel_size_x, initial_pixel_size_y = initial_dx.getPixelSize()

        # Modify the original displacement rasters by resampling them with output resolution asked
        if output_pixel_size != None:
            dx_in_pixels = initial_dx.resize(output_pixel_size, method=method)
            dy_in_pixels = initial_dy.resize(output_pixel_size, method=method)

        # Elsewhere, we will work on a copy of the original displacement rasters
        else:
            dx_in_pixels = initial_dx.copy()
            dy_in_pixels = initial_dy.copy()
            output_pixel_size = initial_pixel_size_x

        # Get metadata
        _, nRows, nCols = dx_in_pixels.getShape()

        # Convert displacements in meters
        dx_in_meters = dx_in_pixels * initial_pixel_size_x
        dy_in_meters = dy_in_pixels * initial_pixel_size_y

        # Switch the displacements if the pair is reversed
        if self.pa_left.th_year > self.pa_right.th_year :
            dx_in_meters *= -1
            dy_in_meters *= -1

        # Compute vector norm (magnitude) in meters
        d_in_meters = (dx_in_meters ** 2 + dy_in_meters ** 2) **0.5

        # Convert into annual velocity
        """
        Ajouter les années en décimal pour convertir les vitesses, plus précis qu'en base 365 jours par 365 jours
        """
        time_gap = abs(self.pa_left.th_year - self.pa_right.th_year)
        dx_in_meters_per_year = dx_in_meters / time_gap
        dy_in_meters_per_year = dy_in_meters / time_gap
        d_in_meters_per_year  = d_in_meters  / time_gap

        # Compute displacement geographic direction (vector orientation)
        array_direction = np.degrees(np.arctan2(dx_in_meters_per_year.array, dy_in_meters_per_year.array))
        direction = d_in_meters.copy()
        direction.array = array_direction

        # Get the Signal Noise Ratio
        snr = self.get_snr_geoim().resize(output_pixel_size, method=method)

        # Make a big GeoIm with all this lovely dataset
        to_vectorize = rt.stack([
            dx_in_pixels,
            dy_in_pixels,
            dx_in_meters,
            dy_in_meters,
            d_in_meters,
            dx_in_meters_per_year,
            dy_in_meters_per_year,
            d_in_meters_per_year,
            direction,
            snr
        ])

        # Use the vectorize raster_tools function to make a vector point on each pixel
        # with an attribute column with the pixel value for each band (here we got 9 band)
        vectors = rt.vectorize(to_vectorize, mode='points').set_crs(epsg=epsg)
        vectors.columns = [
                'dx_in_pixels',
                'dy_in_pixels',
                'dx_in_meters',
                'dy_in_meters',
                'd_in_meters',
                'dx_in_meters_per_year',
                'dy_in_meters_per_year',
                'd_in_meters_per_year',
                'direction',
                'snr',
                'geometry']

        # Write vector layer in geopackage
        if write == True:
            current_vector_layer_name = f"{output_pixel_size}_{self.pa_key}"
            vectors.to_file(self.pa_vect_path, layer=current_vector_layer_name)

            # Copy .qml file
            template_style = Path(Path(__file__).parent, 'resources', 'map_styles', 'vectors-field_rgik-ka-compatible.qml')
            target_style = str(self.pa_vect_path)[:-5]
            cp_command = f"cp {template_style} {target_style}.qml"
            os.system(cp_command)

        return vectors

    def get_pa_perfs(self):
        """
        Reads the performance file and returns the corr timings.
        Read only the 2 first lines, corresponding to the first call. 
        Else, it counts the 0.0000xxx sec when we call the function but the pair is already correlated.
        """
        try:
            with open(self.pa_perf_file_path, "r") as f:
                log_lines = f.readlines()[:2]
                corr_t, corr_eval_t = [float(line.split(':')[-1].strip()[:-2]) for line in log_lines]
            return corr_t, corr_eval_t

        except FileNotFoundError:
            return 0, 0

    def pa_full(self, epsg=2056, vector_res=10, method='average', metric_eval='ncc'):

        # Corr
        self.corr(blank_shot=False)

        try:

            # Save
            self.save_corrdata()

            # Eval
            self.corr_eval(eval_corr_kernel=self.pa_corr_kernel, metric=metric_eval)
            
            # Magn
            self.compute_magnitude()

            # Vectors
            if not self.pa_vect_path.exists():
                self.vectorize(epsg, output_pixel_size=vector_res, method=method)

            return True

        except AssertionError:
            return False

    ### Analyze of the displacement fields
    def get_moving_areas_from_kmeans(self, n_clusters=2, mode='m', save=True):

        """
        Make a segmentation of a displacement 
        raster with K-Means clustering 
        """

        outpath = Path(self.pa_path, f"KMe_N{n_clusters}_{self.pa_key}.tif")

        # Check if the raster is already existing
        if outpath.exists():
            return rt.geoim.Geoim(outpath)

        # Get the interesting GeoIm
        disp = self.get_interesting_geoim(mode)

        # Extract his array
        disp_ar = disp.array

        # Reshape for the clustering
        disp_arX = disp_ar.reshape(-1,1)

        # Create the classifier
        k_means_classifier = cluster.KMeans(n_clusters=n_clusters, n_init=10)

        # Fit to the data
        k_means_classifier.fit(disp_arX)

        # Get the labels
        clusters_labels = k_means_classifier.labels_

        # re-switch the classified vector as image (2D array)
        cluster_disp_ar = clusters_labels.reshape(disp_ar.shape)

        # Assign this array to a new geoim
        cluster_disp = disp.copy()
        cluster_disp.array = cluster_disp_ar

        # Save it
        if save:
            cluster_disp.save(str(outpath))

        return cluster_disp

    def denoise_moving_areas(self, operator_size=30, n_clusters=2, mode='m', save=True):
        """
        Create new raster of moving areas, normally with less noise
        """

        # Build output filepath
        outpath = Path(self.pa_path, f"KMe_N{n_clusters}_Un-{operator_size}_{self.pa_key}.tif")

        # Build a morphological operator
        operator = np.ones((operator_size, operator_size))

        # Get the moving areas from displacement field by k-means clustering
        mas = self.get_moving_areas(n_clusters, mode)

        # Extract the array and convert it compatible with the operator
        mas_ar = mas.array.astype('uint8')

        # Denoise
        mas_denoised_ar = cv.morphologyEx(mas_ar, cv.MORPH_OPEN, operator)

        # Build a new geoim and change his array
        mas_denoised = self.get_moving_areas().copy()
        mas_denoised.array = mas_denoised_ar

        # Save
        if save:
            mas_denoised.save(str(outpath))

        return mas_denoised

    def vectorize_moving_areas(self, epsg=21781, min_surf = '', n_clusters=4, mode='m'):
        """
        Create a geopackage layer with the moving areas outlines
        """
        mask = None
        with rasterio.Env():
            target_path = str(Path(self.pa_path, f"KMe_N{n_clusters}_{self.pa_key}.tif"))
            with rasterio.open(target_path) as src:
                image = src.read(1) # first band
                results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    shapes(image, mask=mask, transform=src.transform)))
        geoms = list(results)
        gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms).set_crs(epsg=epsg)
        if min_surf != '':
            gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster.area / 1000 > min_surf]
            output_path = str(Path(self.pa_path, f"{self.pa_key}_classif-{n_clusters}_min-surf-{min_surf}.gpkg"))
        else:
            output_path = str(Path(self.pa_path, f"{self.pa_key}_classif-{n_clusters}.gpkg"))
        gpd_polygonized_raster.to_file(output_path)

    def get_interesting_geoim(self, mode):
        match mode.lower():
            case 'm':
                target = self.get_magn_geoim()
            case 'x':
                target = self.get_dispX_geoim()
            case 'y':
                target = self.get_dispY_geoim()            
            case 'vm':
                target = self.get_vmagn_geoim()
            case 'vx':
                target = self.get_vx_geoim()
            case 'vy':
                target = self.get_vy_geoim()
        return target

    def get_slice(self, geoLine, mode='m'):
        """
        geoline = [gpd.GeoSeries, path, gpd.GeoDataFrame, shapely.geometry.LineString, (A, B)]
        mode = [m, x, y, vm, vx, vy]
        """
        return self.get_interesting_geoim(mode).inspectGeoLine(geoLine)

    def get_slices(self, geoLine, ribLength, ribStep, ribOrientation='v', mode='m'):
        return self.get_interesting_geoim(mode).inspectRibsAlongthumb(geoLine, ribLength, ribStep, ribOrientation)

    def _get_snr_med_in_row(self, snr, row):
        snr.maskFromVector(row.geometry, 2056)
        snr_med = snr.median()
        snr.unmask()
        return snr_med

    def _get_med_magn_in_row(self, magn, row):
        magn.maskFromVector(row.geometry, 2056)
        magn_med = magn.median()
        magn.unmask()
        return magn_med

    def to_rogi(self, rogi, read_magn=False):
        """
        Read the optical correlation data and compute indexes in each rock glacier 
        """

        # Read pairs rasters
        snr = self.get_snr_geoim()
        magn = self.get_interesting_geoim('m')

        # Patch SNR nodata to 0, else, it falsify the median
        patched_snr = snr.array
        patched_snr[patched_snr < 0] = 0
        snr.array = patched_snr

        # Open all the rock glaciers within the pzone area
        # Sel.geometry is the zone geometry
        rogi_in_zone = vt.spatial_selection(rogi, gpd.GeoDataFrame([{'geometry':self.geometry}]))

        # Write the SNR score into the rogi
        rows = []
        for rg in rogi_in_zone.iloc:
            
            # Write metadata of the pair
            row = self.to_pdserie()

            # Write data of the rock glacier
            for col in rogi_in_zone.columns:
                row[col] = rogi_in_zone[col]

            # Now, go for the analyses of the displacement field in the rock glacier

            # 1) SNR
            rgu_pa_snr_med = self._get_snr_med_in_row(snr, rg)
            if rgu_pa_snr_med < 0:
               rgu_pa_snr_med = 0
            row['rgu_pa_snr_med'] = rgu_pa_snr_med 

            # 2) Score based on the directionality of the disp field compared to the slope
            # TODO

            # 3) Velocity values



            # Add the rgu_pa_data to our dataframe
            rows.append(row)

        return gpd.GeoDataFrame(rows)

# %%
