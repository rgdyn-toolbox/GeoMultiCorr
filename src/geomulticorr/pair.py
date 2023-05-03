import os
from pathlib import Path

import numpy as np
import pandas as pd

from telenvi import raster_tools as rt

import src.geomulticorr.thumb

ROOT_OUTPUTS  = Path(__file__).parent.with_name('temp')
ROOT_TEMPLATE = Path(__file__).parent.with_name('template')

class Pair:

    def __init__(self, session=None, target_path=None, left=None, right=None):

        """
        soit session et target_path sont definis
        soit left et right
        """

        # Construction from a pair path
        if type(target_path) in (str, Path):
            target_path = Path(target_path)
            print(target_path)
            assert Path(target_path).parent.exists() and Path(target_path).parent.name.lower() == 'displacements', 'this path is not leading to a well-formed pair'

            # Get metadata about the thumbs component of the pair
            left_year, left_month, left_day, left_sensor = target_path.name.split('_')[1].split('-')
            right_year, right_month, right_day, right_sensor = target_path.name.split('_')[2].split('-')
            left_date = f"{left_year}-{left_month}-{left_day}"
            right_date = f"{right_year}-{right_month}-{right_day}"
            pz_name = target_path.name.split('_')[0]
            left_key = f"{pz_name}_{left_date}_{left_sensor}.tif"
            right_key = f"{pz_name}_{right_date}_{right_sensor}.tif"

            # Reconstruct their path
            thumbs_pzone_path = Path(session.p_raster_data, pz_name, 'opticals')
            left_path = Path (thumbs_pzone_path, left_key)
            right_path = Path(thumbs_pzone_path, right_key)

            # Make Thumbs from them
            left  = src.geomulticorr.thumb.Thumb(left_path)
            right = src.geomulticorr.thumb.Thumb(right_path)

        # Construction from two thumbs
        assert left.th_pz_name == right.th_pz_name, 'left and right have not the same pzone'
        assert left.th_path != right.th_path, 'left thumb is right thumb'

        # Metadata
        self.pa_pz_name = left.th_pz_name
        self.pa_key = f"{self.pa_pz_name}_{left.th_date}-{left.th_sensor}_{right.th_date}-{right.th_sensor}"
        self.pa_left = left
        self.pa_right = right

        # Base path
        self.pa_path = Path(Path(left.th_path).parent.parent, 'displacements',f"{self.pa_key}")

        # Path for the clip outputs and correlation inputs
        self.pa_inputs_path = Path(self.pa_path, 'inputs')

        # Outputs 1
        self.pa_asp_path    = Path(self.pa_path, 'asp_outputs')
        self.pa_dispf_path  = Path(self.pa_asp_path, f"{self.pa_key}_run-F.tif")
        self.pa_snr_path    = Path(self.pa_asp_path, f"{self.pa_key}_SNR-run-F.tif")

        # Outputs 2
        self.pa_magn_path   = Path(self.pa_path, f"{self.pa_key}_magn.tif")
        self.pa_vect_path   = Path(self.pa_path, f"{self.pa_key}_vect.gpkg")

        # Current state of the pair
        self.pa_status = self.get_status()

        # Pair spatial extent
        self.geometry = left.geometry

    def __repr__(self):
        return f"""---------
type   : GMC_Pair
pzone  : {self.pa_pz_name}
left   : {self.pa_left.th_date}-{self.pa_left.th_sensor}
right  : {self.pa_right.th_date}-{self.pa_right.th_sensor}
status : {self.pa_status}
---------
"""

    def get_status(self):
        if self.pa_path.exists():
            if self.pa_dispf_path.exists():
                pa_status = 'complete'
            elif self.pa_inputs_path.exists():
                pa_status = 'clipped'
            else:
                pa_status = 'corrupt'
        else :
            pa_status = 'empty'
        return pa_status

    def to_pdserie(self):
        return pd.Series({
            'pa_pz_name':self.pa_pz_name,
            'pa_path':str(self.pa_path),
            'pa_left_date':self.pa_left.th_date,
            'pa_left_sensor':self.pa_left.th_sensor,
            'pa_right_date':self.pa_right.th_date,
            'pa_right_sensor':self.pa_right.th_sensor,
            'pa_magn_path':str(self.pa_magn_path),
            'pa_dispf_path':str(self.pa_dispf_path),
            'pa_snr_path':str(self.pa_snr_path),
            'pa_status':self.pa_status,
            'geometry':self.geometry})

    def get_magn_geoim(self):
        try:
            return self.pa_magn_geoim
        except AttributeError:
            self.pa_magn_geoim = rt.pre_process(str(self.pa_magn_path), geoim=True)
            return self.pa_magn_geoim

    def get_snr_geoim(self):
        try:
            return self.pa_snr_geoim
        except AttributeError:
            self.pa_snr_geoim = rt.pre_process(str(self.pa_snr_path), geoim=True)
            return self.pa_snr_geoim

    def get_disp_corr_geoim(self):
            self.pa_dispf_geoim = rt.pre_process(str(self.pa_dispf_path), geoim=True, nBands=3)
            return self.pa_dispf_geoim
        
    def get_dispX_geoim(self):
        try:
            return self.pa_dispX_geoim
        except AttributeError:
            self.pa_dispX_geoim = rt.pre_process(str(self.pa_dispf_path), geoim=True, nBands=1)
            return self.pa_dispX_geoim

    def get_dispY_geoim(self):
        try:
            return self.pa_dispY_geoim
        except AttributeError:
            self.pa_dispY_geoim = rt.pre_process(str(self.pa_dispf_path), geoim=True, nBands=2)
            return self.pa_dispY_geoim
    
    def get_vx_geoim(self):
        return self.get_dispX_geoim() / abs(self.pa_left.th_year - self.pa_right.th_year)

    def get_vy_geoim(self):
        return self.get_dispY_geoim() / abs(self.pa_left.th_year - self.pa_right.th_year)

    def get_vmagn_geoim(self):
        return self.get_magn_geoim() / abs(self.pa_left.th_year - self.pa_right.th_year)

    def clip(self, numBand = 1):
        if self.pa_status in ['complete', 'clipped']:
            return True
        cleft = rt.pre_process(
            target = self.pa_left.get_ds(),
            clip   = self.pa_right.get_ds(),
            nBands = numBand)

        cright = rt.pre_process(
            target = self.pa_right.get_ds(),
            clip   = self.pa_left.get_ds(),
            nBands = numBand)

        if not self.pa_path.exists():
            try:
                self.pa_path.mkdir()
            except FileNotFoundError:
                self.pa_path.parent.mkdir()
                self.pa_path.mkdir()

        if not self.pa_inputs_path.exists():
            self.pa_inputs_path.mkdir()

        rt.write(cleft,  os.path.join(self.pa_inputs_path, self.pa_left.th_key  + '_clipped.tif'))
        rt.write(cright, os.path.join(self.pa_inputs_path, self.pa_right.th_key + '_clipped.tif'))
        self.pa_status='clipped'
        return True

    def corr(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10):
        
        # Check clip
        self.clip()

        # Check existing correlation
        if self.pa_status == 'complete':
            return True

        # Create a directory in a place where ASP can write their outputs : the temp directory, inside GeoMultiCorr app
        temp = Path(ROOT_OUTPUTS, self.pa_key)
        if not temp.exists():
            temp.mkdir()

        # Write correlation command
        corr_command = f"parallel_stereo {self.pa_left.th_path} {self.pa_right.th_path} {temp}/{self.pa_key}_run \
            --correlator-mode \
            --threads-multiprocess 8 \
            --processes 1 \
            --stereo-algorithm {corr_algorithm} \
            --corr-kernel {corr_kernel_size} {corr_kernel_size} \
            --xcorr-threshold {corr_xthreshold} \
            --corr-tile-size 2048 \
            --corr-memory-limit-mb 8000 \
            --save-left-right-disparity-difference\
            --ip-per-tile 200 \
            --min-num-ip 5"

        # Launch
        os.system(corr_command)

    def save_corrdata(self, verbose=False):
        """
        Move the ASP outputs from GeoMultiCorr temporal storage location
        """

        # Hide ugly warnings about symlinks
        verbose_mode = {False:' > /dev/null 2>&1', True:''}

        # get the GeoMultiCorr storage location
        departure = Path(ROOT_OUTPUTS, self.pa_key)
        if not departure.exists():
            return None

        # get the displacements folder path in the current session
        destination = self.pa_asp_path

        # send the command to bring back the temporal data in the current session
        os.system(f"mv {departure} {destination} {verbose_mode[verbose]}")
        if self.pa_asp_path.exists():
            os.system(f"rm -rf {departure}")
            return True
    
    def compute_magnitude(self):

        assert self.pa_dispf_path.exists(), 'this pair is not yet correlate'

        # Check if the file is ever existing
        if self.pa_magn_path.exists():
            return True

        # Open the stack with horizontal and vertical displacements
        xDisp, yDisp = rt.pre_process(str(self.pa_dispf_path), nBands=[1,2], geoim=True).splitBands()

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

    def vectorize(self, output_pixel_size=None, method='average', write=True):
        """
        create a points vector layer, mappable as a vector field in Qgis

        output_pixel_size : space between each point. By default, it is the pixel size of the pair disparity map
        method  : algorithm to use to resample the data (is a output_pixel_size value is given)
        """

        # Get vector components - unit = pixels and referential = matrix (Y top is Y min)
        initial_dx = self.get_dispX_geoim()
        initial_dy = self.get_dispY_geoim()
        initial_pixel_size = initial_dx.getPixelSize()[0]

        # Modify the original displacement rasters by resampling them with output resolution asked
        if output_pixel_size != None:
            dx_in_pixels = initial_dx.resize(output_pixel_size, method=method)
            dy_in_pixels = initial_dy.resize(output_pixel_size, method=method)

        # Elsewhere, we will work on a copy of the original displacement rasters
        else:
            dx_in_pixels = initial_dx.copy()
            dy_in_pixels = initial_dy.copy()
            output_pixel_size = initial_pixel_size

        # Get metadata
        _, nRows, nCols = dx_in_pixels.getShape()

        # Convert displacements in meters
        dx_in_meters = dx_in_pixels * initial_pixel_size
        dy_in_meters = dy_in_pixels * initial_pixel_size

        # Switch from matrixian to geographic spatial referential
        dy_in_meters_switched = dy_in_meters * -1

        # Compute vector norm (magnitude) in meters
        d_in_meters = (dx_in_meters ** 2 + dy_in_meters_switched ** 2) **0.5

        # Convert into annual velocity
        time_gap = abs(self.pa_left.th_year - self.pa_right.th_year)
        dx_in_meters_per_year = dx_in_meters / time_gap
        dy_in_meters_per_year = dy_in_meters / time_gap
        d_in_meters_per_year  = d_in_meters  / time_gap

        # Compute displacement geographic direction (vector orientation)
        array_complex_dx_dy = np.apply_along_axis(lambda args: [complex(*args)], 0, [dx_in_meters.array,dy_in_meters_switched.array]).reshape(nCols, nRows)
        array_direction = np.angle(array_complex_dx_dy, deg=True)
        direction = d_in_meters.copy()
        direction.array = array_direction

        # Make a big GeoIm with all this lovely data
        to_vectorize = rt.stack([
            dx_in_pixels,
            dy_in_pixels,
            dx_in_meters,
            dy_in_meters,
            d_in_meters,
            dx_in_meters_per_year,
            dy_in_meters_per_year,
            d_in_meters_per_year,
            direction
        ])

        vectors = rt.vectorize(to_vectorize).set_crs(epsg=2154)
        
        # Write vector layer in geopackage
        if write == True:
            current_vector_layer_name = f"{output_pixel_size}_{self.pa_key}"
            vectors.to_file(self.pa_vect_path, layer=current_vector_layer_name)

            # Copy .qml file
            template_style = 'styles/vector-field_style_1.qml'
            target_style = str(self.pa_vect_path)[:-5]
            cp_command = f"cp {template_style} {target_style}.qml"
            os.system(cp_command)

        return vectors

    def pa_full(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10, vector_res=20, method='average'):

        # Clip
        # self.clip()

        # Corr
        # self.corr(corr_algorithm, corr_kernel_size, corr_xthreshold)

        # Save
        # self.save_corrdata()

        # Magn
        # self.compute_magnitude()

        # Vectors
        self.vectorize(output_pixel_size=vector_res, method=method)

        return True

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
        return self.get_interesting_geoim(mode).inspectRibsAlongSpine(geoLine, ribLength, ribStep, ribOrientation)

# %%
