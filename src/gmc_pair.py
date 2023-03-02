import os
from pathlib import Path
import pandas as pd
from telenvi import raster_tools as rt
import gmc_thumb as gmc_th
ROOT_OUTPUTS  = Path(__file__).parent.with_name('_temp')

class GMC_Pair:

    def __init__(self, project=None, target_path=None, left=None, right=None):

        """
        soit project et target_path sont definis
        soit left et right
        """

        # Construction from a pair path
        if type(target_path) in (str, Path):
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

            # Reconstruct their path
            thumbs_pzone_path = Path(project.p_raster_data, pz_name, 'opticals')
            left_path = Path (thumbs_pzone_path, left_key)
            right_path = Path(thumbs_pzone_path, right_key)

            # Make Thumbs from them
            left =  gmc_th.GMC_Thumb(left_path)
            right = gmc_th.GMC_Thumb(right_path)

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
        return rt.pre_process(str(self.pa_magn_path), geoim=True)

    def get_snr_geoim(self):
        return rt.pre_process(str(self.pa_snr_path), geoim=True)
    
    def get_dispf_geoim(self):
        return rt.pre_process(str(self.pa_dispf_path), geoim=True)
    
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

    def corr(self, corr_algorithm=2, corr_kernel_size=7, corr_xthreshold=10, active=False):
        
        # Check clip
        self.clip()

        # Check existing correlation
        if self.pa_status == 'completed':
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
        if active:
            os.system(corr_command)

    def move_corrdata(self, verbose=False):
        verbose_mode = {False:' > /dev/null 2>&1', True:''}
        departure = Path(ROOT_OUTPUTS, self.pa_key)
        destination = self.pa_asp_path
        if not departure.exists():
            return None
        os.system(f"mv {departure} {destination} {verbose_mode[verbose]}")
        if self.pa_asp_path.exists():
            os.system(f"rm -rf {departure}")
            return True
    
    def compute_magnitude(self):

        if not self.pa_dispf_path.exists():
            self.corr()
            self.move_corrdata()

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
# %%
