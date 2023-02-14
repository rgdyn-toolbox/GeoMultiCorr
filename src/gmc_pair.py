from pathlib import Path
import pandas as pd
from telenvi import raster_tools as rt
import gmc_thumb as gmc_th

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

        # Attributes writing
        self.pa_pz_name = left.th_pz_name
        self.pa_key = f"{self.pa_pz_name}_{left.th_date}-{left.th_sensor}_{right.th_date}-{right.th_sensor}"
        self.pa_path = Path(Path(left.th_path).parent.parent, 'displacements',f"{self.pa_key}")
        self.pa_dispf_path = Path(self.pa_path, 'asp_outputs', f"{self.pa_key}_run-F.tif")
        self.pa_snr_path = Path(self.pa_path, 'asp_outputs', f"{self.pa_key}_SNR-run-F.tif")
        self.pa_magn_path = Path(self.pa_path, f"{self.pa_key}_magn.tif")
        self.pa_left = left
        self.pa_right = right

        if self.pa_path.exists():
            if self.pa_dispf_path.exists():
                pa_status = 'complete'
            else:
                pa_status = 'corrupt'
        else :
            pa_status = 'empty'

        self.pa_status = pa_status
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
# %%
