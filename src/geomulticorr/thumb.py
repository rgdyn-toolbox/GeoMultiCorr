import re
from pathlib import Path

import pandas as pd

from osgeo import gdal
from telenvi import raster_tools as rt

import geomulticorr.pair as gmc_pair

THUMBNAME_PATTERN = re.compile('^([a-z]|[A-Z]|-|[0-9])+_[0-9]{4}(-[0-9]{2}){2}_.*.(tif|TIF)$')

class Thumb:

    def __init__(self, target_path):
        target_path = Path(target_path)

        # Vérification de l'existence du fichier pointé par l'adresse 
        # et de la validité du nom au regard du pattern défini
        assert target_path.exists(), f'{target_path} : fichier inexistant'
        assert THUMBNAME_PATTERN.match(target_path.name), 'filename don\'t match with thumbname pattern'

        # Extract thumb metadatas
        self.th_path = str(target_path)

        # Mechanical extraction of the pixel size, to be sure
        self.th_px_size=rt.getPixelSize(self.th_path)[0]

        # Build the thumb key
        # Add _th_pix_size does not break the thumbname pattern because it fit into the _.* part of the regex pattern
        self.th_key = f"{target_path.name.split('.')[0]}_{self.th_px_size}"
        self.th_pz_name, self.th_date, self.th_sensor = self.th_key.split('_')[:3]
        self.th_year = int(self.th_date.split('-')[0])
        self.geometry = rt.drawGeomExtent(self.th_path, geomType='shly')
        self.th_valid=0

    def to_pdserie(self):
        return pd.Series({
            'th_pz_name':self.th_pz_name,
            'th_path':self.th_path,
            'th_sensor':self.th_sensor,
            'th_date':self.th_date ,
            'th_year':self.th_year,
            'th_valid':self.th_valid,
            'th_px_size':self.th_px_size,
            'geometry':self.geometry})

    def __add__(self, right):
        return gmc_pair.Pair(or_left=self, or_right=right)

    def __repr__(self):
        return f"""---------
type   : GMC_Thumb
pzone  : {self.th_pz_name}
date   : {self.th_date}
sensor : {self.th_sensor}
pxsize : {self.th_px_size}
---------
"""

    def get_ds(self):
        return rt.Open(str(self.th_path))

    def get_geoim(self):
        return rt.Open(str(self.th_path), load_pixels=True)

    def show(self):
        self.get_geoim().show()

# %%
