import re
from pathlib import Path

import pandas as pd

from osgeo import gdal
from telenvi import raster_tools as rt

import src.geomulticorr.pair

THUMBNAME_PATTERN = re.compile('^([a-z]|[A-Z]|-)+_[0-9]{4}(-[0-9]{2}){2}_.*.(tif|TIF)$')

class Thumb:

    def __init__(self, target_path):
        target_path = Path(target_path)

        # Vérification de l'existence du fichier pointé par l'adresse 
        # et de la validité du nom au regard du pattern défini
        assert target_path.exists(), 'fichier inexistant'
        assert THUMBNAME_PATTERN.match(target_path.name), 'filename don\'t match with gmc_thumbname pattern'

        self.th_path = str(target_path)
        self.th_key = target_path.name.split('.')[0]
        self.th_pz_name, self.th_date, self.th_sensor = self.th_key.split('_')
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
            'geometry':self.geometry})

    def __add__(self, right):
        return src.geomulticorr.pair.GMC_Pair(left=self, right=right)

    def __repr__(self):
        return f"""---------
type   : GMC_Thumb
pzone  : {self.th_pz_name}
date   : {self.th_date}
sensor : {self.th_sensor}
---------
"""

    def get_ds(self):
        return gdal.Open(str(self.th_path))

    def get_geoim(self):
        return rt.GeoIm.GeoIm(str(self.th_path))

    def show(self):
        self.get_geoim().show()

# %%
