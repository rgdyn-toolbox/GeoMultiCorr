from pathlib import Path
import re
from osgeo import gdal
from telenvi import raster_tools as rt
import pandas as pd
import gmc_pair

THUMBNAME_PATTERN = '^([a-z]|[A-Z]|-)+_[0-9]{4}(-[0-9]{2}){2}_.*.(tif|TIF)$'

class GMC_Thumb:

    def __init__(self, th_serie):

        target_path = Path(th_serie.th_path)

        # Vérification de l'existence du fichier pointé par l'adresse 
        # et de la validité du nom au regard du pattern défini
        assert target_path.exists(), 'fichier inexistant'
        assert re.compile(THUMBNAME_PATTERN).match(target_path.name)

        self.th_path = th_serie.th_path
        self.th_key = target_path.name.split('.')[0]
        self.th_sensor = th_serie.th_sensor
        self.th_date = th_serie.th_date
        self.th_year = th_serie.th_year
        self.th_valid = th_serie.th_valid
        self.th_pz_name = th_serie.pz_name
        self.geometry = th_serie.geometry

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
        return gmc_pair.GMC_Pair(self, right)

    def __repr__(self):
        return f"""---------
type   : GMC_Thumb
pzone  : {self.th_pz_name}
date   : {self.th_date}
sensor : {self.th_sensor}
---------
"""

    def get_ds(self):
        return gdal.Open(self.th_path)

    def show(self):
        self.get_geoim().show()

    def get_geoim(self):
        return rt.GeoIm.GeoIm(self.th_path)
