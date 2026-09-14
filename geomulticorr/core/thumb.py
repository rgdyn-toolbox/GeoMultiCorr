# ---------------------------------------------------------------------------- #
#  Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# thumb.py
# creation date: 2026-04-14.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------- #

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from osgeo import gdal
import geoutils as gu

import geomulticorr.core.pair as gmc_pair

THUMBNAME_PATTERN = re.compile(
    "^([a-z]|[A-Z]|-|[0-9])+_[0-9]{4}(-[0-9]{2}){2}_.*.(tif|TIF)$"
)


class Thumb:
    """
    Represents a single optical satellite image (thumbnail) in a processing zone.
    
    A Thumb is a georeferenced raster file containing a satellite image from a specific
    sensor and date. Thumbs are the fundamental building blocks for creating image pairs
    and performing correlation analysis.
    
    The filename must match the pattern: ``<pzone>_YYYY-MM-DD_<sensor>.tif``
    """

    def __init__(self, target_path):
        """
        Initialize a Thumb object from a raster file path.
        
        Validates the filename against the expected pattern and extracts metadata
        (processing zone name, acquisition date, and sensor name) from the filename.
        Computes the footprint geometry and decimal year for temporal analysis.
        
        :param target_path: Path to the optical image file (must be GeoTIFF format).
        :type target_path: str or Path
        
        :raises AssertionError: If the file does not match the expected filename pattern.
        
        :ivar th_path: Full path to the raster file.
        :ivar th_key: Filename without extension (e.g., ``pzone_YYYY-MM-DD_sensor``).
        :ivar th_pz_name: Processing zone name extracted from filename.
        :ivar th_date: Acquisition date in ISO format (``YYYY-MM-DD``).
        :ivar th_sensor: Sensor name (e.g., ``SPOT6``, ``Sentinel2``).
        :ivar th_year: Year of acquisition as integer.
        :ivar th_date_dec: Decimal year for temporal interpolation.
        :ivar th_date_datetime: Acquisition date as pandas Timestamp.
        :ivar geometry: Shapely geometry of the raster footprint.
        :ivar th_valid: Validation flag (0 = not validated).
        :ivar th_res: Ground sample distance in CRS units, read from the file
            header; ``nan`` when it cannot be read.
        """
        target_path = Path(target_path)

        # Vérification de l'existence du fichier pointé par l'adresse
        # et de la validité du nom au regard du pattern défini
        # assert target_path.exists(), 'fichier inexistant'
        assert THUMBNAME_PATTERN.match(
            target_path.name
        ), "filename don't match with thumbname pattern"

        ras = gu.Raster(str(target_path))

        self.th_path = str(target_path)
        self.th_key = target_path.name.split(".")[0]
        self.th_pz_name, self.th_date, self.th_sensor = self.th_key.split("_")
        self.th_year = int(self.th_date.split("-")[0])
        self.th_date_dec = self.th_year + (int(self.th_date.split("-")[1]) - 1) / 12
        self.th_date_datetime = pd.to_datetime(self.th_date)
        self.geometry = ras.footprint.ds.geometry.union_all()
        self.th_valid = 0
        # Ground sample distance, in the CRS' units (metres for a projected CRS).
        #
        # Read from the file header, which is authoritative: it is the pixel size
        # ASP will actually see.  Recording ``sieve_bulk``'s ``target_resolution``
        # instead would be wrong for every thumb registered by
        # ``register_existing_thumbs``, which copies pixels byte-for-byte and
        # never goes through the sieve.
        #
        # Always a real float (``nan`` on failure, never a string or None): the
        # Thumbs layer is rebuilt by concatenating rows read back from the GPKG
        # with rows from ``to_pdserie``, and a mixed dtype there would silently
        # turn the column to ``object`` — the trap ``th_year``/``th_valid``
        # already fell into.
        self.th_res = self._read_resolution(ras)
    #END def

    @staticmethod
    def _read_resolution(ras) -> float:
        """Pixel size of *ras* in CRS units, or ``nan`` when unreadable.

        Degrades rather than raises, matching
        :meth:`~geomulticorr.core.session.Session._thumb_resolution`: a thumb
        with an odd header should still register, just without a resolution, so
        one bad file cannot block ``update_thumbs`` for a whole project.

        :param ras: An open :class:`geoutils.Raster`.
        :returns: Pixel size along x, as a positive float.
        """
        try:
            return abs(float(ras.res[0]))
        except (AttributeError, IndexError, TypeError, ValueError):
            return float("nan")

    def to_pdserie(self):
        """
        Convert Thumb metadata to a pandas Series.
        
        Serializes all thumbnail attributes into a pandas Series for easy integration
        with GeoDataFrames or tabular analyses.
        
        :return: Series containing thumb metadata (date, sensor, coordinates, etc.).
        :rtype: pd.Series
        """
        return pd.Series(
            {
                "th_pz_name": self.th_pz_name,
                "th_path": self.th_path,
                "th_sensor": self.th_sensor,
                "th_date": self.th_date,
                "th_year": self.th_year,
                "th_date_dec": self.th_date_dec,
                "th_date_datetime": self.th_date_datetime,
                "th_valid": self.th_valid,
                "th_res": float(self.th_res),
                "geometry": self.geometry,
            }
        )
    #END def

    def __add__(self, right):
        """
        Create an image pair by adding two Thumbs together.
        
        Overloads the ``+`` operator to create a Pair object from two Thumb instances.
        Syntax: ``pair = thumb_a + thumb_b``
        
        :param right: The second Thumb to pair with this one.
        :type right: Thumb
        
        :return: A Pair object containing both thumbs.
        :rtype: gmc_pair.Pair
        """
        return gmc_pair.Pair(left=self, right=right)
    #END def

    def __repr__(self):
        """
        Return a formatted string representation of the Thumb.
        
        Displays key metadata (processing zone, date, sensor) in a readable format.
        
        :return: Formatted string representation.
        :rtype: str
        """
        return f"""---------
type   : GMC_Thumb
pzone  : {self.th_pz_name}
date   : {self.th_date}
sensor : {self.th_sensor}
---------
"""

    def get_ds(self):
        """
        Open and return the GDAL dataset for this Thumb.
        
        Provides access to the underlying GDAL raster for advanced geospatial operations.
        
        :return: GDAL dataset object.
        :rtype: osgeo.gdal.Dataset
        """
        return gdal.Open(str(self.th_path))
    #END def

    def get_raster(self):
        """
        Load and return the Thumb as a geoutils Raster object.
        
        Provides convenient access to the raster array and metadata through the
        geoutils library for analysis and manipulation.
        
        :return: Raster object wrapping the thumbnail image.
        :rtype: geoutils.Raster
        """
        return gu.Raster(str(self.th_path))
    #END def

    def show_info(self):
        """
        Display detailed information about the Thumb raster.
        
        Prints metadata such as dimensions, coordinate system, bands, and data type.
        
        :return: None
        """
        self.get_raster().info()
    #END def

    def plot(self):
        """
        Display a visualization of the Thumb raster.
        
        Renders the image as an interactive or static plot depending on the
        environment (Jupyter notebooks produce interactive plots).
        
        :return: None
        """
        self.get_raster().plot()
    #END def