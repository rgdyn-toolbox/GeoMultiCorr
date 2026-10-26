"""DEM source inference from file names (supported_sensors.infer_dem_sensor)."""
from __future__ import annotations

import pathlib

import pytest

from geomulticorr.correlation.supported_sensors import (
    DEM_SENSOR_FALLBACK,
    DEM_SENSOR_PATTERNS,
    infer_dem_sensor,
)


@pytest.mark.parametrize("name, expected", [
    ("site_2019-08-14_pleiades-DEM.tif", "pleiades"),
    ("PHR1A_20190814_dsm.tif", "pleiades"),
    ("pneo3_2023_dem.tif", "pleiades"),
    ("SPOT6_2015-09-01_dem.tif", "spot6"),
    ("spot7_20170901_DSM.tif", "spot7"),
    ("WV02_20180704_dem.tif", "worldview"),
    ("IGN_LiDARHD_2022-07-01.tif", "lidar"),
    ("als_2012_dtm.tif", "lidar"),
    ("RGEALTI_2019.tif", "lidar"),
    ("uav_2021-08-03_sfm.tif", "uav"),
    ("drone_20200810.tif", "uav"),
    ("aerial_1978-09-12_dem.tif", "aerial"),
    ("historical_1953_dsm.tif", "aerial"),
    ("dem_2020-01-01.tif", DEM_SENSOR_FALLBACK),
])
def test_filename_patterns(name, expected):
    assert infer_dem_sensor(pathlib.Path("/bank") / name) == expected


def test_lidar_wins_over_ign_aerial():
    """Order matters: IGN distributes lidar too."""
    assert infer_dem_sensor("IGN_lidar_2021.tif") == "lidar"


def test_falls_back_to_parent_folder():
    assert infer_dem_sensor("/bank/pleiades_stereo/2019-08-14_dem.tif") == "pleiades"


@pytest.mark.parametrize("name", [
    "academy_2020-01-01.tif",   # "dem" inside a word is irrelevant now
    "spotted_2020.tif",         # "spot" without 6/7
    "false_2020.tif",           # "als" inside a word
    "valsesia_2020.tif",
])
def test_no_false_positives_inside_words(name):
    assert infer_dem_sensor(pathlib.Path("/x") / name) == DEM_SENSOR_FALLBACK


def test_tokens_are_safe_in_thumb_and_pair_names():
    """Thumb splits its name on '_', Thumb splits its name on '_', and Pair splits pa_key into the two th_keys on '__'."""
    tokens = {t for _, t in DEM_SENSOR_PATTERNS} | {DEM_SENSOR_FALLBACK}
    for token in tokens:
        assert "_" not in token and "-" not in token and token == token.lower()
