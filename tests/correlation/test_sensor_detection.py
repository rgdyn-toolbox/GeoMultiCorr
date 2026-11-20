"""Optical sensor detection from a bank: file name, folders up to the bank root, Planet sidecar.

The case behind it: a Planet order kept as delivered,
``Planet_<site>_..._psorthotile_.../files/<item>_BGRN_SR_clip.tif``, where neither the file name
nor its parent ``files`` names a sensor.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from affine import Affine

from geomulticorr.core.session import Session
from geomulticorr.correlation import supported_sensors as ss

ITEM = "219088_3260007_2016-08-11_0e20"
ASSET = f"{ITEM}_BGRN_SR_clip.tif"


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _sidecar(folder, item=ITEM, item_type="PSOrthoTile"):
    sidecar = folder / f"{item}_metadata.json"
    sidecar.write_text(json.dumps({"properties": {"item_type": item_type}}))
    return sidecar


def test_planet_order_as_delivered_is_found_through_the_order_folder(tmp_path):
    bank = tmp_path / "bank"
    image = _touch(
        bank
        / "Planet_PierreBrune_2016Aug_psorthotile_analytic_sr_udm2"
        / "files"
        / ASSET
    )
    token = ss.detect_sensor_token(image, root=bank)
    assert ss.sensor_normalize(token)["sensor"] == "planetscope"


def test_psorthotile_alone_names_planetscope(tmp_path):
    bank = tmp_path / "bank"
    image = _touch(bank / "PierreBrune_2016Aug_psorthotile_analytic" / "files" / ASSET)
    assert ss.detect_sensor_token(image, root=bank) == "psorthotile"
    assert ss.sensor_normalize("psorthotile")["family"] == "planetscope"


def test_planet_sidecar_is_the_fallback(tmp_path):
    bank = tmp_path / "bank"
    image = _touch(bank / "order_2016" / "files" / ASSET)
    _sidecar(image.parent)
    assert ss.detect_sensor_token(image, root=bank) == "planetscope"


@pytest.mark.parametrize(
    "sidecar_kwargs, content",
    [
        ({"item": "999999_3260007_2016-08-11_0e20"}, None),  # another item's sidecar
        ({"item_type": "REOrthoTile"}, None),  # RapidEye, not PlanetScope
        ({}, "not json"),  # unreadable
    ],
)
def test_planet_sidecar_must_match_and_be_planetscope(
    tmp_path, sidecar_kwargs, content
):
    image = _touch(tmp_path / "bank" / "files" / ASSET)
    sidecar = _sidecar(image.parent, **sidecar_kwargs)
    if content is not None:
        sidecar.write_text(content)
    assert ss.detect_sensor_token(image, root=tmp_path / "bank") == "unknown"


def test_a_folder_above_the_bank_root_never_decides(tmp_path):
    bank = tmp_path / "planetscope_archive" / "bank"
    image = _touch(bank / "order" / "files" / "scene_2016-08-11.tif")
    assert ss.detect_sensor_token(image, root=bank) == "unknown"


def test_without_a_root_only_the_parent_folder_is_read(tmp_path):
    image = _touch(tmp_path / "Planet_order" / "files" / ASSET)
    assert ss.detect_sensor_token(image) == "unknown"
    assert ss.detect_sensor_token(tmp_path / "Planet_order" / ASSET) == "planet"


def test_the_file_name_still_wins(tmp_path):
    bank = tmp_path / "bank"
    image = _touch(
        bank / "Planet_order" / "LC08_L1TP_196028_20210821_20210827_02_T1_B8.TIF"
    )
    assert ss.detect_sensor_token(image, root=bank) == "lc08"


def test_map_single_georaster_reads_a_planet_order_as_delivered(tmp_path):
    bank = tmp_path / "bank"
    image = bank / "Planet_Site_2016Aug_psorthotile_analytic_sr_udm2" / "files" / ASSET
    image.parent.mkdir(parents=True)
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint16",
        crs="EPSG:32632",
        transform=Affine(3, 0, 300000, 0, -3, 5000000),
    ) as dst:
        dst.write(np.ones((1, 4, 4), dtype="uint16"))
    session = Session.__new__(Session)

    found = session.map_single_georaster(image, "EPSG:2154", {"opt"}, bank_root=bank)
    missed = session.map_single_georaster(image, "EPSG:2154", {"opt"})

    assert found["sensor"] == "planetscope"
    assert str(found["acq_date"]) == "2016-08-11"
    assert missed["sensor"] != "planetscope"
