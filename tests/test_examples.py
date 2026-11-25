#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_examples.py
# creation date: 2026-11-20.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------- #
"""The dataset loader, offline: a local ``file://`` URL stands in for the release."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from geomulticorr import examples


def _make_tarball(path, name="demo", content=b"pixels"):
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(f"{name}/data.bin")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A fake release folder, an empty cache, and the loader pointed at both."""
    served = tmp_path / "served" / examples.DATA_VERSION
    served.mkdir(parents=True)
    monkeypatch.setenv("GMC_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        examples, "BASE_URL", f"file://{served.parent}/{{version}}/{{asset}}.tar.gz"
    )
    monkeypatch.setattr(examples, "REGISTRY", {})
    return served


def test_download_extracts_into_the_cache(release):
    examples.REGISTRY["demo"] = _make_tarball(release / "demo.tar.gz")
    root = examples.get_path("demo")
    assert root == examples.cache_dir() / "demo"
    assert (root / "data.bin").read_bytes() == b"pixels"
    assert [p.name for p in examples.cache_dir().iterdir()] == ["demo"]


def test_cache_hit_does_not_download(release):
    examples.REGISTRY["demo"] = _make_tarball(release / "demo.tar.gz")
    first = examples.get_path("demo")
    (release / "demo.tar.gz").unlink()
    assert examples.get_path("demo") == first


def test_locally_built_dataset_needs_no_registry_entry(release):
    local = examples.cache_dir() / "local-build"
    local.mkdir(parents=True)
    assert examples.get_path("local-build") == local


def test_checksum_mismatch_raises_and_leaves_nothing(release):
    _make_tarball(release / "demo.tar.gz")
    examples.REGISTRY["demo"] = "0" * 64
    with pytest.raises(ValueError, match="Checksum mismatch"):
        examples.get_path("demo")
    assert list(examples.cache_dir().iterdir()) == []


def test_unknown_dataset_names_the_available_ones(release):
    examples.REGISTRY["demo"] = "x"
    with pytest.raises(ValueError, match="demo"):
        examples.get_path("nope")


def test_archive_escaping_its_folder_is_refused(release):
    archive = release / "demo.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../outside.bin")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    examples.REGISTRY["demo"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Unsafe archive"):
        examples.get_path("demo")
    assert not (examples.cache_dir().parent / "outside.bin").exists()


def test_default_cache_is_versioned(monkeypatch):
    monkeypatch.delenv("GMC_DATA_DIR", raising=False)
    assert examples.cache_dir().parts[-2:] == ("data", examples.DATA_VERSION)


def _exported_project(root, name="MS_Site_Sensor"):
    """A minimal exported project: a geodatabase and a text file holding placeholders."""
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    from shapely.geometry import Point

    project = root / name
    (project / "Site").mkdir(parents=True)
    gpkg = project / f"GMC_geodatabase_{name}.gpkg"
    thumbs = gpd.GeoDataFrame(
        {"th_path": [f"{examples.PROJECT_TOKEN}/Site/optical/Site_2020-01-01_s2.tif"]},
        geometry=[Point(0, 0)],
        crs=2154,
    )
    thumbs.to_file(gpkg, layer="Thumbs")
    bank = gpd.GeoDataFrame(
        {"file_path": [f"{examples.BANK_TOKEN}/scene.tif"]},
        geometry=[Point(0, 0)],
        crs=2154,
    )
    bank.to_file(gpkg, layer="Protomap")
    pyogrio.write_dataframe(
        pd.DataFrame({"styleName": ["x"]}), gpkg, layer="layer_styles"
    )
    (project / "Site" / "trace.json").write_text(
        f'{{"root": "{examples.PROJECT_TOKEN}"}}'
    )
    return project


def test_relocate_rewrites_every_placeholder(tmp_path):
    import pyogrio

    project = _exported_project(tmp_path)
    changed = examples.relocate_project(project, bank_root=tmp_path / "bank")
    assert changed == 2
    gpkg = next(project.glob("*.gpkg"))
    th_path = pyogrio.read_dataframe(gpkg, layer="Thumbs")["th_path"].iloc[0]
    file_path = pyogrio.read_dataframe(gpkg, layer="Protomap")["file_path"].iloc[0]
    assert th_path == f"{project}/Site/optical/Site_2020-01-01_s2.tif"
    assert file_path == f"{tmp_path / 'bank'}/scene.tif"
    assert str(project) in (project / "Site" / "trace.json").read_text()
    assert len(pyogrio.read_dataframe(gpkg, layer="layer_styles")) == 1


def test_relocate_without_a_bank_keeps_the_bank_placeholder(tmp_path):
    import pyogrio

    project = _exported_project(tmp_path)
    examples.relocate_project(project)
    gpkg = next(project.glob("*.gpkg"))
    file_path = pyogrio.read_dataframe(gpkg, layer="Protomap")["file_path"].iloc[0]
    assert file_path.startswith(examples.BANK_TOKEN)


def test_get_project_copies_and_never_touches_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("GMC_DATA_DIR", str(cache))
    monkeypatch.setattr(examples, "REGISTRY", {})
    cached = _exported_project(cache)
    before = {p: p.read_bytes() for p in cached.rglob("*") if p.is_file()}

    copy = examples.get_project(cached.name, tmp_path / "work")

    assert copy == tmp_path / "work" / cached.name
    assert str(copy) in (copy / "Site" / "trace.json").read_text()
    assert {p: p.read_bytes() for p in cached.rglob("*") if p.is_file()} == before
    with pytest.raises(FileExistsError):
        examples.get_project(cached.name, tmp_path / "work")
    assert examples.get_project(cached.name, tmp_path / "work", overwrite=True) == copy


def test_bank_folder_is_named_after_the_sensor(tmp_path, monkeypatch):
    monkeypatch.setenv("GMC_DATA_DIR", str(tmp_path))
    (tmp_path / "bank_Site_sentinel2" / "sentinel2").mkdir(parents=True)
    assert (
        examples.get_bank("Site", "sentinel2")
        == tmp_path / "bank_Site_sentinel2" / "sentinel2"
    )
