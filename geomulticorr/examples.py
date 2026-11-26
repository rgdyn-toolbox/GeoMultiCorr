#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# examples.py
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
"""Small real datasets for tests and tutorials, fetched on demand.

The data live on the releases of the ``geomulticorr-data`` repository, one tarball per
asset, so nothing heavy is installed with the package. Each study site has, per sensor:

- an **image bank** ``bank_<Site>_<sensor>``: raw images and their metadata, what
  :meth:`Session.map_georasters_bank` scans;
- a **GMC project** ``MS_<Site>_<Sensor>``: geodatabase, mapset and pzone folders, built from
  that bank by GMC, with the output of every stage.

::

    from geomulticorr import examples

    examples.available()
    bank = examples.get_bank("PierreBrune", "sentinel2")
    project = examples.get_project("MS_PierreBrune_Sentinel2", tmp_path)
    session = open_gmc_session(project)

The cache folder is ``$GMC_DATA_DIR`` when set, else
``~/.cache/geomulticorr/data/<DATA_VERSION>/``. The cached copy must stay as it was
downloaded: :func:`get_project` copies a project out of it, because every stage writes into
its project, and rewrites the paths its geodatabase stores.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

#: The data release this version of GeoMultiCorr is tested against.
DATA_VERSION = "v2"

#: ``{asset}`` is one registry key.
BASE_URL = "https://github.com/rgdyn-toolbox/geomulticorr-data/releases/download/{version}/{asset}.tar.gz"

#: name → SHA-256 of the tarball, copied from ``registry.json`` in the data repository.
REGISTRY: dict[str, str] = {
    "bank_PierreBrune_sentinel2": "01434f9583252f05e8a53954b73678359f7b0802a407d036958df52b26035958",
    "MS_PierreBrune_Sentinel2": "c13612f5edd8184bf5958d16e3d39930610dc63193c125f6fb6392407c0fd57b",
    "bank_PierreBrune_landsat": "85b9f1023b3c38f44a1f20ff1a55a33ad0026eb937814408591a892e334f15cc",
    "MS_PierreBrune_Landsat": "9cfda41ac4a675bf062f2478e80eb539de652573f245f762a14e92ad331ccafa",
}

#: Placeholders the data repository writes in place of absolute paths.
PROJECT_TOKEN = "@GMC_PROJECT@"
BANK_TOKEN = "@GMC_BANK@"
_TEXT_SUFFIXES = {".json", ".txt", ".xml"}

_CHUNK_BYTES = 1 << 20
_TIMEOUT_S = 120


def available() -> list[str]:
    """Names of the datasets of this data release."""
    return sorted(REGISTRY)


def cache_dir() -> Path:
    """Folder holding the extracted datasets."""
    configured = os.environ.get("GMC_DATA_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "geomulticorr" / "data" / DATA_VERSION


def get_path(name: str) -> Path:
    """Return the folder of dataset *name*, downloading it on first use.

    A folder already present in the cache is returned as is, so ``GMC_DATA_DIR`` may point
    at locally built datasets, including ones not in the registry.

    :raises ValueError: if *name* is neither cached nor a known dataset.
    """
    root = cache_dir() / name
    if root.is_dir():
        return root
    return download(name)


def get_bank(site: str, sensor: str) -> Path:
    """The image-bank folder of *sensor* at *site*, ready for ``map_georasters_bank``."""
    return get_path(f"bank_{site}_{sensor}") / sensor


def get_project(name: str, dest: str | Path, *, overwrite: bool = False) -> Path:
    """Copy project *name* into *dest* and point its geodatabase at the copy.

    The geodatabase paths point to the copy, and to the matching image bank when that bank
    can be fetched (else ``@GMC_BANK@`` stays). The cache itself is never modified.

    :returns: ``dest/<name>``, which ``open_gmc_session`` opens.
    :raises FileExistsError: if ``dest/<name>`` exists and *overwrite* is False.
    """
    source = get_path(name)
    target = Path(dest) / name
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
        shutil.rmtree(target)
    shutil.copytree(source, target)
    relocate_project(target, bank_root=_bank_root_of(source))
    return target


def relocate_project(root: str | Path, bank_root: str | Path | None = None) -> int:
    """Replace the path placeholders of an exported project; returns the cells changed.

    Rewrites every text column of every geodatabase layer, and the JSON/text/XML files.
    Layers are rewritten through pyogrio, as ``Session._rewrite_geodatabase_paths`` does:
    a SQL ``UPDATE`` would fire the GeoPackage R-tree triggers, which need spatial functions
    plain ``sqlite3`` lacks.
    """
    import pyogrio

    root = Path(root)
    replacements = {PROJECT_TOKEN: str(root)}
    if bank_root is not None:
        replacements[BANK_TOKEN] = str(bank_root)
    changed = 0
    for gpkg in root.glob("*.gpkg"):
        for layer, _ in pyogrio.list_layers(gpkg):
            frame = pyogrio.read_dataframe(gpkg, layer=layer)
            touched = False
            for column in frame.columns:
                if frame[column].dtype != object or column == "geometry":
                    continue
                before = frame[column]
                after = before.map(lambda v: _replace(v, replacements))
                diff = int((after != before).sum())
                if diff:
                    frame[column] = after
                    changed += diff
                    touched = True
            if touched:
                pyogrio.write_dataframe(frame, gpkg, layer=layer)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in _TEXT_SUFFIXES:
            text = path.read_text(errors="replace")
            new = _replace(text, replacements)
            if new != text:
                path.write_text(new)
    return changed


def _replace(value, replacements: dict[str, str]):
    if not isinstance(value, str):
        return value
    for token, path in replacements.items():
        value = value.replace(token, path)
    return value


def _bank_root_of(project_source: Path) -> Path | None:
    """The bank folder named in the project's manifest, if it can be fetched."""
    import json

    manifest = project_source / "manifest.json"
    if not manifest.exists():
        return None
    meta = json.loads(manifest.read_text())
    try:
        return get_bank(meta["site"], meta["sensor"])
    except (KeyError, ValueError, OSError):
        return None


def download(name: str) -> Path:
    """Download, verify and extract dataset *name* into the cache, replacing any copy.

    :raises ValueError: if *name* is unknown, the checksum differs, or the archive holds a
        path that would land outside its folder.
    :raises urllib.error.URLError: if the release cannot be reached.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; available: {available()}")
    target_parent = cache_dir()
    target_parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(version=DATA_VERSION, asset=name)

    # Same filesystem as the cache, so the final rename is atomic.
    with tempfile.TemporaryDirectory(dir=target_parent, prefix=f".{name}-") as tmp:
        archive = Path(tmp) / f"{name}.tar.gz"
        digest = _fetch(url, archive)
        if digest != REGISTRY[name]:
            raise ValueError(
                f"Checksum mismatch for {name!r}: expected {REGISTRY[name]}, got "
                f"{digest}. The download is corrupt or the release has changed."
            )
        extracted = Path(tmp) / "extracted"
        _extract(archive, extracted)
        root = target_parent / name
        if root.exists():
            shutil.rmtree(root)
        (extracted / name).rename(root)
    return root


def _fetch(url: str, target: Path) -> str:
    """Stream *url* into *target* and return its SHA-256."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
        with target.open("wb") as handle:
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                digest.update(chunk)
                handle.write(chunk)
    return digest.hexdigest()


def _extract(archive: Path, target: Path) -> None:
    """Extract with the ``data`` filter, which refuses links and paths leaving *target*."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target, filter="data")
    except tarfile.FilterError as exc:
        raise ValueError(f"Unsafe archive {archive.name}: {exc}") from exc
