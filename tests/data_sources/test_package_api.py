#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_package_api.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""Package-level contracts for geomulticorr.data_sources."""

import re
from pathlib import Path

import pytest

import geomulticorr.data_sources as ds

PKG_DIR = Path(ds.__file__).parent


def test_every_public_name_is_importable():
    missing = [n for n in ds.__all__ if not hasattr(ds, n)]
    assert missing == []


@pytest.mark.parametrize(
    "name",
    [
        "MPC_ImageFinder",
        "PlanetClient",
        "M2MClient",
        "GEEClient",
        "DEMClient",
        "DinamisClient",
        "DINAMIS_DEFAULT_FILENAME_TEMPLATE",
    ],
)
def test_headline_api_is_exported(name):
    assert name in ds.__all__


def test_deprecated_gee_path_still_reexports_the_same_classes():
    """`geomulticorr.gee` is a shim; the classes must be the very same objects."""
    with pytest.warns(DeprecationWarning):
        import importlib
        import geomulticorr.gee as old

        importlib.reload(old)
    assert old.GEEClient is ds.GEEClient
    assert old.ImageFinder is ds.GEE_ImageFinder


# Shapes of real credentials. A match in source means a secret was pasted in.
_SECRET_SHAPES = {
    "Planet key": r"PLAK[0-9a-fA-F]{20,}",
    "JWT": r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}",
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "Google API key": r"AIza[0-9A-Za-z_-]{30,}",
    "signed URL": r"(X-Amz-Signature=|[?&]sig=)[A-Za-z0-9%]{20,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}
# A literal assigned to a credential-named variable or keyword (placeholders excluded).
_LITERAL_CREDENTIAL = re.compile(
    r"""(?ix)\b(api_?key|api_?token|secret_?key|access_?key|password|token)\b\s*[:=]\s*
        (?P<q>["'])(?P<val>[^"']{8,})(?P=q)"""
)
_PLACEHOLDER = re.compile(r"(?i)your|xxx|example|<|\.\.\.|placeholder|changeme|^\{")


@pytest.mark.parametrize("path", sorted(PKG_DIR.glob("*.py")), ids=lambda p: p.name)
def test_no_hardcoded_secret_in_source(path):
    text = path.read_text(encoding="utf-8")
    for label, pattern in _SECRET_SHAPES.items():
        assert not re.search(pattern, text), f"{path.name}: looks like a {label}"
    for m in _LITERAL_CREDENTIAL.finditer(text):
        assert _PLACEHOLDER.search(
            m.group("val")
        ), f"{path.name}: literal assigned to {m.group(1)!r}: {m.group('val')[:12]}..."


def test_credentials_are_only_read_from_the_environment():
    env_names = set()
    for path in PKG_DIR.glob("*.py"):
        env_names |= set(
            re.findall(r'environ(?:\.get)?\(?\[?\s*"([A-Z0-9_]+)"', path.read_text())
        )
    assert {"PL_API_KEY", "USGS_M2M_TOKEN", "MDT_SPOT67_STAC_SECRET_KEY"} <= env_names
