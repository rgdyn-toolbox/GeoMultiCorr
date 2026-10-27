#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# conftest.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""Shared fixtures for the data_sources tests.

Nothing here touches the network or a real credential: every test either
exercises a pure function or replaces the one call that would leave the process.
"""

import pytest

#: every credential variable the data_sources modules read
CREDENTIAL_ENV_VARS = (
    "PL_API_KEY",
    "USGS_M2M_USERNAME",
    "USGS_M2M_TOKEN",
    "USGS_M2M_PASSWORD",
    "MDT_SPOT67_STAC_ACCESS_KEY",
    "MDT_SPOT67_STAC_SECRET_KEY",
    "TLD_ACCESS_KEY",
    "TLD_SECRET_KEY",
)


@pytest.fixture(autouse=True)
def clean_credentials(monkeypatch):
    """Start every test with no credentials in the environment.

    A developer's real ``.env`` must neither leak into the tests (a "missing key"
    test would silently pass for the wrong reason) nor be modified by them.
    """
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
