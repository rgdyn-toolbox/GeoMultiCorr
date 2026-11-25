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
