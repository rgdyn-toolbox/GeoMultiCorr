#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# conftest.py
# creation date: 2026-07-28.
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
"""Shared fixtures for the pure pair-layout helpers (geometry + frame contract)."""

from __future__ import annotations

import pandas as pd
import pytest

from geomulticorr.core.pzone import _strategy_pair_indices


@pytest.fixture
def dates_iso() -> list[str]:
    """Ten irregularly spaced acquisitions spanning three calendar years."""
    return [
        "2020-03-15",
        "2020-06-02",
        "2020-09-21",
        "2020-12-30",
        "2021-04-11",
        "2021-08-08",
        "2021-11-27",
        "2022-02-14",
        "2022-07-19",
        "2022-10-05",
    ]


@pytest.fixture
def dates_ts(dates_iso) -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in dates_iso]


@pytest.fixture
def sensors(dates_iso) -> list[str]:
    """Alternating sensors, so multi-sensor code paths get exercised."""
    return ["planetscope" if i % 2 == 0 else "spot" for i in range(len(dates_iso))]


@pytest.fixture
def index_pairs_consecutive(dates_iso) -> list[tuple[int, int]]:
    return _strategy_pair_indices(len(dates_iso), "consecutive")


@pytest.fixture
def index_pairs_redundancy(dates_iso) -> list[tuple[int, int]]:
    """Emits both (i, j) and (j, i) — the duplicate-couple case."""
    return _strategy_pair_indices(len(dates_iso), "redundancy", max_step=2)
