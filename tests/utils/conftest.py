#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared fixtures for the pure pair-layout helpers (geometry + frame contract)."""
from __future__ import annotations

import pandas as pd
import pytest

from geomulticorr.core.pzone import _strategy_pair_indices


@pytest.fixture
def dates_iso() -> list[str]:
    """Ten irregularly spaced acquisitions spanning three calendar years."""
    return [
        "2020-03-15", "2020-06-02", "2020-09-21", "2020-12-30",
        "2021-04-11", "2021-08-08", "2021-11-27",
        "2022-02-14", "2022-07-19", "2022-10-05",
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
