#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for InversionExtractor._discover_tif_paths().

Locks down that the component-suffixed TOT_<date>_{EW,NS,magn}.tif naming
(see tests/inversion/test_post_process.py) still yields clean, matching
YYYYMMDD date keys across EW/NS/magn — a stray "_EW"/"_NS" left in the key
would silently break every date alignment downstream (available_dates,
compute_stats, extract's wide-format pivot).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from geomulticorr.stats.inversion_extractor import InversionExtractor

_DATES = ["20210901", "20211001"]


def _touch(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@pytest.fixture
def extractor(tmp_path):
    for date in _DATES:
        _touch(tmp_path / "inverse_EW" / f"TOT_{date}_EW.tif")
        _touch(tmp_path / "inverse_NS" / f"TOT_{date}_NS.tif")
        _touch(tmp_path / "inverse_magn" / f"TOT_{date}_magn.tif")

    obj = InversionExtractor.__new__(InversionExtractor)
    obj.inversion = SimpleNamespace(inversion_dir=tmp_path)
    return obj


class TestDiscoverTifPaths:
    def test_date_keys_have_no_component_suffix(self, extractor):
        result = extractor._discover_tif_paths(components=None)

        for comp, suffix in (("EW", "_EW"), ("NS", "_NS"), ("magn", "_magn")):
            assert set(result[comp].keys()) == set(_DATES)
            for date, path in result[comp].items():
                assert path.name == f"TOT_{date}{suffix}.tif"

    def test_dates_align_across_components(self, extractor):
        result = extractor._discover_tif_paths(components=None)
        assert set(result["EW"]) == set(result["NS"]) == set(result["magn"])

    def test_restricts_to_requested_components(self, extractor):
        result = extractor._discover_tif_paths(components=["EW"])
        assert list(result.keys()) == ["EW"]

    def test_backward_compatible_with_unsuffixed_files(self, tmp_path):
        # A pre-change post_process() run left plain TOT_<date>.tif behind —
        # removesuffix("_EW") on an already-bare date string is a no-op.
        _touch(tmp_path / "inverse_EW" / f"TOT_{_DATES[0]}.tif")
        obj = InversionExtractor.__new__(InversionExtractor)
        obj.inversion = SimpleNamespace(inversion_dir=tmp_path)

        result = obj._discover_tif_paths(components=["EW"])

        assert set(result["EW"].keys()) == {_DATES[0]}

    def test_available_dates_and_components_properties(self, extractor):
        extractor.tif_paths = extractor._discover_tif_paths(components=None)
        assert extractor.available_dates == _DATES
        assert sorted(extractor.available_components) == ["EW", "NS", "magn"]

    def test_init_raises_when_nothing_found(self, tmp_path):
        empty = SimpleNamespace(inversion_dir=tmp_path / "no_such_run")
        with pytest.raises(FileNotFoundError):
            InversionExtractor(empty)
