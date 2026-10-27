#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_layout.py
# creation date: 2026-10-19.
#
# Distributed under the GNU Affero General Public License v3 or later
# (https://www.gnu.org/licenses/agpl-3.0.txt), WITHOUT ANY WARRANTY.
# ---------------------------------------------------------------------------- #
"""Archive layout helpers: product-id parsing and server-name sanitisation."""

import pytest

from geomulticorr.data_sources._layout import (
    ORGANIZE_SCHEMES,
    ORGANIZE_SENSOR_PATHROW,
    filename_from_content_disposition,
    landsat_dirs,
    safe_filename,
    sentinel2_dirs,
)


class TestLandsatDirs:
    @pytest.mark.parametrize(
        "display_id, expected",
        [
            ("LC09_L1TP_232083_20230301_20230307_02_T1", ("LANDSAT9", "232_083")),
            ("LC08_L1TP_232083_20230301_20230307_02_T1", ("LANDSAT8", "232_083")),
            ("LE07_L1TP_195029_20200101_20200201_02_T1", ("LANDSAT7", "195_029")),
            ("LT05_L1TP_195029_19990101_20200201_02_T1", ("LANDSAT5", "195_029")),
            ("LM05_L1TP_195029_19990101_20200201_02_T1", ("LANDSAT5", "195_029")),
        ],
    )
    def test_known_products(self, display_id, expected):
        assert landsat_dirs(display_id) == expected

    def test_landsat8_and_9_are_told_apart_by_prefix(self):
        """Both live in one M2M dataset, so the id prefix is the only signal."""
        assert landsat_dirs("LC08_x_000000")[0] != landsat_dirs("LC09_x_000000")[0]

    def test_lowercase_prefix_is_accepted(self):
        assert landsat_dirs("lc08_L1TP_232083_x")[0] == "LANDSAT8"

    @pytest.mark.parametrize("junk", ["nonsense", "", "LC08", "LC08_L1TP_23208_x"])
    def test_junk_degrades_instead_of_raising(self, junk):
        sensor, grid = landsat_dirs(junk)
        assert grid == "unknown"
        assert sensor.startswith("LANDSAT")


class TestSentinel2Dirs:
    def test_tile_from_item_id(self):
        item = "S2A_MSIL2A_20230301T143721_R096_T19HCC_20230302T190112"
        assert sentinel2_dirs(item) == ("SENTINEL2", "T19HCC")

    def test_properties_win_over_the_id(self):
        assert sentinel2_dirs("whatever", {"s2:mgrs_tile": "T32TLR"}) == (
            "SENTINEL2",
            "T32TLR",
        )

    def test_grid_code_form(self):
        assert sentinel2_dirs("whatever", {"grid:code": "MGRS-19HCC"}) == (
            "SENTINEL2",
            "T19HCC",
        )

    def test_unresolvable_tile_is_unknown(self):
        assert sentinel2_dirs("nothing-here") == ("SENTINEL2", "unknown")

    def test_non_string_properties_are_ignored(self):
        item = "S2A_MSIL2A_20230301T143721_R096_T19HCC_20230302T190112"
        assert sentinel2_dirs(item, {"s2:mgrs_tile": None})[1] == "T19HCC"


class TestSafeFilename:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("LC08_B8.TIF", "LC08_B8.TIF"),
            ("  LC08_B8.TIF  ", "LC08_B8.TIF"),
            ('"quoted.tif"', "quoted.tif"),
            ("../../etc/passwd", "passwd"),
            ("a/b\\c.tif", "c.tif"),
            ("..\\..\\win.ini", "win.ini"),
            ("/abs/path/file.tif", "file.tif"),
            (".hidden", "hidden"),
            ("nul\x00byte.tif", "nulbyte.tif"),
        ],
    )
    def test_never_escapes_the_destination(self, raw, expected):
        out = safe_filename(raw)
        assert out == expected
        assert "/" not in out and "\\" not in out and ".." not in out

    @pytest.mark.parametrize("raw", ["///", "", "..", "...", "  "])
    def test_empty_result_uses_fallback(self, raw):
        assert safe_filename(raw) == "download"
        assert safe_filename(raw, fallback="x.bin") == "x.bin"


class TestContentDisposition:
    @pytest.mark.parametrize(
        "header, expected",
        [
            ('attachment; filename="LC08_B8.TIF"', "LC08_B8.TIF"),
            ("attachment; filename=LC08_B8.TIF", "LC08_B8.TIF"),
            ("attachment; filename=LC08_B8.TIF; size=3", "LC08_B8.TIF"),
            ("attachment; filename*=UTF-8''LC08%5FB8.TIF", "LC08_B8.TIF"),
        ],
    )
    def test_forms(self, header, expected):
        assert filename_from_content_disposition(header) == expected

    def test_extended_form_wins_over_plain(self):
        header = "attachment; filename=\"plain.tif\"; filename*=UTF-8''ext.tif"
        assert filename_from_content_disposition(header) == "ext.tif"

    @pytest.mark.parametrize("header", [None, "", "attachment", "inline; name=x"])
    def test_absent_or_nameless_is_none(self, header):
        assert filename_from_content_disposition(header) is None

    def test_traversal_in_header_is_neutralised(self):
        out = filename_from_content_disposition('attachment; filename="../../evil.sh"')
        assert out == "evil.sh"

    def test_encoded_traversal_is_neutralised(self):
        out = filename_from_content_disposition(
            "attachment; filename*=UTF-8''..%2F..%2Fevil.sh"
        )
        assert out == "evil.sh"


def test_organize_schemes_contract():
    assert ORGANIZE_SENSOR_PATHROW in ORGANIZE_SCHEMES
