#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acquisition date/time extraction from vendor metadata.

The XML tag patterns in ``supported_sensors`` are the only route to a date for
band-scoped filenames (Sentinel-2 ``B08.tif``), so their exact shape is
load-bearing for any archive that keeps catalog filenames.
"""
from __future__ import annotations

import re

import pytest

from geomulticorr.correlation.supported_sensors import (
    _GENERIC_XML_DATE_PATTERNS,
    _GENERIC_XML_TIME_PATTERNS,
    _XML_DATE_TAGS,
    _XML_TIME_TAGS,
    _parse_date_from_xml,
    extract_acquisition_date,
)

# How ESA actually writes the granule sensing time in every MTD_TL.xml.
S2_REAL = (
    '<?xml version="1.0" encoding="UTF-8"?><n1:Level-2A_Tile_ID><n1:General_Info>'
    '<TILE_ID metadataLevel="Brief">S2A_OPER_MSI_L2A_TL_EPAE_20230302</TILE_ID>'
    '<SENSING_TIME metadataLevel="Standard">2023-03-01T14:37:21.024Z</SENSING_TIME>'
    "</n1:General_Info></n1:Level-2A_Tile_ID>"
)


class TestAttributesOnTags:
    """Vendor metadata puts attributes on tags; the patterns must tolerate them."""

    def test_sentinel2_real_sensing_time_parses(self):
        """Regression: `<SENSING_TIME metadataLevel="Standard">` used to not match.

        The pattern required an attribute-free `<SENSING_TIME>`, so it failed on
        *every real* Sentinel-2 granule metadata file. Scenes whose filename
        carries no date — which is all of them in a catalog-named archive — were
        then dropped silently by ``map_georasters_bank``.
        """
        assert _parse_date_from_xml(S2_REAL, "sentinel") == ("2023-03-01", (14, 37, 21))

    def test_bare_tag_still_parses(self):
        bare = "<SENSING_TIME>2023-03-01T14:37:21.024Z</SENSING_TIME>"
        assert _parse_date_from_xml(bare, "sentinel") == ("2023-03-01", (14, 37, 21))

    @pytest.mark.parametrize(
        "family, xml, expected_date",
        [
            ("spot", '<IMAGING_DATE version="2">2021-07-14</IMAGING_DATE>', "2021-07-14"),
            ("pleiades", "<IMAGING_DATE>2021-07-14</IMAGING_DATE>", "2021-07-14"),
            (
                "worldview",
                '<FIRSTLINETIME metadataLevel="X">2019-08-01T10:11:12Z</FIRSTLINETIME>',
                "2019-08-01",
            ),
            (
                "planetscope",
                "<eop:acquisitionDate>2017-08-01T09:42:39Z</eop:acquisitionDate>",
                "2017-08-01",
            ),
            (
                "landsat",
                '<DATE_ACQUIRED metadataLevel="Standard">2023-03-01</DATE_ACQUIRED>',
                "2023-03-01",
            ),
        ],
    )
    def test_every_family_tolerates_attributes(self, family, xml, expected_date):
        assert _parse_date_from_xml(xml, family)[0] == expected_date

    def test_separate_time_tag_with_attributes(self):
        xml = (
            '<IMAGING_DATE metadataLevel="Standard">2021-07-14</IMAGING_DATE>'
            '<IMAGING_TIME metadataLevel="Standard">10:30:45</IMAGING_TIME>'
        )
        assert _parse_date_from_xml(xml, "spot") == ("2021-07-14", (10, 30, 45))

    def test_unknown_family_falls_back_to_generic_patterns(self):
        assert _parse_date_from_xml(S2_REAL, None)[0] == "2023-03-01"


class TestTagNamesNotWidened:
    """Admitting attributes must not turn `TIME` into a prefix match."""

    @pytest.mark.parametrize(
        "xml",
        [
            "<TIMESTAMP>2023-03-01</TIMESTAMP>",
            '<TILE_ID metadataLevel="Brief">S2A_OPER</TILE_ID>',
            "<IMAGING_DATETIME>2023-03-01</IMAGING_DATETIME>",
            "<SENSING_TIMESTAMP>2023-03-01</SENSING_TIMESTAMP>",
        ],
    )
    def test_longer_tag_names_do_not_match(self, xml):
        patterns = (
            list(_XML_DATE_TAGS.values())
            + list(_XML_TIME_TAGS.values())
            + _GENERIC_XML_DATE_PATTERNS
            + _GENERIC_XML_TIME_PATTERNS
        )
        assert not [p for p in patterns if re.search(p, xml)]

    def test_no_date_found_returns_none(self):
        assert _parse_date_from_xml("<FOO>bar</FOO>", "sentinel") == (None, None)


class TestExtractAcquisitionDate:
    """End-to-end resolution for the two filename shapes an archive produces."""

    def test_band_scoped_filename_uses_the_xml_sidecar(self, tmp_path):
        """Sentinel-2: the filename carries no date, so the sidecar is the only source."""
        scene = tmp_path / "S2A_MSIL2A_20230301T143721_R096_T19HCC_20230302T190112"
        scene.mkdir()
        (scene / "B08.tif").touch()
        (scene / "MTD_TL.xml").write_text(S2_REAL)

        got = extract_acquisition_date(scene / "B08.tif", sensor="s2")
        assert got is not None
        assert got.strftime("%Y-%m-%d %H:%M:%S") == "2023-03-01 14:37:21"

    def test_band_scoped_filename_without_sidecar_returns_none(self, tmp_path):
        """The metadata dependency is real: no sidecar, no date."""
        scene = tmp_path / "S2A_MSIL2A_20230301T143721_R096_T19HCC_20230302T190112"
        scene.mkdir()
        (scene / "B08.tif").touch()
        assert extract_acquisition_date(scene / "B08.tif", sensor="s2") is None

    def test_landsat_catalog_filename_needs_no_sidecar(self, tmp_path):
        """Landsat band files embed the acquisition date, ahead of the processing date."""
        name = "LC08_L1TP_232083_20230301_20230307_02_T1_B8.TIF"
        (tmp_path / name).touch()
        got = extract_acquisition_date(tmp_path / name, sensor="lc08")
        assert got is not None and got.strftime("%Y-%m-%d") == "2023-03-01"
