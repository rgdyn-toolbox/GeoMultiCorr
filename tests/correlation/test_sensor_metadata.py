#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acquisition date/time extraction from vendor metadata.

The XML tag patterns in ``supported_sensors`` are the only route to a date for
band-scoped filenames (Sentinel-2 ``B08.tif``), so their exact shape is
load-bearing for any archive that keeps catalog filenames.
"""
from __future__ import annotations

import re

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geomulticorr.correlation import supported_sensors as ss
from geomulticorr.correlation.supported_sensors import (
    _GENERIC_XML_DATE_PATTERNS,
    _GENERIC_XML_TIME_PATTERNS,
    _XML_DATE_TAGS,
    _XML_TIME_TAGS,
    _extract_time_components,
    _gdal_info_json,
    _parse_date_from_xml,
    extract_acquisition_date,
    search_date_in_filename,
    sensor_normalize,
    sensors,
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


class TestSensorNormalize:
    """One row per family branch of the sensor-name normaliser."""

    @pytest.mark.parametrize("text", ["", None])
    def test_empty_input_is_unknown_without_family(self, text):
        assert sensor_normalize(text) == {"sensor": "unknown", "platform": None}

    @pytest.mark.parametrize(
        "text, sensor, platform, family",
        [
            ("sentinel-2a", "sentinel-2", "sentinel-A", "sentinel"),
            ("LC08", "landsat8", "landsat8", "landsat"),
            ("landsat 8", "landsat8", "landsat8", "landsat"),
            ("spot 6", "spot6", "spot6", "spot"),
            ("pleiades 1a", "pleiades", "pleiades-1a", "pleiades"),
            ("pneo 3", "pleiadesneo", "pleiadesneo3", "pleiades"),
            ("pneo", "pleiadesneo", None, "pleiades"),
            ("PSScene", "planetscope", "planetscope", "planetscope"),
            ("re3", "rapideye", "rapideye-3", "rapideye"),
            ("rapideye", "rapideye", None, "rapideye"),
            ("wv2", "worldview2", "worldview2", "worldview"),
            ("worldview-3", "worldview3", "worldview3", "worldview"),
            ("geoeye-1", "geoeye1", "geoeye1", "geoeye"),
            ("qb1", "quickbird", "quickbird", "quickbird"),
            ("ikonos", "ikonos", "ikonos", "ikonos"),
            ("swissimage", "aerial", "swissimage", "aerial"),
            ("drone", "aerial", "aerial", "aerial"),
            ("hillshade", "dem", None, "dem"),
            ("dem", "dem", None, "dem"),
        ],
    )
    def test_family_branches(self, text, sensor, platform, family):
        assert sensor_normalize(text) == {
            "sensor": sensor,
            "platform": platform,
            "family": family,
        }

    def test_unrecognised_text_is_unknown_with_a_none_family(self):
        assert sensor_normalize("foo") == {
            "sensor": "unknown",
            "platform": None,
            "family": None,
        }

    def test_s2_tile_code_is_read_as_sentinel(self):
        """Documents the greedy tile-code branch: ``t31tgk`` is taken as Sentinel-2."""
        assert sensor_normalize("t31tgk")["family"] == "sentinel"

    @pytest.mark.parametrize("text,platform", [
        ("ple", None), ("pl1a", "pleiades-1a"), ("PHR1A", "pleiades-1a"),
        ("pleiades 1b", "pleiades-1b"),
    ])
    def test_short_pleiades_tokens_do_not_crash(self, text, platform):
        out = sensor_normalize(text)
        assert out["family"] == "pleiades" and out["sensor"] == "pleiades"
        assert out["platform"] == platform


class TestSensorsHelper:
    def test_one_name_gives_lower_and_upper(self):
        assert sensors(["x"]) == "x|X"

    def test_two_names_are_pipe_joined(self):
        assert sensors(["a", "B"]) == "a|A|b|B"

    def test_none_default_is_not_iterable(self):
        """The ``None`` default cannot be iterated; callers must pass names."""
        with pytest.raises(TypeError):
            sensors()


class TestTimeComponents:
    def test_bare_time_after_planet_date(self):
        assert _extract_time_components("PSScene_20170801_094239_x.tif") == (9, 42, 39)

    def test_plain_colon_time(self):
        assert _extract_time_components("scene at 10:30:45 UTC") == (10, 30, 45)

    def test_no_time_returns_none(self):
        assert _extract_time_components("nothing here") is None


class TestSearchDateFallbacks:
    def test_impossible_date_is_rejected(self):
        assert search_date_in_filename("img_20210230.tif") is None

    def test_impossible_date_with_time_is_rejected_by_every_pattern(self):
        """The 14-digit and T-separated patterns retry on the first 8 digits, then give up."""
        assert search_date_in_filename("img_20210230123000.tif") is None
        assert search_date_in_filename("img_20210230T123000.tif") is None

    def test_valid_date_still_found(self):
        assert search_date_in_filename("img_20210228.tif") == "2021-02-28"


class TestGdalInfoJson:
    def test_missing_file_gives_empty_dict(self, tmp_path):
        assert _gdal_info_json(tmp_path / "nope.tif") == {}

    def test_gdal_failure_gives_empty_dict(self, monkeypatch, tmp_path):
        def boom(_path):
            raise RuntimeError("gdal exploded")

        monkeypatch.setattr(ss.gdal, "Open", boom)
        assert _gdal_info_json(tmp_path / "x.tif") == {}

    def test_readable_raster_gives_info(self, tmp_path):
        path = tmp_path / "ok.tif"
        with rasterio.open(
            path, "w", driver="GTiff", width=4, height=3, count=1, dtype="uint8",
            crs="EPSG:32632", transform=from_origin(500000, 4649000, 3, 3),
        ) as dst:
            dst.write(np.ones((1, 3, 4), "uint8"))
        info = _gdal_info_json(path)
        assert info["size"] == [4, 3]


class TestExtractAcquisitionDateCandidates:
    PS_NAME = "PSScene_20170801_094239_3B.tif"
    PS_XML = "<eop:acquisitionDate>2017-08-01T09:42:39Z</eop:acquisitionDate>"
    OTHER_XML = "<IMAGING_DATE>2016-01-01</IMAGING_DATE>"

    def test_same_stem_sidecar_is_a_candidate(self, tmp_path):
        (tmp_path / self.PS_NAME).touch()
        (tmp_path / "PSScene_20170801_094239_3B.xml").write_text(self.PS_XML)
        got = extract_acquisition_date(tmp_path / self.PS_NAME)
        assert got is not None
        assert got.strftime("%Y-%m-%d %H:%M:%S") == "2017-08-01 09:42:39"

    def test_date_and_time_token_pick_the_right_xml_among_several_scenes(self, tmp_path):
        """A decoy sorted first, and another scene with a different date, must lose."""
        (tmp_path / self.PS_NAME).touch()
        (tmp_path / "20170801_000000_decoy.xml").write_text(self.OTHER_XML)
        (tmp_path / "20170801_094239_meta.xml").write_text(self.PS_XML)
        (tmp_path / "20180101_000000.xml").write_text(self.OTHER_XML)
        got = extract_acquisition_date(tmp_path / self.PS_NAME)
        assert got is not None
        assert got.strftime("%Y-%m-%d %H:%M:%S") == "2017-08-01 09:42:39"

    def test_unreadable_candidate_is_skipped_and_filename_is_the_fallback(self, tmp_path):
        """A directory named ``*.xml`` raises OSError on read; extraction must carry on."""
        name = "LC08_L1TP_232083_20230301_20230307_02_T1_B8.TIF"
        (tmp_path / name).touch()
        (tmp_path / "aaa.xml").mkdir()
        got = extract_acquisition_date(tmp_path / name, sensor="lc08")
        assert got is not None and got.strftime("%Y-%m-%d") == "2023-03-01"

    def test_date_that_fails_strptime_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ss, "search_date_in_filename", lambda _f: "2021-02-30")
        (tmp_path / "x.tif").touch()
        assert extract_acquisition_date(tmp_path / "x.tif") is None


class TestPleiadesFromFilename:
    """A Pléiades file must resolve to sensor + family through the real detection chain
    (``re_searcher`` over ``supported_sensors``, then ``sensor_normalize``), like SPOT/PlanetScope."""

    @staticmethod
    def _detect(filename):
        from geomulticorr.core.session import re_searcher

        token = re_searcher(filename.lower(), ss.sensors(ss.supported_sensors))
        return sensor_normalize(token)

    @pytest.mark.parametrize("filename,sensor,platform", [
        ("chim_2016-11-17_pleiades.tif", "pleiades", None),
        ("pz_2020-01-01_pléiades.tif", "pleiades", None),
        ("pz_2020-01-01_pleiades-1b.tif", "pleiades", None),
        ("pz_2020-01-01_pleiades_1a.tif", "pleiades", None),
        ("pz_2020-01-01_phr1a.tif", "pleiades", "pleiades-1a"),
        ("pz_2020-01-01_PHR1B.tif", "pleiades", "pleiades-1b"),
        ("pz_2020-01-01_ple.tif", "pleiades", None),
        ("pz_2020-01-01_pleiades-neo.tif", "pleiadesneo", None),
        ("pz_2020-01-01_pneo.tif", "pleiadesneo", None),
    ])
    def test_filename_gives_sensor_and_family(self, filename, sensor, platform):
        out = self._detect(filename)
        assert out["sensor"] == sensor
        assert out["family"] == "pleiades"
        assert out["platform"] == platform

    def test_neo_is_listed_before_plain_pleiades(self):
        """Alternation takes the first match, so the longer token must come first."""
        lst = ss.supported_sensors
        assert lst.index("pleiades-neo") < lst.index("pleiades")

    @pytest.mark.parametrize("text,sensor,platform", [
        ("pleiades", "pleiades", None),
        ("pléiades", "pleiades", None),
        ("pleiades-1b", "pleiades", "pleiades-1b"),
        ("pleiades 1a", "pleiades", "pleiades-1a"),
        ("pleiades-neo", "pleiadesneo", None),
        ("pleiades neo 3", "pleiadesneo", "pleiadesneo3"),
        ("PHR1A", "pleiades", "pleiades-1a"),
        ("PNEO4", "pleiadesneo", "pleiadesneo4"),
    ])
    def test_sensor_normalize_variants(self, text, sensor, platform):
        out = sensor_normalize(text)
        assert (out["sensor"], out["platform"], out["family"]) == (sensor, platform, "pleiades")
