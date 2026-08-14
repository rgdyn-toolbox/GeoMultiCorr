"""Tests for Pair.save_correction_masks() — per-pair 1-bit stable-ground masks.

These masks record which pixels the corrections were fitted on. They are
binary, so they are written NBITS=1 + DEFLATE; the size guard below is what
catches an accidental float32 write or a dropped co_opts dict.
"""
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

import geoutils as gu

from geomulticorr.core.pair import Pair


TRANSFORM = Affine(3.0, 0.0, 500000.0, 0.0, -3.0, 9900000.0)
EPSG = 32717


def _reference_raster(height=64, width=64):
    """A float32 displacement-like raster with nodata=0.0, as EW/NS rasters have."""
    return gu.Raster.from_array(
        np.ma.MaskedArray(np.zeros((height, width), dtype="float32")),
        transform=TRANSFORM,
        crs=CRS.from_epsg(EPSG),
        nodata=0.0,
    )


def _blobby_mask(height=64, width=64):
    """Spatially coherent boolean keep-mask (not random noise, which won't compress)."""
    yy, xx = np.mgrid[0:height, 0:width]
    return (np.sin(xx / 9.0) + np.cos(yy / 7.0)) > 0.1


def _make_pair(tmp_path):
    """A Pair with only the attributes save_correction_masks touches.

    The real constructor opens rasters and needs a Session + geodatabase;
    this method depends on nothing but the two mask paths.
    """
    pair = Pair.__new__(Pair)
    pair.pa_key = "TestPz_2024-01-01-planetscope_2024-01-15-planetscope"
    pair.pa_path = tmp_path
    pair.pa_ew_mask_path = tmp_path / f"{pair.pa_key}-F_EWmask.tif"
    pair.pa_ns_mask_path = tmp_path / f"{pair.pa_key}-F_NSmask.tif"
    return pair


class TestMaskPathNaming:
    """The path attributes follow the established -F_<COMPONENT> convention."""

    def test_mask_path_names(self, tmp_path):
        pair = _make_pair(tmp_path)
        assert pair.pa_ew_mask_path.name.endswith("-F_EWmask.tif")
        assert pair.pa_ns_mask_path.name.endswith("-F_NSmask.tif")
        assert pair.pa_ew_mask_path.parent == pair.pa_path


class TestSaveCorrectionMasks:
    """Writing, round-tripping and encoding of the mask GeoTIFFs."""

    def test_writes_both_components(self, tmp_path):
        pair = _make_pair(tmp_path)
        ref = _reference_raster()
        mask = _blobby_mask()

        written = pair.save_correction_masks(mask, ~mask, ref)

        assert set(written) == {"ew", "ns"}
        assert pair.pa_ew_mask_path.exists()
        assert pair.pa_ns_mask_path.exists()
        assert written["ew"] == pair.pa_ew_mask_path
        assert written["ns"] == pair.pa_ns_mask_path

    def test_roundtrip_is_exact(self, tmp_path):
        """Values survive the bool -> uint8 -> 1-bit -> uint8 trip unchanged."""
        pair = _make_pair(tmp_path)
        mask = _blobby_mask()

        pair.save_correction_masks(mask, ~mask, _reference_raster())

        with rasterio.open(pair.pa_ew_mask_path) as ds:
            ew = ds.read(1)
        with rasterio.open(pair.pa_ns_mask_path) as ds:
            ns = ds.read(1)

        assert np.array_equal(ew.astype(bool), mask)
        assert np.array_equal(ns.astype(bool), ~mask)

    def test_dtype_and_values(self, tmp_path):
        pair = _make_pair(tmp_path)
        pair.save_correction_masks(_blobby_mask(), None, _reference_raster())

        with rasterio.open(pair.pa_ew_mask_path) as ds:
            data = ds.read(1)
            assert ds.dtypes[0] == "uint8"

        assert set(np.unique(data)).issubset({0, 1})

    def test_nbits_is_one(self, tmp_path):
        """The 1-bit packing actually reached GDAL."""
        pair = _make_pair(tmp_path)
        pair.save_correction_masks(_blobby_mask(), None, _reference_raster())

        with rasterio.open(pair.pa_ew_mask_path) as ds:
            assert ds.tags(1, ns="IMAGE_STRUCTURE").get("NBITS") == "1"
            assert ds.tags(ns="IMAGE_STRUCTURE").get("COMPRESSION") == "DEFLATE"

    def test_no_nodata_is_set(self, tmp_path):
        """0 means 'not kept', not 'no data'.

        Regression guard: building the mask via reference.copy() would inherit
        the displacement raster's nodata=0.0 and flag every excluded pixel as
        nodata.
        """
        pair = _make_pair(tmp_path)
        pair.save_correction_masks(_blobby_mask(), None, _reference_raster())

        with rasterio.open(pair.pa_ew_mask_path) as ds:
            assert ds.nodata is None

    def test_georeferencing_matches_reference(self, tmp_path):
        pair = _make_pair(tmp_path)
        ref = _reference_raster()
        pair.save_correction_masks(_blobby_mask(), None, ref)

        with rasterio.open(pair.pa_ew_mask_path) as ds:
            assert ds.transform == TRANSFORM
            assert ds.crs == CRS.from_epsg(EPSG)
            assert (ds.height, ds.width) == (64, 64)

    def test_mask_file_is_small(self, tmp_path):
        """512x512 binary mask must stay well under 100 kB.

        Catches a float32 write or dropped co_opts: the same mask as float32
        is roughly an order of magnitude larger.
        """
        pair = _make_pair(tmp_path)
        pair.save_correction_masks(
            _blobby_mask(512, 512), None, _reference_raster(512, 512)
        )

        assert pair.pa_ew_mask_path.stat().st_size < 100_000

    def test_none_component_is_skipped(self, tmp_path):
        pair = _make_pair(tmp_path)

        written = pair.save_correction_masks(None, _blobby_mask(), _reference_raster())

        assert set(written) == {"ns"}
        assert not pair.pa_ew_mask_path.exists()
        assert pair.pa_ns_mask_path.exists()

    def test_both_none_writes_nothing(self, tmp_path):
        pair = _make_pair(tmp_path)

        written = pair.save_correction_masks(None, None, _reference_raster())

        assert written == {}
        assert not pair.pa_ew_mask_path.exists()
        assert not pair.pa_ns_mask_path.exists()

    def test_accepts_3d_single_band_array(self, tmp_path):
        """A (1, h, w) mask is squeezed rather than rejected."""
        pair = _make_pair(tmp_path)
        mask = _blobby_mask()

        pair.save_correction_masks(mask[np.newaxis, :, :], None, _reference_raster())

        with rasterio.open(pair.pa_ew_mask_path) as ds:
            assert ds.count == 1
            assert np.array_equal(ds.read(1).astype(bool), mask)

    def test_overwrites_existing_mask(self, tmp_path):
        """Re-running corrections refreshes the mask instead of failing."""
        pair = _make_pair(tmp_path)
        ref = _reference_raster()
        first = _blobby_mask()

        pair.save_correction_masks(first, None, ref)
        pair.save_correction_masks(~first, None, ref)

        with rasterio.open(pair.pa_ew_mask_path) as ds:
            assert np.array_equal(ds.read(1).astype(bool), ~first)
