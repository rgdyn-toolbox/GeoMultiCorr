"""Tests for grid identity, comparison and the 1-bit writer.

The headline test here is ``test_same_shape_shifted_origin_does_not_match``:
under the old shape-only guard that case was indistinguishable from a perfect
match, so a misaligned DEM was used as-is and produced a silently wrong
topographic correction.
"""
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

import geoutils as gu

from geomulticorr.utils._grid import (
    describe_grid,
    grid_key,
    grids_match,
    regrid_to_ref,
    write_binary_raster,
)


TRANSFORM = Affine(3.0, 0.0, 500000.0, 0.0, -3.0, 9900000.0)
EPSG = 32717


def _raster(height=32, width=32, transform=TRANSFORM, epsg=EPSG, fill=1.0):
    return gu.Raster.from_array(
        np.ma.MaskedArray(np.full((height, width), fill, dtype="float32")),
        transform=transform,
        crs=CRS.from_epsg(epsg),
        nodata=None,
    )


def _blobby_mask(height=64, width=64):
    """Spatially coherent mask — random noise would not compress."""
    yy, xx = np.mgrid[0:height, 0:width]
    return (np.sin(xx / 9.0) + np.cos(yy / 7.0)) > 0.1


class TestGridMatching:
    def test_identical_grids_match(self):
        assert grids_match(_raster(), _raster())

    def test_raster_matches_itself(self):
        r = _raster()
        assert grids_match(r, r)

    def test_different_shape_does_not_match(self):
        assert not grids_match(_raster(32, 32), _raster(32, 33))

    def test_same_shape_shifted_origin_does_not_match(self):
        """THE regression this module exists for.

        Same shape, same resolution, same CRS — origin shifted by half a
        pixel. The old ``shape !=`` guard called this a match and used the
        raster as-is.
        """
        shifted = Affine(3.0, 0.0, 500001.5, 0.0, -3.0, 9900000.0)
        assert not grids_match(_raster(), _raster(transform=shifted))

    def test_same_shape_different_crs_does_not_match(self):
        assert not grids_match(_raster(epsg=32717), _raster(epsg=32718))

    def test_same_shape_different_resolution_does_not_match(self):
        coarse = Affine(6.0, 0.0, 500000.0, 0.0, -6.0, 9900000.0)
        assert not grids_match(_raster(), _raster(transform=coarse))

    def test_float_noise_still_matches(self):
        """A transform that survived a write/read round-trip must match itself."""
        noisy = Affine(
            3.0 + 1e-13, 0.0, 500000.0 + 1e-11, 0.0, -3.0 - 1e-13, 9900000.0
        )
        assert grids_match(_raster(), _raster(transform=noisy))

    def test_band_axis_is_squeezed(self):
        """(1, h, w) and (h, w) describe the same grid."""
        r = _raster(16, 16)
        key_2d = grid_key(r)

        class _Banded:
            data = np.zeros((1, 16, 16))
            transform = TRANSFORM
            crs = CRS.from_epsg(EPSG)

        assert grid_key(_Banded()) == key_2d


class TestGridKeyOnMocks:
    """MockRaster (corrections conftest) has transform/res but no crs."""

    class _Mock:
        def __init__(self, shape=(10, 10), transform=TRANSFORM):
            self.data = np.ma.MaskedArray(np.zeros(shape))
            self.transform = transform

    def test_mock_without_crs_is_hashable_and_comparable(self):
        a, b = self._Mock(), self._Mock()
        assert grid_key(a) == grid_key(b)
        assert grids_match(a, b)

    def test_mock_shape_difference_detected(self):
        assert not grids_match(self._Mock((10, 10)), self._Mock((12, 12)))

    def test_object_without_transform_or_crs(self):
        class _Bare:
            data = np.zeros((4, 4))

        assert grid_key(_Bare()) == ((4, 4), None, None)


class TestDescribeGrid:
    def test_includes_shape_origin_res_and_epsg(self):
        text = describe_grid(_raster(100, 200))
        assert "200x100" in text
        assert "500000" in text
        assert "res 3" in text
        assert "EPSG:32717" in text

    def test_degrades_without_transform_or_crs(self):
        class _Bare:
            data = np.zeros((4, 5))

        assert describe_grid(_Bare()) == "5x4"


class TestRegridToRef:
    def test_matching_grid_returns_same_object_untouched(self):
        src, ref = _raster(), _raster()
        out, regridded = regrid_to_ref(src, ref)
        assert out is src
        assert regridded is False

    def test_mismatched_grid_is_reprojected(self):
        src = _raster(32, 32)
        ref = _raster(16, 16, transform=Affine(6.0, 0.0, 500000.0, 0.0, -6.0, 9900000.0))
        out, regridded = regrid_to_ref(src, ref)
        assert regridded is True
        assert grids_match(out, ref)

    def test_shifted_origin_is_reprojected(self):
        """The silent-failure case now actually warps."""
        shifted = Affine(3.0, 0.0, 500001.5, 0.0, -3.0, 9900000.0)
        src = _raster()
        ref = _raster(transform=shifted)
        out, regridded = regrid_to_ref(src, ref)
        assert regridded is True
        assert grids_match(out, ref)


class TestWriteBinaryRaster:
    def test_roundtrip_is_exact(self, tmp_path):
        mask = _blobby_mask()
        out = write_binary_raster(
            mask, TRANSFORM, CRS.from_epsg(EPSG), tmp_path / "m.tif"
        )
        with rasterio.open(out) as ds:
            assert np.array_equal(ds.read(1).astype(bool), mask)

    def test_encoding_is_one_bit(self, tmp_path):
        out = write_binary_raster(
            _blobby_mask(), TRANSFORM, CRS.from_epsg(EPSG), tmp_path / "m.tif"
        )
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "uint8"
            assert ds.tags(1, ns="IMAGE_STRUCTURE").get("NBITS") == "1"
            assert ds.tags(ns="IMAGE_STRUCTURE").get("COMPRESSION") == "DEFLATE"

    def test_no_nodata(self, tmp_path):
        """0 means 'not set', not 'no data'."""
        out = write_binary_raster(
            _blobby_mask(), TRANSFORM, CRS.from_epsg(EPSG), tmp_path / "m.tif"
        )
        with rasterio.open(out) as ds:
            assert ds.nodata is None

    def test_georeferencing_preserved(self, tmp_path):
        out = write_binary_raster(
            _blobby_mask(64, 64), TRANSFORM, CRS.from_epsg(EPSG), tmp_path / "m.tif"
        )
        with rasterio.open(out) as ds:
            assert ds.transform == TRANSFORM
            assert ds.crs == CRS.from_epsg(EPSG)
            assert (ds.height, ds.width) == (64, 64)

    def test_file_is_small(self, tmp_path):
        """Catches a float32 write or dropped co_opts."""
        out = write_binary_raster(
            _blobby_mask(512, 512), TRANSFORM, CRS.from_epsg(EPSG), tmp_path / "m.tif"
        )
        assert out.stat().st_size < 100_000

    def test_all_ones_is_tiny(self, tmp_path):
        """The reference-grid artifact: uniform data compresses to almost nothing."""
        out = write_binary_raster(
            np.ones((2000, 2000), dtype=bool),
            TRANSFORM,
            CRS.from_epsg(EPSG),
            tmp_path / "grid.tif",
        )
        assert out.stat().st_size < 20_000

    def test_accepts_3d_single_band(self, tmp_path):
        mask = _blobby_mask(32, 32)
        out = write_binary_raster(
            mask[np.newaxis, :, :], TRANSFORM, CRS.from_epsg(EPSG), tmp_path / "m.tif"
        )
        with rasterio.open(out) as ds:
            assert ds.count == 1
            assert np.array_equal(ds.read(1).astype(bool), mask)


class TestLazyRastersStayLazy:
    """Grid comparison must not force a lazy raster to read its pixels.

    ``_shape_of`` used to go through ``.data``, so merely comparing two grids
    pulled entire arrays into memory — which made verifying a freshly written
    DEM cost a full re-read of it.
    """

    def _on_disk(self, tmp_path, name="r.tif"):
        path = tmp_path / name
        _raster(64, 64).to_file(str(path))
        return gu.Raster(str(path), load_data=False)

    def test_grid_key_does_not_load(self, tmp_path):
        r = self._on_disk(tmp_path)
        assert r.is_loaded is False
        grid_key(r)
        assert r.is_loaded is False, "grid_key() triggered a data read"

    def test_describe_grid_does_not_load(self, tmp_path):
        r = self._on_disk(tmp_path)
        describe_grid(r)
        assert r.is_loaded is False, "describe_grid() triggered a data read"

    def test_grids_match_does_not_load(self, tmp_path):
        a = self._on_disk(tmp_path, "a.tif")
        b = self._on_disk(tmp_path, "b.tif")
        assert grids_match(a, b)
        assert a.is_loaded is False and b.is_loaded is False

    def test_lazy_and_loaded_compare_equal(self, tmp_path):
        """The same raster must have one identity, loaded or not."""
        lazy = self._on_disk(tmp_path)
        loaded = gu.Raster(str(tmp_path / "r.tif"), load_data=True)
        assert grid_key(lazy) == grid_key(loaded)
