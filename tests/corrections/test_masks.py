#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for geomulticorr.corrections.masks — boolean mask generators."""

from unittest.mock import MagicMock

import numpy as np
import numpy.ma as ma
import pytest
from affine import Affine

import geopandas as gpd
import geoutils as gu
from rasterio.crs import CRS
from shapely.geometry import box

from geomulticorr.corrections.masks import (
    BaseMask,
    FilterPipeline,
    OutlierFilter,
    CCFilter,
    SlopeMask,
    ShadowMask,
    SnowMask,
    CloudMask,
    StableAreaMask,
)
from geomulticorr.corrections.corrections import RampCorrection

# Matches MockRaster's default transform, so gu.Raster masks built here share the
# mock's georeferencing (10 m pixels, north-up).
AFFINE_10M = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)


class TestOutlierFilter:
    """Tests for OutlierFilter — mask by value bounds."""

    def test_inside_bounds_kept(self):
        """Values inside (-10, 10) → True."""
        from .conftest import MockRaster
        raster = MockRaster(data=ma.array([[1.0, 5.0, -3.0]]))
        result = OutlierFilter((-10, 10)).generate_mask(raster)
        assert result[0, 0] == True
        assert result[0, 1] == True
        assert result[0, 2] == True

    def test_outside_bounds_masked(self):
        """Values outside (-10, 10) → False."""
        from .conftest import MockRaster
        raster = MockRaster(data=ma.array([[15.0, -15.0]]))
        result = OutlierFilter((-10, 10)).generate_mask(raster)
        assert result[0, 0] == False
        assert result[0, 1] == False

    def test_boundary_exclusive(self):
        """Boundary values (-10, 10): strict inequalities."""
        from .conftest import MockRaster
        raster = MockRaster(data=ma.array([[-10.0, 10.0]]))
        result = OutlierFilter((-10, 10)).generate_mask(raster)
        assert result[0, 0] == False
        assert result[0, 1] == False

    def test_nan_masked(self):
        """NaN values → False (not finite)."""
        from .conftest import MockRaster
        raster = MockRaster(data=ma.array([[1.0, np.nan, 5.0]]))
        result = OutlierFilter((-10, 10)).generate_mask(raster)
        assert result[0, 0] == True
        assert result[0, 1] == False
        assert result[0, 2] == True

    def test_masked_input_filled_nan(self):
        """Masked pixel filled with NaN → not finite → False."""
        from .conftest import MockRaster
        data = ma.array([[1.0, 5.0, 3.0]], mask=[[False, True, False]])
        raster = MockRaster(data=data)
        result = OutlierFilter((-10, 10)).generate_mask(raster)
        assert result[0, 0] == True
        assert result[0, 1] == False
        assert result[0, 2] == True

    @pytest.mark.parametrize("lo,hi", [(-5, 5), (0, 100), (-1e3, 1e3)])
    def test_parametrized_thresholds(self, lo, hi):
        """Test various thresholds."""
        from .conftest import MockRaster
        raster = MockRaster(data=ma.array([[lo - 1, (lo + hi) / 2, hi + 1]]))
        result = OutlierFilter((lo, hi)).generate_mask(raster)
        assert result[0, 0] == False  # below lo
        assert result[0, 1] == True   # inside
        assert result[0, 2] == False  # above hi

    def test_default_threshold(self):
        """Default threshold is (-10.0, 10.0)."""
        corr = OutlierFilter()
        assert corr.threshold == (-10.0, 10.0)


class TestCCFilter:
    """Tests for CCFilter — correlation coefficient threshold."""

    def test_above_threshold_kept(self):
        """CC ≥ threshold → True."""
        from .conftest import MockRaster
        target = MockRaster(data=ma.array([[0.0, 0.0, 0.0]]))
        cc = MockRaster(data=ma.array([[0.3, 0.5, 0.8]]))
        result = CCFilter(0.5).generate_mask(target, cc=cc)
        assert result[0, 0] == False
        assert result[0, 1] == True
        assert result[0, 2] == True

    def test_below_threshold_masked(self):
        """CC < threshold → False."""
        from .conftest import MockRaster
        target = MockRaster(data=ma.array([[0.0]]))
        cc = MockRaster(data=ma.array([[0.3]]))
        result = CCFilter(0.5).generate_mask(target, cc=cc)
        assert result[0, 0] == False

    def test_boundary_inclusive(self):
        """CC exactly at threshold (uses >=) → True."""
        from .conftest import MockRaster
        target = MockRaster(data=ma.array([[0.0]]))
        cc = MockRaster(data=ma.array([[0.5]]))
        result = CCFilter(0.5).generate_mask(target, cc=cc)
        assert result[0, 0] == True

    def test_masked_cc_filled_zero(self):
        """Masked CC filled with 0.0 → below threshold → False."""
        from .conftest import MockRaster
        target = MockRaster(data=ma.array([[0.0]]))
        cc = MockRaster(data=ma.array([[0.8]], mask=[[True]]))
        result = CCFilter(0.5).generate_mask(target, cc=cc)
        assert result[0, 0] == False

    def test_return_bool_dtype(self, flat_raster, cc_raster):
        """Result dtype is bool."""
        result = CCFilter(0.5).generate_mask(flat_raster, cc=cc_raster)
        assert result.dtype == bool


class TestSlopeMask:
    """Tests for SlopeMask — slope angle bounds."""

    def test_interior_in_range_kept(self, ramp_raster, slope_dem_raster):
        """Interior slope ≈ 45°, bounds (0, 80): interior → True."""
        result = SlopeMask(min_slope=0, max_slope=80).generate_mask(
            ramp_raster, dem=slope_dem_raster
        )
        assert result[5, 5] == True

    def test_above_max_masked(self, ramp_raster, slope_dem_raster):
        """Interior slope ≈ 45°, max_slope=30: interior → False."""
        result = SlopeMask(min_slope=0, max_slope=30).generate_mask(
            ramp_raster, dem=slope_dem_raster
        )
        assert result[5, 5] == False

    def test_below_min_masked(self, ramp_raster):
        """Flat DEM (slope=0°), min_slope=5: all → False."""
        from .conftest import MockRaster
        flat_dem = MockRaster(data=ma.array(np.ones((10, 10)) * 300.0))
        result = SlopeMask(min_slope=5, max_slope=80).generate_mask(
            ramp_raster, dem=flat_dem
        )
        assert result.all() == False

    def test_dem_shape_mismatch_raises(self, ramp_raster, dem_raster):
        """Mismatched DEM shape → AssertionError."""
        from .conftest import MockRaster
        dem_bad = MockRaster(data=ma.array(np.zeros((15, 15))))
        with pytest.raises(AssertionError, match="must match displacement"):
            SlopeMask().generate_mask(ramp_raster, dem=dem_bad)

    def test_result_bool_shape(self, ramp_raster, slope_dem_raster):
        """Result shape matches input raster, dtype=bool."""
        result = SlopeMask().generate_mask(ramp_raster, dem=slope_dem_raster)
        assert result.shape == ramp_raster.data.shape
        assert result.dtype == bool

    def test_masked_dem_pixel_excluded(self, ramp_raster):
        """Masked DEM pixel → result False."""
        from .conftest import MockRaster
        dem_data = np.ones((10, 10)) * 300.0
        dem_masked = ma.array(dem_data, mask=False)
        dem_masked.mask = np.zeros((10, 10), dtype=bool)
        dem_masked.mask[3, 3] = True
        dem_raster = MockRaster(data=dem_masked)
        result = SlopeMask().generate_mask(ramp_raster, dem=dem_raster)
        assert result[3, 3] == False


class TestShadowMask:
    """Tests for ShadowMask — hillshade-based shadow masking."""

    def test_lit_terrain_kept(self, ramp_raster):
        """Flat DEM, sun at 45° elev, hillshade ≈ 0.707, thresh=0.1: all → True."""
        from .conftest import MockRaster
        flat_dem = MockRaster(data=ma.array(np.ones((10, 10)) * 500.0))
        result = ShadowMask(shadow_threshold=0.1).generate_mask(
            ramp_raster, dem=flat_dem, sun_azimuth_deg=0, sun_elevation_deg=45
        )
        assert result.all() == True

    def test_shadowed_terrain_masked(self, ramp_raster):
        """thresh=0.9 > hillshade ≈ 0.707: all → False."""
        from .conftest import MockRaster
        flat_dem = MockRaster(data=ma.array(np.ones((10, 10)) * 500.0))
        result = ShadowMask(shadow_threshold=0.9).generate_mask(
            ramp_raster, dem=flat_dem, sun_azimuth_deg=0, sun_elevation_deg=45
        )
        assert result.any() == False

    def test_default_threshold(self):
        """Default threshold is 0.1."""
        mask = ShadowMask()
        assert mask.shadow_threshold == 0.1

    def test_missing_dem_raises(self, ramp_raster):
        """Missing dem= kwarg → KeyError."""
        with pytest.raises(KeyError):
            ShadowMask().generate_mask(
                ramp_raster, sun_azimuth_deg=0, sun_elevation_deg=45
            )


class TestSnowMask:
    """Tests for SnowMask — snow detection."""

    def test_disabled_all_true(self, ramp_raster):
        """enabled=False: all pixels → True."""
        result = SnowMask(enabled=False).generate_mask(ramp_raster)
        assert result.all() == True
        assert result.shape == (10, 10)

    def test_numpy_snow_inverted(self, ramp_raster):
        """snow=True pixels masked out: snow inverted to get keep mask."""
        from .conftest import MockRaster
        snow_mask = np.array([[True, False], [False, True]])
        result = SnowMask().generate_mask(
            MockRaster(data=ma.array(np.ones((2, 2)))),
            snow_mask=snow_mask
        )
        assert result[0, 0] == False  # was True → inverted
        assert result[0, 1] == True   # was False → inverted

    def test_raster_snow_inverted(self, ramp_raster):
        """snow_mask as gu.Raster with bool data → inverted."""
        from .conftest import MockRaster
        snow_raster = gu.Raster.from_array(
            np.array([[True, False]]), transform=AFFINE_10M, crs=32632, nodata=None
        )
        small_raster = MockRaster(data=ma.array(np.ones((1, 2))))
        result = SnowMask().generate_mask(small_raster, snow_mask=snow_raster)
        assert result[0, 0] == False
        assert result[0, 1] == True

    def test_no_kwarg_raises(self, ramp_raster):
        """No snow_mask= kwarg → NotImplementedError."""
        with pytest.raises(NotImplementedError, match="snow_mask="):
            SnowMask().generate_mask(ramp_raster)

    def test_enabled_default_true(self):
        """Default enabled=True."""
        mask = SnowMask()
        assert mask.enabled == True


class TestCloudMask:
    """Tests for CloudMask — cloud detection."""

    def test_disabled_all_true(self, ramp_raster):
        """enabled=False: all → True."""
        result = CloudMask(enabled=False).generate_mask(ramp_raster)
        assert result.all() == True

    def test_numpy_cloud_inverted(self, ramp_raster):
        """cloud=True pixels masked out."""
        from .conftest import MockRaster
        cloud_data = np.array([[True, False]])
        small_raster = MockRaster(data=ma.array(np.ones((1, 2))))
        result = CloudMask().generate_mask(small_raster, cloud_mask=cloud_data)
        assert result[0, 0] == False
        assert result[0, 1] == True

    def test_raster_cloud_inverted(self, ramp_raster):
        """Same as numpy case via gu.Raster."""
        from .conftest import MockRaster
        cloud_raster = gu.Raster.from_array(
            np.array([[False, True]]), transform=AFFINE_10M, crs=32632, nodata=None
        )
        small_raster = MockRaster(data=ma.array(np.ones((1, 2))))
        result = CloudMask().generate_mask(small_raster, cloud_mask=cloud_raster)
        assert result[0, 0] == True
        assert result[0, 1] == False

    def test_no_kwarg_raises(self, ramp_raster):
        """No cloud_mask= kwarg → NotImplementedError."""
        with pytest.raises(NotImplementedError, match="cloud_mask="):
            CloudMask().generate_mask(ramp_raster)

    def test_enabled_default_true(self):
        """Default enabled=True."""
        mask = CloudMask()
        assert mask.enabled == True


class TestStableAreaMask:
    """Tests for StableAreaMask — known unstable area exclusion."""

    def test_numpy_bool_passthrough(self, ramp_raster):
        """Boolean array input: pixels marked False stay False."""
        stable_arr = np.ones((10, 10), dtype=bool)
        stable_arr[2, 2] = False
        result = StableAreaMask(stable_arr).generate_mask(ramp_raster)
        assert result[2, 2] == False
        assert result[0, 0] == True

    def test_all_true_numpy(self, ramp_raster):
        """All-True array: all pixels → True."""
        stable_arr = np.ones((10, 10), dtype=bool)
        result = StableAreaMask(stable_arr).generate_mask(ramp_raster)
        assert result.all() == True

    def test_gu_vector_mock(self, ramp_raster):
        """Mock geoutils.Vector with create_mask() → result inverted."""
        mock_vector = MagicMock(spec=gu.Vector)
        from .conftest import MockRaster
        moving_data = np.array([[True, False], [False, True]])
        moving_raster = MockRaster(data=ma.array(moving_data.astype(float)))
        mock_vector.create_mask.return_value = moving_raster
        small_raster = type(ramp_raster)(data=ma.array(np.ones((2, 2))))
        result = StableAreaMask(mock_vector).generate_mask(small_raster)
        assert result[0, 0] == False  # moving → inverted
        assert result[0, 1] == True   # not moving → inverted

    def test_str_path_loads_mask(self, ramp_raster, tmp_path):
        """str/Path to a vector file → pixels inside polygons masked out."""
        import geopandas as gpd
        from shapely.geometry import box
        # ramp_raster: 10×10, transform Affine(10,0,0,0,-10,0) → extent x[0,100], y[-100,0]
        gdf = gpd.GeoDataFrame(geometry=[box(0, -50, 50, 0)], crs="EPSG:2154")
        fn = tmp_path / "moving.gpkg"
        gdf.to_file(fn)
        result = StableAreaMask(str(fn)).generate_mask(ramp_raster)
        assert result.shape == ramp_raster.data.shape
        assert result.dtype == bool
        assert (~result).sum() > 0   # some pixels inside the polygon are masked (moving)
        assert result.sum() > 0      # some pixels outside are kept (stable)

    def test_gdf_loads_mask(self, ramp_raster):
        """GeoDataFrame input → pixels inside polygons masked out."""
        import geopandas as gpd
        from shapely.geometry import box
        gdf = gpd.GeoDataFrame(geometry=[box(0, -50, 50, 0)], crs="EPSG:2154")
        result = StableAreaMask(gdf).generate_mask(ramp_raster)
        assert result.shape == ramp_raster.data.shape
        assert (~result).sum() > 0   # inside polygon → masked
        assert result.sum() > 0      # outside → kept


class TestFilterPipeline:
    """Tests for FilterPipeline — sequential mask combination."""

    def test_construct_with_plus(self):
        """Using +: OutlierFilter() + CCFilter() → FilterPipeline."""
        fp = OutlierFilter() + CCFilter(0.6)
        assert isinstance(fp, FilterPipeline)
        assert len(fp.masks) == 2

    def test_three_mask_chain(self):
        """Three masks chained with +."""
        fp = OutlierFilter() + CCFilter(0.6) + SnowMask(enabled=False)
        assert len(fp.masks) == 3

    def test_compute_ands_masks(self):
        """Masks AND-combined: if one rejects, result is False."""
        from .conftest import MockRaster
        data = np.array([[3.0]])
        raster = MockRaster(data=ma.array(data))
        fp = OutlierFilter((-5, 5)) + CCFilter(0.8)
        cc_data = np.array([[0.6]])
        cc_raster = MockRaster(data=ma.array(cc_data))
        result = fp.compute(raster, cc=cc_raster)
        assert result[0, 0] == False  # CC filter kills it

    def test_compute_all_valid(self, ramp_raster, cc_raster):
        """All-pass masks: all pixels → True."""
        fp = OutlierFilter((-1e6, 1e6)) + CCFilter(0.0)
        result = fp.compute(ramp_raster, cc=cc_raster)
        assert result.all() == True

    def test_compute_all_invalid(self, ramp_raster):
        """Tight threshold kills all pixels."""
        fp = FilterPipeline([OutlierFilter((0, 1))])  # excludes all ramp values
        result = fp.compute(ramp_raster)
        assert result.any() == False

    def test_apply_returns_tuple(self, ramp_raster):
        """apply(xDisp, yDisp) returns tuple of 2 rasters."""
        fp = OutlierFilter()
        xc, yc = fp.apply(ramp_raster, ramp_raster)
        assert hasattr(xc, "data")
        assert hasattr(yc, "data")

    def test_add_correction_raises(self):
        """mask + correction → TypeError."""
        mask = OutlierFilter()
        corr = RampCorrection()
        with pytest.raises(TypeError, match="Cannot chain"):
            mask + corr

    def test_repr(self):
        """repr shows mask names."""
        fp = OutlierFilter() + CCFilter(0.5)
        assert "OutlierFilter" in repr(fp)
        assert "CCFilter" in repr(fp)


class TestBaseMaskApply:
    """Tests for BaseMask.apply() and apply_single()."""

    def test_apply_single_masks_outlier(self, ramp_raster):
        """apply_single: outlier pixel becomes masked."""
        from .conftest import MockRaster
        data = np.array([[1.0, 5.0, 15.0, -3.0]])
        raster = MockRaster(data=ma.array(data))
        result = OutlierFilter((-10, 10)).apply_single(raster)
        assert result.data.mask[0, 2] == True

    def test_apply_single_meta_newly_masked(self, ramp_raster):
        """apply_single sets meta['newly_masked'] and meta['masked_fraction']."""
        from .conftest import MockRaster
        data = np.array([[1.0, 5.0, 15.0]])
        raster = MockRaster(data=ma.array(data))
        corr = OutlierFilter((-10, 10))
        result = corr.apply_single(raster)
        assert corr.meta["newly_masked"] == 1
        assert corr.meta["masked_fraction"] == pytest.approx(1 / 3, abs=1e-9)

    def test_apply_returns_two_rasters(self, ramp_raster):
        """apply(xDisp, yDisp) returns two rasters."""
        corr = OutlierFilter()
        xc, yc = corr.apply(ramp_raster, ramp_raster)
        assert hasattr(xc, "data")
        assert hasattr(yc, "data")

    def test_apply_meta_both_components(self, ramp_raster):
        """apply() sets meta with both x and y component counts."""
        corr = OutlierFilter((-5, 5))
        xc, yc = corr.apply(ramp_raster, ramp_raster)
        assert "newly_masked_x" in corr.meta
        assert "newly_masked_y" in corr.meta
        assert "masked_fraction_x" in corr.meta
        assert "masked_fraction_y" in corr.meta

    def test_apply_preserves_existing_mask(self, partial_mask_raster):
        """Pre-masked pixel stays masked regardless of value."""
        corr = OutlierFilter((-1e6, 1e6))
        result = corr.apply_single(partial_mask_raster)
        assert result.data.mask[0, 0] == True
        assert result.data.mask[9, 9] == True


class TestStableAreaMaskCRS:
    """The rasterizer used to ignore CRS entirely.

    ``rasterio.features.geometry_mask`` reads geometry coordinates directly in
    the transform's coordinate space. Polygons in a different CRS therefore
    landed nowhere near the grid, producing an all-False moving mask — which
    inverts to an all-True *stable* mask. The corrections were then fitted on
    moving terrain, with no error and no warning.
    """

    EPSG = 2154
    TRANSFORM = Affine(10.0, 0.0, 900_000.0, 0.0, -10.0, 6_450_200.0)

    def _raster(self):
        return gu.Raster.from_array(
            np.ma.MaskedArray(np.zeros((20, 30), dtype="float32")),
            transform=self.TRANSFORM,
            crs=CRS.from_epsg(self.EPSG),
            nodata=None,
        )

    def _polygons(self, epsg):
        gdf = gpd.GeoDataFrame(
            geometry=[box(900_050.0, 6_450_050.0, 900_150.0, 6_450_150.0)],
            crs=f"EPSG:{self.EPSG}",
        )
        return gdf if epsg == self.EPSG else gdf.to_crs(epsg=epsg)

    def test_same_crs_masks_expected_pixels(self):
        keep = StableAreaMask(self._polygons(self.EPSG)).generate_mask(self._raster())
        assert 0 < (~keep).sum() < keep.size

    def test_different_crs_is_reprojected_not_silently_empty(self):
        """A WGS84 GeoDataFrame must mask the same pixels as its Lambert twin."""
        raster = self._raster()
        native = StableAreaMask(self._polygons(self.EPSG)).generate_mask(raster)
        wgs84 = StableAreaMask(self._polygons(4326)).generate_mask(raster)

        assert (~wgs84).sum() > 0, "wrong-CRS polygons masked nothing"
        assert np.array_equal(native, wgs84)

    def test_crs_less_polygons_warn(self, caplog_gmc):
        gdf = self._polygons(self.EPSG)
        gdf.crs = None
        with caplog_gmc.at_level("WARNING"):
            StableAreaMask(gdf).generate_mask(self._raster())
        assert "no CRS" in caplog_gmc.text
