"""Terrain derivatives used to correlate multi-temporal DEMs (utils/_terrain.py)."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from geomulticorr.corrections.fit import compute_slope
from geomulticorr.utils._terrain import (
    TERRAIN_PRODUCTS,
    hillshade,
    multidirectional_hillshade,
    shade_to_uint8,
    slope_degrees,
    terrain_product,
)


def _plane(n=20, res=2.0, gx=0.0, gy=0.0):
    """z = gx·easting + gy·northing on a north-up grid (row 0 = north)."""
    rows, cols = np.mgrid[0:n, 0:n].astype(float)
    east = cols * res
    north = (n - 1 - rows) * res
    return gx * east + gy * north


def _cone(n=81, res=1.0, height=40.0):
    rows, cols = np.mgrid[0:n, 0:n].astype(float)
    c = (n - 1) / 2
    r = np.hypot(rows - c, cols - c) * res
    return height - r


def _interior(a):
    return a[1:-1, 1:-1]


class TestSlope:
    @pytest.mark.parametrize("gx, gy", [(0.5, 0.0), (0.0, -1.0), (0.3, 0.4)])
    def test_plane_slope_is_atan_of_gradient(self, gx, gy):
        z = _plane(gx=gx, gy=gy)
        expected = np.degrees(np.arctan(np.hypot(gx, gy)))
        np.testing.assert_allclose(_interior(slope_degrees(z, 2.0, 2.0)), expected, atol=1e-9)

    def test_pixel_size_is_honoured(self):
        """Same elevation array at twice the pixel size is half the gradient."""
        z = _plane(res=1.0, gx=1.0)  # rises 1 m per 1 m pixel
        s1 = _interior(slope_degrees(z, 1.0, 1.0))
        s2 = _interior(slope_degrees(z, 2.0, 2.0))
        np.testing.assert_allclose(s1, 45.0, atol=1e-9)
        np.testing.assert_allclose(s2, np.degrees(np.arctan(0.5)), atol=1e-9)

    def test_nan_propagates_to_neighbours_and_border(self):
        z = _plane(gx=0.2)
        z[10, 10] = np.nan
        s = slope_degrees(z, 2.0, 2.0)
        assert np.isnan(s[9:12, 9:12]).all()
        assert np.isnan(s[0]).all() and np.isnan(s[-1]).all()
        assert np.isnan(s[:, 0]).all() and np.isnan(s[:, -1]).all()
        assert np.isfinite(s[5, 5])

    def test_compute_slope_is_unchanged_by_the_delegation(self):
        """SlopeRampCorrection's predictor must not move: np.gradient, edges included."""
        rng = np.random.default_rng(0)
        z = rng.normal(size=(15, 12)).cumsum(0).cumsum(1)
        transform = Affine(3.0, 0, 0, 0, -2.0, 0)
        drow, dcol = np.gradient(z, 2.0, 3.0)
        legacy = np.degrees(np.arctan(np.sqrt(drow ** 2 + dcol ** 2)))
        np.testing.assert_array_equal(compute_slope(z, transform), legacy)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="slope method"):
            slope_degrees(np.zeros((3, 3)), 1, 1, method="sobel")


class TestHillshade:
    @pytest.mark.parametrize("alt", [30.0, 45.0, 70.0])
    def test_flat_is_sin_altitude(self, alt):
        hs = hillshade(np.zeros((8, 8)), 1.0, 1.0, azimuth=123.0, altitude=alt)
        np.testing.assert_allclose(_interior(hs), np.sin(np.radians(alt)), atol=1e-12)

    def test_slope_facing_the_light_is_brighter(self):
        """Light from the NW: a west-facing slope is lit, a south-facing one is not."""
        faces_west = _plane(gx=0.5)     # rises eastwards → faces west
        faces_south = _plane(gy=0.5)    # rises northwards → faces south
        lit = _interior(hillshade(faces_west, 2.0, 2.0, azimuth=315, altitude=45)).mean()
        shaded = _interior(hillshade(faces_south, 2.0, 2.0, azimuth=315, altitude=45)).mean()
        flat = np.sin(np.radians(45))
        assert lit > flat > shaded

    def test_range_is_unit_interval(self):
        hs = hillshade(_cone(), 1.0, 1.0)
        finite = hs[np.isfinite(hs)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0

    def test_multidirectional_is_less_aspect_dependent(self):
        """On a cone every aspect is present; the multi-light spread is smaller.

        The default NW arc keeps some aspect contrast on purpose; a full circle
        of lights removes it entirely on a constant-slope cone.
        """
        z = _cone()
        single = _interior(hillshade(z, 1.0, 1.0, azimuth=315))
        multi = _interior(multidirectional_hillshade(z, 1.0, 1.0))
        full_circle = _interior(multidirectional_hillshade(z, 1.0, 1.0,
                                                           azimuths=(0, 90, 180, 270)))
        assert np.nanstd(multi) < np.nanstd(single)
        assert np.nanstd(full_circle) < 0.1 * np.nanstd(multi)

    def test_multidirectional_with_one_azimuth_is_single(self):
        z = _cone()
        np.testing.assert_allclose(
            multidirectional_hillshade(z, 1.0, 1.0, azimuths=(200,)),
            hillshade(z, 1.0, 1.0, azimuth=200), equal_nan=True,
        )

    def test_empty_azimuths_raise(self):
        with pytest.raises(ValueError):
            multidirectional_hillshade(np.zeros((3, 3)), 1, 1, azimuths=())


class TestDispatchAndEncoding:
    def test_every_product_dispatches(self):
        z = _cone(21)
        for name in TERRAIN_PRODUCTS:
            assert terrain_product(name, z, 1.0, 1.0).shape == z.shape

    def test_unknown_product_raises(self):
        with pytest.raises(ValueError, match="terrain product"):
            terrain_product("aspect", np.zeros((3, 3)), 1, 1)

    def test_uint8_reserves_zero_for_nan(self):
        out = shade_to_uint8(np.array([[np.nan, 0.0, 0.5, 1.0]]))
        assert out.dtype == np.uint8
        assert out.tolist() == [[0, 1, 128, 255]]
