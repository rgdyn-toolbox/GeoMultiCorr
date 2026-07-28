#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the pure pair-layout geometry helpers (_pairs_geometry)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomulticorr.utils._pairs_geometry import (
    DEFAULT_LINK_RADIUS,
    CircularTimeScale,
    arc_heights,
    arc_ylim,
    bezier_sample,
    chord_control_points,
    chord_polylines,
    color_ring_mesh,
    polar_to_xy,
    polyline_point_and_tangent,
    ring_centerline,
    tangent_marker_angles,
    timeline_arc_control_points,
    timeline_arc_polylines,
    to_decimal_year,
    year_label_placement,
    year_tick_segments,
)


def _legacy_to_dec(date) -> float:
    """The inline formula that used to be duplicated across session.py."""
    dt = pd.Timestamp(date)
    yr = dt.year
    return yr + (dt - pd.Timestamp(yr, 1, 1)) / pd.Timedelta(days=365.25)


class TestToDecimalYear:
    def test_jan_first_is_exact_year(self):
        assert to_decimal_year("2021-01-01") == pytest.approx(2021.0, abs=1e-12)

    def test_scalar_returns_float(self):
        out = to_decimal_year("2021-06-15")
        assert isinstance(out, float)

    def test_sequence_returns_array(self, dates_iso):
        out = to_decimal_year(dates_iso)
        assert isinstance(out, np.ndarray)
        assert out.shape == (len(dates_iso),)

    def test_input_types_agree(self):
        as_str = to_decimal_year("2021-06-15")
        as_ts = to_decimal_year(pd.Timestamp("2021-06-15"))
        as_np = to_decimal_year(np.datetime64("2021-06-15"))
        assert as_str == pytest.approx(as_ts) == pytest.approx(as_np)

    def test_matches_legacy_formula(self, dates_iso):
        """Regression lock — existing figures must not shift."""
        expected = np.array([_legacy_to_dec(d) for d in dates_iso])
        np.testing.assert_allclose(to_decimal_year(dates_iso), expected, rtol=0, atol=1e-9)

    def test_monotonic(self, dates_iso):
        dec = to_decimal_year(dates_iso)
        assert np.all(np.diff(dec) > 0)


class TestCircularTimeScale:
    def test_spans_whole_years(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        assert scale.start == pd.Timestamp("2020-01-01")
        assert scale.end == pd.Timestamp("2023-01-01")

    def test_jan_first_at_top(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        assert scale.angle([scale.start])[0] == pytest.approx(np.pi / 2)

    def test_fraction_in_unit_range_and_monotonic(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        frac = scale.fraction(dates_iso)
        assert np.all((frac >= 0) & (frac < 1))
        assert np.all(np.diff(frac) > 0)

    def test_angle_decreases_clockwise(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        assert np.all(np.diff(scale.angle(dates_iso)) < 0)

    def test_years_inclusive(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        np.testing.assert_array_equal(scale.years, np.array([2020, 2021, 2022]))

    def test_year_angles_one_per_year(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        assert scale.year_angles().shape == (3,)

    def test_single_date_spans_one_year(self):
        scale = CircularTimeScale.from_dates(["2021-07-04"])
        assert scale.start == pd.Timestamp("2021-01-01")
        assert scale.end == pd.Timestamp("2022-01-01")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty date list"):
            CircularTimeScale.from_dates([])


class TestPolarToXY:
    def test_top_of_circle(self):
        x, y = polar_to_xy(1.0, np.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(1.0)

    def test_broadcasts(self):
        x, y = polar_to_xy(2.0, np.array([0.0, np.pi / 2, np.pi]))
        np.testing.assert_allclose(np.hypot(x, y), 2.0)


class TestBezierSample:
    def test_endpoints_exact_quadratic(self):
        net = np.array([[[0.0, 0.0], [1.0, 2.0], [3.0, 0.0]]])
        curve = bezier_sample(net, n_points=9)
        np.testing.assert_allclose(curve[0, 0], net[0, 0])
        np.testing.assert_allclose(curve[0, -1], net[0, -1])

    def test_endpoints_exact_cubic(self):
        net = np.array([[[0.0, 0.0], [0.0, 1.0], [3.0, 1.0], [3.0, 0.0]]])
        curve = bezier_sample(net, n_points=9)
        np.testing.assert_allclose(curve[0, 0], [0.0, 0.0])
        np.testing.assert_allclose(curve[0, -1], [3.0, 0.0])

    def test_midpoint_control_gives_straight_line(self):
        p0, p1 = np.array([0.0, 0.0]), np.array([4.0, 2.0])
        net = np.array([[p0, (p0 + p1) / 2, p1]])
        curve = bezier_sample(net, n_points=11)[0]
        # every sample lies on the p0→p1 segment (2-D cross product == 0)
        d, seg = curve - p0, p1 - p0
        cross = d[:, 0] * seg[1] - d[:, 1] * seg[0]
        np.testing.assert_allclose(cross, 0.0, atol=1e-12)

    def test_output_shape(self):
        net = np.zeros((7, 3, 2))
        assert bezier_sample(net, n_points=12).shape == (7, 12, 2)

    def test_single_net_promoted(self):
        assert bezier_sample(np.zeros((3, 2)), n_points=5).shape == (1, 5, 2)

    def test_bad_degree_raises(self):
        with pytest.raises(ValueError, match="quadratic"):
            bezier_sample(np.zeros((2, 5, 2)))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(m, k, 2\)"):
            bezier_sample(np.zeros((2, 3, 4)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            bezier_sample(np.zeros((1, 3, 2)), n_points=1)


class TestChordGeometry:
    def test_identical_angles_control_on_rim(self):
        net = chord_control_points([0.3], [0.3])
        r_ctrl = np.hypot(*net[0, 1])
        assert r_ctrl == pytest.approx(DEFAULT_LINK_RADIUS)

    def test_antipodal_control_pulled_by_curvature(self):
        net = chord_control_points([0.0], [np.pi], curvature=0.85)
        r_ctrl = np.hypot(*net[0, 1])
        assert r_ctrl == pytest.approx(DEFAULT_LINK_RADIUS * (1 - 0.85))

    def test_endpoints_on_link_radius(self):
        net = chord_control_points([0.0, 1.0, 2.0], [3.0, 4.0, 5.0], link_radius=0.9)
        np.testing.assert_allclose(np.hypot(net[:, 0, 0], net[:, 0, 1]), 0.9)
        np.testing.assert_allclose(np.hypot(net[:, 2, 0], net[:, 2, 1]), 0.9)

    def test_swap_symmetry(self):
        a = chord_control_points([0.4], [2.1])
        b = chord_control_points([2.1], [0.4])
        # same curve, endpoints reversed
        np.testing.assert_allclose(a[0, 1], b[0, 1])
        np.testing.assert_allclose(a[0, 0], b[0, 2])

    def test_all_samples_inside_disc(self):
        ti = np.linspace(0, 2 * np.pi, 25)
        tj = ti[::-1]
        curves = chord_polylines(ti, tj, n_points=24)
        radii = np.hypot(curves[..., 0], curves[..., 1])
        assert radii.max() <= DEFAULT_LINK_RADIUS + 1e-12

    def test_polyline_shape(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        theta = scale.angle(dates_iso)
        curves = chord_polylines(theta[:-1], theta[1:], n_points=16)
        assert curves.shape == (len(dates_iso) - 1, 16, 2)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            chord_control_points([0.0, 1.0], [0.0])


class TestPointAndTangent:
    def test_straight_line_direction(self):
        curve = np.linspace([0.0, 0.0], [4.0, 0.0], 11)[None, :, :]
        pt, tan = polyline_point_and_tangent(curve, position=0.5)
        np.testing.assert_allclose(pt, [[2.0, 0.0]])
        np.testing.assert_allclose(tan, [[1.0, 0.0]])

    def test_reversed_line_flips_direction(self):
        curve = np.linspace([4.0, 0.0], [0.0, 0.0], 11)[None, :, :]
        _, tan = polyline_point_and_tangent(curve)
        np.testing.assert_allclose(tan, [[-1.0, 0.0]])

    def test_position_selects_endpoints(self):
        curve = np.linspace([0.0, 0.0], [4.0, 2.0], 9)[None, :, :]
        start, _ = polyline_point_and_tangent(curve, position=0.0)
        end, _ = polyline_point_and_tangent(curve, position=1.0)
        np.testing.assert_allclose(start, [[0.0, 0.0]])
        np.testing.assert_allclose(end, [[4.0, 2.0]])

    def test_tangents_are_unit_on_real_chords(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        theta = scale.angle(dates_iso)
        curves = chord_polylines(theta[:-1], theta[1:], n_points=16)
        _, tan = polyline_point_and_tangent(curves)
        np.testing.assert_allclose(np.hypot(tan[:, 0], tan[:, 1]), 1.0)

    def test_points_lie_on_the_curve(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        theta = scale.angle(dates_iso)
        curves = chord_polylines(theta[:-1], theta[1:], n_points=17)
        pts, _ = polyline_point_and_tangent(curves, position=0.5)
        np.testing.assert_allclose(pts, curves[:, 8, :])

    def test_degenerate_curve_gives_zero_tangent(self):
        curve = np.zeros((1, 8, 2))
        pt, tan = polyline_point_and_tangent(curve)
        np.testing.assert_allclose(pt, [[0.0, 0.0]])
        np.testing.assert_allclose(tan, [[0.0, 0.0]])
        assert not np.isnan(tan).any()

    def test_repeated_sample_widens_stencil(self):
        """A duplicated sample at the midpoint must not produce a zero tangent."""
        curve = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        _, tan = polyline_point_and_tangent(curve, position=0.5)
        np.testing.assert_allclose(tan, [[1.0, 0.0]])

    def test_out_of_range_position_raises(self):
        with pytest.raises(ValueError, match=r"position must lie in \[0, 1\]"):
            polyline_point_and_tangent(np.zeros((1, 4, 2)), position=1.5)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(m, n, 2\)"):
            polyline_point_and_tangent(np.zeros((2, 4, 3)))


class TestMarkerAngles:
    @pytest.mark.parametrize(
        "tangent, expected",
        [((0, 1), 0.0), ((1, 0), 90.0), ((0, -1), 180.0), ((-1, 0), 270.0)],
    )
    def test_cardinal_directions(self, tangent, expected):
        assert tangent_marker_angles([tangent])[0] == pytest.approx(expected)

    def test_diagonal(self):
        d = 1 / np.sqrt(2)
        assert tangent_marker_angles([(d, d)])[0] == pytest.approx(45.0)

    def test_range_is_wrapped(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        theta = scale.angle(dates_iso)
        _, tan = polyline_point_and_tangent(chord_polylines(theta[:-1], theta[1:]))
        angles = tangent_marker_angles(tan)
        assert np.all((angles >= 0) & (angles < 360))

    def test_zero_tangent_maps_to_zero(self):
        assert tangent_marker_angles([(0.0, 0.0)])[0] == 0.0


class TestArcHeights:
    def test_unmirrored_matches_legacy_expression(self):
        dt = np.array([0.0, 0.05, 1.0, 3.0])
        np.testing.assert_allclose(
            arc_heights(dt, mirror=False), np.maximum(dt * 0.5, 0.05)
        )

    def test_backward_entries_negated(self):
        dt = np.array([1.0, 2.0, 3.0])
        back = np.array([False, True, False])
        h = arc_heights(dt, backward=back)
        np.testing.assert_allclose(h, [0.5, -1.0, 1.5])

    def test_min_height_floors_magnitude(self):
        h = arc_heights([0.0], backward=[True])
        assert h[0] == pytest.approx(-0.05)

    def test_mirror_off_ignores_backward_mask(self):
        h = arc_heights([2.0], backward=[True], mirror=False)
        assert h[0] > 0

    def test_no_mask_keeps_all_positive(self):
        assert np.all(arc_heights([1.0, 2.0]) > 0)


class TestArcYlim:
    def test_positive_heights_reproduce_legacy_range(self):
        """Regression lock on the pre-mirror layout."""
        h = np.array([0.05, 0.5, 1.2])
        assert arc_ylim(h) == pytest.approx((-0.05, 1.2 * 0.85))

    def test_all_negative_heights_still_valid(self):
        bottom, top = arc_ylim(np.array([-0.5, -1.2]))
        assert bottom < top
        assert bottom == pytest.approx(-1.2 * 0.85)
        assert top == pytest.approx(0.05)

    def test_mixed_heights_straddle_zero(self):
        bottom, top = arc_ylim(np.array([-0.8, 0.4]))
        assert bottom < 0 < top

    def test_floor_respected_for_tiny_heights(self):
        bottom, top = arc_ylim(np.array([0.01]))
        assert bottom <= -0.05
        assert top >= 0.05


class TestTimelineArcs:
    def test_matches_legacy_control_net(self):
        """Regression lock against the previous inline _draw_arc verts."""
        x1, x2, h = 2020.5, 2021.75, 0.62
        legacy = np.array([(x1, 0.0), (x1, h), (x2, h), (x2, 0.0)])
        np.testing.assert_allclose(timeline_arc_control_points(x1, x2, h)[0], legacy)

    def test_endpoints_on_baseline(self):
        curves = timeline_arc_polylines([0.0], [2.0], [1.0], n_points=21)
        assert curves[0, 0, 1] == pytest.approx(0.0)
        assert curves[0, -1, 1] == pytest.approx(0.0)

    def test_apex_below_control_height(self):
        h = 1.0
        curves = timeline_arc_polylines([0.0], [2.0], [h], n_points=101)
        # a cubic with two control points at h peaks at 3h/4
        assert curves[0, :, 1].max() == pytest.approx(0.75 * h, rel=1e-3)

    def test_height_broadcasts(self):
        nets = timeline_arc_control_points([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], 0.5)
        assert nets.shape == (3, 4, 2)
        np.testing.assert_allclose(nets[:, 1, 1], 0.5)


class TestRing:
    def test_mesh_shapes(self):
        X, Y, frac = color_ring_mesh(n_segments=36)
        assert X.shape == Y.shape == (2, 37)
        assert frac.shape == (36,)

    def test_mesh_radii_within_ring(self):
        X, Y, _ = color_ring_mesh(n_segments=36, outer_radius=1.0, ring_width=0.045)
        radii = np.hypot(X, Y)
        np.testing.assert_allclose(radii[0], 0.955)
        np.testing.assert_allclose(radii[1], 1.0)

    def test_mesh_frac_increasing_in_unit_range(self):
        _, _, frac = color_ring_mesh(n_segments=50)
        assert np.all(np.diff(frac) > 0)
        assert frac.min() > 0 and frac.max() < 1

    def test_centerline_on_mid_radius(self):
        x, y, frac = ring_centerline(n_points=64, outer_radius=1.0, ring_width=0.04)
        np.testing.assert_allclose(np.hypot(x, y), 0.98)
        assert frac.shape == (64,)
        assert frac[0] == pytest.approx(0.0)

    def test_centerline_starts_at_top(self):
        x, y, _ = ring_centerline(n_points=8)
        assert x[0] == pytest.approx(0.0, abs=1e-12)
        assert y[0] > 0


class TestYearDecorations:
    def test_tick_segments_shape(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        segs = year_tick_segments(scale)
        assert segs.shape == (3, 2, 2)

    def test_tick_straddles_ring_center(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        segs = year_tick_segments(scale, outer_radius=1.0, ring_width=0.045, tick_length=0.075)
        r1 = np.hypot(*segs[0, 0])
        r2 = np.hypot(*segs[0, 1])
        assert r1 == pytest.approx(1.0 - 0.045 / 2 - 0.075 / 2)
        assert r2 == pytest.approx(1.0 - 0.045 / 2 + 0.075 / 2)

    def test_label_placement_shapes(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        x, y, rot, ha = year_label_placement(scale)
        assert x.shape == y.shape == rot.shape == (3,)
        assert len(ha) == 3

    def test_label_rotation_kept_readable(self):
        """Labels are flipped in the left half so they never read upside-down."""
        scale = CircularTimeScale(pd.Timestamp("2000-01-01"), pd.Timestamp("2012-01-01"))
        _, _, rot, ha = year_label_placement(scale)
        assert np.all((rot >= -90) & (rot <= 90))
        assert set(ha) <= {"left", "right"}

    def test_right_half_left_aligned(self, dates_iso):
        scale = CircularTimeScale.from_dates(dates_iso)
        x, _, _, ha = year_label_placement(scale)
        for xi, h in zip(x, ha):
            if xi > 1e-9:
                assert h == "left"
