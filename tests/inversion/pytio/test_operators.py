#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Design-matrix operators: the incidence matrix and the temporal Laplacian.

The Laplacian is a second-derivative operator on an irregular time grid, so
its null space must contain every affine series ``a + b·t`` — on **every**
row, including the one-sided boundary stencils. That property is what makes
the smoothing prior neutral to a constant velocity; a stencil coefficient
error would show up here before it showed up in any inverted map.
"""
from __future__ import annotations

import numpy as np
import pytest

from geomulticorr.inversion.pytio import incidence_matrix, laplacian
from geomulticorr.inversion.pytio.config import T_THRESHOLD


class TestIncidenceMatrix:
    def test_forward_pair_marks_the_spanned_intervals(self):
        J = incidence_matrix(np.array([[0, 3]]), n_dates=5)
        assert J.shape == (1, 4)
        np.testing.assert_array_equal(J[0], [1, 1, 1, 0])

    def test_backward_pair_is_the_negative(self):
        J = incidence_matrix(np.array([[3, 0]]), n_dates=5)
        np.testing.assert_array_equal(J[0], [-1, -1, -1, 0])

    def test_consecutive_pairs_give_the_identity(self):
        pairs = np.array([[i, i + 1] for i in range(4)])
        np.testing.assert_array_equal(incidence_matrix(pairs, 5), np.eye(4))

    def test_row_times_increments_equals_the_pair_displacement(self):
        rng = np.random.default_rng(0)
        u = np.cumsum(rng.normal(size=6))            # cumulative series
        m = np.diff(u)                                # increments
        pairs = np.array([[0, 5], [2, 4], [5, 1]])
        J = incidence_matrix(pairs, 6)
        expected = u[pairs[:, 1]] - u[pairs[:, 0]]
        np.testing.assert_allclose(J @ m, expected)


def _grids():
    return [
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.3, 1.7, 2.0]),
        np.array([2016.6, 2017.5, 2017.53, 2018.6, 2019.5, 2020.6]),
        np.sort(np.random.default_rng(1).uniform(0, 10, size=11)),
    ]


class TestLaplacianNullSpace:
    @pytest.mark.parametrize("t", _grids())
    @pytest.mark.parametrize("scheme", ["3pt", "5pt"])
    @pytest.mark.parametrize("pond_liss", [0, 1, 2])
    def test_annihilates_every_affine_series(self, t, scheme, pond_liss):
        D = laplacian(t, scheme=scheme, pond_liss=pond_liss, first_deriv_zero=False)
        assert D.shape == (len(t), len(t))
        assert np.isfinite(D).all()
        for a, b in ((1.0, 0.0), (0.0, 1.0), (-3.2, 0.7)):
            np.testing.assert_allclose(D @ (a + b * t), 0.0, atol=1e-9)

    @pytest.mark.parametrize("t", _grids())
    def test_first_derivative_row_kills_constants_only(self, t):
        D = laplacian(t, first_deriv_zero=True)
        np.testing.assert_allclose(D @ np.ones_like(t), 0.0, atol=1e-9)
        residual = D @ t
        # row 0 is now a first derivative, so a linear series survives there …
        assert abs(residual[0]) > 1e-6
        # … and nowhere else
        np.testing.assert_allclose(residual[1:], 0.0, atol=1e-9)

    def test_curvature_is_detected(self):
        t = np.linspace(0, 5, 8)
        D = laplacian(t)
        assert np.abs(D @ t**2).max() > 0.1

    def test_needs_three_dates(self):
        with pytest.raises(ValueError, match="3 dates"):
            laplacian(np.array([0.0, 1.0]))

    def test_duplicate_dates_widen_the_stencil_without_nan(self):
        # Two acquisitions closer than T_THRESHOLD in the *middle* of a series
        # (GMC itself never produces this: image_dates is a set of day strings,
        # and one day apart is already 0.0027 yr > T_THRESHOLD).
        t = np.array([0.0, 1.0, 2.0, 3.0, 3.0 + T_THRESHOLD / 2, 4.0, 5.0, 6.0])
        for scheme in ("3pt", "5pt"):
            D = laplacian(t, scheme=scheme)
            assert np.isfinite(D).all()
            np.testing.assert_allclose(D @ np.ones_like(t), 0.0, atol=1e-9)

    def test_pond_liss_changes_the_row_scaling_on_an_irregular_grid(self):
        # On a uniform grid every mode rescales all rows by the same factor,
        # which the mean-pond normalisation divides out again — so the
        # difference only shows on an irregular grid.
        t = np.array([0.0, 0.3, 1.7, 2.0, 3.5, 3.6, 5.0])
        D0 = laplacian(t, pond_liss=0)
        D2 = laplacian(t, pond_liss=2)
        assert not np.allclose(D0, D2)
        np.testing.assert_allclose(laplacian(np.linspace(0, 3, 7), pond_liss=0),
                                   laplacian(np.linspace(0, 3, 7), pond_liss=2))
