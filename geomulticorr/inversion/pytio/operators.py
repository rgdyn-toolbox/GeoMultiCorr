#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# operators.py
# creation date: 2026-09-14.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------- #
"""Design-matrix operators: incidence matrix and temporal Laplacian.

Faithful transcription of `matrice_base` and `matrice_px` from
`invers_pixel_brit_omp.f90` (chronological ordering, itri=0).
"""

import numpy as np

from .config import T_THRESHOLD


def incidence_matrix(pair_idx: np.ndarray, n_dates: int) -> np.ndarray:
    """Signed incidence matrix J (n_pairs, n_dates-1).

    pair_idx[k] = (i1, i2): indices of the first- and second-listed dates of
    pair k; the pair measures d_k = u(t_i2) - u(t_i1). Row k holds +1 on the
    intervals i1..i2-1 when i1 < i2, and -1 on i2..i1-1 otherwise.
    """
    J = np.zeros((len(pair_idx), n_dates - 1), dtype=np.float64)
    for k, (i1, i2) in enumerate(pair_idx):
        if i1 < i2:
            J[k, i1:i2] = 1.0
        else:
            J[k, i2:i1] = -1.0
    return J


def _first_row(t, pond_liss, first_deriv_zero):
    """Row 0 of the Laplacian: first-derivative or one-sided 2nd derivative."""
    row = np.zeros(len(t))
    if first_deriv_zero:  # ider_zero == 0
        dt1 = t[1] - t[0]
        row[0], row[1] = -1.0 / dt1**2, 1.0 / dt1**2
        mean_dt = dt1
    else:
        dt1, dt2, dt3 = t[1] - t[0], t[2] - t[0], t[2] - t[1]
        row[0] = -(1.0 / (dt2 * dt3) - 1.0 / (dt1 * dt3))
        row[1] = -1.0 / (dt1 * dt3)
        row[2] = 1.0 / (dt2 * dt3)
        mean_dt = (dt1 + dt3) / 2.0
    return row, mean_dt


def _last_row(t):
    """Row S-1: one-sided second derivative on the last three epochs."""
    S = len(t)
    row = np.zeros(S)
    dt1, dt2, dt3 = t[-1] - t[-2], t[-1] - t[-3], t[-2] - t[-3]
    row[-2] = -1.0 / (dt1 * dt3)
    row[-1] = -1.0 / (dt2 * dt3) + 1.0 / (dt1 * dt3)
    row[-3] = 1.0 / (dt2 * dt3)
    return row, (dt1 + dt3) / 2.0


def _centered_3pt(t, s):
    row = np.zeros(len(t))
    dt1, dt2, dt3 = t[s] - t[s - 1], t[s + 1] - t[s - 1], t[s + 1] - t[s]
    row[s - 1] = 1.0 / (dt2 * dt1)
    row[s] = -1.0 / (dt2 * dt1) - 1.0 / (dt2 * dt3)
    row[s + 1] = 1.0 / (dt2 * dt3)
    return row, (dt1 + dt3) / 2.0


def _apply_pond(row, mean_dt, pond_liss):
    """Scale one row per ipondliss; return (scaled_row, pondliss_increment)."""
    if pond_liss == 0:
        return row * mean_dt**2, 1.0
    if pond_liss == 2:
        return row * abs(mean_dt), 1.0 / abs(mean_dt)
    return row, 1.0 / mean_dt**2


def _degenerate_row(t, s, pond_liss):
    """Same-date successor (dt3 < threshold): first-derivative penalty."""
    row = np.zeros(len(t))
    dt3 = t[s + 1] - t[s]
    if pond_liss == 0:
        row[s], row[s + 1] = -3.0, 3.0
    elif pond_liss == 2:
        row[s], row[s + 1] = -3.0 / dt3, 3.0 / dt3
    else:
        row[s], row[s + 1] = -1.0 / dt3**2, 1.0 / dt3**2
    return row


def laplacian(t: np.ndarray, scheme: str = "5pt", pond_liss: int = 2,
              first_deriv_zero: bool = False) -> np.ndarray:
    """Second-derivative operator D2 (S, S) on the irregular grid t.

    Transcribes the `matd` construction of `matrice_px` (both iliss schemes,
    all ipondliss modes, duplicate-date stencil widening).
    """
    t = np.asarray(t, dtype=np.float64)
    S = len(t)
    if S < 3:
        raise ValueError("need at least 3 dates for the smoothing operator")
    D = np.zeros((S, S))
    pond = 0.0

    row, mdt = _first_row(t, pond_liss, first_deriv_zero)
    D[0], p = _apply_pond(row, mdt, pond_liss)
    pond += p

    if scheme == "3pt":
        interior = range(1, S - 1)
    else:
        # 5pt: rows 1 and S-2 use the centered 3-point stencil (for S == 3 both
        # are the same row and, like the Fortran, pondliss is accumulated twice)
        D[1], p = _apply_pond(*_centered_3pt(t, 1), pond_liss)
        pond += p
        D[S - 2], p = _apply_pond(*_centered_3pt(t, S - 2), pond_liss)
        pond += p
        interior = range(2, S - 2)

    for s in interior:
        dt3 = t[s + 1] - t[s]
        if dt3 < T_THRESHOLD:
            D[s] = _degenerate_row(t, s, pond_liss)
            continue
        if scheme == "3pt":
            dt1 = t[s] - t[s - 1]
            s_avt = s - 1
            if dt1 < T_THRESHOLD:
                s_avt = s - 2
                dt1 = t[s] - t[s - 2]
            dt2 = t[s + 1] - t[s_avt]
            row = np.zeros(S)
            row[s_avt] = 1.0 / (dt2 * dt1)
            row[s] = -1.0 / (dt2 * dt1) - 1.0 / (dt2 * dt3)
            row[s + 1] = 1.0 / (dt2 * dt3)
            D[s], p = _apply_pond(row, (dt1 + dt3) / 2.0, pond_liss)
            pond += p
        else:
            # 5-point stencil with duplicate-date index shifts
            s_avt, s_avt2 = s - 1, s - 2
            if t[s] - t[s - 1] < T_THRESHOLD:
                s_avt, s_avt2 = s - 2, s - 3
            if t[s_avt] - t[s_avt2] < T_THRESHOLD:
                s_avt2 -= 1
            s_apr2 = s + 2
            if t[s + 2] - t[s + 1] < T_THRESHOLD:
                s_apr2 = s + 3
            dxm2, dxm1 = t[s_avt2] - t[s], t[s_avt] - t[s]
            dx1, dx2 = t[s + 1] - t[s], t[s_apr2] - t[s]
            adt = -(dxm1 * dx1 + dxm1 * dx2 + dx1 * dx2) / (
                dxm2 * (dx2 - dxm2) * (dx1 - dxm2) * (dxm1 - dxm2))
            bdt = -(dxm2 * dx1 + dxm2 * dx2 + dx1 * dx2) / (
                dxm1 * (dx2 - dxm1) * (dx1 - dxm1) * (dxm2 - dxm1))
            cdt = -(dxm2 * dxm1 + dxm2 * dx2 + dxm1 * dx2) / (
                dx1 * (dx2 - dx1) * (dxm1 - dx1) * (dxm2 - dx1))
            ddt = -(dxm2 * dxm1 + dxm2 * dx1 + dxm1 * dx1) / (
                dx2 * (dx1 - dx2) * (dxm1 - dx2) * (dxm2 - dx2))
            row = np.zeros(S)
            row[s_avt2], row[s_avt] = adt, bdt
            row[s] = -(adt + bdt) - (cdt + ddt)
            row[s + 1], row[s_apr2] = cdt, ddt
            D[s], p = _apply_pond(row, (t[s_apr2] - t[s_avt2]) / 4.0, pond_liss)
            pond += p

    row, mdt = _last_row(t)
    D[S - 1], p = _apply_pond(row, mdt, pond_liss)
    pond += p

    return D / (pond / S)
