#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# __init__.py
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
"""pytio — Python port of the NSBAS/TIO per-pixel time-series inversion.

A numpy + xarray + dask re-implementation of ``invers_pixel_brit_omp.f90``
(M.-P. Doin, CNRS/ISTerre; optical variant of Bontemps et al. 2018) and of its
post-processor ``lect_depl_cumule_lin.f``. Validated against the Fortran on a
1900×2100 px, 42-pair scene: ``depl_cumule`` agrees to float32 rounding
(relative RMS ≈ 4·10⁻⁷) with identical no-data masks.

It is the ``launch(mode="python")`` backend of
:class:`~geomulticorr.inversion.tio_inversion.TIOInversion`, and
:class:`TIOConfig` is the single solver configuration both backends read —
:func:`input_tio_text` renders it into the text the Fortran consumes.

Algorithm notes: ``docs/tio_algorithm_guide.md`` and
``docs/tio_inversion_by_hand.md``.

Typical standalone use::

    from geomulticorr.inversion.pytio import TIOConfig, io, invert_stack, fit_velocity

    imgs = io.read_image_list("liste_image_inv")
    d1, d2, w = io.read_pair_list("liste_couple")
    stack = io.load_pair_stack("LN_DATA", d1, d2)
    ds = invert_stack(stack, imgs["dates"], imgs["t"], TIOConfig(),
                      pair_weights=w).compute()
    io.write_depl_cumule(ds.cum_disp, "depl_cumule")
    vel = fit_velocity(ds.cum_disp, t=imgs["elapsed"])
"""

from . import io
from .config import (IPONDER, IPONDER_LINE_INDEX, NODATA_OUT, TIOConfig,
                     input_tio_text)
from .inversion import (common_mask, invert_stack, image_equalization_weights,
                        variance_weights)
from .operators import incidence_matrix, laplacian
from .postprocess import fit_velocity

__all__ = ["TIOConfig", "input_tio_text", "IPONDER", "IPONDER_LINE_INDEX",
           "NODATA_OUT", "io", "invert_stack", "common_mask",
           "variance_weights", "image_equalization_weights",
           "incidence_matrix", "laplacian", "fit_velocity"]
