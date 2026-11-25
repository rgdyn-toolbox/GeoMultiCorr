#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _derived.py
# creation date: 2026-11-25.
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
"""Derived rasters of a corrected pair: magnitude and velocities.

The formulas live here once. ``Pair.get_interesting_geoim`` (raw maps, modes
``m`` / ``vm``) and ``Pair.write_derived`` (corrected maps, written to disk)
both call them, so the two can never drift apart.

The functions take anything that supports ``*``, ``+``, ``**`` and ``/``:
numpy arrays, masked arrays, or :class:`geoutils.Raster`.
"""

from __future__ import annotations

#: ``{product: file stem after "<pa_key>-F_"}``. The file is ``<pa_key>-F_<stem>.tif``,
#: next to ``-F_EW_corr.tif`` and ``-F_NS_corr.tif``.
DERIVED_PRODUCTS: dict[str, str] = {
    "magnitude": "MAG_corr",
    "velocity": "VEL_corr",
    "ew_velocity": "EW_VEL_corr",
    "ns_velocity": "NS_VEL_corr",
}

#: Components each product needs (``"ew"`` / ``"ns"``): a pixel is nodata in the
#: output when it is nodata in any of them.
DERIVED_INPUTS: dict[str, tuple[str, ...]] = {
    "magnitude": ("ew", "ns"),
    "velocity": ("ew", "ns"),
    "ew_velocity": ("ew",),
    "ns_velocity": ("ns",),
}

#: Nodata of the derived rasters. The corrected maps use 0, which is also a
#: legitimate velocity; a distinct value keeps "no data" and "no motion" apart.
DERIVED_NODATA: float = -9999.0


def magnitude(x, y):
    """Horizontal magnitude ``sqrt(x² + y²)``."""
    return (x * x + y * y) ** 0.5


def velocity(displacement, dt_years: float):
    """Displacement per year: ``displacement / dt_years``."""
    return displacement / dt_years


def check_products(products) -> tuple[str, ...]:
    """Validate and normalise a product selection.

    :param products: One product name or an iterable of them.
    :returns: The names as a tuple, in the given order, without duplicates.
    :raises ValueError: A name is not in :data:`DERIVED_PRODUCTS`, or none is given.
    """
    if isinstance(products, str):
        products = (products,)
    names = tuple(dict.fromkeys(products))
    unknown = [p for p in names if p not in DERIVED_PRODUCTS]
    if unknown or not names:
        raise ValueError(
            f"Unknown derived product(s) {unknown or list(products)!r}; "
            f"choose among {sorted(DERIVED_PRODUCTS)}."
        )
    return names


def compute_derived(products, ew, ns, dt_years: float) -> dict:
    """The requested products, from corrected EW / NS arrays.

    :param products: Names from :data:`DERIVED_PRODUCTS`.
    :param ew: EW displacement (metres), masked where there is no data.
    :param ns: NS displacement (metres), masked where there is no data.
    :param dt_years: Temporal baseline of the pair, in years (> 0).
    :returns: ``{product: array}``; masked wherever an input it needs is masked.
    """
    out = {}
    for product in products:
        if product == "magnitude":
            out[product] = magnitude(ew, ns)
        elif product == "velocity":
            out[product] = velocity(magnitude(ew, ns), dt_years)
        elif product == "ew_velocity":
            out[product] = velocity(ew, dt_years)
        elif product == "ns_velocity":
            out[product] = velocity(ns, dt_years)
        else:  # pragma: no cover - check_products rejects it first
            raise ValueError(f"Unknown derived product {product!r}")
    return out
