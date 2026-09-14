#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _durations.py
# creation date: 2026-09-13.
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
"""Temporal-baseline duration strings, in the numpy/xarray style.

``"1Y"``, ``"6M"``, ``"2W"``, ``"30D"`` — the forms already documented by
:meth:`~geomulticorr.core.session.Session.update_pairs`, lifted out of
``session.py`` so the explorers and :mod:`geomulticorr.core.pzone` can share one
implementation. ``pzone`` cannot import from ``session`` (``session`` imports
``pzone``), which is why this lives under ``utils``.

Pure string handling — no GMC imports, no pandas, no I/O.

Public API:

- :func:`parse_dt_days` — the parser; the single source of truth.
- :func:`format_dt_days` — its inverse, for echoing a value back to a user.
- :data:`DT_UNIT_TO_DAYS` — the unit table.
"""
from __future__ import annotations

import re

#: Days per unit. Deliberately approximate — ``M`` is 30 days and ``Y`` is 365,
#: not 30.44 and 365.25.
#:
#: This is a **filter threshold**, where a 0.07 % difference from
#: :data:`~geomulticorr.correlation.corr_params.DAYS_PER_YEAR` cannot change
#: which pairs are selected in any realistic archive. Tightening it here would
#: silently shift every existing ``update_pairs`` call that passed a duration
#: string, so the approximation stays.
DT_UNIT_TO_DAYS: dict[str, int] = {"D": 1, "W": 7, "M": 30, "Y": 365}

#: ``<integer><unit>``, no sign, no whitespace, no compound forms.
_DURATION_RE = re.compile(r"(\d+)([DdWwMmYy])")


def parse_dt_days(value: int | str | None) -> int | None:
    """Convert a duration to integer days.

    Accepts an integer (passed through unchanged), ``None``, or a string of the
    form ``'<n><unit>'`` where unit is one of ``D`` (days), ``W`` (weeks),
    ``M`` (months ≈ 30 d), ``Y`` (years ≈ 365 d). Case-insensitive.

    Examples::

        parse_dt_days(30)     -> 30
        parse_dt_days("30D")  -> 30
        parse_dt_days("6M")   -> 180
        parse_dt_days("1Y")   -> 365
        parse_dt_days("2W")   -> 14

    :param value: Integer days, a duration string, or None.
    :returns: Days as an integer, or None when *value* was None.
    :raises ValueError: On anything else — a bare numeric string (the unit is
        mandatory), a compound form like ``"1Y6M"``, a negative value, or a
        float. Raising is deliberate: silently reinterpreting an unparseable
        threshold would filter a pair network in a way the caller did not ask
        for.
    """
    if value is None or isinstance(value, int):
        return value
    match = _DURATION_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(
            f"Cannot parse duration '{value}'. "
            "Use an integer (days) or a string like '30D', '6M', '1Y', '2W'."
        )
    n, unit = int(match.group(1)), match.group(2).upper()
    return n * DT_UNIT_TO_DAYS[unit]


def format_dt_days(days: int | float | None) -> str:
    """Render *days* as the most readable duration string.

    The rough inverse of :func:`parse_dt_days`, used to echo a resolved value
    back to a user — a widget that was handed ``365`` should be able to show
    ``"1Y"``. Only exact multiples convert; anything else stays in days, so the
    round-trip never loses precision.

    :param days: Days, or None.
    :returns: ``"1Y"``, ``"6M"``, ``"2W"``, ``"45D"`` … or ``""`` for None.
    """
    if days is None:
        return ""
    value = int(days)
    if value <= 0:
        return str(value)
    for unit in ("Y", "M", "W"):
        size = DT_UNIT_TO_DAYS[unit]
        if value % size == 0:
            return f"{value // size}{unit}"
    return f"{value}D"
