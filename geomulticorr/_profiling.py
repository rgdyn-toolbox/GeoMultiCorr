#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# _profiling.py
# creation date: 2026-10-30.
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
"""``@profile`` — the GeoUtils profiler decorator, made exception-safe.

Usage, on any GMC function or method::

    from geomulticorr._profiling import profile

    @profile("gmc.session.apply_pairs_corrections", memprof=True)
    def apply_pairs_corrections(self, ...): ...

It records into ``geoutils.profiler.Profiler`` exactly like
``geoutils.profiler.profile`` and is a no-op until ``Profiler.enable(...)``
is called (the CLI's ``--profile``), so decorated methods cost nothing in a
notebook.

Why not use ``geoutils.profiler.profile`` directly: in geoutils 0.2.5, when the
decorated function **raises**, the wrapper neither pops the call from
``Profiler.running_processes`` (every later call is then filed under the dead
one) nor stops the ``memprof`` sampling thread — a non-daemon thread polling
forever, so the interpreter never exits. Here the function's exception is
caught *inside* the geoutils wrapper, handed out as a value, and re-raised
outside it, so the wrapper always completes its bookkeeping.

``functools.wraps`` is preserved, so ``inspect.signature`` still sees the real
parameters (the CLI's signature guard tests rely on it).
"""

from __future__ import annotations

from functools import wraps

try:
    from geoutils import profiler as _gu_profiler
except ImportError:  # pragma: no cover - geoutils is a hard dependency
    _gu_profiler = None


class _Raised:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def profile(name: str, memprof: bool = False, interval: float = 0.05):
    """Profile the decorated callable under *name* when profiling is enabled."""

    def decorator(func):
        if _gu_profiler is None:
            return func

        def _safe(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseException as exc:  # re-raised below, outside geoutils
                return _Raised(exc)

        profiled = _gu_profiler.profile(name, interval=interval, memprof=memprof)(_safe)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _gu_profiler.Profiler.enabled:
                return func(*args, **kwargs)
            out = profiled(*args, **kwargs)
            if isinstance(out, _Raised):
                raise out.exc
            return out

        return wrapper

    return decorator
