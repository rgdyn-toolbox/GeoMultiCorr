#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# profiling.py
# creation date: 2026-10-24.
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
"""Profiling through the GeoUtils profiler (``geoutils.profiler``).

GMC's heavy methods carry ``@profile("gmc.…")`` from
:mod:`geomulticorr._profiling` (the geoutils decorator, made exception-safe).
It is a no-op until :func:`enable` turns the profiler on, so it costs nothing in
a notebook. geoutils decorates its own ``_reproject`` / ``_crop``, which then
appear nested under the GMC call that triggered them.

``generate_summary`` silently does nothing with no data; :func:`summarise`
returns the written paths so the stage record can list them.
:func:`recover_after_failure` clears the call stack after a failed stage, in
case a geoutils-decorated function (not exception-safe in 0.2.5) raised.
"""

from __future__ import annotations

import pathlib

from geomulticorr._logging import logger

PROFILING_DIRNAME = "profiling"


def enable() -> bool:
    """Turn the profiler on (graphs + raw data). ``False`` if unavailable."""
    try:
        from geoutils.profiler import Profiler

        Profiler.enable(save_graphs=True, save_raw_data=True)
    except Exception as exc:  # plotly/psutil missing, or an older geoutils
        logger.warning(f"Profiling unavailable: {exc}")
        return False
    return True


def is_enabled() -> bool:
    try:
        from geoutils.profiler import Profiler
    except ImportError:
        return False
    return bool(Profiler.enabled)


def stage_wrapper(stage: str):
    """``profile("gmc.stage.<stage>")`` — the root entry of a stage's calls."""
    from geomulticorr._profiling import profile

    return profile(f"gmc.stage.{stage}", memprof=True, interval=0.1)


def recover_after_failure() -> None:
    try:
        from geoutils.profiler import Profiler

        Profiler.running_processes.clear()
    except ImportError:
        pass


def summary_table(info=None) -> list[dict]:
    """Per-name totals: calls, total time, peak memory (MB). Pure on the frame."""
    if info is None:
        try:
            from geoutils.profiler import Profiler

            info = Profiler.get_profiling_info()
        except ImportError:
            return []
    if info is None or len(info) == 0:
        return []
    rows = []
    for name, grp in info.groupby("name", sort=False):
        peak = None
        for mem in grp["memory"]:
            if mem:
                values = [m[1] if isinstance(m, (tuple, list)) else m for m in mem]
                values = [v for v in values if v is not None]
                if values:
                    peak = max(peak or 0.0, float(max(values)))
        rows.append(
            {
                "name": str(name),
                "calls": int(len(grp)),
                "total_s": round(float(grp["time"].sum()), 3),
                "max_s": round(float(grp["time"].max()), 3),
                "peak_mb": None if peak is None else round(peak, 1),
            }
        )
    rows.sort(key=lambda r: r["total_s"], reverse=True)
    return rows


def summarise(run_dir: pathlib.Path | str) -> dict[str, str]:
    """Write ``profiling/`` (icicle graph, memory graphs, raw pickle). Never raises."""
    out = pathlib.Path(run_dir) / PROFILING_DIRNAME
    try:
        from geoutils.profiler import Profiler

        if not Profiler.enabled or len(Profiler.get_profiling_info()) == 0:
            return {}
        Profiler.generate_summary(str(out))
    except Exception as exc:
        logger.warning(f"Could not write the profiling summary: {exc}")
        return {}
    return {p.name: str(p) for p in sorted(out.glob("*")) if p.is_file()}
