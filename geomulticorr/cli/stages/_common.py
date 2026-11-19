#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _common.py
# creation date: 2026-10-29.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ---------------------------------------------------------------------------- #
"""What every stage shares: the context, pair scoping and record helpers."""

from __future__ import annotations

import inspect
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable

from geomulticorr._logging import logger
from geomulticorr.cli.config import RunConfig


@dataclass
class StageContext:
    cfg: RunConfig
    run_dir: pathlib.Path
    #: Injected by tests; otherwise opened on first use.
    session: Any = None
    #: Stage-local scratch shared across stages of one ``gmc run``.
    shared: dict = field(default_factory=dict)

    @property
    def project_root(self) -> pathlib.Path:
        return pathlib.Path(self.cfg.project.path)

    def open_session(self, create: bool = False):
        if self.session is not None:
            return self.session
        root = self.project_root
        if not create and not root.exists():
            raise FileNotFoundError(
                f"project {root} does not exist; run the 'prepare' stage first"
            )
        from geomulticorr.core.session import open_gmc_session

        self.session = open_gmc_session(
            root,
            epsg_code=self.cfg.project.epsg,
            empty_geodatabase=self.cfg.project.empty_geodatabase,
        )
        return self.session

    def image_type(self, stage: str) -> str:
        return self.cfg.image_type_for(stage)

    def pz_names(self) -> list[str]:
        """The header's ``pzones``, or every pzone of the project."""
        if self.cfg.pzones:
            return list(self.cfg.pzones)
        overview = self.open_session().get_pzones_overview()
        if overview is None or not len(overview):
            return []
        return [str(n) for n in overview["pz_name"].tolist()]

    def rel(self, path) -> str:
        path = pathlib.Path(path)
        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path)


def pz_criterias(pz_name: str, extra: str | list[str] | None = "") -> list[str]:
    """Pair/thumb criterias scoped to one pzone.

    ``_search_engine`` matches every criterion (AND) against the **lowercased**
    path, so a criterion is only useful lowercased; ``/<pz>/`` is the pzone's
    folder in both layouts.
    """
    out = [f"/{pz_name.lower()}/"]
    if extra:
        extra = [extra] if isinstance(extra, str) else list(extra)
        out.extend(str(c).lower() for c in extra if str(c))
    return out


def check_kwargs(func: Callable, kwargs: dict, where: str) -> None:
    """Raise ``TypeError`` naming *where* if *func* does not accept a key."""
    params = inspect.signature(func).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return
    unknown = sorted(set(kwargs) - set(params))
    if unknown:
        raise TypeError(
            f"{where}: unknown option(s) {unknown} for {func.__name__}(); "
            f"accepted: {sorted(p for p in params if p != 'self')}"
        )


def accepted(func: Callable, kwargs: dict) -> dict:
    """The subset of *kwargs* that *func* accepts."""
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def clean(value):
    """NaN/inf → ``None`` and numpy scalars → Python, for JSON records."""
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (list, dict, str)):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


PAIR_COLUMNS: dict[str, str] = {
    "pa_status": "status",
    "pa_left_date": "left_date",
    "pa_right_date": "right_date",
    "pa_left_sensor": "left_sensor",
    "pa_right_sensor": "right_sensor",
    "pa_dt_days": "dt_days",
    "pa_ew_nmad_raw": "ew_nmad_raw",
    "pa_ns_nmad_raw": "ns_nmad_raw",
    "pa_ew_nmad_corr": "ew_nmad_corr",
    "pa_ns_nmad_corr": "ns_nmad_corr",
    "pa_cc_quality050": "cc_quality050",
}


def pairs_table(session, criterias, image_type: str | None) -> list[dict]:
    """One small row per pair, read from the Pairs layer (no raster opened)."""
    overview = session.get_pairs_overview(criterias, image_type=image_type)
    if overview is None or not len(overview):
        return []
    rows = []
    for _, row in overview.iterrows():
        out = {"pa_key": pathlib.Path(str(row["pa_path"])).name}
        for col, key in PAIR_COLUMNS.items():
            if col in row.index:
                value = clean(row[col])
                out[key] = None if value is None else _maybe_number(value)
        rows.append(out)
    return rows


def _maybe_number(value):
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return clean(number)
    return value


def thumbs_table(session, pz_name: str, image_type: str) -> list[dict]:
    """``[{date, sensor, res}]`` for the pzone's thumbs of *image_type*."""
    from geomulticorr.utils._image_types import type_column

    overview = session.get_thumbs_overview(pz_criterias(pz_name))
    if overview is None or not len(overview):
        return []
    overview = overview[type_column(overview, "th_type") == image_type]
    rows = []
    for _, row in overview.iterrows():
        rows.append(
            {
                "pz": pz_name,
                "date": str(row.get("th_date", "")),
                "sensor": str(row.get("th_sensor", "")),
                "res": _maybe_number(clean(row.get("th_res"))),
            }
        )
    return sorted(rows, key=lambda r: r["date"])


def median(values) -> float | None:
    vals = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return float(vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid]))


def add_figure_paths(rec, ctx: StageContext, label: str, written) -> None:
    """Register the files a ``save_*_figure`` call returned (dict or list)."""
    if not written:
        return
    if isinstance(written, dict):
        items = written.items()
    else:
        items = ((pathlib.Path(p).suffix.lstrip("."), p) for p in written)
    for fmt, path in items:
        if path:
            rec.artifacts[f"{label}.{fmt}"] = ctx.rel(path)


def log_section(stage: str, pz: str | None = None) -> None:
    logger.info(f"gmc ── {stage}" + (f" ── {pz}" if pz else ""))
