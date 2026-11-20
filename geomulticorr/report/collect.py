#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# collect.py
# creation date: 2026-10-29.
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
"""Everything the report shows, read from a run folder. Pure: no raster opened.

Inputs are the files the CLI writes — ``used_config.yaml``,
``records/<stage>.json``, ``profiling/summary.json`` — so a report can be
re-rendered at any time (``gmc report RUN_DIR``) without the project's rasters
being reachable.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import yaml

from geomulticorr.cli.config import STAGE_ORDER
from geomulticorr.cli.record import read_records

STAGE_TITLES: dict[str, str] = {
    "prepare": "Project and processing zones",
    "crop": "Image archive",
    "similarity": "Image similarity",
    "pairs": "Pair network",
    "correlate": "Image correlation",
    "extract": "Raw displacements",
    "correct": "Corrections",
    "invert": "Time-series inversion",
    "fuse": "Multi-sensor fusion",
}


def collect(run_dir: pathlib.Path | str) -> dict:
    """The report context: config, ordered stage records, warnings, profiling."""
    run_dir = pathlib.Path(run_dir)
    config = _read_yaml(run_dir / "used_config.yaml")
    records = read_records(run_dir)
    ordered = [records[s] for s in STAGE_ORDER if s in records]

    warnings = Counter()
    for rec in ordered:
        for message in rec.get("warnings", []):
            warnings[(rec["stage"], message)] += 1

    return {
        "run_dir": str(run_dir),
        "config": config,
        "project": (config.get("project") or {}).get("path", ""),
        "stages": ordered,
        "titles": STAGE_TITLES,
        "n_failed": sum(1 for r in ordered if r.get("status") == "failed"),
        "total_elapsed_s": round(sum(r.get("elapsed_s", 0.0) for r in ordered), 1),
        "gmc_version": next((r.get("gmc_version") for r in ordered), ""),
        "warnings": [
            {"stage": stage, "message": msg, "count": n}
            for (stage, msg), n in warnings.most_common()
        ],
        "profiling": _read_json(run_dir / "profiling" / "summary.json") or [],
        "profiling_graph": _exists(run_dir / "profiling" / "time_graph.html"),
    }


def _read_yaml(path: pathlib.Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _read_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _exists(path: pathlib.Path) -> str | None:
    return str(path) if path.exists() else None


# ── shaping helpers the template uses ───────────────────────────────────── #


def is_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def is_cell(value) -> bool:
    """Fits a table cell: a scalar or a short list of scalars (a kernel, a box)."""
    if is_scalar(value):
        return True
    return (
        isinstance(value, (list, tuple))
        and len(value) <= 8
        and all(is_scalar(v) for v in value)
    )


def scalar_items(mapping: dict) -> list[tuple[str, object]]:
    """``(key, value)`` for the scalar (and short flat-list/dict) entries."""
    out = []
    for key, value in (mapping or {}).items():
        if is_scalar(value):
            out.append((key, value))
        elif (
            isinstance(value, dict)
            and value
            and all(is_scalar(v) for v in value.values())
        ):
            out.append((key, ", ".join(f"{k}: {fmt(v)}" for k, v in value.items())))
        elif (
            isinstance(value, list)
            and len(value) <= 8
            and all(is_scalar(v) for v in value)
        ):
            out.append((key, ", ".join(fmt(v) for v in value) or "—"))
    return out


def table(rows: list[dict], columns: list[str] | None = None, limit: int | None = None):
    """``(columns, rows-as-lists)`` for a list of flat dicts; nested values dropped."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if columns is None:
        columns = []
        for row in rows:
            for key, value in row.items():
                if key not in columns and is_cell(value):
                    columns.append(key)
    if limit is not None:
        rows = rows[:limit]
    return columns, [[row.get(c) for c in columns] for row in rows]


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1e5 or abs(value) < 1e-3:
            return f"{value:.3g}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def worst_pairs(rows: list[dict], n: int = 10) -> list[dict]:
    """Pairs with the largest corrected NMAD (max of EW and NS)."""

    def key(row):
        vals = [row.get("ew_nmad_corr"), row.get("ns_nmad_corr")]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return max(vals) if vals else -1.0

    ranked = sorted((r for r in rows if key(r) >= 0), key=key, reverse=True)
    return ranked[:n]
