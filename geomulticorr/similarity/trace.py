#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# trace.py
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
"""
The similarity trace: one JSON file holding every similarity run of a pzone.

Pure Python/pandas, no GMC imports. Layout::

    {"format": 1, "pz_name": ..., "image_type": ..., "written_utc": ...,
     "runs": {"<params_hash>": {"params": {...},
                                "score_columns": {"trial_peak_median": true, ...},
                                "updated_utc": ...,
                                "couples": [{...one row per couple...}]}}}

A *run* is every couple measured with the same parameters (``params_hash``).
Re-running with the same parameters reuses its couples and only measures new
ones; other parameters start another run, and earlier runs are kept.

``score`` is recomputed over the **whole run** whenever couples are merged in:
it is a percentile, so adding thumbs shifts every couple's score slightly.
That is the price of a threshold meaning "better than x % of this dataset".

NaN is written as ``null`` — Python's ``json`` would otherwise emit ``NaN``,
which is not JSON and which other readers reject.
"""

from __future__ import annotations

import json
import math
import pathlib
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from geomulticorr.similarity.selection import DEFAULT_SCORE_COLUMNS, rank_score

#: Version of the layout above.
TRACE_FORMAT = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    """Plain JSON value: NaN/inf → None, numpy scalars → Python scalars."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if value is pd.NaT or value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def empty_trace(pz_name: str, image_type: str) -> dict:
    """A trace with no run yet."""
    return {
        "format": TRACE_FORMAT,
        "pz_name": pz_name,
        "image_type": image_type,
        "written_utc": None,
        "runs": {},
    }


def read_trace(path: pathlib.Path | str) -> dict:
    """Read a trace. Raises ``FileNotFoundError`` / ``ValueError`` (bad JSON or
    unknown format), so a caller can tell "absent" from "unreadable"."""
    path = pathlib.Path(path)
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("format") != TRACE_FORMAT:
        raise ValueError(f"{path} is not a similarity trace (format {TRACE_FORMAT})")
    doc.setdefault("runs", {})
    return doc


def write_trace(path: pathlib.Path | str, doc: Mapping) -> pathlib.Path | None:
    """Write *doc* as indented JSON. **Never raises**: a trace is a convenience
    and losing it must not cost the computation; returns None on failure."""
    path = pathlib.Path(path)
    doc = dict(doc, written_utc=_now())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2, allow_nan=False)
        tmp.replace(path)
    except (OSError, TypeError, ValueError):
        return None
    return path


def run_frame(doc: Mapping, params_hash: str | None = None) -> pd.DataFrame:
    """The couples of one run as a frame (``None`` → the most recently updated run,
    by the run's ``seq`` counter — timestamps tie within a second).

    Raises ``KeyError`` when the trace has no such run.
    """
    runs = doc.get("runs", {})
    if not runs:
        raise KeyError("the similarity trace holds no run")
    if params_hash is None:
        params_hash = max(runs, key=lambda h: runs[h].get("seq", 0))
    if params_hash not in runs:
        raise KeyError(f"no similarity run {params_hash!r}; have {sorted(runs)}")
    frame = pd.DataFrame(runs[params_hash].get("couples", []))
    return frame.replace({None: np.nan}).infer_objects()


def merge_run(
    doc: dict,
    params_hash: str,
    params: Mapping,
    new: pd.DataFrame,
    score_columns: Mapping[str, bool] = DEFAULT_SCORE_COLUMNS,
) -> pd.DataFrame:
    """Merge *new* couples into run *params_hash* of *doc* (in place) and
    rescore the whole run. A couple measured again replaces its old row.

    :returns: The full run after the merge, with ``score``.
    """
    run = doc["runs"].setdefault(params_hash, {"params": dict(params), "couples": []})
    old = pd.DataFrame(run.get("couples", []))
    frames = [f for f in (old, new) if not f.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["left_key", "right_key"], keep="last"
    )
    merged = merged.replace({None: np.nan}).infer_objects().reset_index(drop=True)
    present = {c: up for c, up in score_columns.items() if c in merged.columns}
    merged["score"] = rank_score(merged, present) if present else np.nan
    run["params"] = {k: _jsonable(v) for k, v in params.items()}
    run["score_columns"] = dict(score_columns)
    run["updated_utc"] = _now()
    run["seq"] = 1 + max((r.get("seq", 0) for r in doc["runs"].values()), default=0)
    run["couples"] = [
        {k: _jsonable(v) for k, v in row.items()}
        for row in merged.to_dict(orient="records")
    ]
    return merged
