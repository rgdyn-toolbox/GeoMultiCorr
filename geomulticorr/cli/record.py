#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# record.py
# creation date: 2026-10-25.
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
"""Stage records: the small files the report is built from.

Every stage leaves ``<run_dir>/records/<stage>.json`` holding what the report
needs and nothing heavy — parameters, counts, key statistics, the paths of the
figures and traces it wrote, the warnings it logged and its wall time. The
report reads these and never re-opens a raster.

Each record is also appended, one JSON line, to ``records/history.jsonl``, so
re-running a stage refreshes ``<stage>.json`` without losing the earlier run.

Same rules as the other GMC traces (``correlation_plan_*.json``,
``inverse_*_parameters.json``):

- **a write failure is logged, never raised** — losing a trace must not cost a
  finished stage;
- ``written_utc`` makes the file not byte-stable by design; no test may assert
  byte equality.

:func:`record_stage` is usable from a notebook as well as from the CLI::

    with record_stage("correct", run_dir, config={...}) as rec:
        results = session.apply_pairs_corrections(...)
        rec.summary["n_pairs"] = len(results)
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from geomulticorr._logging import logger, severity
from geomulticorr.inversion._run_parameters import _jsonable

RECORDS_DIRNAME = "records"
HISTORY_FILENAME = "history.jsonl"


@dataclass
class StageRecord:
    stage: str
    status: str = "running"  # ok | partial | failed
    started_utc: str = ""
    elapsed_s: float = 0.0
    gmc_version: str = ""
    pzones: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    #: Small dict of key numbers: counts, medians, fractions.
    summary: dict = field(default_factory=dict)
    #: ``{label: path}`` of figures, traces and tables, relative to the project
    #: when possible.
    artifacts: dict = field(default_factory=dict)
    #: Optional per-pair rows: ``{"pa_key":…, "status":…, …}``.
    pairs: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    written_utc: str = ""

    def add_artifact(self, label: str, path, root: pathlib.Path | None = None) -> None:
        """Register *path* under *label*; relative to *root* when inside it."""
        if path is None:
            return
        path = pathlib.Path(path)
        if root is not None:
            try:
                path = path.resolve().relative_to(pathlib.Path(root).resolve())
            except ValueError:
                pass
        self.artifacts[label] = str(path)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


class _WarningCollector(logging.Handler):
    """Keeps the text of every GMC record ranked WARNING or above."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if severity(record.levelno) >= logging.WARNING:
            try:
                self.messages.append(record.getMessage())
            except Exception:  # never let a log line break a stage
                pass


def records_dir(run_dir: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(run_dir) / RECORDS_DIRNAME


def write_record(
    run_dir: pathlib.Path | str, record: StageRecord
) -> pathlib.Path | None:
    """Write ``records/<stage>.json`` and append to the history. Never raises."""
    record.written_utc = datetime.now(timezone.utc).isoformat()
    try:
        out = records_dir(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        doc = record.to_dict()
        path = out / f"{record.stage}.json"
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(out / HISTORY_FILENAME, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return path
    except Exception as exc:  # a trace must never cost the stage
        logger.warning(f"Could not write the {record.stage!r} stage record: {exc}")
        return None


def read_records(run_dir: pathlib.Path | str) -> dict[str, dict]:
    """``{stage: record dict}`` for every ``records/<stage>.json`` present."""
    out: dict[str, dict] = {}
    folder = records_dir(run_dir)
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"Skipping unreadable record {path.name}: {exc}")
            continue
        out[doc.get("stage", path.stem)] = doc
    return out


@contextmanager
def record_stage(
    stage: str,
    run_dir: pathlib.Path | str,
    config: dict | None = None,
    pzones: list[str] | None = None,
) -> Iterator[StageRecord]:
    """Time a stage, capture its warnings, and write its record on exit.

    The record is written whether the block succeeds or raises; on an
    exception ``status`` is ``"failed"``, ``error`` holds the message, and the
    exception propagates. A block that sets ``status = "partial"`` keeps it.
    """
    from geomulticorr import __version__

    rec = StageRecord(
        stage=stage,
        started_utc=datetime.now(timezone.utc).isoformat(),
        gmc_version=__version__,
        pzones=list(pzones or []),
        config=dict(config or {}),
    )
    collector = _WarningCollector()
    gmc_logger = logging.getLogger("GMC")
    gmc_logger.addHandler(collector)
    t0 = time.perf_counter()
    try:
        yield rec
    except BaseException as exc:
        rec.status = "failed"
        rec.error = f"{type(exc).__name__}: {exc}"
        raise
    else:
        if rec.status == "running":
            rec.status = "ok"
    finally:
        gmc_logger.removeHandler(collector)
        rec.elapsed_s = round(time.perf_counter() - t0, 3)
        rec.warnings.extend(collector.messages)
        write_record(run_dir, rec)
