#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# render.py
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
"""Run folder → ``report.html`` (Jinja2) → ``report.pdf`` (plutoprint).

The HTML is self-contained next to an ``assets/`` folder: report figures are
drawn from the records, and the PNG figures the stages already wrote (pair
network, correlation design map, weights) are copied in, so the report folder
can be zipped and sent. plutoprint is imported lazily; without it the PDF is
skipped with a warning and the HTML is still written, the same contract as
xDEM's ``generate_pdf``.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
from datetime import datetime, timezone

from geomulticorr._logging import logger
from geomulticorr.report import collect as _collect
from geomulticorr.report import figures as _fig

TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"
ASSETS_DIRNAME = "assets"


def render_report(
    run_dir: pathlib.Path | str,
    *,
    outputs=None,
    pdf: bool | None = None,
) -> dict[str, pathlib.Path]:
    """Render the report of *run_dir*.

    :param outputs: An ``OutputsConfig``; default read from ``used_config.yaml``.
    :param pdf: Force the PDF on/off; default ``outputs.report.pdf``.
    :returns: ``{"html": path, "pdf": path}`` for the files written.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    from geomulticorr.cli.config import OutputsConfig

    run_dir = pathlib.Path(run_dir)
    ctx = _collect.collect(run_dir)
    if not ctx["stages"]:
        logger.warning(f"No stage record in {run_dir}; nothing to report")
        return {}
    if outputs is None:
        outputs = OutputsConfig.model_validate(ctx["config"].get("outputs") or {})
    want_pdf = outputs.report.pdf if pdf is None else pdf

    assets = run_dir / ASSETS_DIRNAME
    project = pathlib.Path(ctx["project"]) if ctx["project"] else run_dir
    ctx["figures"] = {}
    ctx["files"] = {}
    for rec in ctx["stages"]:
        figs, files = _stage_assets(rec, project, run_dir, assets, outputs)
        ctx["figures"][rec["stage"]] = figs
        ctx["files"][rec["stage"]] = files
    ts = _time_series_figure(ctx, run_dir, assets, outputs)
    if ts:
        ctx["figures"].setdefault("invert", []).append(ts)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt"] = _collect.fmt
    env.globals.update(
        scalar_items=_collect.scalar_items,
        table=_collect.table,
        worst_pairs=_collect.worst_pairs,
        is_scalar=_collect.is_scalar,
    )
    title = outputs.report.title or f"GeoMultiCorr report — {project.name}"
    html = env.get_template("report.html.j2").render(
        **ctx,
        title=title,
        level=outputs.level,
        worst_n=outputs.report.worst_pairs,
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        css=(TEMPLATES_DIR / "style.css").read_text(encoding="utf-8"),
    )
    written: dict[str, pathlib.Path] = {}
    html_path = run_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    written["html"] = html_path
    logger.save(f"report → {html_path}")

    if want_pdf:
        pdf_path = write_pdf(html_path, run_dir / "report.pdf")
        if pdf_path is not None:
            written["pdf"] = pdf_path
    return written


def write_pdf(html_path: pathlib.Path, pdf_path: pathlib.Path) -> pathlib.Path | None:
    """HTML → PDF with plutoprint (A4, narrow margins). ``None`` if unavailable."""
    try:
        import plutoprint
    except ImportError:
        logger.warning("plutoprint is not installed: report.pdf skipped (HTML written)")
        return None
    try:
        book = plutoprint.Book(plutoprint.PAGE_SIZE_A4, plutoprint.PAGE_MARGINS_NARROW)
        book.load_url(str(html_path))
        book.write_to_pdf(str(pdf_path))
    except Exception as exc:
        logger.warning(f"report.pdf failed: {exc}")
        return None
    logger.save(f"report → {pdf_path}")
    return pdf_path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "figure"


def _stage_assets(rec, project, run_dir, assets, outputs):
    """Figures drawn from the record, plus copies of the PNGs the stage wrote."""
    stage = rec["stage"]
    figs: list[dict] = []
    files: list[dict] = []
    folder = assets / stage

    def add(fig, name, caption):
        try:
            path = _fig.save(fig, folder / f"{name}.png")
            figs.append({"src": os.path.relpath(path, run_dir), "caption": caption})
        except Exception as exc:
            logger.warning(f"report figure {stage}/{name} failed: {exc}")

    summary = rec.get("summary") or {}
    try:
        thumbs = summary.get("thumbs") or [
            t
            for entry in (summary.get("pzones") or {}).values()
            if isinstance(entry, dict)
            for t in entry.get("thumbs", []) or []
        ]
        if stage in ("crop", "pairs") and thumbs:
            add(
                _fig.plot_archive_timeline(thumbs)[0],
                "archive",
                "Acquisitions per sensor",
            )
        if stage == "pairs" and rec.get("pairs"):
            add(
                _fig.plot_dt_histogram(rec["pairs"])[0], "dt_hist", "Temporal baselines"
            )
        if stage == "correct" and rec.get("pairs"):
            add(
                _fig.plot_nmad_before_after(rec["pairs"])[0],
                "nmad",
                "Stable-ground NMAD per pair, before vs after the corrections",
            )
    except Exception as exc:  # a figure must never cost the report
        logger.warning(f"report figures for {stage} failed: {exc}")

    for label, rel in (rec.get("artifacts") or {}).items():
        src = pathlib.Path(rel)
        if not src.is_absolute():
            src = project / src
        if src.suffix.lower() == ".png" and src.exists():
            name = label[:-4] if label.lower().endswith(".png") else label
            dst = folder / f"{_slug(name)}.png"
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                figs.append({"src": os.path.relpath(dst, run_dir), "caption": name})
            except OSError as exc:
                logger.warning(f"could not copy {src}: {exc}")
        else:
            files.append({"label": label, "path": str(src), "exists": src.exists()})
    return figs, files


def _time_series_figure(ctx, run_dir, assets, outputs) -> dict | None:
    """Cumulative displacement at ``outputs.report.points``, one line per inversion.

    The only place the report reads rasters, and only by sampling them at a
    handful of points through ``InversionExtractor``.
    """
    points = outputs.report.points
    if points is None:
        return None
    sources = {}
    project = pathlib.Path(ctx["project"]) if ctx["project"] else run_dir
    for rec in ctx["stages"]:
        for label, rel in (rec.get("artifacts") or {}).items():
            if label.endswith(("/inversion_dir", "/fusion_dir")):
                path = pathlib.Path(rel)
                sources[label.rsplit("/", 1)[0]] = (
                    path if path.is_absolute() else project / path
                )
    if not sources:
        return None
    try:
        from geomulticorr.stats.inversion_extractor import InversionExtractor
        from geomulticorr.utils.gmc_functions import plot_cumulative_time_series

        frames = {
            label: InversionExtractor(path).extract(
                points,
                layer=outputs.report.points_layer,
                buffer_m=outputs.report.buffer_m,
            )
            for label, path in sources.items()
            if path.exists()
        }
        if not frames:
            return None
        fig = plot_cumulative_time_series(frames)
        path = _fig.save(fig, assets / "invert" / "time_series.png")
    except Exception as exc:
        logger.warning(f"report time series failed: {exc}")
        return None
    return {
        "src": os.path.relpath(path, run_dir),
        "caption": f"Cumulative displacement at {pathlib.Path(points).name}",
    }
