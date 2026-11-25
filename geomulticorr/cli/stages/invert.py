#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# invert.py
# creation date: 2026-11-19.
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
"""``gmc invert`` — one ``TIOInversion`` per run and pzone: prepare, launch, trace."""

from __future__ import annotations

import json
import pathlib

from geomulticorr._logging import logger
from geomulticorr.cli.registry import build_filter_pipeline
from geomulticorr.cli.stages._common import (
    StageContext,
    add_figure_paths,
    check_kwargs,
    clean,
    log_section,
    pz_criterias,
)
from geomulticorr.cli.stages.correct import moving_areas_resolver


def run(ctx: StageContext, section, rec) -> None:
    from geomulticorr.inversion.pytio.config import TIOConfig
    from geomulticorr.inversion.tio_inversion import TIOInversion

    session = ctx.open_session()
    image_type = ctx.image_type("invert")
    names = ctx.pz_names()

    runs = []
    for spec in section.runs:
        for pz in [spec.pz_name] if spec.pz_name else names:
            log_section("invert", f"{pz} / {spec.name}")
            pairs = session.get_pairs(
                criterias=pz_criterias(pz, spec.criterias), image_type=image_type
            )
            if not pairs:
                raise ValueError(
                    f"inversion {spec.name!r}: no pair in {pz!r} matches "
                    f"criterias {spec.criterias!r}"
                )
            inv = TIOInversion(
                session,
                pairs,
                inversion_name=spec.name,
                filter_pipeline=build_filter_pipeline(
                    spec.masks, resolve_auto=moving_areas_resolver(session, pz)
                ),
                tio_binaries_dir=spec.tio_binaries_dir,
            )
            entry = {"name": spec.name, "pz": pz, "n_pairs_selected": len(pairs)}
            if spec.nmad_threshold is not None:
                kept, dropped = inv.filter_pairs_by_nmad(spec.nmad_threshold)
                entry["n_pairs_dropped_nmad"] = len(dropped)

            check_kwargs(inv.prepare_inversion, spec.weights, "invert.runs.weights")
            inv.prepare_inversion(solver=TIOConfig(**spec.solver), **spec.weights)
            entry["n_pairs"] = len(inv.pairs)
            entry["n_dates"] = len(inv.image_dates)
            entry["network"] = _components(inv)
            entry["disconnected"] = len(entry["network"]) > 1

            if spec.weights_figure:
                try:
                    written = inv.save_weights_figure(formats=("png",))
                    add_figure_paths(rec, ctx, f"{pz}/{spec.name}/weights", written)
                except Exception as exc:  # a figure must not cost the inversion
                    logger.warning(f"weights figure for {spec.name!r} failed: {exc}")

            if spec.launch:
                inv.launch(mode=spec.mode, write_residuals=spec.write_residuals)
                post = spec.post_process
                if post is None:
                    post = spec.mode == "local"
                if post and spec.mode != "python":
                    inv.post_process()

            entry.update(_read_outputs(ctx, inv, rec, pz, spec.name))
            runs.append(entry)

    rec.pzones = names
    rec.summary["image_type"] = image_type
    rec.summary["runs"] = runs


def _components(inv) -> list[dict]:
    try:
        comps = inv.network_components()
    except Exception as exc:
        logger.debug(f"network components unavailable: {exc}")
        return []
    out = []
    for comp in comps:
        out.append({k: clean(v) for k, v in comp.items() if not isinstance(v, set)})
    return out


def _read_outputs(ctx, inv, rec, pz: str, name: str) -> dict:
    """Traces and the solver's diagnostics, summarised (small rasters only)."""
    from geomulticorr.stats.inversion_extractor import discover_tot_rasters

    inv_dir = pathlib.Path(inv.inversion_dir)
    rec.add_artifact(f"{pz}/{name}/inversion_dir", inv_dir, ctx.project_root)
    out: dict = {}
    try:
        tots = discover_tot_rasters(inv_dir)
        out["n_tot_dates"] = {c: len(v) for c, v in tots.items()}
    except Exception:
        out["n_tot_dates"] = {}
    for direction in ("EW", "NS"):
        trace = (
            inv_dir / f"inverse_{direction}" / f"inverse_{direction}_parameters.json"
        )
        if trace.exists():
            rec.add_artifact(
                f"{pz}/{name}/{direction}_parameters", trace, ctx.project_root
            )
            try:
                doc = json.loads(trace.read_text(encoding="utf-8"))
                weights = doc.get("weights", {})
                out[f"{direction}_weights"] = {
                    "mode": weights.get("mode"),
                    "summary": weights.get("summary"),
                    "applied_by_solver": weights.get("applied_by_solver"),
                }
            except (OSError, ValueError):
                pass
        rms = inv_dir / f"inverse_{direction}" / "diagnostics" / "rms.tif"
        if rms.exists():
            out[f"{direction}_rms"] = _percentiles(rms)
    return out


def _percentiles(path: pathlib.Path) -> dict | None:
    try:
        import numpy as np
        import rasterio

        with rasterio.open(path) as src:
            data = src.read(1, masked=True).astype("float64").filled(np.nan)
        data = data[np.isfinite(data)]
        if not data.size:
            return None
        p50, p90, p99 = np.percentile(data, [50, 90, 99])
        return {"p50": float(p50), "p90": float(p90), "p99": float(p99)}
    except Exception as exc:
        logger.debug(f"rms summary of {path} failed: {exc}")
        return None
