#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# fuse.py
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
"""``gmc fuse`` — calibrate one per-sensor inversion onto another and merge."""

from __future__ import annotations

import json

from geomulticorr._logging import logger
from geomulticorr.cli.stages._common import (
    StageContext,
    clean,
    log_section,
    median,
    pz_criterias,
)


def run(ctx: StageContext, section, rec) -> None:
    from geomulticorr.inversion.fusion import fuse_inversions
    from geomulticorr.inversion.tio_inversion import resolve_inversion_dir

    session = ctx.open_session()
    image_type = ctx.image_type("fuse")
    names = ctx.pz_names()

    runs = []
    for spec in section.runs:
        for pz in [spec.pz_name] if spec.pz_name else names:
            log_section("fuse", f"{pz} / {spec.name}")
            ref_dir = resolve_inversion_dir(session, pz, spec.reference)
            sec_dir = resolve_inversion_dir(session, pz, spec.secondary)
            out_dir = resolve_inversion_dir(session, pz, spec.name)
            entry = {"name": spec.name, "pz": pz}

            # Closure first: it is what tells which side should be the reference.
            for side, crit, series_dir in (
                ("reference", spec.closure_reference_criterias, sec_dir),
                ("secondary", spec.closure_secondary_criterias, ref_dir),
            ):
                if crit is None:
                    continue
                entry[f"closure_{side}_pairs"] = _closure(
                    ctx, session, rec, pz, spec, side, crit, series_dir, image_type
                )

            fuse_inversions(
                ref_dir,
                sec_dir,
                out_dir,
                model=spec.model,
                overlap=spec.overlap,
                min_epochs=spec.min_epochs,
                reference_label=spec.reference_label,
                secondary_label=spec.secondary_label,
                overwrite=spec.overwrite,
            )
            trace = out_dir / "fusion_parameters.json"
            rec.add_artifact(f"{pz}/{spec.name}/fusion_dir", out_dir, ctx.project_root)
            if trace.exists():
                rec.add_artifact(
                    f"{pz}/{spec.name}/fusion_parameters", trace, ctx.project_root
                )
                try:
                    doc = json.loads(trace.read_text(encoding="utf-8"))
                    entry["fusion"] = {
                        k: doc.get(k)
                        for k in ("model", "overlap", "reference", "secondary")
                        if k in doc
                    }
                except (OSError, ValueError):
                    pass
            runs.append(entry)

    rec.pzones = names
    rec.summary["runs"] = runs


def _closure(ctx, session, rec, pz, spec, side, crit, series_dir, image_type):
    """Pairs of one side against the *other* side's series; CSV + medians."""
    from geomulticorr.inversion._stack import load_cumulative_stack
    from geomulticorr.inversion.fusion import (
        closure_summary,
        pair_closure_against_series,
    )

    try:
        pairs = session.get_pairs(
            criterias=pz_criterias(pz, crit), image_type=image_type
        )
        series = load_cumulative_stack(series_dir)
        table = closure_summary(pair_closure_against_series(pairs, series))
    except Exception as exc:  # informative, not blocking
        logger.warning(f"fuse: closure of the {side} pairs failed: {exc}")
        return None
    path = ctx.run_dir / "tables" / f"closure_{pz}_{spec.name}_{side}.csv"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        rec.artifacts[f"{pz}/{spec.name}/closure_{side}"] = str(path)
    except OSError as exc:
        logger.warning(f"fuse: could not write {path.name}: {exc}")
    out = {"n_pairs": int(table["pair"].nunique()) if len(table) else 0}
    for comp, grp in table.groupby("component") if len(table) else []:
        out[str(comp)] = {
            "median_rate_bias_m_yr": median(grp["rate_bias_m_yr"].tolist()),
            "median_nmad": median(grp["nmad"].tolist()),
        }
    return {k: clean(v) for k, v in out.items()}
