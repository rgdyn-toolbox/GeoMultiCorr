#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# similarity.py
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
"""``gmc similarity`` — predict which thumb couples will correlate."""

from __future__ import annotations

from geomulticorr.cli.config import forwarded
from geomulticorr.cli.stages._common import StageContext, clean, log_section, median

#: Columns summarised in the record — the validated predictors first.
SUMMARY_COLUMNS = ("trial_peak_median", "coh_mean", "trial_valid_frac", "pearson")


def run(ctx: StageContext, section, rec) -> None:
    session = ctx.open_session()
    image_type = ctx.image_type("similarity")
    names = ctx.pz_names()
    kwargs = forwarded(section)
    level = ctx.cfg.outputs.level

    per_pz = {}
    for pz in names:
        log_section("similarity", pz)
        frame = session.compute_thumbs_similarity(
            pz,
            image_type=image_type,
            progress=False,
            **kwargs,
            **section.report_kwargs,
        )
        rec.add_artifact(
            f"{pz}/similarity_csv",
            session.similarity_path(pz, image_type),
            ctx.project_root,
        )
        n = 0 if frame is None else int(len(frame))
        entry = {"n_couples": n}
        if n:
            for col in SUMMARY_COLUMNS:
                if col in frame.columns:
                    entry[f"median_{col}"] = median(frame[col].tolist())
            entry["top"] = _top_rows(frame, 10 if level == 1 else None)
        per_pz[pz] = entry

    rec.pzones = names
    rec.summary["image_type"] = image_type
    rec.summary["pzones"] = per_pz


def _top_rows(frame, limit: int | None) -> list[dict]:
    keep = [
        c
        for c in ("left_key", "right_key", "dt_days", *SUMMARY_COLUMNS)
        if c in frame.columns
    ]
    sort_col = next((c for c in SUMMARY_COLUMNS if c in frame.columns), None)
    sub = frame.sort_values(sort_col, ascending=False) if sort_col else frame
    if limit is not None:
        sub = sub.head(limit)
    return [
        {k: clean(v) for k, v in row.items()} for row in sub[keep].to_dict("records")
    ]
