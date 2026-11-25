#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# extract.py
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
"""``gmc extract`` — ASP disparity → metric EW/NS/CC rasters + raw stats."""

from __future__ import annotations

from geomulticorr.cli.config import forwarded
from geomulticorr.cli.stages._common import (
    StageContext,
    log_section,
    median,
    pairs_table,
    pz_criterias,
)


def run(ctx: StageContext, section, rec) -> None:
    session = ctx.open_session()
    image_type = ctx.image_type("extract")
    names = ctx.pz_names()
    kwargs = forwarded(section, exclude={"criterias"})

    per_pz = {}
    for pz in names:
        log_section("extract", pz)
        crit = pz_criterias(pz, section.criterias)
        results = (
            session.extract_pairs_raw_displacements(
                criterias=crit, image_type=image_type, **kwargs
            )
            or {}
        )
        failed = sorted(k for k, v in results.items() if v is None)
        rows = pairs_table(session, crit, image_type)
        per_pz[pz] = {
            "n_pairs": len(results),
            "n_failed": len(failed),
            "failed": failed,
            "median_ew_nmad_raw": median(r.get("ew_nmad_raw") for r in rows),
            "median_ns_nmad_raw": median(r.get("ns_nmad_raw") for r in rows),
            "median_cc_quality050": median(r.get("cc_quality050") for r in rows),
        }
        if failed:
            rec.status = "partial"
        rec.pairs.extend(rows)

    rec.pzones = names
    rec.summary["image_type"] = image_type
    rec.summary["pzones"] = per_pz
