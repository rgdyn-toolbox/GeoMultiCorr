#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_stable_stats_sync.py
# creation date: 2026-11-25.
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
"""Stable-ground NMAD reaches the Pairs layer and the console summary."""

import json
import types

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from geomulticorr.core.console.tables import print_apply_corrections_summary
from geomulticorr.core.session import STABLE_STATS_SYNC, Session, _nmad_before_after


def _stats(raw_ew, corr_ew):
    block = lambda v: {"ew": {"nmad": v}, "ns": {"nmad": v / 2}}  # noqa: E731
    return {
        "raw_corr_stats": block(9.0),
        "final_corrected_stats": block(8.0),
        "correction_stats": {"A": block(8.0)},
        "raw_corr_stats_stable": block(raw_ew),
        "final_corrected_stats_stable": block(corr_ew),
    }


def _stub(tmp_path, with_stable=True):
    js_path = tmp_path / "k.json"
    js = _stats(0.93, 0.41)
    if not with_stable:
        js.pop("raw_corr_stats_stable"), js.pop("final_corrected_stats_stable")
    js_path.write_text(json.dumps(js))
    pair = types.SimpleNamespace(
        pa_key="k", pa_path=tmp_path / "k", pa_stats_path=js_path
    )
    pairs = gpd.GeoDataFrame(
        {"pa_path": [str(pair.pa_path)]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:32632",
    )
    stub = types.SimpleNamespace(
        _pairs=pairs, path_geodb=tmp_path / "db.gpkg", get_pairs=lambda *a: [pair]
    )
    stub.sync_pairs_stats = lambda *a, **k: Session.sync_pairs_stats(stub, *a, **k)
    return stub


def test_sync_writes_the_stable_columns(tmp_path):
    stub = _stub(tmp_path)
    out = Session.sync_corrected_stats(stub, n_corrections=2)
    row = out.iloc[0]
    assert row["pa_ew_nmad_stable_raw"] == 0.93
    assert row["pa_ew_nmad_stable_corr"] == 0.41
    assert row["pa_ns_nmad_stable_corr"] == 0.205
    # the existing columns keep their meaning (all valid pixels)
    assert row["pa_ew_nmad_corr"] == 8.0
    assert set(STABLE_STATS_SYNC) <= set(out.columns)


def test_sync_gives_nan_without_stable_ground(tmp_path):
    out = Session.sync_corrected_stats(_stub(tmp_path, with_stable=False))
    assert np.isnan(out.iloc[0]["pa_ew_nmad_stable_corr"])
    assert out.iloc[0]["pa_ew_nmad_corr"] == 8.0


class TestConsole:
    def test_before_after_text(self):
        raw = {"ew": {"nmad": 0.93}}
        corr = {"ew": {"nmad": 0.41}}
        assert _nmad_before_after(raw, corr, "ew") == "0.93 → 0.41"
        assert _nmad_before_after(None, None, "ew") == "—"
        assert _nmad_before_after(raw, {"ew": {"nmad": float("nan")}}, "ew") == (
            "0.93 → ?"
        )

    def test_table_has_the_stable_columns(self):
        row = dict(
            pair="k", median="ok", ramp_topo="ok", stats="ok", ew_corr="ok",
            ns_corr="ok", nmad_ew="0.93 → 0.41", nmad_ns="—",
        )  # fmt: skip
        table = print_apply_corrections_summary([row], print_summary=False)
        assert [c.header for c in table.columns][-2:] == [
            "NMAD EW (stable)",
            "NMAD NS (stable)",
        ]

    def test_old_rows_without_the_keys_still_render(self):
        row = dict(
            pair="k", median="ok", ramp_topo="ok", stats="ok", ew_corr="ok",
            ns_corr="ok",
        )  # fmt: skip
        table = print_apply_corrections_summary([row], print_summary=False)
        assert len(table.columns) == 6
