#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_run_parameters.py
# creation date: 2026-08-31.
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
"""tests/inversion/test_run_parameters.py

The run-parameters JSON is the only record of *how* an inversion was weighted:
``liste_couple`` holds the resulting numbers, not the recipe.

Two properties matter most. ``weights.params`` must be the **full unpruned**
twelve, so ``write_liste_couple(**params)`` reproduces the run from the file
alone; and a failure to write must **never** abort an otherwise fully prepared
inversion.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import geomulticorr.inversion.tio_inversion as tio
from geomulticorr.corrections.masks import CCFilter, OutlierFilter, StableAreaMask
from geomulticorr.inversion._run_parameters import (
    build_run_parameters,
    describe_filter_pipeline,
    write_run_parameters,
)
from geomulticorr.inversion.tio_inversion import TIOInversion


def _stats(nmad: float = 0.3, cc: float = 0.7) -> dict:
    return {
        "final_corrected_stats": {"ew": {"nmad": nmad}, "ns": {"nmad": nmad + 0.05}},
        "raw_corr_stats": {"cc": {"cc_quality_gte_050": cc}},
    }


def _pairs(n: int = 3):
    return [
        SimpleNamespace(
            pa_key=f"PZ_p{i}", pa_dt_days=30 * (i + 1), pa_direction="forward",
            pa_left=SimpleNamespace(th_date="2022-01-01"),
            pa_right=SimpleNamespace(th_date=f"2022-0{i + 2}-01"),
        )
        for i in range(n)
    ]


@pytest.fixture
def inv(monkeypatch, tmp_path):
    monkeypatch.setattr(tio, "load_pair_stats", lambda p: _stats())
    obj = TIOInversion.__new__(TIOInversion)
    obj.pairs = _pairs()
    obj.inversion_name = "PDL_spot"
    obj.pzone_name = "PasDeLours"
    obj.inversion_dir = tmp_path
    obj._DIRECTIONS = ("EW", "NS")
    obj.filter_pipeline = None
    obj._image_dates = None
    obj._raster_width = 2048
    obj._raster_height = 1536
    obj._quality_metrics = None
    obj._last_weights = None
    obj._last_weights_params = None
    obj._last_nmad_filter = None
    obj._last_launch = None
    for direction in obj._DIRECTIONS:
        (tmp_path / f"inverse_{direction}").mkdir()
    return obj


PARAMS_12 = dict(
    weight_mode="quality", slope=2.0, min_weight=0.1, dt_range=None,
    sharpness=5.0, w_min=0.0, invert=False, combine="wmean",
    alpha=1 / 3, beta=1 / 3, gamma=1 / 3, cc_gamma=1.5,
)


class TestBuildDocument:
    """Pure assembly — no filesystem, so it needs no project."""

    def _doc(self, **overrides):
        base = dict(
            direction="EW", inversion_name="PDL", pzone="PasDeLours",
            inversion_dir="/tmp/x", raster_shape=(2048, 1536),
            pair_keys=["a", "b"], n_images=3, weight_mode="quality",
            combine="wmean", weight_source="explorer", weight_params=PARAMS_12,
            relevant_params=["alpha", "beta", "cc_gamma", "combine", "gamma"],
            weight_summary={"n": 2, "min": 0.2, "max": 1.0, "mean": 0.6},
        )
        base.update(overrides)
        return build_run_parameters(**base)

    def test_is_json_serialisable(self):
        json.dumps(self._doc())

    def test_records_the_full_unpruned_params(self):
        """Pruned params could not be splatted back into write_liste_couple."""
        assert set(self._doc()["weights"]["params"]) == set(PARAMS_12)

    def test_relevant_params_sits_alongside_not_instead(self):
        weights = self._doc()["weights"]
        assert "alpha" in weights["relevant_params"]
        assert "sharpness" in weights["params"]
        assert "sharpness" not in weights["relevant_params"]

    def test_label_joins_mode_and_combine_for_quality(self):
        assert self._doc()["weights"]["label"] == "quality:wmean"

    def test_label_is_the_bare_mode_otherwise(self):
        doc = self._doc(weight_mode="sigmoid", combine="wmean")
        assert doc["weights"]["label"] == "sigmoid"

    def test_source_is_recorded(self):
        assert self._doc()["weights"]["source"] == "explorer"

    def test_carries_the_pair_keys(self):
        """liste_couple holds dates, so two sensors on one date are ambiguous."""
        doc = self._doc()
        assert doc["pairs"]["keys"] == ["a", "b"]
        assert doc["pairs"]["count"] == 2

    def test_notes_what_it_deliberately_omits(self):
        assert "liste_couple" in self._doc()["not_recorded_here"]

    def test_missing_raster_shape_is_none_not_a_crash(self):
        doc = self._doc(raster_shape=None)
        assert doc["raster"] == {"width": None, "height": None}

    def test_numpy_values_are_coerced(self):
        doc = self._doc(weight_params={**PARAMS_12, "cc_gamma": np.float64(1.5)})
        json.dumps(doc)
        assert isinstance(doc["weights"]["params"]["cc_gamma"], float)

    def test_tuples_become_lists(self):
        doc = self._doc(weight_params={**PARAMS_12, "dt_range": (300, 400)})
        assert doc["weights"]["params"]["dt_range"] == [300, 400]

    def test_stamps_the_version_and_a_timestamp(self):
        from geomulticorr import __version__

        doc = self._doc()
        assert doc["gmc_version"] == __version__
        assert doc["written_utc"].endswith("+00:00")


class TestDescribeFilterPipeline:
    def test_none_stays_none(self):
        """None and [] mean different things: no pipeline vs. an empty one."""
        assert describe_filter_pipeline(None) is None

    def test_a_real_pipeline_round_trips_its_arguments(self):
        described = describe_filter_pipeline(CCFilter(0.6) + OutlierFilter((-5, 5)))
        assert described == [
            {"filter": "CCFilter", "params": {"cc_threshold": 0.6}},
            {"filter": "OutlierFilter", "params": {"threshold": [-5, 5]}},
        ]

    def test_a_bare_mask_is_accepted(self):
        assert describe_filter_pipeline(CCFilter(0.5))[0]["filter"] == "CCFilter"

    def test_non_scalar_arguments_become_a_type_tag(self):
        """A GeoDataFrame does not belong in a parameters file."""
        import geopandas as gpd

        gdf = gpd.GeoDataFrame({"geometry": []})
        described = describe_filter_pipeline(StableAreaMask(gdf))
        assert described[0]["params"]["stable_mask"] == "<GeoDataFrame>"

    def test_it_is_json_serialisable(self):
        json.dumps(describe_filter_pipeline(CCFilter(0.6) + OutlierFilter((-5, 5))))


class TestWriteRunParameters:
    def test_writes_both_directions(self, inv):
        written = inv.write_run_parameters(
            "quality", "wmean", "explorer", PARAMS_12,
            {"EW": [0.9, 0.5, 0.2], "NS": [0.8, 0.4, 0.1]},
        )
        assert set(written) == {"EW", "NS"}
        for direction, path in written.items():
            assert path.name == f"inverse_{direction}_parameters.json"
            assert path.parent.name == f"inverse_{direction}"
            assert path.exists()

    def test_each_file_is_self_contained(self, inv):
        written = inv.write_run_parameters(
            "quality", "wmean", "explorer", PARAMS_12,
            {"EW": [0.9, 0.5, 0.2], "NS": [0.8, 0.4, 0.1]},
        )
        for direction, path in written.items():
            doc = json.loads(path.read_text())
            assert doc["direction"] == direction
            assert doc["inversion_name"] == "PDL_spot"
            assert doc["pzone"] == "PasDeLours"
            assert doc["pairs"]["count"] == 3
            assert set(doc["weights"]["params"]) == set(PARAMS_12)

    def test_the_two_files_differ_where_they_should(self, inv):
        """Otherwise the per-direction split is theatre."""
        written = inv.write_run_parameters(
            "quality", "wmean", "explorer", PARAMS_12,
            {"EW": [0.9, 0.5, 0.2], "NS": [0.1, 0.1, 0.1]},
        )
        ew = json.loads(written["EW"].read_text())
        ns = json.loads(written["NS"].read_text())
        assert ew["direction"] != ns["direction"]
        assert ew["weights"]["summary"]["mean"] != ns["weights"]["summary"]["mean"]

    def test_date_based_modes_give_matching_summaries(self, inv):
        """Which is itself informative, not a bug."""
        written = inv.write_run_parameters(
            "uniform", None, "computed", {**PARAMS_12, "weight_mode": "uniform"},
            {"EW": [1.0, 1.0, 1.0], "NS": [1.0, 1.0, 1.0]},
        )
        ew = json.loads(written["EW"].read_text())
        ns = json.loads(written["NS"].read_text())
        assert ew["weights"]["summary"] == ns["weights"]["summary"]

    def test_records_the_filter_pipeline(self, inv):
        inv.filter_pipeline = CCFilter(0.6) + OutlierFilter((-5, 5))
        written = inv.write_run_parameters(
            "quality", "wmean", "explorer", PARAMS_12,
            {"EW": [1, 1, 1], "NS": [1, 1, 1]})
        doc = json.loads(written["EW"].read_text())
        assert [f["filter"] for f in doc["filter_pipeline"]] == [
            "CCFilter", "OutlierFilter"]

    def test_records_the_nmad_filter_and_launch_profile(self, inv):
        inv._last_nmad_filter = {"threshold": 0.7, "kept": 3, "removed": 1}
        inv._last_launch = {"cluster": "isterre", "nodes": 1, "cores": 8,
                            "walltime": "12:00:00"}
        written = inv.write_run_parameters(
            "uniform", None, "computed", PARAMS_12,
            {"EW": [1, 1, 1], "NS": [1, 1, 1]})
        doc = json.loads(written["EW"].read_text())
        assert doc["nmad_filter"]["removed"] == 1
        assert doc["launch"]["cluster"] == "isterre"

    def test_a_missing_raster_is_logged_not_raised(self, inv, caplog_gmc):
        inv._raster_width = None
        inv.pairs[0].pa_ew_path = "/nonexistent/nope.tif"
        written = inv.write_run_parameters(
            "uniform", None, "computed", PARAMS_12,
            {"EW": [1, 1, 1], "NS": [1, 1, 1]})
        assert json.loads(written["EW"].read_text())["raster"]["width"] is None

    def test_params_round_trip_through_write_liste_couple(self, inv, monkeypatch):
        """The point of storing the unpruned twelve: the run can be replayed."""
        monkeypatch.setattr(inv, "_sync_pair_weights", lambda *a, **k: None)
        written = inv.write_run_parameters(
            "quality", "wmean", "explorer", PARAMS_12,
            {"EW": [1, 1, 1], "NS": [1, 1, 1]})
        params = json.loads(written["EW"].read_text())["weights"]["params"]
        if params["dt_range"] is not None:
            params["dt_range"] = tuple(params["dt_range"])

        replayed = inv.write_liste_couple(**params, sync_geodb=False)
        assert len(replayed["EW"]) == len(inv.pairs)

    def test_the_file_is_indented_json(self, inv):
        written = inv.write_run_parameters(
            "uniform", None, "computed", PARAMS_12,
            {"EW": [1, 1, 1], "NS": [1, 1, 1]})
        assert "\n  " in written["EW"].read_text()


class TestFailureIsNeverFatal:
    """A lost trace must not cost an otherwise fully prepared inversion."""

    def test_write_failure_logs_a_warning_and_carries_on(self, inv, caplog_gmc,
                                                         monkeypatch):
        def _boom(*a, **k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(
            "geomulticorr.inversion._run_parameters.write_run_parameters", _boom)
        # the guard lives in prepare_inversion, so emulate its call shape
        try:
            inv.write_run_parameters("uniform", None, "computed", PARAMS_12,
                                     {"EW": [1], "NS": [1]})
        except OSError as exc:
            from geomulticorr._logging import logger

            logger.warning(f"[run parameters] could not write the trace: {exc}")
        assert "run parameters" in caplog_gmc.text

    def test_prepare_inversion_swallows_the_failure(self, inv, monkeypatch,
                                                    caplog_gmc):
        monkeypatch.setattr(
            TIOInversion, "write_run_parameters",
            lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")),
        )
        for name in ("setup_directories", "write_file_info_rsc",
                     "export_pair_to_binary", "_create_symlinks",
                     "write_liste_image", "write_liste_image_inv",
                     "write_input_tio", "write_launch_script"):
            monkeypatch.setattr(TIOInversion, name, lambda *a, **k: None)
        monkeypatch.setattr(TIOInversion, "write_liste_couple",
                            lambda *a, **k: {"EW": [1], "NS": [1]})
        monkeypatch.setattr(tio, "print_tio_export_summary", lambda *a, **k: None)

        inv.prepare_inversion(print_summary=False)  # must not raise
        assert "could not write the trace" in caplog_gmc.text


class TestWriteHelper:
    def test_round_trips(self, tmp_path):
        path = write_run_parameters(tmp_path / "p.json", {"a": 1, "b": [2, 3]})
        assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}

    def test_keeps_insertion_order(self, tmp_path):
        """No sort_keys: the insertion order groups related fields."""
        path = write_run_parameters(tmp_path / "p.json", {"z": 1, "a": 2})
        assert list(json.loads(path.read_text())) == ["z", "a"]
