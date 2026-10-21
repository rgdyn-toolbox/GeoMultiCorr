#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Connectivity of the pair network — the check the solver cannot make itself.

Two sensors paired only among themselves give two components. The solver
still returns a series (the smoothing prior fills the gap) with a clean
closure RMS and ``rank_defect == 0`` — see
``tests/inversion/pytio/test_kernel.py::TestDisconnectedNetwork`` — so the
warning here is the only place a user learns that the offset between the
components was never measured.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import geomulticorr.inversion.tio_inversion as tio
from geomulticorr.inversion.tio_inversion import (TIOInversion,
                                                  connected_date_components)


class TestConnectedDateComponents:
    def test_two_components(self):
        assert connected_date_components([("a", "b"), ("b", "c"), ("d", "e")]) == \
            [["a", "b", "c"], ["d", "e"]]

    def test_empty(self):
        assert connected_date_components([]) == []

    def test_orientation_and_duplicates_collapse(self):
        edges = [("b", "a"), ("a", "b"), ("a", "b"), ("c", "b")]
        assert connected_date_components(edges) == [["a", "b", "c"]]

    def test_self_loop_is_a_component_of_one(self):
        assert connected_date_components([("x", "x")]) == [["x"]]

    def test_ordered_by_earliest_date(self):
        edges = [("20220101", "20230101"), ("20160101", "20170101")]
        comps = connected_date_components(edges)
        assert comps[0][0] == "20160101"
        assert comps[1][0] == "20220101"

    def test_a_bridge_pair_joins_the_components(self):
        edges = [("a", "b"), ("c", "d"), ("b", "c")]
        assert len(connected_date_components(edges)) == 1


def _pair(d1, d2, s1=None, s2=None):
    left = SimpleNamespace(th_date=d1)
    right = SimpleNamespace(th_date=d2)
    if s1 is not None:
        left.th_sensor = s1
    if s2 is not None:
        right.th_sensor = s2
    return SimpleNamespace(pa_key=f"PZ_{d1}-{s1}_{d2}-{s2}", pa_dt_days=30,
                           pa_left=left, pa_right=right)


PS = [("2016-08-16", "2017-07-03"), ("2017-07-03", "2018-07-29"), ("2018-07-29", "2019-07-13")]
SPOT = [("2018-09-27", "2021-08-29"), ("2021-08-29", "2024-07-17"), ("2018-09-27", "2024-07-17")]


def _inv(pairs, tmp_path):
    obj = TIOInversion.__new__(TIOInversion)
    obj.pairs = pairs
    obj.inversion_name = "t"
    obj.pzone_name = "PZ"
    obj.inversion_dir = tmp_path
    obj._DIRECTIONS = ("EW", "NS")
    obj._image_dates = None
    obj.filter_pipeline = None
    obj._last_launch = None
    obj._last_network = None
    obj._quality_metrics = None
    obj._last_weights = None
    obj._last_weights_params = None
    obj._last_nmad_filter = None
    obj.solver = tio.TIOConfig()
    for d in obj._DIRECTIONS:
        (tmp_path / f"inverse_{d}").mkdir(exist_ok=True)
    return obj


class TestNetworkComponents:
    def test_two_sensor_groups(self, tmp_path):
        pairs = ([_pair(a, b, "planetscope", "planetscope") for a, b in PS]
                 + [_pair(a, b, "SPOT6", "spot7") for a, b in SPOT])
        comps = _inv(pairs, tmp_path).network_components()
        assert len(comps) == 2
        ps, spot = comps
        assert ps["sensors"] == ["planetscope"]
        assert ps["start"] == "20160816" and ps["end"] == "20190713"
        assert ps["n_dates"] == 4 and ps["n_pairs"] == 3
        assert spot["sensors"] == ["spot6", "spot7"]
        assert spot["n_dates"] == 3 and spot["n_pairs"] == 3
        assert ps["n_pairs"] + spot["n_pairs"] == len(pairs)

    def test_bridge_pair_makes_one_component(self, tmp_path):
        pairs = ([_pair(a, b, "planetscope", "planetscope") for a, b in PS]
                 + [_pair(a, b, "spot6", "spot6") for a, b in SPOT]
                 + [_pair("2018-07-29", "2018-09-27", "planetscope", "spot6")])
        comps = _inv(pairs, tmp_path).network_components()
        assert len(comps) == 1
        assert comps[0]["sensors"] == ["planetscope", "spot6"]
        assert comps[0]["n_pairs"] == 7

    def test_pairs_without_sensor_attribute(self, tmp_path):
        comps = _inv([_pair(a, b) for a, b in PS], tmp_path).network_components()
        assert len(comps) == 1
        assert comps[0]["sensors"] == []

    def test_not_cached_follows_pair_list_mutation(self, tmp_path):
        inv = _inv([_pair(a, b, "s", "s") for a, b in PS + SPOT], tmp_path)
        assert len(inv.network_components()) == 2
        inv.pairs = inv.pairs[:3]
        assert len(inv.network_components()) == 1


class TestPrepareInversionWarning:
    def _stub(self, monkeypatch):
        for name in ("setup_directories", "write_file_info_rsc", "export_pair_to_binary",
                     "_create_symlinks", "write_liste_image", "write_liste_image_inv",
                     "write_input_tio", "write_launch_script", "write_run_parameters"):
            monkeypatch.setattr(TIOInversion, name, lambda *a, **k: None)
        monkeypatch.setattr(TIOInversion, "write_liste_couple",
                            lambda *a, **k: {"EW": [1], "NS": [1]})
        monkeypatch.setattr(tio, "print_tio_export_summary", lambda *a, **k: None)

    def test_disconnected_network_warns_and_is_stashed(self, tmp_path, monkeypatch, caplog_gmc):
        self._stub(monkeypatch)
        pairs = ([_pair(a, b, "planetscope", "planetscope") for a, b in PS]
                 + [_pair(a, b, "spot6", "spot7") for a, b in SPOT])
        inv = _inv(pairs, tmp_path)
        inv.prepare_inversion(print_summary=False)
        assert "2 disconnected components" in caplog_gmc.text
        assert "20180927–20240717" in caplog_gmc.text
        assert "[spot6,spot7]" in caplog_gmc.text
        assert "geomulticorr.inversion.fusion" in caplog_gmc.text
        assert inv._last_network["n_components"] == 2

    def test_connected_network_does_not_warn(self, tmp_path, monkeypatch, caplog_gmc):
        self._stub(monkeypatch)
        inv = _inv([_pair(a, b, "planetscope", "planetscope") for a, b in PS], tmp_path)
        inv.prepare_inversion(print_summary=False)
        assert "disconnected" not in caplog_gmc.text
        assert "1 connected component" in caplog_gmc.text
        assert inv._last_network["n_components"] == 1

    def test_network_reaches_the_trace(self, tmp_path, monkeypatch):
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        monkeypatch.setattr("geomulticorr.inversion._run_parameters.build_run_parameters", fake_build)
        monkeypatch.setattr("geomulticorr.inversion._run_parameters.write_run_parameters",
                            lambda path, doc: path)
        pairs = [_pair(a, b, "s", "s") for a, b in PS + SPOT]
        inv = _inv(pairs, tmp_path)
        inv._raster_width, inv._raster_height = 4, 3
        inv._last_network = {"n_components": 2, "components": inv.network_components()}
        inv.write_run_parameters("uniform", None, "computed", {}, {"EW": [1], "NS": [1]})
        assert captured["network"]["n_components"] == 2
        assert captured["solver"]["iponder"] == 2
