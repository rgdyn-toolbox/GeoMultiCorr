#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_main.py
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
"""The ``gmc`` command end to end on a stub session, plus argument handling."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import yaml

from geomulticorr.cli import main as gmc_main


def _write_cfg(tmp_path, root, **sections):
    path = tmp_path / "cfg.yaml"
    data = {"project": {"path": str(root)}, "pzones": ["Alpha"], **sections}
    path.write_text(yaml.safe_dump(data))
    return path


def test_template_to_stdout_is_valid_yaml(capsys):
    assert gmc_main.main(["correct", "--template-config"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["correct"]["corrections"]


def test_template_to_file(tmp_path):
    out = tmp_path / "t.yaml"
    assert gmc_main.main(["run", "--template-config", str(out)]) == 0
    assert "invert" in yaml.safe_load(out.read_text())


def test_missing_section_exit_2(tmp_path, stub_session, capsys):
    cfg = _write_cfg(tmp_path, stub_session.root, pairs={})
    assert gmc_main.main(["correct", str(cfg)]) == 2
    assert "no 'correct:' section" in capsys.readouterr().err


def test_invalid_config_exit_2(tmp_path, stub_session, capsys):
    cfg = _write_cfg(tmp_path, stub_session.root, pairs={"nope": 1})
    assert gmc_main.main(["pairs", str(cfg)]) == 2
    assert "nope" in capsys.readouterr().err


def test_unknown_only_exit_2(tmp_path, stub_session):
    cfg = _write_cfg(tmp_path, stub_session.root, pairs={})
    assert gmc_main.main(["run", str(cfg), "--only", "pears"]) == 2


def test_default_run_dir_keyed_on_config(tmp_path, stub_session):
    cfg_path = _write_cfg(tmp_path, stub_session.root)
    cfg = gmc_main.load_config(cfg_path)
    assert (
        gmc_main.default_run_dir(cfg, cfg_path)
        == stub_session.root / "gmc_runs" / "cfg"
    )


def test_run_stages_writes_everything(make_cfg, stub_session, tmp_path):
    cfg = make_cfg(pairs={"figures": []}, correct={"masks": ["SlopeMask"]})
    cfg.outputs.report.pdf = False
    run_dir = tmp_path / "run"
    n_figs = len(plt.get_fignums())
    code = gmc_main.run_stages(cfg, cfg.stages(), run_dir, session=stub_session)
    assert code == 0
    assert (run_dir / "used_config.yaml").exists()
    assert (run_dir / "gmc.log").exists()
    for stage in ("pairs", "correct"):
        doc = json.loads((run_dir / "records" / f"{stage}.json").read_text())
        assert doc["status"] == "ok"
    html = (run_dir / "report.html").read_text()
    assert "Corrections" in html and "Pair network" in html
    assert (run_dir / "assets" / "correct" / "nmad.png").exists()
    assert len(plt.get_fignums()) == n_figs  # no figure leaked


def test_failure_stops_and_is_reported(make_cfg, stub_session, tmp_path):
    def boom(**kwargs):
        raise RuntimeError("extract exploded")

    stub_session.extract_pairs_raw_displacements = boom
    cfg = make_cfg(extract={}, correct={})
    cfg.outputs.report.pdf = False
    run_dir = tmp_path / "run"
    assert gmc_main.run_stages(cfg, cfg.stages(), run_dir, session=stub_session) == 1
    assert (
        json.loads((run_dir / "records" / "extract.json").read_text())["status"]
        == "failed"
    )
    assert not (run_dir / "records" / "correct.json").exists()
    assert "extract exploded" in (run_dir / "report.html").read_text()


def test_profiled_run(make_cfg, stub_session, tmp_path):
    from geoutils.profiler import Profiler

    cfg = make_cfg(pairs={"figures": []})
    cfg.outputs.profiling = True
    cfg.outputs.report.pdf = False
    run_dir = tmp_path / "run"
    try:
        assert gmc_main.run_stages(cfg, ["pairs"], run_dir, session=stub_session) == 0
    finally:
        Profiler.enable(False, False)
    table = json.loads((run_dir / "profiling" / "summary.json").read_text())
    assert table[0]["name"] == "gmc.stage.pairs"
    assert (run_dir / "profiling" / "time_graph.html").exists()


def test_report_command_rerenders(make_cfg, stub_session, tmp_path, capsys):
    cfg = make_cfg(pairs={"figures": []})
    cfg.outputs.report.pdf = False
    run_dir = tmp_path / "run"
    gmc_main.run_stages(cfg, ["pairs"], run_dir, session=stub_session, report=False)
    assert not (run_dir / "report.html").exists()
    assert gmc_main.main(["report", str(run_dir), "--no-pdf"]) == 0
    assert (run_dir / "report.html").exists()


def test_dry_run_writes_nothing(tmp_path, stub_session, capsys):
    cfg = _write_cfg(tmp_path, tmp_path / "absent_project", pairs={})
    assert gmc_main.main(["pairs", str(cfg), "--dry-run"]) == 0
    assert "stages  : pairs" in capsys.readouterr().out
    assert not (tmp_path / "absent_project").exists()
