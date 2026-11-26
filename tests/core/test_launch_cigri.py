#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_launch_cigri.py
# creation date: 2026-11-20.
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
"""``launch_pairs_correlation(cluster="cigri…")`` and ``cigri_status`` on a stub session.

Nothing reaches a grid: ``gridsub`` / ``gridstat`` go through a recorded
``subprocess.run`` in :mod:`geomulticorr.scheduler.cigri`.
"""

from __future__ import annotations

import functools
import json
import types

import pytest

from geomulticorr.core import session as session_mod
from geomulticorr.core.session import Session
from geomulticorr.scheduler import cigri as cigri_mod

PZ = "Zone"


def _pair(root, pz, key, pa_type="opt"):
    folder = root / pz / "image_correlation" / key
    folder.mkdir(parents=True)
    (folder / f"{key}_CorrelationJob.sh").write_text("echo hi\n")
    return types.SimpleNamespace(
        pa_key=key, pa_path=folder, pa_pz_name=pz, pa_type=pa_type
    )


@pytest.fixture
def grid(tmp_path, monkeypatch):
    """A stub session with pairs in two pzones, and a recorded grid."""
    pairs = [
        _pair(tmp_path, "Zone", "Zone_2020-08-15_s2__2021-08-15_s2"),
        _pair(tmp_path, "Zone", "Zone_2021-08-15_s2__2022-08-15_s2"),
        _pair(tmp_path, "Other", "Other_2020-08-15_s2__2021-08-15_s2"),
    ]
    state = types.SimpleNamespace(calls=[], next_id=100, synced=[], stdout=None)

    def fake_run(cmd, **kwargs):
        state.calls.append((cmd, kwargs))
        if state.stdout is not None:
            return types.SimpleNamespace(stdout=state.stdout, stderr="", returncode=0)
        state.next_id += 1
        return types.SimpleNamespace(
            stdout=f"Campaign successfully submitted\nCAMPAIGN_ID={state.next_id}\n",
            stderr="",
            returncode=0,
        )

    # One module object: this also records anything the OAR/local path would run.
    assert session_mod.subprocess is cigri_mod.subprocess
    monkeypatch.setattr(cigri_mod.subprocess, "run", fake_run)
    monkeypatch.setenv("GMC_CIGRI_HOST", "local")

    stub = types.SimpleNamespace(
        get_pairs=lambda *a, **k: list(pairs),
        _sync_pairs_status=lambda p: state.synced.extend(p),
        pz_dir=lambda pz, kind: tmp_path / pz / kind,
        get_pzones_overview=lambda *a, **k: {
            "pz_name": types.SimpleNamespace(tolist=lambda: ["Zone", "Other"])
        },
    )
    stub._launch_cigri_campaigns = functools.partial(
        Session._launch_cigri_campaigns, stub
    )

    def launch(**kwargs):
        kwargs.setdefault("print_summary", False)
        state.codes = Session.launch_pairs_correlation(stub, **kwargs)
        return state

    state.launch = launch
    state.stub = stub
    state.root = tmp_path
    state.pairs = pairs
    return state


def _jdl(root, pz, name=None):
    return json.loads(
        (
            root / pz / "image_correlation" / f"cigri_{name or 'gmc_' + pz}.jdl"
        ).read_text()
    )


class TestDryRun:
    def test_one_campaign_per_pzone_and_nothing_submitted(self, grid, caplog_gmc):
        grid.launch(cluster="cigri", dry_run=True)
        assert grid.codes == [0, 0, 0]
        assert grid.calls == []
        assert grid.synced == []
        assert len(_jdl(grid.root, "Zone")["params"]) == 2
        assert len(_jdl(grid.root, "Other")["params"]) == 1
        assert "DRY RUN: gridsub -f" in caplog_gmc.text

    def test_files_are_flat_in_image_correlation(self, grid):
        grid.launch(cluster="cigri", dry_run=True)
        folder = grid.root / "Zone" / "image_correlation"
        files = sorted(p.name for p in folder.iterdir() if p.is_file())
        assert files == [
            "cigri_gmc_Zone.jdl",
            "cigri_gmc_Zone_params.txt",
            "gmc_cigri_job.sh",
        ]
        # Only the pair folders, no new directory.
        assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == sorted(
            p.pa_key for p in grid.pairs if p.pa_pz_name == "Zone"
        )

    def test_jdl_comes_from_the_existing_arguments(self, grid):
        grid.launch(
            cluster="cigri-test:dahu,luke", dry_run=True, cores=4, walltime="03:00:00"
        )
        jdl = _jdl(grid.root, "Zone")
        assert jdl["resources"] == "/nodes=1/core=4"
        assert jdl["walltime"] == "03:00:00"
        assert jdl["test_mode"] == "true"
        assert jdl["type"] == "best-effort"
        assert jdl["project"] == "rgdyn"
        assert list(jdl["clusters"]) == ["dahu", "luke"]

    def test_param_line_requires_both_outputs(self, grid):
        grid.launch(cluster="cigri", dry_run=True)
        pair = grid.pairs[0]
        line = _jdl(grid.root, "Zone")["params"][0].split()
        assert line[0] == pair.pa_key
        assert line[1] == str(pair.pa_path / f"{pair.pa_key}_CorrelationJob.sh")
        assert line[2:] == [
            str(pair.pa_path / f"{pair.pa_key}-F.tif"),
            str(pair.pa_path / f"{pair.pa_key}-F-ncc.tif"),
        ]

    def test_without_correval_only_the_disparity_is_required(self, grid):
        grid.launch(cluster="cigri", dry_run=True, correval=False)
        assert len(_jdl(grid.root, "Zone")["params"][0].split()) == 3

    def test_image_types_get_separate_campaigns(self, grid):
        # pa_key is unique only within one type: CiGri needs unique job names.
        twin = _pair(grid.root / "slope", "Zone", grid.pairs[0].pa_key, pa_type="slope")
        grid.pairs.append(twin)
        grid.launch(cluster="cigri", dry_run=True)
        assert len(_jdl(grid.root, "Zone")["params"]) == 2
        assert len(_jdl(grid.root, "Zone", "gmc_Zone_slope")["params"]) == 1


class TestSubmission:
    def test_submits_once_per_pzone_and_stashes_ids(self, grid):
        grid.launch(cluster="cigri:dahu")
        assert grid.codes == [0, 0, 0]
        assert [c[0] for c, _ in grid.calls] == ["gridsub", "gridsub"]  # no oarsub/bash
        assert grid.stub._last_cigri_campaigns == {
            "gmc_Zone": (101, None),
            "gmc_Other": (102, None),
        }
        assert (
            grid.root / "Zone" / "image_correlation" / "cigri_campaign_101.json"
        ).exists()
        assert grid.synced == []  # asynchronous, like oarsub

    def test_failed_submission_gives_code_1_for_its_pairs(self, grid):
        grid.stdout = "Error: admission rule"
        grid.launch(cluster="cigri")
        assert grid.codes == [1, 1, 1]

    def test_status_of_the_last_campaigns(self, grid):
        grid.launch(cluster="cigri")
        grid.stdout = json.dumps(
            {
                "id": 101,
                "name": "gmc_Zone",
                "state": "terminated",
                "finished_jobs": 2,
                "total_jobs": 2,
                "has_events": False,
            }
        )
        statuses = Session.cigri_status(grid.stub, print_summary=False)
        assert [s.is_finished for s in statuses] == [True, True]
        assert grid.calls[-1][0] == ["gridstat", "-c", "102", "-d"]

    def test_status_after_a_restart_reads_the_records(self, grid):
        grid.launch(cluster="cigri")
        del grid.stub._last_cigri_campaigns
        grid.stdout = json.dumps(
            {"id": 0, "state": "in_treatment", "finished_jobs": 0, "total_jobs": 1}
        )
        statuses = Session.cigri_status(grid.stub, print_summary=False)
        asked = sorted(c[2] for c, _ in grid.calls if c[0] == "gridstat")
        assert asked == ["101", "102"]
        assert len(statuses) == 2

    def test_status_without_any_campaign_warns(self, grid, caplog_gmc):
        assert Session.cigri_status(grid.stub, print_summary=False) == []
        assert "no campaign known" in caplog_gmc.text


def test_prepare_scripts_for_cigri_have_no_oar_header(tmp_path):
    from geomulticorr.correlation.correlation import ASP

    asp = ASP.__new__(ASP)
    asp._build_stereo_cmd = lambda *a, **k: ["parallel_stereo", "L", "R", "out"]
    asp._build_correval_cmd = lambda **k: ["corr_eval", "out"]
    pair = types.SimpleNamespace(
        pa_left_link_path=tmp_path / "L.tif",
        pa_right_link_path=tmp_path / "R.tif",
        pa_path=tmp_path,
        pa_key="k",
    )
    text = asp.write_bash_script(
        pair, cluster="cigri:dahu,luke", conda_env="E"
    ).read_text()
    assert "#OAR" not in text
    assert "source $HOME/.bashrc\nconda activate E\n" in text
    assert "parallel_stereo" in text and "corr_eval" in text


def test_tio_rejects_cigri():
    from geomulticorr.scheduler.profiles import validate_cluster

    with pytest.raises(ValueError, match="pair correlation"):
        validate_cluster("cigri")
