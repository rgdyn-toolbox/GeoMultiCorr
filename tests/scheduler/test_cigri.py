#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_cigri.py
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
"""CiGri campaigns: JDL, runner, grid commands. No grid needed: ``run=`` records."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from geomulticorr.scheduler import cigri
from geomulticorr.scheduler.cigri import (
    CigriError,
    CigriJob,
    CigriOptions,
    build_jdl,
    campaign_events,
    campaign_status,
    cancel_campaign,
    fix_campaign_events,
    parse_campaign_id,
    read_campaign_record,
    resolve_host,
    runner_script_text,
    submit_campaign,
    write_campaign,
)


def _job(name="A", root="/b"):
    return CigriJob(
        name, Path(f"{root}/{name}/job.sh"), (Path(f"{root}/{name}/o.tif"),)
    )


def _jdl(**kw):
    kw.setdefault("walltime", "02:00:00")
    kw.setdefault("resources", "/nodes=1/core=8")
    return build_jdl(
        "gmc_Z",
        "/b/run.sh",
        kw.pop("jobs", [_job()]),
        kw.pop("clusters", ["dahu"]),
        **kw,
    )


class Recorder:
    """A ``subprocess.run`` stand-in returning canned answers."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.calls = []
        self.answer = types.SimpleNamespace(
            stdout=stdout, stderr=stderr, returncode=returncode
        )

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return self.answer


# ─────────────────────────────────────────────────────────────────────────────


class TestBuildJdl:
    def test_keys_and_param_line(self):
        jdl = _jdl(project="rgdyn")
        assert jdl["name"] == "gmc_Z"
        assert jdl["params"] == ["A /b/A/job.sh /b/A/o.tif"]
        assert jdl["exec_file"] == "/b/run.sh"
        assert jdl["type"] == "best-effort"
        assert jdl["walltime"] == "02:00:00"
        assert jdl["resources"] == "/nodes=1/core=8"
        assert jdl["project"] == "rgdyn"
        assert jdl["clusters"] == {"dahu": {}}
        json.dumps(jdl)  # JSON-ready

    def test_booleans_are_strings(self):
        # CiGri compares test_mode == "true": a JSON true would be ignored.
        assert _jdl(test_mode=True)["test_mode"] == "true"
        assert _jdl()["test_mode"] == "false"
        jdl = _jdl(
            clusters={"dahu": {"temporal_grouping": False}},
            extra={"dimensional_grouping": True},
        )
        assert jdl["clusters"]["dahu"]["temporal_grouping"] == "false"
        assert jdl["dimensional_grouping"] == "true"

    def test_per_cluster_overrides_are_kept(self):
        jdl = _jdl(clusters={"dahu-oar3": {"cluster_job_type": "devel"}, "luke": {}})
        assert jdl["clusters"] == {
            "dahu-oar3": {"cluster_job_type": "devel"},
            "luke": {},
        }

    def test_duplicate_job_names_raise(self):
        with pytest.raises(ValueError, match="unique"):
            _jdl(jobs=[_job("A"), _job("A", root="/c")])

    def test_whitespace_in_a_token_raises(self):
        job = CigriJob("A", Path("/my project/job.sh"))
        with pytest.raises(ValueError, match="whitespace"):
            _jdl(jobs=[job])

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"jobs": []}, "at least one job"),
            ({"job_type": "fast"}, "job_type"),
            ({"walltime": "2h"}, "HH:MM:SS"),
            ({"clusters": []}, "at least one CiGri cluster"),
            ({"max_jobs": 0}, "max_jobs"),
            ({"extra": {"params": []}}, "cannot override"),
        ],
    )
    def test_invalid_inputs(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _jdl(**kw)

    def test_optional_keys(self):
        jdl = _jdl(
            max_jobs=50, prologue=["set -e"], epilogue=["true"], exec_directory="/b"
        )
        assert (jdl["max_jobs"], jdl["prologue"], jdl["epilogue"]) == (
            50,
            ["set -e"],
            ["true"],
        )
        assert jdl["exec_directory"] == "/b"


class TestOptions:
    def test_coerce(self):
        assert CigriOptions.coerce(None) == CigriOptions()
        assert CigriOptions.coerce({"test_mode": True}).test_mode is True

    def test_unknown_key_raises(self):
        with pytest.raises(TypeError, match="test_mod"):
            CigriOptions.coerce({"test_mod": True})


class TestResolveHost:
    def test_env_var_wins(self):
        assert (
            resolve_host({"GMC_CIGRI_HOST": "grid.alias"}, which=lambda _: "/x")
            == "grid.alias"
        )

    def test_env_local_means_here(self):
        assert resolve_host({"GMC_CIGRI_HOST": "local"}, which=lambda _: None) is None

    def test_gridsub_on_path_means_here(self):
        assert resolve_host({}, which=lambda _: "/usr/bin/gridsub") is None

    def test_otherwise_ssh_to_killeen(self):
        assert resolve_host({}, which=lambda _: None) == "killeen.ciment"


# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def campaign(tmp_path):
    jobs = [_job("A", str(tmp_path)), _job("B", str(tmp_path))]
    return write_campaign(
        jobs,
        tmp_path,
        {"clusters": ["dahu", "luke"]},
        name="gmc_Z",
        walltime="01:00:00",
        resources="/nodes=1/core=4",
        project="rgdyn",
    )


class TestWriteCampaign:
    def test_files_are_flat_with_the_cigri_prefix(self, campaign, tmp_path):
        names = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
        assert names == [
            "cigri_gmc_Z.jdl",
            "cigri_gmc_Z_params.txt",
            "gmc_cigri_job.sh",
        ]
        assert not [p for p in tmp_path.iterdir() if p.is_dir()]

    def test_jdl_on_disk(self, campaign, tmp_path):
        jdl = json.loads(campaign.jdl_path.read_text())
        assert jdl["exec_file"] == str(tmp_path.resolve() / "gmc_cigri_job.sh")
        assert jdl["exec_directory"] == str(tmp_path.resolve())
        assert list(jdl["clusters"]) == ["dahu", "luke"]
        params = (tmp_path / "cigri_gmc_Z_params.txt").read_text().splitlines()
        assert params == jdl["params"]

    def test_runner_is_executable(self, campaign):
        assert os.access(campaign.runner_path, os.X_OK)


class TestSubmit:
    def test_local_submission(self, campaign):
        run = Recorder(stdout="Campaign successfully submitted\nCAMPAIGN_ID=19289\n")
        assert submit_campaign(campaign, None, run=run) == 19289
        assert run.calls[0][0] == ["gridsub", "-f", str(campaign.jdl_path)]
        record = read_campaign_record(campaign.workdir / "cigri_campaign_19289.json")
        assert record["campaign_id"] == 19289
        assert [j["name"] for j in record["jobs"]] == ["A", "B"]

    def test_ssh_submission_pipes_the_jdl(self, campaign):
        run = Recorder(stdout="CAMPAIGN_ID=7\n")
        submit_campaign(campaign, "killeen.ciment", run=run)
        cmd, kwargs = run.calls[0]
        assert cmd[:4] == ["ssh", "-o", "BatchMode=yes", "killeen.ciment"]
        assert "gridsub -f" in cmd[4]
        assert json.loads(kwargs["input"]) == campaign.jdl
        assert (
            read_campaign_record(campaign.workdir / "cigri_campaign_7.json")["host"]
            == "killeen.ciment"
        )

    def test_admission_error_with_exit_zero_raises(self, campaign):
        # gridsub exits 0 even when the campaign is refused.
        run = Recorder(stderr="Admission rule error: \n  too many jobs", returncode=0)
        with pytest.raises(CigriError, match="Admission rule error"):
            submit_campaign(campaign, None, run=run)
        assert campaign.campaign_id is None

    def test_missing_gridsub_is_explained(self, campaign):
        def run(cmd, **kw):
            raise FileNotFoundError(cmd[0])

        with pytest.raises(CigriError, match="killeen.ciment"):
            submit_campaign(campaign, None, run=run)

    def test_parse_campaign_id(self):
        assert parse_campaign_id("CAMPAIGN_ID=42") == 42
        assert parse_campaign_id("Error: nope") is None


class TestStatusCommands:
    def test_status(self):
        body = {
            "id": 5,
            "name": "gmc_Z",
            "state": "in_treatment",
            "finished_jobs": 3,
            "total_jobs": 4,
            "has_events": True,
        }
        run = Recorder(stdout=json.dumps(body))
        status = campaign_status(5, "killeen.ciment", run=run)
        assert run.calls[0][0] == [
            "ssh",
            "-o",
            "BatchMode=yes",
            "killeen.ciment",
            "gridstat -c 5 -d",
        ]
        assert (status.progress, status.is_finished, status.has_events) == (
            0.75,
            False,
            True,
        )
        assert "3/4" in str(status)

    def test_status_not_json(self):
        with pytest.raises(CigriError, match="no JSON"):
            campaign_status(5, run=Recorder(stdout="API server not reachable"))

    def test_status_not_found(self):
        with pytest.raises(CigriError, match="not found"):
            campaign_status(5, run=Recorder(stdout='{"status": 404}'))

    def test_events(self):
        run = Recorder(stdout=json.dumps({"items": [{"id": 1, "code": "BLACKLIST"}]}))
        assert campaign_events(5, run=run) == [{"id": 1, "code": "BLACKLIST"}]
        assert run.calls[0][0] == ["gridstat", "-c", "5", "-e", "-d"]

    def test_fix_events_resubmits_by_default(self):
        run = Recorder()
        fix_campaign_events(5, run=run)
        assert run.calls[0][0] == ["gridevents", "-c", "5", "-f", "-r"]

    def test_cancel_reports_failures(self):
        run = Recorder(stderr="Failed to cancel campaign 5: forbidden.")
        with pytest.raises(CigriError, match="forbidden"):
            cancel_campaign(5, run=run)


# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestRunnerScript:
    @pytest.fixture
    def setup(self, tmp_path):
        runner = tmp_path / "gmc_cigri_job.sh"
        runner.write_text(runner_script_text())
        pair = tmp_path / "pair"
        pair.mkdir()
        out = pair / "k-F.tif"
        script = pair / "k_CorrelationJob.sh"
        launch = tmp_path / "launch"
        launch.mkdir()
        return types.SimpleNamespace(
            runner=runner, pair=pair, out=out, script=script, launch=launch
        )

    def _run(self, s):
        env = dict(os.environ, CIGRI_CAMPAIGN_ID="9", OAR_JOB_ID="77")
        return subprocess.run(
            ["bash", str(s.runner), "k", str(s.script), str(s.out)],
            cwd=s.launch,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_syntax(self, setup):
        assert subprocess.run(["bash", "-n", str(setup.runner)]).returncode == 0

    def test_runs_the_job_in_its_folder_and_logs_there(self, setup):
        setup.script.write_text('echo running in "$PWD"\necho data > k-F.tif\n')
        assert self._run(setup).returncode == 0
        log = (setup.pair / "k_cigri.log").read_text()
        assert f"running in {setup.pair}" in log and "k: done" in log

    def test_existing_outputs_skip_the_job(self, setup):
        setup.out.write_text("data")
        setup.script.write_text("exit 1\n")
        assert self._run(setup).returncode == 0
        assert "nothing to do" in (setup.pair / "k_cigri.log").read_text()

    def test_missing_output_is_a_failure(self, setup):
        setup.script.write_text("true\n")
        assert self._run(setup).returncode == 3

    def test_job_failure_is_propagated(self, setup):
        setup.script.write_text("exit 5\n")
        assert self._run(setup).returncode == 5

    def test_own_oar_cigri_files_are_removed(self, setup):
        mine = [setup.launch / f"OAR.cigri.9.77.{e}" for e in ("stdout", "stderr")]
        other = setup.launch / "OAR.cigri.9.78.stdout"
        for f in (*mine, other):
            f.write_text("")
        setup.script.write_text("echo data > k-F.tif\n")
        assert self._run(setup).returncode == 0
        assert not any(f.exists() for f in mine)
        assert other.exists()
