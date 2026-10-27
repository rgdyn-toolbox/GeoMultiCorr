#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the shared OAR/HPC helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geomulticorr.utils import hpc_tools
from geomulticorr.utils.hpc_tools import (
    CLUSTER_CHOICES,
    cluster_base_env,
    oar_header,
    oarsub_submit,
    run_local,
    validate_cluster,
)


@pytest.fixture
def recorded_run(monkeypatch):
    """Replace ``subprocess.run`` with a recorder; return the call list."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hpc_tools.subprocess, "run", fake_run)
    return calls


class TestValidateCluster:
    @pytest.mark.parametrize("cluster", CLUSTER_CHOICES)
    def test_accepts_every_documented_choice(self, cluster):
        assert validate_cluster(cluster) is None

    def test_rejects_unknown_cluster(self):
        with pytest.raises(ValueError, match="cluster must be one of"):
            validate_cluster("bogus")


class TestOarHeader:
    @pytest.mark.parametrize("cluster", [None, "local"])
    def test_no_scheduler_no_header(self, cluster):
        assert oar_header("job", 1, 8, "01:00:00", cluster) == []

    @pytest.mark.parametrize(
        "cluster, project",
        [("gricad", "rgdyn"), ("isterre", "iste-equ-risques")],
    )
    def test_default_project_per_cluster(self, cluster, project):
        text = "".join(oar_header("job", 2, 4, "02:00:00", cluster))
        assert f"#OAR --project {project}\n" in text
        assert "#OAR -n job\n" in text
        assert "#OAR -l /nodes=2/core=4,walltime=02:00:00\n" in text

    def test_explicit_project_wins(self):
        text = "".join(oar_header("job", 1, 8, "01:00:00", "gricad", project="mine"))
        assert "#OAR --project mine\n" in text
        assert "rgdyn" not in text

    def test_logs_land_in_cwd_by_default(self):
        text = "".join(oar_header("job", 1, 8, "01:00:00", "gricad"))
        assert "#OAR -O ./job_%jobid%.out\n" in text
        assert "#OAR -E ./job_%jobid%.err\n" in text

    def test_logs_land_in_output_dir_when_given(self, tmp_path):
        text = "".join(
            oar_header("job", 1, 8, "01:00:00", "isterre", output_dir=tmp_path)
        )
        assert f"#OAR -O {tmp_path / 'job_%jobid%.out'}\n" in text
        assert f"#OAR -E {tmp_path / 'job_%jobid%.err'}\n" in text

    def test_besteffort_appends_last_line(self):
        lines = oar_header("job", 1, 8, "01:00:00", "gricad", besteffort=True)
        assert lines[-1] == "#OAR -t besteffort\n"

    def test_no_besteffort_by_default(self):
        lines = oar_header("job", 1, 8, "01:00:00", "gricad")
        assert not any("besteffort" in ln for ln in lines)


class TestClusterBaseEnv:
    def test_gricad_sources_bashrc(self):
        assert cluster_base_env("gricad") == ["source $HOME/.bashrc\n"]

    def test_isterre_sources_soft_env(self):
        assert cluster_base_env("isterre") == ["source /soft/env.bash\n"]

    @pytest.mark.parametrize("cluster", [None, "local"])
    def test_local_needs_nothing(self, cluster):
        assert cluster_base_env(cluster) == []


class TestSubmitAndRun:
    def test_oarsub_submit_defaults_cwd_to_script_dir(self, recorded_run, tmp_path):
        script = tmp_path / "job.sh"
        oarsub_submit(script)
        ((cmd, kwargs),) = recorded_run
        assert cmd == "oarsub -S ./job.sh"
        assert kwargs == {"shell": True, "cwd": str(tmp_path), "check": True}

    def test_oarsub_submit_explicit_cwd(self, recorded_run, tmp_path):
        other = tmp_path / "elsewhere"
        oarsub_submit(tmp_path / "job.sh", cwd=other)
        assert recorded_run[0][1]["cwd"] == str(other)

    def test_run_local_inherits_cwd_when_none(self, recorded_run):
        run_local("echo hi")
        ((cmd, kwargs),) = recorded_run
        assert cmd == "echo hi"
        assert kwargs == {"shell": True, "cwd": None, "check": True}

    def test_run_local_stringifies_cwd(self, recorded_run, tmp_path):
        run_local("echo hi", cwd=Path(tmp_path))
        assert recorded_run[0][1]["cwd"] == str(tmp_path)
