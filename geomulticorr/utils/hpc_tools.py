#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# hpc_tools.py
# creation date: 2026-05-15.
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
"""Shared HPC utilities for OAR cluster job management.

Provides cluster-profile constants, OAR header generation, environment-setup
helpers, job submission, and local execution — consumed by both the correlation
(ASP) and inversion (TIO) modules.

Supported cluster profiles
--------------------------
- ``None`` / ``"local"`` : run commands directly on the current machine
- ``"gricad"``           : GRICAD cluster (OAR + conda via bashrc)
- ``"isterre"``          : ISTerre cluster (OAR + modules + soft env)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_CHOICES: tuple = (None, "local", "gricad", "isterre")

_OAR_PROJECT: dict[str, str] = {
    "gricad":  "rgdyn",
    "isterre": "iste-equ-risques",
}

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_cluster(cluster) -> None:
    """Raise ``ValueError`` if *cluster* is not in :data:`CLUSTER_CHOICES`."""
    if cluster not in CLUSTER_CHOICES:
        raise ValueError(
            f"cluster must be one of {CLUSTER_CHOICES}, got {cluster!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# OAR header generation
# ─────────────────────────────────────────────────────────────────────────────

def oar_header(
    job_name: str,
    nodes: int,
    cores: int,
    walltime: str,
    cluster: str,
    project: str | None = None,
    besteffort: bool = False,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Return ``#OAR`` directive lines for the given cluster profile.

    Parameters
    ----------
    job_name:
        OAR job name (``#OAR -n``).
    nodes, cores:
        Number of nodes and cores per node.
    walltime:
        Maximum job duration in ``HH:MM:SS``.
    cluster:
        One of :data:`CLUSTER_CHOICES`.  Returns ``[]`` for ``None`` or
        ``"local"`` (no scheduler needed).
    project:
        OAR project allocation name.  Defaults to the canonical project for
        *cluster* (``"rgdyn"`` for gricad, ``"iste-equ-risques"`` for isterre).
    besteffort:
        If True, append ``#OAR -t besteffort`` (uses idle resources).
    output_dir:
        Directory where ``-O`` stdout and ``-E`` stderr files are written.
        When ``None``, files land in the current working directory (``./``).

    Returns
    -------
    list[str]
        Ready-to-join lines including trailing newlines.
    """
    if cluster not in ("gricad", "isterre"):
        return []
    if project is None:
        project = _OAR_PROJECT[cluster]
    _out = Path(output_dir) / f"{job_name}_%jobid%.out" if output_dir else f"./{job_name}_%jobid%.out"
    _err = Path(output_dir) / f"{job_name}_%jobid%.err" if output_dir else f"./{job_name}_%jobid%.err"
    lines = [
        f"#OAR -n {job_name}\n",
        f"#OAR -l /nodes={nodes}/core={cores},walltime={walltime}\n",
        f"#OAR -O {_out}\n",
        f"#OAR -E {_err}\n",
        f"#OAR --project {project}\n",
    ]
    if besteffort:
        lines.append("#OAR -t besteffort\n")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Cluster base environment
# ─────────────────────────────────────────────────────────────────────────────

def cluster_base_env(cluster: str) -> list[str]:
    """Return the base environment-setup lines for *cluster*.

    This covers only the cluster-level source command.  Application-specific
    ``module load`` and ``conda activate`` lines are the caller's responsibility.

    Returns
    -------
    list[str]
        ``["source $HOME/.bashrc\\n"]`` for gricad,
        ``["source /soft/env.bash\\n"]`` for isterre,
        ``[]`` for ``None`` / ``"local"``.
    """
    if cluster == "gricad":
        return ["source $HOME/.bashrc\n"]
    if cluster == "isterre":
        return ["source /soft/env.bash\n"]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Job submission and local execution
# ─────────────────────────────────────────────────────────────────────────────

def oarsub_submit(script_path: Path, cwd: Path | None = None) -> None:
    """Submit *script_path* to OAR via ``oarsub -S``.

    Parameters
    ----------
    script_path:
        Absolute path to the bash script to submit.
    cwd:
        Working directory for the ``oarsub`` call.  Defaults to
        ``script_path.parent``.
    """
    if cwd is None:
        cwd = script_path.parent
    subprocess.run(
        f"oarsub -S ./{script_path.name}",
        shell=True,
        cwd=str(cwd),
        check=True,
    )


def run_local(cmd: str, cwd: Path | None = None) -> None:
    """Run *cmd* locally via ``subprocess.run(shell=True, check=True)``.

    Parameters
    ----------
    cmd:
        Shell command string to execute.
    cwd:
        Working directory.  If ``None``, inherits from the current process.
    """
    subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
    )
