#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# profiles.py
# creation date: 2026-11-13.
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
"""Cluster profiles: one table that says how each execution target works.

A *profile* is what used to be spread over ``if cluster == "gricad"`` branches:
which scheduler submits the job, which OAR project it is billed to, and which
lines set up the environment inside the job script.

=========  ==========  ==========================================================
Profile    Scheduler   What happens
=========  ==========  ==========================================================
``local``  ``local``   ``bash <script>`` on this machine (also ``cluster=None``)
``gricad`` ``oar``     one ``oarsub -S`` per script (``#OAR`` header in the script)
``isterre`` ``oar``    same, ISTerre modules and micromamba
``cigri``  ``cigri``   **one** CiGri campaign for all scripts (GRICAD grid; the
                       scripts carry no ``#OAR`` header, CiGri passes the options)
=========  ==========  ==========================================================

The grid target is chosen like any other, through the cluster string, which
also names the grid clusters and the test mode (:func:`parse_cluster`):

==========================  =====================================================
``"cigri"``                 grid cluster ``dahu`` (:data:`DEFAULT_GRID_CLUSTERS`)
``"cigri:dahu,luke"``       those grid clusters (``gridclusters`` on killeen)
``"cigri-test"``            test mode: one job at a time per cluster, high priority
``"cigri-test:dahu,luke"``  both
==========================  =====================================================

``CLUSTER_CHOICES`` keeps its historical four values: it is what the TIO
inversion accepts, and a whole inversion is one job, which a grid campaign does
not help with. Correlation, a bag of independent pair jobs, accepts
:data:`CORRELATION_CLUSTER_CHOICES`, which adds ``"cigri"``.

No GMC import here: the module is used by ``correlation`` and ``inversion``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ClusterProfile",
    "ClusterSpec",
    "PROFILES",
    "DEFAULT_GRID_CLUSTERS",
    "parse_cluster",
    "is_grid_cluster",
    "CLUSTER_CHOICES",
    "GRID_CHOICES",
    "CORRELATION_CLUSTER_CHOICES",
    "get_profile",
    "validate_cluster",
    "oar_header",
    "cluster_base_env",
    "asp_env_lines",
]


@dataclass(frozen=True)
class ClusterProfile:
    """How one execution target is reached.

    Attributes
    ----------
    name:
        The ``cluster=`` value that selects it.
    scheduler:
        ``"local"``, ``"oar"`` (one ``oarsub`` per script) or ``"cigri"`` (one grid
        campaign for every script).
    project:
        Default OAR project the jobs are billed to (``None`` for local).
    base_env:
        Cluster-level setup lines, written first in the job script.
    asp_env:
        ASP-specific lines written after ``base_env``; ``{conda_env}`` is
        substituted. An empty string writes a blank line.
    """

    name: str
    scheduler: str
    project: str | None = None
    base_env: tuple[str, ...] = ()
    asp_env: tuple[str, ...] = ()

    @property
    def is_local(self) -> bool:
        return self.scheduler == "local"


_GRICAD_ENV = ("source $HOME/.bashrc",)
_GRICAD_ASP = ("conda activate {conda_env}",)

PROFILES: dict[str, ClusterProfile] = {
    "local": ClusterProfile("local", "local"),
    "gricad": ClusterProfile("gricad", "oar", "rgdyn", _GRICAD_ENV, _GRICAD_ASP),
    "isterre": ClusterProfile(
        "isterre",
        "oar",
        "iste-equ-risques",
        ("source /soft/env.bash",),
        ("module load stereopipeline/3.6.0", "", "micromamba activate {conda_env}"),
    ),
    # CiGri jobs run on the GRICAD clusters (dahu, bigfoot, luke), so the job
    # environment is GRICAD's. Only the submission differs.
    "cigri": ClusterProfile("cigri", "cigri", "rgdyn", _GRICAD_ENV, _GRICAD_ASP),
}

#: Targets that accept any job, including a single long one (the TIO inversion).
CLUSTER_CHOICES: tuple = (None, "local", "gricad", "isterre")
#: Grid targets: many independent jobs submitted as one campaign.
GRID_CHOICES: tuple = ("cigri",)
#: What the correlation step accepts.
CORRELATION_CLUSTER_CHOICES: tuple = CLUSTER_CHOICES + GRID_CHOICES

#: Backward-compatible view of the per-profile projects.
_OAR_PROJECT: dict[str, str] = {
    name: p.project for name, p in PROFILES.items() if p.scheduler == "oar"
}


#: Grid clusters used by a bare ``"cigri"``.
DEFAULT_GRID_CLUSTERS: tuple[str, ...] = ("dahu",)

_GRID_SPEC_RE = re.compile(
    r"^(?P<grid>cigri)(?P<test>-test)?(?::(?P<clusters>[^:]*))?$"
)
_GRID_CLUSTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ClusterSpec:
    """A parsed ``cluster=`` value: its profile plus the grid-only settings."""

    profile: ClusterProfile
    grid_clusters: tuple[str, ...] = ()
    test_mode: bool = False

    @property
    def is_grid(self) -> bool:
        return self.profile.scheduler == "cigri"


def is_grid_cluster(cluster) -> bool:
    """True for every spelling of a grid target (``"cigri"``, ``"cigri-test:dahu"``…).

    Only the prefix is checked, so a malformed spelling is still recognised as
    *meant* for the grid and gets the grid error message, not "unknown cluster".
    """
    return isinstance(cluster, str) and cluster.split(":", 1)[0] in (
        "cigri",
        "cigri-test",
    )


def parse_cluster(cluster) -> ClusterSpec:
    """Parse a ``cluster=`` value into a :class:`ClusterSpec`.

    ``None`` and the historical names map to their profile. Grid spellings
    (see the module docstring) carry the grid cluster list and the test mode.
    Raises ``ValueError`` on anything else.
    """
    if cluster is None:
        return ClusterSpec(PROFILES["local"])
    if is_grid_cluster(cluster):
        match = _GRID_SPEC_RE.match(cluster)
        names: tuple[str, ...] = DEFAULT_GRID_CLUSTERS
        if match and match.group("clusters") is not None:
            names = tuple(n.strip() for n in match.group("clusters").split(","))
        if (
            match is None
            or not names
            or not all(_GRID_CLUSTER_NAME_RE.match(n) for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                f"invalid grid cluster {cluster!r}: expected 'cigri', "
                f"'cigri:dahu,luke', 'cigri-test' or 'cigri-test:dahu'"
            )
        return ClusterSpec(
            PROFILES["cigri"], names, test_mode=bool(match.group("test"))
        )
    if cluster in PROFILES:
        return ClusterSpec(PROFILES[cluster])
    raise ValueError(
        f"cluster must be one of {CORRELATION_CLUSTER_CHOICES} (or 'cigri:<clusters>', "
        f"'cigri-test[:<clusters>]'), got {cluster!r}"
    )


def get_profile(cluster: str | None) -> ClusterProfile:
    """Return the profile of *cluster* (``None`` means ``"local"``)."""
    return parse_cluster(cluster).profile


def validate_cluster(cluster, choices: tuple = CLUSTER_CHOICES) -> None:
    """Raise ``ValueError`` if *cluster* is not accepted by *choices*.

    The default keeps the historical behaviour (no grid target). When *choices*
    contains ``"cigri"`` (:data:`CORRELATION_CLUSTER_CHOICES`), every grid
    spelling is accepted too and its syntax is checked.
    """
    if is_grid_cluster(cluster):
        if not any(c in GRID_CHOICES for c in choices):
            raise ValueError(
                f"cluster must be one of {choices}, got {cluster!r} ('cigri' "
                f"submits a campaign of independent jobs and is only available "
                f"for the pair correlation)"
            )
        parse_cluster(cluster)
        return
    if cluster not in choices:
        raise ValueError(f"cluster must be one of {choices}, got {cluster!r}")


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

    Returns ``[]`` for every profile that is not submitted with ``oarsub -S``:
    local runs need none, and CiGri passes the OAR options on the command line
    itself (from the campaign JDL), so a header there would be dead text.

    Parameters
    ----------
    job_name:
        OAR job name (``#OAR -n``).
    nodes, cores:
        Number of nodes and cores per node.
    walltime:
        Maximum job duration in ``HH:MM:SS``.
    cluster:
        A ``cluster=`` value.
    project:
        OAR project allocation name. Defaults to the profile's project.
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
    profile = PROFILES.get(cluster) if cluster is not None else None
    if profile is None or profile.scheduler != "oar":
        return []
    if project is None:
        project = profile.project
    _out = (
        Path(output_dir) / f"{job_name}_%jobid%.out"
        if output_dir
        else f"./{job_name}_%jobid%.out"
    )
    _err = (
        Path(output_dir) / f"{job_name}_%jobid%.err"
        if output_dir
        else f"./{job_name}_%jobid%.err"
    )
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


def cluster_base_env(cluster: str) -> list[str]:
    """Return the base environment-setup lines for *cluster*.

    This covers only the cluster-level source command. Application-specific
    ``module load`` and ``conda activate`` lines are the caller's responsibility
    (see :func:`asp_env_lines` for ASP).

    Returns
    -------
    list[str]
        ``["source $HOME/.bashrc\\n"]`` for gricad and cigri,
        ``["source /soft/env.bash\\n"]`` for isterre,
        ``[]`` for ``None`` / ``"local"``.
    """
    profile = PROFILES.get(cluster) if cluster is not None else None
    if profile is None:
        return []
    return [f"{line}\n" for line in profile.base_env]


def asp_env_lines(cluster: str | None, conda_env: str = "gmc_env") -> list[str]:
    """Return the full environment block of an ASP job script for *cluster*.

    Base environment, then the ASP lines, framed by blank lines; ``[]`` for a
    local run (the script uses resolved binary paths instead).
    """
    profile = get_profile(cluster)
    if profile.is_local:
        return []
    lines = ["\n"] + cluster_base_env(profile.name)
    lines += [f"{line.format(conda_env=conda_env)}\n" for line in profile.asp_env]
    lines.append("\n")
    return lines
