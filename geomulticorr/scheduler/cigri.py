#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# cigri.py
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
"""CiGri grid campaigns (GRICAD) for bags of independent jobs.

`CiGri <https://github.com/oar-team/cigri>`_ is a light grid on top of the OAR
clusters of GRICAD (dahu, bigfoot, luke…). Instead of one ``oarsub`` per pair,
the whole set of pairs is submitted **once**, as a *campaign*: CiGri feeds the
jobs to the clusters at the rate they can take, mostly in **best-effort** mode
(idle cores, zero priority), and **re-submits** any job that a normal job
killed. A hundred pair correlations is exactly the bag-of-tasks it is built for.

How a campaign maps onto GMC
----------------------------
* One **job** per pair. Its parameter line is
  ``<pa_key> <pair script> <required output> …``. CiGri uses the first token as
  the job name (it must be unique) and runs, on a cluster node,
  ``export CIGRI_CAMPAIGN_ID=<id>; <exec_file> <parameter line>``
  (``lib/cigri-joblib.rb``).
* ``exec_file`` is one generic runner, :data:`RUNNER_NAME`, written next to the
  JDL. It runs the pair's own ``<pa_key>_CorrelationJob.sh`` (the same script
  the other modes run) and is **idempotent**, which best-effort requires: a job
  whose outputs exist exits 0 at once, and a job that finished without them
  exits non-zero so CiGri raises an event instead of counting a success.
* The JDL (JSON) carries the OAR options (resources, walltime, project), so the
  pair scripts of this mode have no ``#OAR`` header.

Assumption: the project lives on a file system mounted by every cluster of the
campaign (``/bettik`` or ``/summer`` at GRICAD), and the session was opened
with those same paths. This is the assumption the ``gricad`` OAR mode already
makes; nothing is staged through iRODS.

Things that will break if changed carelessly
--------------------------------------------
* **JDL booleans are strings.** CiGri compares ``test_mode == "true"``
  (``lib/cigri-iolib.rb``); a JSON ``true`` silently disables the test mode.
* **``gridsub`` exits 0 on failure.** It prints errors to stderr and
  ``CAMPAIGN_ID=<n>`` to stdout only on success, so the id is parsed and its
  absence raises; the exit status is not trusted.
* **No whitespace in a parameter token.** CiGri appends the line to the command
  and OAR splits it on blanks; a path with a space would shift every argument.

Submission host
---------------
``gridsub`` / ``gridstat`` exist only on the CiGri frontend (``killeen``,
``ciment-grid.u-ga.fr``). With ``host=None`` they are run here (GMC is running on
killeen); with ``host="killeen.ciment"`` they are run through ``ssh``, the JDL
being piped on stdin, so nothing has to be copied to killeen by hand. ``ssh``
runs with ``BatchMode=yes``: a missing key fails at once instead of waiting for
a password in a notebook cell.

Every function that touches the outside world takes ``run=`` (default:
:func:`subprocess.run`, looked up at call time), so tests never need a grid.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from geomulticorr._logging import logger

__all__ = [
    "CigriError",
    "CigriOptions",
    "CigriJob",
    "CigriCampaign",
    "CampaignStatus",
    "CIGRI_JOB_TYPES",
    "RUNNER_NAME",
    "DEFAULT_FRONTEND",
    "HOST_ENV_VAR",
    "resolve_host",
    "build_jdl",
    "runner_script_text",
    "write_campaign",
    "submit_campaign",
    "campaign_status",
    "campaign_events",
    "fix_campaign_events",
    "cancel_campaign",
    "parse_campaign_id",
    "write_campaign_record",
    "read_campaign_record",
]

#: Values of the JDL ``type`` field. Anything but best-effort may need the
#: platform admins' approval.
CIGRI_JOB_TYPES = ("best-effort", "normal", "semi-best-effort", "nightly")

#: File name of the generic job runner written next to the JDL.
RUNNER_NAME = "gmc_cigri_job.sh"

#: ssh alias of the GRICAD CiGri frontend (killeen, ciment-grid.u-ga.fr), as set
#: up in ``~/.ssh/config`` by the GRICAD documentation (``Host *.ciment``).
DEFAULT_FRONTEND = "killeen.ciment"

#: Overrides the frontend: an ssh alias, or ``"local"`` to run gridsub here.
HOST_ENV_VAR = "GMC_CIGRI_HOST"

_WALLTIME_RE = re.compile(r"^\d+:\d{1,2}:\d{1,2}$")
_CAMPAIGN_ID_RE = re.compile(r"CAMPAIGN_ID=(\d+)")
_FINISHED_STATES = ("terminated", "cancelled")

Runner = Callable[..., Any]


class CigriError(RuntimeError):
    """A CiGri command failed or answered something unexpected."""


# ─────────────────────────────────────────────────────────────────────────────
# Options and jobs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CigriOptions:
    """User-facing settings of a campaign.

    Attributes
    ----------
    clusters:
        Grid clusters to use: a list of names (``["dahu"]``) or a mapping of name
        to per-cluster JDL overrides (``{"dahu": {"max_jobs": 50}}``). List the
        available ones with ``gridclusters`` on killeen. Every cluster must
        mount the project's file system.
    job_type:
        JDL ``type``; see :data:`CIGRI_JOB_TYPES`. ``"best-effort"`` is the point
        of the grid. For ``dahu-oar3`` in ``"normal"`` mode the GRICAD docs ask for
        ``{"dahu-oar3": {"cluster_job_type": "devel"}}``.
    project:
        OAR project billed (``None`` → the ``cigri`` profile's, ``rgdyn``).
    host:
        ``ssh`` alias of the CiGri frontend, or ``None`` when GMC runs on it.
    test_mode:
        One job at a time per cluster, at high priority. Run a campaign in test
        mode first, check it with :func:`campaign_status`, cancel it, then submit
        the real one: a broken environment then costs one failed job, not a
        hundred.
    name:
        Campaign name (``None`` → chosen by the caller).
    walltime / resources:
        ``None`` → derived by the caller from its ``walltime`` / ``cores``. Keep
        the walltime close to the real job length: a best-effort job that runs
        longer is more likely to be killed and re-run from the start.
    max_jobs:
        Cap on the jobs CiGri keeps on one cluster for this campaign.
    prologue / epilogue:
        Shell lines run once per cluster before the first / after the last job.
    extra:
        Any other JDL key, written verbatim (``properties``, ``temporal_grouping``…).
    """

    clusters: Sequence[str] | Mapping[str, Mapping[str, Any]] = ("dahu",)
    job_type: str = "best-effort"
    project: str | None = None
    host: str | None = None
    test_mode: bool = False
    name: str | None = None
    walltime: str | None = None
    resources: str | None = None
    max_jobs: int | None = None
    prologue: Sequence[str] = ()
    epilogue: Sequence[str] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: "CigriOptions | Mapping[str, Any] | None") -> "CigriOptions":
        """Accept an instance, a plain mapping (CLI/YAML) or ``None``.

        Unknown keys raise ``TypeError`` naming them: a misspelt ``test_mod``
        must not silently submit a full campaign.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"cigri options must be a mapping, got {type(value)!r}")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise TypeError(
                f"unknown cigri option(s) {unknown}; accepted: {sorted(known)}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class CigriJob:
    """One job of a campaign: a unique name, the script to run, its outputs."""

    name: str
    script: Path
    outputs: tuple[Path, ...] = ()

    def param_line(self) -> str:
        """The JDL parameter line: ``<name> <script> <output> …``."""
        tokens = [self.name, str(self.script), *(str(o) for o in self.outputs)]
        for tok in tokens:
            if not tok or any(c.isspace() for c in tok):
                raise ValueError(
                    f"CiGri parameter tokens cannot be empty or contain whitespace "
                    f"(OAR splits the line on blanks): {tok!r} in job {self.name!r}"
                )
        return " ".join(tokens)


@dataclass
class CigriCampaign:
    """A campaign written to disk, ready to submit."""

    name: str
    workdir: Path
    jdl_path: Path
    runner_path: Path
    jdl: dict
    jobs: list[CigriJob]
    host: str | None = None
    campaign_id: int | None = None

    def to_record(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "host": self.host,
            "workdir": str(self.workdir),
            "jdl_path": str(self.jdl_path),
            "jdl": self.jdl,
            "jobs": [
                {
                    "name": j.name,
                    "script": str(j.script),
                    "outputs": [str(o) for o in j.outputs],
                }
                for j in self.jobs
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pure assembly
# ─────────────────────────────────────────────────────────────────────────────


def _jdl_bool(value: bool) -> str:
    return "true" if value else "false"


def build_jdl(
    name: str,
    exec_file: str | Path,
    jobs: Sequence[CigriJob],
    clusters: Sequence[str] | Mapping[str, Mapping[str, Any]],
    *,
    walltime: str,
    resources: str,
    job_type: str = "best-effort",
    project: str | None = None,
    test_mode: bool = False,
    exec_directory: str | Path | None = None,
    max_jobs: int | None = None,
    prologue: Sequence[str] = (),
    epilogue: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """Assemble a CiGri JDL (a JSON-ready dict). Pure; validates its inputs.

    Settings valid for every cluster are written globally (CiGri accepts any
    cluster setting there); per-cluster mappings in *clusters* override them.
    """
    if not name or any(c.isspace() for c in name):
        raise ValueError(f"campaign name must be one word, got {name!r}")
    if not jobs:
        raise ValueError("a CiGri campaign needs at least one job")
    names = [j.name for j in jobs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"CiGri job names (first parameter token) must be unique; duplicated: "
            f"{dupes[:3]}"
        )
    if job_type not in CIGRI_JOB_TYPES:
        raise ValueError(f"job_type must be one of {CIGRI_JOB_TYPES}, got {job_type!r}")
    if not _WALLTIME_RE.match(str(walltime)):
        raise ValueError(f"walltime must be HH:MM:SS, got {walltime!r}")
    if isinstance(clusters, str):
        clusters = [clusters]
    if isinstance(clusters, Mapping):
        cluster_block = {str(k): dict(v or {}) for k, v in clusters.items()}
    else:
        cluster_block = {str(c): {} for c in clusters}
    if not cluster_block:
        raise ValueError("at least one CiGri cluster is required (e.g. ['dahu'])")
    if max_jobs is not None and int(max_jobs) < 1:
        raise ValueError(f"max_jobs must be >= 1, got {max_jobs!r}")

    jdl: dict[str, Any] = {
        "name": name,
        "params": [j.param_line() for j in jobs],
        "exec_file": str(exec_file),
        "type": job_type,
        "walltime": str(walltime),
        "resources": str(resources),
        "test_mode": _jdl_bool(test_mode),
    }
    if exec_directory is not None:
        jdl["exec_directory"] = str(exec_directory)
    if project:
        jdl["project"] = project
    if max_jobs is not None:
        jdl["max_jobs"] = int(max_jobs)
    if prologue:
        jdl["prologue"] = list(prologue)
    if epilogue:
        jdl["epilogue"] = list(epilogue)
    for key, value in (extra or {}).items():
        if key in ("name", "params", "clusters"):
            raise ValueError(f"extra cannot override {key!r}")
        jdl[key] = _jdl_bool(value) if isinstance(value, bool) else value
    for section in cluster_block.values():
        for key, value in list(section.items()):
            if isinstance(value, bool):
                section[key] = _jdl_bool(value)
    jdl["clusters"] = cluster_block
    return jdl


def runner_script_text(env_lines: Sequence[str] = ()) -> str:
    """The generic, idempotent job runner (``exec_file`` of every campaign).

    Usage on a node: ``gmc_cigri_job.sh <name> <job script> [<output> …]``.
    *env_lines* are not needed for the pair scripts (they set up their own
    environment) and are written before the job for other uses.
    """
    env = "".join(line if line.endswith("\n") else f"{line}\n" for line in env_lines)
    return f"""#!/bin/bash
# GeoMultiCorr CiGri job runner, written by geomulticorr.scheduler.cigri.
# Usage: {RUNNER_NAME} <job_name> <job_script> [<required_output> ...]
#
# Idempotent, as best-effort requires: CiGri may kill this job and run it
# again, possibly on another cluster. A job whose outputs all exist exits 0
# at once; a job script that ends without them exits 3, so CiGri reports an
# event instead of counting a success.
#
# Output goes to <job_name>_cigri.log next to the job script (appended, so a
# re-submitted job keeps the history of its earlier attempts). CiGri's own
# OAR.cigri.<campaign>.<oar job>.std* files in the launch directory are removed
# on exit; only a job killed by best-effort leaves them behind.
set -u
if [ "$#" -lt 2 ]; then
    echo "usage: $0 <job_name> <job_script> [<required_output> ...]" >&2
    exit 64
fi
name="$1"
script="$2"
shift 2

launch_dir="$PWD"
cleanup() {{
    if [ -n "${{CIGRI_CAMPAIGN_ID:-}}" ] && [ -n "${{OAR_JOB_ID:-}}" ]; then
        rm -f "$launch_dir/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stdout" \
              "$launch_dir/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stderr"
    fi
}}
trap cleanup EXIT

job_dir="$(dirname "$script")"
if [ -d "$job_dir" ] && [ -w "$job_dir" ]; then
    exec >>"$job_dir/${{name}}_cigri.log" 2>&1
fi
echo "[gmc] $(date -Is) $name: start on $(hostname) (campaign ${{CIGRI_CAMPAIGN_ID:-?}}, OAR job ${{OAR_JOB_ID:-?}})"

if [ "$#" -gt 0 ]; then
    complete=1
    for f in "$@"; do [ -s "$f" ] || complete=0; done
    if [ "$complete" -eq 1 ]; then
        echo "[gmc] $name: outputs already present, nothing to do"
        exit 0
    fi
fi
{env}
cd "$job_dir" || {{ echo "[gmc] $name: cannot cd to $job_dir" >&2; exit 2; }}
bash "$script"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[gmc] $name: job script exited with $rc" >&2
    exit "$rc"
fi
for f in "$@"; do
    if [ ! -s "$f" ]; then
        echo "[gmc] $name: expected output missing: $f" >&2
        exit 3
    fi
done
echo "[gmc] $(date -Is) $name: done"
"""


def parse_campaign_id(stdout: str) -> int | None:
    """Return the id printed by ``gridsub`` (``CAMPAIGN_ID=<n>``), or ``None``."""
    match = _CAMPAIGN_ID_RE.search(stdout or "")
    return int(match.group(1)) if match else None


def resolve_host(
    env: Mapping[str, str] | None = None, which=shutil.which
) -> str | None:
    """Where the grid commands run: ``None`` (here) or an ssh alias.

    1. ``$GMC_CIGRI_HOST`` if set (``"local"`` means here);
    2. here, if ``gridsub`` is on ``PATH`` (GMC is running on killeen);
    3. otherwise ssh to :data:`DEFAULT_FRONTEND`.
    """
    env = os.environ if env is None else env
    value = (env.get(HOST_ENV_VAR) or "").strip()
    if value:
        return None if value.lower() == "local" else value
    if which("gridsub"):
        return None
    return DEFAULT_FRONTEND


# ─────────────────────────────────────────────────────────────────────────────
# Files
# ─────────────────────────────────────────────────────────────────────────────


def write_campaign(
    jobs: Sequence[CigriJob],
    workdir: str | Path,
    options: CigriOptions | Mapping[str, Any] | None,
    *,
    name: str,
    walltime: str,
    resources: str,
    project: str | None = None,
    env_lines: Sequence[str] = (),
) -> CigriCampaign:
    """Write the runner, the JDL and a readable parameter file into *workdir*.

    Files (flat, no subfolder): ``gmc_cigri_job.sh``, ``cigri_<name>.jdl`` and
    ``cigri_<name>_params.txt``; :func:`submit_campaign` adds
    ``cigri_campaign_<id>.json``.

    *name*, *walltime*, *resources* and *project* are the caller's defaults;
    values set in *options* win. CiGri's own ``OAR.cigri.*`` stdout/stderr files
    land in *workdir* too (it is the JDL ``exec_directory``).
    """
    opts = CigriOptions.coerce(options)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    campaign_name = opts.name or name

    runner = workdir / RUNNER_NAME
    runner.write_text(runner_script_text(env_lines))
    runner.chmod(runner.stat().st_mode | 0o755)

    jdl = build_jdl(
        campaign_name,
        runner,
        jobs,
        opts.clusters,
        walltime=opts.walltime or walltime,
        resources=opts.resources or resources,
        job_type=opts.job_type,
        project=opts.project or project,
        test_mode=opts.test_mode,
        exec_directory=workdir,
        max_jobs=opts.max_jobs,
        prologue=opts.prologue,
        epilogue=opts.epilogue,
        extra=opts.extra,
    )
    jdl_path = workdir / f"cigri_{campaign_name}.jdl"
    jdl_path.write_text(json.dumps(jdl, indent=2) + "\n")
    # Not read by CiGri (the JDL holds the params inline); kept for a human and
    # for `gridsub -c <id> -F` should jobs need adding by hand.
    (workdir / f"cigri_{campaign_name}_params.txt").write_text(
        "".join(f"{line}\n" for line in jdl["params"])
    )
    return CigriCampaign(
        name=campaign_name,
        workdir=workdir,
        jdl_path=jdl_path,
        runner_path=runner,
        jdl=jdl,
        jobs=list(jobs),
        host=opts.host,
    )


def write_campaign_record(campaign: CigriCampaign, path: str | Path | None = None):
    """Write the campaign trace JSON. A failure is logged, never raised."""
    if path is None:
        stem = (
            f"cigri_campaign_{campaign.campaign_id}"
            if campaign.campaign_id is not None
            else f"cigri_campaign_{campaign.name}"
        )
        path = campaign.workdir / f"{stem}.json"
    path = Path(path)
    record = campaign.to_record()
    record["written_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    try:
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(path)
        return path
    except OSError as exc:
        logger.warning(f"Could not write the CiGri campaign record {path}: {exc}")
        return None


def read_campaign_record(path: str | Path) -> dict:
    """Read a campaign trace written by :func:`write_campaign_record`."""
    return json.loads(Path(path).read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Grid commands
# ─────────────────────────────────────────────────────────────────────────────


def _command(args: Sequence[str], host: str | None) -> list[str]:
    """``args`` here, or the same command line run on *host* through ssh."""
    if host is None:
        return list(args)
    return ["ssh", "-o", "BatchMode=yes", host, shlex.join(args)]


def _call(args, host, run: Runner | None, input: str | None = None, what: str = ""):
    # Resolved at call time, so patching ``subprocess.run`` reaches it.
    run = run or subprocess.run
    cmd = _command(args, host) if isinstance(args, (list, tuple)) else args
    try:
        result = run(cmd, input=input, capture_output=True, text=True)
    except FileNotFoundError as exc:
        where = f"on {host}" if host else "here (pass host='killeen.ciment' to use ssh)"
        raise CigriError(
            f"{what or args[0]}: command not found {where}: {exc}"
        ) from exc
    return result


def submit_campaign(
    campaign: CigriCampaign,
    host: str | None = None,
    *,
    run: Runner | None = None,
    write_record: bool = True,
) -> int:
    """Submit *campaign* with ``gridsub``; return and store the campaign id.

    With a *host*, the JDL is piped through ``ssh`` into a temporary file on the
    frontend, so the local JDL path does not have to exist there.
    """
    host = host if host is not None else campaign.host
    if host is None:
        result = _call(
            ["gridsub", "-f", str(campaign.jdl_path)], None, run, what="gridsub"
        )
    else:
        remote = (
            'f=$(mktemp --suffix=.jdl) && cat > "$f" && gridsub -f "$f"; '
            'rc=$?; rm -f "$f"; exit $rc'
        )
        cmd = ["ssh", "-o", "BatchMode=yes", host, remote]
        result = _call(
            cmd, None, run, input=campaign.jdl_path.read_text(), what="gridsub"
        )
    campaign_id = parse_campaign_id(result.stdout)
    if campaign_id is None:
        # gridsub exits 0 on an admission error: the id is the only proof.
        detail = (result.stderr or "").strip() or (result.stdout or "").strip()
        raise CigriError(
            f"gridsub did not return a campaign id (exit {result.returncode}): "
            f"{detail or 'no output'}"
        )
    campaign.campaign_id = campaign_id
    campaign.host = host
    logger.info(
        f"CiGri campaign {campaign_id} '{campaign.name}' submitted: "
        f"{len(campaign.jobs)} job(s) on {', '.join(campaign.jdl['clusters'])}"
        f"{' [TEST MODE]' if campaign.jdl.get('test_mode') == 'true' else ''}"
    )
    if write_record:
        write_campaign_record(campaign)
    return campaign_id


@dataclass
class CampaignStatus:
    """The fields of ``gridstat -c <id> -d`` that matter, plus the raw JSON."""

    campaign_id: int
    name: str | None
    state: str | None
    finished_jobs: int
    total_jobs: int
    has_events: bool
    raw: dict

    @property
    def progress(self) -> float:
        return self.finished_jobs / self.total_jobs if self.total_jobs else 0.0

    @property
    def is_finished(self) -> bool:
        return self.state in _FINISHED_STATES

    def __str__(self) -> str:
        events = " (open events: run campaign_events)" if self.has_events else ""
        return (
            f"campaign {self.campaign_id} '{self.name}': {self.state}, "
            f"{self.finished_jobs}/{self.total_jobs} ({self.progress:.0%}){events}"
        )

    @classmethod
    def from_json(cls, data: Mapping[str, Any], campaign_id: int) -> "CampaignStatus":
        return cls(
            campaign_id=int(data.get("id", campaign_id)),
            name=data.get("name"),
            state=data.get("state"),
            finished_jobs=int(data.get("finished_jobs") or 0),
            total_jobs=int(data.get("total_jobs") or 0),
            has_events=bool(data.get("has_events")),
            raw=dict(data),
        )


def _json_call(args, host, run, what) -> Any:
    result = _call(args, host, run, what=what)
    text = (result.stdout or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        detail = (result.stderr or "").strip() or text
        raise CigriError(f"{what}: no JSON answer: {detail or 'no output'}") from None


def campaign_status(
    campaign_id: int, host: str | None = None, *, run: Runner | None = None
) -> CampaignStatus:
    """Status of a campaign, from ``gridstat -c <id> -d`` (JSON dump)."""
    data = _json_call(
        ["gridstat", "-c", str(int(campaign_id)), "-d"], host, run, "gridstat"
    )
    if not isinstance(data, Mapping) or data.get("status") == 404:
        raise CigriError(f"campaign {campaign_id} not found")
    return CampaignStatus.from_json(data, int(campaign_id))


def campaign_events(
    campaign_id: int, host: str | None = None, *, run: Runner | None = None
) -> list[dict]:
    """Open events of a campaign (failed jobs, blacklisted clusters…)."""
    data = _json_call(
        ["gridstat", "-c", str(int(campaign_id)), "-e", "-d"], host, run, "gridstat"
    )
    if isinstance(data, Mapping):
        return list(data.get("items") or [])
    return list(data or [])


def fix_campaign_events(
    campaign_id: int,
    host: str | None = None,
    *,
    resubmit: bool = True,
    run: Runner | None = None,
) -> None:
    """Close every open event of a campaign (``gridevents -c <id> -f [-r]``).

    With *resubmit* (the default) the jobs behind the events are run again;
    without it they are lost. Fix the cause first (expired OAR token, broken
    environment): closing an event also lifts the cluster's blacklist.
    """
    args = ["gridevents", "-c", str(int(campaign_id)), "-f"]
    if resubmit:
        args.append("-r")
    result = _call(args, host, run, what="gridevents")
    if result.returncode != 0 or (result.stderr or "").strip():
        raise CigriError(
            f"gridevents: {(result.stderr or '').strip() or result.returncode}"
        )


def cancel_campaign(
    campaign_id: int, host: str | None = None, *, run: Runner | None = None
) -> None:
    """Cancel a campaign and its queued and running jobs (``griddel -c <id>``)."""
    result = _call(["griddel", "-c", str(int(campaign_id))], host, run, what="griddel")
    if result.returncode != 0 or (result.stderr or "").strip():
        raise CigriError(
            f"griddel: {(result.stderr or '').strip() or result.returncode}"
        )
