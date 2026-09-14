#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _correlation_plan.py
# creation date: 2026-09-10.
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
"""The correlation-plan JSON: which parameters apply to which pairs, and why.

Modelled on :mod:`geomulticorr.inversion._run_parameters`, and for the same
reason: correlation parameters chosen per group are a *recipe*, and a recipe
that exists only as a set of scattered ``<pa_key>_CorrParameters.txt`` files
cannot be reviewed, diffed or replayed.  The plan records the assumptions that
drove the derivation alongside its result, so months later the question "why is
this group's kernel 15 px?" has an answer in one file.

Written to ``<pzone>/image_correlation/correlation_plan_<name>.json``.

Two invariants inherited from the inversion trace:

- **A write failure is logged, never raised.** Losing a trace must not cost an
  otherwise fully prepared correlation run.
- ``written_utc`` makes the file **not byte-stable** by design: it is a log, so
  no test may assert byte-equality, and re-running the explorer always shows a
  diff.

The plan is round-trippable: feeding one to
:func:`~geomulticorr.utils._corrparams_frame.corrparams_frame_from_plan`
reproduces the frame the explorer derived, with no session open.

Public API:

- :func:`build_correlation_plan` — pure assembly, no I/O.
- :func:`write_correlation_plan` / :func:`read_correlation_plan`.
- :func:`plan_path` — the canonical location for a named plan.
- :func:`resolve_pair_params` — the ``scalar -> group -> override`` precedence.
"""
from __future__ import annotations

import json
import pathlib

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from geomulticorr._logging import logger

#: Assumptions a plan records.  Exactly the derivation inputs, so a reader can
#: reproduce the numbers without guessing what the explorer's sliders were on.
PLAN_ASSUMPTION_KEYS: tuple[str, ...] = (
    "velocity_m_yr", "footprint_m", "c_px", "k", "coreg_px", "margin_px",
    "flow_azimuth_deg", "strain_rate_per_yr", "corr_algorithm",
    "subpixel_mode", "tau_days", "dt_bin_ratio",
    "explicit_search", "explicit_search_above_days",
    "uniform_kernel", "uniform_kernel_px", "uniform_search", "uniform_search_box",
)

#: ASP parameters a group or an override may carry.  A strict subset of
#: :meth:`~geomulticorr.core.session.Session.prepare_pairs_correlation`'s
#: signature, so the resolved dict can be splatted straight into
#: ``build_correlation_params`` without filtering.
PLAN_PARAM_KEYS: tuple[str, ...] = (
    "corr_algorithm", "corr_kernel", "corr_search", "corr_xthreshold",
    "corr_seed_mode", "subpixel_mode", "subpixel_kernel", "prefilter_mode",
    "cost_mode",
)

#: Schema version, so a future reader can branch rather than misparse.
PLAN_SCHEMA_VERSION: int = 1


def _jsonable(value: Any) -> Any:
    """Coerce *value* to something ``json.dump`` accepts, eagerly.

    Same house style as the inversion trace: coerce at the point of computation
    rather than installing a custom encoder.  Tuples become lists, numpy
    scalars become Python scalars, and anything unrecognised degrades to its
    ``repr`` — a surprising type produces a readable string instead of an
    exception in the middle of a write.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(round(value, 6))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (ValueError, TypeError):
            pass
    return repr(value)


def plan_path(directory: pathlib.Path | str, name: str = "default") -> pathlib.Path:
    """Canonical path of a named plan inside *directory*.

    :param directory: The pzone's ``image_correlation`` folder.
    :param name: Plan name; ``"default"`` when unnamed.
    :returns: ``<directory>/correlation_plan_<name>.json``.
    """
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(name)) or "default"
    return pathlib.Path(directory) / f"correlation_plan_{safe}.json"


def build_correlation_plan(
    *,
    name: str = "default",
    pzone: str = "",
    assumptions: Mapping[str, Any] | None = None,
    groups: Mapping[str, Mapping[str, Any]] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    pairs: Sequence[Mapping[str, Any]] | None = None,
    relevant_params: Sequence[str] | None = None,
    dt_bin_edges: Sequence[float] | None = None,
) -> dict:
    """Assemble the plan document. Pure — no I/O, no filesystem access.

    :param name: Plan name, used in the filename.
    :param pzone: Pzone the plan covers.
    :param assumptions: The derivation inputs; keys outside
        :data:`PLAN_ASSUMPTION_KEYS` are kept but flagged by their absence from
        ``relevant_params``.
    :param groups: ``{group_key: {"params": {...}, "n_pairs": int, ...}}``.
    :param overrides: ``{pa_key: {...}}`` for individually pinned pairs.
    :param pairs: Per-pair facts — ``pa_key``, ``dt_days``, ``resolution_m``,
        ``sensor_i``, ``sensor_j``, ``pz``, ``group`` — enough for
        :func:`~geomulticorr.utils._corrparams_frame.corrparams_frame_from_plan`
        to replay the frame.
    :param relevant_params: Assumption names that actually changed this run, for
        a human reading the file.
    :param dt_bin_edges: The geometric Δt bin edges the grouping used.
    :returns: A JSON-safe dict.
    """
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "name": str(name),
        "pzone": str(pzone),
        "assumptions": {k: _jsonable(v) for k, v in dict(assumptions or {}).items()},
        "relevant_params": sorted(str(p) for p in (relevant_params or [])),
        "dt_bin_edges": _jsonable(list(dt_bin_edges or [])),
        "groups": {
            str(key): _jsonable(dict(block)) for key, block in dict(groups or {}).items()
        },
        "overrides": {
            str(key): _jsonable(dict(block)) for key, block in dict(overrides or {}).items()
        },
        "pairs": [_jsonable(dict(row)) for row in (pairs or [])],
        "not_recorded_here": [
            "the resolved ASP settings per pair — see <pa_key>_CorrParameters.txt "
            "in each pair folder, which this plan generates",
            "correlation results and quality statistics — see the per-pair stats JSON",
        ],
    }


def write_correlation_plan(
    path: pathlib.Path | str, document: Mapping[str, Any]
) -> pathlib.Path | None:
    """Write *document* to *path* as indented JSON.

    House style: ``indent=2``, no ``sort_keys`` (insertion order groups related
    fields), no custom encoder — values are coerced eagerly by
    :func:`build_correlation_plan`.

    **Never raises.** A trace is a convenience; failing to write one must not
    abort a correlation run that is otherwise ready to launch.

    :param path: Destination file.
    :param document: The plan mapping.
    :returns: The path written, or None when the write failed.
    """
    path = pathlib.Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(document), f, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"Could not write correlation plan to {path}: {exc}")
        return None
    logger.file(f"Correlation plan written: {path}")
    return path


def read_correlation_plan(path: pathlib.Path | str) -> dict:
    """Read a plan back from *path*.

    Unlike the writer this **does** raise: a caller asking for a specific plan
    that is missing or corrupt has nothing sensible to proceed with, and
    silently substituting defaults would apply the wrong parameters to every
    pair without saying so.

    :param path: The plan file.
    :returns: The plan mapping.
    :raises FileNotFoundError: When the file does not exist.
    :raises ValueError: When the file is not valid JSON or not a plan.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Correlation plan not found: {path}")
    try:
        with open(path) as f:
            document = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Correlation plan {path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or "schema_version" not in document:
        raise ValueError(f"{path} does not look like a correlation plan")
    return document


def resolve_pair_params(
    pa_key: str,
    group: str,
    scalars: Mapping[str, Any],
    *,
    groups: Mapping[str, Mapping[str, Any]] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict, str]:
    """Effective ASP parameters for one pair, and where they came from.

    Precedence is ``scalars -> group -> override``: the call's scalar arguments
    are the floor, a group's parameters refine them, and a per-pair override
    wins outright.  With neither a group nor an override the scalars pass
    through untouched, which is what keeps the extended
    ``prepare_pairs_correlation`` bit-identical to today for existing callers.

    Only keys in :data:`PLAN_PARAM_KEYS` are taken from the plan; anything else
    a plan happens to carry is ignored rather than splatted into a function that
    does not accept it.

    :param pa_key: The pair's key.
    :param group: The pair's group key.
    :param scalars: The caller's scalar arguments.
    :param groups: ``{group_key: {"params": {...}}}`` or ``{group_key: {...}}``.
    :param overrides: ``{pa_key: {...}}``.
    :returns: ``(params, source)`` with *source* one of ``"scalar"``,
        ``"group"``, ``"override"``.
    """
    params = {k: v for k, v in dict(scalars).items() if k in PLAN_PARAM_KEYS}
    source = "scalar"

    group_block = dict((groups or {}).get(group) or {})
    # Accept both the nested {"params": {...}} shape a plan writes and a flat
    # mapping, so a hand-written plan does not need the extra nesting.
    group_params = dict(group_block.get("params") or group_block)
    for key in PLAN_PARAM_KEYS:
        if key in group_params:
            params[key] = group_params[key]
            source = "group"

    override_params = dict((overrides or {}).get(pa_key) or {})
    for key in PLAN_PARAM_KEYS:
        if key in override_params:
            params[key] = override_params[key]
            source = "override"

    # JSON has no tuples: a replayed plan hands back lists, but ASP's writers
    # index positionally and the summary compares by value, so normalise here
    # rather than at every call site.
    for key in ("corr_kernel", "subpixel_kernel", "corr_search"):
        value = params.get(key)
        if isinstance(value, list):
            params[key] = tuple(value)

    return params, source


#: Effective-parameter name -> the ASP key ``build_correlation_params`` writes.
#: ``subpixel_mode`` is the one rename: the session and the plan call it that,
#: the ASP settings file calls it ``subpixel-mode``, and
#: ``build_correlation_params`` calls the argument
#: ``subpixel_refinement_mode``.
_PARAM_TO_ASP_KEY: dict[str, str] = {
    "corr_algorithm": "stereo-algorithm",
    "corr_kernel": "corr-kernel",
    "corr_search": "corr-search",
    "corr_xthreshold": "xcorr-threshold",
    "corr_seed_mode": "corr-seed-mode",
    "subpixel_mode": "subpixel-mode",
    "subpixel_kernel": "subpixel-kernel",
    "prefilter_mode": "prefilter-mode",
    "cost_mode": "cost-mode",
}


def parse_params_file(path: pathlib.Path | str) -> dict[str, str]:
    """Read an ASP ``key value`` settings file back into a dict.

    The inverse of
    :meth:`~geomulticorr.correlation.correlation.ASP.build_correlation_params`.
    Comment and blank lines are skipped; a bare flag (``save-left-right-...``)
    maps to ``""``.

    :param path: The ``<pa_key>_CorrParameters.txt`` file.
    :returns: ``{asp_key: value_string}``; empty when the file is unreadable.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            out[key.strip()] = value.strip()
    except OSError as exc:
        logger.warning(f"Could not read correlation parameters from {path}: {exc}")
        return {}
    return out


def _format_asp_value(value: Any) -> str:
    """Render *value* the way ``build_correlation_params`` writes it."""
    if isinstance(value, (tuple, list)):
        return " ".join(str(int(v)) for v in value)
    return str(value)


def params_differ_from_file(
    path: pathlib.Path | str,
    effective: Mapping[str, Any],
    *,
    save_disparity_difference: bool = True,
) -> bool:
    """Whether *effective* differs from the parameters already on disk.

    This is what makes a re-run with changed parameters actually take effect.
    Reusing a job script on mere existence made a parameter change a silent
    no-op — the settings file was never regenerated, so a freshly tuned kernel
    was discarded and the previous one re-run with nothing in the log.

    A missing or unreadable file counts as "different": regenerating is cheap
    and always correct, whereas assuming a match would re-run the wrong
    settings.

    The SGM/MGM override is applied to *effective* before comparing, because
    the file records post-override values — otherwise every ``asp_mgm`` run
    would look changed forever.

    :param path: The ``<pa_key>_CorrParameters.txt`` file.
    :param effective: Resolved parameters, as :func:`resolve_pair_params`
        returns them.
    :param save_disparity_difference: Whether the bare
        ``save-left-right-disparity-difference`` flag is expected.
    :returns: True when the file should be regenerated.
    """
    from geomulticorr.correlation.corr_params import sgm_kernel_override

    existing = parse_params_file(path)
    if not existing:
        return True

    wanted = dict(effective)
    kernel, subpixel, cost, _ = sgm_kernel_override(
        str(wanted.get("corr_algorithm", "asp_bm")),
        tuple(wanted.get("corr_kernel", (21, 21))),
        tuple(wanted.get("subpixel_kernel", (21, 21))),
        int(wanted.get("cost_mode", 2)),
    )
    wanted["corr_kernel"] = kernel
    wanted["subpixel_kernel"] = subpixel
    wanted["cost_mode"] = cost

    for name, asp_key in _PARAM_TO_ASP_KEY.items():
        if name not in wanted:
            continue
        value = wanted[name]
        if value is None:
            # corr_search=None means "let ASP auto-detect", which writes no line
            # at all; a line being present is therefore a real difference.
            if asp_key in existing:
                return True
            continue
        if existing.get(asp_key) != _format_asp_value(value):
            return True

    has_flag = "save-left-right-disparity-difference" in existing
    return has_flag != bool(save_disparity_difference)
