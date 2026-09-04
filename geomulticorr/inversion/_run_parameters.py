#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _run_parameters.py
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
"""The run-parameters trace written beside each direction's TIO inputs.

Per-pair weights already reach disk in four places (``liste_couple``, each pair's
stats JSON, and four geodatabase columns), but the **recipe** that produced them
did not: ``save_pair_weight``'s ``params`` field only ever held
``{"weight_mode": ...}``, so ``slope``, ``sharpness``, ``cc_gamma`` and the rest
were lost the moment the kernel died — along with the filter pipeline, the NMAD
threshold and the launch profile.

This module builds a self-contained JSON document per direction, written to::

    <inversion_dir>/inverse_EW/inverse_EW_parameters.json
    <inversion_dir>/inverse_NS/inverse_NS_parameters.json

``setup_directories`` already creates both folders, so nothing here calls
``mkdir``.

**Deliberately excluded**, because they are already on disk in the same folder
and duplicating them invites the two copies to disagree: the per-pair weight
vectors, the acquisition dates (``liste_image``) and the TIO solver settings
(``input_tio``, whose generator takes no arguments — there is nothing to record).

``written_utc`` makes the file **not byte-stable** by design: it is a log, so no
test may assert byte-equality, and re-running ``prepare_inversion`` will always
show a diff.
"""
from __future__ import annotations

import inspect
import json
import pathlib
from datetime import datetime, timezone

from geomulticorr._logging import logger

#: What the file deliberately does not carry, recorded in the file itself so a
#: reader is not left wondering whether it was an omission.
_NOT_RECORDED = (
    "per-pair weights → liste_couple + each pair's stats JSON; "
    "acquisition dates → liste_image; "
    "raster geometry → binary/File_info.rsc; "
    "solver settings → input_tio"
)

#: Constructor arguments that are data, not configuration — recorded as a type
#: tag rather than serialised.  ``StableAreaMask.stable_mask`` is typically a
#: GeoDataFrame or an ndarray, neither of which belongs in a parameters file.
_OPAQUE_ARGS: frozenset[str] = frozenset({"stable_mask"})


def _jsonable(value):
    """Coerce *value* to something ``json.dump`` accepts, eagerly.

    Follows the house style of coercing at the point of computation rather than
    installing a custom encoder: tuples become lists, numpy scalars become
    Python scalars, and anything else unrecognised becomes its ``repr``, so a
    surprising type degrades to a readable string instead of raising.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(round(value, 6))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    # numpy scalars and anything else that quacks like a number
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (ValueError, TypeError):
            pass
    return repr(value)


def describe_filter_pipeline(pipeline) -> list[dict] | None:
    """Capture a ``FilterPipeline`` (or a bare mask) as JSON-able dicts.

    Every mask in :mod:`geomulticorr.corrections.masks` stores its constructor
    arguments under identically-named instance attributes
    (``CCFilter.cc_threshold``, ``SlopeMask.max_slope``,
    ``OutlierFilter.threshold``…), so the arguments are recovered by walking the
    ``__init__`` signature rather than by maintaining a per-class table that
    would silently go stale when a mask gains a parameter.

    :param pipeline: A ``FilterPipeline``, a single ``BaseMask``, or ``None``.
    :returns: ``[{"filter": name, "params": {...}}, …]``, or ``None`` when no
        pipeline was used — ``None`` and ``[]`` mean different things, and the
        distinction is worth keeping in the file.
    """
    if pipeline is None:
        return None

    masks = getattr(pipeline, "masks", None)
    if masks is None:
        masks = [pipeline]  # a single BaseMask used on its own

    described: list[dict] = []
    for mask in masks:
        params: dict = {}
        try:
            signature = inspect.signature(type(mask).__init__)
            names = [n for n in signature.parameters if n != "self"]
        except (TypeError, ValueError):
            names = []
        for name in names:
            if not hasattr(mask, name):
                continue
            value = getattr(mask, name)
            params[name] = (
                f"<{type(value).__name__}>" if name in _OPAQUE_ARGS
                else _jsonable(value)
            )
        described.append({"filter": type(mask).__name__, "params": params})
    return described


def build_run_parameters(
    *,
    direction: str,
    inversion_name: str,
    pzone: str,
    inversion_dir: pathlib.Path | str,
    raster_shape: tuple[int, int] | None,
    pair_keys: list[str],
    n_images: int,
    weight_mode: str,
    combine: str | None,
    weight_source: str,
    weight_params: dict,
    relevant_params: list[str],
    weight_summary: dict,
    filter_pipeline=None,
    nmad_filter: dict | None = None,
    launch: dict | None = None,
) -> dict:
    """Assemble the run-parameters document for one direction.

    Pure: it takes everything it records and touches no filesystem, so it is
    testable without a project.

    The two per-direction files genuinely differ in *direction* and in
    *weight_summary* — statistics of that direction's own weight vector.  For
    date-based modes the two summaries coincide, which is itself informative.

    :param direction: ``"EW"`` or ``"NS"``.
    :param weight_params: The full **unpruned** twelve weighting parameters, so
        ``write_liste_couple(**params)`` reproduces the run from the file alone.
    :param relevant_params: Which of those actually affected this mode — for a
        human reading the file, not for replay.
    :returns: A JSON-able dict.
    """
    from geomulticorr import __version__

    width, height = raster_shape if raster_shape else (None, None)
    document = {
        "gmc_version": __version__,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "inversion_name": inversion_name,
        "pzone": pzone,
        "direction": direction,
        "inversion_dir": str(inversion_dir),
        "raster": {"width": width, "height": height},
        "pairs": {
            "count": len(pair_keys),
            "n_images": int(n_images),
            "keys": list(pair_keys),
        },
        "weights": {
            "mode": weight_mode,
            "combine": combine,
            "label": (f"{weight_mode}:{combine}"
                      if combine and weight_mode in ("quality", "quality_spatial")
                      else weight_mode),
            "source": weight_source,
            "params": _jsonable(weight_params),
            "relevant_params": sorted(relevant_params),
            "summary": _jsonable(weight_summary),
        },
        "filter_pipeline": describe_filter_pipeline(filter_pipeline),
        "nmad_filter": _jsonable(nmad_filter),
        "launch": _jsonable(launch),
        "not_recorded_here": _NOT_RECORDED,
    }
    return document


def write_run_parameters(path: pathlib.Path | str, document: dict) -> pathlib.Path:
    """Write *document* to *path* as indented JSON.

    House style: ``indent=2``, no ``sort_keys`` (the insertion order groups
    related fields), no custom encoder — values are coerced eagerly by
    :func:`build_run_parameters`.
    """
    path = pathlib.Path(path)
    with open(path, "w") as f:
        json.dump(document, f, indent=2)
    logger.file(f"Run parameters written: {path}")
    return path
