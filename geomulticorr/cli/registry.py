#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# registry.py
# creation date: 2026-10-27.
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
"""YAML step specs ↔ the corrections package's mask and correction objects.

A step is written either as a bare class name or as a one-key mapping of the
class name to its constructor arguments::

    masks:       [{CCFilter: {cc_threshold: 0.5}}, {StableAreaMask: auto}]
    corrections: [MedianCentering, {DirectionalBiasCorrection: {angle: null}}]

The tables are read from :mod:`geomulticorr.corrections`' ``__all__``, not from
``__subclasses__()``: ``BaseMask`` itself subclasses ``BaseCorrection``, so a
subclass walk would offer every mask as a correction too.

The inverse direction, :func:`describe_steps`, reuses
:func:`~geomulticorr.inversion._run_parameters.describe_filter_pipeline` (which
recovers constructor arguments by signature walk), so a pipeline built here
describes itself back to the spec it came from.
"""

from __future__ import annotations

from typing import Any, Callable

from geomulticorr import corrections as _corr
from geomulticorr.corrections import BaseCorrection, BaseMask

#: The argument value that makes ``StableAreaMask`` use the pzone's stored
#: moving-area outlines (``Session.get_moving_areas``).
AUTO = "auto"


def _classes(base: type, exclude: type | None = None) -> dict[str, type]:
    out = {}
    for name in _corr.__all__:
        obj = getattr(_corr, name)
        if not isinstance(obj, type) or not issubclass(obj, base):
            continue
        if obj.__name__.startswith("Base") or obj.__name__.endswith("Pipeline"):
            continue
        if exclude is not None and issubclass(obj, exclude):
            continue
        out[name] = obj
    return out


MASKS: dict[str, type] = _classes(BaseMask)
CORRECTIONS: dict[str, type] = _classes(BaseCorrection, exclude=BaseMask)


def _split(spec) -> tuple[str, Any]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict) and len(spec) == 1:
        ((name, args),) = spec.items()
        return name, {} if args is None else args
    raise ValueError(
        f"a step is a class name or a one-key mapping {{Name: {{args}}}}, got {spec!r}"
    )


def _build(spec, table: dict[str, type], kind: str, resolve_auto: Callable | None):
    name, args = _split(spec)
    if name not in table:
        raise ValueError(f"unknown {kind} {name!r}; choose from {sorted(table)}")
    cls = table[name]
    if args == AUTO:
        if resolve_auto is None:
            raise ValueError(f"{name}: 'auto' needs a session to resolve it")
        return cls(resolve_auto(name))
    if not isinstance(args, dict):
        # a single positional argument, e.g. {OutlierFilter: [-10, 10]}
        args = {_first_param(cls): args}
    try:
        return cls(**args)
    except TypeError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def _first_param(cls: type) -> str:
    import inspect

    params = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    if not params:
        raise ValueError(f"{cls.__name__} takes no arguments")
    return params[0]


def build_filter_pipeline(specs: list, resolve_auto: Callable | None = None):
    """``FilterPipeline`` from mask specs, or ``None`` for an empty list.

    :param resolve_auto: ``fn(class_name) -> argument`` for ``{Name: auto}``;
        the stages pass one that returns the pzone's moving areas.
    """
    if not specs:
        return None
    steps = [_build(s, MASKS, "mask", resolve_auto) for s in specs]
    return _chain(steps)


def build_correction_pipeline(specs: list):
    """``CorrectionPipeline`` from correction specs, or ``None`` for an empty list."""
    if not specs:
        return None
    steps = [_build(s, CORRECTIONS, "correction", None) for s in specs]
    return _chain(steps)


def _chain(steps):
    if len(steps) == 1:
        # `+` is what builds a pipeline; a single step still needs wrapping so
        # every consumer receives the same type.
        from geomulticorr.corrections import CorrectionPipeline, FilterPipeline

        wrap = FilterPipeline if isinstance(steps[0], BaseMask) else CorrectionPipeline
        return wrap([steps[0]])
    out = steps[0]
    for step in steps[1:]:
        out = out + step
    return out


def describe_steps(pipeline) -> list:
    """Spec list recovered from a pipeline (masks or corrections)."""
    from geomulticorr.inversion._run_parameters import describe_filter_pipeline

    if pipeline is None:
        return []
    steps = getattr(pipeline, "masks", None)
    if steps is None:
        steps = getattr(pipeline, "steps", [pipeline])
    out = []
    for item in describe_filter_pipeline(type("P", (), {"masks": steps})()) or []:
        out.append(
            {item["filter"]: item["params"]} if item["params"] else item["filter"]
        )
    return out
