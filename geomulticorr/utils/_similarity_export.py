#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _similarity_export.py
# creation date: 2026-10-30.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
# ---------------------------------------------------------------------------- #
"""Naming and writing the figures of ``Session.explore_similarity_pairs``.

A separate module from :mod:`geomulticorr.utils._pairs_export` (which validates
``view`` against the pairing explorer's views) that **shares its primitives**
rather than copying them, like ``_corrparams_export`` and ``_weights_export``.

File stems are a pure function of the selection parameters, so re-saving the
same selection refreshes the same files. ``html`` goes through plotly; ``png``,
``jpg``, ``pdf`` and ``svg`` through the matplotlib twins, never ``kaleido``.

Public API: :func:`similarity_figure_stem`, :func:`save_similarity_figure`.
"""

from __future__ import annotations

import pathlib

from typing import Sequence

import pandas as pd

from geomulticorr._logging import logger
from geomulticorr.utils._pairs_export import FIGURE_FORMATS, _slug, _unique_path


def _number(value) -> str:
    """``0.8`` → ``0.8``, ``2.0`` → ``2``: short and filesystem-safe."""
    return _slug(f"{float(value):g}")


def similarity_figure_stem(
    view: str,
    *,
    pz_name: str = "",
    image_type: str | None = "opt",
    similarity_metric: str = "score",
    min_similarity: float | None = None,
    per_image: int | None = None,
    cross_sensor_only: bool = False,
    max_dt_days: int | None = None,
    min_dt_days: int | None = None,
    sensor_filter: str | None = None,
    **_ignored,
) -> str:
    """Collision-free file stem encoding the selection a figure shows.

    Shape (segments omitted when not applicable)::

        {pzone|all}[_{type}]_selection_{view}_{metric}[{threshold}][_cross][_per{n}][_dt{min}-{max}][_{sensor}]

    e.g. ``PasDeLours_selection_network_score_cross_per2_dt0-730``. The keywords
    are those of ``Session._last_similarity_params``, so the stash splats in
    (``strategy`` and ``both_directions`` are accepted and ignored).
    """
    parts = [_slug(pz_name) or "all"]
    if image_type and str(image_type).lower() != "opt":
        parts.append(_slug(image_type))
    metric = _slug(similarity_metric)
    if min_similarity is not None:
        metric += _number(min_similarity)
    parts += ["selection", _slug(view), metric]
    if cross_sensor_only:
        parts.append("cross")
    if per_image:
        parts.append(f"per{int(per_image)}")
    if max_dt_days is not None or min_dt_days is not None:
        parts.append(f"dt{int(min_dt_days or 0)}-{int(max_dt_days or 0)}")
    if sensor_filter:
        parts.append(_slug(sensor_filter))
    return "_".join(p for p in parts if p)


def save_similarity_figure(
    sim: pd.DataFrame,
    selected: pd.DataFrame | None,
    out_dir: pathlib.Path | str,
    *,
    view: str = "matrix",
    formats: str | Sequence[str] = ("html", "png"),
    stem: str | None = None,
    title: str = "",
    metric: str = "score",
    both_directions: bool = True,
    pz_name: str = "",
    dpi: int = 300,
    plotlyjs: bool | str = True,
    overwrite: bool = True,
) -> dict[str, pathlib.Path]:
    """Write one similarity-explorer figure per requested format.

    :param sim: The similarity frame (the whole run).
    :param selected: The selected couples.
    :param out_dir: Destination directory; created if missing.
    :param view: ``"matrix"`` or ``"network"``.
    :param formats: Any of :data:`~geomulticorr.utils._pairs_export.FIGURE_FORMATS`.
    :param stem: File stem; ``{pz}_selection_{view}`` when omitted.
    :param title: Figure title, applied to both backends.
    :param metric: Column the matrix shows.
    :param both_directions: Draw the backward pair of each couple (network).
    :param pz_name: Pzone written into the pairs frame of the network view.
    :param dpi: Raster resolution for the matplotlib output.
    :param plotlyjs: Passed to ``write_html``; ``True`` inlines plotly.js.
    :param overwrite: Replace a file of the same name (default); ``False``
        appends ``_01``, ``_02``… instead.
    :returns: ``{format: path}`` for every file written.
    :raises ValueError: Unknown view or format, an empty similarity frame, or
        an empty selection for the network view (nothing to draw).
    """
    from geomulticorr.utils._similarity_plotly import (
        SIMILARITY_VIEWS,
        figure_similarity_view,
        selection_pairs_frame,
    )

    if view not in SIMILARITY_VIEWS:
        raise ValueError(
            f"Unknown view '{view}'. Valid options: {list(SIMILARITY_VIEWS)}."
        )
    if isinstance(formats, str):
        formats = (formats,)
    formats = tuple(dict.fromkeys(f.lower().lstrip(".") for f in formats))
    unknown = [f for f in formats if f not in FIGURE_FORMATS]
    if unknown:
        raise ValueError(
            f"Unknown format(s) {unknown}. Valid options: {list(FIGURE_FORMATS)}."
        )
    if not formats:
        raise ValueError("No output format requested.")
    if sim is None or len(sim) == 0:
        raise ValueError("Cannot save a similarity figure from an empty frame.")
    if view == "network" and (selected is None or len(selected) == 0):
        raise ValueError("No couple selected: the network view would be empty.")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{_slug(pz_name) or 'all'}_selection_{view}"

    def _target(suffix: str) -> pathlib.Path:
        return (
            (out_dir / f"{stem}.{suffix}")
            if overwrite
            else _unique_path(out_dir, stem, suffix)
        )

    written: dict[str, pathlib.Path] = {}
    if "html" in formats:
        path = _target("html")
        figure_similarity_view(
            view,
            sim,
            selected,
            metric=metric,
            both_directions=both_directions,
            title=title,
        ).write_html(str(path), include_plotlyjs=plotlyjs, full_html=True)
        written["html"] = path
        logger.info(f"Wrote {path}")

    image_formats = [f for f in formats if f != "html"]
    if image_formats:
        import matplotlib.pyplot as plt

        from geomulticorr.utils import gmc_functions as gmc_fn

        fig = None
        try:
            if view == "matrix":
                fig = gmc_fn.plot_similarity_selection(
                    sim, selected, metric=metric, fig_name=title or None
                )
            else:
                frame = selection_pairs_frame(selected, both_directions, pz=pz_name)
                fig, _ = gmc_fn.plot_pairs_network(
                    frame, color_by="sensor", fig_name=title or None
                )
            for fmt in image_formats:
                path = _target(fmt)
                # No bbox_inches="tight": the builders call tight_layout().
                fig.savefig(str(path), dpi=dpi)
                written[fmt] = path
                logger.info(f"Wrote {path}")
        finally:
            # Never leak the figure: one per click of the Save button otherwise.
            if fig is not None:
                plt.close(fig)
    return written
