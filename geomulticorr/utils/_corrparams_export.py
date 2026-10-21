#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _corrparams_export.py
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
"""Naming and writing correlation-parameter figures.

Deliberately a separate module from :mod:`geomulticorr.utils._pairs_export` —
that one validates ``view`` against the *pairing* explorer's dispatch table —
but it **shares its primitives** rather than copying them, exactly as
:mod:`geomulticorr.utils._weights_export` does.

File stems are a **pure function of the parameters**: no timestamp, no pair
count, so re-running a notebook or a sweep refreshes the same file set rather
than accumulating copies, and a downstream step can reference a figure by path.
The stem is pruned by
:func:`~geomulticorr.utils._corrparams_frame.relevant_corr_keys`, so moving a
slider that changed nothing about the run does not silently start a second file.

``kaleido`` is **not** a dependency: static output goes through the matplotlib
twins in :mod:`geomulticorr.utils.gmc_functions`, never ``fig.write_image``.

Public API: :func:`corrparams_figure_stem`, :func:`save_corrparams_figure`.
"""
from __future__ import annotations

import pathlib

from typing import Any, Mapping, Sequence

import pandas as pd

from geomulticorr._logging import logger
from geomulticorr.utils._corrparams_frame import relevant_corr_keys
from geomulticorr.utils._pairs_export import (
    FIGURE_FORMATS,
    _accepted_kwargs,
    _filter_kwargs,
    _slug,
    _unique_path,
)

#: Order the parameters appear in a stem, so the same settings always produce
#: byte-identical names regardless of dict ordering.
_STEM_KEY_ORDER: tuple[str, ...] = (
    "velocity_m_yr", "footprint_m", "c_px", "k",
    "coreg_px", "margin_px", "flow_azimuth_deg", "strain_rate_per_yr",
    "subpixel_mode", "dt_bin_ratio",
    "explicit_search_above_days", "uniform_kernel_px", "uniform_search_box",
)

#: Short forms, so a stem carrying six parameters stays a filename rather than a
#: sentence.
_STEM_ABBREV: dict[str, str] = {
    "velocity_m_yr": "v",
    "footprint_m": "L",
    "c_px": "c",
    "k": "k",
    "coreg_px": "coreg",
    "margin_px": "margin",
    "flow_azimuth_deg": "az",
    "strain_rate_per_yr": "strain",
    "subpixel_mode": "sub",
    "dt_bin_ratio": "bin",
    "explicit_search_above_days": "srchabove",
    "uniform_kernel_px": "fixK",
    "uniform_search_box": "fixS",
}

_DEFAULT_FIGSIZE: dict[str, tuple[float, float]] = {
    "design_map": (9, 5.5),
    "kernel_search": (9, 5),
    "snr": (9, 5),
    "cost": (9, 5),
}


def _stem_fragment(key: str, value: Any) -> str:
    """One ``key=value`` fragment, in filesystem-safe form."""
    if value is None:
        return ""
    label = _STEM_ABBREV.get(key, key)
    if isinstance(value, float):
        # trim the trailing zeros a plain str() leaves on 0.15000000000000002
        return f"{label}{_slug(f'{value:g}')}"
    return f"{label}{_slug(value)}"


def corrparams_figure_stem(
    view: str,
    *,
    pz_name: str = "",
    corr_algorithm: str = "asp_bm",
    **params: Any,
) -> str:
    """Collision-free file stem encoding the derivation that produced a figure.

    Shape: ``{pzone|all}_{view}_{algorithm}[_{param}{value}…]``, e.g.
    ``Chimborazo_design_map_asp_bm_v5_L60_c0.15_k3``.

    Only parameters that actually changed the derivation appear, via
    :func:`~geomulticorr.utils._corrparams_frame.relevant_corr_keys` — so an
    SGM run carries no footprint (the kernel is forced to 9x9 regardless) and a
    run with no flow azimuth carries no ``azNone``.

    :param view: One of the keys of
        :data:`~geomulticorr.utils._corrparams_plotly.VIEW_BUILDERS`.
    :param pz_name: Pzone name; ``"all"`` when omitted.
    :param corr_algorithm: ASP ``stereo-algorithm`` value.
    :param params: The derivation assumptions.
    :returns: The stem, without an extension.
    """
    relevant = relevant_corr_keys(
        corr_algorithm,
        flow_azimuth_deg=params.get("flow_azimuth_deg"),
        strain_rate_per_yr=params.get("strain_rate_per_yr"),
    )
    parts = [_slug(pz_name) if pz_name else "all", _slug(view), _slug(corr_algorithm)]
    for key in _STEM_KEY_ORDER:
        if key not in relevant or key not in params:
            continue
        fragment = _stem_fragment(key, params[key])
        if fragment:
            parts.append(fragment)
    return "_".join(p for p in parts if p)


def save_corrparams_figure(
    frame: pd.DataFrame,
    out_dir: pathlib.Path | str,
    *,
    view: str = "design_map",
    formats: str | Sequence[str] = ("html", "png"),
    stem: str | None = None,
    title: str = "",
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    plotlyjs: bool | str = True,
    view_kwargs: Mapping[str, Any] | None = None,
    overwrite: bool = True,
) -> dict[str, pathlib.Path]:
    """Write one correlation-parameters figure per requested format.

    :param frame: A correlation-parameters frame.
    :param out_dir: Destination directory; created if missing.
    :param view: Which view to render.
    :param formats: Any of :data:`~geomulticorr.utils._pairs_export.FIGURE_FORMATS`.
    :param stem: File stem; :func:`corrparams_figure_stem` supplies one when
        omitted.
    :param title: Figure title, applied to both backends.
    :param dpi: Raster resolution for the matplotlib output.
    :param figsize: Matplotlib figure size; a per-view default when omitted.
    :param plotlyjs: Passed to ``write_html``. ``True`` (the default) inlines
        plotly.js so the page works offline.
    :param view_kwargs: Figure options, filtered per backend so a keyword only
        one of them understands is dropped with a log line rather than raising.
    :param overwrite: Replace an existing file of the same name (the default —
        stems are deterministic, so re-running refreshes rather than
        accumulates). ``False`` appends ``_01``, ``_02``… instead.
    :returns: ``{format: path}`` for every file written.
    :raises ValueError: On an unknown view or format, or an empty frame.
    """
    from geomulticorr.utils._corrparams_plotly import VIEW_BUILDERS

    if view not in VIEW_BUILDERS:
        raise ValueError(
            f"Unknown view '{view}'. Valid options: {list(VIEW_BUILDERS)}."
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
    if frame is None or len(frame) == 0:
        raise ValueError(
            "Cannot save a correlation-parameters figure from an empty frame."
        )

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or corrparams_figure_stem(view)
    view_kwargs = dict(view_kwargs or {})

    def _target(suffix: str) -> pathlib.Path:
        return (out_dir / f"{stem}.{suffix}") if overwrite else _unique_path(
            out_dir, stem, suffix
        )

    written: dict[str, pathlib.Path] = {}

    if "html" in formats:
        builder = VIEW_BUILDERS[view]
        kw = _filter_kwargs(view_kwargs, _accepted_kwargs(builder), f"figure_{view}")
        path = _target("html")
        builder(frame, title=title, **kw).write_html(
            str(path), include_plotlyjs=plotlyjs, full_html=True
        )
        written["html"] = path
        logger.info(f"Wrote {path}")

    image_formats = [f for f in formats if f != "html"]
    if image_formats:
        import matplotlib.pyplot as plt

        from geomulticorr.utils import gmc_functions as gmc_fn

        builder = gmc_fn.CORRPARAMS_MPL_BUILDERS[view]
        drawer = getattr(gmc_fn, f"_draw_correlation_{view}_on_ax", None)
        # the public builder forwards **style to its drawer, so the accepted set
        # is the union of both signatures
        accepted = _accepted_kwargs(builder, drawer)
        kw = _filter_kwargs(view_kwargs, accepted, builder.__name__)

        fig = None
        try:
            # plotly calls it `title`, matplotlib calls it `fig_name`
            fig, _ = builder(
                frame,
                figsize=figsize or _DEFAULT_FIGSIZE.get(view, (9, 5)),
                fig_name=title or None,
                **kw,
            )
            for fmt in image_formats:
                path = _target(fmt)
                # No bbox_inches="tight": it forces a second full renderer pass,
                # and the builder already called fig.tight_layout().
                fig.savefig(str(path), dpi=dpi)
                written[fmt] = path
                logger.info(f"Wrote {path}")
        finally:
            # Never leak the figure: the explorer's save button would otherwise
            # accumulate one per click and dump strays under %matplotlib inline.
            if fig is not None:
                plt.close(fig)

    return written
