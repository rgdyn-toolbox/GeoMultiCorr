#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _weights_export.py
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
"""Writing TIO weight figures to disk, in both interactive and publication form.

The weights counterpart of :mod:`geomulticorr.utils._pairs_export`, and it shares
that module's primitives (:data:`~geomulticorr.utils._pairs_export.FIGURE_FORMATS`,
``_slug``, ``_unique_path``, ``_accepted_kwargs``, ``_filter_kwargs``) rather than
re-implementing them.

It is a *separate module* because ``save_pairs_figure`` validates its ``view``
against ``_pairs_plotly.VIEW_BUILDERS`` and looks up a per-view default figure
size — and those view keys are enumerated straight into the pairing explorer's
view dropdown, so a ``"weights"`` entry added there would show up as a pairing
view the explorer cannot draw.

One call writes the figure twice:

- **``.html``** — the plotly figure, self-contained by default so it opens with
  no network access.
- **``.png`` / ``.jpg`` / ``.pdf`` / ``.svg``** — the matplotlib twin, via
  :func:`geomulticorr.utils.gmc_functions.plot_inversion_weights`.  **Never**
  ``kaleido``, which is not a GMC dependency.

Public API:

- :func:`weights_figure_stem` — collision-free stem encoding the weighting setup.
- :func:`save_weights_figure` — render and write one file per requested format.
"""
from __future__ import annotations

import pathlib
from typing import Sequence

import pandas as pd

from geomulticorr._logging import logger
from geomulticorr.utils._pairs_export import (
    FIGURE_FORMATS,
    _accepted_kwargs,
    _filter_kwargs,
    _slug,
    _unique_path,
)
from geomulticorr.utils._weights_frame import relevant_weight_keys

#: Order the parameters appear in a stem, so the same settings always produce
#: byte-identical names regardless of dict ordering.
_STEM_KEY_ORDER: tuple[str, ...] = (
    "combine", "cc_gamma", "alpha", "beta", "gamma",
    "sharpness", "slope", "min_weight", "w_min", "invert", "dt_range",
)

_DEFAULT_FIGSIZE: tuple[float, float] = (9, 5)


def _stem_fragment(key: str, value) -> str:
    """One ``key=value`` fragment, in filesystem-safe form."""
    if key == "invert":
        return "invert" if value else ""
    if key == "dt_range":
        if value is None:
            return ""
        lo, hi = value
        return f"dt{int(lo)}-{int(hi)}"
    if isinstance(value, float):
        # trim the trailing zeros a plain str() leaves on 0.3333333333333333
        return f"{key}{_slug(f'{value:g}')}"
    return f"{key}{_slug(value)}"


def weights_figure_stem(
    *,
    inversion_name: str = "",
    pz_name: str = "",
    weight_mode: str = "",
    **params,
) -> str:
    """Build a file stem encoding the pzone, the inversion and the weighting setup.

    A **pure function of the parameters** — no timestamp, no pair count — so
    re-running a notebook refreshes the same files instead of accumulating
    copies, and a downstream step can reference a figure by path.  This is the
    same rule :func:`~geomulticorr.utils._pairs_export.pairs_figure_stem` follows.

    Only the parameters that *affect* the chosen mode appear, via
    :func:`~geomulticorr.utils._weights_frame.relevant_weight_keys`, so moving an
    irrelevant slider does not silently start a second file.  Note this makes
    ``dt_range`` load-bearing: it is in the relevant set for
    ``relative_temporal``/``sigmoid``, without which a ``dt_range=(300, 400)`` run
    and a ``dt_range=None`` run would collide on one name.

    Shape (segments omitted when not applicable)::

        {pzone|all}_{inversion}_weights_{mode}[_{param}{value}…]

    e.g. ``PasDeLours_PDL-spot_weights_quality_combinegeomean_cc_gamma1.5``.

    :param inversion_name: The inversion these weights belong to.
    :param pz_name: Pzone name; ``"all"`` when empty.
    :param weight_mode: The weighting mode.
    :param params: Any of the twelve weighting parameters; irrelevant ones are
        dropped rather than encoded.
    :returns: The stem, without an extension.
    """
    parts = [_slug(pz_name) or "all", _slug(inversion_name), "weights",
             _slug(weight_mode)]

    relevant = relevant_weight_keys(weight_mode, params.get("combine"))
    for key in _STEM_KEY_ORDER:
        if key not in relevant or key not in params:
            continue
        fragment = _stem_fragment(key, params[key])
        if fragment:
            parts.append(fragment)

    return "_".join(p for p in parts if p)


def save_weights_figure(
    frame: pd.DataFrame,
    out_dir: pathlib.Path | str,
    *,
    formats: str | Sequence[str] = ("html", "png"),
    stem: str | None = None,
    title: str = "",
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    plotlyjs: bool | str = True,
    view_kwargs: dict | None = None,
    overwrite: bool = True,
) -> dict[str, pathlib.Path]:
    """Write the weights figure once per requested format.

    :param frame: A weights frame (see :mod:`geomulticorr.utils._weights_frame`).
    :param out_dir: Destination directory; created if missing.
    :param formats: Any of :data:`~geomulticorr.utils._pairs_export.FIGURE_FORMATS`.
    :param stem: File stem; :func:`weights_figure_stem` supplies one when omitted.
    :param title: Figure title, applied to both backends.
    :param dpi: Raster resolution for the matplotlib output.
    :param figsize: Matplotlib figure size; ``(9, 5)`` when omitted.
    :param plotlyjs: Passed to ``write_html``.  ``True`` (the default) inlines
        plotly.js so the page works offline.
    :param view_kwargs: Figure options (``directions``, ``height``, ``alpha``,
        ``markersize``), filtered per backend so a keyword only one of them
        understands is dropped with a log line rather than raising.
    :param overwrite: Replace an existing file of the same name (the default —
        stems are deterministic, so re-running refreshes rather than accumulates).
        ``False`` appends ``_01``, ``_02``… instead.
    :returns: ``{format: path}`` for every file written.
    :raises ValueError: On an unknown format, or an empty frame.
    """
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
        raise ValueError("Cannot save a weights figure from an empty weights frame.")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or weights_figure_stem()
    view_kwargs = dict(view_kwargs or {})

    def _target(suffix: str) -> pathlib.Path:
        return (out_dir / f"{stem}.{suffix}") if overwrite else _unique_path(
            out_dir, stem, suffix)

    written: dict[str, pathlib.Path] = {}

    if "html" in formats:
        from geomulticorr.utils._weights_plotly import figure_weights

        kw = _filter_kwargs(view_kwargs, _accepted_kwargs(figure_weights),
                            "figure_weights")
        path = _target("html")
        figure_weights(frame, title=title, **kw).write_html(
            str(path), include_plotlyjs=plotlyjs, full_html=True
        )
        written["html"] = path
        logger.info(f"Wrote {path}")

    image_formats = [f for f in formats if f != "html"]
    if image_formats:
        import matplotlib.pyplot as plt

        from geomulticorr.utils import gmc_functions as gmc_fn

        builder = gmc_fn.plot_inversion_weights
        # the public builder forwards **style to its drawer, so the accepted set
        # is the union of both signatures
        accepted = _accepted_kwargs(builder, gmc_fn._draw_inversion_weights_on_ax)
        kw = _filter_kwargs(view_kwargs, accepted, "plot_inversion_weights")

        fig = None
        try:
            # plotly calls it `title`, matplotlib calls it `fig_name`
            fig, _ = builder(
                frame, figsize=figsize or _DEFAULT_FIGSIZE,
                fig_name=title or None, **kw,
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
