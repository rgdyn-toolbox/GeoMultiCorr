#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _pairs_export.py
# creation date: 2026-07-27.
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
"""Writing pair figures to disk, in both interactive and publication form.

A saved view comes out twice from a single call:

- **`.html`** — the plotly figure, self-contained by default so it opens with no
  network access.  Requires no extra dependency (in particular **not** ``kaleido``,
  which plotly would need for static image export and which is not a GMC dependency).
- **`.png` / `.pdf` / `.svg`** — the matplotlib twin of the same view, via
  :data:`geomulticorr.utils.gmc_functions.PAIRS_MPL_BUILDERS`.

The entry point takes a **pairs frame**, not a ``Session``: the pairing-strategy
explorer works on *candidate* pairs that are never written to the geodatabase, so
anything that re-read pairs from disk could not save what is on screen.
:meth:`~geomulticorr.core.session.Session.save_pairs_figure` is a thin wrapper that
supplies the output directory and a default file stem.

Public API:

- :data:`FIGURE_FORMATS` — what can be written.
- :func:`pairs_figure_stem` — collision-free file stem encoding the pairing setup.
- :func:`save_pairs_figure` — render and write one file per requested format.
"""
from __future__ import annotations

import inspect
import pathlib
import re
from typing import Sequence

import pandas as pd

from geomulticorr._logging import logger

#: Formats this module can write.  ``html`` goes through plotly, the rest through
#: the matplotlib twins — no ``kaleido`` anywhere.  ``jpg`` needs Pillow, which
#: matplotlib already depends on.
FIGURE_FORMATS: tuple[str, ...] = ("html", "png", "jpg", "pdf", "svg")

#: Sensible figure size per view, used when the caller does not pass one.
_DEFAULT_FIGSIZE: dict[str, tuple[float, float]] = {
    "baseline": (10, 5),
    "chord": (10, 10),
    "network": (14, 5),
    "dt_hist": (8, 5),
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value) -> str:
    """Filesystem-safe form of *value* (empty string for ``None``)."""
    if value is None:
        return ""
    return _UNSAFE.sub("-", str(value)).strip("-")


def pairs_figure_stem(
    view: str,
    *,
    strategy: str = "",
    pz_name: str = "",
    max_step: int | None = None,
    max_dt_days: int | None = None,
    min_dt_days: int | None = None,
    sensor_filter: str | None = None,
) -> str:
    """Build a file stem that encodes the pzone, the view and the pairing setup.

    The name is a **pure function of the parameters** — no timestamp, no pair
    count — so re-running a notebook or a sweep refreshes the same files instead
    of accumulating copies, and downstream steps can reference a figure by path.

    The keyword names are deliberately **exactly** the six keys of
    ``Session._last_pairs_params`` (plus *view*), so a caller can splat that dict
    in.  That dict is also splatted into
    :meth:`~geomulticorr.core.session.Session.update_pairs`, which accepts those
    six and nothing else — hence nothing may ever be added to it.

    Shape (segments omitted when not applicable)::

        {pzone|all}_{view}_{strategy}[_maxstep{n}][_dt{min}-{max}][_{sensor}]

    e.g. ``Chimborazo_chord_redundancy_maxstep17_dt60-800``.

    :returns: The stem, without an extension.
    """
    parts = [_slug(pz_name) or "all", _slug(view), _slug(strategy)]
    if max_step is not None:
        parts.append(f"maxstep{int(max_step)}")
    if max_dt_days is not None or min_dt_days is not None:
        parts.append(f"dt{int(min_dt_days or 0)}-{int(max_dt_days or 0)}")
    if sensor_filter:
        parts.append(_slug(sensor_filter))

    return "_".join(p for p in parts if p)


def _unique_path(out_dir: pathlib.Path, stem: str, suffix: str) -> pathlib.Path:
    """``out_dir/stem.suffix``, with a ``_01``, ``_02``… guard against collisions.

    Only used when ``overwrite=False``; the default naming is deterministic and
    replaces in place.
    """
    path = out_dir / f"{stem}.{suffix}"
    counter = 1
    while path.exists():
        path = out_dir / f"{stem}_{counter:02d}.{suffix}"
        counter += 1
    return path


def _accepted_kwargs(func, extra_of=None) -> set[str]:
    """Parameter names *func* accepts, optionally unioned with another callable's."""
    names = set(inspect.signature(func).parameters)
    if extra_of is not None:
        names |= set(inspect.signature(extra_of).parameters)
    return names


def _filter_kwargs(kwargs: dict, accepted: set[str], target: str) -> dict:
    """Keep only what *target* accepts, logging what was dropped."""
    kept = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        logger.info(f"save_pairs_figure: ignoring {dropped} — not accepted by {target}.")
    return kept


def save_pairs_figure(
    frame: pd.DataFrame,
    out_dir: pathlib.Path | str,
    *,
    view: str = "baseline",
    formats: str | Sequence[str] = ("html", "png"),
    stem: str | None = None,
    title: str = "",
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    plotlyjs: bool | str = True,
    view_kwargs: dict | None = None,
    overwrite: bool = True,
) -> dict[str, pathlib.Path]:
    """Write one pair figure per requested format.

    :param frame: A pairs frame (see :mod:`geomulticorr.utils._pairs_frame`).
    :param out_dir: Destination directory; created if missing.
    :param view: ``"baseline"`` | ``"chord"`` | ``"network"`` | ``"dt_hist"``.
    :param formats: Any of :data:`FIGURE_FORMATS`.
    :param stem: File stem; :func:`pairs_figure_stem` supplies one when omitted.
    :param title: Figure title, applied to both backends.
    :param dpi: Raster resolution for the matplotlib output.
    :param figsize: Matplotlib figure size; a per-view default is used when omitted.
    :param plotlyjs: Passed to ``write_html``.  ``True`` (the default) inlines
        plotly.js so the page works offline — the same offline-first reasoning that
        rules ``go.FigureWidget`` out of the explorer.  ``"cdn"`` gives a ~3 MB
        smaller file that needs a network connection to render.
    :param view_kwargs: View options (``dedupe``, ``show_arrows``, ``color_by``,
        ``mirror_direction``, ``nbins``…), filtered per backend so a keyword that
        only one of them understands is dropped with a log line rather than raising.
    :param overwrite: Replace an existing file of the same name (the default —
        stems are deterministic, so re-running refreshes rather than accumulates).
        ``False`` appends ``_01``, ``_02``… instead.
    :returns: ``{format: path}`` for every file written.
    :raises ValueError: On an unknown view or format, or an empty frame.
    """
    from geomulticorr.utils import _pairs_plotly as gmc_plotly
    from geomulticorr.utils.gmc_functions import PAIRS_MPL_BUILDERS

    if view not in gmc_plotly.VIEW_BUILDERS:
        raise ValueError(
            f"Unknown view '{view}'. Valid options: {sorted(gmc_plotly.VIEW_BUILDERS)}."
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
        raise ValueError("Cannot save a pair figure from an empty pairs frame.")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or pairs_figure_stem(view)
    view_kwargs = dict(view_kwargs or {})

    def _target(suffix: str) -> pathlib.Path:
        return (out_dir / f"{stem}.{suffix}") if overwrite else _unique_path(
            out_dir, stem, suffix)

    written: dict[str, pathlib.Path] = {}

    if "html" in formats:
        builder = gmc_plotly.VIEW_BUILDERS[view]
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

        builder = PAIRS_MPL_BUILDERS[view]
        # the public builder forwards **style to its _draw_*_on_ax drawer, so the
        # accepted set is the union of both signatures
        drawer = getattr(gmc_fn, f"_draw_pairs_{view}_on_ax", None)
        accepted = _accepted_kwargs(builder, drawer)
        kw = _filter_kwargs(view_kwargs, accepted, f"plot_pairs_{view}")

        fig = None
        try:
            # plotly calls it `title`, matplotlib calls it `fig_name`
            fig, _ = builder(
                frame, figsize=figsize or _DEFAULT_FIGSIZE[view],
                fig_name=title or None, **kw,
            )
            for fmt in image_formats:
                path = _target(fmt)
                fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
                written[fmt] = path
                logger.info(f"Wrote {path}")
        finally:
            # Never leak the figure: the explorer's save button would otherwise
            # accumulate one per click and dump strays under %matplotlib inline.
            if fig is not None:
                plt.close(fig)

    return written
