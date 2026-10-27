#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# _image_types.py
# creation date: 2026-09-30.
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
"""
The image-type table: which kinds of thumbs a pzone can hold, and where.

The image type is carried by the **folder**, never by the filename, so the
``<pz>_<date>_<sensor>[_<res>m]`` filename and the ``pa_key`` format stay untouched::

    <pz>/optical/<pz>_<date>_<sensor>.tif                    -> "opt"
    <pz>/elevation/<type>/<pz>_<date>_<sensor>.tif           -> "dem" | "hillshade" | ...
    <pz>/image_correlation/<pa_key>/                         -> "opt" pairs
    <pz>/image_correlation/<type>/<pa_key>/                  -> elevation pairs

Lives under ``utils`` with no GMC imports because ``thumb`` and ``pair`` need it
and cannot import ``session`` (which imports them). ``session`` re-exports it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_TYPE_OPTICAL = "opt"
#: Pzone subfolder holding every DEM-derived thumb type.
ELEVATION_DIRNAME = "elevation"
#: Thumb types stored under ``<pz>/elevation/<type>/``.
ELEVATION_IMAGE_TYPES = ("dem", "hillshade", "mdhillshade", "slope")
IMAGE_TYPES = (IMAGE_TYPE_OPTICAL, *ELEVATION_IMAGE_TYPES)

_OPTICAL_DIRNAMES = ("optical", "opticals")


def normalise_image_type(image_type: str | None) -> str:
    """Lower-case and validate *image_type*; ``None`` means optical."""
    t = IMAGE_TYPE_OPTICAL if image_type is None else str(image_type).strip().lower()
    if t not in IMAGE_TYPES:
        raise ValueError(
            f"Unknown image_type {image_type!r}; expected one of {IMAGE_TYPES}"
        )
    return t


def image_type_of_thumb_path(path) -> str:
    """Image type of a thumb, read from its folder.

    ``<pz>/elevation/<type>/x.tif`` gives ``<type>``; everything else —
    ``optical/``, legacy ``opticals/`` or an unexpected folder — gives ``"opt"``,
    which is what every thumb was before image types existed.
    """
    p = Path(path)
    parent = p.parent
    if parent.name in ELEVATION_IMAGE_TYPES and parent.parent.name == ELEVATION_DIRNAME:
        return parent.name
    return IMAGE_TYPE_OPTICAL


def pzone_root_of_thumb_path(path) -> Path:
    """Pzone folder containing a thumb (one level deeper for elevation types)."""
    p = Path(path)
    if image_type_of_thumb_path(p) == IMAGE_TYPE_OPTICAL:
        return p.parent.parent
    return p.parent.parent.parent


def pair_subdir(image_type: str) -> str:
    """Folder under ``image_correlation/`` for *image_type* (``""`` for optical)."""
    t = normalise_image_type(image_type)
    return "" if t == IMAGE_TYPE_OPTICAL else t


def type_column(df: pd.DataFrame, column: str) -> pd.Series:
    """``df[column]`` with rows written before the column existed read as ``"opt"``."""
    if column not in df.columns:
        return pd.Series(IMAGE_TYPE_OPTICAL, index=df.index, dtype=object)
    return df[column].fillna(IMAGE_TYPE_OPTICAL).astype(str)
