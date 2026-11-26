#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# __init__.py
# creation date: 2026-11-19.
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
"""One module per stage, each ``run(ctx, section, record) -> None``.

Stages are thin: they turn a config section into the existing ``Session`` /
``TIOInversion`` calls and fill the stage record. No processing logic lives
here — anything a stage computes beyond bookkeeping belongs in the library.
"""

from __future__ import annotations

import importlib
from typing import Callable

from geomulticorr.cli.config import STAGE_ORDER


def stage_runner(name: str) -> Callable:
    if name not in STAGE_ORDER:
        raise ValueError(f"unknown stage {name!r}; choose from {STAGE_ORDER}")
    return importlib.import_module(f"geomulticorr.cli.stages.{name}").run
