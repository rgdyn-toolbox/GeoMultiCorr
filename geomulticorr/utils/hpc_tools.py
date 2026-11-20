#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# hpc_tools.py
# creation date: 2026-05-15.
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
"""Shared HPC utilities — kept as the historical import path.

The implementation moved to :mod:`geomulticorr.scheduler` (cluster profiles,
OAR, and the CiGri grid). Every name that lived here is re-exported unchanged,
so ``from geomulticorr.utils.hpc_tools import oar_header`` keeps working.

Supported cluster profiles
--------------------------
- ``None`` / ``"local"`` : run commands directly on the current machine
- ``"gricad"``           : GRICAD cluster (OAR + conda via bashrc)
- ``"isterre"``          : ISTerre cluster (OAR + modules + soft env)
- ``"cigri"``            : GRICAD CiGri grid, correlation only (one campaign
  for all pairs, best-effort, automatic re-submission); ``"cigri:dahu,luke"``
  names the grid clusters, ``"cigri-test[:…]"`` runs the campaign in test mode
"""

from __future__ import annotations

import subprocess  # noqa: F401  (tests patch ``hpc_tools.subprocess.run``)

from geomulticorr.scheduler.profiles import (  # noqa: F401
    _OAR_PROJECT,
    CLUSTER_CHOICES,
    CORRELATION_CLUSTER_CHOICES,
    GRID_CHOICES,
    asp_env_lines,
    cluster_base_env,
    get_profile,
    is_grid_cluster,
    oar_header,
    parse_cluster,
    validate_cluster,
)
from geomulticorr.scheduler.oar import oarsub_submit, run_local  # noqa: F401
