#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# __init__.py
# creation date: 2026-11-20.
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
"""Where GMC jobs run: local, OAR clusters (GRICAD, ISTerre) and the CiGri grid.

* :mod:`~geomulticorr.scheduler.profiles` — the cluster-profile table,
  ``#OAR`` headers and job-script environment lines.
* :mod:`~geomulticorr.scheduler.oar` — ``oarsub -S`` and local execution.
* :mod:`~geomulticorr.scheduler.cigri` — CiGri campaigns: JDL, idempotent
  runner, ``gridsub`` / ``gridstat`` / ``gridevents`` / ``griddel``.

No module here imports ``geomulticorr.core``; :mod:`geomulticorr.utils.hpc_tools`
re-exports the historical names.
"""

from geomulticorr.scheduler.profiles import (
    CLUSTER_CHOICES,
    CORRELATION_CLUSTER_CHOICES,
    GRID_CHOICES,
    PROFILES,
    DEFAULT_GRID_CLUSTERS,
    ClusterProfile,
    ClusterSpec,
    asp_env_lines,
    is_grid_cluster,
    parse_cluster,
    cluster_base_env,
    get_profile,
    oar_header,
    validate_cluster,
)
from geomulticorr.scheduler.oar import oarsub_submit, run_local
from geomulticorr.scheduler.cigri import (
    CampaignStatus,
    CigriCampaign,
    CigriError,
    CigriJob,
    CigriOptions,
    campaign_events,
    campaign_status,
    cancel_campaign,
    fix_campaign_events,
    resolve_host,
    submit_campaign,
    write_campaign,
)

__all__ = [
    "CLUSTER_CHOICES",
    "CORRELATION_CLUSTER_CHOICES",
    "GRID_CHOICES",
    "PROFILES",
    "DEFAULT_GRID_CLUSTERS",
    "ClusterProfile",
    "ClusterSpec",
    "asp_env_lines",
    "is_grid_cluster",
    "parse_cluster",
    "cluster_base_env",
    "get_profile",
    "oar_header",
    "validate_cluster",
    "oarsub_submit",
    "run_local",
    "CampaignStatus",
    "CigriCampaign",
    "CigriError",
    "CigriJob",
    "CigriOptions",
    "campaign_events",
    "campaign_status",
    "cancel_campaign",
    "fix_campaign_events",
    "resolve_host",
    "submit_campaign",
    "write_campaign",
]
