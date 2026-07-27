#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# __init__.py
# creation date: 2026-07-15.
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
"""Console output helpers: shared rich console + summary-table builders."""
from geomulticorr.core.console.console import _rich_console
from geomulticorr.core.console.tables import (
    print_register_thumbs_summary,
    print_sieve_bulk_summary,
    print_prepare_correlation_summary,
    print_launch_correlation_summary,
    print_sync_after_cluster_summary,
    print_reset_outputs_summary,
    print_extract_displacements_summary,
    print_apply_corrections_summary,
    print_tio_export_summary,
)
