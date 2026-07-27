#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# console.py
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
"""Shared rich console singleton.

A single :class:`rich.console.Console` instance is shared across the whole
package so that progress bars (Live regions) and summary tables write to the
same stream — two different Console objects writing to the same terminal breaks
rich's live-region rendering.
"""
from rich.console import Console as _RichConsole

# Module-level singleton. Import this everywhere a rich console is needed
# (progress bars in session.py / tio_inversion.py, and the summary tables in
# tables.py) so all output goes through one Console.
_rich_console = _RichConsole(highlight=False)
