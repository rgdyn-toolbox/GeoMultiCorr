#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# tables.py
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
"""Rich summary-table builders for GeoMultiCorr pipeline steps.

Each ``print_*_summary`` function builds a :class:`rich.table.Table` from the
``summary_rows`` collected by a pipeline method, prints it through the shared
:data:`_rich_console` when ``print_summary`` is true, and returns the table so a
future CLI or test can render/capture it instead.

The table layouts (titles, columns, styles, embedded rich markup) are kept
identical to the original inline blocks in ``session.py`` / ``tio_inversion.py``.
"""
from rich.table import Table

from geomulticorr.core.console.console import _rich_console


def print_register_thumbs_summary(rows, pzone_name, *, print_summary=True) -> Table:
    """Summary for ``Session.register_existing_thumbs``."""
    table = Table(title=f"register_existing_thumbs — pzone '{pzone_name}'", show_lines=False)
    table.add_column("Source", style="default", no_wrap=True)
    table.add_column("Thumb", style="cyan", no_wrap=True)
    table.add_column("Status", style="default")
    for row in rows:
        table.add_row(row["src"], row["thumb"], row["status"])
    # Original block is double-guarded (`print_summary and summary_rows`); the
    # caller keeps the row guard, so only re-check print_summary here.
    if print_summary:
        _rich_console.print(table)
    return table


def print_sieve_bulk_summary(rows, target_resolution, *, print_summary=True) -> Table:
    """Summary for ``Session.sieve_bulk``."""
    table = Table(title="sieve_bulk — processed thumbs", show_lines=False)
    table.add_column("Thumb",   style="cyan", no_wrap=True)
    table.add_column("Sensor",  style="default")
    table.add_column("Res (m)", style="default", justify="right")
    table.add_column("Status",  style="default")
    for row in rows:
        table.add_row(row["name"], row["sensor"], str(target_resolution), row["status"])
    if print_summary:
        _rich_console.print(table)
    return table


def print_prepare_correlation_summary(rows, *, print_summary=True) -> Table:
    """Summary for ``Session.prepare_pairs_correlation``."""
    table = Table(title="prepare_pairs_correlation — summary", show_lines=False)
    table.add_column("Pair",         style="cyan", no_wrap=True)
    table.add_column("Directory",    style="dim")
    table.add_column("Symlink",      style="dim")
    table.add_column("Corr. params", style="dim")
    table.add_column("Job Script",   style="dim")
    for rec in rows:
        table.add_row(rec["pair"], rec["directory"], rec["symlink"], rec["params"], rec["script"])
    if print_summary:
        _rich_console.print(table)
    return table


def print_launch_correlation_summary(rows, *, print_summary=True) -> Table:
    """Summary for ``Session.launch_pairs_correlation``."""
    table = Table(title="launch_pairs_correlation — summary", show_lines=False)
    table.add_column("Pair", style="cyan", no_wrap=True)
    table.add_column("Status")
    for rec in rows:
        table.add_row(rec["pair"], rec["status"])
    if print_summary:
        _rich_console.print(table)
    return table


def print_sync_after_cluster_summary(rows, *, print_summary=True) -> Table:
    """Summary for ``Session.sync_pairs_after_cluster``."""
    table = Table(title="sync_pairs_after_cluster — summary", show_lines=False)
    table.add_column("Pair", style="cyan", no_wrap=True)
    table.add_column("Correlated OK")
    table.add_column("Files deleted", justify="right")
    table.add_column("DB status", style="dim")
    for rec in rows:
        table.add_row(rec["pair"], rec["correlated"], rec["deleted"], rec["db_status"])
    if print_summary:
        _rich_console.print(table)
    return table


def print_reset_outputs_summary(rows, *, print_summary=True) -> Table:
    """Summary for ``Session.reset_pairs_outputs``."""
    table = Table(title="reset_pairs_outputs — summary", show_lines=False)
    table.add_column("Pair", style="cyan", no_wrap=True)
    table.add_column("Status before", style="dim")
    table.add_column("Files deleted", justify="right")
    table.add_column("Status after")
    for rec in rows:
        table.add_row(rec["pair"], rec["status_before"], rec["deleted"], rec["status_after"])
    if print_summary:
        _rich_console.print(table)
    return table


def print_extract_displacements_summary(rows, target_resolution, resampling, *, print_summary=True) -> Table:
    """Summary for ``Session.extract_pairs_raw_displacements``.

    The ``Resampling`` column is only present when ``target_resolution`` is set.
    """
    table = Table(title="extract_pairs_raw_displacements — summary", show_lines=False)
    table.add_column("Pair", style="cyan", no_wrap=True)
    table.add_column("EW", justify="center")
    table.add_column("NS", justify="center")
    table.add_column("CC", justify="center")
    table.add_column("Raw plot", justify="center")
    table.add_column("Stats", justify="center")
    if target_resolution is not None:
        table.add_column("Resampling", justify="center", style="dim")
    for rec in rows:
        row_data = [rec["pair"], rec["ew"], rec["ns"], rec["cc"], rec["plot"], rec["stats"]]
        if target_resolution is not None:
            row_data.append(f"{resampling} @ {target_resolution} m")
        table.add_row(*row_data)
    if print_summary:
        _rich_console.print(table)
    return table


def print_apply_corrections_summary(rows, *, print_summary=True) -> Table:
    """Summary for ``Session.apply_pairs_corrections``."""
    table = Table(title="apply_pairs_corrections — summary", show_lines=False)
    table.add_column("Pair",        style="cyan", no_wrap=True)
    table.add_column("Median corr", justify="center")
    table.add_column("Ramp/Topo",   justify="center")
    table.add_column("Stats",       justify="center")
    table.add_column("EW corr",     justify="center")
    table.add_column("NS corr",     justify="center")
    for rec in rows:
        table.add_row(
            rec["pair"], rec["median"], rec["ramp_topo"],
            rec["stats"], rec["ew_corr"], rec["ns_corr"],
        )
    if print_summary:
        _rich_console.print(table)
    return table


def print_tio_export_summary(rows, *, print_summary=True) -> Table:
    """Summary for ``TIOInversion.prepare`` pair export.

    ``rows`` is a list of ``(pair_key, status)`` tuples.
    """
    table = Table(title="TIO prepare — pair export summary", show_lines=False)
    table.add_column("Pair", style="cyan", no_wrap=True)
    table.add_column("Status")
    for key, status in rows:
        table.add_row(key, status)
    if print_summary:
        _rich_console.print(table)
    return table
