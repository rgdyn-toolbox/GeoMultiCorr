#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# oar.py
# creation date: 2026-11-13.
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
"""Direct execution: ``oarsub -S`` per script, or ``bash`` on this machine."""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["oarsub_submit", "run_local"]


def oarsub_submit(script_path: Path, cwd: Path | None = None) -> None:
    """Submit *script_path* to OAR via ``oarsub -S``.

    Parameters
    ----------
    script_path:
        Absolute path to the bash script to submit.
    cwd:
        Working directory for the ``oarsub`` call. Defaults to
        ``script_path.parent``.
    """
    if cwd is None:
        cwd = script_path.parent
    subprocess.run(
        f"oarsub -S ./{script_path.name}",
        shell=True,
        cwd=str(cwd),
        check=True,
    )


def run_local(cmd: str, cwd: Path | None = None) -> None:
    """Run *cmd* locally via ``subprocess.run(shell=True, check=True)``.

    Parameters
    ----------
    cmd:
        Shell command string to execute.
    cwd:
        Working directory. If ``None``, inherits from the current process.
    """
    subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
    )
