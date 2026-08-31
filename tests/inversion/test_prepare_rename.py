#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_prepare_rename.py
# creation date: 2026-08-30.
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
"""tests/inversion/test_prepare_rename.py

``TIOInversion.prepare`` was renamed to ``prepare_inversion``; the old name
survives as a deprecated forwarder because several notebooks still call it, and
a hard break would fail only *after* ``setup_directories()`` had already run.

These use ``TIOInversion.__new__`` (the pattern in ``test_weights.py``) so no
project, geodatabase or TIO binaries are needed.
"""

import inspect
import warnings

import pytest

from geomulticorr.inversion.tio_inversion import TIOInversion


class TestPrepareInversionRename:
    """The new name is the real method; the old one forwards to it."""

    def test_prepare_inversion_exists(self):
        assert callable(getattr(TIOInversion, "prepare_inversion", None))
    # END def

    def test_prepare_alias_still_exists(self):
        """Notebooks calling inv.prepare(...) must keep working until 0.7.0."""
        assert callable(getattr(TIOInversion, "prepare", None))
    # END def

    def test_prepare_inversion_accepts_verbose(self):
        """Signature check, so a future refactor cannot silently drop the knob."""
        params = inspect.signature(TIOInversion.prepare_inversion).parameters
        assert "verbose" in params
        assert params["verbose"].default is False
    # END def

    def test_alias_forwards_arguments(self):
        inv = TIOInversion.__new__(TIOInversion)
        seen = {}
        inv.prepare_inversion = lambda *a, **k: seen.update(kwargs=k, args=a) or "done"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = inv.prepare(weight_mode="uniform", cores=8)

        assert result == "done"
        assert seen["kwargs"] == {"weight_mode": "uniform", "cores": 8}
    # END def

    def test_alias_emits_deprecation_warning(self):
        inv = TIOInversion.__new__(TIOInversion)
        inv.prepare_inversion = lambda *a, **k: None

        with pytest.warns(DeprecationWarning, match="prepare_inversion"):
            inv.prepare()
    # END def

    def test_alias_also_warns_on_the_gmc_channel(self, caplog_gmc):
        """Notebooks routinely filter DeprecationWarning out entirely."""
        inv = TIOInversion.__new__(TIOInversion)
        inv.prepare_inversion = lambda *a, **k: None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            inv.prepare()

        assert "prepare_inversion" in caplog_gmc.text
    # END def

    def test_nothing_in_the_library_calls_the_alias(self):
        """Keeps the alias safe under ``-W error::DeprecationWarning``.

        Parsed rather than grepped: the alias's own warning text mentions
        ``TIOInversion.prepare()``, and a substring search cannot tell that
        string apart from a real call.
        """
        import ast
        import pathlib

        src = pathlib.Path(inspect.getfile(TIOInversion)).parent.parent
        hits = []
        for path in sorted(src.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "prepare"
                ):
                    hits.append(f"{path.name}:{node.lineno}")
        assert hits == [], f"internal callers of the deprecated alias: {hits}"
    # END def
# END class
