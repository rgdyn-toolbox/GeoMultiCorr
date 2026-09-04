#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_weights_export.py
# creation date: 2026-08-31.
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
"""tests/utils/test_weights_export.py

Two properties carry the weight here:

* the file **stem is a pure function of the parameters** — so re-running a
  notebook refreshes the same files rather than accumulating copies;
* the matplotlib figure is **always closed**, since the explorer's save button
  would otherwise leak one figure per click.
"""
from __future__ import annotations

import sys

import matplotlib
import pytest

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

from geomulticorr.utils._weights_export import (  # noqa: E402
    save_weights_figure,
    weights_figure_stem,
)
from geomulticorr.utils._weights_frame import weights_frame  # noqa: E402


@pytest.fixture
def frame():
    return weights_frame(
        ["pz_a", "pz_b", "pz_c"], [30.0, 90.0, 200.0],
        [0.9, 0.6, 0.3], [0.8, 0.5, 0.2],
        nmad_ew=[0.2, 0.3, 0.4], nmad_ns=[0.25, 0.35, 0.45],
        cc=[0.8, 0.7, 0.6], corr_direction=["Forward", "Forward", "Backward"],
    )


@pytest.fixture
def quality_params():
    return dict(weight_mode="quality", combine="geomean", cc_gamma=1.5,
                invert=False, sharpness=5.0, slope=2.0, min_weight=0.1,
                w_min=0.0, alpha=1 / 3, beta=1 / 3, gamma=1 / 3, dt_range=None)


class TestFigureStem:
    """A pure function of the parameters: no timestamp, no counts."""

    def test_same_parameters_give_the_same_stem(self, quality_params):
        first = weights_figure_stem(inversion_name="PDL", pz_name="PasDeLours",
                                    **quality_params)
        second = weights_figure_stem(inversion_name="PDL", pz_name="PasDeLours",
                                     **quality_params)
        assert first == second

    def test_stem_names_the_pzone_inversion_and_mode(self, quality_params):
        stem = weights_figure_stem(inversion_name="PDL_spot",
                                   pz_name="PasDeLours", **quality_params)
        assert stem.startswith("PasDeLours_PDL_spot_weights_quality")

    def test_missing_pzone_becomes_all(self, quality_params):
        assert weights_figure_stem(**quality_params).startswith("all_")

    def test_a_relevant_parameter_changes_the_stem(self, quality_params):
        base = weights_figure_stem(**quality_params)
        changed = weights_figure_stem(**{**quality_params, "cc_gamma": 2.0})
        assert base != changed

    def test_an_irrelevant_parameter_does_not(self, quality_params):
        """Moving a slider the mode ignores must not start a second file."""
        base = weights_figure_stem(**quality_params)
        changed = weights_figure_stem(**{**quality_params, "sharpness": 11.0})
        assert base == changed

    def test_alpha_only_appears_under_wmean(self, quality_params):
        geomean = weights_figure_stem(**quality_params)
        wmean = weights_figure_stem(**{**quality_params, "combine": "wmean"})
        assert "alpha" not in geomean
        assert "alpha" in wmean

    def test_dt_range_separates_two_otherwise_identical_sigmoid_runs(self):
        """Without dt_range in the relevant set, these two collide on one name."""
        common = dict(weight_mode="sigmoid", sharpness=8.0, w_min=0.2, invert=True)
        derived = weights_figure_stem(**common, dt_range=None)
        explicit = weights_figure_stem(**common, dt_range=(300, 400))
        assert derived != explicit
        assert "dt300-400" in explicit

    def test_invert_is_a_flag_not_a_value(self, quality_params):
        on = weights_figure_stem(**{**quality_params, "invert": True})
        off = weights_figure_stem(**{**quality_params, "invert": False})
        assert "invert" in on and "invert" not in off

    def test_floats_do_not_leak_repr_noise(self):
        stem = weights_figure_stem(weight_mode="quality", combine="wmean",
                                   alpha=1 / 3, beta=1 / 3, gamma=1 / 3)
        assert "0.3333333333333333" not in stem

    def test_stem_is_filesystem_safe(self):
        stem = weights_figure_stem(inversion_name="a/b c", pz_name="x:y",
                                   weight_mode="quality", combine="geo mean")
        assert not set(stem) & set("/\\: ")


class TestSaveWeightsFigure:
    def test_writes_the_requested_formats(self, frame, tmp_path):
        paths = save_weights_figure(frame, tmp_path, formats=("html", "png"),
                                    stem="fig")
        assert set(paths) == {"html", "png"}
        for path in paths.values():
            assert path.exists() and path.stat().st_size > 0

    def test_creates_the_destination(self, frame, tmp_path):
        out = tmp_path / "figures" / "inversion"
        save_weights_figure(frame, out, formats="png", stem="fig")
        assert (out / "fig.png").exists()

    def test_format_strings_are_normalised(self, frame, tmp_path):
        paths = save_weights_figure(frame, tmp_path, formats=(".PNG", "png"),
                                    stem="fig")
        assert set(paths) == {"png"}

    def test_a_single_format_string_is_accepted(self, frame, tmp_path):
        assert set(save_weights_figure(frame, tmp_path, formats="pdf",
                                       stem="fig")) == {"pdf"}

    def test_overwrite_replaces_in_place(self, frame, tmp_path):
        save_weights_figure(frame, tmp_path, formats="png", stem="fig")
        save_weights_figure(frame, tmp_path, formats="png", stem="fig")
        assert [p.name for p in tmp_path.glob("*.png")] == ["fig.png"]

    def test_overwrite_false_appends_a_counter(self, frame, tmp_path):
        save_weights_figure(frame, tmp_path, formats="png", stem="fig")
        save_weights_figure(frame, tmp_path, formats="png", stem="fig",
                            overwrite=False)
        assert sorted(p.name for p in tmp_path.glob("*.png")) == [
            "fig.png", "fig_01.png"]

    def test_unknown_format_raises(self, frame, tmp_path):
        with pytest.raises(ValueError, match="Unknown format"):
            save_weights_figure(frame, tmp_path, formats="tiff")

    def test_empty_frame_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_weights_figure(weights_frame([], [], [], []), tmp_path)

    def test_one_direction_only(self, frame, tmp_path):
        paths = save_weights_figure(frame, tmp_path, formats=("html", "png"),
                                    stem="fig", view_kwargs={"directions": "EW"})
        assert len(paths) == 2

    def test_a_backend_specific_kwarg_is_dropped_not_raised(self, frame, tmp_path):
        """`height` is plotly-only; matplotlib must not choke on it."""
        paths = save_weights_figure(frame, tmp_path, formats=("html", "png"),
                                    stem="fig", view_kwargs={"height": 700})
        assert set(paths) == {"html", "png"}


class TestNoFigureLeak:
    """The explorer's save button calls this on every click."""

    def test_no_figure_is_left_open(self, frame, tmp_path):
        plt.close("all")
        before = len(plt.get_fignums())
        save_weights_figure(frame, tmp_path, formats=("png", "pdf"), stem="fig")
        assert len(plt.get_fignums()) == before

    def test_no_figure_leaks_when_saving_fails(self, frame, tmp_path, monkeypatch):
        """The close lives in a `finally`, so a mid-write failure still closes."""
        plt.close("all")
        before = len(plt.get_fignums())

        import matplotlib.figure

        def _boom(self, *a, **k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _boom)
        with pytest.raises(RuntimeError):
            save_weights_figure(frame, tmp_path, formats="png", stem="fig")
        assert len(plt.get_fignums()) == before


class TestKaleidoIsNeverNeeded:
    """Static output goes through the matplotlib twin, never fig.write_image."""

    def test_kaleido_is_not_imported(self, frame, tmp_path):
        assert "kaleido" not in sys.modules
        save_weights_figure(frame, tmp_path, formats=("html", "png", "pdf", "svg"),
                            stem="fig")
        assert "kaleido" not in sys.modules
