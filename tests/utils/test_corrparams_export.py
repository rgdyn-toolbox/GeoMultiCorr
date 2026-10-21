#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for correlation-parameter figure naming and writing."""
from __future__ import annotations

import pathlib

import matplotlib
import pytest

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

from geomulticorr.utils._corrparams_export import (  # noqa: E402
    corrparams_figure_stem,
    save_corrparams_figure,
)
from geomulticorr.utils._corrparams_frame import (  # noqa: E402
    corrparams_frame_from_pairs,
    empty_corrparams_frame,
)
from geomulticorr.utils._corrparams_plotly import VIEW_BUILDERS  # noqa: E402

ASSUMPTIONS = dict(velocity_m_yr=5.0, footprint_m=60.0, c_px=0.15, k=3.0,
                   coreg_px=2.0, margin_px=2.0)


@pytest.fixture
def frame():
    return corrparams_frame_from_pairs(
        ["a", "b", "c"], [30.0, 400.0, 2000.0], [3.0, 3.0, 3.0],
        sensor_i=["ps"] * 3, velocity_m_yr=2.0,
    )


class TestStem:
    def test_shape(self):
        stem = corrparams_figure_stem("design_map", pz_name="Chimborazo", **ASSUMPTIONS)
        assert stem.startswith("Chimborazo_design_map_asp_bm")

    def test_no_pzone_becomes_all(self):
        assert corrparams_figure_stem("snr").startswith("all_snr_")

    def test_is_a_pure_function_of_the_parameters(self):
        """No timestamp, no pair count — re-running refreshes the same files."""
        a = corrparams_figure_stem("design_map", pz_name="PZ", **ASSUMPTIONS)
        b = corrparams_figure_stem("design_map", pz_name="PZ", **ASSUMPTIONS)
        assert a == b

    def test_key_order_is_fixed_regardless_of_dict_order(self):
        forward = corrparams_figure_stem("snr", velocity_m_yr=5.0, footprint_m=60.0)
        reverse = corrparams_figure_stem("snr", footprint_m=60.0, velocity_m_yr=5.0)
        assert forward == reverse

    def test_a_changed_parameter_changes_the_stem(self):
        base = corrparams_figure_stem("design_map", **ASSUMPTIONS)
        other = corrparams_figure_stem("design_map", **{**ASSUMPTIONS, "k": 2.0})
        assert base != other

    def test_irrelevant_parameters_are_pruned(self):
        """SGM forces 9x9, so a footprint in the stem would name a no-op."""
        stem = corrparams_figure_stem("design_map", corr_algorithm="asp_mgm",
                                      **ASSUMPTIONS)
        assert "L60" not in stem

    def test_unset_optional_refinements_leave_no_fragment(self):
        stem = corrparams_figure_stem("design_map", flow_azimuth_deg=None, **ASSUMPTIONS)
        assert "az" not in stem

    def test_set_optional_refinements_appear(self):
        stem = corrparams_figure_stem("design_map", flow_azimuth_deg=90.0, **ASSUMPTIONS)
        assert "az90" in stem

    def test_floats_are_trimmed(self):
        stem = corrparams_figure_stem("snr", c_px=0.15000000000000002)
        assert "c0.15" in stem
        assert "0.15000" not in stem

    def test_stem_is_filesystem_safe(self):
        stem = corrparams_figure_stem("design_map", pz_name="Pas De Lours/2")
        assert "/" not in stem and " " not in stem


class TestSave:
    def test_writes_every_requested_format(self, frame, tmp_path):
        out = save_corrparams_figure(frame, tmp_path, view="design_map",
                                     formats=("html", "png"), stem="s")
        assert set(out) == {"html", "png"}
        for path in out.values():
            assert path.exists() and path.stat().st_size > 0

    @pytest.mark.parametrize("view", list(VIEW_BUILDERS))
    def test_every_view_saves(self, view, frame, tmp_path):
        out = save_corrparams_figure(frame, tmp_path, view=view,
                                     formats=("png",), stem=f"s_{view}")
        assert out["png"].exists()

    def test_html_is_self_contained(self, frame, tmp_path):
        out = save_corrparams_figure(frame, tmp_path, view="snr",
                                     formats=("html",), stem="s")
        # inlined plotly.js, so the page works offline
        assert out["html"].stat().st_size > 100_000

    def test_rerunning_refreshes_rather_than_accumulates(self, frame, tmp_path):
        for _ in range(3):
            save_corrparams_figure(frame, tmp_path, view="snr",
                                   formats=("png",), stem="s")
        assert len(list(tmp_path.glob("*.png"))) == 1

    def test_overwrite_false_keeps_every_version(self, frame, tmp_path):
        for _ in range(3):
            save_corrparams_figure(frame, tmp_path, view="snr", formats=("png",),
                                   stem="s", overwrite=False)
        assert len(list(tmp_path.glob("*.png"))) == 3

    def test_creates_the_directory(self, frame, tmp_path):
        target = tmp_path / "deep" / "deeper"
        save_corrparams_figure(frame, target, view="snr", formats=("png",), stem="s")
        assert target.exists()

    def test_unknown_view_raises(self, frame, tmp_path):
        with pytest.raises(ValueError, match="Unknown view"):
            save_corrparams_figure(frame, tmp_path, view="nope")

    def test_unknown_format_raises(self, frame, tmp_path):
        with pytest.raises(ValueError, match="Unknown format"):
            save_corrparams_figure(frame, tmp_path, view="snr", formats=("tiff",))

    def test_empty_frame_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty frame"):
            save_corrparams_figure(empty_corrparams_frame(), tmp_path, view="snr")

    def test_no_format_raises(self, frame, tmp_path):
        with pytest.raises(ValueError, match="No output format"):
            save_corrparams_figure(frame, tmp_path, view="snr", formats=())

    def test_unknown_view_kwargs_are_dropped_not_raised(self, frame, tmp_path):
        out = save_corrparams_figure(
            frame, tmp_path, view="cost", formats=("png",), stem="s",
            view_kwargs={"tau_days": 900.0, "nonsense": 1},
        )
        assert out["png"].exists()

    def test_does_not_leak_matplotlib_figures(self, frame, tmp_path):
        """The explorer's save button would otherwise leak one per click."""
        before = len(plt.get_fignums())
        for view in VIEW_BUILDERS:
            save_corrparams_figure(frame, tmp_path, view=view,
                                   formats=("png",), stem=f"s_{view}")
        assert len(plt.get_fignums()) == before

    def test_closes_the_figure_even_when_saving_fails(self, frame, tmp_path):
        before = len(plt.get_fignums())
        with pytest.raises(Exception):
            save_corrparams_figure(frame, tmp_path / "x", view="snr",
                                   formats=("png",), stem="s", dpi="not-a-dpi")
        assert len(plt.get_fignums()) == before

    def test_default_stem_is_used_when_omitted(self, frame, tmp_path):
        out = save_corrparams_figure(frame, tmp_path, view="cost", formats=("png",))
        assert out["png"].name.startswith("all_cost_")


class TestNoKaleido:
    """``kaleido`` is not a dependency; static output uses the mpl twins.

    Asserted against the parsed syntax tree rather than the raw text, so the
    module can keep *documenting* the rule without tripping its own test.
    """

    @staticmethod
    def _tree():
        import ast

        return ast.parse(
            pathlib.Path("geomulticorr/utils/_corrparams_export.py").read_text()
        )

    def test_never_calls_write_image(self):
        import ast

        calls = [
            node.func.attr
            for node in ast.walk(self._tree())
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "write_image" not in calls

    def test_never_imports_kaleido(self):
        import ast

        imported = set()
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any("kaleido" in name for name in imported)

    def test_static_output_goes_through_the_matplotlib_twins(self):
        source = pathlib.Path("geomulticorr/utils/_corrparams_export.py").read_text()
        assert "CORRPARAMS_MPL_BUILDERS" in source
