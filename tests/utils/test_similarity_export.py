"""Naming and writing the similarity-explorer figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from geomulticorr.similarity.selection import select_couples
from geomulticorr.utils import _similarity_export as se
from geomulticorr.utils import gmc_functions as gf
from tests.utils.test_similarity_plotly import _sim

STASH = {
    "strategy": "similarity",
    "pz_name": "PasDeLours",
    "image_type": "opt",
    "similarity_metric": "score",
    "min_similarity": None,
    "per_image": 2,
    "cross_sensor_only": True,
    "max_dt_days": 730,
    "min_dt_days": None,
    "both_directions": True,
    "sensor_filter": None,
}


class TestStem:
    def test_stash_splats_and_is_deterministic(self):
        a = se.similarity_figure_stem("network", **STASH)
        assert a == "PasDeLours_selection_network_score_cross_per2_dt0-730"
        assert (
            se.similarity_figure_stem("network", **dict(reversed(STASH.items()))) == a
        )

    def test_threshold_type_and_sensor(self):
        stem = se.similarity_figure_stem(
            "matrix",
            pz_name="PZ",
            image_type="hillshade",
            similarity_metric="trial_sigma_px",
            min_similarity=0.75,
            sensor_filter="spot",
        )
        assert stem == "PZ_hillshade_selection_matrix_trial_sigma_px0.75_spot"

    def test_defaults(self):
        assert se.similarity_figure_stem("matrix") == "all_selection_matrix_score"


class TestSave:
    @pytest.mark.parametrize("view", ["matrix", "network"])
    def test_all_formats_no_leak(self, tmp_path, view):
        plt.close("all")
        sim = _sim()
        sel = select_couples(sim, threshold=0.0)
        paths = se.save_similarity_figure(
            sim, sel, tmp_path, view=view, formats=("html", "png", "pdf", "svg")
        )
        assert set(paths) == {"html", "png", "pdf", "svg"}
        assert all(p.exists() and p.stat().st_size > 0 for p in paths.values())
        assert plt.get_fignums() == []

    def test_overwrite_false_numbers(self, tmp_path):
        sim = _sim()
        first = se.save_similarity_figure(sim, None, tmp_path, formats="png")
        second = se.save_similarity_figure(
            sim, None, tmp_path, formats="png", overwrite=False
        )
        assert second["png"].name == first["png"].stem + "_01.png"

    def test_errors(self, tmp_path):
        sim = _sim()
        with pytest.raises(ValueError, match="Unknown view"):
            se.save_similarity_figure(sim, None, tmp_path, view="chord")
        with pytest.raises(ValueError, match="Unknown format"):
            se.save_similarity_figure(sim, None, tmp_path, formats="gif")
        with pytest.raises(ValueError, match="empty frame"):
            se.save_similarity_figure(sim.iloc[0:0], None, tmp_path)
        with pytest.raises(ValueError, match="No couple selected"):
            se.save_similarity_figure(sim, sim.iloc[0:0], tmp_path, view="network")

    def test_empty_selection_matrix_still_saves(self, tmp_path):
        paths = se.save_similarity_figure(
            _sim(), _sim().iloc[0:0], tmp_path, formats="png"
        )
        assert paths["png"].exists()


class TestStaticMatrix:
    def test_selection_outlined_in_both_triangles(self):
        sim = _sim()
        fig = gf.plot_similarity_selection(sim, sim.head(2), metric="coh_mean")
        try:
            (ax,) = [a for a in fig.axes if a.get_title()]
            (outline,) = ax.collections
            assert len(outline.get_offsets()) == 4
            assert len(ax.images) == 1
        finally:
            plt.close(fig)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="not a column"):
            gf.plot_similarity_selection(_sim(), metric="absent")

    def test_overview_matrices_unchanged(self):
        fig = gf.plot_similarity_matrices(_sim())
        try:
            panels = [a for a in fig.axes if a.images]
            assert len(panels) == 4
            assert [a.get_title() for a in panels] == list(gf.SIMILARITY_MATRIX_METRICS)
            assert not any(a.collections for a in panels)
        finally:
            plt.close(fig)
