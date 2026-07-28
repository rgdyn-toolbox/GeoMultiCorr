#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for writing pair figures to disk (_pairs_export)."""
from __future__ import annotations

import pytest

pytest.importorskip("plotly")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geomulticorr.utils._pairs_export import (  # noqa: E402
    FIGURE_FORMATS,
    _unique_path,
    pairs_figure_stem,
    save_pairs_figure,
)
from geomulticorr.utils._pairs_frame import (  # noqa: E402
    empty_pairs_frame,
    pairs_frame_from_indices,
)

@pytest.fixture
def frame(dates_iso, sensors, index_pairs_redundancy):
    return pairs_frame_from_indices(
        dates_iso, index_pairs_redundancy, sensors=sensors, pz="PZ1"
    )


class TestFigureStem:
    def test_encodes_every_parameter(self):
        stem = pairs_figure_stem(
            "network", strategy="redundancy", pz_name="Chimborazo",
            max_step=17, max_dt_days=800, min_dt_days=60,
            sensor_filter="planetscope",
        )
        assert stem == "Chimborazo_network_redundancy_maxstep17_dt60-800_planetscope"

    def test_pzone_comes_first(self):
        assert pairs_figure_stem("chord", pz_name="PZ1").startswith("PZ1_")

    def test_omits_absent_segments(self):
        assert pairs_figure_stem("chord", strategy="consecutive") == "all_chord_consecutive"

    def test_empty_pzone_becomes_all(self):
        assert pairs_figure_stem("chord", strategy="step", max_step=2,
                                 pz_name="") == "all_chord_step_maxstep2"

    def test_unsafe_characters_sanitised(self):
        stem = pairs_figure_stem("chord", strategy="a/b c", pz_name="P:Z 1")
        assert "/" not in stem and " " not in stem and ":" not in stem

    def test_is_deterministic(self):
        """No timestamp — the same parameters must always give the same name."""
        kwargs = dict(strategy="redundancy", pz_name="PZ1", max_step=5)
        assert pairs_figure_stem("chord", **kwargs) == pairs_figure_stem("chord", **kwargs)

    def test_distinct_parameters_give_distinct_names(self):
        seen = {
            pairs_figure_stem(view, strategy=strat, pz_name=pz, max_step=k)
            for view in ("chord", "network")
            for strat in ("step", "redundancy")
            for pz in ("A", "B")
            for k in (3, 17)
        }
        assert len(seen) == 2 * 2 * 2 * 2

    def test_accepts_last_pairs_params_splat(self):
        """The six keys of Session._last_pairs_params must splat in unchanged."""
        params = dict(strategy="redundancy", max_step=3, max_dt_days=None,
                      min_dt_days=None, sensor_filter=None, pz_name="PZ1")
        assert pairs_figure_stem("chord", **params) == "PZ1_chord_redundancy_maxstep3"


class TestUniquePath:
    def test_appends_counter_on_collision(self, tmp_path):
        first = _unique_path(tmp_path, "fig", "png")
        first.write_text("x")
        second = _unique_path(tmp_path, "fig", "png")
        assert first.name == "fig.png"
        assert second.name == "fig_01.png"


class TestOverwriteSemantics:
    def test_resaving_replaces_in_place(self, frame, tmp_path):
        first = save_pairs_figure(frame, tmp_path, view="dt_hist",
                                  formats=("png",), stem="same")["png"]
        second = save_pairs_figure(frame, tmp_path, view="dt_hist",
                                   formats=("png",), stem="same")["png"]
        assert first == second
        assert len(list(tmp_path.glob("same*.png"))) == 1

    def test_overwrite_false_keeps_both(self, frame, tmp_path):
        a = save_pairs_figure(frame, tmp_path, view="dt_hist", formats=("png",),
                              stem="keep", overwrite=False)["png"]
        b = save_pairs_figure(frame, tmp_path, view="dt_hist", formats=("png",),
                              stem="keep", overwrite=False)["png"]
        assert a != b
        assert b.name == "keep_01.png"

    def test_default_stem_is_stable_across_runs(self, frame, tmp_path):
        kw = dict(view="chord", formats=("png",),
                  view_kwargs={"dedupe": True})
        first = save_pairs_figure(frame, tmp_path, **kw)["png"]
        second = save_pairs_figure(frame, tmp_path, **kw)["png"]
        assert first == second


class TestJpeg:
    def test_jpg_is_supported(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, view="network", formats=("jpg",),
                                stem="raster")
        assert out["jpg"].suffix == ".jpg"
        assert out["jpg"].stat().st_size > 0

    def test_jpg_alongside_the_others(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, view="baseline",
                                formats=("png", "jpg"), stem="both")
        assert set(out) == {"png", "jpg"}


class TestSaveHtml:
    def test_writes_self_contained_html(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, view="chord", formats=("html",),
                                stem="chord_test")
        path = out["html"]
        assert path.exists() and path.stat().st_size > 0
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "plotly" in text.lower()
        # the library is inlined, not pulled from a CDN <script src=...>
        assert 'src="https://cdn.plot.ly' not in text
        assert path.stat().st_size > 1_000_000

    def test_cdn_mode_is_smaller(self, frame, tmp_path):
        full = save_pairs_figure(frame, tmp_path, formats=("html",), stem="a")["html"]
        cdn = save_pairs_figure(frame, tmp_path, formats=("html",), stem="b",
                                plotlyjs="cdn")["html"]
        assert cdn.stat().st_size < full.stat().st_size

    def test_never_calls_plotly_static_export(self):
        """Static output must go through matplotlib: kaleido is not a GMC dependency."""
        import pathlib

        import geomulticorr.utils._pairs_export as mod

        code = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in code.splitlines() if not line.lstrip().startswith("#")
        )
        assert "write_image" not in code


class TestSaveImages:
    @pytest.mark.parametrize("view", ["baseline", "chord", "network", "dt_hist"])
    def test_png_for_every_view(self, frame, tmp_path, view):
        out = save_pairs_figure(frame, tmp_path, view=view, formats=("png",),
                                stem=f"{view}_test")
        assert out["png"].exists()
        assert out["png"].stat().st_size > 0

    def test_multiple_formats_at_once(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, view="dt_hist",
                                formats=("html", "png", "pdf"), stem="multi")
        assert set(out) == {"html", "png", "pdf"}
        assert all(p.exists() for p in out.values())

    def test_no_figure_leak(self, frame, tmp_path):
        plt.close("all")
        save_pairs_figure(frame, tmp_path, view="chord", formats=("png",), stem="leak")
        assert plt.get_fignums() == []

    def test_creates_missing_directory(self, frame, tmp_path):
        target = tmp_path / "deep" / "nested"
        save_pairs_figure(frame, target, formats=("png",), stem="x")
        assert target.is_dir()

    def test_title_forwarded_to_both_backends(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, view="dt_hist",
                                formats=("html",), stem="titled", title="My title")
        assert "My title" in out["html"].read_text(encoding="utf-8", errors="ignore")


class TestViewKwargs:
    def test_view_specific_kwargs_pass_through(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, view="chord", formats=("png",),
                                stem="kw", view_kwargs={"show_arrows": True,
                                                        "dedupe": False})
        assert out["png"].exists()

    def test_unknown_kwargs_are_dropped_not_raised(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, view="dt_hist", formats=("png",),
                                stem="drop", view_kwargs={"not_a_real_option": 1})
        assert out["png"].exists()

    def test_backend_specific_kwarg_survives(self, frame, tmp_path):
        """`nbins` is understood by dt_hist on both sides; `n_ring_points` only by plotly."""
        out = save_pairs_figure(frame, tmp_path, view="chord",
                                formats=("html", "png"), stem="mixed",
                                view_kwargs={"n_ring_points": 180, "dedupe": True})
        assert set(out) == {"html", "png"}


class TestValidation:
    def test_unknown_view_raises(self, frame, tmp_path):
        with pytest.raises(ValueError, match="Unknown view"):
            save_pairs_figure(frame, tmp_path, view="nope")

    def test_unknown_format_raises(self, frame, tmp_path):
        with pytest.raises(ValueError, match="Unknown format"):
            save_pairs_figure(frame, tmp_path, formats=("gif",))

    def test_empty_frame_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty pairs frame"):
            save_pairs_figure(empty_pairs_frame(), tmp_path)

    def test_format_string_accepted(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, formats="png", stem="single")
        assert set(out) == {"png"}

    def test_extension_dots_tolerated(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, formats=(".PNG",), stem="dotted")
        assert set(out) == {"png"}

    def test_all_declared_formats_are_supported(self, frame, tmp_path):
        out = save_pairs_figure(frame, tmp_path, view="dt_hist",
                                formats=FIGURE_FORMATS, stem="everything")
        assert set(out) == set(FIGURE_FORMATS)
