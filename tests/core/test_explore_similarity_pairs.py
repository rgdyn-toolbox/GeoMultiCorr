"""Session.explore_similarity_pairs — headless and widget, on a temp project."""

from __future__ import annotations

import contextlib
import inspect
import io

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

from geomulticorr.core.session import Session
from geomulticorr.similarity import selection as sim_selection
from tests.similarity.conftest import SPOT, make_similarity_project

KW = dict(kernel=21, search=4, tile=32, progress=False, plots=False)


@pytest.fixture
def session(tmp_path):
    s = make_similarity_project(tmp_path)
    s.compute_thumbs_similarity("TestPz", **KW)
    return s


def _widgets(box):
    """Flatten the explorer's widget tree."""
    out, stack = [], [box]
    while stack:
        w = stack.pop()
        out.append(w)
        stack.extend(getattr(w, "children", ()))
    return out


def _find(box, description):
    return next(
        w for w in _widgets(box) if getattr(w, "description", None) == description
    )


class TestHeadless:
    @pytest.mark.parametrize("view", ["matrix", "network"])
    def test_returns_frame_and_figure(self, session, view):
        sel, fig = session.explore_similarity_pairs(
            "TestPz", view=view, interactive=False, min_similarity=0.0
        )
        assert isinstance(sel, pd.DataFrame) and len(sel) == 6
        assert isinstance(fig, go.Figure)

    def test_threshold_and_cross(self, session):
        sel, _ = session.explore_similarity_pairs(
            "TestPz", interactive=False, cross_sensor_only=True, per_image=1
        )
        assert len(sel) == 1 and set(sel.anchor_key) == {SPOT}

    def test_stash_commits_the_same_couples(self, session):
        sel, _ = session.explore_similarity_pairs(
            "TestPz", interactive=False, min_similarity=0.5, both_directions=False
        )
        layer = session.update_pairs(**session._last_similarity_params)
        assert len(layer) == len(sel)

    def test_stash_keys_are_update_pairs_parameters(self, session):
        session.explore_similarity_pairs("TestPz", interactive=False)
        accepted = set(inspect.signature(Session.update_pairs).parameters) - {"self"}
        assert set(session._last_similarity_params) <= accepted
        assert session._last_similarity_params["strategy"] == "similarity"

    def test_empty_selection_never_raises(self, session):
        for view in ("matrix", "network"):
            sel, fig = session.explore_similarity_pairs(
                "TestPz", view=view, interactive=False, min_similarity=1.01
            )
            assert sel.empty and isinstance(fig, go.Figure)

    def test_bad_arguments(self, session):
        for removed in ("pie", "chord", "table"):
            with pytest.raises(ValueError, match="Unknown view"):
                session.explore_similarity_pairs(
                    "TestPz", view=removed, interactive=False
                )
        with pytest.raises(TypeError, match="Unexpected"):
            session.explore_similarity_pairs("TestPz", interactive=False, max_dt=3)
        with pytest.raises(ValueError, match="similarity_metric"):
            session.explore_similarity_pairs(
                "TestPz", interactive=False, similarity_metric="nope"
            )

    def test_no_trace(self, tmp_path):
        s = make_similarity_project(tmp_path)
        with pytest.raises(FileNotFoundError, match="compute_thumbs_similarity"):
            s.explore_similarity_pairs("TestPz", interactive=False)


class TestWidget:
    @pytest.fixture
    def counted(self, monkeypatch):
        calls = []
        real = sim_selection.select_couples

        def _spy(*a, **k):
            calls.append(k)
            return real(*a, **k)

        monkeypatch.setattr(sim_selection, "select_couples", _spy)
        return calls

    def _open(self, session, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return session.explore_similarity_pairs("TestPz", **kw)

    def test_one_recompute_per_action(self, session, counted):
        box = self._open(session)
        assert len(counted) == 1
        with contextlib.redirect_stdout(io.StringIO()):
            _find(box, "threshold").value = 0.7
        assert len(counted) == 2
        with contextlib.redirect_stdout(io.StringIO()):
            _find(box, "metric").value = "trial_sigma_px"  # resets the slider too
        assert len(counted) == 3
        with contextlib.redirect_stdout(io.StringIO()):
            _find(box, "view").value = "network"  # redraw only
        assert len(counted) == 3

    def test_metric_switch_resets_to_permissive(self, session):
        box = self._open(session)
        with contextlib.redirect_stdout(io.StringIO()):
            _find(box, "threshold").value = 0.9
            _find(box, "metric").value = "trial_sigma_px"
        thr = _find(box, "threshold")
        assert thr.value == pytest.approx(thr.max)  # lower is better → max passes all
        assert session._last_similarity_params["min_similarity"] is None

    def test_stash_follows_controls(self, session):
        box = self._open(session)
        with contextlib.redirect_stdout(io.StringIO()):
            _find(box, "cross-sensor only").value = True
            _find(box, "per image").value = 2
            _find(box, "max Δt").value = "1Y"
        p = session._last_similarity_params
        assert (
            p["cross_sensor_only"] and p["per_image"] == 2 and p["max_dt_days"] == 365
        )

    def test_bad_dt_shows_red_status_and_keeps_last(self, session):
        box = self._open(session, max_dt_days="1Y")
        with contextlib.redirect_stdout(io.StringIO()):
            _find(box, "max Δt").value = "banana"
        html = [w.value for w in _widgets(box) if type(w).__name__ == "HTML"]
        assert any("c0392b" in h for h in html)
        assert session._last_similarity_params["max_dt_days"] == 365

    def test_box_is_returned_not_displayed(self, session, monkeypatch):
        """Jupyter shows the returned box; displaying it too drew it twice."""
        import ipywidgets
        from IPython import display as ipy_display

        shown = []
        monkeypatch.setattr(ipy_display, "display", lambda *a, **k: shown.extend(a))
        box = self._open(session)
        assert isinstance(box, ipywidgets.VBox)
        assert not any(isinstance(o, ipywidgets.Widget) for o in shown)

    def test_save_button_writes_the_view(self, session, counted):
        plt.close("all")
        box = self._open(session, cross_sensor_only=True, per_image=1)
        n = len(counted)
        _find(box, "formats").value = ("html", "png")
        with contextlib.redirect_stdout(io.StringIO()):
            _find(box, "view").value = "network"
        _find(box, "Save figure").click()
        out = session.pz_dir("TestPz", "figures") / "similarity"
        stem = "TestPz_selection_network_score_cross_per1"
        assert (out / f"{stem}.html").exists() and (out / f"{stem}.png").exists()
        html = " ".join(w.value for w in _widgets(box) if type(w).__name__ == "HTML")
        assert "saved to" in html
        assert len(counted) == n  # saving does not recompute
        assert plt.get_fignums() == []

    def test_save_failure_is_shown_not_raised(self, session, monkeypatch):
        from geomulticorr.utils import _similarity_export

        def _boom(*a, **k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(_similarity_export, "save_similarity_figure", _boom)
        box = self._open(session)
        _find(box, "Save figure").click()
        html = " ".join(w.value for w in _widgets(box) if type(w).__name__ == "HTML")
        assert "save failed" in html and "disk full" in html


class TestSaveSimilarityFigure:
    def test_from_the_stash(self, session):
        plt.close("all")
        session.explore_similarity_pairs(
            "TestPz", interactive=False, cross_sensor_only=True, per_image=1
        )
        paths = session.save_similarity_figure("matrix", formats=("png", "svg"))
        assert set(paths) == {"png", "svg"}
        assert paths["png"].name == "TestPz_selection_matrix_score_cross_per1.png"
        assert paths["png"].parent == session.pz_dir("TestPz", "figures") / "similarity"
        assert plt.get_fignums() == []

    def test_explicit_selection(self, session):
        paths = session.save_similarity_figure(
            "network", "TestPz", formats="png", min_similarity=0.0
        )
        assert paths["png"].name == "TestPz_selection_network_score0.png"

    def test_needs_a_pzone(self, tmp_path):
        s = make_similarity_project(tmp_path)
        with pytest.raises(ValueError, match="pz_name"):
            s.save_similarity_figure("matrix")
