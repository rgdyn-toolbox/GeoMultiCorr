"""Explorer views and static overview figures for image similarity."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from geomulticorr.similarity.selection import select_couples
from geomulticorr.utils import _similarity_plotly as sp
from geomulticorr.utils import gmc_functions as gf


def _sim():
    keys = [
        ("PZ_2018-01-01_spot6", "2018-01-01", "spot6"),
        ("PZ_2018-06-01_planetscope", "2018-06-01", "planetscope"),
        ("PZ_2019-06-01_planetscope", "2019-06-01", "planetscope"),
        ("PZ_2020-06-01_spot7", "2020-06-01", "spot7"),
    ]
    rows = []
    rng = np.random.default_rng(0)
    for i in range(4):
        for j in range(i + 1, 4):
            rows.append(
                dict(
                    left_key=keys[i][0],
                    right_key=keys[j][0],
                    left_date=keys[i][1],
                    right_date=keys[j][1],
                    left_sensor=keys[i][2],
                    right_sensor=keys[j][2],
                    dt_days=365 * (j - i),
                    score=rng.uniform(),
                    trial_valid_frac=rng.uniform(0.8, 1),
                    trial_peak_median=rng.uniform(0.5, 0.9),
                    trial_sigma_px=rng.uniform(0.3, 1.2),
                    coh_mean=rng.uniform(0.1, 0.4),
                )
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("view", list(sp.SIMILARITY_VIEWS))
@pytest.mark.parametrize("selected", ["some", "none", "empty"])
def test_views_never_raise(view, selected):
    sim = _sim()
    sel = {
        "some": select_couples(sim, threshold=0.3),
        "none": None,
        "empty": sim.iloc[0:0],
    }[selected]
    assert isinstance(sp.figure_similarity_view(view, sim, sel), go.Figure)


def test_only_matrix_and_network():
    assert list(sp.SIMILARITY_VIEWS) == ["matrix", "network"]
    assert not hasattr(sp, "figure_selection_table")


def test_view_survives_garbage():
    assert isinstance(
        sp.figure_similarity_view("matrix", pd.DataFrame(), None), go.Figure
    )
    assert isinstance(sp.figure_similarity_view("nope", _sim(), None), go.Figure)
    assert isinstance(
        sp.figure_similarity_matrix(_sim(), None, metric="absent"), go.Figure
    )


def test_matrix_outlines_selection_symmetrically():
    sim = _sim()
    sel = sim.head(2)
    fig = sp.figure_similarity_matrix(sim, sel)
    assert len(fig.data) == 2 and len(fig.data[1].x) == 4


def test_selection_pairs_frame_directions():
    sel = _sim().head(3)
    assert len(sp.selection_pairs_frame(sel, both_directions=False)) == 3
    f = sp.selection_pairs_frame(sel, both_directions=True)
    assert len(f) == 6 and set(f.direction) == {"forward", "backward"}
    assert sp.selection_pairs_frame(sel.iloc[0:0]).empty


def test_static_figures_close_cleanly():
    plt.close("all")
    sim = _sim()
    for fig in (gf.plot_similarity_matrices(sim), gf.plot_similarity_by_sensor(sim)):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    table = pd.DataFrame({"cc_median": [0.5, np.nan]}, index=["score", "dt_days"])
    fig, ax = gf.plot_similarity_validation(table, n=12)
    assert "12 couples" in ax.get_title()
    plt.close(fig)
    assert plt.get_fignums() == []


def test_matrix_figure_has_one_panel_per_metric():
    fig = gf.plot_similarity_matrices(
        _sim(), metrics={"score": "viridis", "absent": "x"}
    )
    assert len([a for a in fig.axes if a.get_title()]) == 1
    plt.close(fig)
