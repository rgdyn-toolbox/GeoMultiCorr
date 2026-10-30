"""Session.compute_thumbs_similarity on a real (temporary) GMC project:
the JSON trace, the stored score, and the overview figures."""

from __future__ import annotations

import json
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geomulticorr.core.session import SIMILARITY_ID_COLUMNS
from geomulticorr.similarity.raster import REPORT_KEYS
from tests.similarity.conftest import (
    EPSG,
    ORIGIN,
    PS_OTHER,
    PS_SAME_1,
    PS_SAME_2,
    SPOT,
    make_similarity_project,
)

KW = dict(kernel=21, search=4, tile=32, progress=False, plots=False)


@pytest.fixture
def session(tmp_path):
    return make_similarity_project(tmp_path)


def _strict_json(path):
    """Parse refusing NaN/Infinity, which are not JSON."""

    def _bad(token):
        raise ValueError(f"non-JSON constant {token}")

    return json.loads(pathlib.Path(path).read_text(), parse_constant=_bad)


def test_all_couples_columns_and_order(session):
    f = session.compute_thumbs_similarity("TestPz", **KW)
    assert len(f) == 6
    assert list(f.columns) == list(SIMILARITY_ID_COLUMNS) + list(REPORT_KEYS) + [
        "score"
    ]
    assert (f["left_date"] < f["right_date"]).all()
    assert f["cross_sensor"].dtype == bool and f["cross_sensor"].sum() == 3
    assert f["score"].between(0, 1).all()


def test_ranks_same_ground_above_other_ground(session):
    f = session.compute_thumbs_similarity("TestPz", **KW).set_index("right_key")
    spot_rows = f[f["left_key"] == SPOT]
    assert (
        spot_rows.loc[PS_SAME_1, "trial_valid_frac"]
        > 0.9
        > spot_rows.loc[PS_OTHER, "trial_valid_frac"]
    )
    assert spot_rows.loc[PS_SAME_1, "score"] > spot_rows.loc[PS_OTHER, "score"]
    assert spot_rows.loc[PS_SAME_1, "res"] == 3.0


def test_cross_sensor_is_by_family(session, tmp_path):
    """spot6 ↔ spot7 is not cross-sensor."""
    from tests.similarity.conftest import scene, write_tif

    spot, _ = scene(seed=0)
    write_tif(session.thumb_dir("TestPz") / "TestPz_2023-07-01_spot7.tif", spot, 1.5)
    session.update_thumbs()
    f = session.compute_thumbs_similarity("TestPz", **KW)
    row = f[(f.left_key == SPOT) & (f.right_key == "TestPz_2023-07-01_spot7")]
    assert not row["cross_sensor"].item()
    g = session.compute_thumbs_similarity("TestPz", cross_sensor_only=True, **KW)
    assert len(g) == 3 * 2  # two SPOT images × three PlanetScope


def test_cross_sensor_only_and_strategy(session):
    f = session.compute_thumbs_similarity("TestPz", cross_sensor_only=True, **KW)
    assert len(f) == 3 and f["cross_sensor"].all()
    g = session.compute_thumbs_similarity("TestPz", couples="consecutive", **KW)
    assert len(g) == 3


class TestTrace:
    def test_json_written_and_strict(self, session):
        f = session.compute_thumbs_similarity("TestPz", **KW)
        path = session.similarity_path("TestPz")
        assert path.suffix == ".json" and path.exists()
        doc = _strict_json(path)
        assert doc["format"] == 1 and doc["pz_name"] == "TestPz"
        (run,) = doc["runs"].values()
        assert len(run["couples"]) == 6 and "kernel" in run["params"]
        assert run["score_columns"]["trial_sigma_px"] is False
        assert f["params_hash"].iloc[0] in doc["runs"]

    def test_nan_round_trips_as_null(self, session):
        f = session.compute_thumbs_similarity("TestPz", **KW)
        doc = _strict_json(session.similarity_path("TestPz"))
        (run,) = doc["runs"].values()
        nulls = [k for c in run["couples"] for k, v in c.items() if v is None]
        back = session.load_thumbs_similarity("TestPz")
        assert back[nulls[0]].isna().any() if nulls else True
        assert f["coh_cutoff_m"].isna().sum() == back["coh_cutoff_m"].isna().sum()

    def test_load_matches_compute(self, session):
        f = session.compute_thumbs_similarity("TestPz", **KW)
        back = session.load_thumbs_similarity("TestPz")
        pd.testing.assert_frame_equal(
            f.reset_index(drop=True),
            back[f.columns].reset_index(drop=True),
            check_dtype=False,
        )

    def test_reused_then_new_run_on_new_params(self, session):
        first = session.compute_thumbs_similarity("TestPz", **KW)
        again = session.compute_thumbs_similarity("TestPz", **KW)
        pd.testing.assert_frame_equal(first, again, check_dtype=False)
        changed = session.compute_thumbs_similarity("TestPz", min_peak=0.6, **KW)
        assert changed["params_hash"].iloc[0] != first["params_hash"].iloc[0]
        doc = _strict_json(session.similarity_path("TestPz"))
        assert set(doc["runs"]) == {
            first["params_hash"].iloc[0],
            changed["params_hash"].iloc[0],
        }
        # explicit run, and latest by default
        old = session.load_thumbs_similarity(
            "TestPz", params_hash=first["params_hash"].iloc[0]
        )
        assert len(old) == 6
        latest = session.load_thumbs_similarity("TestPz")
        assert latest["params_hash"].iloc[0] == changed["params_hash"].iloc[0]

    def test_score_rescored_when_couples_are_added(self, session):
        """A percentile shifts as the run grows; the stored score follows."""
        sub = session.compute_thumbs_similarity(
            "TestPz", couples="consecutive", **KW
        ).set_index(["left_key", "right_key"])
        full = session.compute_thumbs_similarity("TestPz", **KW).set_index(
            ["left_key", "right_key"]
        )
        stored = session.load_thumbs_similarity("TestPz").set_index(
            ["left_key", "right_key"]
        )
        assert len(stored) == 6
        assert not np.allclose(
            sub["score"].to_numpy(), full.loc[sub.index, "score"].to_numpy()
        )
        np.testing.assert_allclose(
            full["score"].to_numpy(), stored.loc[full.index, "score"].to_numpy()
        )

    def test_trace_lives_in_the_similarity_folder(self, session):
        session.compute_thumbs_similarity("TestPz", **KW)
        path = session.similarity_path("TestPz")
        assert path == session.pz_dir("TestPz") / "similarity" / path.name
        assert path.exists()

    def test_trace_in_image_correlation_is_moved_once(self, session):
        """Projects measured before the similarity/ folder keep their runs."""
        first = session.compute_thumbs_similarity("TestPz", **KW)
        new = session.similarity_path("TestPz")
        old = session.pz_dir("TestPz", "image_correlation") / new.name
        new.replace(old)
        back = session.load_thumbs_similarity("TestPz")
        assert new.exists() and not old.exists()
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True),
            back[first.columns].reset_index(drop=True),
            check_dtype=False,
        )

    def test_old_csv_ignored(self, session, caplog):
        csv = (
            session.pz_dir("TestPz", "image_correlation")
            / session.similarity_path("TestPz").with_suffix(".csv").name
        )
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text("left_key,right_key\nx,y\n")
        f = session.compute_thumbs_similarity("TestPz", **KW)
        assert len(f) == 6 and session.similarity_path("TestPz").exists()

    def test_write_failure_is_logged_not_raised(self, session, monkeypatch):
        from geomulticorr.similarity import trace

        monkeypatch.setattr(trace, "write_trace", lambda *a, **k: None)
        f = session.compute_thumbs_similarity("TestPz", **KW)
        assert len(f) == 6 and not session.similarity_path("TestPz").exists()

    def test_load_errors(self, session):
        with pytest.raises(FileNotFoundError, match="compute_thumbs_similarity"):
            session.load_thumbs_similarity("TestPz")
        session.compute_thumbs_similarity("TestPz", **KW)
        with pytest.raises(KeyError):
            session.load_thumbs_similarity("TestPz", params_hash="nope")


def test_moving_areas_auto_used_when_loaded(session):
    before = session.compute_thumbs_similarity("TestPz", save=False, **KW)
    session.build_reference_grid("TestPz", resolution=3.0)
    poly = box(ORIGIN[0], ORIGIN[1] - 150, ORIGIN[0] + 150, ORIGIN[1])
    session.load_moving_areas("TestPz", gpd.GeoDataFrame(geometry=[poly], crs=EPSG))
    after = session.compute_thumbs_similarity("TestPz", save=False, **KW)
    assert (after["n_pixels"] < before["n_pixels"]).all()
    assert (after["params_hash"] != before["params_hash"]).all()


def test_reference_grid(session):
    session.build_reference_grid("TestPz", resolution=6.0)
    f = session.compute_thumbs_similarity("TestPz", grid="reference", save=False, **KW)
    assert (f["res"] == 6.0).all()


def test_errors(session):
    with pytest.raises(ValueError, match="grid"):
        session.compute_thumbs_similarity("TestPz", grid="finest", **KW)
    with pytest.raises(ValueError, match="at least two"):
        session.compute_thumbs_similarity("TestPz", sensor_filter="spot", **KW)
    with pytest.raises(TypeError, match="unknown"):
        session.compute_thumbs_similarity("TestPz", kernal=21, progress=False)


def test_resolves_thumbs_through_layout_not_stored_path(session):
    """A moved project (stale th_path) still finds its thumbs."""
    session._thumbs["th_path"] = [
        "/somewhere/else/" + p.split("/")[-1] for p in session._thumbs["th_path"]
    ]
    f = session.compute_thumbs_similarity("TestPz", save=False, **KW)
    assert len(f) == 6


# --------------------------------------------------------------------------- #
# overview figures and validation
# --------------------------------------------------------------------------- #
def _fake_correlate(session, couples):
    """Commit the couples as pairs and write the stats JSON ASP extraction and
    corrections would leave, with CC decreasing along *couples*."""
    session.update_pairs(strategy="manual", couples=couples, both_directions=False)
    root = session.pair_dir("TestPz")
    for k, p in enumerate(session._pairs["pa_path"]):
        name = pathlib.Path(p).name
        (root / name).mkdir(parents=True, exist_ok=True)
        stats = {
            "raw_corr_stats": {
                "cc": {"median": 0.9 - 0.1 * k, "cc_quality_gte_050": 0.95 - 0.1 * k}
            },
            "final_corrected_stats": {
                "ew": {"nmad": 0.5 + 0.2 * k},
                "ns": {"nmad": 0.5 + 0.2 * k},
            },
        }
        (root / name / f"{name}.json").write_text(json.dumps(stats))


def test_figures_written_without_leak(session):
    plt.close("all")
    session.compute_thumbs_similarity("TestPz", **dict(KW, plots=True))
    out = session.pz_dir("TestPz", "figures") / "similarity"
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "TestPz_similarity_by_sensor.png",
        "TestPz_similarity_matrices.png",
    ]
    assert plt.get_fignums() == []


def test_validation_figure_once_pairs_are_correlated(session):
    sim = session.compute_thumbs_similarity("TestPz", **KW)
    _fake_correlate(session, sim.sort_values("score", ascending=False))
    outcomes = session._similarity_outcomes("TestPz")
    assert len(outcomes) == 6 and outcomes["nmad_m"].notna().all()
    session.compute_thumbs_similarity(
        "TestPz", **dict(KW, plots=True, formats=("png", "svg"))
    )
    out = session.pz_dir("TestPz", "figures") / "similarity"
    assert (out / "TestPz_similarity_validation.png").exists()
    assert (out / "TestPz_similarity_validation.svg").exists()
    assert plt.get_fignums() == []


def test_outcomes_empty_without_pairs(session):
    assert session._similarity_outcomes("TestPz").empty


def test_figure_failure_never_raises(session, monkeypatch):
    from geomulticorr.utils import gmc_functions

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(gmc_functions, "plot_similarity_matrices", _boom)
    f = session.compute_thumbs_similarity("TestPz", **dict(KW, plots=True))
    assert len(f) == 6
    out = session.pz_dir("TestPz", "figures") / "similarity"
    assert not (out / "TestPz_similarity_matrices.png").exists()
    assert plt.get_fignums() == []


def test_similarity_validation_public(session):
    sim = session.compute_thumbs_similarity("TestPz", **KW)
    matched, table = session.similarity_validation("TestPz")
    assert matched.empty and table.empty
    _fake_correlate(session, sim.sort_values("score", ascending=False))
    matched, table = session.similarity_validation("TestPz")
    assert len(matched) == 6 and "nmad_px" in matched
    # CC was written roughly decreasing with score (pairs-layer order) → positive ρ
    assert table.loc["score", "cc_median"] > 0.5
