#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_thumbs_similarity.py
# creation date: 2026-10-30.
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
def _fake_correlate(session, couples, both_directions=False):
    """Commit the couples as pairs and write the stats JSON ASP extraction and
    corrections would leave, with CC decreasing along *couples*."""
    session.update_pairs(
        strategy="manual", couples=couples, both_directions=both_directions
    )
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


# --------------------------------------------------------------------------- #
# similarity after correlation (validation=True)
# --------------------------------------------------------------------------- #
def _write_asp_outputs(session, corrected=True):
    """What ASP and the corrections leave in each pair folder: a zero
    disparity ``-F.tif`` and an NCC map on the left thumb's grid, and (with
    *corrected*) small corrected EW/NS maps (0 is their no-data, so not 0)."""
    import rasterio

    from geomulticorr.similarity.raster import Grid

    root = session.pair_dir("TestPz")
    rng = np.random.default_rng(0)
    for p in session._pairs["pa_path"]:
        name = pathlib.Path(p).name
        pz, _, rest = name.partition("_")
        left = session.thumb_dir("TestPz") / f"{pz}_{rest.split('__')[0]}.tif"
        g = Grid.of(left)
        prof = dict(
            driver="GTiff",
            height=g.shape[0],
            width=g.shape[1],
            dtype="float32",
            crs=g.crs,
            transform=g.transform,
        )
        zeros = np.zeros(g.shape, "float32")
        with rasterio.open(root / name / f"{name}-F.tif", "w", count=3, **prof) as d:
            d.write(np.stack([zeros, zeros, zeros + 1]))
        with rasterio.open(
            root / name / f"{name}-F-ncc.tif", "w", count=1, nodata=-32768, **prof
        ) as d:
            d.write(zeros + 0.8, 1)
        if corrected:
            for comp in ("EW", "NS"):
                with rasterio.open(
                    root / name / f"{name}-F_{comp}_corr.tif",
                    "w",
                    count=1,
                    nodata=0,
                    **prof,
                ) as d:
                    d.write((0.01 + 0.001 * rng.random(g.shape)).astype("float32"), 1)


def _pair_jsons(session):
    root = session.pair_dir("TestPz")
    names = [pathlib.Path(p).name for p in session._pairs["pa_path"]]
    return {n: root / n / f"{n}.json" for n in names}


@pytest.fixture
def correlated(session):
    sim = session.compute_thumbs_similarity("TestPz", **KW)
    _fake_correlate(session, sim.sort_values("score", ascending=False))
    _write_asp_outputs(session)
    return session


def _count_after(monkeypatch):
    from geomulticorr.similarity import raster as sim_raster

    calls = []
    real = sim_raster.compare_after_correlation
    monkeypatch.setattr(
        sim_raster,
        "compare_after_correlation",
        lambda *a, **k: calls.append(a[0]) or real(*a, **k),
    )
    return calls


def test_validation_false_is_unchanged(correlated):
    matched, table = correlated.similarity_validation("TestPz")
    assert not any(c.endswith(("_raw", "_corr")) for c in matched.columns)
    assert "before_vs_raw" not in table.columns
    for path in _pair_jsons(correlated).values():
        assert "similarity" not in _strict_json(path)


def test_validation_writes_before_and_after_into_pair_json(correlated):
    matched, table = correlated.similarity_validation(
        "TestPz", validation=True, progress=False
    )
    for path in _pair_jsons(correlated).values():
        js = _strict_json(path)  # NaN written as null
        assert js["raw_corr_stats"]["cc"]["median"] > 0  # other sections intact
        block = js["similarity"]
        assert set(block) == {
            "params",
            "couple",
            "orientation",
            "before",
            "after",
            "written_utc",
        }
        assert "params_hash" not in block and "params_hash" not in block["params"]
        assert block["params"]["kernel"] == 21
        assert block["orientation"] == "forward"
        assert set(block["after"]) == {"raw", "corrected"}
        raw, corr = block["after"]["raw"], block["after"]["corrected"]
        assert set(raw["indices"]) == set(REPORT_KEYS)
        assert raw["ncc_median"] == pytest.approx(0.8)
        assert corr["nmad_m"] == pytest.approx(
            np.hypot(corr["nmad_ew_m"], corr["nmad_ns_m"])
        )
        assert "score" in block["before"]
    for col in ("trial_peak_median_raw", "coh_mean_corr", "ncc_median", "nmad_corr_px"):
        assert matched[col].notna().all(), col
    assert (matched["asp_pairs"] == 1).all()
    assert {"before_vs_raw", "before_vs_corr", "ncc_median"} <= set(table.columns)
    assert "trial_peak_median_raw" in table.index


def test_zero_disparity_reproduces_before(correlated):
    """Same thumbs, same grid, same mask and no shift: after == before."""
    correlated.similarity_validation("TestPz", validation=True, progress=False)
    checked = 0
    for path in _pair_jsons(correlated).values():
        block = _strict_json(path)["similarity"]
        if "spot" in "".join(block["couple"]):
            continue  # cross-resolution: before ran on the coarser grid
        checked += 1
        before, after = block["before"], block["after"]["raw"]["indices"]
        for key in REPORT_KEYS:
            b = np.nan if before[key] is None else before[key]
            a = np.nan if after[key] is None else after[key]
            assert np.isclose(a, b, equal_nan=True), key
    assert checked == 3  # the three PlanetScope couples


def test_second_call_reuses_and_a_touched_file_recomputes(correlated, monkeypatch):
    import os

    calls = _count_after(monkeypatch)
    correlated.similarity_validation("TestPz", validation=True, progress=False)
    n_pairs = len(_pair_jsons(correlated))
    assert len(calls) == 2 * n_pairs  # raw + corrected per pair
    calls.clear()
    m2, _ = correlated.similarity_validation("TestPz", validation=True, progress=False)
    assert calls == [] and m2["trial_peak_median_raw"].notna().all()
    name = next(iter(_pair_jsons(correlated)))
    ew = correlated.pair_dir("TestPz") / name / f"{name}-F_EW_corr.tif"
    st = ew.stat()
    os.utime(ew, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    correlated.similarity_validation("TestPz", validation=True, progress=False)
    assert len(calls) == 2


def test_without_corrections_corrected_is_null(session):
    sim = session.compute_thumbs_similarity("TestPz", **KW)
    _fake_correlate(session, sim.sort_values("score", ascending=False))
    _write_asp_outputs(session, corrected=False)
    matched, _ = session.similarity_validation(
        "TestPz", validation=True, progress=False
    )
    for path in _pair_jsons(session).values():
        assert _strict_json(path)["similarity"]["after"]["corrected"] is None
    assert "trial_peak_median_raw" in matched
    assert not any(c.endswith("_corr") for c in matched.columns)


def test_missing_disparity_is_skipped(correlated):
    jsons = _pair_jsons(correlated)
    name = next(iter(jsons))
    (correlated.pair_dir("TestPz") / name / f"{name}-F.tif").unlink()
    matched, _ = correlated.similarity_validation(
        "TestPz", validation=True, progress=False
    )
    assert "similarity" not in _strict_json(jsons[name])
    assert (
        sum("similarity" in _strict_json(p) for n, p in jsons.items() if n != name)
        == len(jsons) - 1
    )
    assert matched["trial_peak_median_raw"].isna().sum() == 1


def test_write_failure_never_raises(correlated, monkeypatch):
    from geomulticorr.stats import stats as gmc_stats

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(gmc_stats, "update_stats_file", _boom)
    matched, _ = correlated.similarity_validation(
        "TestPz", validation=True, progress=False
    )
    assert matched["trial_peak_median_raw"].notna().all()  # still returned


def test_forward_and_backward_pairs_are_averaged(session):
    sim = session.compute_thumbs_similarity("TestPz", **KW)
    _fake_correlate(
        session, sim.sort_values("score", ascending=False), both_directions=True
    )
    _write_asp_outputs(session)
    matched, _ = session.similarity_validation(
        "TestPz", validation=True, progress=False
    )
    assert len(matched) == 6 and (matched["asp_pairs"] == 2).all()
    orients = {
        _strict_json(p)["similarity"]["orientation"]
        for p in _pair_jsons(session).values()
    }
    assert orients == {"forward", "backward"}


def test_real_pair_fit_masks_are_named_stable(correlated):
    """The fit masks are -F_EW_stable.tif / -F_NS_stable.tif (not the old *mask names)."""
    pair = correlated.get_pairs()[0]
    assert pair.pa_ew_mask_path.name == f"{pair.pa_key}-F_EW_stable.tif"
    assert pair.pa_ns_mask_path.name == f"{pair.pa_key}-F_NS_stable.tif"
