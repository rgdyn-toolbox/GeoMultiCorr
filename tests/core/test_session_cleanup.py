#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_session_cleanup.py
# creation date: 2026-11-18.
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
"""Session behaviour pinned by the first cleanup of ``session.py``.

``launch_pairs_correlation`` runs on a stub session, with ``subprocess.run`` and
``ASP`` replaced, so no test needs an ASP install.
"""

from __future__ import annotations

import inspect
import types

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

import geomulticorr.correlation as gmc_correlation
from geomulticorr import open_gmc_session
from geomulticorr.core import session as session_mod
from geomulticorr.core.session import Session
from geomulticorr.correlation import correlation as correlation_mod
from geomulticorr.correlation.correlation import ASP
from geomulticorr.corrections import CCFilter, OutlierFilter, StableAreaMask

KEY = "Zone_2020-08-15_sentinel2__2021-08-15_sentinel2"

#: What a real ``parallel_stereo`` run of ASP 3.6.0 left beside ``_KEEP_SUFFIXES``.
ASP_3_6_INTERMEDIATES = (
    "-D.tif",
    "-D_sub.tif",
    "-GoodPixelMap.tif",
    "-L.vwip",
    "-L__R.match",
    "-L_sub.tif",
    "-R.vwip",
    "-R_sub.tif",
    "-Zone_2020-08-15_sentinel2-stats.tif",
    "-dirList.txt",
    "-info.txt",
    "-lMask.tif",
    "-lMask_sub.tif",
    "-log-stereo_corr-11-04-1245-84413.txt",
    "-rMask.tif",
    "-rMask_sub.tif",
    "-stereo-status.txt",
    "-stereo.default",
    "-tiles-index.txt",
    "-tiles.dbf",
    "-tiles.qml",
    "-tiles.shp",
    "-tiles.shx",
)

#: What GMC writes into a pair folder after correlation.
GMC_PRODUCTS = (
    "-F_EW.tif",
    "-F_NS.tif",
    "-F_CC.tif",
    "-F_EW_corr.tif",
    "-F_NS_corr.tif",
    "-F_EW_stable.tif",
    "-F_NS_stable.tif",
    ".json",
    "_raw_disp.jpg",
    "_00_MedianCentering_disp.jpg",
    "_corrections_disp.jpg",
    "_magn.tif",
    "_vect.gpkg",
)


class _FakeASP:
    """Records cleaning; building it fails, as on a machine without ASP."""

    cleaned: list = []

    def __init__(self, session=None, asp_bin_dir=None):
        raise FileNotFoundError("Unable to find ASP executable 'parallel_stereo'.")

    @classmethod
    def clean_pair_directory(cls, pair):
        cls.cleaned.append(pair.pa_key)
        return []


@pytest.fixture
def launch(tmp_path, monkeypatch):
    """Call ``launch_pairs_correlation`` on one prepared pair; expose what it did."""
    (tmp_path / "k_CorrelationJob.sh").write_text("echo hi\n")
    pair = types.SimpleNamespace(pa_key="k", pa_path=tmp_path)
    state = types.SimpleNamespace(commands=[], synced=[], returncode=0, script=None)
    state.script = tmp_path / "k_CorrelationJob.sh"
    stub = types.SimpleNamespace(
        get_pairs=lambda *a, **k: [pair],
        _sync_pairs_status=lambda pairs: state.synced.extend(pairs),
    )

    def fake_run(cmd, **kwargs):
        state.commands.append(cmd)
        return types.SimpleNamespace(returncode=state.returncode, stderr="boom")

    _FakeASP.cleaned = []
    monkeypatch.setattr(session_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(gmc_correlation, "ASP", _FakeASP)

    def _call(**kwargs):
        kwargs.setdefault("print_summary", False)
        state.codes = Session.launch_pairs_correlation(stub, **kwargs)
        return state

    return _call


class TestLaunchDryRun:
    def test_logs_the_local_command_and_runs_nothing(self, launch, caplog_gmc):
        state = launch(dry_run=True)
        assert state.codes == [0]
        assert state.commands == []
        assert state.synced == []
        assert f"DRY RUN: bash {state.script}" in caplog_gmc.text

    def test_logs_the_cluster_command(self, launch, caplog_gmc):
        state = launch(dry_run=True, cluster="isterre")
        assert state.commands == []
        assert f"DRY RUN: oarsub -S {state.script.resolve()}" in caplog_gmc.text

    @pytest.mark.parametrize("cluster", [None, "isterre"])
    def test_works_without_asp_installed(self, launch, cluster):
        assert launch(dry_run=True, cluster=cluster).codes == [0]
        assert _FakeASP.cleaned == []


class TestLaunchReal:
    def test_local_run_cleans_and_syncs(self, launch):
        state = launch()
        assert state.commands == [["bash", str(state.script)]]
        assert state.codes == [0]
        assert _FakeASP.cleaned == ["k"]
        assert len(state.synced) == 1

    def test_clean_false_does_not_clean(self, launch):
        assert launch(clean=False).codes == [0]
        assert _FakeASP.cleaned == []

    def test_failed_script_is_not_cleaned(self, launch):
        launch(dry_run=True).returncode = 3
        state = launch()
        assert state.codes == [3]
        assert _FakeASP.cleaned == []

    def test_cluster_submission_does_not_clean_or_sync(self, launch):
        state = launch(cluster="isterre")
        assert state.commands == [["oarsub", "-S", str(state.script.resolve())]]
        assert _FakeASP.cleaned == []
        assert state.synced == []


class TestSyncPairsAfterCluster:
    """Cleaning must never delete GMC's products, and needs no ASP install."""

    @pytest.fixture
    def pair_dir(self, tmp_path, monkeypatch):
        # Nothing here runs ASP, so a machine without it must work.
        monkeypatch.setattr(correlation_mod.shutil, "which", lambda *a, **k: None)
        for suffix in (*ASP_3_6_INTERMEDIATES, *GMC_PRODUCTS, *ASP._KEEP_SUFFIXES):
            (tmp_path / f"{KEY}{suffix}").write_text("x")
        return tmp_path

    @staticmethod
    def _sync(pair_dir, status="complete", **kwargs):
        pair = types.SimpleNamespace(
            pa_key=KEY, pa_path=pair_dir, get_status=lambda: status
        )
        synced = []
        stub = types.SimpleNamespace(
            get_pairs=lambda *a, **k: [pair],
            _sync_pairs_status=lambda pairs: synced.extend(pairs),
        )
        Session.sync_pairs_after_cluster(stub, print_summary=False, **kwargs)
        return synced

    def test_keeps_what_later_steps_produced(self, pair_dir):
        self._sync(pair_dir)
        for suffix in (*GMC_PRODUCTS, *ASP._KEEP_SUFFIXES):
            assert (pair_dir / f"{KEY}{suffix}").exists(), suffix

    def test_removes_the_asp_intermediates(self, pair_dir):
        self._sync(pair_dir)
        for suffix in ASP_3_6_INTERMEDIATES:
            assert not (pair_dir / f"{KEY}{suffix}").exists(), suffix

    def test_clean_false_removes_nothing_and_still_syncs(self, pair_dir):
        before = sorted(pair_dir.iterdir())
        assert len(self._sync(pair_dir, clean=False)) == 1
        assert sorted(pair_dir.iterdir()) == before

    def test_incomplete_pair_is_not_cleaned(self, pair_dir):
        before = sorted(pair_dir.iterdir())
        assert len(self._sync(pair_dir, status="corrupt")) == 1
        assert sorted(pair_dir.iterdir()) == before


class TestIntermediateFiles:
    def test_works_on_the_class_without_asp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(correlation_mod.shutil, "which", lambda *a, **k: None)
        (tmp_path / f"{KEY}-D.tif").write_text("x")
        pair = types.SimpleNamespace(pa_key=KEY, pa_path=tmp_path)
        assert ASP.clean_pair_directory(pair) == [tmp_path / f"{KEY}-D.tif"]

    def test_log_in_the_pzone_name_protects_the_products(self, tmp_path):
        key = "Zone-log-a_2020-08-15_s__2021-08-15_s"
        for name in (f"{key}-F_EW.tif", f"{key}.json", f"{key}-log-stereo_corr-1.txt"):
            (tmp_path / name).write_text("x")
        assert ASP._intermediate_files(tmp_path, key) == [
            tmp_path / f"{key}-log-stereo_corr-1.txt"
        ]

    def test_other_prefixes_and_symlinks_are_left_alone(self, tmp_path):
        (tmp_path / "other-D.tif").write_text("x")
        (tmp_path / "target").write_text("x")
        (tmp_path / f"{KEY}-D.tif").symlink_to(tmp_path / "target")
        assert ASP._intermediate_files(tmp_path, KEY) == []


class TestDeprecatedMethods:
    def test_insert_pzone_to_gdb_warns_and_still_writes(self, tmp_path):
        session = open_gmc_session(tmp_path / "proj", epsg_code=32632)
        with pytest.warns(DeprecationWarning, match="insert_pzone"):
            session.insert_pzone_to_gdb(box(0, 0, 10, 10), "Zone", "Z")
        stored = gpd.read_file(session.path_geodb, layer="Pzones", engine="pyogrio")
        assert "Zone" in set(stored.pz_name)


def test_map_georasters_bank_signature():
    params = inspect.signature(Session.map_georasters_bank).parameters
    assert "save_georaster_bank" not in params
    assert "exclude_patterns" in params


class TestApplyCorrectionsFilterArgument:
    """The per-pair loop swallows exceptions, so these assert on what it returns."""

    @staticmethod
    def _run(tmp_path, filter_pipeline):
        transform = rasterio.transform.from_origin(0, 100, 10, 10)
        rng = np.random.default_rng(0)
        paths = {}
        for name, data in (
            ("EW", rng.normal(1.0, 0.1, (10, 10))),
            ("NS", rng.normal(-1.0, 0.1, (10, 10))),
            ("ncc", np.full((10, 10), 0.9)),
        ):
            paths[name] = tmp_path / f"k-F_{name}.tif"
            with rasterio.open(
                paths[name],
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="float32",
                crs="EPSG:32632",
                transform=transform,
            ) as dst:
                dst.write(data.astype("float32"), 1)
        pair = types.SimpleNamespace(
            pa_key="k",
            pa_path=tmp_path,
            pa_ew_path=paths["EW"],
            pa_ns_path=paths["NS"],
            pa_cc_path=paths["ncc"],
            pa_ew_mask_path=tmp_path / "k-F_EW_stable.tif",
            pa_ns_mask_path=tmp_path / "k-F_NS_stable.tif",
            pa_stats_path=tmp_path / "k.json",
            save_correction_masks=lambda x, y, reference: {},
        )
        stub = types.SimpleNamespace(get_pairs=lambda *a, **k: [pair])
        return Session.apply_pairs_corrections(
            stub, filter_pipeline=filter_pipeline, save_plot=False, print_summary=False
        )["k"]

    def test_filter_pipeline_writes_the_corrected_maps(self, tmp_path):
        result = self._run(tmp_path, CCFilter(0.3) + OutlierFilter((-10, 10)))
        assert result is not None
        assert result["ew"].exists() and result["ns"].exists()

    def test_single_mask_writes_the_corrected_maps(self, tmp_path):
        assert self._run(tmp_path, CCFilter(0.3)) is not None


class TestApplyCorrectionsMovingGround:
    """Corrections are fitted on stable ground; moving ground stays in the output."""

    STABLE_VALUE, MOVING_VALUE = 1.0, 5.0

    @classmethod
    def _setup(cls, tmp_path, with_moving_file):
        import geopandas as gpd
        import geoutils as gu
        from shapely.geometry import box

        transform = rasterio.transform.from_origin(0, 100, 10, 10)
        ew = np.full((10, 10), cls.STABLE_VALUE)
        ew[:6] = cls.MOVING_VALUE  # 60 % of the scene moves: the global median lies
        # a small gradient, so a centred stable pixel is not exactly 0 (= nodata)
        ew[6:] += np.linspace(-0.2, 0.2, 40).reshape(4, 10)
        paths = {}
        for name, data in (
            ("EW", ew),
            ("NS", np.zeros((10, 10)) + 0.5),
            ("ncc", np.full((10, 10), 0.9)),
        ):
            paths[name] = tmp_path / f"k-F_{name}.tif"
            with rasterio.open(
                paths[name],
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="float32",
                crs="EPSG:32632",
                transform=transform,
            ) as dst:
                dst.write(data.astype("float32"), 1)
        written = {}
        pair = types.SimpleNamespace(
            pa_key="k",
            pa_pz_name="Z",
            pa_path=tmp_path,
            pa_ew_path=paths["EW"],
            pa_ns_path=paths["NS"],
            pa_cc_path=paths["ncc"],
            pa_ew_mask_path=tmp_path / "k-F_EW_stable.tif",
            pa_ns_mask_path=tmp_path / "k-F_NS_stable.tif",
            pa_stats_path=tmp_path / "k.json",
            save_correction_masks=lambda x, y, reference: written.update(x=x, y=y)
            or {},
        )
        moving_file = tmp_path / "moving.gpkg"
        if with_moving_file:  # the top 6 rows: y from 40 to 100
            gpd.GeoDataFrame(geometry=[box(0, 40, 100, 100)], crs="EPSG:32632").to_file(
                moving_file, layer="Z_moving-areas"
            )
        stub = types.SimpleNamespace(
            get_pairs=lambda *a, **k: [pair],
            moving_areas_path=lambda pz: moving_file,
            get_moving_areas=lambda pz: gu.Vector(
                gpd.read_file(moving_file, layer="Z_moving-areas")
            ),
        )
        stub._with_moving_areas = lambda *a: Session._with_moving_areas(stub, *a)
        return stub, pair, written

    @classmethod
    def _run(cls, tmp_path, monkeypatch, with_moving_file=True, **kwargs):
        import geomulticorr.stats.stats as gmc_stats
        from geomulticorr.corrections import MedianCentering

        stub, pair, written = cls._setup(tmp_path, with_moving_file)
        # the stats JSON needs a whole Pair; it is covered by its own tests
        monkeypatch.setattr(gmc_stats, "save_corrected_stats", lambda *a, **k: {})
        monkeypatch.setattr(gmc_stats, "save_final_corrected_stats", lambda *a, **k: {})
        monkeypatch.setattr(
            gmc_stats,
            "save_stable_stats",
            lambda *a, **k: {"raw_corr_stats_stable": {}},
        )
        result = Session.apply_pairs_corrections(
            stub,
            MedianCentering(),
            save_plot=False,
            print_summary=False,
            sync_geodb=False,
            overwrite=True,
            **kwargs,
        )["k"]
        corrected = rasterio.open(result["ew"]).read(1) if result else None
        return result, corrected, written

    def test_auto_fits_outside_moving_areas_and_keeps_them(self, tmp_path, monkeypatch):
        result, corrected, written = self._run(tmp_path, monkeypatch)
        assert result is not None
        # stable pixels centred to zero, moving ones keep their signal (5 - 1)
        assert np.allclose(
            corrected[6:], np.linspace(-0.2, 0.2, 40).reshape(4, 10), atol=1e-5
        )
        assert np.allclose(corrected[:6], self.MOVING_VALUE - self.STABLE_VALUE)
        # the persisted mask is the fit mask: moving ground excluded
        assert written["x"][6:].all() and not written["x"][:6].any()

    def test_none_fits_on_all_ground(self, tmp_path, monkeypatch):
        _, corrected, written = self._run(tmp_path, monkeypatch, moving_areas=None)
        assert np.allclose(corrected[:6], 0.0)  # centred on the moving median
        assert written == {}  # no filter pipeline, no mask files

    def test_no_moving_areas_warns_and_fits_everywhere(
        self, tmp_path, monkeypatch, caplog_gmc
    ):
        _, corrected, _ = self._run(tmp_path, monkeypatch, with_moving_file=False)
        assert np.allclose(corrected[:6], 0.0)
        assert any("No moving areas loaded" in r.message for r in caplog_gmc.records)

    def test_explicit_stable_area_mask_is_not_duplicated(self, tmp_path):
        stable = np.ones((10, 10), bool)
        stable[:6] = False
        stub, _, _ = self._setup(tmp_path, True)
        out = Session._with_moving_areas(
            stub, StableAreaMask(stable) + CCFilter(0.3), "auto", ["Z"]
        )
        assert [type(m).__name__ for m in out["Z"].masks] == [
            "StableAreaMask",
            "CCFilter",
        ]

    def test_quality_mask_still_removes_pixels_from_the_output(
        self, tmp_path, monkeypatch
    ):
        result, _, _ = self._run(
            tmp_path,
            monkeypatch,
            filter_pipeline=CCFilter(0.3) + OutlierFilter((0.0, 3.0)),
        )
        raw = rasterio.open(result["ew"]).read(1)
        assert np.all(raw[:6] == 0.0)  # the 5 m/yr block is a quality outlier: nodata
        assert (raw[6:] != 0.0).sum() >= 38  # the stable rows are still there

    def test_stable_stats_are_written_for_the_raw_field_and_each_step(
        self, tmp_path, monkeypatch
    ):
        import geomulticorr.stats.stats as gmc_stats

        calls = {"raw": [], "step": []}
        stub, pair, _ = self._setup(tmp_path, True)
        monkeypatch.setattr(
            gmc_stats,
            "save_stable_stats",
            lambda pair, section, x, y, stable, **k: calls["raw"].append(
                (section, int(stable[0].sum()))
            )
            or {"raw_corr_stats_stable": {}},
        )
        monkeypatch.setattr(
            gmc_stats,
            "save_corrected_stats",
            lambda pair, x, y, name, **k: calls["step"].append(
                (name, int(k["stable"][0].sum()))
            )
            or {},
        )
        monkeypatch.setattr(gmc_stats, "save_final_corrected_stats", lambda *a, **k: {})
        from geomulticorr.corrections import MedianCentering

        Session.apply_pairs_corrections(
            stub,
            MedianCentering(),
            save_plot=False,
            print_summary=False,
            sync_geodb=False,
            overwrite=True,
        )
        # 40 stable pixels: the 4 rows outside the moving areas
        assert calls["raw"] == [("raw_corr_stats_stable", 40)]
        assert calls["step"] == [("MedianCentering", 40)]


class TestApplyCorrectionsMemory:
    """Each pair's memory is released before the next one starts."""

    @staticmethod
    def _run(tmp_path, monkeypatch, **kwargs):
        calls = []
        monkeypatch.setattr(
            session_mod, "_release_memory", lambda: calls.append("release")
        )
        result, corrected, _ = TestApplyCorrectionsMovingGround._run(
            tmp_path, monkeypatch, **kwargs
        )
        return calls, result, corrected

    def test_memory_is_released_once_per_pair(self, tmp_path, monkeypatch):
        calls, result, _ = self._run(tmp_path, monkeypatch)
        assert result is not None
        assert calls == ["release"]

    def test_memory_is_released_when_a_pair_fails(self, tmp_path, monkeypatch):
        """A missing dem= makes the pair fail; the release still happens."""
        from geomulticorr.corrections import TopoCorrection

        monkeypatch.setattr(
            session_mod, "_release_memory", lambda: calls.append("release")
        )
        calls: list = []
        stub, pair, _ = TestApplyCorrectionsMovingGround._setup(tmp_path, False)
        results = Session.apply_pairs_corrections(
            stub,
            TopoCorrection(),
            save_plot=False,
            print_summary=False,
            sync_geodb=False,
            overwrite=True,
            moving_areas=None,
        )
        assert results["k"] is None
        assert calls == ["release"]

    def test_no_raster_outlives_the_pair(self, tmp_path, monkeypatch):
        """After a pair is done nothing keeps its rasters alive (no stray reference)."""
        import gc

        import geoutils as gu

        alive_after = []

        def check():
            gc.collect()
            alive_after.append(sum(isinstance(o, gu.Raster) for o in gc.get_objects()))

        monkeypatch.setattr(session_mod, "_release_memory", check)
        gc.collect()
        before = sum(isinstance(o, gu.Raster) for o in gc.get_objects())
        TestApplyCorrectionsMovingGround._run(tmp_path, monkeypatch)
        assert alive_after == [before]

    def test_save_and_close_releases_the_figure_even_if_the_write_fails(self, tmp_path):
        import matplotlib.pyplot as plt

        mine, other = plt.figure(), plt.figure()
        mine.add_subplot().plot([0, 1])
        with pytest.raises(Exception):
            session_mod._save_and_close(mine, tmp_path / "missing" / "f.jpg", 50)
        assert mine.axes == []  # cleared: its arrays are released
        assert mine.number not in plt.get_fignums()
        assert other.number in plt.get_fignums()  # the user's figures stay open
        plt.close(other)

    def test_release_memory_runs(self):
        session_mod._release_memory()  # gc + malloc_trim, never raises
