# -*- coding: utf-8 -*-
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
from geomulticorr.corrections import CCFilter, OutlierFilter

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
    "-F_EWmask.tif",
    "-F_NSmask.tif",
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
            pa_ew_mask_path=tmp_path / "k-F_EWmask.tif",
            pa_ns_mask_path=tmp_path / "k-F_NSmask.tif",
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

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="apply_pairs_corrections accepts a single mask, then calls "
        "filter_pipeline.compute(), which only FilterPipeline has; every pair fails.",
    )
    def test_single_mask_writes_the_corrected_maps(self, tmp_path):
        assert self._run(tmp_path, CCFilter(0.3)) is not None
