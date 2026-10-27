"""Unit tests for the Pair methods that do not need ASP or a Session.

Pairs are built with ``Pair.__new__`` (the real constructor opens rasters and
needs a geodatabase) and only the attributes each method touches. ASP and the
shell (``os.system``) are replaced by recorders, so the command strings are
checked but nothing runs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import geoutils as gu
import matplotlib.pyplot as plt

import geomulticorr.core.pair as gmc_pair
from geomulticorr.core.pair import Pair

KEY = "Pz_2024-01-01_planetscope__2024-01-15_planetscope"


def _thumb(root: Path, date: str, th_type="opt", pz="Pz"):
    d = dt.datetime.strptime(date, "%Y-%m-%d")
    sub = root / pz / "optical"
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{pz}_{date}_planetscope.tif"
    path.write_bytes(b"x")
    return SimpleNamespace(
        th_pz_name=pz,
        th_path=path,
        th_type=th_type,
        th_date=date,
        th_date_datetime=d,
        th_sensor="planetscope",
        th_key=f"{pz}_{date}_planetscope",
        geometry=None,
    )


def _pair(tmp_path, status="empty") -> Pair:
    p = Pair.__new__(Pair)
    p.session = None
    p.pa_pz_name = "Pz"
    p.pa_key = KEY
    p.pa_type = "opt"
    p.pa_path = tmp_path / "image_correlation" / KEY
    for attr, suffix in (
        ("pa_raw_left_path", "-L.tif"),
        ("pa_raw_right_path", "-R.tif"),
        ("pa_disparity_rd_path", "-RD.tif"),
        ("pa_disparity_f_path", "-F.tif"),
        ("pa_cc_raw_path", "-F-ncc.tif"),
        ("pa_ew_path", "-F_EW.tif"),
        ("pa_ns_path", "-F_NS.tif"),
        ("pa_cc_path", "-F_CC.tif"),
        ("pa_magn_path", "_magn.tif"),
        ("pa_plot_raw_path", "_raw_disp.jpg"),
        ("pa_stats_path", ".json"),
    ):
        setattr(p, attr, p.pa_path / f"{KEY}{suffix}")
    left = SimpleNamespace(
        th_date="2024-01-01",
        th_sensor="planetscope",
        th_date_datetime=dt.datetime(2024, 1, 1),
        th_path=tmp_path / "left.tif",
        th_key="Pz_2024-01-01_planetscope",
    )
    right = SimpleNamespace(
        th_date="2024-01-15",
        th_sensor="planetscope",
        th_date_datetime=dt.datetime(2024, 1, 15),
        th_path=tmp_path / "right.tif",
        th_key="Pz_2024-01-15_planetscope",
    )
    left.th_path.write_bytes(b"x")
    right.th_path.write_bytes(b"x")
    p.pa_left, p.pa_right = left, right
    p.pa_left_link_path = p.pa_path / f"{left.th_key}.tif"
    p.pa_right_link_path = p.pa_path / f"{right.th_key}.tif"
    p.pa_dt_days, p.pa_dt_months, p.pa_dt_years = 14, 0.46, 0.5
    p.pa_direction = "forward"
    p.pa_status = status
    p.geometry = None
    return p


def _write_disparity(path: Path, shape=(8, 6), res=3.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack([np.full(shape, 1.0), np.full(shape, 2.0), np.ones(shape)]).astype(
        "float32"
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=3,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(500000, 4649000, res, res),
    ) as dst:
        dst.write(data)


def _write_single(path: Path, shape=(8, 6), value=1.0, res=3.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(500000, 4649000, res, res),
    ) as dst:
        dst.write(np.full(shape, value, dtype="float32"), 1)


class _FakeRaster:
    """Stand-in for gu.Raster: records ctor args, scales on ``*``."""

    calls: list = []

    def __init__(self, path, **kw):
        self.path, self.kw = path, kw
        self.res = (3.0, 3.0)
        _FakeRaster.calls.append((path, kw))

    def __mul__(self, f):
        return ("scaled", self.path, f)


@pytest.fixture
def fake_raster(monkeypatch):
    _FakeRaster.calls = []
    monkeypatch.setattr(gmc_pair.gu, "Raster", _FakeRaster)
    return _FakeRaster


class TestConstruction:
    def test_ill_formed_path_asserts(self, tmp_path):
        with pytest.raises(AssertionError, match="well-formed"):
            Pair(session=None, target_path=tmp_path / "nowhere" / KEY)

    def test_bad_key_under_image_correlation_asserts(self, tmp_path):
        bad = tmp_path / "image_correlation" / "Pz_notapair"
        bad.parent.mkdir()
        with pytest.raises(AssertionError, match="well-formed"):
            Pair(session=SimpleNamespace(), target_path=bad)

    def test_from_two_thumbs(self, tmp_path):
        a, b = _thumb(tmp_path, "2024-01-01"), _thumb(tmp_path, "2024-01-15")
        p = Pair(left=a, right=b)
        assert p.pa_key == KEY
        assert p.pa_dt_days == 14 and p.pa_direction == "forward"
        assert p.pa_status == "empty"
        assert p.pa_path.name == KEY and p.pa_type == "opt"

    def test_backward_direction(self, tmp_path):
        a, b = _thumb(tmp_path, "2024-01-15"), _thumb(tmp_path, "2024-01-01")
        assert Pair(left=a, right=b).pa_direction == "backward"

    def test_same_thumb_rejected(self, tmp_path):
        a = _thumb(tmp_path, "2024-01-01")
        with pytest.raises(AssertionError, match="left thumb is right"):
            Pair(left=a, right=a)

    def test_different_pzone_rejected(self, tmp_path):
        a, b = _thumb(tmp_path, "2024-01-01"), _thumb(
            tmp_path, "2024-01-15", pz="Other"
        )
        with pytest.raises(AssertionError, match="same pzone"):
            Pair(left=a, right=b)

    def test_different_image_types_rejected(self, tmp_path):
        a = _thumb(tmp_path, "2024-01-01")
        b = _thumb(tmp_path, "2024-01-15", th_type="slope")
        with pytest.raises(AssertionError, match="image types"):
            Pair(left=a, right=b)


class TestStatusAndReprs:
    def test_repr_lists_key_facts(self, tmp_path):
        text = repr(_pair(tmp_path))
        for token in ("GMC_Pair", "Pz", "2024-01-01-planetscope", "forward", "14 days"):
            assert token in text

    def test_status_empty(self, tmp_path):
        assert _pair(tmp_path).get_status() == "empty"

    def test_status_corrupt(self, tmp_path):
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        assert p.get_status() == "corrupt"

    def test_status_clipped(self, tmp_path):
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        p.pa_left_link_path.touch()
        p.pa_right_link_path.touch()
        assert p.get_status() == "clipped"

    def test_status_complete(self, tmp_path):
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        p.pa_disparity_f_path.touch()
        assert p.get_status() == "complete"

    def test_check_corr_outputs(self, tmp_path):
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        (p.pa_path / f"{KEY}-F.tif").touch()
        out = p.check_corr_outputs()
        assert out["-F.tif"] is True
        assert set(out) == set(gmc_pair.ASP._KEEP_SUFFIXES)
        assert not all(out.values())

    def test_to_pdserie(self, tmp_path):
        s = _pair(tmp_path).to_pdserie()
        assert s["pa_pz_name"] == "Pz" and s["pa_dt_days"] == 14
        assert s["pa_type"] == "opt" and s["pa_left_sensor"] == "planetscope"


class TestRasterLoaders:
    def test_magn_and_cc_cached(self, tmp_path, fake_raster):
        p = _pair(tmp_path)
        m1, m2 = p.get_magn_raster(), p.get_magn_raster()
        c1, c2 = p.get_cc_raster(), p.get_cc_raster()
        assert m1 is m2 and c1 is c2
        assert [c[0] for c in fake_raster.calls] == [
            str(p.pa_magn_path),
            str(p.pa_cc_raw_path),
        ]

    def test_disp_corr_raster_reads_three_bands_once(self, tmp_path, fake_raster):
        p = _pair(tmp_path)
        assert p.get_disp_corr_raster() is p.get_disp_corr_raster()
        assert len(fake_raster.calls) == 1
        assert fake_raster.calls[0][1]["nBands"] == 3

    def test_xdisp_first_call_scales_by_pixel_size(self, tmp_path, fake_raster):
        out = _pair(tmp_path).get_xDisp_raster()
        assert out[0] == "scaled" and out[2] == 3.0

    def test_ydisp_flips_sign_after_xdisp(self, tmp_path, fake_raster):
        p = _pair(tmp_path)
        p.get_xDisp_raster()
        out = p.get_yDisp_raster()
        assert out[0] == "scaled" and out[2] == -3.0

    def test_ydisp_works_standalone(self, tmp_path, fake_raster):
        out = _pair(tmp_path).get_yDisp_raster()
        assert out[0] == "scaled" and out[2] == -3.0

    def test_ydisp_is_idempotent(self, tmp_path, fake_raster):
        p = _pair(tmp_path)
        assert p.get_yDisp_raster() == p.get_yDisp_raster()

    def test_xdisp_is_idempotent(self, tmp_path, fake_raster):
        p = _pair(tmp_path)
        assert p.get_xDisp_raster() == p.get_xDisp_raster()

    def test_velocities_divide_by_baseline(self, tmp_path, monkeypatch):
        p = _pair(tmp_path)
        monkeypatch.setattr(p, "get_xDisp_raster", lambda: np.array([1.0, 2.0]))
        monkeypatch.setattr(p, "get_yDisp_raster", lambda: np.array([3.0]))
        monkeypatch.setattr(p, "compute_magnitude", lambda: np.array([4.0]))
        assert p.get_xVel().tolist() == [2.0, 4.0]
        assert p.get_yVel().tolist() == [6.0]
        assert p.get_vMagn_raster().tolist() == [8.0]


class TestClip:
    def test_creates_symlinks_and_sets_status(self, tmp_path):
        p = _pair(tmp_path)
        assert p.clip() is True
        assert p.pa_left_link_path.is_symlink() and p.pa_right_link_path.is_symlink()
        assert p.pa_status == "clipped"
        assert p.get_status() == "clipped"

    def test_idempotent_with_existing_links(self, tmp_path):
        p = _pair(tmp_path)
        p.clip()
        p.pa_status = "empty"
        assert p.clip() is True

    @pytest.mark.parametrize("status", ["clipped", "complete"])
    def test_early_return_creates_nothing(self, tmp_path, status):
        p = _pair(tmp_path, status=status)
        assert p.clip() is True
        assert not p.pa_path.exists()


class TestLegacyCorrelation:
    def test_launch_correlation_complete_skips_asp(self, tmp_path, monkeypatch):
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        p.pa_disparity_f_path.touch()
        monkeypatch.setattr(
            gmc_pair, "ASP", lambda **k: pytest.fail("ASP must not run")
        )
        assert p.launch_correlation() is True

    def test_launch_correlation_delegates_to_asp(self, tmp_path, monkeypatch):
        seen = {}

        class FakeASP:
            def __init__(self, session=None):
                seen["session"] = session

            def run_correlation(self, **kw):
                seen.update(kw)
                return 0

        monkeypatch.setattr(gmc_pair, "ASP", FakeASP)
        p = _pair(tmp_path)
        assert p.launch_correlation() == 0
        assert seen["left"] == p.pa_left.th_path
        assert seen["params_file_path"] == p.pa_path / "corr_params.txt"

    def test_corr_builds_parallel_stereo_command(self, tmp_path, monkeypatch):
        cmds = []
        monkeypatch.setattr(gmc_pair, "ROOT_OUTPUTS", tmp_path / "temp")
        (tmp_path / "temp").mkdir()
        monkeypatch.setattr(gmc_pair.os, "system", lambda c: cmds.append(c) or 0)
        p = _pair(tmp_path)
        assert p.corr(corr_algorithm=3, corr_kernel_size=9, corr_xthreshold=5) is True
        assert (tmp_path / "temp" / KEY).is_dir()
        assert "parallel_stereo" in cmds[0]
        assert "--stereo-algorithm 3" in cmds[0]
        assert "--corr-kernel 9 9" in cmds[0]
        assert "--xcorr-threshold 5" in cmds[0]

    def test_corr_skips_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gmc_pair.os, "system", lambda c: pytest.fail("no shell"))
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        p.pa_disparity_f_path.touch()
        assert p.corr() is True

    def test_corr_eval_runs_command_and_renames(self, tmp_path, monkeypatch):
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        cmds = []

        def fake_system(cmd):
            cmds.append(cmd)
            if cmd.startswith("corr_eval"):
                p.pa_cc_raw_path.touch()
            return 0

        monkeypatch.setattr(gmc_pair.os, "system", fake_system)
        assert p.corr_eval(corr_kernel_size=11, metric="ncc") is True
        assert "--kernel-size 11 11" in cmds[0] and "--metric ncc" in cmds[0]
        assert cmds[1].startswith("cp ")
        assert p.pa_cc_path.name == f"{KEY}_CC.tif" and p.pa_cc_path.exists()

    def test_corr_eval_early_return_when_cc_exists(self, tmp_path):
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        p.pa_cc_raw_path.touch()
        assert p.corr_eval() is True

    @pytest.mark.parametrize("verbose,redirect", [(False, "/dev/null"), (True, None)])
    def test_save_corrdata(self, tmp_path, monkeypatch, verbose, redirect):
        cmds = []
        monkeypatch.setattr(gmc_pair, "ROOT_OUTPUTS", tmp_path / "temp")
        monkeypatch.setattr(gmc_pair.os, "system", lambda c: cmds.append(c) or 0)
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)  # as if the move had succeeded
        assert p.save_corrdata(verbose=verbose) is True
        assert cmds[0].startswith("mv ") and cmds[1].startswith("rm -rf ")
        assert ("/dev/null" in cmds[0]) is bool(redirect)

    def test_save_corrdata_reports_failed_move(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gmc_pair, "ROOT_OUTPUTS", tmp_path / "temp")
        monkeypatch.setattr(gmc_pair.os, "system", lambda c: 0)
        assert _pair(tmp_path).save_corrdata() is False

    def test_compute_magnitude_requires_disparity(self, tmp_path):
        with pytest.raises(AssertionError, match="not yet correlate"):
            _pair(tmp_path).compute_magnitude()


class TestExtractRawDisplacements:
    @pytest.fixture
    def pair_with_disparity(self, tmp_path, monkeypatch):
        p = _pair(tmp_path)
        _write_disparity(p.pa_disparity_f_path)
        monkeypatch.setattr(
            gmc_pair.gmc_stats, "save_raw_corr_stats", lambda pair: {"stats": "ok"}
        )
        return p

    def test_requires_disparity(self, tmp_path):
        with pytest.raises(AssertionError, match="Disparity raster not found"):
            _pair(tmp_path).extract_raw_displacements()

    def test_writes_metre_rasters(self, pair_with_disparity):
        p = pair_with_disparity
        assert p.extract_raw_displacements(save_plot=False) == {"stats": "ok"}
        ew = gu.Raster(str(p.pa_ew_path))
        ns = gu.Raster(str(p.pa_ns_path))
        assert float(np.nanmedian(ew.data)) == pytest.approx(3.0)  # 1 px * 3 m
        assert float(np.nanmedian(ns.data)) == pytest.approx(
            -6.0
        )  # 2 px * 3 m, flipped
        assert not p.pa_plot_raw_path.exists()

    def test_plot_and_cc_copy(self, pair_with_disparity, monkeypatch):
        p = pair_with_disparity
        _write_single(p.pa_cc_raw_path, value=0.8)
        figs = []

        def fake_plot(**kw):
            fig = plt.figure()
            figs.append(kw)
            return fig

        monkeypatch.setattr(gmc_pair, "plot_show_raw_results", fake_plot)
        p.extract_raw_displacements(save_plot=True)
        assert p.pa_plot_raw_path.exists()
        assert p.pa_cc_path.exists()
        assert figs[0]["fig_name"] == KEY and figs[0]["ncc"] is not None

    def test_plot_without_ncc(self, pair_with_disparity, monkeypatch):
        seen = {}

        def fake_plot(**kw):
            seen.update(kw)
            return plt.figure()

        monkeypatch.setattr(gmc_pair, "plot_show_raw_results", fake_plot)
        pair_with_disparity.extract_raw_displacements(save_plot=True)
        assert seen["ncc"] is None

    def test_target_resolution_resamples(self, pair_with_disparity):
        p = pair_with_disparity
        p.extract_raw_displacements(save_plot=False, target_resolution=6.0)
        assert gu.Raster(str(p.pa_ew_path)).res[0] == pytest.approx(6.0)

    def test_canonical_grid(self, pair_with_disparity):
        p = pair_with_disparity
        bounds = rasterio.coords.BoundingBox(500000, 4648976, 500018, 4649000)
        p.extract_raw_displacements(
            save_plot=False,
            target_resolution=6.0,
            canonical_bounds=bounds,
            canonical_grid_size=(3, 4),
        )
        r = gu.Raster(str(p.pa_ew_path))
        assert (r.width, r.height) == (3, 4)


class TestVectorizeMovingAreas:
    def _write_classes(self, p, n_clusters=2):
        path = p.pa_path / f"KMe_N{n_clusters}_{KEY}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.zeros((40, 40), dtype="uint8")
        data[5:25, 5:25] = 1  # 20x20 px block
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=40,
            width=40,
            count=1,
            dtype="uint8",
            crs="EPSG:32632",
            transform=from_origin(500000, 4649000, 10.0, 10.0),
        ) as dst:
            dst.write(data, 1)

    def test_writes_gpkg(self, tmp_path):
        import geopandas as gpd

        p = _pair(tmp_path)
        self._write_classes(p)
        p.vectorize_moving_areas(epsg=32632, n_clusters=2)
        gdf = gpd.read_file(p.pa_path / f"{KEY}_classif-2.gpkg")
        assert set(gdf["raster_val"]) == {0.0, 1.0}

    def test_min_surf_filter_and_name(self, tmp_path):
        import geopandas as gpd

        p = _pair(tmp_path)
        self._write_classes(p)
        p.vectorize_moving_areas(epsg=32632, min_surf=10, n_clusters=2)
        out = p.pa_path / f"{KEY}_classif-2_min-surf-10.gpkg"
        gdf = gpd.read_file(out)
        assert len(gdf) >= 1 and (gdf.area / 1000 > 10).all()

    def test_missing_raster_raises(self, tmp_path):
        p = _pair(tmp_path)
        p.pa_path.mkdir(parents=True)
        with pytest.raises(Exception):
            p.vectorize_moving_areas(epsg=32632)


class TestInterestingRasterAndSlices:
    @pytest.mark.parametrize(
        "mode,attr",
        [
            ("X", "get_xDisp_raster"),
            ("y", "get_yDisp_raster"),
            ("vx", "get_xVel"),
            ("VY", "get_yVel"),
        ],
    )
    def test_mode_dispatch(self, tmp_path, monkeypatch, mode, attr):
        p = _pair(tmp_path)
        monkeypatch.setattr(p, attr, lambda: attr)
        assert p.get_interesting_geoim(mode) == attr

    def test_magnitude_and_velocity_magnitude(self, tmp_path, monkeypatch):
        p = _pair(tmp_path)
        monkeypatch.setattr(p, "get_xDisp_raster", lambda: np.array([3.0]))
        monkeypatch.setattr(p, "get_yDisp_raster", lambda: np.array([4.0]))
        assert p.get_interesting_geoim("m").tolist() == [5.0]
        assert p.get_interesting_geoim("vm").tolist() == [5.0 / p.pa_dt_years]

    def test_unknown_mode_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown mode"):
            _pair(tmp_path).get_interesting_geoim("nope")

    def test_get_slice_delegates(self, tmp_path, monkeypatch):
        calls = []
        fake = SimpleNamespace(
            inspectGeoLine=lambda line: calls.append(("line", line)) or "slice",
            inspectRibsAlongthumb=lambda *a: calls.append(("ribs", a)) or "ribs",
        )
        p = _pair(tmp_path)
        monkeypatch.setattr(p, "get_interesting_geoim", lambda mode: fake)
        assert p.get_slice("LINE", mode="x") == "slice"
        assert p.get_slices("LINE", 100, 10, "h") == "ribs"
        assert calls == [("line", "LINE"), ("ribs", ("LINE", 100, 10, "h"))]
