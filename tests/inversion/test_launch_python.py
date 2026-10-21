#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``TIOInversion.launch(mode="python")`` — the in-process solver backend.

A minimal inversion folder is assembled by hand (the files
``prepare_inversion`` writes: ``binary/<d1>_<d2>_<EW|NS>`` pair maps,
``liste_image_inv``, ``liste_couple``) around a known linear truth, so the
GeoTIFFs the backend writes can be checked against it. No Fortran binaries,
no project, no session.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geomulticorr.inversion import TIOConfig
from geomulticorr.inversion.pytio import io as tio_io
from geomulticorr.inversion.tio_inversion import TIOInversion

W, H = 5, 6
DATES = ["20200101", "20201201", "20220201", "20221201", "20240101"]
T = np.array([2020.0, 2020.9, 2022.1, 2022.9, 2024.0])
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def _velocity(component):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    base = 0.5 if component == "EW" else -0.3
    return base * (1 + 0.1 * (yy * W + xx))


def _truth(component):
    """(S, H, W) cumulative displacement, zero at the first date."""
    return _velocity(component)[None] * (T - T[0])[:, None, None]


def _write_ref_tif(path: Path) -> None:
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": W,
               "height": H, "crs": "EPSG:32632",
               "transform": from_origin(500000, 4649000, 3.0, 3.0), "nodata": np.nan}
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(np.zeros((H, W), np.float32), 1)


@pytest.fixture
def inv(tmp_path):
    inversion_dir = tmp_path / "inv"
    (inversion_dir / "binary").mkdir(parents=True)
    for d in ("EW", "NS"):
        (inversion_dir / f"inverse_{d}").mkdir()
        u = _truth(d)
        for i, j in PAIRS:
            tio_io.write_r4(u[j] - u[i], str(inversion_dir / "binary" / f"{DATES[i]}_{DATES[j]}_{d}"))
        (inversion_dir / f"inverse_{d}" / "liste_image_inv").write_text(
            "".join(f"{date} {t:.4f} {t - T[0]:.4f} 0\n" for date, t in zip(DATES, T)))
        (inversion_dir / f"inverse_{d}" / "liste_couple").write_text(
            "".join(f"{DATES[i]} {DATES[j]} 1.000000\n" for i, j in PAIRS))

    ref = tmp_path / "ref_EW.tif"
    _write_ref_tif(ref)

    obj = TIOInversion.__new__(TIOInversion)
    obj.inversion_dir = inversion_dir
    obj.inversion_name = "t"
    obj.pairs = [SimpleNamespace(pa_ew_path=ref)]
    obj._DIRECTIONS = ("EW", "NS")
    obj._raster_width = W
    obj._raster_height = H
    obj.solver = TIOConfig()
    obj.invers_pixel_omp_bin = obj.lect_depl_cumule_lin_bin = None
    obj._last_launch = None
    return obj


def _read(path):
    with rasterio.open(str(path)) as src:
        return src.read(1), src.crs, src.transform


class TestPythonLaunch:
    def test_writes_full_height_georeferenced_tot_rasters(self, inv):
        out = inv.launch(mode="python")
        assert set(out) == {"EW", "NS"}
        for d in ("EW", "NS"):
            truth = _truth(d)
            for k, date in enumerate(DATES):
                path = inv.inversion_dir / f"inverse_{d}" / f"TOT_{date}_{d}.tif"
                assert path.exists(), path
                data, crs, transform = _read(path)
                assert data.shape == (H, W)               # full height, not H-1
                assert crs.to_epsg() == 32632
                assert transform == from_origin(500000, 4649000, 3.0, 3.0)
                np.testing.assert_allclose(data, truth[k], atol=1e-4)

    def test_diagnostics_and_magnitude(self, inv):
        inv.launch(mode="python")
        diag = inv.inversion_dir / "inverse_EW" / "diagnostics"
        for stem in ("rms", "rank_defect", "n_pairs", "n_images",
                     "velocity", "velocity_error", "rho"):
            assert (diag / f"{stem}.tif").exists(), stem
        rms, *_ = _read(diag / "rms.tif")
        assert np.nanmax(rms) < 1e-4
        n_pairs, *_ = _read(diag / "n_pairs.tif")
        assert (n_pairs == len(PAIRS)).all()
        vel, *_ = _read(diag / "velocity.tif")
        np.testing.assert_allclose(vel, _velocity("EW"), atol=1e-4)
        # no residual rasters unless asked
        assert not list(diag.glob("residual_*.tif"))

        magn = inv.inversion_dir / "inverse_magn" / f"TOT_{DATES[-1]}_magn.tif"
        assert magn.exists()
        data, *_ = _read(magn)
        expected = np.hypot(_truth("EW")[-1], _truth("NS")[-1])
        np.testing.assert_allclose(data, expected, atol=1e-3)

    def test_returned_datasets_carry_the_solver_diagnostics(self, inv):
        out = inv.launch(mode="python")
        ds = out["EW"]
        for var in ("cum_disp", "residual", "rms", "rank_defect", "TOT", "velocity"):
            assert var in ds, var
        assert ds.residual.dims == ("pair_out", "y", "x")
        assert ds.residual.sizes["pair_out"] == len(PAIRS)
        assert np.abs(ds.residual.values).max() < 1e-4
        assert ds["time"].values[0] == pytest.approx(T[0])     # decimal years kept
        assert inv._last_launch["backend"] == "python"
        assert inv._last_launch["directions"] == ["EW", "NS"]
        assert inv._last_launch["seconds"] >= 0

    def test_single_direction_writes_no_magnitude(self, inv):
        out = inv.launch(direction="EW", mode="python")
        assert list(out) == ["EW"]
        assert not (inv.inversion_dir / "inverse_magn").exists()
        assert not list((inv.inversion_dir / "inverse_NS").glob("TOT_*.tif"))

    def test_residual_rasters_on_request(self, inv):
        inv.launch(direction="EW", mode="python", write_residuals=True)
        files = sorted((inv.inversion_dir / "inverse_EW" / "diagnostics").glob("residual_*.tif"))
        assert len(files) == len(PAIRS)
        assert files[0].name == f"residual_{DATES[0]}_{DATES[1]}.tif"

    def test_solver_settings_are_honoured(self, inv):
        """A bad pair with weight 0 in liste_couple is removed under weight_mode='file'."""
        d = "EW"
        u = _truth(d)
        k = 3
        i, j = PAIRS[k]
        tio_io.write_r4(u[j] - u[i] + 4.0, str(inv.inversion_dir / "binary" / f"{DATES[i]}_{DATES[j]}_{d}"))
        lines = [f"{DATES[a]} {DATES[b]} {'0.000000' if n == k else '1.000000'}\n"
                 for n, (a, b) in enumerate(PAIRS)]
        (inv.inversion_dir / "inverse_EW" / "liste_couple").write_text("".join(lines))

        good = inv.launch(direction="EW", mode="python")["EW"]
        np.testing.assert_allclose(good.TOT.values, u, atol=1e-4)

        inv.solver = TIOConfig(weight_mode="none")       # column 3 not read: the 0 is ignored
        unweighted = inv.launch(direction="EW", mode="python")["EW"]
        assert np.abs(unweighted.TOT.values - u).max() > 0.1

    def test_variance_mode_refuses_a_grid_too_small_for_its_sampling_region(self, inv):
        inv.solver = TIOConfig.legacy()
        with pytest.raises(ValueError, match="variance"):
            inv.launch(direction="EW", mode="python")

    def test_post_process_after_a_python_run_only_derives_magnitude(self, inv, caplog_gmc):
        inv.launch(mode="python")
        magn = inv.inversion_dir / "inverse_magn" / f"TOT_{DATES[0]}_magn.tif"
        magn.unlink()
        inv.post_process()
        assert magn.exists()
        assert "nothing to convert" in caplog_gmc.text
        assert "no TOT_* files found" not in caplog_gmc.text

    def test_python_mode_needs_no_cluster_validation(self, inv):
        inv.cluster = "isterre"                          # scripts written for a cluster …
        inv.launch(direction="EW", mode="python")        # … is irrelevant in-process
