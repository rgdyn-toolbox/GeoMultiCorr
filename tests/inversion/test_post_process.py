#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for TIOInversion.post_process() / _tot_to_geotiff() naming.

Locks down the TOT_<date>_<component>.tif convention (EW/NS/magn all
suffixed) introduced to replace the previous unsuffixed TOT_<date>.tif,
which left the two directions indistinguishable once copied elsewhere.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geomulticorr.inversion.tio_inversion import TIOInversion

_WIDTH = 3
_HEIGHT = 4  # tot_height = HEIGHT - 1 = 3 (lect_depl_cumule_lin convention)
_DATES = ["20210901", "20211001"]


def _write_ref_tif(path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": _WIDTH,
        "height": _HEIGHT,
        "crs": "EPSG:32632",
        "transform": from_origin(500000, 4649000, 1.0, 1.0),
        "nodata": np.nan,
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(np.zeros((_HEIGHT, _WIDTH), dtype=np.float32), 1)


def _write_tot_binary(path: Path) -> None:
    data = np.ones((_HEIGHT - 1, _WIDTH), dtype=np.float32)
    data.tofile(str(path))


@pytest.fixture
def inv(tmp_path):
    inversion_dir = tmp_path / "inversion_test"
    for d in ("inverse_EW", "inverse_NS"):
        (inversion_dir / d).mkdir(parents=True)
        for date in _DATES:
            _write_tot_binary(inversion_dir / d / f"TOT_{date}")

    ref_tif = tmp_path / "ref.tif"
    _write_ref_tif(ref_tif)

    inversion = TIOInversion.__new__(TIOInversion)
    inversion.inversion_dir = inversion_dir
    inversion.pairs = [SimpleNamespace(pa_ew_path=ref_tif)]
    inversion._raster_width = _WIDTH
    inversion._raster_height = _HEIGHT
    return inversion


class TestTotTifName:
    """The single name-builder shared by the writer, the skip-check and magnitude."""

    def test_builds_component_suffixed_name(self):
        out = TIOInversion._tot_tif_name(Path("/x/inverse_EW"), "20210901", "EW")
        assert out == Path("/x/inverse_EW/TOT_20210901_EW.tif")

    def test_magn_component(self):
        out = TIOInversion._tot_tif_name(Path("/x/inverse_magn"), "20210901", "magn")
        assert out == Path("/x/inverse_magn/TOT_20210901_magn.tif")


class TestTotToGeotiff:
    def test_writes_direction_suffixed_file(self, inv):
        tot = inv.inversion_dir / "inverse_EW" / f"TOT_{_DATES[0]}"
        ref = inv.pairs[0].pa_ew_path

        out = inv._tot_to_geotiff(tot, ref, direction="EW")

        assert out == tot.parent / f"TOT_{_DATES[0]}_EW.tif"
        assert out.exists()
        with rasterio.open(str(out)) as ds:
            assert ds.width == _WIDTH
            assert ds.height == _HEIGHT - 1


class TestPostProcess:
    def test_produces_suffixed_tifs_for_all_components(self, inv):
        inv.post_process()

        for date in _DATES:
            ew = inv.inversion_dir / "inverse_EW" / f"TOT_{date}_EW.tif"
            ns = inv.inversion_dir / "inverse_NS" / f"TOT_{date}_NS.tif"
            magn = inv.inversion_dir / "inverse_magn" / f"TOT_{date}_magn.tif"
            assert ew.exists()
            assert ns.exists()
            assert magn.exists()

        # No unsuffixed leftovers of the old naming scheme.
        assert not (inv.inversion_dir / "inverse_EW" / f"TOT_{_DATES[0]}.tif").exists()
        assert not (inv.inversion_dir / "inverse_magn" / f"TOT_{_DATES[0]}.tif").exists()

    def test_second_call_skips_existing_outputs(self, inv, monkeypatch):
        inv.post_process()

        monkeypatch.setattr(
            inv, "_tot_to_geotiff",
            lambda *a, **k: pytest.fail("should not reconvert an existing output"),
        )
        monkeypatch.setattr(
            inv, "_compute_magnitude",
            lambda *a, **k: pytest.fail("should not recompute an existing magnitude"),
        )

        inv.post_process(overwrite=False)  # must not raise via the patches above

    def test_direction_only_skips_magnitude(self, inv):
        inv.post_process(direction="EW")

        for date in _DATES:
            assert (inv.inversion_dir / "inverse_EW" / f"TOT_{date}_EW.tif").exists()
        assert list((inv.inversion_dir / "inverse_NS").glob("*.tif")) == []
        assert not (inv.inversion_dir / "inverse_magn").exists()
