#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistics on stable ground: the numbers that judge a correction."""

from __future__ import annotations

import geoutils as gu
import numpy as np
import numpy.ma as ma
import pytest
import rasterio

from geomulticorr.corrections import RampCorrection
from geomulticorr.stats.stats import (
    compute_stable_stats,
    load_pair_stats,
    save_corrected_stats,
    save_final_corrected_stats,
    save_stable_stats,
)

NOISE = 0.05


def _scene(seed=0, blob=slice(5, 30)):
    """A tilted stable plane + noise, with a 15 m/yr moving blob in one corner."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[:80, :80]
    field = 0.05 * rows + 0.03 * cols + rng.normal(0, NOISE, (80, 80))
    moving = np.zeros((80, 80), bool)
    moving[blob, blob] = True
    field[moving] += 15.0
    raster = gu.Raster.from_array(
        field.astype("float32"),
        transform=rasterio.transform.from_origin(0, 800, 10, 10),
        crs="EPSG:32632",
        nodata=-9999,
    )
    return raster, ~moving


def test_stable_nmad_judges_the_ramp_correction_and_the_blob_survives():
    raster, stable = _scene()
    before = compute_stable_stats(raster, raster, (stable, stable))["ew"]["nmad"]
    corrected = RampCorrection().fit_and_apply(raster, stable_mask=stable)
    after = compute_stable_stats(corrected, corrected, (stable, stable))["ew"]["nmad"]
    assert before > 5 * NOISE  # the tilt dominates
    assert after == pytest.approx(NOISE, rel=0.2)  # only the noise is left
    assert np.ma.getdata(corrected.data)[10, 10] > 10.0  # moving ground kept


def test_every_pixel_stats_are_not_the_stable_stats():
    """A large moving area inflates the all-pixel NMAD; the stable one ignores it."""
    raster, stable = _scene(blob=slice(0, 55))  # ~47 % of the scene
    corrected = RampCorrection().fit_and_apply(raster, stable_mask=stable)
    stable_n = compute_stable_stats(corrected, corrected, (stable, stable))
    everywhere = compute_stable_stats(
        corrected, corrected, (np.ones_like(stable), np.ones_like(stable))
    )
    assert everywhere["ew"]["nmad"] > 2 * stable_n["ew"]["nmad"]
    assert stable_n["ew"]["count_valid"] == int(stable.sum())


def test_already_masked_pixels_stay_excluded():
    raster, stable = _scene()
    data = raster.data.copy()
    data[40:, :] = ma.masked
    raster = raster.copy(new_array=data)
    out = compute_stable_stats(raster, raster, (stable, stable))["ew"]
    assert out["count_valid"] == int((stable & ~np.ma.getmaskarray(data)).sum())


class TestPersistence:
    def test_save_corrected_stats_writes_the_stable_block(self, mock_pair):
        raster, stable = _scene()
        save_corrected_stats(
            mock_pair, raster, raster, "RampCorrection", stable=(stable, stable)
        )
        js = load_pair_stats(mock_pair)
        assert "RampCorrection" in js["correction_stats"]
        stable_block = js["correction_stats_stable"]["RampCorrection"]
        assert stable_block["ew"]["count_valid"] == int(stable.sum())
        # the existing block keeps its meaning: every valid pixel
        assert js["correction_stats"]["RampCorrection"]["ew"]["count_valid"] == 6400

    def test_without_stable_a_stale_entry_is_removed(self, mock_pair):
        raster, stable = _scene()
        save_corrected_stats(mock_pair, raster, raster, "Step", stable=(stable, stable))
        save_corrected_stats(mock_pair, raster, raster, "Step")
        assert "Step" not in load_pair_stats(mock_pair)["correction_stats_stable"]

    def test_final_block_is_a_copy_of_the_last_step(self, mock_pair):
        raster, stable = _scene()
        save_corrected_stats(mock_pair, raster, raster, "A", stable=(stable, stable))
        save_corrected_stats(mock_pair, raster, raster, "B", stable=(stable, stable))
        save_final_corrected_stats(mock_pair, "B")
        js = load_pair_stats(mock_pair)
        assert js["final_corrected_stats_stable"] == js["correction_stats_stable"]["B"]

    def test_final_block_is_cleared_when_there_is_no_stable_ground(self, mock_pair):
        raster, stable = _scene()
        save_corrected_stats(mock_pair, raster, raster, "A", stable=(stable, stable))
        save_final_corrected_stats(mock_pair, "A")
        save_corrected_stats(mock_pair, raster, raster, "A")  # a later run, no mask
        save_final_corrected_stats(mock_pair, "A")
        assert load_pair_stats(mock_pair)["final_corrected_stats_stable"] == {}

    def test_raw_section_is_written_flat(self, mock_pair):
        raster, stable = _scene()
        save_stable_stats(
            mock_pair, "raw_corr_stats_stable", raster, raster, (stable, stable)
        )
        assert set(load_pair_stats(mock_pair)["raw_corr_stats_stable"]) == {"ew", "ns"}
