#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_mask_roles.py
# creation date: 2026-11-25.
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
"""Mask roles: ``quality`` masks shape the output, ``fit`` masks only the fit."""

import numpy as np
import numpy.ma as ma
import pytest

from geomulticorr.corrections.masks import (
    CCFilter,
    CloudMask,
    FilterPipeline,
    OutlierFilter,
    ShadowMask,
    SlopeMask,
    SnowMask,
    StableAreaMask,
    as_pipeline,
)

from .conftest import MockRaster


def _raster(values=None):
    values = np.arange(16, dtype=float).reshape(4, 4) if values is None else values
    return MockRaster(data=ma.array(values))


class TestRoles:
    @pytest.mark.parametrize("cls", [StableAreaMask, SlopeMask])
    def test_fit_role(self, cls):
        assert cls.ROLE == "fit"

    @pytest.mark.parametrize(
        "cls", [CCFilter, OutlierFilter, SnowMask, CloudMask, ShadowMask]
    )
    def test_quality_role(self, cls):
        assert cls.ROLE == "quality"

    def test_pipeline_partitions_masks(self):
        stable = StableAreaMask(np.ones((4, 4), bool))
        cc, out = CCFilter(0.5), OutlierFilter((-1, 1))
        fp = cc + stable + out
        assert fp.quality_masks() == [cc, out]
        assert fp.fit_masks() == [stable]


class TestComputeSplit:
    def test_equals_compute_without_fit_masks(self):
        r = _raster()
        fp = OutlierFilter((2, 12)) + OutlierFilter((0, 10))
        q, f = fp.compute_split(r)
        assert np.array_equal(q, fp.compute(r))
        assert np.array_equal(f, q)

    def test_fit_mask_only_narrows_the_fit(self):
        r = _raster()
        stable = np.ones((4, 4), bool)
        stable[:2] = False  # top half "moving"
        fp = OutlierFilter((0.5, 100)) + StableAreaMask(stable)
        q, f = fp.compute_split(r)
        assert q.sum() == 15  # only value 0 removed by quality
        assert f.sum() == 8  # bottom half, all >= 1
        assert q[0, 1]  # moving but good quality: kept in the output
        assert not f[0, 1]  # ...and excluded from the fit

    def test_fit_is_subset_of_quality(self):
        rng = np.random.default_rng(0)
        r = _raster(rng.normal(size=(20, 20)))
        stable = rng.random((20, 20)) > 0.4
        q, f = (OutlierFilter((-1, 1)) + StableAreaMask(stable)).compute_split(r)
        assert not (f & ~q).any()

    def test_each_mask_generated_once(self):
        calls = []

        class Spy(OutlierFilter):
            def generate_mask(self, raster, **kw):
                calls.append(type(self).__name__)
                return super().generate_mask(raster, **kw)

        class SpyFit(StableAreaMask):
            def generate_mask(self, raster, **kw):
                calls.append(type(self).__name__)
                return super().generate_mask(raster, **kw)

        fp = Spy((-100, 100)) + SpyFit(np.ones((4, 4), bool))
        fp.compute_split(_raster())
        assert sorted(calls) == ["Spy", "SpyFit"]

    def test_compute_and_apply_unchanged(self):
        """The legacy entry points still apply every mask, whatever its role."""
        r = _raster()
        stable = np.ones((4, 4), bool)
        stable[:2] = False
        fp = OutlierFilter((-1, 100)) + StableAreaMask(stable)
        assert fp.compute(r).sum() == 8
        xo, _ = fp.apply(r, r)
        assert xo.data.mask.sum() == 8


class TestApplyQuality:
    def test_fit_masks_do_not_remove_pixels(self):
        r = _raster()
        stable = np.ones((4, 4), bool)
        stable[:2] = False
        fp = OutlierFilter((0.5, 100)) + StableAreaMask(stable)
        xo, yo = fp.apply_quality(r, r)
        assert xo.data.mask.sum() == 1  # only value 0
        assert yo.data.mask.sum() == 1
        assert not xo.data.mask[0, 1]

    def test_preserves_existing_mask(self):
        data = ma.array(np.ones((4, 4)), mask=np.eye(4, dtype=bool))
        xo, _ = FilterPipeline([CCFilter(0.0)]).apply_quality(
            _raster(data), _raster(data), cc=_raster(np.ones((4, 4)))
        )
        assert xo.data.mask.sum() == 4

    def test_only_fit_masks_is_identity(self):
        r = _raster()
        fp = FilterPipeline([StableAreaMask(np.zeros((4, 4), bool))])
        xo, _ = fp.apply_quality(r, r)
        assert not np.ma.getmaskarray(xo.data).any()


class TestAsPipeline:
    def test_none(self):
        assert as_pipeline(None) is None

    def test_single_mask_wrapped(self):
        m = CCFilter(0.5)
        fp = as_pipeline(m)
        assert isinstance(fp, FilterPipeline) and fp.masks == [m]

    def test_pipeline_returned_as_is(self):
        fp = CCFilter(0.5) + OutlierFilter((-1, 1))
        assert as_pipeline(fp) is fp

    def test_rejects_other_types(self):
        with pytest.raises(TypeError):
            as_pipeline("CCFilter")
