#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# test_validation.py
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
"""Similarity vs realised correlation (similarity/validation.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomulticorr.similarity import validation as val


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "Chimborazo_2016-11-17_planetscope_3m__2017-08-14_planetscope_3m",
            (
                "Chimborazo_2016-11-17_planetscope_3m",
                "Chimborazo_2017-08-14_planetscope_3m",
            ),
        ),
        (  # backward pair → same sorted couple
            "PDL_2021-08-29_spot7__2018-07-29_planetscope",
            ("PDL_2018-07-29_planetscope", "PDL_2021-08-29_spot7"),
        ),
        ("PasDeLours_2014-09-01-spot6_2018-09-27-spot6", None),  # old '-' key
        ("PDL_a__b__c", None),
        ("PDL___b", None),
    ],
)
def test_couple_key_from_pair_name(name, expected):
    assert val.couple_key_from_pair_name(name) == expected


def test_spearman_table():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    f = pd.DataFrame(
        {
            "trial_peak_median": x,
            "dt_days": rng.normal(size=40),
            "cc_median": x + 0.01 * rng.normal(size=40),
            "nmad_px": -x,
            "constant": 1.0,
        }
    )
    t = val.spearman_table(
        f, predictors=("trial_peak_median", "dt_days", "constant", "absent")
    )
    assert list(t.index) == ["trial_peak_median", "dt_days", "constant"]
    assert list(t.columns) == ["cc_median", "nmad_px"]
    assert t.loc["trial_peak_median", "cc_median"] > 0.95
    assert t.loc["trial_peak_median", "nmad_px"] == pytest.approx(-1.0)
    assert t.loc["constant"].isna().all()


def test_spearman_too_few_rows_is_nan():
    f = pd.DataFrame({"score": [1, 2, 3], "cc_median": [1, 2, 3]})
    assert val.spearman_table(f).isna().all().all()


def test_match_outcomes():
    sim = pd.DataFrame(
        {
            "left_key": ["A", "B"],
            "right_key": ["B", "C"],
            "res": [3.0, 1.5],
            "score": [0.9, 0.1],
        }
    )
    outcomes = pd.DataFrame(
        {"cc_median": [0.8], "nmad_m": [1.5]}, index=pd.Index([("A", "B")])
    )
    m = val.match_outcomes(sim, outcomes)
    assert len(m) == 1 and m["nmad_px"].item() == 0.5 and "couple" not in m


def _block(corrected=True):
    return {
        "before": {"trial_peak_median": 0.7},
        "after": {
            "raw": {
                "indices": {"trial_peak_median": 0.9, "coh_mean": None},
                "disp_valid_frac": 0.95,
                "ncc_median": 0.8,
                "ncc_q50": 0.9,
                "sources": {"disp": 1, "ncc": 2},
            },
            "corrected": (
                {
                    "indices": {"trial_peak_median": 0.85},
                    "disp_valid_frac": 0.9,
                    "nmad_m": 1.2,
                    "nmad_px": 0.4,
                    "sources": {"ew": 3, "ns": 4},
                }
                if corrected
                else None
            ),
        },
    }


def test_flatten_similarity_block():
    row = val.flatten_similarity_block(_block())
    assert row["trial_peak_median_raw"] == 0.9
    assert row["trial_peak_median_corr"] == 0.85
    assert np.isnan(row["coh_mean_raw"])  # null → NaN
    assert row["disp_valid_frac_raw"] == 0.95 and row["disp_valid_frac_corr"] == 0.9
    assert row["ncc_median"] == 0.8 and row["ncc_q50"] == 0.9
    assert row["nmad_corr_m"] == 1.2 and row["nmad_corr_px"] == 0.4
    assert "sources" not in row and "trial_peak_median" not in row


def test_flatten_without_corrections():
    row = val.flatten_similarity_block(_block(corrected=False))
    assert "trial_peak_median_raw" in row
    assert not any(k.endswith("_corr") for k in row) and "nmad_corr_m" not in row
    assert val.flatten_similarity_block(None) == {}


def test_with_asp_twins():
    assert val.with_asp_twins(["a", "b"]) == [
        "a",
        "b",
        "a_raw",
        "b_raw",
        "a_corr",
        "b_corr",
    ]


def test_before_after_table():
    x = np.arange(8.0)
    frame = pd.DataFrame(
        {"coh_mean": x, "coh_mean_raw": x * 2, "coh_mean_corr": -x, "dt_days": x}
    )
    table = val.before_after_table(frame)
    assert list(table.columns) == ["before_vs_raw", "before_vs_corr"]
    assert list(table.index) == ["coh_mean"]  # dt_days has no twin
    assert table.loc["coh_mean", "before_vs_raw"] == pytest.approx(1.0)
    assert table.loc["coh_mean", "before_vs_corr"] == pytest.approx(-1.0)
