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
