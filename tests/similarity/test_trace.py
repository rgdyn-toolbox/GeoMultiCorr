"""The similarity JSON trace (similarity/trace.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from geomulticorr.similarity import trace as tr


def _rows(n, seed=0, nan=False):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(
        {
            "left_key": [f"P_a{i}" for i in range(n)],
            "right_key": [f"P_b{i}" for i in range(n)],
            "trial_peak_median": rng.uniform(0.4, 0.9, n),
            "coh_mean": rng.uniform(0.1, 0.4, n),
            "trial_valid_frac": rng.uniform(0.7, 1.0, n),
            "trial_sigma_px": rng.uniform(0.3, 1.5, n),
            "flag": [True] * n,
            "count": np.arange(n, dtype=np.int64),
        }
    )
    if nan:
        f.loc[0, "coh_mean"] = np.nan
    return f


def test_roundtrip_strict_json(tmp_path):
    doc = tr.empty_trace("PZ", "opt")
    merged = tr.merge_run(doc, "h1", {"kernel": 21}, _rows(5, nan=True))
    path = tr.write_trace(tmp_path / "t.json", doc)

    def _bad(token):
        raise ValueError(token)

    raw = json.loads(path.read_text(), parse_constant=_bad)  # no NaN tokens
    assert raw["runs"]["h1"]["couples"][0]["coh_mean"] is None
    back = tr.run_frame(tr.read_trace(path))
    assert np.isnan(back.loc[0, "coh_mean"])
    pd.testing.assert_series_equal(back["score"], merged["score"], check_names=False)
    assert back["flag"].dtype == bool and back["count"].dtype.kind == "i"


def test_score_is_rank_over_run_and_follows_merges():
    doc = tr.empty_trace("PZ", "opt")
    a = tr.merge_run(doc, "h", {}, _rows(4, seed=1))
    assert a["score"].between(0, 1).all()
    more = _rows(6, seed=2)
    more["left_key"] = more["left_key"] + "x"
    b = tr.merge_run(doc, "h", {}, more)
    assert len(b) == 10 and len(doc["runs"]["h"]["couples"]) == 10
    common = a["left_key"]
    assert not np.allclose(
        a.set_index("left_key")["score"], b.set_index("left_key").loc[common, "score"]
    )


def test_remeasured_couple_replaces_old_row():
    doc = tr.empty_trace("PZ", "opt")
    tr.merge_run(doc, "h", {}, _rows(3))
    again = _rows(1)
    again.loc[0, "trial_peak_median"] = 0.123
    out = tr.merge_run(doc, "h", {}, again)
    assert len(out) == 3
    assert out.set_index("left_key").loc["P_a0", "trial_peak_median"] == 0.123


def test_latest_run_by_sequence_not_timestamp():
    doc = tr.empty_trace("PZ", "opt")
    tr.merge_run(doc, "first", {}, _rows(2))
    tr.merge_run(doc, "second", {}, _rows(2))
    assert tr.run_frame(doc).equals(tr.run_frame(doc, "second"))
    tr.merge_run(doc, "first", {}, pd.DataFrame())  # reused → latest again
    assert tr.run_frame(doc).equals(tr.run_frame(doc, "first"))


def test_errors(tmp_path):
    with pytest.raises(KeyError):
        tr.run_frame(tr.empty_trace("PZ", "opt"))
    doc = tr.empty_trace("PZ", "opt")
    tr.merge_run(doc, "h", {}, _rows(2))
    with pytest.raises(KeyError):
        tr.run_frame(doc, "nope")
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": 99}')
    with pytest.raises(ValueError):
        tr.read_trace(bad)
    with pytest.raises(FileNotFoundError):
        tr.read_trace(tmp_path / "missing.json")


def test_write_never_raises(tmp_path):
    target = tmp_path / "file"
    target.write_text("x")
    assert (
        tr.write_trace(target / "sub" / "t.json", tr.empty_trace("PZ", "opt")) is None
    )
