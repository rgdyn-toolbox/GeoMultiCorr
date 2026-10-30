"""Choosing cross-sensor bridges from a similarity frame (similarity/selection.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomulticorr.similarity import selection as sel


def _frame():
    """2 SPOT images × 4 PlanetScope images, plus same-sensor rows to ignore.

    SPOT-A's couples are all better than SPOT-B's, so a plain top-N would
    pick only SPOT-A — the case per-image selection exists for.
    """
    rows = []
    spots = {"S_2018_spot6": 0.0, "S_2021_spot7": -0.2}
    for s, penalty in spots.items():
        for k, ps in enumerate(["P_2017", "P_2019", "P_2020", "P_2022"]):
            rows.append(
                dict(
                    left_key=ps,
                    right_key=s,
                    left_sensor="planetscope",
                    right_sensor="spot6" if "spot6" in s else "spot7",
                    dt_days=200 * (k + 1),
                    trial_peak_median=0.8 - 0.05 * k + penalty,
                    coh_mean=0.3 - 0.01 * k + penalty / 10,
                    trial_valid_frac=0.99 - 0.01 * k + penalty / 10,
                    trial_sigma_px=0.5 + 0.1 * k - penalty,
                )
            )
    rows.append(
        dict(
            left_key="S_2018_spot6",
            right_key="S_2021_spot7",
            left_sensor="spot6",
            right_sensor="spot7",
            dt_days=1000,
            trial_peak_median=0.99,
            coh_mean=0.9,
            trial_valid_frac=1.0,
            trial_sigma_px=0.1,
        )
    )
    rows.append(
        dict(
            left_key="P_2017",
            right_key="P_2019",
            left_sensor="planetscope",
            right_sensor="planetscope",
            dt_days=700,
            trial_peak_median=0.99,
            coh_mean=0.9,
            trial_valid_frac=1.0,
            trial_sigma_px=0.1,
        )
    )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "sensor, family",
    [
        ("spot6", "spot"),
        ("SPOT7", "spot"),
        ("planetscope", "planetscope"),
        ("landsat8", "landsat"),
        ("sentinel-2", "sentinel"),
        ("pleiades-1a", "pleiades"),
        ("worldview3", "worldview"),
    ],
)
def test_sensor_family(sensor, family):
    assert sel.sensor_family(sensor) == family


class TestRankScore:
    def test_direction_respected(self):
        f = pd.DataFrame(
            {
                "trial_peak_median": [0.9, 0.5],
                "coh_mean": [0.3, 0.1],
                "trial_valid_frac": [1.0, 0.8],
                "trial_sigma_px": [0.4, 1.2],
            }
        )
        s = sel.rank_score(f)
        assert s[0] == 1.0 and s[1] < s[0]

    def test_scale_free(self):
        """Multiplying one metric by 1000 changes nothing."""
        f = _frame()
        g = f.assign(trial_sigma_px=f.trial_sigma_px * 1000)
        pd.testing.assert_series_equal(sel.rank_score(f), sel.rank_score(g))

    def test_missing_column(self):
        with pytest.raises(KeyError, match="coh_mean"):
            sel.rank_score(pd.DataFrame({"trial_peak_median": [1.0]}))


class TestSelectBridges:
    def test_every_anchor_image_gets_partners(self):
        out = sel.select_bridge_couples(_frame(), per_image=2, min_valid_frac=None)
        assert out.groupby("anchor_key").size().to_dict() == {
            "S_2018_spot6": 2,
            "S_2021_spot7": 2,
        }
        assert set(out.rank_in_anchor) == {1, 2}

    def test_same_family_never_a_bridge(self):
        out = sel.select_bridge_couples(_frame(), per_image=10, min_valid_frac=None)
        assert (out.family_left != out.family_right).all()
        assert len(out) == 8

    def test_minority_anchor_is_spot(self):
        out = sel.select_bridge_couples(_frame(), min_valid_frac=None)
        assert out.anchor_key.str.contains("spot").all()
        assert not out.partner_key.str.contains("spot").any()

    def test_explicit_anchor(self):
        out = sel.select_bridge_couples(
            _frame(), anchor="planetscope", per_image=1, min_valid_frac=None
        )
        assert set(out.anchor_key) == {"P_2017", "P_2019", "P_2020", "P_2022"}
        with pytest.raises(ValueError, match="not a sensor family"):
            sel.select_bridge_couples(_frame(), anchor="landsat")

    def test_best_partner_first(self):
        out = sel.select_bridge_couples(_frame(), per_image=1, min_valid_frac=None)
        assert set(out.partner_key) == {"P_2017"}

    def test_gates(self):
        f = _frame()
        out = sel.select_bridge_couples(
            f, per_image=10, max_dt_days=400, min_valid_frac=None
        )
        assert (out.dt_days <= 400).all() and len(out) == 4
        out = sel.select_bridge_couples(
            f, per_image=10, min_peak=0.7, min_valid_frac=None
        )
        assert (out.trial_peak_median >= 0.7).all()
        out = sel.select_bridge_couples(
            f, per_image=10, max_sigma_px=0.65, min_valid_frac=None
        )
        assert (out.trial_sigma_px <= 0.65).all()

    def test_everything_gated_out_is_empty_not_error(self):
        out = sel.select_bridge_couples(_frame(), min_peak=0.999)
        assert out.empty and "score" in out.columns

    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            sel.select_bridge_couples(_frame(), per_image=0)
        with pytest.raises(KeyError, match="dt_days"):
            sel.select_bridge_couples(_frame().drop(columns="dt_days"))


class TestCoverage:
    def test_reports_unbridged_images(self):
        f = _frame()
        out = sel.select_bridge_couples(
            f, min_peak=0.7, per_image=2, min_valid_frac=None
        )
        cov = sel.bridge_coverage(f, out).set_index("anchor_key")
        assert cov.loc["S_2018_spot6", "n_bridges"] == 2
        assert cov.loc["S_2021_spot7", "n_bridges"] == 0
        assert np.isnan(cov.loc["S_2021_spot7", "best_score"])

    def test_empty_selection(self):
        assert sel.bridge_coverage(
            _frame(), sel.select_bridge_couples(_frame(), min_peak=0.999)
        ).empty


class TestSelectCouples:
    def test_stored_score_used_not_recomputed(self):
        f = _frame()
        f["score"] = np.linspace(0, 1, len(f))
        out = sel.select_couples(f, threshold=0.5)
        assert (out["score"] >= 0.5).all() and len(out) == (f["score"] >= 0.5).sum()

    def test_score_computed_over_input_when_absent(self):
        f = _frame()
        expected = dict(zip(zip(f.left_key, f.right_key), sel.rank_score(f)))
        out = sel.select_couples(f)
        assert len(out) == len(f)
        for k, s in zip(zip(out.left_key, out.right_key), out.score):
            assert s == pytest.approx(expected[k])

    def test_lower_is_better_metric(self):
        out = sel.select_couples(_frame(), metric="trial_sigma_px", threshold=0.6)
        assert (out["trial_sigma_px"] <= 0.6).all() and len(out) > 0

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="unknown metric"):
            sel.select_couples(_frame(), metric="nope")

    def test_per_image_without_cross_counts_both_ends(self):
        out = sel.select_couples(_frame(), per_image=1)
        imgs = pd.concat([out.left_key, out.right_key])
        full = pd.concat([_frame().left_key, _frame().right_key]).unique()
        assert set(imgs) == set(full)  # every image keeps at least its best couple
        assert len(out) < len(_frame())

    def test_cross_and_per_image_match_bridges(self):
        a = sel.select_couples(_frame(), cross_sensor_only=True, per_image=2)
        b = sel.select_bridge_couples(_frame(), per_image=2, min_valid_frac=None)
        assert set(zip(a.anchor_key, a.partner_key)) == set(
            zip(b.anchor_key, b.partner_key)
        )

    def test_per_image_none_keeps_all(self):
        assert len(sel.select_couples(_frame(), cross_sensor_only=True)) == 8

    def test_bad_per_image(self):
        with pytest.raises(ValueError):
            sel.select_couples(_frame(), per_image=0)


class TestMatrixHelpers:
    def test_order_by_date(self):
        f = pd.DataFrame(
            {
                "left_key": ["b", "a"],
                "right_key": ["c", "b"],
                "left_date": ["2020-01-02", "2019-01-01"],
                "right_date": ["2021-01-01", "2020-01-02"],
                "left_sensor": ["x", "y"],
                "right_sensor": ["z", "x"],
                "score": [0.2, 0.9],
            }
        )
        assert sel.thumb_order(f) == ["a", "b", "c"]
        m = sel.similarity_matrix(f, "score")
        assert m.loc["a", "b"] == m.loc["b", "a"] == 0.9
        assert np.isnan(m.loc["a", "c"]) and np.isnan(m.loc["a", "a"])
        assert sel.thumb_sensors(f) == {"b": "x", "a": "y", "c": "z"}
