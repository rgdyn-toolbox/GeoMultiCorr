"""update_pairs as the single pairing entry point: strategy="similarity",
strategy="manual", and the min_similarity filter on date strategies."""

from __future__ import annotations

import inspect

import pytest

from geomulticorr.core.session import Session
from tests.similarity.conftest import (
    PS_OTHER,
    PS_SAME_1,
    PS_SAME_2,
    SPOT,
    make_similarity_project,
)

KW = dict(kernel=21, search=4, tile=32, progress=False, plots=False)


@pytest.fixture
def session(tmp_path):
    return make_similarity_project(tmp_path)


@pytest.fixture
def measured(session):
    session.compute_thumbs_similarity("TestPz", **KW)
    return session


def _couples(layer):
    return {
        tuple(sorted((f"TestPz_{a}_{sa}", f"TestPz_{b}_{sb}")))
        for a, sa, b, sb in zip(
            layer.pa_left_date,
            layer.pa_left_sensor,
            layer.pa_right_date,
            layer.pa_right_sensor,
        )
    }


class TestSimilarityStrategy:
    def test_threshold_on_score(self, measured):
        sim = measured.load_thumbs_similarity("TestPz")
        thr = sim["score"].median()
        layer = measured.update_pairs(
            strategy="similarity", min_similarity=thr, both_directions=False
        )
        expected = {
            tuple(sorted(k))
            for k, s in zip(zip(sim.left_key, sim.right_key), sim.score)
            if s >= thr
        }
        assert _couples(layer) == expected
        assert set(layer.pa_direction) == {"forward"}

    def test_both_directions_default(self, measured):
        layer = measured.update_pairs(strategy="similarity", min_similarity=0.0)
        assert len(layer) == 12
        assert set(layer.pa_direction) == {"forward", "backward"}

    def test_bridges_per_image(self, measured):
        layer = measured.update_pairs(
            strategy="similarity",
            cross_sensor_only=True,
            per_image=1,
            both_directions=False,
        )
        assert len(layer) == 1
        assert _couples(layer) == {tuple(sorted((SPOT, PS_SAME_1)))} or _couples(
            layer
        ) == {tuple(sorted((SPOT, PS_SAME_2)))}

    def test_lower_is_better_metric(self, measured):
        sim = measured.load_thumbs_similarity("TestPz")
        thr = sim["trial_sigma_px"].median()
        layer = measured.update_pairs(
            strategy="similarity",
            similarity_metric="trial_sigma_px",
            min_similarity=thr,
            both_directions=False,
        )
        kept = sim[sim.trial_sigma_px <= thr]
        assert len(layer) == len(kept)

    def test_dt_and_sensor_filter(self, measured):
        layer = measured.update_pairs(
            strategy="similarity",
            sensor_filter="planetscope",
            max_dt_days="1Y",
            both_directions=False,
        )
        assert set(layer.pa_left_sensor) == {"planetscope"}
        assert (layer.pa_dt_days <= 365).all()

    def test_appends_to_date_pairs(self, measured):
        before = measured.update_pairs(strategy="consecutive", pz_name="TestPz")
        bridge = measured.update_pairs(
            strategy="similarity",
            cross_sensor_only=True,
            per_image=1,
            both_directions=False,
        )
        measured.update_pairs(strategy="consecutive", pz_name="TestPz")
        layer = measured.update_pairs(
            strategy="similarity",
            cross_sensor_only=True,
            per_image=1,
            both_directions=False,
            append=True,
        )
        # union of both, a pair already built by "consecutive" is not duplicated
        assert set(layer["pa_path"]) == set(before["pa_path"]) | set(bridge["pa_path"])

    def test_unvalidated_thumbs_skipped(self, measured):
        th = measured._thumbs
        th.loc[th["th_path"].str.contains(PS_SAME_1), "th_valid"] = "0"
        layer = measured.update_pairs(
            strategy="similarity", min_similarity=0.0, both_directions=False
        )
        assert len(layer) == 3  # the 3 couples without PS_SAME_1
        assert not layer.pa_path.str.contains(PS_SAME_1.split("_", 1)[1]).any()

    def test_missing_trace_raises(self, session):
        with pytest.raises(FileNotFoundError, match="compute_thumbs_similarity"):
            session.update_pairs(strategy="similarity", min_similarity=0.5)

    def test_unknown_metric(self, measured):
        with pytest.raises(ValueError, match="unknown metric"):
            measured.update_pairs(strategy="similarity", similarity_metric="nope")


class TestFilterOnDateStrategies:
    def test_min_similarity_drops_weak_pairs(self, measured):
        all_pairs = measured.update_pairs(strategy="redundancy", max_step=3)
        sim = measured.load_thumbs_similarity("TestPz")
        thr = sim["score"].median()
        kept = measured.update_pairs(
            strategy="redundancy", max_step=3, min_similarity=thr
        )
        assert 0 < len(kept) < len(all_pairs)
        strong = {
            tuple(sorted(k))
            for k, s in zip(zip(sim.left_key, sim.right_key), sim.score)
            if s >= thr
        }
        assert _couples(kept) <= strong

    def test_unmeasured_pairs_dropped_with_warning(self, session):
        session.compute_thumbs_similarity(
            "TestPz", couples="consecutive", **KW
        )  # only 3 of 6 couples measured
        layer = session.update_pairs(strategy="step", max_step=3, min_similarity=0.0)
        assert len(layer) == 3

    def test_bridge_arguments_rejected_on_date_strategies(self, measured):
        with pytest.raises(ValueError, match="strategy='similarity'"):
            measured.update_pairs(strategy="consecutive", per_image=2)
        with pytest.raises(ValueError, match="strategy='similarity'"):
            measured.update_pairs(strategy="consecutive", cross_sensor_only=True)


class TestManual:
    def test_tuples_and_frame(self, measured):
        layer = measured.update_pairs(
            strategy="manual", couples=[(SPOT, PS_SAME_2)], both_directions=False
        )
        assert len(layer) == 1 and layer.pa_direction.iloc[0] == "forward"
        sim = measured.load_thumbs_similarity("TestPz").head(2)
        layer = measured.update_pairs(strategy="manual", couples=sim)
        assert len(layer) == 4

    def test_dedupe_on_append(self, session):
        c = [(SPOT, PS_OTHER)]
        session.update_pairs(strategy="manual", couples=c)
        layer = session.update_pairs(strategy="manual", couples=c, append=True)
        assert len(layer) == 2

    def test_errors(self, session):
        with pytest.raises(ValueError, match="needs couples"):
            session.update_pairs(strategy="manual")
        with pytest.raises(ValueError, match="only used with strategy='manual'"):
            session.update_pairs(strategy="consecutive", couples=[(SPOT, PS_OTHER)])
        with pytest.raises(FileNotFoundError):
            session.update_pairs(strategy="manual", couples=[(SPOT, "TestPz_nope")])
        with pytest.raises(ValueError, match="itself"):
            session.update_pairs(strategy="manual", couples=[(SPOT, SPOT)])
        with pytest.raises(ValueError, match="at least one"):
            session.update_pairs(strategy="manual", couples=[])


def test_add_pairs_is_gone():
    assert not hasattr(Session, "add_pairs")


def test_pairing_explorer_stash_still_commits(session):
    """The seven-key stash of explore_pairs_strategy still splats."""
    session.explore_pairs_strategy(
        strategy="consecutive", interactive=False, savefig=False
    )
    layer = session.update_pairs(**session._last_pairs_params)
    assert len(layer) == 3


def test_new_parameters_have_defaults():
    sig = inspect.signature(Session.update_pairs)
    for name in (
        "min_similarity",
        "similarity_metric",
        "per_image",
        "cross_sensor_only",
        "both_directions",
        "couples",
    ):
        assert sig.parameters[name].default is not inspect.Parameter.empty
