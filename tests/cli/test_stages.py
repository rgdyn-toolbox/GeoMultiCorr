"""Each stage against a stub session: the calls it makes and the record it fills."""

from __future__ import annotations

import pytest

from geomulticorr.cli.record import StageRecord
from geomulticorr.cli.stages import stage_runner
from geomulticorr.cli.stages._common import (
    StageContext,
    check_kwargs,
    pz_criterias,
)
from geomulticorr.corrections import CorrectionPipeline, FilterPipeline


def _run(stage, cfg, session, tmp_path):
    ctx = StageContext(cfg=cfg, run_dir=tmp_path / "run", session=session)
    rec = StageRecord(stage=stage)
    stage_runner(stage)(ctx, getattr(cfg, stage), rec)
    return rec


def test_pz_criterias_lowercased_and_scoped():
    assert pz_criterias("PasDeLours", ["SPOT"]) == ["/pasdelours/", "spot"]
    assert pz_criterias("A", "") == ["/a/"]


def test_check_kwargs_against_real_session():
    from geomulticorr.core.session import Session

    check_kwargs(Session.prepare_pairs_correlation, {"corr_kernel": [9, 9]}, "x")
    with pytest.raises(TypeError, match="corr_kernal"):
        check_kwargs(Session.prepare_pairs_correlation, {"corr_kernal": 1}, "x")


class TestPairs:
    def test_groups_and_forced_append(self, make_cfg, stub_session, tmp_path):
        cfg = make_cfg(
            pairs={
                "groups": [
                    {
                        "strategy": "redundancy",
                        "max_step": 2,
                        "sensor_filter": "planetscope",
                    },
                    {"strategy": "redundancy", "max_step": 1, "sensor_filter": "spot"},
                ],
                "figures": ["chord"],
            }
        )
        rec = _run("pairs", cfg, stub_session, tmp_path)
        calls = stub_session.called("update_pairs")
        assert [c["append"] for c in calls] == [False, True]
        assert calls[0]["pz_name"] == "Alpha" and calls[0]["image_type"] == "opt"
        assert calls[1]["sensor_filter"] == "spot"
        (fig,) = stub_session.called("save_pairs_figure")
        assert fig["criterias"] == ["/alpha/"] and fig["pz_name"] == "Alpha"
        assert "Alpha/pairs_chord.png" in rec.artifacts
        summary = rec.summary["pzones"]["Alpha"]
        assert summary["n_pairs"] == 2 and summary["n_components"] == 1
        assert len(summary["thumbs"]) == 2  # the dem thumb is filtered out
        assert rec.pairs[0]["dt_days"] == 366.0


class TestCorrect:
    def test_pipelines_and_auto(self, make_cfg, stub_session, tmp_path):
        cfg = make_cfg(
            correct={
                "masks": [
                    {"CCFilter": {"cc_threshold": 0.5}},
                    {"StableAreaMask": "auto"},
                ],
                "corrections": ["MedianCentering", "RampCorrection"],
                "criterias": "SPOT",
            }
        )
        rec = _run("correct", cfg, stub_session, tmp_path)
        (call,) = stub_session.called("apply_pairs_corrections")
        assert isinstance(call["correction_pipeline"], CorrectionPipeline)
        assert isinstance(call["filter_pipeline"], FilterPipeline)
        assert call["filter_pipeline"].masks[1].stable_mask == "MOVING"
        assert call["criterias"] == ["/alpha/", "spot"]
        pz = rec.summary["pzones"]["Alpha"]
        assert pz["n_pairs"] == 2 and pz["n_failed"] == 0
        assert pz["median_ew_nmad_ratio"] == pytest.approx((0.5 + 0.75) / 2)
        assert rec.summary["corrections"][0] == {"MedianCentering": {"stat": "median"}}


class TestExtract:
    def test_failed_pairs_make_partial(self, make_cfg, stub_session, tmp_path):
        cfg = make_cfg(extract={"target_resolution": 3.0})
        rec = _run("extract", cfg, stub_session, tmp_path)
        (call,) = stub_session.called("extract_pairs_raw_displacements")
        assert call["target_resolution"] == 3.0
        assert rec.status == "partial"
        assert rec.summary["pzones"]["Alpha"]["failed"] == ["b"]


class TestCorrelate:
    def test_prepare_then_launch(self, make_cfg, stub_session, tmp_path):
        cfg = make_cfg(correlate={"options": {"corr_kernel": [9, 9]}, "dry_run": True})
        rec = _run("correlate", cfg, stub_session, tmp_path)
        (prep,) = stub_session.called("prepare_pairs_correlation")
        assert prep["corr_kernel"] == [9, 9] and prep["plan"] is None
        (launch,) = stub_session.called("launch_pairs_correlation")
        assert launch["dry_run"] is True and "corr_kernel" not in launch
        entry = rec.summary["pzones"]["Alpha"]
        assert entry["n_scripts"] == 2 and entry["n_failed"] == 1
        assert rec.status == "partial"

    def test_sync_only(self, make_cfg, stub_session, tmp_path):
        cfg = make_cfg(correlate={"sync": True})
        _run("correlate", cfg, stub_session, tmp_path)
        assert stub_session.called("sync_pairs_after_cluster")
        assert not stub_session.called("prepare_pairs_correlation")
