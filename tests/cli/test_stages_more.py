"""prepare / crop / similarity / invert / fuse stages, with the heavy objects faked."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from geomulticorr.cli.record import StageRecord
from geomulticorr.cli.stages import stage_runner
from geomulticorr.cli.stages._common import StageContext


def _run(stage, cfg, session, tmp_path):
    ctx = StageContext(cfg=cfg, run_dir=tmp_path / "run", session=session)
    rec = StageRecord(stage=stage)
    stage_runner(stage)(ctx, getattr(cfg, stage), rec)
    return rec


def _recorder(session, name, result=None):
    def method(*args, **kwargs):
        session.calls.append((name, {"args": args, **kwargs}))
        return result(*args, **kwargs) if callable(result) else result

    setattr(session, name, method)


class TestPrepare:
    def test_creates_missing_pzone_and_layers(self, make_cfg, stub_session, tmp_path):
        for name in ("insert_pzone_from_file", "insert_pzone"):
            _recorder(stub_session, name)
        for name in (
            "build_reference_grid",
            "build_reference_dem",
            "load_moving_areas",
        ):
            _recorder(stub_session, name, stub_session.root / "Alpha" / f"{name}.tif")
        cfg = make_cfg(
            prepare={
                "pzones": [
                    {"name": "Alpha", "shortname": "A", "source": "/x.gpkg"},
                    {
                        "name": "Beta",
                        "shortname": "B",
                        "coordinates": [[0, 0], [1, 0], [1, 1]],
                    },
                ],
                "reference_grid": {"resolution": 3.0},
                "reference_dem": {"source": "/dem.tif", "pz_name": "Alpha"},
                "moving_areas": {"source": "/m.gpkg", "buffer": 10},
            }
        )
        rec = _run("prepare", cfg, stub_session, tmp_path)
        assert not stub_session.called("insert_pzone_from_file")  # Alpha exists
        (beta,) = stub_session.called("insert_pzone")
        assert beta["args"][1] == "Beta"
        assert stub_session.called("build_reference_grid")[0]["resolution"] == 3.0
        assert stub_session.called("load_moving_areas")[0]["buffer"] == 10
        assert rec.artifacts["Alpha/reference_dem"] == "Alpha/build_reference_dem.tif"
        assert rec.summary["created"] == ["Beta"]

    def test_unknown_pz_name(self, make_cfg, stub_session, tmp_path):
        cfg = make_cfg(prepare={"reference_grid": {"resolution": 3, "pz_name": "Nope"}})
        with pytest.raises(ValueError, match="Nope"):
            _run("prepare", cfg, stub_session, tmp_path)

    def test_pzone_needs_geometry(self, make_cfg, stub_session, tmp_path):
        cfg = make_cfg(prepare={"pzones": {"name": "Gamma", "shortname": "G"}})
        with pytest.raises(ValueError, match="source"):
            _run("prepare", cfg, stub_session, tmp_path)


class TestCrop:
    def test_map_sieve_register(self, make_cfg, stub_session, tmp_path):
        _recorder(stub_session, "map_georasters_bank", pd.DataFrame({"a": [1, 2, 3]}))
        _recorder(stub_session, "sieve_bulk", {"Alpha": ["t1", "t2"]})
        _recorder(stub_session, "update_thumbs")
        _recorder(stub_session, "register_existing_thumbs", ["x"])
        _recorder(stub_session, "backfill_thumb_resolution")
        cfg = make_cfg(
            crop={
                "georasters_bank_path": "/bank",
                "target_resolution": 3.0,
                "exclude_patterns": ["_old"],
                "register_thumbs": {"sources": "/done", "pz_name": "Alpha"},
                "backfill_resolution": True,
            }
        )
        rec = _run("crop", cfg, stub_session, tmp_path)
        (mapping,) = stub_session.called("map_georasters_bank")
        assert (
            mapping["exclude_patterns"] == ["_old"]
            and "target_resolution" not in mapping
        )
        (sieve,) = stub_session.called("sieve_bulk")
        assert sieve["target_resolution"] == 3.0 and sieve["pzone_filter"] == ["Alpha"]
        assert "exclude_patterns" not in sieve and "georasters_bank_path" not in sieve
        assert stub_session.called("update_thumbs")
        assert rec.summary["written"] == {"Alpha": 2}
        assert rec.summary["registered"] == {"Alpha": 1}
        assert rec.summary["n_thumbs"] == 2 and rec.summary["thumbs_per_sensor"]


class TestSimilarity:
    def test_summary_and_top(self, make_cfg, stub_session, tmp_path):
        frame = pd.DataFrame(
            {
                "left_key": ["a", "b"],
                "right_key": ["c", "d"],
                "trial_peak_median": [0.2, 0.9],
                "coh_mean": [0.1, np.nan],
            }
        )
        _recorder(stub_session, "compute_thumbs_similarity", frame)
        _recorder(stub_session, "similarity_path", stub_session.root / "sim.csv")
        cfg = make_cfg(similarity={"report_kwargs": {"kernel": 15}})
        rec = _run("similarity", cfg, stub_session, tmp_path)
        (call,) = stub_session.called("compute_thumbs_similarity")
        assert call["kernel"] == 15 and call["progress"] is False
        assert "report_kwargs" not in call
        entry = rec.summary["pzones"]["Alpha"]
        assert entry["n_couples"] == 2
        assert (
            entry["top"][0]["left_key"] == "b" and entry["top"][0]["coh_mean"] is None
        )


class FakeInversion:
    instances: list = []

    def __init__(
        self, session, pairs, inversion_name, filter_pipeline, tio_binaries_dir
    ):
        self.session, self.pairs, self.name = session, list(pairs), inversion_name
        self.filter_pipeline = filter_pipeline
        self.inversion_dir = session.root / "Alpha" / "inversion" / inversion_name
        self.calls = []
        FakeInversion.instances.append(self)

    @property
    def image_dates(self):
        return ["20200101", "20210101", "20220101"]

    def filter_pairs_by_nmad(self, threshold):
        dropped = self.pairs[1:]
        self.pairs = self.pairs[:1]
        return self.pairs, dropped

    def prepare_inversion(self, weight_mode=None, solver=None, sync_geodb=True):
        self.calls.append(("prepare", weight_mode, solver))

    def network_components(self):
        return [{"dates": ["20200101"], "sensors": ["spot6"]}, {"dates": ["20210101"]}]

    def save_weights_figure(self, formats=()):
        return {"png": self.inversion_dir / "w.png"}

    def launch(self, mode="local", write_residuals=False):
        import rasterio

        self.calls.append(("launch", mode))
        for d in ("EW", "NS"):
            folder = self.inversion_dir / f"inverse_{d}"
            (folder / "diagnostics").mkdir(parents=True, exist_ok=True)
            (folder / f"inverse_{d}_parameters.json").write_text(
                json.dumps({"weights": {"mode": "uniform", "summary": {"n": 2}}})
            )
            (folder / f"TOT_20200101_{d}.tif").write_bytes(b"")
            with rasterio.open(
                folder / "diagnostics" / "rms.tif",
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="float32",
            ) as dst:
                dst.write(np.array([[1, 2], [3, np.nan]], "float32"), 1)

    def post_process(self):
        self.calls.append(("post_process",))


@pytest.fixture
def fake_inversion(monkeypatch, stub_session):
    import geomulticorr.inversion.tio_inversion as tio

    FakeInversion.instances = []
    monkeypatch.setattr(tio, "TIOInversion", FakeInversion)
    stub_session.get_pairs = lambda criterias="", image_type=None: (
        stub_session.calls.append(("get_pairs", {"criterias": criterias}))
        or ["p1", "p2"]
    )
    return FakeInversion


class TestInvert:
    def test_full_run(self, make_cfg, stub_session, tmp_path, fake_inversion):
        cfg = make_cfg(
            invert={
                "runs": {
                    "name": "inv",
                    "criterias": "spot",
                    "masks": [{"CCFilter": {"cc_threshold": 0.4}}],
                    "nmad_threshold": 0.7,
                    "weights": {"weight_mode": "quality"},
                    "solver": {"reweight_iterations": 2},
                    "mode": "local",
                }
            }
        )
        rec = _run("invert", cfg, stub_session, tmp_path)
        (inv,) = fake_inversion.instances
        assert stub_session.called("get_pairs")[0]["criterias"] == ["/alpha/", "spot"]
        assert inv.filter_pipeline.masks[0].cc_threshold == 0.4
        kind, mode, solver = inv.calls[0]
        assert (kind, mode, solver.reweight_iterations) == ("prepare", "quality", 2)
        assert ("post_process",) in inv.calls  # local Fortran → post-process
        (run,) = rec.summary["runs"]
        assert run["n_pairs_dropped_nmad"] == 1 and run["n_pairs"] == 1
        assert run["disconnected"] is True
        assert run["EW_weights"]["summary"] == {"n": 2}
        assert run["EW_rms"]["p50"] == pytest.approx(2.0)
        assert run["n_tot_dates"] == {"EW": 1, "NS": 1}
        assert "Alpha/inv/weights.png" in rec.artifacts

    def test_python_mode_skips_post_process(
        self, make_cfg, stub_session, tmp_path, fake_inversion
    ):
        cfg = make_cfg(
            invert={"runs": {"name": "inv", "mode": "python", "weights_figure": False}}
        )
        _run("invert", cfg, stub_session, tmp_path)
        assert ("post_process",) not in fake_inversion.instances[0].calls

    def test_no_pairs_raises(self, make_cfg, stub_session, tmp_path, fake_inversion):
        stub_session.get_pairs = lambda **kw: []
        cfg = make_cfg(invert={"runs": {"name": "inv"}})
        with pytest.raises(ValueError, match="no pair"):
            _run("invert", cfg, stub_session, tmp_path)

    def test_unknown_weight_key(self, make_cfg, stub_session, tmp_path, fake_inversion):
        cfg = make_cfg(
            invert={"runs": {"name": "inv", "weights": {"wieght_mode": "x"}}}
        )
        with pytest.raises(TypeError, match="wieght_mode"):
            _run("invert", cfg, stub_session, tmp_path)


class TestFuse:
    def test_fusion_and_closure(self, make_cfg, stub_session, tmp_path, monkeypatch):
        import geomulticorr.inversion._stack as stack
        import geomulticorr.inversion.fusion as fusion

        seen = {}

        def fake_fuse(ref, sec, out, **kwargs):
            seen.update(ref=ref, sec=sec, out=out, **kwargs)
            out.mkdir(parents=True, exist_ok=True)
            (out / "fusion_parameters.json").write_text(
                json.dumps({"model": kwargs["model"]})
            )

        table = pd.DataFrame(
            {
                "pair": ["p", "p"],
                "component": ["EW", "NS"],
                "rate_bias_m_yr": [0.1, -0.2],
                "nmad": [0.3, 0.4],
            }
        )
        monkeypatch.setattr(fusion, "fuse_inversions", fake_fuse)
        monkeypatch.setattr(
            fusion, "pair_closure_against_series", lambda pairs, series: "C"
        )
        monkeypatch.setattr(fusion, "closure_summary", lambda closure: table)
        monkeypatch.setattr(stack, "load_cumulative_stack", lambda d: "S")
        stub_session.get_pairs = lambda criterias="", image_type=None: ["p"]
        cfg = make_cfg(
            fuse={
                "runs": {
                    "name": "fused",
                    "reference": "a",
                    "secondary": "b",
                    "model": "offset",
                    "closure_reference_criterias": "spot",
                }
            }
        )
        rec = _run("fuse", cfg, stub_session, tmp_path)
        assert seen["ref"].name == "a" and seen["out"].name == "fused"
        assert seen["model"] == "offset"
        (run,) = rec.summary["runs"]
        assert run["fusion"] == {"model": "offset"}
        assert run["closure_reference_pairs"]["EW"]["median_rate_bias_m_yr"] == 0.1
        assert "closure_secondary_pairs" not in run
        assert pathlib.Path(rec.artifacts["Alpha/fused/closure_reference"]).exists()
