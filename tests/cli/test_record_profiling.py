"""Stage records and the exception-safe profiler wrapper."""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from geomulticorr._logging import logger
from geomulticorr.cli import profiling
from geomulticorr.cli.record import (
    read_records,
    record_stage,
    write_record,
    StageRecord,
)


class TestRecord:
    def test_ok_record_written_with_warnings(self, tmp_path):
        with record_stage("pairs", tmp_path, config={"a": 1}, pzones=["P"]) as rec:
            logger.warning("watch out")
            logger.info("chatter")
            rec.summary["n"] = 3
        doc = read_records(tmp_path)["pairs"]
        assert doc["status"] == "ok"
        assert doc["summary"] == {"n": 3}
        assert doc["warnings"] == ["watch out"]
        assert doc["elapsed_s"] >= 0
        assert doc["written_utc"]

    def test_failed_record_and_reraise(self, tmp_path):
        with pytest.raises(RuntimeError):
            with record_stage("correct", tmp_path):
                raise RuntimeError("boom")
        doc = read_records(tmp_path)["correct"]
        assert doc["status"] == "failed"
        assert "boom" in doc["error"]

    def test_partial_kept(self, tmp_path):
        with record_stage("extract", tmp_path) as rec:
            rec.status = "partial"
        assert read_records(tmp_path)["extract"]["status"] == "partial"

    def test_history_appends(self, tmp_path):
        for _ in range(2):
            with record_stage("pairs", tmp_path):
                pass
        lines = (tmp_path / "records" / "history.jsonl").read_text().splitlines()
        assert len(lines) == 2 and json.loads(lines[0])["stage"] == "pairs"

    def test_write_failure_logged_not_raised(self, tmp_path, caplog_gmc):
        blocker = tmp_path / "records"
        blocker.write_text("a file where a folder should be")
        assert write_record(tmp_path, StageRecord(stage="pairs")) is None
        assert "Could not write" in caplog_gmc.text

    def test_artifact_relative_to_root(self, tmp_path):
        rec = StageRecord(stage="x")
        rec.add_artifact("fig", tmp_path / "a" / "b.png", tmp_path)
        rec.add_artifact("ext", "/elsewhere/c.png", tmp_path)
        assert rec.artifacts == {"fig": "a/b.png", "ext": "/elsewhere/c.png"}

    def test_nan_and_numpy_are_json_safe(self, tmp_path):
        import numpy as np

        with record_stage("x", tmp_path) as rec:
            rec.summary = {"a": np.float32(1.5), "b": (1, 2)}
        assert read_records(tmp_path)["x"]["summary"] == {"a": 1.5, "b": [1, 2]}


@pytest.fixture
def profiler():
    from geoutils.profiler import Profiler

    Profiler.enable(save_graphs=False, save_raw_data=True)
    yield Profiler
    Profiler.enable(save_graphs=False, save_raw_data=False)
    Profiler.running_processes.clear()


class TestProfileWrapper:
    def test_noop_when_disabled(self):
        from geoutils.profiler import Profiler

        from geomulticorr._profiling import profile

        Profiler.enable(False, False)

        @profile("t.f")
        def f(x):
            return x + 1

        assert f(1) == 2
        assert len(Profiler.get_profiling_info()) == 0

    def test_records_and_keeps_signature(self, profiler):
        import inspect

        from geomulticorr._profiling import profile

        @profile("t.g", memprof=True, interval=0.01)
        def g(a, b=2):
            # geoutils reads the samples right after signalling the sampler to
            # stop; an instant function can finish before the first sample.
            time.sleep(0.1)
            return a * b

        assert g(3) == 6
        assert list(inspect.signature(g).parameters) == ["a", "b"]
        info = profiler.get_profiling_info()
        assert info["name"].tolist() == ["t.g"]
        assert info["memory"].iloc[0]

    def test_exception_is_reraised_and_stack_is_clean(self, profiler):
        from geomulticorr._profiling import profile

        @profile("t.bad", memprof=True, interval=0.01)
        def bad():
            raise KeyError("nope")

        with pytest.raises(KeyError):
            bad()
        assert profiler.running_processes == []

        @profile("t.after")
        def after():
            return 1

        after()
        row = profiler.get_profiling_info().set_index("name").loc["t.after"]
        assert row["level"] == 0


def test_summary_table_aggregates():
    info = pd.DataFrame(
        {
            "name": ["a", "a", "b"],
            "time": [1.0, 2.0, 0.5],
            "memory": [[(0, 100.0), (1, 150.0)], None, None],
            "level": [0, 0, 1],
        }
    )
    rows = profiling.summary_table(info)
    assert rows[0] == {
        "name": "a",
        "calls": 2,
        "total_s": 3.0,
        "max_s": 2.0,
        "peak_mb": 150.0,
    }
    assert rows[1]["peak_mb"] is None
