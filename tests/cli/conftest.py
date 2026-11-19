"""A stub session recording every call, and a minimal config on a temp project."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from geomulticorr.cli.config import RunConfig


class StubSession:
    """Just enough of ``Session`` for the stages; every call is recorded."""

    def __init__(self, root: pathlib.Path, pzones=("Alpha",)):
        self.root = root
        self.calls: list[tuple[str, dict]] = []
        self.pzones = list(pzones)
        self.results: dict = {}

    def _rec(self, name, **kwargs):
        self.calls.append((name, kwargs))

    def called(self, name):
        return [kw for n, kw in self.calls if n == name]

    # ── read side ──────────────────────────────────────────────────────── #
    def get_pzones_overview(self, pz_name=""):
        return pd.DataFrame(
            {"pz_name": self.pzones, "pz_shortname": self.pzones, "pz_surf": 1.0}
        )

    def get_pairs_overview(self, criterias="", image_type=None):
        pz = self.pzones[0]
        base = self.root / pz / "image_correlation"
        return pd.DataFrame(
            {
                "pa_pz_name": [pz, pz],
                "pa_path": [
                    str(base / f"{pz}_2020-01-01_spot6__2021-01-01_spot7"),
                    str(base / f"{pz}_2021-01-01_spot7__2022-01-01_spot6"),
                ],
                "pa_left_date": ["2020-01-01", "2021-01-01"],
                "pa_right_date": ["2021-01-01", "2022-01-01"],
                "pa_left_sensor": ["spot6", "spot7"],
                "pa_right_sensor": ["spot7", "spot6"],
                "pa_dt_days": ["366", "365"],
                "pa_direction": ["forward", "forward"],
                "pa_status": ["corrected", "corrected"],
                "pa_ew_nmad_raw": [0.8, 1.2],
                "pa_ns_nmad_raw": [0.7, 1.0],
                "pa_ew_nmad_corr": [0.4, 0.9],
                "pa_ns_nmad_corr": [0.3, float("nan")],
            }
        )

    def get_thumbs_overview(self, criterias=""):
        return pd.DataFrame(
            {
                "th_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
                "th_sensor": ["spot6", "spot7", "spot6"],
                "th_res": [1.5, 1.5, 1.5],
                "th_type": ["opt", "opt", "dem"],
            }
        )

    def get_moving_areas(self, pz):
        self._rec("get_moving_areas", pz=pz)
        return "MOVING"

    def pz_dir(self, pz, kind=None):
        return self.root / pz / (kind or "")

    # ── write side ─────────────────────────────────────────────────────── #
    def update_pairs(self, **kwargs):
        self._rec("update_pairs", **kwargs)

    def save_pairs_figure(self, view, **kwargs):
        self._rec("save_pairs_figure", view=view, **kwargs)
        return {"png": self.root / "figures" / f"{view}.png"}

    def extract_pairs_raw_displacements(self, **kwargs):
        self._rec("extract_pairs_raw_displacements", **kwargs)
        return self.results.get("extract", {"a": {"ok": 1}, "b": None})

    def apply_pairs_corrections(
        self, correction_pipeline=None, filter_pipeline=None, **kwargs
    ):
        self._rec(
            "apply_pairs_corrections",
            correction_pipeline=correction_pipeline,
            filter_pipeline=filter_pipeline,
            **kwargs,
        )
        return {"a": {"ew": "x"}, "b": {"ew": "y"}}

    def prepare_pairs_correlation(self, **kwargs):
        self._rec("prepare_pairs_correlation", **kwargs)
        return ["s1", "s2"]

    def launch_pairs_correlation(
        self,
        criterias="",
        cluster=None,
        dry_run=False,
        overwrite=False,
        image_type=None,
    ):
        self._rec(
            "launch_pairs_correlation",
            criterias=criterias,
            cluster=cluster,
            dry_run=dry_run,
            overwrite=overwrite,
            image_type=image_type,
        )
        return [0, 1]

    def sync_pairs_after_cluster(self, **kwargs):
        self._rec("sync_pairs_after_cluster", **kwargs)


@pytest.fixture
def stub_session(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    return StubSession(root)


@pytest.fixture
def make_cfg(stub_session):
    def _make(**sections) -> RunConfig:
        data = {"project": {"path": str(stub_session.root)}, "pzones": ["Alpha"]}
        data.update(sections)
        return RunConfig.model_validate(data)

    return _make
