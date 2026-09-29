#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``input_tio`` is rendered from ``TIOConfig`` — and the weights finally reach the solver.

Until 0.6.3 ``_build_input_tio_text`` hard-coded the *"weighting by
interferogram variance"* line to ``0``; the Fortran reads ``liste_couple``'s
third column only when it is ``2``, so every pair weight GMC ever wrote was
ignored. These tests pin the new default, the byte-exact reproduction of the
old block through ``TIOConfig.legacy()``, the storage of the resolved config
on the instance (so the trace records what was written), and the warning that
fires when weights are written the solver will not read.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import geomulticorr.inversion.tio_inversion as tio
from geomulticorr.inversion import TIOConfig
from geomulticorr.inversion.pytio import IPONDER_LINE_INDEX
from geomulticorr.inversion.tio_inversion import TIOInversion, _build_input_tio_text
from tests.inversion.pytio.test_config import LEGACY_TEXT


@pytest.fixture
def inv(tmp_path, monkeypatch):
    monkeypatch.setattr(tio, "load_pair_stats", lambda p: {
        "final_corrected_stats": {"ew": {"nmad": 0.3}, "ns": {"nmad": 0.3}},
        "raw_corr_stats": {"cc": {"cc_quality_gte_050": 0.7}},
    })
    obj = TIOInversion.__new__(TIOInversion)
    obj.pairs = [
        SimpleNamespace(pa_key=f"PZ_p{i}", pa_dt_days=30 * (i + 1),
                        pa_left=SimpleNamespace(th_date="2022-01-01"),
                        pa_right=SimpleNamespace(th_date=f"2022-0{i + 2}-01"))
        for i in range(3)
    ]
    obj.inversion_name = "t"
    obj.pzone_name = "PZ"
    obj.inversion_dir = tmp_path
    obj._DIRECTIONS = ("EW", "NS")
    obj._image_dates = None
    obj._quality_metrics = None
    obj._last_weights = None
    obj._last_weights_params = None
    obj.solver = TIOConfig()
    obj.session = SimpleNamespace(sync_pairs_weights=lambda *a, **k: None)
    for d in obj._DIRECTIONS:
        (tmp_path / f"inverse_{d}").mkdir()
    return obj


class TestBuildInputTioText:
    def test_default_requests_the_file_weights(self):
        lines = _build_input_tio_text().splitlines()
        assert lines[IPONDER_LINE_INDEX].startswith("2 ")

    def test_legacy_reproduces_the_old_block_verbatim(self):
        assert _build_input_tio_text(solver=TIOConfig.legacy()) == LEGACY_TEXT

    def test_filenames_still_flow_through(self):
        lines = _build_input_tio_text("img", "cpl").splitlines()
        assert lines[10] == "img" and lines[12] == "cpl"

    def test_a_tuned_config_lands_in_the_text(self):
        lines = _build_input_tio_text(solver=TIOConfig(gamma=0.02, reweight_iterations=2)).splitlines()
        assert lines[0].startswith("0.0200 ")
        assert lines[5].startswith("2 ")


class TestWriteInputTio:
    def test_writes_both_directions_with_the_default(self, inv):
        inv.write_input_tio()
        for d in inv._DIRECTIONS:
            text = (inv.inversion_dir / f"inverse_{d}" / "input_tio").read_text()
            assert text.splitlines()[IPONDER_LINE_INDEX].startswith("2 ")
        assert inv.solver.weight_mode == "file"

    def test_explicit_solver_is_stored_on_the_instance(self, inv):
        inv.write_input_tio(solver=TIOConfig.legacy())
        text = (inv.inversion_dir / "inverse_EW" / "input_tio").read_text()
        assert text == LEGACY_TEXT
        assert inv.solver.weight_mode == "variance"

    def test_none_keeps_the_stored_solver(self, inv):
        inv.solver = TIOConfig(gamma=0.05)
        inv.write_input_tio()
        text = (inv.inversion_dir / "inverse_NS" / "input_tio").read_text()
        assert text.splitlines()[0].startswith("0.0500 ")

    def test_instance_built_without_solver_attribute_still_works(self, inv):
        del inv.solver
        inv.write_input_tio()
        assert inv.solver.weight_mode == "file"

    def test_wrong_type_raises(self, inv):
        with pytest.raises(TypeError, match="TIOConfig"):
            inv.write_input_tio(solver={"gamma": 0.1})


class TestPrepareInversionSolverArgument:
    def _stub_everything(self, monkeypatch, calls):
        for name in ("write_file_info_rsc", "export_pair_to_binary",
                     "_create_symlinks", "write_liste_image", "write_liste_image_inv",
                     "write_input_tio", "write_launch_script", "write_run_parameters"):
            monkeypatch.setattr(TIOInversion, name, lambda *a, **k: None)
        monkeypatch.setattr(TIOInversion, "setup_directories",
                            lambda self: calls.append("setup_directories"))
        monkeypatch.setattr(TIOInversion, "write_liste_couple",
                            lambda *a, **k: {"EW": [1], "NS": [1]})
        monkeypatch.setattr(tio, "print_tio_export_summary", lambda *a, **k: None)

    def test_bad_solver_raises_before_anything_is_written(self, inv, monkeypatch):
        calls: list = []
        self._stub_everything(monkeypatch, calls)
        with pytest.raises(TypeError, match="TIOConfig"):
            inv.prepare_inversion(solver="file", print_summary=False)
        assert calls == []

    def test_solver_is_stored_and_logged(self, inv, monkeypatch, caplog_gmc):
        calls: list = []
        self._stub_everything(monkeypatch, calls)
        inv.prepare_inversion(solver=TIOConfig(gamma=0.02, reweight_iterations=2),
                              print_summary=False)
        assert inv.solver.gamma == 0.02
        assert "iponder=2" in caplog_gmc.text
        assert "reweight_iterations=2" in caplog_gmc.text
        assert calls == ["setup_directories"]

    def test_default_solver_when_none(self, inv, monkeypatch):
        self._stub_everything(monkeypatch, [])
        del inv.solver
        inv.prepare_inversion(print_summary=False)
        assert inv.solver.weight_mode == "file"


class TestIgnoredWeightsWarning:
    def test_legacy_solver_with_a_weight_mode_warns(self, inv, caplog_gmc):
        inv.solver = TIOConfig.legacy()
        inv.write_liste_couple(weight_mode="temporal", sync_geodb=False)
        assert "will NOT read them" in caplog_gmc.text
        assert "iponder=0" in caplog_gmc.text

    def test_legacy_solver_with_explicit_vectors_warns(self, inv, caplog_gmc):
        inv.solver = TIOConfig.legacy()
        inv.write_liste_couple(weights=[0.5, 1.0, 0.2], sync_geodb=False)
        assert "will NOT read them" in caplog_gmc.text

    def test_legacy_solver_with_uniform_weights_is_silent(self, inv, caplog_gmc):
        inv.solver = TIOConfig.legacy()
        inv.write_liste_couple(sync_geodb=False)
        assert "will NOT read them" not in caplog_gmc.text

    def test_file_solver_never_warns(self, inv, caplog_gmc):
        inv.solver = TIOConfig()
        inv.write_liste_couple(weight_mode="temporal", sync_geodb=False)
        assert "will NOT read them" not in caplog_gmc.text
        text = (inv.inversion_dir / "inverse_EW" / "liste_couple").read_text()
        assert len(text.splitlines()) == 3


class TestOptionalBinaries:
    def test_missing_binaries_do_not_raise_at_construction(self, tmp_path, caplog_gmc):
        session = SimpleNamespace(_legacy_layout=False,
                                  pz_dir=lambda pz, kind=None: tmp_path / pz / str(kind))
        pairs = [SimpleNamespace(pa_key="PZ_2020-01-01-s_2020-02-01-s")]
        inv = TIOInversion(session, pairs, "t", tio_binaries_dir=tmp_path / "nowhere")
        assert inv.invers_pixel_omp_bin is None
        assert inv.lect_depl_cumule_lin_bin is None
        assert "launch(mode='python')" in caplog_gmc.text
        assert inv.solver.weight_mode == "file"

    def test_fortran_launch_without_binaries_raises(self, inv):
        inv.invers_pixel_omp_bin = inv.lect_depl_cumule_lin_bin = None
        with pytest.raises(RuntimeError, match="tio_binaries_dir"):
            inv.launch(mode="local")

    def test_cluster_script_without_binaries_raises(self, inv):
        inv.invers_pixel_omp_bin = inv.lect_depl_cumule_lin_bin = None
        with pytest.raises(RuntimeError, match="tio_binaries_dir"):
            inv.write_launch_script(cluster="isterre")

    def test_local_script_without_binaries_is_skipped_not_fatal(self, inv, caplog_gmc):
        inv.invers_pixel_omp_bin = inv.lect_depl_cumule_lin_bin = None
        inv.write_launch_script()
        assert not list(inv.inversion_dir.glob("inverse_*/launch_TIO_inv_*.sh"))
        assert inv.cluster == "local"
        assert inv._last_launch["script"] is None
        assert "launch(mode='python')" in caplog_gmc.text
