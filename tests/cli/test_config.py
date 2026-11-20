"""The YAML layer: validation, templates, path resolution, and API agreement."""

from __future__ import annotations

import inspect

import pytest
import yaml
from pydantic import ValidationError

from geomulticorr.cli import config as cfgmod
from geomulticorr.cli.config import (
    SECTION_MODELS,
    STAGE_ORDER,
    RunConfig,
    forwarded,
    load_config,
    template_config,
)


def _method(target: str):
    """``"Session.update_pairs"`` → the function object."""
    from geomulticorr.core.session import Session

    owner, name = target.split(".")
    assert owner == "Session"
    return getattr(Session, name)


def _params(target: str) -> set[str]:
    return set(inspect.signature(_method(target)).parameters) - {"self"}


class TestSignatureGuard:
    """Every forwarded field must be a parameter of one of the section's targets."""

    @pytest.mark.parametrize(
        "stage", [s for s in STAGE_ORDER if SECTION_MODELS[s]._targets]
    )
    def test_forwarded_fields_are_parameters(self, stage):
        model = SECTION_MODELS[stage]
        accepted = set().union(*(_params(t) for t in model._targets))
        fields = set(model.model_fields) - set(model._cli_only)
        assert fields <= accepted, sorted(fields - accepted)

    def test_pair_group_matches_update_pairs(self):
        fields = set(cfgmod.PairGroup.model_fields)
        assert fields <= _params("Session.update_pairs")

    def test_inversion_run_weight_keys_go_through_prepare_inversion(self):
        from geomulticorr.inversion.tio_inversion import TIOInversion

        params = set(inspect.signature(TIOInversion.prepare_inversion).parameters)
        assert {"weight_mode", "sensor_weights", "solver"} <= params

    def test_fusion_run_matches_fuse_inversions(self):
        from geomulticorr.inversion.fusion import fuse_inversions

        params = set(inspect.signature(fuse_inversions).parameters)
        own = {
            "name",
            "reference",
            "secondary",
            "pz_name",
            "closure_reference_criterias",
            "closure_secondary_criterias",
        }
        assert set(cfgmod.FusionRun.model_fields) - own <= params

    def test_defaults_match_method_defaults(self):
        """A field's default is the method's default, so an empty section is a no-arg call."""
        for stage in ("extract", "similarity"):
            model = SECTION_MODELS[stage]
            (target,) = model._targets
            sig = inspect.signature(_method(target)).parameters
            for name, field in model.model_fields.items():
                if name in model._cli_only or name not in sig:
                    continue
                default = sig[name].default
                ours = field.get_default(call_default_factory=True)
                if isinstance(default, tuple):
                    default = list(default)
                assert ours == default, (stage, name)


class TestValidation:
    def test_unknown_key_raises(self):
        with pytest.raises(ValidationError, match="grops"):
            RunConfig.model_validate(
                {"project": {"path": "/x"}, "pairs": {"grops": []}}
            )

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ValidationError):
            RunConfig.model_validate({"project": {"path": "/x"}, "pair": {}})

    def test_single_mapping_accepted_for_list(self):
        cfg = RunConfig.model_validate(
            {
                "project": {"path": "/x"},
                "pairs": {"groups": {"strategy": "step", "max_step": 2}},
            }
        )
        assert cfg.pairs.groups[0].strategy == "step"

    def test_duration_strings_kept_for_update_pairs(self):
        cfg = RunConfig.model_validate(
            {"project": {"path": "/x"}, "pairs": {"groups": [{"max_dt_days": "2Y"}]}}
        )
        assert cfg.pairs.groups[0].max_dt_days == "2Y"

    def test_stages_in_canonical_order_and_process_flag(self):
        cfg = RunConfig.model_validate(
            {
                "project": {"path": "/x"},
                "correct": {},
                "pairs": {},
                "extract": {"process": False},
            }
        )
        assert cfg.stages() == ["pairs", "correct"]

    def test_image_type_override(self):
        cfg = RunConfig.model_validate(
            {
                "project": {"path": "/x"},
                "image_type": "slope",
                "pairs": {"image_type": "opt"},
            }
        )
        assert cfg.image_type_for("pairs") == "opt"
        assert cfg.image_type_for("correct") == "slope"

    def test_forwarded_drops_cli_only(self):
        section = cfgmod.ExtractConfig(target_resolution=3.0)
        out = forwarded(section)
        assert out["target_resolution"] == 3.0
        assert "process" not in out and "image_type" not in out


class TestLoad:
    def test_relative_paths_resolve_against_yaml(self, tmp_path):
        (tmp_path / "sub").mkdir()
        path = tmp_path / "sub" / "c.yaml"
        path.write_text(
            "project: {path: ../proj}\n"
            "prepare: {reference_dem: {source: dem.tif}}\n"
            "correlate: {plan: plans/p.json}\n"
        )
        cfg = load_config(path)
        assert cfg.project.path == (tmp_path / "proj").resolve()
        assert (
            cfg.prepare.reference_dem[0].source
            == (tmp_path / "sub" / "dem.tif").resolve()
        )
        assert cfg.correlate.plan == (tmp_path / "sub" / "plans" / "p.json").resolve()

    def test_none_string(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("project: {path: /x}\nextract: {target_resolution: None}\n")
        assert load_config(path).extract.target_resolution is None

    def test_wrong_suffix(self, tmp_path):
        path = tmp_path / "c.txt"
        path.write_text("project: {path: /x}")
        with pytest.raises(ValueError, match="yaml"):
            load_config(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


class TestTemplate:
    @pytest.mark.parametrize("stage", ["run", *STAGE_ORDER])
    def test_template_validates(self, stage):
        data = yaml.safe_load(template_config(stage))
        cfg = RunConfig.model_validate(data)
        assert cfg.project.path

    def test_template_holds_defaults(self):
        data = yaml.safe_load(template_config("extract"))
        cfg = RunConfig.model_validate(data)
        assert cfg.extract == cfgmod.ExtractConfig()

    def test_unknown_stage(self):
        with pytest.raises(ValueError, match="unknown stage"):
            template_config("nope")
