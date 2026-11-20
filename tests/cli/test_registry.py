"""YAML step specs ↔ mask/correction pipelines."""

from __future__ import annotations

import pytest

from geomulticorr.cli.registry import (
    CORRECTIONS,
    MASKS,
    build_correction_pipeline,
    build_filter_pipeline,
    describe_steps,
)
from geomulticorr.corrections import (
    CorrectionPipeline,
    FilterPipeline,
    StableAreaMask,
)


def test_tables_do_not_mix_masks_and_corrections():
    assert "CCFilter" in MASKS and "CCFilter" not in CORRECTIONS
    assert "RampCorrection" in CORRECTIONS and "RampCorrection" not in MASKS
    assert not any(
        n.startswith("Base") or n.endswith("Pipeline") for n in {**MASKS, **CORRECTIONS}
    )


def test_round_trip_masks():
    specs = [
        {"CCFilter": {"cc_threshold": 0.4}},
        {"SlopeMask": {"min_slope": 0.0, "max_slope": 35.0}},
    ]
    pipeline = build_filter_pipeline(specs)
    assert isinstance(pipeline, FilterPipeline)
    assert describe_steps(pipeline) == specs
    assert describe_steps(build_filter_pipeline(describe_steps(pipeline))) == specs


def test_round_trip_corrections_with_defaults_filled():
    pipeline = build_correction_pipeline(
        ["MedianCentering", {"RampCorrection": {"lambda_reg": 0.1}}]
    )
    assert isinstance(pipeline, CorrectionPipeline)
    assert describe_steps(pipeline) == [
        {"MedianCentering": {"stat": "median"}},
        {"RampCorrection": {"lambda_reg": 0.1}},
    ]


def test_single_step_is_still_a_pipeline():
    assert isinstance(build_correction_pipeline(["RampCorrection"]), CorrectionPipeline)
    assert isinstance(build_filter_pipeline(["SlopeMask"]), FilterPipeline)


def test_empty_is_none():
    assert build_filter_pipeline([]) is None
    assert build_correction_pipeline([]) is None


def test_positional_argument():
    pipeline = build_filter_pipeline([{"OutlierFilter": [-5, 5]}])
    assert describe_steps(pipeline) == [{"OutlierFilter": {"threshold": [-5, 5]}}]


def test_auto_resolves_through_callback():
    seen = []
    pipeline = build_filter_pipeline(
        [{"StableAreaMask": "auto"}],
        resolve_auto=lambda name: seen.append(name) or "AREAS",
    )
    (mask,) = pipeline.masks
    assert isinstance(mask, StableAreaMask) and mask.stable_mask == "AREAS"
    assert seen == ["StableAreaMask"]


def test_auto_without_session_raises():
    with pytest.raises(ValueError, match="auto"):
        build_filter_pipeline([{"StableAreaMask": "auto"}])


@pytest.mark.parametrize(
    "spec, match",
    [
        ("NoSuchMask", "unknown mask"),
        ({"CCFilter": {"thr": 1}}, "CCFilter"),
        ({"A": 1, "B": 2}, "one-key mapping"),
    ],
)
def test_bad_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        build_filter_pipeline([spec])


def test_mask_is_not_a_correction():
    with pytest.raises(ValueError, match="unknown correction"):
        build_correction_pipeline(["CCFilter"])
