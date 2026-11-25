#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# config.py
# creation date: 2026-10-29.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------- #
"""The YAML configuration of the ``gmc`` command line, as pydantic models.

One file drives every stage: a shared header (``project``, ``pzones``,
``image_type``, ``outputs``) and one optional section per stage. ``gmc <stage>``
reads its own section; ``gmc run`` executes, in :data:`STAGE_ORDER`, every
section that is present and has ``process: true``.

Three rules hold across the models:

- **Unknown keys raise** (``extra="forbid"``). A misspelt key silently replaced
  by its default is exactly the failure the explorers' ``TypeError`` on unknown
  ``**defaults`` exists to prevent.
- **Field names are the target method's parameter names.** Each section lists,
  in ``_targets``, the methods its fields are forwarded to, and
  ``tests/cli/test_config.py`` asserts every forwarded field is a parameter of
  one of them — so the YAML layer cannot drift from the API. Fields that only the
  CLI reads are listed in ``_cli_only``.
- **Defaults are the method defaults.** A section left empty calls the method
  exactly as a notebook would with no arguments.

This module imports nothing from :mod:`geomulticorr.core`, so templates and
validation never open a project.
"""

from __future__ import annotations

import pathlib
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Canonical execution order of ``gmc run``.
STAGE_ORDER: tuple[str, ...] = (
    "prepare",
    "crop",
    "similarity",
    "pairs",
    "correlate",
    "extract",
    "correct",
    "invert",
    "fuse",
)

Duration = int | str | None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class _Section(_Model):
    """A stage section: ``process`` switches it off without deleting it."""

    #: ``Class.method`` names the forwarded fields must be parameters of.
    _targets: ClassVar[tuple[str, ...]] = ()
    #: Fields read by the CLI itself, never forwarded.
    _cli_only: ClassVar[frozenset[str]] = frozenset({"process", "image_type"})

    process: bool = True
    #: Overrides the header ``image_type`` for this stage only.
    image_type: str | None = None


def _one_or_many(cls, value):
    """Accept a single mapping where a list of them is expected."""
    if value is None:
        return []
    return [value] if isinstance(value, dict) else value


# ── header ──────────────────────────────────────────────────────────────── #


class ProjectConfig(_Model):
    path: pathlib.Path
    epsg: int | str | None = None
    empty_geodatabase: pathlib.Path | None = None


class ReportOutputs(_Model):
    html: bool = True
    pdf: bool = True
    title: str | None = None
    #: Points (gpkg/shp) where the report samples the inverted time series.
    points: pathlib.Path | None = None
    points_layer: str | None = None
    buffer_m: float = 0.0
    #: How many worst pairs (by corrected NMAD) the report lists.
    worst_pairs: int = 10


class OutputsConfig(_Model):
    #: Run folder; ``None`` → ``<project>/gmc_runs/<config stem>/``.
    path: pathlib.Path | None = None
    #: 1 = summary tables and a few figures; 2 = per-pair tables, every figure.
    level: Literal[1, 2] = 1
    report: ReportOutputs = Field(default_factory=ReportOutputs)
    profiling: bool = False


# ── prepare ─────────────────────────────────────────────────────────────── #


class PzoneSpec(_Model):
    name: str
    shortname: str
    #: Polygon file (gpkg/shp/geojson) or raster whose footprint is used …
    source: pathlib.Path | None = None
    layer: str | None = None
    index: int = 0
    dissolve: bool = False
    #: … or explicit coordinates, with their CRS.
    coordinates: list[tuple[float, float]] | None = None
    crs: int | str | None = None

    @field_validator("coordinates")
    @classmethod
    def _closed_enough(cls, value):
        if value is not None and len(value) < 3:
            raise ValueError("a pzone polygon needs at least 3 vertices")
        return value


class _PerPzone(_Model):
    #: Restrict to one pzone; ``None`` applies to every selected pzone.
    pz_name: str | None = None
    overwrite: bool = False


class ReferenceGridSpec(_PerPzone):
    resolution: float


class ReferenceDemSpec(_PerPzone):
    source: pathlib.Path
    resampling: str = "bilinear"


class MovingAreasSpec(_PerPzone):
    source: pathlib.Path
    layer: str | None = None
    buffer: float | None = None
    clip: bool = True


class PrepareConfig(_Section):
    _targets = (
        "Session.insert_pzone_from_file",
        "Session.insert_pzone",
        "Session.build_reference_grid",
        "Session.build_reference_dem",
        "Session.load_moving_areas",
    )
    _cli_only = _Section._cli_only | {
        "pzones",
        "reference_grid",
        "reference_dem",
        "moving_areas",
    }

    pzones: list[PzoneSpec] = Field(default_factory=list)
    reference_grid: list[ReferenceGridSpec] = Field(default_factory=list)
    reference_dem: list[ReferenceDemSpec] = Field(default_factory=list)
    moving_areas: list[MovingAreasSpec] = Field(default_factory=list)

    _many = field_validator(
        "pzones", "reference_grid", "reference_dem", "moving_areas", mode="before"
    )(_one_or_many)


# ── crop ────────────────────────────────────────────────────────────────── #


class RegisterSpec(_Model):
    sources: pathlib.Path | list[pathlib.Path]
    pz_name: str
    glob_pattern: str = "*.tif"
    move: bool = False
    overwrite: bool = False


class CropConfig(_Section):
    """Map a raw image bank, sieve it onto every pzone, register the thumbs."""

    _targets = ("Session.map_georasters_bank", "Session.sieve_bulk")
    _cli_only = _Section._cli_only | {"register_thumbs", "backfill_resolution"}

    # map_georasters_bank
    georasters_bank_path: pathlib.Path | None = None
    extensions: str | list[str] | None = None
    exclude_patterns: list[str] | None = None
    max_workers: int | None = None
    # sieve_bulk
    suffix: str = ""
    target_resolution: float = 1.5
    resampling_method: str = "bilinear"
    selected_band: int | list[int] = 1
    out_dtype: str = "uint16"
    skip_existing: bool = True
    n_threads: int = 4
    memory_limit_mb: int = 1024
    use_multiproc_threshold_mb: int = 2048
    terrain_products: list[str] = Field(
        default_factory=lambda: ["mdhillshade", "slope"]
    )
    keep_dem: bool = True
    hillshade_azimuth: float = 315.0
    hillshade_altitude: float = 45.0
    z_factor: float = 1.0
    resolution_in_name: bool = True
    mask_vector: pathlib.Path | None = None
    mask_buffer_value: float = 0.0
    mask_layer: str | None = None
    # already-cropped thumbs, registered as they are
    register_thumbs: list[RegisterSpec] = Field(default_factory=list)
    backfill_resolution: bool = False

    _many = field_validator("register_thumbs", mode="before")(_one_or_many)

    MAP_KEYS: ClassVar[tuple[str, ...]] = (
        "extensions",
        "exclude_patterns",
        "max_workers",
    )


# ── similarity ──────────────────────────────────────────────────────────── #


class SimilarityConfig(_Section):
    _targets = ("Session.compute_thumbs_similarity",)
    _cli_only = _Section._cli_only | {"report_kwargs"}

    couples: str = "all"
    max_step: int | None = None
    sensor_filter: str | None = None
    cross_sensor_only: bool = False
    moving_areas: Literal["auto"] | None = "auto"
    moving_buffer: float = 0.0
    grid: Literal["coarser", "reference"] = "coarser"
    overwrite: bool = False
    plots: bool = True
    formats: list[str] = Field(default_factory=lambda: ["png"])
    #: Forwarded as ``**report_kwargs`` (``kernel``, ``downsample``…).
    report_kwargs: dict[str, Any] = Field(default_factory=dict)


# ── pairs ───────────────────────────────────────────────────────────────── #


class PairGroup(_Model):
    """One ``update_pairs`` call. Field names are its parameters."""

    strategy: str = "consecutive"
    max_step: int | None = None
    max_dt_days: Duration = None
    min_dt_days: Duration = None
    sensor_filter: str | None = None
    append: bool = False
    min_similarity: float | None = None
    similarity_metric: str = "score"
    per_image: int | None = None
    cross_sensor_only: bool = False
    both_directions: bool = True
    couples: list[tuple[str, str]] | None = None


class PairsConfig(_Section):
    _targets = ("Session.update_pairs",)
    _cli_only = _Section._cli_only | {"groups", "figures", "formats"}

    #: ``[]`` commits nothing: the stage only reports the pairs already there.
    groups: list[PairGroup] = Field(default_factory=lambda: [PairGroup()])
    #: Views written with ``save_pairs_figure`` after the pairs are committed.
    figures: list[str] = Field(default_factory=lambda: ["chord", "baseline"])
    formats: list[str] = Field(default_factory=lambda: ["png"])

    _many = field_validator("groups", mode="before")(_one_or_many)


# ── correlate ───────────────────────────────────────────────────────────── #


class CorrelateConfig(_Section):
    """Derive (or read) a correlation plan, write the ASP scripts, launch them."""

    _targets = (
        "Session.prepare_pairs_correlation",
        "Session.launch_pairs_correlation",
    )
    _cli_only = _Section._cli_only | {
        "explore",
        "launch",
        "sync",
        "options",
        "plan_name",
    }

    criterias: str | list[str] = ""
    #: An existing ``correlation_plan_<name>.json``; wins over ``explore``.
    plan: pathlib.Path | None = None
    #: Derive the plan with ``explore_correlation_params(interactive=False)``;
    #: keys are its ``**defaults`` (``velocity_m_yr``, ``footprint_m``…).
    explore: dict[str, Any] | None = None
    plan_name: str = "default"
    cluster: str | None = None
    asp_bin_dir: pathlib.Path | None = None
    overwrite: bool = False
    overwrite_scripts: bool = False
    nodes: int = 1
    cores: int = 8
    walltime: str = "12:00:00"
    correval: bool = True
    launch: bool = True
    dry_run: bool = False
    #: Instead of preparing/launching, collect finished cluster jobs.
    sync: bool = False
    #: Any other ``prepare_pairs_correlation`` keyword (``corr_kernel``…),
    #: checked against its signature when the stage runs.
    options: dict[str, Any] = Field(default_factory=dict)


# ── extract / correct ───────────────────────────────────────────────────── #


class ExtractConfig(_Section):
    _targets = ("Session.extract_pairs_raw_displacements",)

    criterias: str | list[str] = ""
    save_plot: bool = True
    overwrite: bool = False
    target_resolution: float | None = None
    resampling: str = "bilinear"
    verbose: bool = False


#: ``"MedianCentering"`` or ``{"CCFilter": {"cc_threshold": 0.5}}``.
StepSpec = str | dict[str, Any]


class CorrectConfig(_Section):
    _targets = ("Session.apply_pairs_corrections",)
    _cli_only = _Section._cli_only | {"masks", "corrections", "options", "derived"}

    criterias: str | list[str] = ""
    masks: list[StepSpec] = Field(default_factory=list)
    corrections: list[StepSpec] = Field(
        default_factory=lambda: ["MedianCentering", "RampCorrection"]
    )
    overwrite: bool = False
    save_plot: bool = True
    plot_level: Literal["light", "full"] = "light"
    verbose: bool = False
    #: ``"auto"``: fit outside the pzone's stored moving areas when ``masks`` holds
    #: no ``StableAreaMask``; a path to a vector file; ``null`` fits on all ground.
    moving_areas: str | None = "auto"
    #: Forwarded as ``**kwargs`` to the corrections (e.g. ``dem: auto``).
    options: dict[str, Any] = Field(default_factory=dict)
    #: Rasters written after the corrections (``Session.write_derived_rasters``):
    #: any of ``magnitude``, ``velocity``, ``ew_velocity``, ``ns_velocity``. Empty = none.
    derived: list[Literal["magnitude", "velocity", "ew_velocity", "ns_velocity"]] = (
        Field(default_factory=list)
    )


# ── invert ──────────────────────────────────────────────────────────────── #


class InversionRun(_Model):
    name: str
    #: Pair selection, AND-ed with the pzone (``get_pairs(criterias=)``).
    criterias: str | list[str] = ""
    pz_name: str | None = None
    masks: list[StepSpec] = Field(default_factory=list)
    tio_binaries_dir: pathlib.Path | None = None
    #: ``filter_pairs_by_nmad(threshold=)``; ``None`` keeps every pair.
    nmad_threshold: float | None = None
    #: ``prepare_inversion`` keywords (``weight_mode``, ``sensor_weights``…).
    weights: dict[str, Any] = Field(default_factory=dict)
    #: ``TIOConfig`` fields.
    solver: dict[str, Any] = Field(default_factory=dict)
    mode: Literal["python", "local", "isterre"] = "python"
    launch: bool = True
    #: Fortran runs only; ``None`` → ``True`` unless ``mode == "python"``.
    post_process: bool | None = None
    write_residuals: bool = False
    weights_figure: bool = True


class InvertConfig(_Section):
    _cli_only = _Section._cli_only | {"runs"}

    runs: list[InversionRun] = Field(default_factory=list)

    _many = field_validator("runs", mode="before")(_one_or_many)


# ── fuse ────────────────────────────────────────────────────────────────── #


class FusionRun(_Model):
    name: str
    reference: str
    secondary: str
    pz_name: str | None = None
    model: Literal["offset+rate", "offset"] = "offset+rate"
    overlap: tuple[str, str] | None = None
    min_epochs: int = 2
    reference_label: str = "reference"
    secondary_label: str = "secondary"
    overwrite: bool = True
    #: Pair criterias for the closure check of each side against the other's
    #: series (``pair_closure_against_series``); ``None`` skips it.
    closure_reference_criterias: str | list[str] | None = None
    closure_secondary_criterias: str | list[str] | None = None


class FuseConfig(_Section):
    _cli_only = _Section._cli_only | {"runs"}

    runs: list[FusionRun] = Field(default_factory=list)

    _many = field_validator("runs", mode="before")(_one_or_many)


# ── the whole file ──────────────────────────────────────────────────────── #


SECTION_MODELS: dict[str, type[_Section]] = {
    "prepare": PrepareConfig,
    "crop": CropConfig,
    "similarity": SimilarityConfig,
    "pairs": PairsConfig,
    "correlate": CorrelateConfig,
    "extract": ExtractConfig,
    "correct": CorrectConfig,
    "invert": InvertConfig,
    "fuse": FuseConfig,
}


class RunConfig(_Model):
    project: ProjectConfig
    #: Pzones the stages loop over; ``None`` → every pzone of the project.
    pzones: list[str] | None = None
    image_type: str = "opt"
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)

    prepare: PrepareConfig | None = None
    crop: CropConfig | None = None
    similarity: SimilarityConfig | None = None
    pairs: PairsConfig | None = None
    correlate: CorrelateConfig | None = None
    extract: ExtractConfig | None = None
    correct: CorrectConfig | None = None
    invert: InvertConfig | None = None
    fuse: FuseConfig | None = None

    @field_validator("pzones", mode="before")
    @classmethod
    def _str_to_list(cls, value):
        return [value] if isinstance(value, str) else value

    def stages(self) -> list[str]:
        """Sections present and switched on, in execution order."""
        return [
            s
            for s in STAGE_ORDER
            if getattr(self, s) is not None and getattr(self, s).process
        ]

    def image_type_for(self, stage: str) -> str:
        section = getattr(self, stage, None)
        return (section.image_type if section else None) or self.image_type


def forwarded(section: _Section, *, exclude: set[str] = frozenset()) -> dict:
    """The section's fields that are forwarded to its target methods."""
    skip = set(section._cli_only) | set(exclude)
    return {k: v for k, v in section.model_dump().items() if k not in skip}


def _none_strings(value):
    """``"None"`` / ``"null"`` typed as strings become ``None`` (as in xDEM)."""
    if isinstance(value, dict):
        return {k: _none_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_none_strings(v) for v in value]
    if isinstance(value, str) and value in ("None", "none"):
        return None
    return value


def load_config(path: str | pathlib.Path) -> RunConfig:
    """Read and validate a YAML configuration file.

    Every relative path in the file is resolved against the YAML file's folder
    (not the working directory), so a config can travel with its data.

    :raises FileNotFoundError: The file does not exist.
    :raises ValueError: Wrong suffix, not a mapping.
    :raises pydantic.ValidationError: Invalid content.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"expected a .yaml/.yml file, got {path.name}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: the top level must be a mapping")
    cfg = RunConfig.model_validate(_none_strings(raw))
    _resolve_paths(cfg, path.parent.resolve())
    return cfg


def _resolve_paths(model: BaseModel, base: pathlib.Path) -> None:
    """Make every relative ``Path`` field absolute against *base*, in place."""

    def fix(value):
        if isinstance(value, pathlib.Path):
            return value if value.is_absolute() else (base / value).resolve()
        if isinstance(value, list):
            return [fix(v) for v in value]
        if isinstance(value, BaseModel):
            _resolve_paths(value, base)
        return value

    for name in type(model).model_fields:
        value = getattr(model, name)
        fixed = fix(value)
        if fixed is not value:
            object.__setattr__(model, name, fixed)


def dump_config(cfg: RunConfig) -> str:
    """The resolved configuration as YAML (``used_config.yaml``)."""
    data = cfg.model_dump(mode="json", exclude_none=False)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def template_config(stage: str = "run") -> str:
    """A commented YAML template with every default filled in.

    :param stage: ``"run"`` for the whole file, or one stage name for the
        header plus that section only.
    """
    sections = STAGE_ORDER if stage == "run" else (stage,)
    unknown = [s for s in sections if s not in SECTION_MODELS]
    if unknown:
        raise ValueError(f"unknown stage {unknown[0]!r}; choose from {STAGE_ORDER}")
    header = {
        "project": {"path": "path/to/project", "epsg": 2154},
        "pzones": None,
        "image_type": "opt",
        "outputs": OutputsConfig().model_dump(mode="json"),
    }
    lines = [
        "# GeoMultiCorr configuration — every value below is the default.",
        "# Only `project.path` is required; delete what you do not change.",
        yaml.safe_dump(header, sort_keys=False),
    ]
    examples = _SECTION_EXAMPLES
    for s in sections:
        body = SECTION_MODELS[s]().model_dump(mode="json")
        body.update(examples.get(s, {}))
        doc = (SECTION_MODELS[s].__doc__ or "").strip().splitlines()
        if doc:
            lines.append(f"# {s}: {doc[0]}")
        lines.append(yaml.safe_dump({s: body}, sort_keys=False))
    return "\n".join(lines)


#: Non-default illustrations for list sections whose default is empty.
_SECTION_EXAMPLES: dict[str, dict] = {
    "prepare": {
        "pzones": [{"name": "MyZone", "shortname": "MZ", "source": "aoi.gpkg"}],
        "reference_grid": [{"resolution": 3.0}],
    },
    "correct": {
        "masks": [{"CCFilter": {"cc_threshold": 0.5}}, {"StableAreaMask": "auto"}],
    },
    "invert": {
        "runs": [
            {
                "name": "my_inversion",
                "weights": {"weight_mode": "quality_spatial"},
                "solver": {"reweight_iterations": 2},
                "mode": "python",
            }
        ]
    },
    "fuse": {
        "process": False,
        "runs": [{"name": "fused", "reference": "inv_a", "secondary": "inv_b"}],
    },
}
