#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# main.py
# creation date: 2026-10-24.
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
"""The ``gmc`` command.

::

    gmc <stage> CONFIG.yaml [--output DIR] [--dry-run] [--profile] [--no-report]
    gmc run     CONFIG.yaml [--only pairs,correct] [...]
    gmc report  RUN_DIR | CONFIG.yaml [--no-pdf]
    gmc <stage|run> --template-config [PATH]

Stages: prepare, crop, similarity, pairs, correlate, extract, correct, invert,
fuse. Every command writes into one *run folder* (``outputs.path``, default
``<project>/gmc_runs/<config stem>/``): ``used_config.yaml``, ``gmc.log``,
``records/<stage>.json``, ``profiling/`` and ``report.{html,pdf}``. Because the
folder is keyed on the config file, running the stages one by one accumulates
their records in the same place and the report covers them all.

Exit codes: 0 success, 1 a stage failed, 2 invalid configuration.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import traceback
from typing import Sequence

from geomulticorr._logging import logger
from geomulticorr.cli.config import (
    STAGE_ORDER,
    RunConfig,
    dump_config,
    load_config,
    template_config,
)

RUNS_DIRNAME = "gmc_runs"
LOG_FILENAME = "gmc.log"
USED_CONFIG_FILENAME = "used_config.yaml"


def default_run_dir(cfg: RunConfig, config_path: pathlib.Path | None) -> pathlib.Path:
    if cfg.outputs.path is not None:
        path = pathlib.Path(cfg.outputs.path)
        if not path.is_absolute() and config_path is not None:
            path = (config_path.parent / path).resolve()
        return path
    stem = config_path.stem if config_path is not None else "run"
    return pathlib.Path(cfg.project.path) / RUNS_DIRNAME / stem


class _FileLog:
    """Mirror GMC log records into ``<run_dir>/gmc.log`` while active."""

    def __init__(self, path: pathlib.Path) -> None:
        self.handler = logging.FileHandler(path, encoding="utf-8")
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-10s %(message)s")
        )

    def __enter__(self):
        logging.getLogger("GMC").addHandler(self.handler)
        return self

    def __exit__(self, *exc) -> None:
        logging.getLogger("GMC").removeHandler(self.handler)
        self.handler.close()


def run_stages(
    cfg: RunConfig,
    stages: Sequence[str],
    run_dir: pathlib.Path,
    *,
    session=None,
    report: bool = True,
) -> int:
    """Execute *stages* in order, recording each; render the report at the end.

    Stops at the first failing stage (later stages depend on it) but still
    writes the profiling summary and the report, so a failed run is documented
    as well as a successful one.

    :param session: Inject an already-open session (tests, notebooks).
    :returns: Exit code — 0, or 1 if a stage failed.
    """
    from geomulticorr.cli import profiling
    from geomulticorr.cli.record import record_stage
    from geomulticorr.cli.stages import stage_runner
    from geomulticorr.cli.stages._common import StageContext

    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / USED_CONFIG_FILENAME).write_text(dump_config(cfg), encoding="utf-8")

    ctx = StageContext(cfg=cfg, run_dir=run_dir, session=session)
    code = 0
    with _FileLog(run_dir / LOG_FILENAME):
        if cfg.outputs.profiling:
            profiling.enable()
        for stage in stages:
            section = getattr(cfg, stage)
            runner = stage_runner(stage)
            if profiling.is_enabled():
                runner = profiling.stage_wrapper(stage)(runner)
            try:
                with record_stage(
                    stage,
                    run_dir,
                    config=section.model_dump(mode="json"),
                    pzones=cfg.pzones,
                ) as rec:
                    runner(ctx, section, rec)
                logger.success(f"gmc ── {stage}: {rec.status} in {rec.elapsed_s:.1f} s")
            except Exception as exc:
                logger.error(f"gmc ── {stage} failed: {type(exc).__name__}: {exc}")
                # the GMC console formatter drops exc_info: log the text itself
                logger.debug(traceback.format_exc())
                profiling.recover_after_failure()
                code = 1
                break
        if cfg.outputs.profiling:
            written = profiling.summarise(run_dir)
            if written:
                _write_profiling_table(run_dir)
        if report:
            render(run_dir, cfg)
    return code


def _write_profiling_table(run_dir: pathlib.Path) -> None:
    import json

    from geomulticorr.cli import profiling

    try:
        path = run_dir / profiling.PROFILING_DIRNAME / "summary.json"
        path.write_text(
            json.dumps(profiling.summary_table(), indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning(f"Could not write the profiling table: {exc}")


def render(
    run_dir: pathlib.Path, cfg: RunConfig | None = None, pdf: bool | None = None
):
    """Render ``report.html`` (and ``report.pdf``); never raises."""
    try:
        from geomulticorr.report import render_report

        outputs = cfg.outputs if cfg is not None else None
        return render_report(run_dir, outputs=outputs, pdf=pdf)
    except Exception as exc:
        logger.warning(f"Report not rendered: {type(exc).__name__}: {exc}")
        return {}


def _dry_run(cfg: RunConfig, stages: Sequence[str], run_dir: pathlib.Path) -> None:
    """Print what would run; open the project read-only to count the selection."""
    import yaml

    print(f"project : {cfg.project.path}")
    print(f"run dir : {run_dir}")
    print(f"stages  : {', '.join(stages) or '(none)'}")
    session = None
    if pathlib.Path(cfg.project.path).exists():
        try:
            from geomulticorr.cli.stages._common import StageContext

            ctx = StageContext(cfg=cfg, run_dir=run_dir)
            session = ctx.open_session()
            print(f"pzones  : {', '.join(ctx.pz_names())}")
        except Exception as exc:
            print(f"pzones  : (project not readable: {exc})")
    for stage in stages:
        body = getattr(cfg, stage).model_dump(mode="json")
        print(f"\n── {stage}")
        print(yaml.safe_dump(body, sort_keys=False).rstrip())
        if session is not None and stage in ("correlate", "extract", "correct"):
            _print_selection(cfg, session, stage)


def _print_selection(cfg, session, stage) -> None:
    from geomulticorr.cli.stages._common import StageContext, pz_criterias

    ctx = StageContext(cfg=cfg, run_dir=pathlib.Path("."), session=session)
    crit = getattr(cfg, stage).criterias
    for pz in ctx.pz_names():
        overview = session.get_pairs_overview(
            pz_criterias(pz, crit), image_type=ctx.image_type(stage)
        )
        print(f"   {pz}: {0 if overview is None else len(overview)} pair(s) selected")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmc",
        description="GeoMultiCorr — YAML-driven processing stages and reports.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument(
            "config", nargs="?", type=pathlib.Path, help="YAML configuration"
        )
        p.add_argument(
            "--output", type=pathlib.Path, help="run folder (overrides outputs.path)"
        )
        p.add_argument(
            "--template-config",
            nargs="?",
            const="-",
            metavar="PATH",
            help="write a YAML template with every default to PATH "
            "(stdout without PATH), then exit",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="validate and print the plan; write nothing",
        )
        p.add_argument(
            "--profile",
            action="store_true",
            help="profile time and memory (GeoUtils profiler)",
        )
        p.add_argument(
            "--no-report", action="store_true", help="do not render the report"
        )
        p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    for stage in STAGE_ORDER:
        common(sub.add_parser(stage, help=f"run the '{stage}' stage"))
    p_run = sub.add_parser("run", help="run every section present, in order")
    common(p_run)
    p_run.add_argument("--only", help="comma-separated subset of stages")

    p_rep = sub.add_parser("report", help="(re-)render the report of a run folder")
    p_rep.add_argument("target", type=pathlib.Path, help="run folder or YAML config")
    p_rep.add_argument("--no-pdf", action="store_true", help="HTML only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from pydantic import ValidationError

    args = _parser().parse_args(argv)

    if args.command == "report":
        target = args.target
        cfg = None
        if target.suffix.lower() in (".yaml", ".yml"):
            cfg = load_config(target)
            target = default_run_dir(cfg, target)
        written = render(target, cfg, pdf=False if args.no_pdf else None)
        for path in (written or {}).values():
            print(path)
        return 0 if written else 1

    if args.verbose:
        logging.getLogger("GMC").setLevel(logging.DEBUG)
    if args.template_config:
        text = template_config(args.command)
        if args.template_config == "-":
            print(text)
        else:
            pathlib.Path(args.template_config).write_text(text, encoding="utf-8")
            print(f"template written to {args.template_config}", file=sys.stderr)
        return 0
    if args.config is None:
        print("gmc: a CONFIG file is required", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        print(f"gmc: invalid configuration {args.config}:\n{exc}", file=sys.stderr)
        return 2
    if args.output is not None:
        cfg.outputs.path = args.output.resolve()
    if args.profile:
        cfg.outputs.profiling = True

    if args.command == "run":
        stages = cfg.stages()
        if args.only:
            wanted = [s.strip() for s in args.only.split(",") if s.strip()]
            bad = [s for s in wanted if s not in STAGE_ORDER]
            if bad:
                print(f"gmc: unknown stage(s) {bad}", file=sys.stderr)
                return 2
            stages = [s for s in stages if s in wanted]
    else:
        if getattr(cfg, args.command) is None:
            print(
                f"gmc: {args.config.name} has no '{args.command}:' section "
                f"(see `gmc {args.command} --template-config`)",
                file=sys.stderr,
            )
            return 2
        stages = [args.command]

    run_dir = default_run_dir(cfg, args.config)
    if args.dry_run:
        _dry_run(cfg, stages, run_dir)
        return 0
    return run_stages(cfg, stages, run_dir, report=not args.no_report)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
