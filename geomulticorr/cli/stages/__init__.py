"""One module per stage, each ``run(ctx, section, record) -> None``.

Stages are thin: they turn a config section into the existing ``Session`` /
``TIOInversion`` calls and fill the stage record. No processing logic lives
here — anything a stage computes beyond bookkeeping belongs in the library.
"""

from __future__ import annotations

import importlib
from typing import Callable

from geomulticorr.cli.config import STAGE_ORDER


def stage_runner(name: str) -> Callable:
    if name not in STAGE_ORDER:
        raise ValueError(f"unknown stage {name!r}; choose from {STAGE_ORDER}")
    return importlib.import_module(f"geomulticorr.cli.stages.{name}").run
