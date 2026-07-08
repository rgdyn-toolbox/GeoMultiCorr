#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# tio_inversion.py
# creation date: 2026-05-12.
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
"""TIO inversion preparation module.

Provides :class:`TIOInversion` — an OOP driver for the full preparation pipeline
of the *invers_pixel* TIO Fortran code: correction of displacement fields,
ENVI binary export, symlink creation, and all required text files.

Typical usage
-------------
>>> filter_pipeline = CCFilter(cc_threshold=0.5) + OutlierFilter(threshold=(-15, 15))
>>> inv = TIOInversion(session, pairs, inversion_name="rock_glacier_2021_2024",
...                    filter_pipeline=filter_pipeline)
>>> inv.prepare()
>>> inv.launch(mode="local")
"""
from __future__ import annotations

import math
import shutil
import stat
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
import geoutils as gu
from rich.console import Console as _RichConsole, Group as _RichGroup
from rich.live import Live as _RichLive
from rich.progress import (
    Progress, TextColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text as _RichText

from geomulticorr.stats import load_pair_stats
from geomulticorr._logging import logger
from geomulticorr.utils.hpc_tools import (
    oar_header,
    cluster_base_env,
    oarsub_submit,
    run_local,
    validate_cluster,
)

_rich_console = _RichConsole(highlight=False)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _year_dec(d: datetime) -> float:
    """Convert a :class:`~datetime.datetime` to a day-precision decimal year.

    Computes the fractional year by dividing elapsed days since the start of
    the year by the total number of days in that year (365 or 366), giving
    day-level precision (e.g. 1 January 2021 → 2021.000; 1 Sept 2021 → 2021.671).

    :param d: The date to convert.
    :type d: datetime.datetime
    :returns: Decimal year with day precision (e.g. ``2021.671``).
    :rtype: float

    Example
    -------
    .. code-block:: python

        from datetime import datetime
        from geomulticorr.inversion.tio_inversion import _year_dec

        _year_dec(datetime(2021, 9, 1))   # → 2021.671...
        _year_dec(datetime(2021, 1, 1))   # → 2021.0
    """
    year_start = datetime(d.year, 1, 1)
    year_end   = datetime(d.year + 1, 1, 1)
    return d.year + (d - year_start).days / (year_end - year_start).days


# ─────────────────────────────────────────────────────────────────────────────
# Quality-weight math (pure helpers, no I/O — used by compute_pair_weights)
# ─────────────────────────────────────────────────────────────────────────────


def normalise_invert(values: list[float]) -> list[float]:
    """Min-max normalise then invert, so the SMALLEST value maps to 1.0.

    Used to turn an "error" metric (e.g. NMAD, where lower is better) into a
    sub-weight in ``[0, 1]`` where the best (lowest) pair gets ``1.0`` and the
    worst (highest) gets ``0.0``.

    Non-finite entries (``nan``/``inf``) are treated as **neutral** and map to
    ``1.0`` (they must not drag a pair's weight down on missing data — callers
    log a warning separately). A degenerate span (all finite values equal) maps
    every finite entry to ``1.0``.

    :param values: Sequence of metric values (any may be non-finite).
    :type values: list[float]
    :returns: Sub-weights in ``[0, 1]``, aligned to *values*.
    :rtype: list[float]
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [1.0] * len(arr)
    vmin = float(finite.min())
    vmax = float(finite.max())
    span = vmax - vmin
    out: list[float] = []
    for v in arr:
        if not np.isfinite(v):
            out.append(1.0)
        elif span <= 0.0:
            out.append(1.0)
        else:
            out.append(1.0 - (float(v) - vmin) / span)
    return out


def normalise_dt(dt_values: list[float], invert: bool = False) -> list[float]:
    """Min-max normalise Δt to a sub-weight in ``[0, 1]``.

    By default the **shortest** baseline maps to ``1.0`` and the longest to
    ``0.0`` (short pairs decorrelate less → higher weight). With *invert* the
    direction is reversed (longer baselines favoured — useful for slow
    phenomena). Non-finite entries map to ``1.0`` (neutral); a degenerate span
    maps every entry to ``1.0``.

    :param dt_values: Temporal baselines (e.g. days), aligned to the pairs.
    :type dt_values: list[float]
    :param invert: Favour longer baselines when ``True``.
    :type invert: bool
    :returns: Sub-weights in ``[0, 1]``.
    :rtype: list[float]
    """
    arr = np.asarray(dt_values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [1.0] * len(arr)
    vmin = float(finite.min())
    vmax = float(finite.max())
    span = vmax - vmin
    out: list[float] = []
    for v in arr:
        if not np.isfinite(v) or span <= 0.0:
            out.append(1.0)
            continue
        t_norm = (float(v) - vmin) / span      # 0 = shortest, 1 = longest
        out.append(t_norm if invert else 1.0 - t_norm)
    return out


def combine_weights(
    w_nmad: float,
    w_cc: float,
    w_dt: float,
    method: str = "geomean",
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
    gamma: float = 1.0 / 3.0,
) -> float:
    """Combine three ``[0, 1]`` sub-weights into a single weight in ``[0, 1]``.

    A non-finite sub-weight is treated as **neutral** (``1.0``) so a pair with a
    missing component is not silently zeroed.

    :param w_nmad: NMAD sub-weight (higher = lower NMAD = better).
    :param w_cc: CC-quality sub-weight.
    :param w_dt: Temporal sub-weight.
    :param method: ``"geomean"`` (cube root of the product), ``"wmean"``
        (weighted arithmetic mean with *alpha*/*beta*/*gamma*, renormalised to
        sum 1), or ``"product"`` (plain product).
    :type method: str
    :param alpha: Weight of *w_nmad* for ``"wmean"``.
    :param beta: Weight of *w_cc* for ``"wmean"``.
    :param gamma: Weight of *w_dt* for ``"wmean"``.
    :returns: Combined weight clamped to ``[0, 1]``.
    :rtype: float
    :raises ValueError: If *method* is not recognised.
    """
    n = 1.0 if not np.isfinite(w_nmad) else float(w_nmad)
    c = 1.0 if not np.isfinite(w_cc) else float(w_cc)
    d = 1.0 if not np.isfinite(w_dt) else float(w_dt)

    if method == "geomean":
        w = (max(n, 0.0) * max(c, 0.0) * max(d, 0.0)) ** (1.0 / 3.0)
    elif method == "product":
        w = n * c * d
    elif method == "wmean":
        total = alpha + beta + gamma
        if total <= 0.0:
            a = b = g = 1.0 / 3.0
        else:
            a, b, g = alpha / total, beta / total, gamma / total
        w = a * n + b * c + g * d
    else:
        raise ValueError(
            f"Unknown combine method '{method}'. "
            "Valid options: 'geomean', 'wmean', 'product'."
        )
    return max(0.0, min(1.0, w))


def _find_tio_executable(bin_name: str, tio_bin_dir: Path | None = None) -> str:
    """Return the full path to a TIO executable.

    Searches *tio_bin_dir* first (if set), then the system PATH.

    :param bin_name: Name of the TIO binary to locate (e.g. ``'invers_pixel_omp'``).
    :type bin_name: str
    :param tio_bin_dir: Directory containing the compiled TIO binaries.
        When ``None``, the system ``PATH`` is searched instead.
    :type tio_bin_dir: pathlib.Path or None
    :returns: Absolute path to the resolved executable.
    :rtype: str
    :raises FileNotFoundError: If the binary cannot be found in *tio_bin_dir*
        or on the system ``PATH``.
    """
    search_path = str(tio_bin_dir) if tio_bin_dir else None
    exe = shutil.which(bin_name, path=search_path)
    if exe is None:
        raise FileNotFoundError(
            f"Unable to find TIO executable '{bin_name}'. "
            f"Ensure the TIO binaries are compiled and available at {tio_bin_dir or 'system PATH'}. "
            f"Install at /home/cusicand/TIO_install or pass tio_binaries_dir= to TIOInversion()."
        )
    return exe


def _build_input_tio_text(
        liste_image_inv_fn: str = "liste_image_inv",
        liste_couple_fn: str = "liste_couple"
        ) -> str:
    """Build the standard ``invers_pixel`` parameter block written to ``input_tio``.

    Produces the multi-line text expected by ``invers_pixel_omp`` as its
    stdin input file. All smoothing, referencing, and weighting settings are
    hard-coded to the GMC defaults; only the filenames are customisable.

    :param liste_image_inv_fn: Filename of the image-date list, relative to the
        inversion directory (e.g. ``'liste_image_inv'``).
    :type liste_image_inv_fn: str
    :param liste_couple_fn: Filename of the pair/weight list, relative to the
        inversion directory (e.g. ``'liste_couple'``).
    :type liste_couple_fn: str
    :returns: Multi-line parameter string terminated by a newline, ready to be
        written directly to ``input_tio``.
    :rtype: str
    """
    lines = [
        "0.0030  %  smoothing coefficient (threshold = 0.0001)",
        "1   %   remove points with large RMS misclosure  (y=0;n=1)",
        "1.2 %  threshold on RMS misclosure (in rad) ?",
        "1  % range and azimuth sampling ?",
        "0 % iterations to correct unwrapping errors (y:nb_of_iterations,n:0)",
        "0 % iterations to weight pixels of interferograms with large residual? (y:nb_of_iterations,n:0)",
        "0.2 % Scaling value for weighting residuals (in same unit as input files)",
        "0 % iterations to mask (tiny weight) pixels of interferograms with large residual? (y:nb_of_iterations,n:0)",
        "4 % threshold on residual, defining clearly wrong values (in same unit as input files)",
        "1    %   elimination of outliers by the median ? (y=0,n=1)",
        f"{liste_image_inv_fn}",
        "0    % sort by date (0) ou by another variable (1) ?",
        f"{liste_couple_fn}",
        "1   % interferogram format (RMG : 0; R4 :1) (date1-date2_pre_inv.unw or date1-date2.r4)",
        "3100.   %  include interferograms with bperp lower than maximal baseline",
        "1 % Weight input interferograms by coherence or correlation maps ? (y:0,n:1)",
        "0 % coherence file format (RMG : 0; R4 :1) (date1-date2.cor or date1-date2-CC.r4)",
        "1   %   minimal number of interferams using each image",
        "1     % interferograms weighting so that the weight per image is the same (y=0;n=1)",
        "0.6 % maximum fraction of discarded interferograms",
        "0 %  Would you like to restrict the area of inversion ?(y=1,n=0)",
        "1 735 1500 1585  %Give four corners, lower, left, top, right in file pixel coord",
        "1  %    referencing of interferograms by bands (1) or corners (2) ?",
        "5  %     band NW-SW(1), SW-SE(2), NW-NE(3), average of three bands(4), no referencement(5) ?",
        "1   %   Weigthing by image quality (y:0,n:1) ?",
        "0   %  Weigthing by interferogram variance (y:0,n:1) ?  or user given weight (2)?",
        "1    % use of covariance (y:0,n:1) ? (Obsolete)",
        "0   % include a baseline term in inversion ? (y:1;n:0) Requires smoothing !",
        "1   % smoothing by Laplacian, computed with a scheme at 3pts (0) or 5pts (1) ?",
        "2   % weigthed smoothing by the average time step (y:0; n:1, int:2) ?",
        "1    % put the first derivative to zero (y:0; n:1)?",
    ]
    return "\n".join(lines) + "\n"


def _build_launch_script(
    inv_dir: Path,
    width: int,
    height: int,
    n_images: int,
    invers_pixel_bin: str,
    lect_depl_bin: str,
    nodes: int = 1,
    cores: int = 8,
    walltime: str = "12:00:00",
    cluster: str | None = None,
) -> str:
    """Generate the bash launch script for ``invers_pixel_omp``.

    When *cluster* is ``None``, returns a plain bash script with no scheduler
    headers, using the caller-supplied binary paths directly.  When *cluster*
    is ``'isterre'`` or ``'gricad'``, prepends OAR scheduler headers and
    cluster environment setup via
    :func:`~geomulticorr.utils.hpc_tools.oar_header` and
    :func:`~geomulticorr.utils.hpc_tools.cluster_base_env`.

    :param inv_dir: Absolute path to the inversion direction directory
        (e.g. ``inversion_name/inverse_EW``).  Used as the working directory
        in the script and as the ``<`` redirect target for ``input_tio``.
    :type inv_dir: pathlib.Path
    :param width: Raster width in pixels (number of columns).
    :type width: int
    :param height: Raster height in pixels (number of rows).  The
        ``lect_depl_cumule_lin`` call uses ``height - 1``.
    :type height: int
    :param n_images: Number of unique acquisition dates (lines in
        ``liste_image``).
    :type n_images: int
    :param invers_pixel_bin: Full path (local mode) or bare name (HPC mode)
        of the ``invers_pixel_omp`` executable.
    :type invers_pixel_bin: str
    :param lect_depl_bin: Full path (local mode) or bare name (HPC mode) of
        the ``lect_depl_cumule_lin`` executable.
    :type lect_depl_bin: str
    :param nodes: Number of compute nodes to request (OAR, HPC mode only).
    :type nodes: int
    :param cores: Number of CPU cores to request per node (OAR, HPC mode).
    :type cores: int
    :param walltime: Maximum job duration in ``HH:MM:SS`` format (OAR).
    :type walltime: str
    :param cluster: Cluster target — ``None`` for local execution,
        ``'isterre'`` or ``'gricad'`` for OAR-managed HPC clusters.
    :type cluster: str or None
    :returns: Complete bash script content as a single string.
    :rtype: str
    """
    job_name = f"TIO_inv_{inv_dir.name}"
    lines = ["#!/bin/bash\n"]

    if cluster is not None:
        lines += [f"{l}" for l in oar_header(job_name, nodes, cores, walltime, cluster, output_dir=inv_dir)]
        if lines[-1] != "\n":
            lines.append("\n")
        lines += cluster_base_env(cluster)
        lines += [
            "module purge\n",
            "module load intel-devel/18.0.1\n",
        ]

    lines += [
        "\n",
        f"{invers_pixel_bin} <{inv_dir}/input_tio\n",
        "nbd=$(wc -l liste_image)\n",
        "nbdat=${nbd:0:3}\n",
        'echo "$nbd"\n',
        'echo "$nbdat"\n',
        "\n",
        f"{lect_depl_bin} {width} {height - 1} {n_images} 1 1\n",
    ]
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GDAL helpers
# ─────────────────────────────────────────────────────────────────────────────

def tiff2bin(image_path: str | Path, outbin_path: str | Path) -> None:
    """Convert a GeoTIFF to a headerless ENVI Float32 binary file.

    Wraps ``gdal_translate -of envi -ot Float32 --config GDAL_PAM_ENABLED NO``
    to produce the raw binary format required by ``invers_pixel_omp`` as input.
    The ``.hdr`` sidecar written by GDAL is ignored; TIO reads geometry from
    ``File_info.rsc`` instead.

    :param image_path: Path to the source GeoTIFF raster.
    :type image_path: str or pathlib.Path
    :param outbin_path: Destination path for the ENVI binary output (no
        extension — TIO convention).
    :type outbin_path: str or pathlib.Path
    :raises subprocess.CalledProcessError: If ``gdal_translate`` exits with a
        non-zero return code.

    Example
    -------
    .. code-block:: python

        from geomulticorr.inversion.tio_inversion import tiff2bin
        from pathlib import Path

        tiff2bin(Path("pair_EW.tif"), Path("binary/20210815_20220301_EW"))
    """
    cmd = (
        f"gdal_translate -of envi -ot Float32 --config GDAL_PAM_ENABLED NO"
        f" {image_path} {outbin_path}"
    )
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


# ─────────────────────────────────────────────────────────────────────────────
# TIOInversion
# ─────────────────────────────────────────────────────────────────────────────

class TIOInversion:
    """Prepare and launch a TIO time-series inversion from a set of Pair objects.

    Parameters
    ----------
    session:
        An open GMC Session.
    pairs:
        List of Pair objects whose EW/NS displacements will be inverted.
        All pairs must share the same raster grid (resolution, extent, pzone).
    inversion_name:
        Name for the inversion run (e.g., "rock_glacier_2021_2024").
        Outputs are stored in ``{pzone_path}/inversion_{inversion_name}``.
        Defaults to ``"inversion"`` if not provided.
    filter_pipeline:
        Optional FilterPipeline (CCFilter, OutlierFilter, SlopeMask, …) applied
        to each pair's pre-corrected displacements before binary export.
        When None, all pixels are kept.
    tio_binaries_dir:
        Path to compiled TIO inversion binaries (invers_pixel_omp, lect_depl_cumule_lin).
        Defaults to ``/home/cusicand/TIO_install``.
        Raises ``FileNotFoundError`` if binaries are not found in the directory.
    """

    _DIRECTIONS = ("EW", "NS")

    def __init__(
        self,
        session,
        pairs: list,
        inversion_name: str = "inversion",
        filter_pipeline=None,
        tio_binaries_dir: str | Path = "/home/cusicand/TIO_install",
    ) -> None:
        if not pairs:
            raise ValueError("At least one pair is required")

        self.session             = session
        self.pairs               = list(pairs)
        self.inversion_name      = inversion_name
        self.filter_pipeline     = filter_pipeline
        self.tio_binaries_dir    = Path(tio_binaries_dir)

        # Derive pzone from first pair (all pairs must be from same pzone)
        first_pair = self.pairs[0]
        # Extract pzone from pair key format: {pzone}_{date1}-{sensor1}_{date2}-{sensor2}
        pzone_name = first_pair.pa_key.split("_")[0]

        # Build inversion directory within pzone structure:
        # {raster_data}/{pzone_name}/inversion_{inversion_name}
        pzone_path = Path(session.path_raster_data) / pzone_name
        self.inversion_dir = pzone_path / f"inversion_{inversion_name}"

        logger.info(f"TIO inversion '{inversion_name}' will be stored in: {self.inversion_dir}")

        # Validate TIO binaries exist
        self.invers_pixel_omp_bin = _find_tio_executable("invers_pixel_omp", self.tio_binaries_dir)
        self.lect_depl_cumule_lin_bin = _find_tio_executable("lect_depl_cumule_lin", self.tio_binaries_dir)
        logger.info(f"TIO binaries found: invers_pixel_omp, lect_depl_cumule_lin")

        self._image_dates: list[str] | None = None
        self._raster_width:  int | None = None
        self._raster_height: int | None = None
        self.cluster: str | None = None

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TIOInversion(inversion_name='{self.inversion_name}', "
            f"pairs={len(self.pairs)}, "
            f"images={len(self.image_dates)}, "
            f"filter={'yes' if self.filter_pipeline else 'none'})"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def image_dates(self) -> list[str]:
        """Sorted unique image dates (``YYYYmmdd``) across all pairs.

        Derived lazily from ``pa_left.th_date`` and ``pa_right.th_date`` of
        every pair in ``self.pairs``. The result is cached after the first
        access; it is invalidated when :meth:`filter_pairs_by_nmad` removes
        pairs from the list.

        :rtype: list[str]
        """
        if self._image_dates is None:
            dates: set[str] = set()
            for p in self.pairs:
                dates.add(p.pa_left.th_date.replace("-", ""))
                dates.add(p.pa_right.th_date.replace("-", ""))
            self._image_dates = sorted(dates)
        return self._image_dates

    @property
    def raster_shape(self) -> tuple[int, int]:
        """Raster dimensions ``(width_cols, height_rows)`` in pixels.

        Read once from the EW displacement raster of the first pair
        (``pairs[0].pa_ew_path``) and cached. All pairs must share the same
        grid; only the first is inspected.

        :rtype: tuple[int, int]
        """
        if self._raster_width is None:
            r = gu.Raster(str(self.pairs[0].pa_ew_path))
            self._raster_width  = r.width
            self._raster_height = r.height
        return self._raster_width, self._raster_height

    # ── Private helpers ───────────────────────────────────────────────────────

    def _pair_bin_stem(self, pair) -> str:
        """Build the binary file stem ``'{yyyymmdd1}_{yyyymmdd2}'`` for a pair.

        Strips hyphens from both acquisition dates and joins them with an
        underscore, matching the naming convention used in ``binary/`` and
        ``LN_DATA/``.

        :param pair: A GMC Pair object with ``pa_left.th_date`` and
            ``pa_right.th_date`` attributes in ``YYYY-MM-DD`` format.
        :type pair: geomulticorr.core.pair.Pair
        :returns: Date stem of the form ``'20210101_20220301'``.
        :rtype: str
        """
        d1 = pair.pa_left.th_date.replace("-", "")
        d2 = pair.pa_right.th_date.replace("-", "")
        return f"{d1}_{d2}"

    def _load_and_filter(self, pair) -> tuple:
        """Load displacement rasters for *pair* and apply the filter pipeline.

        Prefers corrected rasters (``pa_ew_corr_path``, ``pa_ns_corr_path``)
        when they exist; falls back to raw ASP outputs (``pa_ew_path``,
        ``pa_ns_path``) otherwise. When ``self.filter_pipeline`` is set,
        the pipeline is applied with the correlation-coefficient map if
        available (``pa_cc_path``).

        :param pair: A GMC Pair object providing raster paths and optional
            correlation-coefficient map.
        :type pair: geomulticorr.core.pair.Pair
        :returns: Tuple ``(xDisp, yDisp)`` — filtered EW and NS displacement
            rasters as :class:`geoutils.Raster` objects.
        :rtype: tuple[geoutils.Raster, geoutils.Raster]
        """

        ew_src = pair.pa_ew_corr_path if pair.pa_ew_corr_path.exists() else pair.pa_ew_path
        ns_src = pair.pa_ns_corr_path if pair.pa_ns_corr_path.exists() else pair.pa_ns_path
        xDisp = gu.Raster(str(ew_src))
        yDisp = gu.Raster(str(ns_src))

        if self.filter_pipeline is None:
            return xDisp, yDisp

        fp_kwargs = {}
        if pair.pa_cc_path.exists():
            fp_kwargs["cc"] = gu.Raster(str(pair.pa_cc_path))
        return self.filter_pipeline.apply(xDisp, yDisp, **fp_kwargs)

    def _save_corrected(self, pair, xDisp, yDisp) -> tuple[Path, Path]:
        """Save filtered displacement rasters to the ``corrected/`` subfolder.

        Files are named ``{yyyymmdd1}_{yyyymmdd2}_EW.tif`` and
        ``{yyyymmdd1}_{yyyymmdd2}_NS.tif``, mirroring the binary stem
        convention used by :meth:`_pair_bin_stem`.

        :param pair: The source GMC Pair (used to derive the output filename).
        :type pair: geomulticorr.core.pair.Pair
        :param xDisp: Filtered EW displacement raster.
        :type xDisp: geoutils.Raster
        :param yDisp: Filtered NS displacement raster.
        :type yDisp: geoutils.Raster
        :returns: Paths to the saved ``(EW, NS)`` GeoTIFFs.
        :rtype: tuple[pathlib.Path, pathlib.Path]
        """
        stem    = self._pair_bin_stem(pair)
        ew_path = self.inversion_dir / "corrected" / f"{stem}_EW.tif"
        ns_path = self.inversion_dir / "corrected" / f"{stem}_NS.tif"
        xDisp.save(str(ew_path))
        yDisp.save(str(ns_path))
        return ew_path, ns_path

    def _make_executable(self, path: Path) -> None:
        """Set the executable bit on *path* for owner, group, and others.

        :param path: File to mark as executable.
        :type path: pathlib.Path
        """
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # ── Directory setup ───────────────────────────────────────────────────────

    def setup_directories(self) -> None:
        """Create the full inversion directory tree if it does not exist.

        Directories created (relative to ``self.inversion_dir``):

        * ``corrected/`` — filtered GeoTIFF outputs before binary conversion.
        * ``binary/`` — ENVI Float32 binaries and ``File_info.rsc``.
        * ``inverse_EW/LN_DATA/`` — EW inversion working directory with pair symlinks.
        * ``inverse_NS/LN_DATA/`` — NS inversion working directory with pair symlinks.
        * ``inverse_magn/`` — post-inversion magnitude GeoTIFFs.

        Existing directories are left untouched (``exist_ok=True``).
        """
        for d in [
            self.inversion_dir / "corrected",
            self.inversion_dir / "binary",
            self.inversion_dir / "inverse_EW" / "LN_DATA",
            self.inversion_dir / "inverse_NS" / "LN_DATA",
            self.inversion_dir / "inverse_magn",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Per-pair data export ──────────────────────────────────────────────────

    def export_pair_to_binary(self, pair, overwrite: bool = False) -> tuple[Path, Path]:
        """Load, filter, and export one pair's displacements to ENVI Float32 binary.

        Full chain: load pre-corrected (or raw) rasters → apply
        ``self.filter_pipeline`` → save GeoTIFFs to ``corrected/`` → convert
        to headerless ENVI binary in ``binary/`` via :func:`tiff2bin` → set
        executable bit.  Both output binaries are named
        ``{yyyymmdd1}_{yyyymmdd2}_{EW|NS}``.

        :param pair: GMC Pair object to process.
        :type pair: geomulticorr.core.pair.Pair
        :param overwrite: When ``False`` (default), skip conversion if both
            binary files already exist.  Set to ``True`` to force reprocessing.
        :type overwrite: bool
        :returns: Paths to the EW and NS ENVI binary files.
        :rtype: tuple[pathlib.Path, pathlib.Path]

        Example
        -------
        .. code-block:: python

            ew_bin, ns_bin = inv.export_pair_to_binary(pair, overwrite=True)
        """
        stem   = self._pair_bin_stem(pair)
        ew_bin = self.inversion_dir / "binary" / f"{stem}_EW"
        ns_bin = self.inversion_dir / "binary" / f"{stem}_NS"

        # Check if outputs already exist
        if not overwrite and ew_bin.exists() and ns_bin.exists():
            logger.file(f"Pair {pair.pa_key}: binary files already exist, skipping (use overwrite=True to redo)")
            return ew_bin, ns_bin

        xDisp, yDisp    = self._load_and_filter(pair)
        ew_tiff, ns_tiff = self._save_corrected(pair, xDisp, yDisp)
        tiff2bin(ew_tiff, ew_bin)
        tiff2bin(ns_tiff, ns_bin)
        self._make_executable(ew_bin)
        self._make_executable(ns_bin)
        logger.file(f"Exported {pair.pa_key} to ENVI binary: {ew_bin.name}, {ns_bin.name}")
        return ew_bin, ns_bin

    def _create_symlinks(self, pair) -> None:
        """Create ``.r4`` and ``.r4.rsc`` symlinks in each ``LN_DATA/`` directory.

        ``invers_pixel_omp`` discovers pairs through symlinks in
        ``inverse_{EW|NS}/LN_DATA/``.  The naming convention is::

            binary/{d1}_{d2}_EW      →  inverse_EW/LN_DATA/{d1}-{d2}.r4
            binary/File_info.rsc     →  inverse_EW/LN_DATA/{d1}-{d2}.r4.rsc

        (identically for ``NS``).  Existing symlinks are left untouched.

        :param pair: GMC Pair object providing the two acquisition dates.
        :type pair: geomulticorr.core.pair.Pair
        """
        d1   = pair.pa_left.th_date.replace("-", "")
        d2   = pair.pa_right.th_date.replace("-", "")
        stem = f"{d1}_{d2}"
        r4   = f"{d1}-{d2}.r4"
        file_info = (self.inversion_dir / "binary" / "File_info.rsc").resolve()

        for direction in self._DIRECTIONS:
            bin_src  = (self.inversion_dir / "binary" / f"{stem}_{direction}").resolve()
            ln_dir   = self.inversion_dir / f"inverse_{direction}" / "LN_DATA"
            r4_link  = ln_dir / r4
            rsc_link = ln_dir / f"{r4}.rsc"
            if not r4_link.exists() and not r4_link.is_symlink():
                r4_link.symlink_to(bin_src)
            if not rsc_link.exists() and not rsc_link.is_symlink():
                rsc_link.symlink_to(file_info)

    def _tot_to_geotiff(self, tot_file: Path, ref_raster_path: Path) -> Path:
        """Convert a raw ENVI Float32 TOT binary to a georeferenced GeoTIFF.

        ``lect_depl_cumule_lin`` is called with ``height - 1`` rows, so TOT
        files contain ``(height - 1) × width`` float32 values.  CRS and
        geotransform (top-left anchor + pixel size) are copied from
        *ref_raster_path* via a ``rasterio`` profile, with ``height`` updated
        to ``height - 1``.  Pixels that are zero or non-finite are set to NaN.

        :param tot_file: Path to the raw TOT binary (no extension, e.g.
            ``inverse_EW/TOT_20210901``).
        :type tot_file: pathlib.Path
        :param ref_raster_path: Path to any same-grid GeoTIFF whose CRS and
            geotransform will be copied (typically ``pairs[0].pa_ew_path``).
        :type ref_raster_path: pathlib.Path
        :returns: Path to the newly written GeoTIFF (same stem, ``.tif``
            extension, LZW-compressed).
        :rtype: pathlib.Path
        """

        width, height = self.raster_shape
        tot_height    = height - 1  # lect_depl_cumule_lin uses height-1

        data = np.fromfile(str(tot_file), dtype=np.float32).reshape(tot_height, width)
        mask = ~np.isfinite(data) | (data == 0)
        data[mask] = np.nan

        out = tot_file.with_suffix(".tif")
        with rasterio.open(str(ref_raster_path)) as ref_ds:
            profile = ref_ds.profile.copy()
        profile.update(
            height=tot_height,
            width=width,
            count=1,
            dtype=np.float32,
            nodata=np.nan,
            compress="lzw",
        )
        with rasterio.open(str(out), "w", **profile) as dst:
            dst.write(data, 1)
        return out

    def _compute_magnitude(self, ew_tif: Path, ns_tif: Path, out_tif: Path) -> Path:
        """Compute displacement magnitude ``sqrt(EW² + NS²)`` and write to *out_tif*.

        Uses :class:`geoutils.Raster` arithmetic, which preserves the CRS,
        geotransform, and NoData mask from the input rasters.  Mirrors the
        pattern used in ``pair.py`` for single-pair magnitude maps.

        :param ew_tif: Path to the East–West displacement GeoTIFF.
        :type ew_tif: pathlib.Path
        :param ns_tif: Path to the North–South displacement GeoTIFF.
        :type ns_tif: pathlib.Path
        :param out_tif: Destination path for the magnitude GeoTIFF.
        :type out_tif: pathlib.Path
        :returns: *out_tif* (the path passed in), for convenience.
        :rtype: pathlib.Path
        """

        ew   = gu.Raster(str(ew_tif))
        ns   = gu.Raster(str(ns_tif))
        magn = (ew ** 2 + ns ** 2) ** 0.5
        magn.save(str(out_tif))
        return out_tif

    # ── TIO text files ────────────────────────────────────────────────────────

    def write_file_info_rsc(self) -> Path:
        """Write ``binary/File_info.rsc`` with the raster pixel geometry.

        Pixel dimensions are read from ``pairs[0].pa_ew_path`` via
        :attr:`raster_shape`.  The file follows the *ocslide* convention:
        ``WIDTH`` = columns, ``FILE_LENGTH`` = rows, with zero-based
        ``XMIN``/``XMAX``/``YMIN``/``YMAX`` bounds.

        The file is also marked executable (TIO convention).

        :returns: Path to the written ``File_info.rsc``.
        :rtype: pathlib.Path

        Example
        -------
        .. code-block:: python

            rsc_path = inv.write_file_info_rsc()
            print(rsc_path.read_text())
        """
        width, height = self.raster_shape
        text = (
            f"FILE_LENGTH                              {height}\n"
            f"WIDTH                                    {width}\n"
            f"XMIN                                     0\n"
            f"XMAX                                     {width - 1}\n"
            f"YMIN                                     0\n"
            f"YMAX                                     {height - 1}\n"
        )
        out = self.inversion_dir / "binary" / "File_info.rsc"
        out.write_text(text)
        self._make_executable(out)
        return out

    def write_liste_image(self) -> None:
        """Write ``liste_image`` (one ``YYYYmmdd`` per line) to both inversion directories.

        The file lists every unique acquisition date derived from
        :attr:`image_dates`.  It is written to both ``inverse_EW/liste_image``
        and ``inverse_NS/liste_image`` and marked executable.
        """
        content = "\n".join(self.image_dates) + "\n"
        for direction in self._DIRECTIONS:
            p = self.inversion_dir / f"inverse_{direction}" / "liste_image"
            p.write_text(content)
            self._make_executable(p)

    def write_liste_image_inv(self) -> None:
        """Write ``liste_image_inv`` with four columns per date line.

        Each row contains: ``YYYYmmdd``, decimal year, Δ decimal year from
        the first acquisition, and a trailing ``0``.  Written to both
        ``inverse_EW/`` and ``inverse_NS/`` and marked executable.
        """
        dates_dt  = [datetime.strptime(d, "%Y%m%d") for d in self.image_dates]
        dec_years = [_year_dec(d) for d in dates_dt]
        d0        = dec_years[0]
        lines = [
            f"{d} {y:.6f} {abs(y - d0):.6f} 0\n"
            for d, y in zip(self.image_dates, dec_years)
        ]
        content = "".join(lines)
        for direction in self._DIRECTIONS:
            p = self.inversion_dir / f"inverse_{direction}" / "liste_image_inv"
            p.write_text(content)
            self._make_executable(p)

    def _pair_quality_metrics(
        self,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Collect ``(ew_nmad, ns_nmad, cc, dt_days)`` per pair for ``'quality'`` mode.

        EW/NS NMAD come from ``final_corrected_stats`` (falling back to
        ``raw_corr_stats``); CC is ``raw_corr_stats.cc.cc_quality_gte_050``; Δt is
        ``pair.pa_dt_days``. Missing values are recorded as ``nan`` (treated as
        neutral downstream) with a warning — a pair is never dropped here.

        :returns: Four lists (ew_nmad, ns_nmad, cc, dt_days), each aligned to
            ``self.pairs``.
        :rtype: tuple[list[float], list[float], list[float], list[float]]
        """
        def _num(v):
            return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else float("nan")

        ew_nmads: list[float] = []
        ns_nmads: list[float] = []
        ccs: list[float] = []
        dts: list[float] = []
        for pair in self.pairs:
            try:
                js = load_pair_stats(pair)
            except FileNotFoundError:
                logger.warning(
                    f"[quality weight] no stats for {pair.pa_key} — "
                    "NMAD/CC treated as neutral"
                )
                js = {}
            block = js.get("final_corrected_stats") or js.get("raw_corr_stats", {})
            ew_nmads.append(_num(block.get("ew", {}).get("nmad", float("nan"))))
            ns_nmads.append(_num(block.get("ns", {}).get("nmad", float("nan"))))
            cc = js.get("raw_corr_stats", {}).get("cc", {}).get(
                "cc_quality_gte_050", float("nan")
            )
            ccs.append(_num(cc))
            dts.append(float(pair.pa_dt_days))
        return ew_nmads, ns_nmads, ccs, dts

    def compute_pair_weights(
        self,
        weight_mode: str = "uniform",
        *,
        slope: float = 2.0,
        min_weight: float = 0.1,
        dt_range: tuple[float, float] | None = None,
        sharpness: float = 5.0,
        w_min: float = 0.0,
        invert: bool = False,
        combine: str = "geomean",
        alpha: float = 1.0 / 3.0,
        beta: float = 1.0 / 3.0,
        gamma: float = 1.0 / 3.0,
        cc_gamma: float = 1.0,
        direction: str = "max",
    ) -> list[float]:
        """Compute one weight per pair (aligned to ``self.pairs``), in ``[0, 1]``.

        This is the single source of pair weights consumed by
        :meth:`write_liste_couple`, :meth:`plot_weights`, and
        :meth:`explore_weights`. All modes return weights in ``[0, 1]``.

        Modes (date-based): ``'uniform'``, ``'temporal'``, ``'relative_temporal'``,
        ``'sigmoid'``, ``'parametric'`` — see :meth:`write_liste_couple` for the
        formulae; behaviour is unchanged.

        New quality mode ``'quality'`` combines corrected **NMAD**, **CC-quality**
        (``cc_quality_gte_050``) and **Δt** into a single weight:

        * ``w_nmad`` = :func:`normalise_invert` of the NMAD selected by
          *direction* — ``ew_nmad`` (``"EW"``), ``ns_nmad`` (``"NS"``), or the
          per-pair ``max`` (``"max"``) — normalised within that direction
          (lowest NMAD → 1.0),
        * ``w_cc``   = ``cc_quality_gte_050 ** cc_gamma``,
        * ``w_dt``   = :func:`normalise_dt` of ``pa_dt_days`` (short Δt → 1.0,
          reversed by *invert*),

        combined by :func:`combine_weights` using *combine* (``'geomean'``,
        ``'wmean'`` with *alpha*/*beta*/*gamma*, or ``'product'``). Pairs missing
        a component contribute it as neutral (1.0).

        :param weight_mode: Weighting strategy (see above).
        :param slope: ``'parametric'`` power exponent.
        :param min_weight: ``'parametric'`` floor.
        :param dt_range: ``(min_days, max_days)`` bounds for ``'relative_temporal'``
            / ``'sigmoid'`` (derived from the pairs when ``None``).
        :param sharpness: ``'sigmoid'`` steepness.
        :param w_min: Floor rescaling ``w → w_min + (1 − w_min)·w`` for all modes
            except ``'uniform'`` and ``'parametric'``.
        :param invert: Reverse the temporal direction (``'relative_temporal'``,
            ``'sigmoid'``, and the Δt term of ``'quality'``).
        :param combine: Combination method for ``'quality'``.
        :param alpha: ``'quality'``/``'wmean'`` weight of the NMAD term.
        :param beta: ``'quality'``/``'wmean'`` weight of the CC term.
        :param gamma: ``'quality'``/``'wmean'`` weight of the Δt term.
        :param cc_gamma: Exponent applied to the CC sub-weight in ``'quality'``.
        :param direction: NMAD source for ``'quality'`` — ``"EW"``, ``"NS"``, or
            ``"max"`` (per-pair maximum). Ignored by date-based modes.
        :returns: Weights aligned to ``self.pairs``.
        :rtype: list[float]
        """
        n = len(self.pairs)

        if weight_mode == "quality":
            ew_nmads, ns_nmads, ccs, dts = self._pair_quality_metrics()
            if direction == "EW":
                nmads = ew_nmads
            elif direction == "NS":
                nmads = ns_nmads
            else:  # "max" — worse of the two directions per pair
                nmads = [
                    (max(e, s) if (math.isfinite(e) and math.isfinite(s))
                     else (e if math.isfinite(e) else s))
                    for e, s in zip(ew_nmads, ns_nmads)
                ]
            w_nmad = normalise_invert(nmads)
            w_dt = normalise_dt(dts, invert=invert)
            w_cc = [
                (c ** cc_gamma) if math.isfinite(c) else float("nan") for c in ccs
            ]
            return [
                combine_weights(w_nmad[i], w_cc[i], w_dt[i], combine, alpha, beta, gamma)
                for i in range(n)
            ]

        # ── date-based modes ──────────────────────────────────────────────────
        if weight_mode == "parametric":
            midpoints = []
            for pair in self.pairs:
                d1 = pair.pa_left.th_date.replace("-", "")
                d2 = pair.pa_right.th_date.replace("-", "")
                y1 = _year_dec(datetime.strptime(d1, "%Y%m%d"))
                y2 = _year_dec(datetime.strptime(d2, "%Y%m%d"))
                midpoints.append((y1 + y2) / 2.0)
            t_min   = min(midpoints)
            t_max   = max(midpoints)
            t_range = (t_max - t_min) or 1.0

        if weight_mode in ("relative_temporal", "sigmoid"):
            if dt_range is not None:
                _dt_min = dt_range[0] / 365.25
                _dt_max = dt_range[1] / 365.25
            else:
                dt_yrs = []
                for pair in self.pairs:
                    d1 = pair.pa_left.th_date.replace("-", "")
                    d2 = pair.pa_right.th_date.replace("-", "")
                    y1 = _year_dec(datetime.strptime(d1, "%Y%m%d"))
                    y2 = _year_dec(datetime.strptime(d2, "%Y%m%d"))
                    dt_yrs.append(abs(y2 - y1))
                _dt_min = min(dt_yrs)
                _dt_max = max(dt_yrs)
            _dt_span = (_dt_max - _dt_min) or 1.0

        weights: list[float] = []
        for i, pair in enumerate(self.pairs):
            if weight_mode == "temporal":
                d1 = pair.pa_left.th_date.replace("-", "")
                d2 = pair.pa_right.th_date.replace("-", "")
                y1 = _year_dec(datetime.strptime(d1, "%Y%m%d"))
                y2 = _year_dec(datetime.strptime(d2, "%Y%m%d"))
                w  = 1.0 / math.pow(1.0 + math.pow(y2 - y1, 2), 2)
            elif weight_mode == "relative_temporal":
                d1 = pair.pa_left.th_date.replace("-", "")
                d2 = pair.pa_right.th_date.replace("-", "")
                y1     = _year_dec(datetime.strptime(d1, "%Y%m%d"))
                y2     = _year_dec(datetime.strptime(d2, "%Y%m%d"))
                t_norm = max(0.0, min(1.0, (abs(y2 - y1) - _dt_min) / _dt_span))
                if invert:
                    t_norm = 1.0 - t_norm
                w      = 1.0 / math.pow(1.0 + math.pow(t_norm, 2), 2)
            elif weight_mode == "sigmoid":
                d1 = pair.pa_left.th_date.replace("-", "")
                d2 = pair.pa_right.th_date.replace("-", "")
                y1     = _year_dec(datetime.strptime(d1, "%Y%m%d"))
                y2     = _year_dec(datetime.strptime(d2, "%Y%m%d"))
                t_norm = max(0.0, min(1.0, (abs(y2 - y1) - _dt_min) / _dt_span))
                if invert:
                    t_norm = 1.0 - t_norm
                w      = 1.0 / (1.0 + math.exp(sharpness * (t_norm - 0.5)))
            elif weight_mode == "parametric":
                t_norm = (midpoints[i] - t_min) / t_range
                w = min_weight + (1.0 - min_weight) * math.pow(t_norm, slope)
            else:
                w = 1.0
            # Apply w_min floor (skipped for uniform=1.0 and parametric which has its own floor)
            if w_min > 0.0 and weight_mode not in ("uniform", "parametric"):
                w = w_min + (1.0 - w_min) * w
            weights.append(w)
        return weights

    def write_liste_couple(
        self,
        weight_mode: str | None = None,
        slope: float = 2.0,
        min_weight: float = 0.1,
        dt_range: tuple[float, float] | None = None,
        sharpness: float = 5.0,
        w_min: float = 0.0,
        invert: bool = False,
        combine: str = "geomean",
        alpha: float = 1.0 / 3.0,
        beta: float = 1.0 / 3.0,
        gamma: float = 1.0 / 3.0,
        cc_gamma: float = 1.0,
        weights: list[float] | dict | None = None,
        sync_geodb: bool = True,
    ) -> dict[str, list[float]]:
        """Write ``liste_couple`` (date1, date2, weight) for each pair in ``self.pairs``.

        Only pairs present in ``self.pairs`` are listed — not all permutations
        of image dates.  Weights are computed **per direction** by
        :meth:`compute_pair_weights` (EW-NMAD → ``inverse_EW/liste_couple``,
        NS-NMAD → ``inverse_NS/liste_couple``) unless a precomputed *weights*
        vector/mapping is supplied.  For date-based modes both directions are
        identical.  Each file is marked executable.

        :param weight_mode: Weighting strategy. ``None`` (default) means
            ``'uniform'`` when computing, but when explicit *weights* are supplied
            the recorded ``pa_weight_mode`` is taken from
            :meth:`explore_weights` (if these are its weights) or labelled
            ``'explicit'`` — so the DB always reflects what produced the weights.
            One of:

            * ``'uniform'`` — weight = 1.0 for all pairs.
            * ``'temporal'`` — ``1 / (1 + Δt²)²`` where Δt is in absolute
              decimal years.  Works best when pairs span a wide Δt range.
            * ``'relative_temporal'`` — same Lorentzian decay, but Δt is
              normalised to ``[0, 1]`` within *dt_range* (days).  If
              *dt_range* is ``None``, bounds are derived from the actual pair
              set.  Shortest pair → w = 1.0; longest → minimum weight.
            * ``'sigmoid'`` — ``1 / (1 + exp(sharpness × (t_norm − 0.5)))``.
              Produces a steep S-shaped step: short pairs get w ≈ 1.0, long
              pairs get w ≈ 0.0, with a sharp transition in the middle.
              *sharpness* controls the steepness.  Uses *dt_range* for
              normalisation (same as ``'relative_temporal'``).
            * ``'parametric'`` — ``min_weight + (1 − min_weight) × t_norm^slope``,
              where *t_norm* is the pair's temporal midpoint normalised to
              ``[0, 1]`` over the full survey period.  Older pairs receive
              lower weight.
            * ``'quality'`` — combines corrected NMAD, CC-quality and Δt (see
              :meth:`compute_pair_weights`); tuned by *combine*, *alpha*,
              *beta*, *gamma*, *cc_gamma*.

        :type weight_mode: str
        :param slope: Power exponent for ``'parametric'`` mode.
        :type slope: float
        :param min_weight: Minimum weight floor for ``'parametric'`` mode.
            Pairs at the earliest temporal midpoint receive this weight.
        :type min_weight: float
        :param dt_range: ``(min_days, max_days)`` normalisation bounds for
            ``'relative_temporal'`` and ``'sigmoid'``.  Pairs shorter than
            *min_days* are clamped to w = 1.0; pairs longer than *max_days*
            are clamped to the minimum weight.  When ``None`` (default),
            bounds are derived from the actual pair set
            (e.g. ``dt_range=(300, 400)``).
        :type dt_range: tuple[float, float] or None
        :param sharpness: Steepness of the sigmoid transition.  Higher values
            produce a harder step (10 → near-binary; 2 → gentle S-curve).
        :type sharpness: float
        :param w_min: Minimum weight floor for all modes except ``'uniform'``
            and ``'parametric'``.  Raw weights are rescaled as
            ``w = w_min + (1 − w_min) × w_raw``, so the output range becomes
            ``[w_min, 1.0]`` instead of ``[0.0, 1.0]``.  Default ``0.0``
            means no floor (e.g. ``w_min=0.25``).
        :type w_min: float
        :param invert: When ``True``, reverses the weighting direction for
            ``'relative_temporal'``, ``'sigmoid'`` and the Δt term of
            ``'quality'`` — longer pairs get higher weight.  No effect on
            ``'uniform'``, ``'temporal'`` or ``'parametric'``.
        :type invert: bool
        :param combine: Combination method for ``'quality'`` (``'geomean'``,
            ``'wmean'``, ``'product'``).
        :param alpha: ``'quality'`` weight of the NMAD term (``'wmean'``).
        :param beta: ``'quality'`` weight of the CC term (``'wmean'``).
        :param gamma: ``'quality'`` weight of the Δt term (``'wmean'``).
        :param cc_gamma: Exponent on the CC sub-weight in ``'quality'``.
        :param weights: Optional precomputed weights bypassing the mode math.
            Either a single list (length ``len(self.pairs)``, written to **both**
            directions) or a mapping ``{"EW": [...], "NS": [...]}`` for
            per-direction weights (e.g. ``inv._last_weights`` from
            :meth:`explore_weights`).
        :type weights: list[float] | dict | None
        :param sync_geodb: When ``True`` (default), persist each pair's EW/NS
            weight to its stats JSON and sync ``pa_ew_weight``/``pa_ns_weight``/
            ``pa_weight``/``pa_weight_mode`` to the Pairs layer via
            :meth:`~geomulticorr.core.session.Session.sync_pairs_weights`.
        :type sync_geodb: bool
        :returns: The weights written, as ``{"EW": [...], "NS": [...]}``.
        :rtype: dict[str, list[float]]
        """
        def _check(vec, label):
            if len(vec) != len(self.pairs):
                raise ValueError(
                    f"weights[{label}] has length {len(vec)} but there are "
                    f"{len(self.pairs)} pairs."
                )
            return [float(w) for w in vec]

        if weights is not None:
            if isinstance(weights, dict):
                w_ew = _check(weights["EW"], "EW")
                w_ns = _check(weights["NS"], "NS")
            else:  # single vector → both directions
                w_ew = w_ns = _check(weights, "EW/NS")
            # Resolve the recorded mode label for explicit weights: an explicit
            # weight_mode wins; otherwise reuse what explore_weights tuned (when
            # these are its weights); else be honest and call it "explicit".
            if weight_mode is not None:
                res_mode, res_combine = weight_mode, combine
            elif weights is getattr(self, "_last_weights", None) and \
                    getattr(self, "_last_weight_mode", None):
                res_mode = self._last_weight_mode
                res_combine = getattr(self, "_last_combine", combine)
            else:
                res_mode, res_combine = "explicit", None
        else:
            res_mode = weight_mode or "uniform"
            res_combine = combine
            common = dict(
                slope=slope, min_weight=min_weight, dt_range=dt_range,
                sharpness=sharpness, w_min=w_min, invert=invert,
                combine=combine, alpha=alpha, beta=beta, gamma=gamma, cc_gamma=cc_gamma,
            )
            w_ew = self.compute_pair_weights(res_mode, direction="EW", **common)
            w_ns = self.compute_pair_weights(res_mode, direction="NS", **common)

        for direction, wvec in (("EW", w_ew), ("NS", w_ns)):
            lines = []
            for pair, w in zip(self.pairs, wvec):
                d1 = pair.pa_left.th_date.replace("-", "")
                d2 = pair.pa_right.th_date.replace("-", "")
                lines.append(f"{d1} {d2} {w:.6f}\n")
            p = self.inversion_dir / f"inverse_{direction}" / "liste_couple"
            p.write_text("".join(lines))
            self._make_executable(p)

        if sync_geodb:
            self._sync_pair_weights(w_ew, w_ns, res_mode, res_combine)

        return {"EW": w_ew, "NS": w_ns}

    def _sync_pair_weights(
        self,
        w_ew: list[float],
        w_ns: list[float],
        weight_mode: str,
        combine: str | None = None,
    ) -> None:
        """Persist per-direction weights to each stats JSON and sync them to the DB.

        Writes a ``weight`` section (``ew``/``ns``/``mode``/``combine``) into every
        pair's stats JSON, then promotes ``pa_ew_weight``/``pa_ns_weight``/
        ``pa_weight``/``pa_weight_mode`` into the Pairs layer via
        :meth:`~geomulticorr.core.session.Session.sync_pairs_weights`.
        """
        import geomulticorr.stats.stats as gmc_stats

        mode_label = weight_mode if weight_mode != "quality" else f"quality:{combine}"
        for pair, ew, ns in zip(self.pairs, w_ew, w_ns):
            try:
                gmc_stats.save_pair_weight(
                    pair, float(ew), float(ns), mode=mode_label,
                    combine=combine, params={"weight_mode": weight_mode},
                )
            except Exception as exc:  # never let a bad JSON abort the write
                logger.warning(f"[weight sync] could not save weight for {pair.pa_key}: {exc}")
        try:
            self.session.sync_pairs_weights(self.pairs)
        except Exception as exc:
            logger.warning(f"[weight sync] geodatabase sync skipped: {exc}")

    # ── Weight visualisation / exploration ──────────────────────────────────────

    #: Controls that actually affect each weighting mode (drives the explorer's
    #: show/hide logic). ``direction`` and ``mode`` are always shown.
    _MODE_CONTROLS: dict[str, set[str]] = {
        "uniform": set(),
        "temporal": {"w_min"},
        "relative_temporal": {"w_min", "invert"},
        "sigmoid": {"sharpness", "w_min", "invert"},
        "parametric": {"slope", "min_weight"},
        "quality": {"combine", "cc_gamma", "invert"},
    }

    def plot_weights(self, modes: list[str] | None = None, direction: str = "max",
                     **params):
        """Static Plotly figure comparing weight-vs-Δt across weighting modes.

        Overlays one marker series per mode so the modes can be compared
        side-by-side before committing. Requires plotly.

        :param modes: Modes to overlay. Defaults to
            ``["uniform", "relative_temporal", "sigmoid", "quality"]``.
        :param direction: NMAD direction for ``'quality'`` (``"EW"``/``"NS"``/``"max"``).
        :param params: Extra keyword arguments forwarded to
            :meth:`compute_pair_weights` (e.g. ``combine``, ``sharpness``).
        :returns: A :class:`plotly.graph_objects.Figure`.
        """
        import plotly.graph_objects as go

        modes = modes or ["uniform", "relative_temporal", "sigmoid", "quality"]
        dts = [float(p.pa_dt_days) for p in self.pairs]
        keys = [p.pa_key for p in self.pairs]

        fig = go.Figure()
        for mode in modes:
            w = self.compute_pair_weights(mode, direction=direction, **params)
            fig.add_trace(
                go.Scatter(
                    x=dts, y=w, mode="markers", name=mode, text=keys,
                    hovertemplate="mode=" + mode
                    + "<br>pair=%{text}<br>Δt=%{x:.0f} d<br>weight=%{y:.3f}<extra></extra>",
                )
            )
        fig.update_layout(
            title=f"TIO pair weights — '{self.inversion_name}'",
            xaxis_title="Temporal baseline Δt (days)",
            yaxis_title="weight", yaxis_range=[-0.02, 1.02],
            template="plotly_white",
        )
        return fig

    def explore_weights(self, weight_mode: str = "quality", **defaults):
        """Interactive Plotly + ipywidgets tool to tune and preview pair weights.

        Renders a live scatter of **weight vs Δt** with one point per pair **per
        direction** (EW circles + NS diamonds — 2 × ``len(pairs)`` points in
        ``both`` view, one series otherwise). Sliders/dropdowns recompute the
        weights on every change, and only the controls that affect the selected
        mode are shown. The current per-direction weights are stored on
        ``self._last_weights`` (a ``{"EW": [...], "NS": [...]}`` dict) so they can
        be written directly::

            inv.explore_weights()                       # tune interactively
            inv.write_liste_couple(weights=inv._last_weights)
            # or inv.prepare(weights=inv._last_weights)

        Requires plotly and ipywidgets (Jupyter). Returns the assembled
        ``ipywidgets.VBox`` (also displays inline in a notebook).

        :param weight_mode: Initial mode selected in the dropdown.
        :param defaults: Initial values for any control (``direction``,
            ``combine``, ``alpha``, ``beta``, ``gamma``, ``cc_gamma``,
            ``sharpness``, ``slope``, ``min_weight``, ``w_min``, ``invert``).
        :returns: An ``ipywidgets.VBox`` containing the controls and figure.
        """
        import ipywidgets as widgets
        import plotly.graph_objects as go

        ew_nmads, ns_nmads, ccs, dts = self._pair_quality_metrics()
        keys = [p.pa_key for p in self.pairs]
        dirs = [str(getattr(p, "pa_direction", "")).capitalize() for p in self.pairs]
        # list-of-lists (not np.column_stack) so float NMAD/CC keep their dtype
        # alongside the string correlation direction.
        cd_ew = [[e, c, d] for e, c, d in zip(ew_nmads, ccs, dirs)]
        cd_ns = [[n, c, d] for n, c, d in zip(ns_nmads, ccs, dirs)]

        def _compute():
            common = dict(
                combine=c_combine.value, alpha=c_alpha.value, beta=c_beta.value,
                gamma=c_gamma.value, cc_gamma=c_ccg.value, sharpness=c_sharp.value,
                slope=c_slope.value, min_weight=c_minw.value, w_min=c_wmin.value,
                invert=c_invert.value,
            )
            w_ew = self.compute_pair_weights(c_mode.value, direction="EW", **common)
            w_ns = self.compute_pair_weights(c_mode.value, direction="NS", **common)
            return w_ew, w_ns

        # ── controls ──
        c_mode = widgets.Dropdown(
            options=["uniform", "temporal", "relative_temporal", "sigmoid",
                     "parametric", "quality"],
            value=weight_mode, description="mode")
        c_dir = widgets.Dropdown(options=["both", "EW", "NS"],
                                 value=defaults.get("direction", "both"),
                                 description="direction")
        c_combine = widgets.Dropdown(options=["geomean", "wmean", "product"],
                                     value=defaults.get("combine", "geomean"),
                                     description="combine")
        c_alpha = widgets.FloatSlider(value=defaults.get("alpha", 1/3), min=0, max=1,
                                      step=0.05, description="α nmad")
        c_beta = widgets.FloatSlider(value=defaults.get("beta", 1/3), min=0, max=1,
                                     step=0.05, description="β cc")
        c_gamma = widgets.FloatSlider(value=defaults.get("gamma", 1/3), min=0, max=1,
                                      step=0.05, description="γ dt")
        c_ccg = widgets.FloatSlider(value=defaults.get("cc_gamma", 1.0), min=0.1, max=3.0,
                                    step=0.1, description="cc_gamma")
        c_sharp = widgets.FloatSlider(value=defaults.get("sharpness", 5.0), min=1, max=12,
                                      step=0.5, description="sharpness")
        c_slope = widgets.FloatSlider(value=defaults.get("slope", 2.0), min=0.5, max=5,
                                      step=0.1, description="slope")
        c_minw = widgets.FloatSlider(value=defaults.get("min_weight", 0.1), min=0, max=1,
                                     step=0.05, description="min_weight")
        c_wmin = widgets.FloatSlider(value=defaults.get("w_min", 0.0), min=0, max=0.9,
                                     step=0.05, description="w_min")
        c_invert = widgets.Checkbox(value=defaults.get("invert", False), description="invert")

        # map control name -> widget, for show/hide
        tunables = {
            "combine": c_combine, "alpha": c_alpha, "beta": c_beta, "gamma": c_gamma,
            "cc_gamma": c_ccg, "sharpness": c_sharp, "slope": c_slope,
            "min_weight": c_minw, "w_min": c_wmin, "invert": c_invert,
        }

        def _apply_visibility():
            relevant = set(self._MODE_CONTROLS.get(c_mode.value, set()))
            # α/β/γ only matter for the weighted-mean combine of quality mode
            if c_mode.value == "quality" and c_combine.value == "wmean":
                relevant |= {"alpha", "beta", "gamma"}
            for name, w in tunables.items():
                w.layout.display = None if name in relevant else "none"

        w_ew, w_ns = _compute()
        self._last_weights = {"EW": w_ew, "NS": w_ns}
        self._last_weight_mode = c_mode.value
        self._last_combine = c_combine.value

        fig = go.FigureWidget(
            data=[
                go.Scatter(x=dts, y=w_ew, mode="markers", name="EW", text=keys,
                           customdata=cd_ew, opacity=0.75,
                           marker=dict(size=9, symbol="circle", color="#1f77b4"),
                           hovertemplate="pair_key=%{text}<br>weight=%{y:.3f}"
                                         "<br>map=EW"
                                         "<br>corr_dir=%{customdata[2]}"
                                         "<br>NMAD=%{customdata[0]:.3f}"
                                         "<br>CC=%{customdata[1]:.2f}"
                                         "<br>Δt=%{x:.0f} d<extra></extra>"),
                go.Scatter(x=dts, y=w_ns, mode="markers", name="NS", text=keys,
                           customdata=cd_ns, opacity=0.75,
                           marker=dict(size=9, symbol="diamond", color="#ff7f0e"),
                           hovertemplate="pair_key=%{text}<br>weight=%{y:.3f}"
                                         "<br>map=NS"
                                         "<br>corr_dir=%{customdata[2]}"
                                         "<br>NMAD=%{customdata[0]:.3f}"
                                         "<br>CC=%{customdata[1]:.2f}"
                                         "<br>Δt=%{x:.0f} d<extra></extra>"),
            ]
        )
        fig.update_layout(
            title=f"TIO pair weights — '{self.inversion_name}' [{c_mode.value}]",
            xaxis_title="Temporal baseline Δt (days)",
            yaxis_title="weight", yaxis_range=[-0.02, 1.02],
            template="plotly_white", height=460,
        )

        def _apply_direction():
            show_ew = c_dir.value in ("both", "EW")
            show_ns = c_dir.value in ("both", "NS")
            fig.data[0].visible = show_ew
            fig.data[1].visible = show_ns

        def _update(_change=None):
            _apply_visibility()
            w_ew, w_ns = _compute()
            self._last_weights = {"EW": w_ew, "NS": w_ns}
            self._last_weight_mode = c_mode.value
            self._last_combine = c_combine.value
            with fig.batch_update():
                fig.data[0].y = w_ew
                fig.data[1].y = w_ns
                _apply_direction()
                fig.layout.title.text = (
                    f"TIO pair weights — '{self.inversion_name}' [{c_mode.value}]"
                )

        for c in (c_mode, c_dir, *tunables.values()):
            c.observe(_update, names="value")

        _apply_visibility()
        _apply_direction()

        row1 = widgets.HBox([c_mode, c_dir, c_combine, c_invert])
        row2 = widgets.HBox([c_alpha, c_beta, c_gamma, c_ccg])
        row3 = widgets.HBox([c_sharp, c_slope, c_minw, c_wmin])
        return widgets.VBox([row1, row2, row3, fig])

    def write_input_tio(self) -> None:
        """Write the ``input_tio`` parameter file to both inversion directories.

        Content is produced by :func:`_build_input_tio_text` with GMC defaults.
        The file is written to ``inverse_EW/input_tio`` and
        ``inverse_NS/input_tio`` and marked executable.
        """
        content = _build_input_tio_text()
        for direction in self._DIRECTIONS:
            p = self.inversion_dir / f"inverse_{direction}" / "input_tio"
            p.write_text(content)
            self._make_executable(p)

    def write_launch_script(
        self,
        cluster: str | None = None,
        nodes: int = 1,
        cores: int = 8,
        walltime: str = "12:00:00",
    ) -> None:
        """Write bash launch scripts to ``inverse_EW/`` and ``inverse_NS/``.

        When *cluster* is ``None``, produces a plain bash script with the
        locally resolved binary paths and no scheduler headers.  When
        *cluster* is ``'isterre'`` or ``'gricad'``, prepends OAR headers
        via :func:`~geomulticorr.utils.hpc_tools.oar_header` and uses bare
        binary names (resolved via the cluster ``PATH`` after module load).

        Scripts are named ``launch_TIO_inv_{EW|NS}.sh`` and marked executable.

        :param cluster: Execution target — ``None`` for local execution,
            ``'isterre'`` or ``'gricad'`` for OAR-managed HPC clusters.
        :type cluster: str or None
        :param nodes: Number of compute nodes to request (OAR only).
        :type nodes: int
        :param cores: Number of CPU cores per node (OAR only).
        :type cores: int
        :param walltime: Maximum wall-clock time in ``HH:MM:SS`` (OAR only).
        :type walltime: str

        Example
        -------
        .. code-block:: python

            inv.write_launch_script()                    # local
            inv.write_launch_script(cluster="gricad",
                                    cores=16,
                                    walltime="06:00:00") # HPC
        """
        validate_cluster(cluster)

        invers_pixel_bin = self.invers_pixel_omp_bin
        lect_depl_bin    = self.lect_depl_cumule_lin_bin

        width, height = self.raster_shape
        n_images = len(self.image_dates)
        for direction in self._DIRECTIONS:
            inv_dir = self.inversion_dir / f"inverse_{direction}"
            content = _build_launch_script(
                inv_dir, width, height, n_images,
                invers_pixel_bin=invers_pixel_bin,
                lect_depl_bin=lect_depl_bin,
                nodes=nodes, cores=cores, walltime=walltime,
                cluster=cluster,
            )
            p = inv_dir / f"launch_TIO_inv_{direction}.sh"
            p.write_text(content)
            self._make_executable(p)
            mode = "local" if cluster is None else cluster
            logger.info(f"Launch script written [{mode}]: {p.name}")

        # Remember which cluster profile the scripts on disk were written
        # for, so launch() can catch a mode mismatch instead of failing at
        # `oarsub` time with a script that has the wrong (or no) header.
        self.cluster = "local" if cluster is None else cluster

    # ── Orchestration ─────────────────────────────────────────────────────────

    def filter_pairs_by_nmad(
        self,
        threshold: float = 0.7,
        correction_name: str | None = None,
    ) -> tuple[list, list]:
        """Remove pairs whose NMAD exceeds *threshold* from ``self.pairs``.

        Reads each pair's stats JSON at ``pair.pa_stats_path``.  When
        *correction_name* is given, that named section of ``correction_stats``
        is used; when *correction_name* is ``None`` the canonical
        ``final_corrected_stats`` section (the fully-corrected field) is used.
        Falls back to ``raw_corr_stats`` when no corrected stats are available.
        Pairs whose stats file is missing are retained with a warning so the
        inversion is not silently broken.

        ``self.pairs`` is updated in-place and the :attr:`image_dates` cache
        is cleared so the date list reflects the filtered pair set.

        :param threshold: Maximum NMAD in metres.  ``max(nmad_EW, nmad_NS)``
            is compared against this value.
        :type threshold: float
        :param correction_name: Key inside ``correction_stats`` to read.
            When ``None``, the ``final_corrected_stats`` section is used.
        :type correction_name: str or None
        :returns: Two lists — ``(good, bad)`` — pairs retained and pairs removed.
        :rtype: tuple[list, list]

        Example
        -------
        .. code-block:: python

            good, bad = inv.filter_pairs_by_nmad(threshold=0.5)
            print(f"Kept {len(good)}, removed {len(bad)} pairs")
        """

        good: list = []
        bad:  list = []

        for pair in self.pairs:
            try:
                stats = load_pair_stats(pair)
            except FileNotFoundError:
                logger.warning(f"[NMAD filter] no stats for {pair.pa_key} — keeping")
                good.append(pair)
                continue

            nmad_ew = nmad_ns = float("nan")

            if correction_name:
                section = stats.get("correction_stats", {}).get(correction_name, {})
            else:
                section = stats.get("final_corrected_stats", {})
            if section:
                nmad_ew = section.get("ew", {}).get("nmad", float("nan"))
                nmad_ns = section.get("ns", {}).get("nmad", float("nan"))

            if math.isnan(nmad_ew) and math.isnan(nmad_ns):
                raw = stats.get("raw_corr_stats", {})
                nmad_ew = raw.get("ew", {}).get("nmad", float("nan"))
                nmad_ns = raw.get("ns", {}).get("nmad", float("nan"))

            nmad_max = max(
                nmad_ew if not math.isnan(nmad_ew) else 0.0,
                nmad_ns if not math.isnan(nmad_ns) else 0.0,
            )
            if nmad_max <= threshold:
                good.append(pair)
            else:
                bad.append(pair)

        self.pairs = good
        self._image_dates = None

        if bad:
            logger.info(f"@GMC TIO ── NMAD filter (≤{threshold} m): "
                        f"kept {len(good)}, removed {len(bad)} pair(s)")
        else:
            logger.info(f"@GMC TIO ── NMAD filter: all {len(good)} pairs within threshold")
        return good, bad

    def prepare(
        self,
        weight_mode: str | None = None,
        slope: float = 2.0,
        min_weight: float = 0.1,
        dt_range: tuple[float, float] | None = None,
        sharpness: float = 5.0,
        w_min: float = 0.0,
        invert: bool = False,
        combine: str = "geomean",
        alpha: float = 1.0 / 3.0,
        beta: float = 1.0 / 3.0,
        gamma: float = 1.0 / 3.0,
        cc_gamma: float = 1.0,
        weights: list[float] | dict | None = None,
        sync_geodb: bool = True,
        cluster: str | None = None,
        nodes: int = 1,
        cores: int = 8,
        walltime: str = "12:00:00",
        overwrite: bool = False,
        print_summary: bool = True,
    ) -> None:
        """Run the full TIO preparation pipeline.

        Executes these steps in order:

        1. Create the directory tree via :meth:`setup_directories`.
        2. Write ``binary/File_info.rsc`` via :meth:`write_file_info_rsc`.
        3. For each pair: filter → save corrected TIFFs → convert to ENVI
           binary → create symlinks in ``LN_DATA/``.
        4. Write ``liste_image``, ``liste_image_inv``, ``liste_couple``,
           ``input_tio``, and bash launch scripts.

        After :meth:`prepare` completes, call :meth:`launch` to start the
        inversion.

        :param weight_mode: Pair weighting strategy passed to
            :meth:`write_liste_couple`.  One of ``'uniform'``, ``'temporal'``,
            ``'relative_temporal'``, ``'sigmoid'``, ``'parametric'`` or
            ``'quality'``.  ``None`` (default) computes ``'uniform'``; when
            *weights* are supplied the recorded mode comes from
            :meth:`explore_weights` (or ``'explicit'``).
        :type weight_mode: str | None
        :param slope: Power exponent for ``'parametric'`` weighting.
        :type slope: float
        :param min_weight: Minimum weight floor for ``'parametric'`` mode.
        :type min_weight: float
        :param dt_range: ``(min_days, max_days)`` normalisation bounds for
            ``'relative_temporal'`` and ``'sigmoid'``.  Derived from the
            actual pairs when ``None``.
        :type dt_range: tuple[float, float] or None
        :param sharpness: Sigmoid steepness for ``'sigmoid'`` mode.
        :type sharpness: float
        :param w_min: Global minimum weight floor (all modes except
            ``'uniform'`` and ``'parametric'``).  See
            :meth:`write_liste_couple` for details.
        :type w_min: float
        :param invert: Reverse weighting direction for ``'relative_temporal'``,
            ``'sigmoid'`` and the Δt term of ``'quality'`` so longer Δt pairs
            receive higher weight.
        :type invert: bool
        :param combine: Combination method for ``'quality'`` mode
            (``'geomean'``, ``'wmean'``, ``'product'``).
        :param alpha: ``'quality'`` NMAD-term weight (``'wmean'``).
        :param beta: ``'quality'`` CC-term weight (``'wmean'``).
        :param gamma: ``'quality'`` Δt-term weight (``'wmean'``).
        :param cc_gamma: Exponent on the CC sub-weight in ``'quality'``.
        :param weights: Optional precomputed weight vector (e.g. from
            :meth:`explore_weights`); bypasses mode math when given.
        :param sync_geodb: Persist per-pair weights to the stats JSON and sync
            ``pa_weight``/``pa_weight_mode`` to the Pairs layer (default ``True``).
        :param cluster: Cluster target for the launch script — ``None``
            (local), ``'isterre'``, or ``'gricad'``.
        :type cluster: str or None
        :param nodes: Number of compute nodes (OAR, cluster mode only).
        :type nodes: int
        :param cores: CPU cores per node (OAR, cluster mode only).
        :type cores: int
        :param walltime: Maximum job duration ``HH:MM:SS`` (OAR only).
        :type walltime: str
        :param overwrite: When ``True``, re-process pairs even if their
            binary files already exist.
        :type overwrite: bool
        :param print_summary: Print a rich summary table of per-pair export
            status once the loop completes (default ``True``).
        :type print_summary: bool

        Example
        -------
        .. code-block:: python

            inv.prepare(
                weight_mode="sigmoid",
                dt_range=(300, 400),
                sharpness=8,
                w_min=0.2,
                invert=True,
            )
            inv.launch(mode="local")
        """
        validate_cluster(cluster)

        logger.info(f"TIO ── preparing '{self.inversion_name}' "
                    f"({len(self.pairs)} pairs, {len(self.image_dates)} images)")

        self.setup_directories()
        logger.info(f"Creating directory tree → {self.inversion_dir}")

        self.write_file_info_rsc()
        logger.info(f"Writing 'File_info.rsc'")

        summary_rows: list[tuple[str, str]] = []
        pair_name = _RichText("", style="cyan")
        progress = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=_rich_console,
        )
        task = progress.add_task("", total=len(self.pairs))

        with _RichLive(
            _RichGroup(pair_name, progress),
            console=_rich_console,
            transient=False,
            refresh_per_second=10,
        ) as live:
            for pair in self.pairs:
                pair_name.plain = pair.pa_key
                live.refresh()

                stem   = self._pair_bin_stem(pair)
                ew_bin = self.inversion_dir / "binary" / f"{stem}_EW"
                ns_bin = self.inversion_dir / "binary" / f"{stem}_NS"
                skipped = (not overwrite) and ew_bin.exists() and ns_bin.exists()

                self.export_pair_to_binary(pair, overwrite=overwrite)
                self._create_symlinks(pair)

                summary_rows.append(
                    (pair.pa_key, "skipped" if skipped else "[green]exported[/green]")
                )
                progress.advance(task)
                live.refresh()

        if print_summary:
            table = Table(title="TIO prepare — pair export summary", show_lines=False)
            table.add_column("Pair", style="cyan", no_wrap=True)
            table.add_column("Status")
            for key, status in summary_rows:
                table.add_row(key, status)
            _rich_console.print(table)

        self.write_liste_image()
        self.write_liste_image_inv()
        self.write_liste_couple(
            weight_mode=weight_mode,
            slope=slope,
            min_weight=min_weight,
            dt_range=dt_range,
            sharpness=sharpness,
            w_min=w_min,
            invert=invert,
            combine=combine,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            cc_gamma=cc_gamma,
            weights=weights,
            sync_geodb=sync_geodb,
        )
        self.write_input_tio()
        self.write_launch_script(cluster=cluster, nodes=nodes, cores=cores, walltime=walltime)
        logger.info(f"TIO input files written to inverse_EW/ and inverse_NS/")

        mode = "local" if cluster is None else cluster
        logger.info(f"TIO ── ready. Call inv.launch(mode='{mode}').")

    def launch(self, direction: str = "both", mode: str | None = "local") -> None:
        """Run or submit the TIO inversion jobs.

        *mode* mirrors the ``cluster`` values used by :meth:`prepare` /
        :meth:`write_launch_script` — ``None``/``'local'`` runs
        ``invers_pixel_omp`` and ``lect_depl_cumule_lin`` directly on the
        current machine via :func:`~geomulticorr.utils.hpc_tools.run_local`;
        ``'gricad'`` or ``'isterre'`` submits the launch script written for
        that cluster via :func:`~geomulticorr.utils.hpc_tools.oarsub_submit`.

        The launch script's OAR header is baked in at
        :meth:`write_launch_script` time, not at launch time, so *mode* must
        match the cluster the scripts were last written for — otherwise a
        ``ValueError`` is raised instead of silently submitting a script with
        a missing/mismatched header.

        Call :meth:`post_process` after the inversion completes to convert
        TOT binary outputs to GeoTIFF.

        :param direction: Which inversion direction to run — ``'EW'``,
            ``'NS'``, or ``'both'`` (default).
        :type direction: str
        :param mode: Execution target — ``None``/``'local'`` runs binaries
            directly; ``'gricad'``/``'isterre'`` submits via OAR.
        :type mode: str or None

        Example
        -------
        .. code-block:: python

            inv.launch(mode="local")           # run locally
            inv.launch(mode="isterre")         # submit to the ISTerre cluster
            inv.launch(mode="gricad")          # submit to the GRICAD cluster
            inv.launch(direction="EW")         # only EW direction
        """
        validate_cluster(mode)
        mode = "local" if mode is None else mode

        targets = self._DIRECTIONS if direction == "both" else (direction.upper(),)

        if mode == "local":
            width, height = self.raster_shape
            for d in targets:
                inv_dir = self.inversion_dir / f"inverse_{d}"
                n_images = len((inv_dir / "liste_image").read_text().splitlines())
                logger.info(f"TIO ── running locally: inverse_{d} ({n_images} images)")
                run_local(
                    f"{self.invers_pixel_omp_bin} < {inv_dir}/input_tio",
                    cwd=inv_dir,
                )
                run_local(
                    f"{self.lect_depl_cumule_lin_bin}"
                    f" {width} {height - 1} {n_images} 1 1",
                    cwd=inv_dir,
                )
        else:
            if self.cluster != mode:
                raise ValueError(
                    f"launch(mode={mode!r}) requested, but the launch scripts on "
                    f"disk were written for cluster={self.cluster!r}. Call "
                    f"prepare(cluster={mode!r}) or write_launch_script(cluster={mode!r}) "
                    f"first so the OAR header matches the target cluster."
                )
            for d in targets:
                inv_dir = self.inversion_dir / f"inverse_{d}"
                script  = inv_dir / f"launch_TIO_inv_{d}.sh"
                logger.info(f"TIO ── submitting {script.name} [{mode}]")
                oarsub_submit(script, cwd=inv_dir)

    def post_process(self, direction: str = "both", overwrite: bool = False) -> None:
        """Convert TOT binary inversion outputs to GeoTIFF and compute magnitude maps.

        Must be called after :meth:`launch` has completed.  Processes
        ``TOT_YYYYMMDD`` raw ENVI Float32 binary files produced by
        ``lect_depl_cumule_lin`` in ``inverse_EW/`` and ``inverse_NS/``.

        Steps:

        1. For each direction: convert ``TOT_*`` binaries to georeferenced
           GeoTIFF via :meth:`_tot_to_geotiff` (CRS and geotransform inherited
           from ``pairs[0].pa_ew_path``).
        2. When *direction* is ``'both'``: for every date where both EW and NS
           GeoTIFFs exist, compute ``sqrt(EW² + NS²)`` via
           :meth:`_compute_magnitude` and save to ``inverse_magn/``.

        :param direction: Which directions to process — ``'EW'``, ``'NS'``,
            or ``'both'`` (default).
        :type direction: str
        :param overwrite: When ``False`` (default), skip conversion if a
            ``.tif`` already exists alongside the binary.  Set to ``True``
            to force re-conversion.
        :type overwrite: bool

        Example
        -------
        .. code-block:: python

            inv.launch(mode="local")
            inv.post_process()                   # EW + NS + magnitude
            inv.post_process(direction="EW")     # EW only
            inv.post_process(overwrite=True)     # force re-conversion
        """
        targets  = self._DIRECTIONS if direction == "both" else (direction.upper(),)
        ref_path = self.pairs[0].pa_ew_path
        tifs: dict[str, dict[str, Path]] = {d: {} for d in self._DIRECTIONS}

        for d in targets:
            inv_dir  = self.inversion_dir / f"inverse_{d}"
            tot_files = sorted(
                f for f in inv_dir.glob("TOT_*") if f.suffix == ""
            )
            if not tot_files:
                logger.warning(f"post_process: no TOT_* files found in {inv_dir.name}/")
                continue

            logger.info(f"post_process ── converting {len(tot_files)} TOT files [{d}]")
            for tot in tot_files:
                out = tot.with_suffix(".tif")
                if not overwrite and out.exists():
                    logger.info(f"  {tot.name}.tif already exists, skipping")
                    tifs[d][tot.stem.replace("TOT_", "")] = out
                    continue
                self._tot_to_geotiff(tot, ref_path)
                tifs[d][tot.stem.replace("TOT_", "")] = out
                logger.file(f"  {out.name}")

        # Magnitude only when both directions are available
        if direction == "both":
            magn_dir   = self.inversion_dir / "inverse_magn"
            magn_dir.mkdir(parents=True, exist_ok=True)
            common_dates = sorted(set(tifs["EW"]) & set(tifs["NS"]))
            logger.info(f"post_process ── computing magnitude for {len(common_dates)} dates")
            for date in common_dates:
                out = magn_dir / f"TOT_{date}.tif"
                if not overwrite and out.exists():
                    logger.info(f"  TOT_{date}.tif magnitude already exists, skipping")
                    continue
                self._compute_magnitude(tifs["EW"][date], tifs["NS"][date], out)
                logger.file(f"  magnitude: {out.name}")

        logger.info("post_process ── done.")
