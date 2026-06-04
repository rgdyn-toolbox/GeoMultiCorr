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
>>> pipeline = (CCFilter(cc_threshold=0.5)
...           + OutlierFilter(threshold=(-15, 15))
...           + MedianCentering()
...           + RampCorrection())
>>> inv = TIOInversion(session, pairs, name="rock_glacier_2021_2024",
...                    correction_pipeline=pipeline,
...                    correction_kwargs={"stable_mask": "stable_areas.geojson"})
>>> inv.prepare()
>>> inv.launch()
"""
from __future__ import annotations

import math
import stat
import subprocess
from datetime import datetime
from pathlib import Path

from geomulticorr._logging import logger
from geomulticorr.utils.hpc_tools import (
    oar_header,
    cluster_base_env,
    oarsub_submit,
    run_local,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _year_dec(d: datetime) -> float:
    """Day-precision decimal year (e.g. 2021.671)."""
    year_start = datetime(d.year, 1, 1)
    year_end   = datetime(d.year + 1, 1, 1)
    return d.year + (d - year_start).days / (year_end - year_start).days


def _has_cc_filter(pipeline) -> bool:
    """Return True if *pipeline* contains a CCFilter step."""
    from geomulticorr.corrections import CCFilter, CorrectionPipeline
    if isinstance(pipeline, CorrectionPipeline):
        return any(isinstance(s, CCFilter) for s in pipeline.steps)
    return isinstance(pipeline, CCFilter)


def _build_input_tio_text(
        liste_image_inv_fn: str = "liste_image_inv",
        liste_couple_fn: str = "liste_couple"
        ) -> str:
    """Standard invers_pixel parameter block."""
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


def _build_hpc_script(
    inv_dir: Path,
    width: int,
    height: int,
    n_images: int,
    tio_binaries_dir: Path,
    nodes: int = 1,
    cores: int = 8,
    walltime: str = "12:00:00",
    cluster: str = "isterre",
) -> str:
    """OAR bash launch script for invers_pixel_omp.

    Uses :func:`~geomulticorr.utils.hpc_tools.oar_header` and
    :func:`~geomulticorr.utils.hpc_tools.cluster_base_env` for the scheduler
    and environment sections; TIO-specific module loads and binary paths are
    added here.
    """
    job_name = f"TIO_inv_{inv_dir.name}"
    lines = ["#!/bin/bash\n"]
    lines += [f"{l}" for l in oar_header(job_name, nodes, cores, walltime, cluster)]
    if lines[-1] != "\n":
        lines.append("\n")
    lines += cluster_base_env(cluster)
    lines += [
        "module purge\n",
        "module load intel-devel/18.0.1\n",
        "\n",
        f"{tio_binaries_dir}/invers_pixel_omp <{inv_dir}/input_tio\n",
        "nbd=$(wc -l liste_image)\n",
        "nbdat=${nbd:0:3}\n",
        'echo "$nbd"\n',
        'echo "$nbdat"\n',
        "\n",
        f"{tio_binaries_dir}/lect_depl_cumule_lin {width} {height - 1} {n_images} 1 1\n",
    ]
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GDAL helpers
# ─────────────────────────────────────────────────────────────────────────────

def tiff2bin(image_path: str | Path, outbin_path: str | Path) -> None:
    """Convert a GeoTIFF to ENVI Float32 binary (TIO mandatory input format).

    Wraps ``gdal_translate -of envi -ot Float32 --config GDAL_PAM_ENABLED NO``.
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
        All pairs must share the same raster grid (resolution, extent).
    name:
        Sub-folder name created inside ``session.path_tio_inversion_outs``.
    correction_pipeline:
        Optional composable CorrectionPipeline applied to each pair's EW/NS
        rasters before binary export.  When None, raw displacements are used.
    correction_kwargs:
        Extra keyword arguments forwarded to ``pipeline.apply()``
        (e.g. ``stable_mask``, ``dem``).
        ``cc`` is auto-injected from ``pair.pa_cc_raw_path`` when CCFilter is
        detected in the pipeline.
    """

    _DIRECTIONS = ("EW", "NS")

    def __init__(
        self,
        session,
        pairs: list,
        name: str,
        correction_pipeline=None,
        correction_kwargs: dict | None = None,
        tio_binaries_dir: str | Path = "/home/cusicand/TIO",
    ) -> None:
        self.session             = session
        self.pairs               = list(pairs)
        self.name                = name
        self.correction_pipeline = correction_pipeline
        self.correction_kwargs   = dict(correction_kwargs or {})
        self.tio_binaries_dir    = Path(tio_binaries_dir)
        self.inversion_dir       = Path(session.path_tio_inversion_outs) / name
        self._image_dates: list[str] | None = None
        self._raster_width:  int | None = None
        self._raster_height: int | None = None

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TIOInversion(name='{self.name}', "
            f"pairs={len(self.pairs)}, "
            f"images={len(self.image_dates)}, "
            f"correction={'yes' if self.correction_pipeline else 'none'})"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def image_dates(self) -> list[str]:
        """Sorted unique image dates (YYYYmmdd) across all pairs."""
        if self._image_dates is None:
            dates: set[str] = set()
            for p in self.pairs:
                dates.add(p.pa_left.th_date.replace("-", ""))
                dates.add(p.pa_right.th_date.replace("-", ""))
            self._image_dates = sorted(dates)
        return self._image_dates

    @property
    def raster_shape(self) -> tuple[int, int]:
        """``(width_cols, height_rows)`` in pixels, read from pairs[0].pa_ew_path."""
        if self._raster_width is None:
            import geoutils as gu
            r = gu.Raster(str(self.pairs[0].pa_ew_path))
            self._raster_width  = r.width
            self._raster_height = r.height
        return self._raster_width, self._raster_height

    # ── Private helpers ───────────────────────────────────────────────────────

    def _pair_bin_stem(self, pair) -> str:
        """Return ``'{yyyymmdd1}_{yyyymmdd2}'`` used as binary file stem."""
        d1 = pair.pa_left.th_date.replace("-", "")
        d2 = pair.pa_right.th_date.replace("-", "")
        return f"{d1}_{d2}"

    def _apply_corrections(self, pair) -> tuple:
        """Load pa_ew_path / pa_ns_path, apply pipeline, return (xDisp, yDisp)."""
        import geoutils as gu
        xDisp = gu.Raster(str(pair.pa_ew_path))
        yDisp = gu.Raster(str(pair.pa_ns_path))
        if self.correction_pipeline is None:
            return xDisp, yDisp
        kwargs = dict(self.correction_kwargs)
        if _has_cc_filter(self.correction_pipeline) and "cc" not in kwargs:
            kwargs["cc"] = gu.Raster(str(pair.pa_cc_raw_path))
        return self.correction_pipeline.apply(xDisp, yDisp, **kwargs)

    def _save_corrected(self, pair, xDisp, yDisp) -> tuple[Path, Path]:
        """Save corrected gu.Raster objects to corrected/ subfolder."""
        stem    = self._pair_bin_stem(pair)
        ew_path = self.inversion_dir / "corrected" / f"{stem}_EW.tif"
        ns_path = self.inversion_dir / "corrected" / f"{stem}_NS.tif"
        xDisp.save(str(ew_path))
        yDisp.save(str(ns_path))
        return ew_path, ns_path

    def _make_executable(self, path: Path) -> None:
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # ── Directory setup ───────────────────────────────────────────────────────

    def setup_directories(self) -> None:
        """Create corrected/, binary/, inverse_EW/LN_DATA/, inverse_NS/LN_DATA/."""
        for d in [
            self.inversion_dir / "corrected",
            self.inversion_dir / "binary",
            self.inversion_dir / "inverse_EW" / "LN_DATA",
            self.inversion_dir / "inverse_NS" / "LN_DATA",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Per-pair data export ──────────────────────────────────────────────────

    def export_pair_to_binary(self, pair) -> tuple[Path, Path]:
        """Apply corrections, save corrected TIFFs, and convert to ENVI binary.

        Returns
        -------
        (ew_bin_path, ns_bin_path)
        """
        xDisp, yDisp    = self._apply_corrections(pair)
        ew_tiff, ns_tiff = self._save_corrected(pair, xDisp, yDisp)
        stem   = self._pair_bin_stem(pair)
        ew_bin = self.inversion_dir / "binary" / f"{stem}_EW"
        ns_bin = self.inversion_dir / "binary" / f"{stem}_NS"
        tiff2bin(ew_tiff, ew_bin)
        tiff2bin(ns_tiff, ns_bin)
        self._make_executable(ew_bin)
        self._make_executable(ns_bin)
        return ew_bin, ns_bin

    def _create_symlinks(self, pair) -> None:
        """Create .r4 and .r4.rsc symlinks in inverse_EW/LN_DATA/ and inverse_NS/LN_DATA/.

        Naming convention:
          binary/{yyyymmdd1}_{yyyymmdd2}_EW  →  inverse_EW/LN_DATA/{yyyymmdd1}-{yyyymmdd2}.r4
          binary/File_info.rsc              →  inverse_EW/LN_DATA/{yyyymmdd1}-{yyyymmdd2}.r4.rsc
        (same for NS)
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

    # ── TIO text files ────────────────────────────────────────────────────────

    def write_file_info_rsc(self) -> Path:
        """Write binary/File_info.rsc with pixel geometry derived from the first pair.

        Uses the ocslide convention: ``WIDTH`` = number of columns,
        ``FILE_LENGTH`` = number of rows.
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
        """Write liste_image (one YYYYmmdd per line) to inverse_EW/ and inverse_NS/."""
        content = "\n".join(self.image_dates) + "\n"
        for direction in self._DIRECTIONS:
            p = self.inversion_dir / f"inverse_{direction}" / "liste_image"
            p.write_text(content)
            self._make_executable(p)

    def write_liste_image_inv(self) -> None:
        """Write liste_image_inv with columns: YYYYmmdd, decimal_year, delta_from_first, 0."""
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

    def write_liste_couple(
        self,
        weight_mode: str = "uniform",
        slope: float = 2.0,
        min_weight: float = 0.1,
    ) -> None:
        """Write liste_couple (date1, date2, weight) for each actual pair.

        Only pairs present in ``self.pairs`` are listed — not all permutations
        of image dates.

        Parameters
        ----------
        weight_mode:
            ``'uniform'``    → weight = 1.0 for all pairs.
            ``'temporal'``   → 1 / (1 + Δt²)² (shorter intervals get higher weight).
            ``'parametric'`` → w = min_weight + (1 − min_weight) × t_norm^slope,
                               where t_norm is the pair's temporal midpoint
                               normalised to [0, 1] over the full survey period.
                               Older pairs (earlier midpoints) receive lower weight.
        slope:
            Power exponent for ``'parametric'`` mode (default 2.0).
        min_weight:
            Minimum weight floor for ``'parametric'`` mode (default 0.1).
        """
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

        lines = []
        for i, pair in enumerate(self.pairs):
            d1 = pair.pa_left.th_date.replace("-", "")
            d2 = pair.pa_right.th_date.replace("-", "")
            if weight_mode == "temporal":
                y1 = _year_dec(datetime.strptime(d1, "%Y%m%d"))
                y2 = _year_dec(datetime.strptime(d2, "%Y%m%d"))
                w  = 1.0 / math.pow(1.0 + math.pow(y2 - y1, 2), 2)
            elif weight_mode == "parametric":
                t_norm = (midpoints[i] - t_min) / t_range
                w = min_weight + (1.0 - min_weight) * math.pow(t_norm, slope)
            else:
                w = 1.0
            lines.append(f"{d1} {d2} {w:.6f}\n")
        content = "".join(lines)
        for direction in self._DIRECTIONS:
            p = self.inversion_dir / f"inverse_{direction}" / "liste_couple"
            p.write_text(content)
            self._make_executable(p)

    def write_input_tio(self) -> None:
        """Write the standard invers_pixel input_tio parameter file."""
        content = _build_input_tio_text()
        for direction in self._DIRECTIONS:
            p = self.inversion_dir / f"inverse_{direction}" / "input_tio"
            p.write_text(content)
            self._make_executable(p)

    def write_launch_script(
        self,
        nodes: int = 1,
        cores: int = 8,
        walltime: str = "12:00:00",
    ) -> None:
        """Write OAR HPC bash launch scripts for inverse_EW/ and inverse_NS/.

        Parameters
        ----------
        nodes, cores, walltime:
            OAR resource allocation parameters.
        """
        width, height = self.raster_shape
        n_images = len(self.image_dates)
        for direction in self._DIRECTIONS:
            inv_dir = self.inversion_dir / f"inverse_{direction}"
            content = _build_hpc_script(
                inv_dir, width, height, n_images,
                tio_binaries_dir=self.tio_binaries_dir,
                nodes=nodes, cores=cores, walltime=walltime,
            )
            p = inv_dir / f"launch_TIO_inv_{direction}.sh"
            p.write_text(content)
            self._make_executable(p)

    # ── Orchestration ─────────────────────────────────────────────────────────

    def filter_pairs_by_nmad(
        self,
        threshold: float = 0.7,
        correction_name: str | None = None,
    ) -> tuple[list, list]:
        """Remove pairs whose NMAD exceeds *threshold* from ``self.pairs``.

        Reads each pair's stats JSON (``pair.pa_stats_path``).  For pairs with
        corrected stats, the section keyed by *correction_name* is used; when
        *correction_name* is None, the last recorded corrected section is used.
        Falls back to ``raw_corr_stats`` if no corrected stats are available.
        Pairs whose stats file is missing are kept with a warning.

        Parameters
        ----------
        threshold:
            Maximum NMAD in meters.  ``max(nmad_EW, nmad_NS)`` is compared
            against this value (default 0.7 m).
        correction_name:
            Key inside ``corrected_stats`` to read.  When None, the last entry
            is used.

        Returns
        -------
        good, bad : list, list
            Pairs retained and pairs removed.  ``self.pairs`` is updated in place
            and the ``image_dates`` cache is cleared.
        """
        from geomulticorr.stats import load_pair_stats

        good: list = []
        bad:  list = []

        for pair in self.pairs:
            try:
                stats = load_pair_stats(pair)
            except FileNotFoundError:
                logger.warning(f"[NMAD filter] no stats for {pair.pa_key} — keeping")
                good.append(pair)
                continue

            corrected = stats.get("corrected_stats", {})
            nmad_ew = nmad_ns = float("nan")

            if corrected:
                if correction_name and correction_name in corrected:
                    key = correction_name
                else:
                    key = list(corrected)[-1]
                section = corrected[key]
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
        weight_mode: str = "uniform",
        slope: float = 2.0,
        min_weight: float = 0.1,
        nodes: int = 1,
        cores: int = 8,
        walltime: str = "12:00:00",
    ) -> None:
        """Full TIO preparation pipeline.

        Steps
        -----
        1. Create folder tree (corrected/, binary/, inverse_EW/LN_DATA/, inverse_NS/LN_DATA/)
        2. Write binary/File_info.rsc
        3. Per pair: apply corrections → save corrected TIFFs → ENVI binary → symlinks
        4. Write liste_image, liste_image_inv, liste_couple, input_tio, launch scripts

        Parameters
        ----------
        weight_mode:
            Passed to :meth:`write_liste_couple` — ``'uniform'``, ``'temporal'``,
            or ``'parametric'``.
        slope:
            Exponent for ``weight_mode='parametric'`` (default 2.0).
        min_weight:
            Minimum weight floor for ``weight_mode='parametric'`` (default 0.1).
        nodes, cores, walltime:
            OAR resource parameters for the HPC launch script.
        """
        logger.info(f"TIO ── preparing '{self.name}' "
                    f"({len(self.pairs)} pairs, {len(self.image_dates)} images)")

        self.setup_directories()
        logger.info(f"Creating directory tree → {self.inversion_dir}")

        self.write_file_info_rsc()
        logger.info(f"Writing 'File_info.rsc'")

        for i, pair in enumerate(self.pairs, 1):
            logger.info(f"TIO making pairs ({i}/{len(self.pairs)}) "
                        f"{pair.pa_left.th_date} → {pair.pa_right.th_date}")
            self.export_pair_to_binary(pair)
            self._create_symlinks(pair)

        self.write_liste_image()
        self.write_liste_image_inv()
        self.write_liste_couple(weight_mode=weight_mode, slope=slope, min_weight=min_weight)
        self.write_input_tio()
        self.write_launch_script(nodes=nodes, cores=cores, walltime=walltime)
        logger.info(f"TIO input files written to inverse_EW/ and inverse_NS/")
        logger.info(f"TIO ── ready. Call inv.launch() to submit to HPC.")

    def launch(self, direction: str = "both", mode: str = "hpc") -> None:
        """Run or submit TIO inversion jobs.

        Parameters
        ----------
        direction:
            ``'EW'``, ``'NS'``, or ``'both'`` (default).
        mode:
            ``'hpc'`` (default) — submit via ``oarsub -S`` using the written
            launch script.
            ``'local'`` — run ``invers_pixel_omp`` and ``lect_depl_cumule_lin``
            directly on the current machine (no scheduler).
        """
        targets = self._DIRECTIONS if direction == "both" else (direction.upper(),)

        if mode == "local":
            width, height = self.raster_shape
            for d in targets:
                inv_dir = self.inversion_dir / f"inverse_{d}"
                n_images = len((inv_dir / "liste_image").read_text().splitlines())
                logger.info(f"TIO ── running locally: inverse_{d} ({n_images} images)")
                run_local(
                    f"{self.tio_binaries_dir}/invers_pixel_omp < {inv_dir}/input_tio",
                    cwd=inv_dir,
                )
                run_local(
                    f"{self.tio_binaries_dir}/lect_depl_cumule_lin"
                    f" {width} {height - 1} {n_images} 1 1",
                    cwd=inv_dir,
                )
        else:
            for d in targets:
                inv_dir = self.inversion_dir / f"inverse_{d}"
                script  = inv_dir / f"launch_TIO_inv_{d}.sh"
                logger.info(f"TIO ── submitting {script.name}")
                oarsub_submit(script, cwd=inv_dir)
