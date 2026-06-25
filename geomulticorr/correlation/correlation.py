#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# correlation.py
# creation date: 2026-05-08.
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
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from geomulticorr._logging import logger
from geomulticorr.utils.hpc_tools import (
    CLUSTER_CHOICES,
    validate_cluster,
    oar_header,
    cluster_base_env,
)


class ASP:
    """Helper class to build and run Ames Stereo Pipeline (ASP) correlation commands."""

    def __init__(
        self,
        session=None,
        asp_bin_dir: str | Path | None = None,
    ) -> None:
        """Initialize ASP helper with optional binary directory.

        :param session: GMC session object (optional).
        :param asp_bin_dir: Directory containing ASP binaries. If None, search system PATH.
        :raises FileNotFoundError: If ``parallel_stereo`` binary is not found.
        """
        self.session = session
        self.asp_bin_dir = Path(asp_bin_dir) if asp_bin_dir else None
        self.find_asp_executable("parallel_stereo")

    def find_asp_executable(self, bin_name: str) -> str:
        """Return the full path to an ASP executable.

        Searches *asp_bin_dir* first (if set), then the system PATH.

        :param bin_name: Name of the ASP binary to find.
        :return: Full path to the executable.
        :raises FileNotFoundError: If the executable is not found.
        """
        search_path = str(self.asp_bin_dir) if self.asp_bin_dir else None
        exe = shutil.which(bin_name, path=search_path)
        if exe is None:
            raise FileNotFoundError(
                f"Unable to find ASP executable '{bin_name}'. "
                "Install ASP and ensure it is on your PATH, or pass asp_bin_dir= to ASP(). "
                "See https://stereopipeline.readthedocs.io/en/latest/installation.html"
            )
        return exe

    # ------------------------------------------------------------------
    # Parameter-file helpers
    # ------------------------------------------------------------------

    def build_correlation_params(
        self,
        pre_alignment: str | None = None,
        pre_normalization: bool = False,
        individual_normalization: bool = False,
        prefilter_mode: int | None = 2,
        prefilter_kernel_width: float | None = 1.4,
        corr_seed_mode: int | None = 1,
        cost_mode: int | None = 2,
        corr_algorithm: str = "asp_bm",
        corr_kernel: tuple[int, int] = (21, 21),
        n_levels: int | None = 5,
        corr_search: tuple[int, int, int, int] | None = None,
        corr_xthreshold: int = 10,
        subpixel_refinement_mode: int = 2,
        subpixel_kernel: tuple[int, int] = (21, 21),
        save_disparity_difference: bool = True,
        min_ip_points: int | None = 2,
        ip_per_tile: int | None = 200,
        nodata_value: float | None = None,
        params_file_path: str | Path | None = None,
    ) -> list[str]:
        """Build ASP correlation parameters and write them to a settings file.

        The file uses ASP's bare ``key value`` format (no ``--`` prefix).
        Note: SGM/MGM algorithms override ``corr_kernel`` to (9, 9) and ``cost_mode`` to 4.

        :param pre_alignment: Pre-alignment method (e.g., ``"homography"``).
        :param pre_normalization: Force use of entire dynamic range.
        :param individual_normalization: Normalize each image independently.
        :param prefilter_mode: Prefilter mode (0, 1, or 2). None disables.
        :param prefilter_kernel_width: Prefilter kernel width (used if mode >= 2).
        :param corr_seed_mode: Seed correlation mode.
        :param cost_mode: Cost mode for matching (1-4).
        :param corr_algorithm: Correlation algorithm (``asp_bm``, ``asp_sgm``, ``asp_mgm``, etc.).
        :param corr_kernel: Correlation kernel size as (height, width).
        :param n_levels: Number of coarse-to-fine levels. None disables.
        :param corr_search: Search range as (x_min, y_min, x_max, y_max). None uses ASP auto-detection.
            Best used with ``corr_seed_mode=0`` to skip the low-resolution disparity stage entirely.
            Example: ``(-80, -2, 20, 2)`` = large horizontal shift, small vertical shift.
        :param corr_xthreshold: Cross-correlation threshold.
        :param subpixel_refinement_mode: Subpixel refinement mode.
        :param subpixel_kernel: Subpixel kernel size as (height, width).
        :param save_disparity_difference: Save left-right disparity difference.
        :param min_ip_points: Minimum interest points. None disables.
        :param ip_per_tile: Interest points per tile. None disables.
        :param nodata_value: Nodata value for input images.
        :param params_file_path: Output path for the parameters file. If None, uses ``corr_params.txt``.
        :return: List of parameter lines written to the file.
        """
        params: list[str] = []
        params.append("# -*- mode: sh -*-\n")
        params.append("# Stereo Pipeline Correlation Parameters\n")
        params.append("# --------------------------------------\n")
        params.append("# Pre-Processing / stereo_pprc\n")
        params.append("# --------------------------------------\n")

        if pre_alignment is not None:
            params.append(f"alignment-method {pre_alignment}\n")
        if pre_normalization:
            params.append("force-use-entire-range\n")
        if individual_normalization:
            params.append("individually-normalize\n")

        if prefilter_mode is not None:
            params.append(f"prefilter-mode {prefilter_mode}\n")
            if prefilter_mode >= 2 and prefilter_kernel_width is not None:
                params.append(f"prefilter-kernel-width {prefilter_kernel_width}\n")

        if nodata_value is not None:
            params.append(f"nodata-value {nodata_value}\n")

        # SGM/MGM algorithms require a small kernel and cost-mode 4.
        if corr_algorithm in ("asp_sgm", "asp_mgm", "asp_final_mgm"):
            logger.warning(f"'{corr_algorithm}' requires corr-kernel ≤ 9×9. Overriding to 9 9.")
            corr_kernel = (9, 9)
            subpixel_kernel = (9, 9)
            cost_mode = 4

        params.append("# --------------------------------------\n")
        params.append("# Integer Correlation / stereo_corr\n")
        params.append("# --------------------------------------\n")
        params.append(f"cost-mode {cost_mode}\n")
        params.append(f"stereo-algorithm {corr_algorithm}\n")
        params.append(f"corr-seed-mode {corr_seed_mode}\n")
        params.append(f"corr-kernel {corr_kernel[0]} {corr_kernel[1]}\n")
        if n_levels is not None:
            params.append(f"corr-max-levels {n_levels}\n")
        if corr_search is not None:
            params.append(
                f"corr-search {corr_search[0]} {corr_search[1]} "
                f"{corr_search[2]} {corr_search[3]}\n"
            )
        params.append(f"xcorr-threshold {corr_xthreshold}\n")

        params.append("# --------------------------------------\n")
        params.append("# Subpixel Refinement / stereo_rfne\n")
        params.append("# --------------------------------------\n")
        params.append(f"subpixel-mode {subpixel_refinement_mode}\n")
        params.append(f"subpixel-kernel {subpixel_kernel[0]} {subpixel_kernel[1]}\n")
        if save_disparity_difference:
            params.append("save-left-right-disparity-difference\n")
        if min_ip_points is not None:
            params.append(f"min-num-ip {min_ip_points}\n")
        if ip_per_tile is not None:
            params.append(f"ip-per-tile {ip_per_tile}\n")

        self.write_correlation_parameters_file(params, params_file_path)
        return params

    def write_correlation_parameters_file(
        self,
        params: list[str],
        param_file_path: str | Path | None = None,
    ) -> None:
        """Write *params* lines to *param_file_path* (default: ``corr_params.txt``).

        :param params: List of parameter lines to write.
        :param param_file_path: Output file path. If None, uses ``corr_params.txt``.
        """
        if param_file_path is None:
            param_file_path = Path("corr_params.txt")
        param_file_path = Path(param_file_path)
        param_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(param_file_path, "w") as f:
            f.writelines(params)

    # ------------------------------------------------------------------
    # Command building (shared between run_correlation & write_bash_script)
    # ------------------------------------------------------------------

    def _build_stereo_cmd(
        self,
        left: str | Path,
        right: str | Path,
        out_prefix: str | Path,
        corr_params_file: str | Path | None = None,
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        corr_tile_size: int = 2048,
        stereo_bin: str = "parallel_stereo",
    ) -> list[str]:
        """Return the full ``parallel_stereo`` command as a list of strings.

        :param left: Path to the left (reference) image.
        :param right: Path to the right (secondary) image.
        :param out_prefix: Output path prefix (directory + base name, no extension).
        :param corr_params_file: Path to ASP settings file. If None, uses ASP defaults.
        :param processes: Number of parallel processes.
        :param threads_multiprocess: Threads per process in multiprocess mode.
        :param threads_singleprocess: Threads in single-process mode.
        :param corr_memory_limit_mb: Memory limit per process in MB.
        :param corr_tile_size: Tile size in pixels for correlation.
        :param stereo_bin: Path or name of the ``parallel_stereo`` binary.
        :return: Command as a list of strings (ready for subprocess).
        """
        cmd_stereo: list[str] = []
        cmd_stereo.append(str(stereo_bin))
        cmd_stereo.append("--correlator-mode")
        if corr_params_file is not None:
            cmd_stereo.append(f"-s {str(corr_params_file)}")
        # END if
        cmd_stereo.append(f"--processes {str(processes)}")
        cmd_stereo.append(f"--threads-multiprocess {str(threads_multiprocess)}")
        cmd_stereo.append(f"--threads-singleprocess {str(threads_singleprocess)}")
        cmd_stereo.append(f"--corr-memory-limit-mb {str(corr_memory_limit_mb)}")
        cmd_stereo.append(f"--corr-tile-size {str(corr_tile_size)}")
        cmd_stereo.append(str(left))
        cmd_stereo.append(str(right))
        cmd_stereo.append(str(out_prefix))

        return cmd_stereo

    def _build_correval_cmd(
        self,
        out_prefix: str | Path,
        corr_algorithm: str = "asp_bm",
        corr_kernel: tuple[int, int] = (21, 21),
        metric: str = "ncc",
        correval_bin: str = "corr_eval",
    ) -> list[str]:
        """Return the ``corr_eval`` command as a list of strings.

        The prefilter mode is derived from *corr_algorithm*:

        - ``asp_bm`` → ``--prefilter-mode 2``
        - ``asp_sgm`` / ``asp_mgm`` / ``asp_final_mgm`` → ``--prefilter-mode 0``

        :param out_prefix: Output prefix path for the correlation output.
        :param corr_algorithm: Correlation algorithm name (controls prefilter mode).
        :param corr_kernel: Correlation kernel size as (height, width) tuple.
        :param metric: Similarity metric to compute (e.g., ``"ncc"``).
        :param correval_bin: Path or name of the ``corr_eval`` binary.
        :return: Command as a list of strings (ready for subprocess).
        """
        if corr_algorithm in ("asp_sgm", "asp_mgm", "asp_final_mgm"):
            prefilter_mode = 0
        else:
            prefilter_mode = 2

        p = Path(out_prefix)
        left = str(p.parent / f"{p.name}-L.tif")
        right = str(p.parent / f"{p.name}-R.tif")
        disp = str(p.parent / f"{p.name}-F.tif")
        output = str(p.parent / f"{p.name}-F")

        cmd_correval: list[str] = []
        cmd_correval.append(str(correval_bin))
        cmd_correval.append(f"--prefilter-mode {str(prefilter_mode)}")
        cmd_correval.append(
            f"--kernel-size {str(corr_kernel[0])} {str(corr_kernel[1])}"
        )
        cmd_correval.append(f"--metric {metric}")
        cmd_correval.append(left)
        cmd_correval.append(right)
        cmd_correval.append(disp)
        cmd_correval.append(output)

        return cmd_correval

    # ------------------------------------------------------------------
    # Correlation execution
    # ------------------------------------------------------------------

    def run_correlation(
        self,
        left: str | Path,
        right: str | Path,
        out_prefix: str | Path,
        corr_params_file: str | Path | None = None,
        corr_tile_size: int = 2048,
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        dry_run: bool = False,
        clean: bool = False,
        correval: bool = True,
        corr_algorithm: str = "asp_bm",
        corr_kernel: tuple[int, int] = (21, 21),
        metric: str = "ncc",
    ) -> int:
        """Run ``parallel_stereo`` on a pair of images, then optionally run ``corr_eval``.

        :param left: Path to the left (reference) image.
        :param right: Path to the right (secondary) image.
        :param out_prefix: Output path prefix (directory + base name, no extension).
        :param corr_params_file: Path to ASP settings file. If None, uses ASP defaults.
        :param corr_tile_size: Tile size in pixels for correlation.
        :param processes: Number of parallel processes.
        :param threads_multiprocess: Threads per process in multiprocess mode.
        :param threads_singleprocess: Threads in single-process mode.
        :param corr_memory_limit_mb: Memory limit per process in MB.
        :param dry_run: Print commands without executing them.
        :param clean: Delete intermediate files after successful correlation.
        :param correval: Run ``corr_eval`` to compute metrics after correlation.
        :param corr_algorithm: Algorithm used (controls ``corr_eval`` prefilter mode).
        :param corr_kernel: Kernel size used (forwarded to ``corr_eval --kernel-size``).
        :param metric: Metric passed to ``corr_eval --metric`` (default: ``ncc``).
        :return: ``parallel_stereo`` return code (0 = success).
        """
        stereo_bin = self.find_asp_executable("parallel_stereo")
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

        call = self._build_stereo_cmd(
            left,
            right,
            out_prefix,
            corr_params_file,
            processes,
            threads_multiprocess,
            threads_singleprocess,
            corr_memory_limit_mb,
            corr_tile_size,
            stereo_bin=stereo_bin,
        )

        if dry_run:
            logger.info("DRY RUN (parallel_stereo): " + " ".join(call))
            if correval:
                correval_bin = self.find_asp_executable("corr_eval")
                ce_call = self._build_correval_cmd(
                    out_prefix,
                    corr_algorithm,
                    corr_kernel,
                    metric,
                    correval_bin=correval_bin,
                )
                logger.info("DRY RUN (corr_eval): " + " ".join(ce_call))
            return 0

        result = subprocess.run(
            call,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"DEBUG: returncode is {result.returncode}")
            logger.error(f"ASP parallel_stereo error:\n{result.stderr}")
            return result.returncode

        if correval:
            correval_bin = self.find_asp_executable("corr_eval")
            ce_call = self._build_correval_cmd(
                out_prefix,
                corr_algorithm,
                corr_kernel,
                metric,
                correval_bin=correval_bin,
            )
            ce_result = subprocess.run(
                ce_call,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if ce_result.returncode != 0:
                logger.error(f"ASP corr_eval error:\n{ce_result.stderr}")

        if clean:
            out_dir = Path(out_prefix).parent
            for f in sorted(out_dir.iterdir()):
                if f.is_file() and not any(
                    f.name.endswith(s) for s in self._KEEP_SUFFIXES
                ):
                    f.unlink()

        return result.returncode

    def run_for_pair(
        self,
        pair,
        corr_tile_size: int = 2048,
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        dry_run: bool = False,
        clean: bool = False,
        correval: bool = True,
        metric: str = "ncc",
        **corr_params_kwargs,
    ) -> int:
        """Run correlation for a GMC ``Pair`` object, then optionally run ``corr_eval``.

        Expects the pair to have been clipped first (``pair.clip()``).
        Writes a settings file inside the pair directory, then calls
        ``parallel_stereo`` and optionally ``corr_eval``.

        :param pair: A ``geomulticorr.pair.Pair`` instance with ``pa_status`` ``'clipped'`` or ``'complete'``.
        :param corr_tile_size: Tile size in pixels for correlation.
        :param processes: Number of parallel processes.
        :param threads_multiprocess: Threads per process in multiprocess mode.
        :param threads_singleprocess: Threads in single-process mode.
        :param corr_memory_limit_mb: Memory limit per process in MB.
        :param dry_run: Print commands without executing them.
        :param clean: Delete intermediate files after successful correlation.
        :param correval: Run ``corr_eval`` after successful correlation.
        :param metric: Metric for ``corr_eval`` (default: ``ncc``).
        :param corr_params_kwargs: Additional arguments forwarded to :meth:`build_correlation_params`.
        :return: ``parallel_stereo`` return code.
        """
        if pair.get_status() == "complete":
            logger.info(f"Pair '{pair.pa_key}' is already complete. Skipping.")
            return 0

        left = pair.pa_left_link_path
        right = pair.pa_right_link_path
        out_prefix = pair.pa_path / pair.pa_key
        params_file = pair.pa_path / "corr_params.txt"

        self.build_correlation_params(
            params_file_path=params_file,
            **corr_params_kwargs,
        )

        # Extract the algorithm/kernel used so corr_eval gets the right settings.
        corr_algorithm = corr_params_kwargs.get("corr_algorithm", "asp_bm")
        corr_kernel = corr_params_kwargs.get("corr_kernel", (21, 21))

        return self.run_correlation(
            left=left,
            right=right,
            out_prefix=out_prefix,
            corr_params_file=params_file,
            corr_tile_size=corr_tile_size,
            processes=processes,
            threads_multiprocess=threads_multiprocess,
            threads_singleprocess=threads_singleprocess,
            corr_memory_limit_mb=corr_memory_limit_mb,
            dry_run=dry_run,
            clean=clean,
            correval=correval,
            corr_algorithm=corr_algorithm,
            corr_kernel=corr_kernel,
            metric=metric,
        )

    # ------------------------------------------------------------------
    # Bash script generation
    # ------------------------------------------------------------------

    def write_bash_script(
        self,
        pair,
        corr_params_file: str | Path | None = None,
        cluster: str | None = None,
        nodes: int = 1,
        cores: int = 8,
        walltime: str = "12:00:00",
        besteffort: bool = False,
        conda_env: str = "gmc_env",
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        corr_tile_size: int = 2048,
        correval: bool = True,
        corr_algorithm: str = "asp_bm",
        corr_kernel: tuple[int, int] = (21, 21),
        metric: str = "ncc",
    ) -> Path:
        """Write a bash script to run ``parallel_stereo`` for *pair*.

        The script is written to ``pair.pa_path / "{pair.pa_key}_CorrelationJob.sh"`` and
        made executable. When *cluster* is set, OAR scheduler headers are
        prepended and the appropriate environment setup is included.

        :param pair: A ``geomulticorr.pair.Pair`` instance.
        :param corr_params_file: Path to ASP settings file (from :meth:`build_correlation_params`). If None, uses default.
        :param cluster: Cluster target: ``None``, ``"local"``, ``"gricad"``, or ``"isterre"``.
        :param nodes: Number of OAR nodes (cluster mode only).
        :param cores: Number of cores per node (cluster mode only).
        :param walltime: OAR walltime in ``HH:MM:SS`` format (cluster mode only).
        :param besteffort: Enable OAR besteffort mode (cluster mode only).
        :param conda_env: Conda environment name to activate in the script.
        :param processes: Number of parallel processes.
        :param threads_multiprocess: Threads per process in multiprocess mode.
        :param threads_singleprocess: Threads in single-process mode.
        :param corr_memory_limit_mb: Memory limit per process in MB.
        :param corr_tile_size: Tile size in pixels for correlation.
        :param correval: Include ``corr_eval`` command in the script.
        :param corr_algorithm: Correlation algorithm (controls ``corr_eval`` prefilter mode).
        :param corr_kernel: Correlation kernel size as (height, width) tuple.
        :param metric: Metric for ``corr_eval`` (default: ``ncc``).
        :return: Path to the written bash script.
        """
        validate_cluster(cluster)

        left = pair.pa_left_link_path
        right = pair.pa_right_link_path
        out_prefix = pair.pa_path / pair.pa_key

        if corr_params_file is None:
            corr_params_file = pair.pa_path / f"{pair.pa_key}_CorrParameters.txt"

        # For cluster scripts, the binary is loaded via module load / PATH.
        # For local, use the resolved full path so the script is self-contained.
        if cluster in (None, "local"):
            stereo_bin = self.find_asp_executable("parallel_stereo")
        else:
            stereo_bin = "parallel_stereo"

        cmd_parts = self._build_stereo_cmd(
            left,
            right,
            out_prefix,
            corr_params_file,
            processes,
            threads_multiprocess,
            threads_singleprocess,
            corr_memory_limit_mb,
            corr_tile_size,
            stereo_bin=stereo_bin,
        )
        # Format as a readable multiline bash command
        cmd_str = " \\\n    ".join(cmd_parts)

        job_name = f"ImgCorr_{pair.pa_key}"
        lines: list[str] = ["#!/bin/bash\n"]

        lines += oar_header(
            job_name, nodes, cores, walltime, cluster,
            besteffort=besteffort,
            output_dir=pair.pa_path,
        )
        if cluster in ("gricad", "isterre"):
            lines += ["\n"] + cluster_base_env(cluster)
            if cluster == "gricad":
                lines += [f"conda activate {conda_env}\n"]
            else:  # isterre
                lines += [
                    "module load stereopipeline/3.6.0\n",
                    "\n",
                    f"micromamba activate {conda_env}\n",
                ]
            lines.append("\n")

        lines.append(f"{cmd_str}\n")

        if correval:
            if cluster in (None, "local"):
                correval_bin = self.find_asp_executable("corr_eval")
            else:
                correval_bin = "corr_eval"
            ce_parts = self._build_correval_cmd(
                out_prefix=out_prefix,
                corr_algorithm=corr_algorithm,
                corr_kernel=corr_kernel,
                metric=metric,
                correval_bin=correval_bin,
            )
            ce_str = " \\\n    ".join(ce_parts)
            lines.append(f"\n{ce_str}\n")

        bash_path = pair.pa_path / f"{pair.pa_key}_CorrelationJob.sh"
        bash_path.parent.mkdir(parents=True, exist_ok=True)
        bash_path.write_text("".join(lines))
        bash_path.chmod(bash_path.stat().st_mode | 0o755)
        return bash_path

    # ------------------------------------------------------------------
    # Output cleanup
    # ------------------------------------------------------------------

    _KEEP_SUFFIXES = (
        "-F.tif",
        "-L-R-disp-diff.tif",
        "-L.tif",
        "-R.tif",
        "-RD.tif",
        "-F-ncc.tif",
        "_CorrelationJob.sh",
        "_CorrParameters.txt",
    )

    def clean_pair_directory(self, pair, dry_run: bool = False) -> list[Path]:
        """Delete intermediate ASP files from *pair*'s output directory.

        Keeps only files whose names end with one of:
        ``-F.tif``, ``-L-R-disp-diff.tif``, ``-L.tif``, ``-R.tif``, ``-RD.tif``, ``-F-ncc.tif``.

        :param pair: A ``geomulticorr.pair.Pair`` instance.
        :param dry_run: List files that would be deleted without removing them.
        :return: List of deleted (or would-be-deleted) paths.
        """
        asp_dir = pair.pa_path
        if not asp_dir.exists():
            logger.warning(f"Pair directory does not exist: {asp_dir}")
            return []

        removed: list[Path] = []
        for f in sorted(asp_dir.iterdir()):
            if not f.is_file() or f.is_symlink():
                continue
            if any(f.name.endswith(suffix) for suffix in self._KEEP_SUFFIXES):
                continue
            if dry_run:
                logger.info(f"DRY RUN — would delete: {f.name}")
            else:
                f.unlink()
            removed.append(f)

        action = "Would delete" if dry_run else "Deleted"
        logger.info(f"{action} {len(removed)} file(s) from '{asp_dir}'.")
        return removed
