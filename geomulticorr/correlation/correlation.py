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

# Supported OAR cluster profiles.
_CLUSTER_CHOICES = (None, "local", "gricad", "isterre")


class ASP:
    """Helper class to build and run Ames Stereo Pipeline (ASP) correlation commands."""

    def __init__(
        self,
        session=None,
        asp_bin_dir: str | Path = None,
    ):
        self.session = session
        self.asp_bin_dir = Path(asp_bin_dir) if asp_bin_dir else None
        # Validate that parallel_stereo is reachable on construction.
        self.find_asp_executable("parallel_stereo")

    def find_asp_executable(self, bin_name: str) -> str:
        """Return the full path to an ASP executable.

        Searches *asp_bin_dir* first (if set), then the system PATH.

        Raises:
            FileNotFoundError: If the executable is not found.
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
        params_file_path: str | Path = None,
    ) -> list[str]:
        """Build ASP correlation parameters and write them to a settings file.

        The file uses ASP's bare ``key value`` format (no ``--`` prefix).

        Returns:
            list[str]: Lines written to the file.
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
            print(
                f"'{corr_algorithm}' requires corr-kernel ≤ 9×9. Overriding to 9 9."
            )
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
        param_file_path: str | Path = None,
    ) -> None:
        """Write *params* lines to *param_file_path* (default: ``corr_params.txt``)."""
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
        corr_params_file: str | Path = None,
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        corr_tile_size: int = 2048,
        stereo_bin: str = "parallel_stereo",
    ) -> list[str]:
        """Return the full ``parallel_stereo`` command as a list of strings."""
        cmd = [
            str(stereo_bin),
            str(left),
            str(right),
            str(out_prefix),
            "--correlator-mode",
            "--processes", str(processes),
            "--threads-multiprocess", str(threads_multiprocess),
            "--threads-singleprocess", str(threads_singleprocess),
            "--corr-memory-limit-mb", str(corr_memory_limit_mb),
            "--corr-tile-size", str(corr_tile_size),
        ]
        if corr_params_file is not None:
            cmd.extend(["-s", str(corr_params_file)])
        return cmd

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
        """
        if corr_algorithm in ("asp_sgm", "asp_mgm", "asp_final_mgm"):
            prefilter_mode = 0
        else:
            prefilter_mode = 2

        p = Path(out_prefix)
        left  = str(p.parent / f"{p.name}-L.tif")
        right = str(p.parent / f"{p.name}-R.tif")
        disp  = str(p.parent / f"{p.name}-F.tif")

        return [
            str(correval_bin),
            "--prefilter-mode", str(prefilter_mode),
            "--kernel-size", str(corr_kernel[0]), str(corr_kernel[1]),
            "--metric", metric,
            left, right, disp, str(p),
        ]

    # ------------------------------------------------------------------
    # Correlation execution
    # ------------------------------------------------------------------

    def run_correlation(
        self,
        left: str | Path,
        right: str | Path,
        out_prefix: str | Path,
        corr_params_file: str | Path = None,
        corr_tile_size: int = 2048,
        processes: int = 1,
        threads_multiprocess: int = 8,
        threads_singleprocess: int = 8,
        corr_memory_limit_mb: int = 8000,
        dry_run: bool = False,
        clean: bool = False,
        correval: bool = False,
        corr_algorithm: str = "asp_bm",
        corr_kernel: tuple[int, int] = (21, 21),
        metric: str = "ncc",
    ) -> int:
        """Run ``parallel_stereo`` on a pair of images, then optionally run ``corr_eval``.

        Args:
            left: Path to the left (reference) image.
            right: Path to the right (secondary) image.
            out_prefix: Output path prefix (directory + base name, no extension).
            corr_params_file: Path to an ASP settings file.  If *None*, ASP
                uses its built-in defaults.
            dry_run: Print commands without executing them.
            clean: Delete intermediate files after a successful correlation.
            correval: Run ``corr_eval`` to compute NCC metrics after correlation.
            corr_algorithm: Algorithm used (controls ``corr_eval`` prefilter mode).
            corr_kernel: Kernel size used (forwarded to ``corr_eval --kernel-size``).
            metric: Metric passed to ``corr_eval --metric`` (default: ``ncc``).

        Returns:
            int: ``parallel_stereo`` return code (0 = success).
        """
        stereo_bin = self.find_asp_executable("parallel_stereo")
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

        call = self._build_stereo_cmd(
            left, right, out_prefix, corr_params_file,
            processes, threads_multiprocess, threads_singleprocess,
            corr_memory_limit_mb, corr_tile_size,
            stereo_bin=stereo_bin,
        )

        if dry_run:
            print("DRY RUN (parallel_stereo):", " ".join(call))
            if correval:
                correval_bin = self.find_asp_executable("corr_eval")
                ce_call = self._build_correval_cmd(
                    out_prefix, corr_algorithm, corr_kernel, metric,
                    correval_bin=correval_bin,
                )
                print("DRY RUN (corr_eval):      ", " ".join(ce_call))
            return 0

        result = subprocess.run(
            call,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print("ASP parallel_stereo error:\n", result.stderr)
            return result.returncode

        if correval:
            correval_bin = self.find_asp_executable("corr_eval")
            ce_call = self._build_correval_cmd(
                out_prefix, corr_algorithm, corr_kernel, metric,
                correval_bin=correval_bin,
            )
            ce_result = subprocess.run(
                ce_call,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if ce_result.returncode != 0:
                print("ASP corr_eval error:\n", ce_result.stderr)

        if clean:
            out_dir = Path(out_prefix).parent
            for f in sorted(out_dir.iterdir()):
                if f.is_file() and not any(f.name.endswith(s) for s in self._KEEP_SUFFIXES):
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
        correval: bool = False,
        metric: str = "ncc",
        **corr_params_kwargs,
    ) -> int:
        """Run correlation for a GMC ``Pair`` object, then optionally run ``corr_eval``.

        Expects the pair to have been clipped first (``pair.clip()``).
        Writes a settings file inside the pair directory, then calls
        ``parallel_stereo`` and optionally ``corr_eval``.

        Args:
            pair: A ``geomulticorr.pair.Pair`` instance with ``pa_status``
                ``'clipped'`` or ``'complete'``.
            dry_run: Print commands without executing them.
            clean: Delete intermediate files after a successful correlation.
            correval: Run ``corr_eval`` after a successful correlation.
            metric: Metric for ``corr_eval`` (default: ``ncc``).
            **corr_params_kwargs: Forwarded to :meth:`build_correlation_params`.

        Returns:
            int: ``parallel_stereo`` return code.
        """
        if pair.get_status() == "complete":
            print(f"Pair '{pair.pa_key}' is already complete. Skipping.")
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
        corr_kernel    = corr_params_kwargs.get("corr_kernel", (21, 21))

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
        corr_params_file: str | Path = None,
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

        The script is written to ``pair.pa_path / "correlation_job.sh"`` and
        made executable.  When *cluster* is set, OAR scheduler headers are
        prepended and the appropriate environment setup is included.

        Args:
            pair: A ``geomulticorr.pair.Pair`` instance.
            corr_params_file: Path to the ASP settings file (written by
                :meth:`build_correlation_params`).  If *None*, the default
                ``pair.pa_path / "corr_params.txt"`` is used.
            cluster: One of ``None`` / ``"local"`` / ``"gricad"`` / ``"isterre"``.
            nodes: Number of OAR nodes (cluster only).
            cores: Number of cores per node (cluster only).
            walltime: OAR walltime in ``HH:MM:SS`` (cluster only).
            besteffort: Enable OAR besteffort mode (cluster only).
            conda_env: Conda environment name to activate.
            processes / threads_* / corr_memory_limit_mb / corr_tile_size:
                Forwarded to the ``parallel_stereo`` command.

        Returns:
            Path to the written bash script.
        """
        if cluster not in _CLUSTER_CHOICES:
            raise ValueError(
                f"cluster must be one of {_CLUSTER_CHOICES}, got {cluster!r}"
            )

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
            left, right, out_prefix, corr_params_file,
            processes, threads_multiprocess, threads_singleprocess,
            corr_memory_limit_mb, corr_tile_size,
            stereo_bin=stereo_bin,
        )
        # Format as a readable multiline bash command
        cmd_str = " \\\n    ".join(cmd_parts)

        job_name = f"ImgCorr_{pair.pa_key}"
        lines: list[str] = ["#!/bin/bash\n"]

        if cluster in ("gricad", "isterre"):
            lines += [
                f"#OAR -n {job_name}\n",
                f"#OAR -l /nodes={nodes}/core={cores},walltime={walltime}\n",
                f"#OAR -O ./{job_name}_%jobid%.out\n",
                f"#OAR -E ./{job_name}_%jobid%.err\n",
            ]
            if besteffort:
                lines.append("#OAR -t besteffort\n")

            if cluster == "gricad":
                lines += [
                    "#OAR --project rgdyn\n",
                    "\n",
                    "source $HOME/.bashrc\n",
                    f"conda activate {conda_env}\n",
                ]
            else:  # isterre
                lines += [
                    "#OAR --project iste-equ-risques\n",
                    "\n",
                    "source /soft/env.bash\n",
                    "module load stereopipeline/3.6.0\n",
                    "\n",
                    "source /usr/contrib/all/anaconda3/anaconda3.rc\n",
                    f"conda activate {conda_env}\n",
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
        "-ncc.tif",
    )

    def clean_pair_directory(self, pair, dry_run: bool = False) -> list[Path]:
        """Delete intermediate ASP files from *pair*'s output directory.

        Keeps only files whose names end with one of:
        ``-F.tif``, ``-L-R-disp-diff.tif``, ``-L.tif``, ``-R.tif``,
        ``-RD.tif``, ``-ncc.tif``.

        Args:
            pair: A ``geomulticorr.pair.Pair`` instance.
            dry_run: List files that would be deleted without removing them.

        Returns:
            List of deleted (or would-be-deleted) paths.
        """
        asp_dir = pair.pa_path
        if not asp_dir.exists():
            print(f"Pair directory does not exist: {asp_dir}")
            return []

        removed: list[Path] = []
        for f in sorted(asp_dir.iterdir()):
            if not f.is_file() or f.is_symlink():
                continue
            if any(f.name.endswith(suffix) for suffix in self._KEEP_SUFFIXES):
                continue
            if dry_run:
                print(f"DRY RUN — would delete: {f.name}")
            else:
                f.unlink()
            removed.append(f)

        action = "Would delete" if dry_run else "Deleted"
        print(f"{action} {len(removed)} file(s) from '{asp_dir}'.")
        return removed
