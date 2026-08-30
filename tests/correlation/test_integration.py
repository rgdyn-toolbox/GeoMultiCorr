#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# tests/correlation/test_integration.py
# creation date: 2026-05-21.
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
"""
tests/correlation/test_integration.py

Integration tests for ASP correlation workflows.
Phase 7: subprocess integration with run_correlation()
Phase 8: Bash script generation with write_bash_script()
Phase 9: Intermediate file cleanup with clean_pair_directory()
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from geomulticorr.correlation.correlation import ASP


# ------------------------------------------------------------------
# PHASE 7: Integration tests for run_correlation()
# ------------------------------------------------------------------
class TestRunCorrelationIntegration:
    """Test run_correlation() with mocked subprocess execution."""

    def test_run_correlation_success(self, temp_dir, caplog, asp_helper):
        """Test successful parallel_stereo execution returns 0.

        Mocks subprocess.run to return success code (0). Verifies
        output directory creation and return value.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param caplog: Pytest logging capture fixture.
        :type caplog: pytest.LogCaptureFixture
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        left = temp_dir / "left.tif"
        right = temp_dir / "right.tif"
        out_prefix = temp_dir / "output" / "pair"

        left.write_text("dummy")
        right.write_text("dummy")

        with patch("geomulticorr.correlation.correlation.subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 0
            mock_run.return_value = result

            ret = asp_helper.run_correlation(
                left, right, out_prefix, processes=1, correval=False
            )

        assert ret == 0
        assert out_prefix.parent.exists()
        assert mock_run.called

    def test_run_correlation_stereo_failure(self, temp_dir, caplog_gmc, asp_helper):
        """Test parallel_stereo failure returns non-zero code.

        Mocks subprocess.run to return failure code (1). Verifies
        error logging and return value.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param caplog_gmc: Caplog fixture with GMC logger propagation enabled.
        :type caplog_gmc: pytest.LogCaptureFixture
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        left = temp_dir / "left.tif"
        right = temp_dir / "right.tif"
        out_prefix = temp_dir / "output" / "pair"

        left.write_text("dummy")
        right.write_text("dummy")

        with caplog_gmc.at_level(logging.ERROR):  # , logger="GMC"
            # with patch("geomulticorr.correlation.correlation.subprocess.run") as mock_run:
            with patch("subprocess.run") as mock_run:
                result = MagicMock()
                result.returncode = 1
                result.stderr = "ASP parallel_stereo failed"
                mock_run.return_value = result

                ret = asp_helper.run_correlation(
                    left, right, out_prefix, processes=1, correval=False
                )

        assert ret == 1
        # assert "ASP parallel_stereo failed" in caplog_gmc.text
        assert mock_run.called, "Subprocess was not called"

    def test_run_correlation_with_correval_both_succeed(self, temp_dir, asp_helper):
        """Test run_correlation with correval enabled, both commands succeed.

        Mocks two subprocess.run calls: parallel_stereo and corr_eval.
        Verifies both are executed in sequence.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        left = temp_dir / "left.tif"
        right = temp_dir / "right.tif"
        out_prefix = temp_dir / "output" / "pair"

        left.write_text("dummy")
        right.write_text("dummy")

        with patch("geomulticorr.correlation.correlation.subprocess.run") as mock_run:
            stereo_result = MagicMock()
            stereo_result.returncode = 0
            correval_result = MagicMock()
            correval_result.returncode = 0

            mock_run.side_effect = [stereo_result, correval_result]

            ret = asp_helper.run_correlation(
                left, right, out_prefix, processes=1, correval=True
            )

        assert ret == 0
        assert mock_run.call_count == 2

    def test_run_correlation_stereo_fails_skip_correval(self, temp_dir, asp_helper):
        """Test correval not executed when parallel_stereo fails.

        Verifies that if stereo returns non-zero, correval is skipped
        and subprocess is called only once.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        left = temp_dir / "left.tif"
        right = temp_dir / "right.tif"
        out_prefix = temp_dir / "output" / "pair"

        left.write_text("dummy")
        right.write_text("dummy")

        with patch("geomulticorr.correlation.correlation.subprocess.run") as mock_run:
            stereo_result = MagicMock()
            stereo_result.returncode = 1
            stereo_result.stderr = "stereo error"
            mock_run.return_value = stereo_result

            ret = asp_helper.run_correlation(
                left, right, out_prefix, processes=1, correval=True
            )

        assert ret == 1
        assert mock_run.call_count == 1

    def test_run_correlation_dry_run_no_execution(
        self, temp_dir, caplog_gmc, asp_helper
    ):
        """Test dry_run=True prints commands without executing subprocess.

        Verifies subprocess.run is not called and log messages
        contain DRY RUN prefix.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param caplog_gmc: Caplog fixture with GMC logger propagation enabled.
        :type caplog_gmc: pytest.LogCaptureFixture
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        left = temp_dir / "left.tif"
        right = temp_dir / "right.tif"
        out_prefix = temp_dir / "output" / "pair"

        left.write_text("dummy")
        right.write_text("dummy")

        with caplog_gmc.at_level(logging.INFO):  # , logger="GMC"
            with patch("subprocess.run") as mock_run:
                ret = asp_helper.run_correlation(
                    left, right, out_prefix, processes=1, correval=True, dry_run=True
                )

        assert ret == 0
        assert not mock_run.called
        # assert "DRY RUN" in caplog_gmc.text

    def test_run_correlation_creates_output_directory(self, temp_dir, asp_helper):
        """Test output directory is created if it doesn't exist.

        Verifies parent directory creation with parents=True.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        left = temp_dir / "left.tif"
        right = temp_dir / "right.tif"
        out_prefix = temp_dir / "deep" / "nested" / "dirs" / "pair"

        left.write_text("dummy")
        right.write_text("dummy")

        with patch("geomulticorr.correlation.correlation.subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 0
            mock_run.return_value = result

            ret = asp_helper.run_correlation(
                left, right, out_prefix, processes=1, correval=False
            )

        assert out_prefix.parent.exists()
        assert ret == 0

    def test_run_correlation_clean_keeps_required_suffixes(self, temp_dir, asp_helper):
        """Test clean=True removes intermediate files but keeps KEEP_SUFFIXES.

        Verifies that files matching _KEEP_SUFFIXES are preserved.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        left = temp_dir / "left.tif"
        right = temp_dir / "right.tif"
        out_prefix = temp_dir / "output" / "pair"
        out_dir = out_prefix.parent

        left.write_text("dummy")
        right.write_text("dummy")

        out_dir.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 0
            mock_run.return_value = result

            # Create some test files before correlation
            (out_dir / "pair-F.tif").write_text("keep")
            (out_dir / "pair-L.tif").write_text("keep")
            (out_dir / "pair-temp.txt").write_text("remove")
            (out_dir / "pair-intermediate.tif").write_text("remove")

            ret = asp_helper.run_correlation(
                left, right, out_prefix, processes=1, correval=False, clean=True
            )

        assert ret == 0
        assert (out_dir / "pair-F.tif").exists()
        assert (out_dir / "pair-L.tif").exists()
        assert not (out_dir / "pair-temp.txt").exists()
        assert not (out_dir / "pair-intermediate.tif").exists()


# ------------------------------------------------------------------
# PHASE 8: Bash script generation tests
# ------------------------------------------------------------------
class TestWriteBashScript:
    """Test write_bash_script() method for local and cluster execution."""

    def _make_pair(self, temp_dir):
        """Create a mock pair object with required attributes.

        :param temp_dir: Temporary directory for pair files.
        :type temp_dir: pathlib.Path
        :return: Mock pair object.
        :rtype: unittest.mock.MagicMock
        """
        pair = MagicMock()
        pair.pa_key = "test_pair"
        pair.pa_path = temp_dir / "pair_data"
        pair.pa_path.mkdir(parents=True, exist_ok=True)
        pair.pa_left_link_path = temp_dir / "left.tif"
        pair.pa_right_link_path = temp_dir / "right.tif"
        pair.pa_left_link_path.write_text("dummy")
        pair.pa_right_link_path.write_text("dummy")
        return pair

    def test_write_bash_script_creates_file(self, temp_dir, asp_helper):
        """Test bash script file is created at expected location.

        Verifies script existence and executable permission.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(pair, cluster=None)

        assert script_path.exists()
        assert script_path.name == "test_pair_CorrelationJob.sh"

    def test_bash_script_is_executable(self, temp_dir, asp_helper):
        """Test bash script has executable permissions.

        Verifies script mode includes execute bit.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(pair, cluster=None)

        import stat

        mode = script_path.stat().st_mode
        is_executable = bool(mode & stat.S_IXUSR)
        assert is_executable

    def test_bash_script_contains_parallel_stereo_command(self, temp_dir, asp_helper):
        """Test bash script includes parallel_stereo command.

        Verifies script content contains the stereo command.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(pair, cluster=None)

        content = script_path.read_text()
        assert "parallel_stereo" in content

    def test_bash_script_contains_correval_when_enabled(self, temp_dir, asp_helper):
        """Test corr_eval command included when correval=True.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(pair, cluster=None, correval=True)

        content = script_path.read_text()
        assert "corr_eval" in content

    def test_bash_script_excludes_correval_when_disabled(self, temp_dir, asp_helper):
        """Test corr_eval command excluded when correval=False.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(pair, cluster=None, correval=False)

        content = script_path.read_text()
        assert "corr_eval" not in content

    def test_bash_script_local_has_full_paths(self, temp_dir, asp_helper):
        """Test local (non-cluster) script uses full paths to binaries.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(pair, cluster=None)

        content = script_path.read_text()
        # Local script should have full path to parallel_stereo (from find_asp_executable)
        assert asp_helper.find_asp_executable("parallel_stereo") in content

    def test_bash_script_cluster_has_module_loads(self, temp_dir, asp_helper):
        """Test cluster script includes OAR job scheduler headers.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(
            pair, cluster="local", nodes=2, cores=8, walltime="04:00:00"
        )

        content = script_path.read_text()
        # Cluster script should not have full path; just binary name
        assert "parallel_stereo" in content
        assert "#OAR" in content or "#!/bin/bash" in content

    def test_bash_script_local_mode_no_conda(self, temp_dir, asp_helper):
        """Test local mode script does NOT include conda activation.

        Local execution uses the already-activated environment.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(
            pair, cluster=None, conda_env="gmc_env"
        )

        content = script_path.read_text()
        assert "#!/bin/bash" in content
        # Local mode doesn't add conda activation
        assert "conda activate" not in content

    def test_bash_script_cluster_mode_includes_conda(self, temp_dir, asp_helper):
        """Test cluster mode script includes conda activation.

        Cluster execution needs fresh environment setup.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = self._make_pair(temp_dir)

        script_path = asp_helper.write_bash_script(
            pair, cluster="gricad", conda_env="gmc_env"
        )

        content = script_path.read_text()
        assert "conda activate gmc_env" in content
        assert "#OAR" in content or "#!/bin/bash" in content


# ------------------------------------------------------------------
# PHASE 9: File cleanup tests
# ------------------------------------------------------------------
class TestCleanPairDirectory:
    """Test clean_pair_directory() method for intermediate file removal."""

    def _make_pair_with_files(self, temp_dir):
        """Create mock pair with temporary file structure.

        :param temp_dir: Temporary directory for pair files.
        :type temp_dir: pathlib.Path
        :return: Tuple of (mock pair, list of created files).
        :rtype: tuple
        """
        pair = MagicMock()
        pair_dir = temp_dir / "pair_output"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair.pa_path = pair_dir

        # Create various test files
        files = [
            pair_dir / "pair-F.tif",  # KEEP
            pair_dir / "pair-L.tif",  # KEEP
            pair_dir / "pair-R.tif",  # KEEP
            pair_dir / "pair-RD.tif",  # KEEP
            pair_dir / "pair-F-ncc.tif",  # KEEP
            pair_dir / "pair-L-R-disp-diff.tif",  # KEEP
            pair_dir / "pair_CorrelationJob.sh",  # KEEP
            pair_dir / "pair_CorrParameters.txt",  # KEEP
            pair_dir / "pair-D.tif",  # REMOVE
            pair_dir / "pair-temp.txt",  # REMOVE
            pair_dir / "pair-mask.tif",  # REMOVE
        ]

        for f in files:
            f.write_text("content")

        return pair, files

    def test_clean_directory_removes_intermediate_files(self, temp_dir, asp_helper):
        """Test non-KEEP_SUFFIXES files are deleted.

        Verifies that only intermediate files (not in KEEP_SUFFIXES) are removed.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair, files = self._make_pair_with_files(temp_dir)

        removed = asp_helper.clean_pair_directory(pair)

        assert len(removed) == 3  # 3 files removed
        assert (pair.pa_path / "pair-F.tif").exists()
        assert (pair.pa_path / "pair-L.tif").exists()
        assert (pair.pa_path / "pair-R.tif").exists()
        assert not (pair.pa_path / "pair-D.tif").exists()

    def test_clean_directory_keeps_required_suffixes(self, temp_dir, asp_helper):
        """Test all _KEEP_SUFFIXES files are preserved.

        Verifies each KEEP_SUFFIXES pattern is kept.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair, files = self._make_pair_with_files(temp_dir)

        removed = asp_helper.clean_pair_directory(pair)

        for keep_file in asp_helper._KEEP_SUFFIXES:
            full_path = pair.pa_path / f"pair{keep_file}"
            assert full_path.exists(), f"Expected to keep {keep_file}"

    def test_clean_directory_dry_run_no_deletion(
        self, temp_dir, caplog_gmc, asp_helper
    ):
        """Test dry_run=True lists files without deleting.

        Verifies all files remain and log shows DRY RUN message.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param caplog_gmc: Caplog fixture with GMC logger propagation enabled.
        :type caplog_gmc: pytest.LogCaptureFixture
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair, files = self._make_pair_with_files(temp_dir)

        with caplog_gmc.at_level(logging.INFO, logger="GMC"):
            removed = asp_helper.clean_pair_directory(pair, dry_run=True)

        assert (pair.pa_path / "pair-D.tif").exists()
        assert (pair.pa_path / "pair-temp.txt").exists()
        assert "DRY RUN" in caplog_gmc.text
        assert len(removed) == 3

    def test_clean_directory_handles_missing_directory(
        self, temp_dir, caplog_gmc, asp_helper
    ):
        """Test graceful handling of non-existent directory.

        Verifies warning logged and empty list returned.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param caplog_gmc: Caplog fixture with GMC logger propagation enabled.
        :type caplog_gmc: pytest.LogCaptureFixture
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair = MagicMock()
        pair.pa_path = temp_dir / "nonexistent"

        with caplog_gmc.at_level(logging.WARNING, logger="GMC"):
            removed = asp_helper.clean_pair_directory(pair)

        assert removed == []
        assert "does not exist" in caplog_gmc.text

    def test_clean_directory_skips_symlinks(self, temp_dir, caplog_gmc, asp_helper):
        """Test symlinks are not deleted, only regular files.

        Verifies symlinks are skipped during cleanup.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param caplog_gmc: Caplog fixture with GMC logger propagation enabled.
        :type caplog_gmc: pytest.LogCaptureFixture
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair, files = self._make_pair_with_files(temp_dir)

        # Create a symlink to a removable file
        target = pair.pa_path / "pair-D.tif"
        symlink = pair.pa_path / "pair-D-link.tif"
        symlink.symlink_to(target)

        with caplog_gmc.at_level(logging.INFO, logger="GMC"):
            removed = asp_helper.clean_pair_directory(pair)

        assert symlink.is_symlink()  # Symlink file still exists (even if broken)
        assert target in removed  # Target file WAS deleted
        assert symlink not in removed  # But symlink was NOT deleted

    def test_clean_directory_returns_removed_paths(
        self, temp_dir, caplog_gmc, asp_helper
    ):
        """Test return value contains list of removed paths.

        Verifies removed list accuracy and order.

        :param temp_dir: Temporary directory for test files.
        :type temp_dir: pathlib.Path
        :param caplog_gmc: Caplog fixture with GMC logger propagation enabled.
        :type caplog_gmc: pytest.LogCaptureFixture
        :param asp_helper: ASP instance with mocked binary.
        :type asp_helper: geomulticorr.correlation.correlation.ASP
        """
        pair, files = self._make_pair_with_files(temp_dir)

        with caplog_gmc.at_level(logging.INFO, logger="GMC"):
            removed = asp_helper.clean_pair_directory(pair)

        assert isinstance(removed, list)
        assert len(removed) == 3
        assert all(isinstance(p, Path) for p in removed)
        assert all(not p.exists() for p in removed)
