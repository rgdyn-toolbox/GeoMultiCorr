#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# test_asp.py
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
"""Tests for ASP class initialization and binary discovery."""

import pytest
from pathlib import Path
from unittest.mock import patch
from geomulticorr.correlation import ASP

# --------------------------------------------------------------------------- #
# INITIALIZATION TESTS
# --------------------------------------------------------------------------- #
class TestASPInitialization:
    """Test ASP.__init__ method for correct binary discovery and error handling."""

    @patch('shutil.which')
    def test_init_success_with_mock_binary(self, mock_which):
        """ASP initializes when parallel_stereo found."""
        mock_which.return_value = '/usr/local/bin/parallel_stereo'
        asp = ASP(session=None, asp_bin_dir=None)

        assert asp is not None
        mock_which.assert_called()
    #END def

    @patch('shutil.which')
    def test_init_raises_when_binary_not_found(self, mock_which):
        """ASP raises FileNotFoundError without binary."""
        mock_which.return_value = None

        with pytest.raises(FileNotFoundError) as exc_info:
            ASP(session=None, asp_bin_dir=None)

        assert "parallel_stereo" in str(exc_info.value)
        assert "PATH" in str(exc_info.value)    
        #END with
    #END def

    @patch('shutil.which')
    def test_init_with_custom_bin_dir(self, mock_which):
        """ASP should accept custom binary directory."""
        mock_which.return_value = "/custom/bin/parallel_stereo"

        asp = ASP(session=None, asp_bin_dir="/custom/bin")

        assert asp.asp_bin_dir == Path("/custom/bin")
        # shutil.which should have searched in the custom directory
        mock_which.assert_called()
    # END def
# END class

# --------------------------------------------------------------------------- #
# BINARY FINDING TESTS
# --------------------------------------------------------------------------- #
class TestFindASPExecutable:
    """Test find_asp_executable method."""

    @pytest.fixture
    def asp_with_mock(self):
        """Create ASP instance with mocked binary."""
        with patch('shutil.which', return_value="/fake/bin/parallel_stereo"):
            yield ASP(session=None, asp_bin_dir=None)
    # END def

    @patch('shutil.which')
    def test_finds_executable_in_system_path(self, mock_which):
        """Should find executable using shutil.which."""
        mock_which.return_value = "/usr/bin/parallel_stereo"

        with patch('shutil.which') as init_mock:
            init_mock.return_value = "/usr/bin/parallel_stereo"
            asp = ASP()

        with patch('shutil.which', return_value="/usr/bin/corr_eval") as search_mock:
            result = asp.find_asp_executable("corr_eval")

        assert result == "/usr/bin/corr_eval"
    # END def

    @patch('shutil.which')
    def test_raises_when_executable_not_found(self, mock_which):
        """Should raise FileNotFoundError for missing executables."""
        # First call: init succeeds with parallel_stereo
        # Second call: find fails for corr_eval
        mock_which.side_effect = [
            "/usr/bin/parallel_stereo",  # init call
            None,  # find call
        ]

        with patch('shutil.which') as init_mock:
            init_mock.return_value = "/usr/bin/parallel_stereo"
            asp = ASP()

        with patch('shutil.which', return_value=None):
            with pytest.raises(FileNotFoundError) as exc_info:
                asp.find_asp_executable("missing_binary")

            assert "missing_binary" in str(exc_info.value)
    # END def
# END class