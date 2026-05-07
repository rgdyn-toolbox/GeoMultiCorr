#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# _typing.py
# creation date: 2026-05-07.
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

"""Typing aliases for internal use."""

from __future__ import annotations

import sys
from typing import Any, List, Tuple, Union

import numpy as np

# Only for Python >= 3.9
if sys.version_info.minor >= 9:

    from numpy.typing import (  # this syntax works starting on Python 3.9
        ArrayLike,
        DTypeLike,
        NDArray,
    )

    # Mypy has issues with the builtin Number type (https://github.com/python/mypy/issues/3186)
    Number = Union[int, float, np.integer[Any], np.floating[Any]]

    # Simply define here if they exist
    DTypeLike = DTypeLike
    ArrayLike = ArrayLike

    # Use NDArray wrapper to easily define numerical (float or int) N-D array types, and boolean N-D array types
    NDArrayNum = NDArray[Union[np.floating[Any], np.integer[Any]]]
    NDArrayBool = NDArray[np.bool_]
    # Define numerical (float or int) masked N-D array type
    MArrayNum = np.ma.masked_array[Any, np.dtype[Union[np.floating[Any], np.integer[Any]]]]
    MArrayBool = np.ma.masked_array[Any, np.dtype[np.bool_]]

# For backward compatibility before Python 3.9
else:

    # Mypy has issues with the builtin Number type (https://github.com/python/mypy/issues/3186)
    Number = Union[int, float, np.integer, np.floating]  # type: ignore

    # Make an array-like type (since the array-like numpy type only exists in numpy>=1.20)
    DTypeLike = Union[str, type, np.dtype]  # type: ignore
    ArrayLike = Union[np.ndarray, np.ma.masked_array, List[Any], Tuple[Any]]  # type: ignore

    # Define generic types for NumPy array and masked-array (behaves as "Any" before 3.9 and plugin)
    NDArrayNum = np.ndarray  # type: ignore
    NDArrayBool = np.ndarray  # type: ignore
    MArrayNum = np.ma.masked_array  # type: ignore
    MArrayBool = np.ma.masked_array  # type: ignore