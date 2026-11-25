#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# check_license_header.py
# creation date: 2026-11-25.
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
"""Check that every Python file starts with the GeoMultiCorr licence header."""

from __future__ import annotations
import argparse, re, sys
from datetime import datetime
from pathlib import Path

TEMPLATE = Path(__file__).with_name("license-header.txt")
PLACEHOLDERS = {
    "{year}": r"(?P<year>\d{4})",
    "{date}": r"(?P<date>\d{4}-\d{2}-\d{2})",
    "{filename}": r"(?P<filename>[^\n]+)",
}


def build_pattern() -> re.Pattern[str]:
    text = TEMPLATE.read_text(encoding="utf-8").rstrip("\n")
    pattern = re.escape(text)
    for key, regex in PLACEHOLDERS.items():
        pattern = pattern.replace(re.escape(key), regex)
    return re.compile(r"\A" + pattern + r"\n")


def check(path: Path, pattern: re.Pattern[str]) -> str | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return f"cannot read file: {error}"
    match = pattern.match(content)
    if match is None:
        return f"header missing or different from {TEMPLATE.name}"
    if match.group("filename") != path.name:
        return f"header names '{match.group('filename')}', file is '{path.name}'"
    try:
        created = datetime.strptime(match.group("date"), "%Y-%m-%d")
    except ValueError:
        return f"creation date '{match.group('date')}' is not a valid date"
    if match.group("year") != str(created.year):
        return (
            f"copyright year {match.group('year')} differs from the year of "
            f"the creation date {match.group('date')}"
        )
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    pattern = build_pattern()
    failed = False
    for path in parser.parse_args(argv).files:
        error = check(path, pattern)
        if error:
            print(f"{path}: {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
