#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# add_license_header.py
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
"""Add or repair the GeoMultiCorr licence header (one-off, DRY-RUN by default).

For every Python file that does not pass ``check_license_header.py``:

* a missing header is inserted from ``license-header.txt``;
* an old or partial licence header (Apache variant, no file name, wrong
  shebang, wrong file name, ...) is replaced by the template;
* the shebang and coding lines are always the template's;
* the module docstring and the code after the header are never touched.

The creation date is, in this order: the ``creation date:`` line of the old
header, the date the file was first added in git, the file modification time.
The copyright year is the year of that date, as the checker requires.

Nothing is written unless ``--write`` is given. Use ``--diff`` to see the
changes. Files are read and written with ``\\n`` line endings.

Usage::

    python .github/scripts/add_license_header.py              # report only
    python .github/scripts/add_license_header.py --diff       # report + diffs
    python .github/scripts/add_license_header.py --write      # apply
    python .github/scripts/add_license_header.py a.py b.py    # chosen files
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_license_header import TEMPLATE, build_pattern  # noqa: E402

DEFAULT_SCOPE = ("geomulticorr/", "tests/")
CODING_LINE = re.compile(r"#.*coding[:=]")
OLD_DATE = re.compile(r"creation date:\s*(\d{4}-\d{2}-\d{2})")
LICENCE_WORDS = re.compile(r"copyright|licen[sc]e|GeoMultiCorr", re.IGNORECASE)


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def default_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", "*.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    names = [n for n in out.stdout.splitlines() if n.startswith(DEFAULT_SCOPE)]
    return [root / n for n in sorted(names)]


def valid_date(text: str) -> bool:
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def git_creation_date(path: Path) -> str | None:
    """Date of the commit that added the file (oldest), None if untracked."""
    out = subprocess.run(
        ["git", "log", "--follow", "--diff-filter=A", "--format=%as", "--", path.name],
        cwd=path.parent,
        capture_output=True,
        text=True,
    )
    lines = out.stdout.split()
    return lines[-1] if out.returncode == 0 and lines else None


def split_old_header(content: str) -> tuple[str, int, str | None]:
    """Return (body, removed_line_count, old_creation_date).

    ``body`` is the file without its shebang/coding lines and without the old
    licence comment block. A top comment block that is not a licence header
    (no "copyright"/"license"/"GeoMultiCorr" in it) is kept.
    """
    lines = content.splitlines(keepends=True)
    i = 0
    while i < len(lines) and i < 2:
        if lines[i].startswith("#!") or CODING_LINE.match(lines[i]):
            i += 1
        else:
            break
    removed = i
    old_date = None
    j = i
    while j < len(lines) and lines[j].startswith("#"):
        j += 1
    block = "".join(lines[i:j])
    if block and LICENCE_WORDS.search(block):
        found = OLD_DATE.search(block)
        if found and valid_date(found.group(1)):
            old_date = found.group(1)
        removed += j - i
        i = j
    return "".join(lines[i:]).lstrip("\n"), removed, old_date


def build_new_content(path: Path, content: str, template: str) -> tuple[str, str]:
    """Return (new content, short description of what was done)."""
    body, removed, old_date = split_old_header(content)
    if old_date:
        date, source = old_date, "date kept from old header"
    elif git_date := git_creation_date(path):
        date, source = git_date, "date from git"
    else:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        date, source = mtime.strftime("%Y-%m-%d"), "date from file mtime"
    header = (
        template.replace("{filename}", path.name)
        .replace("{date}", date)
        .replace("{year}", date[:4])
    )
    action = f"replace {removed} old line(s)" if removed else "insert header"
    # Existing files put the module docstring right under the header; any
    # other code gets one blank line of separation.
    gap = "" if not body or body.startswith(('"""', "'''", 'r"""', "r'''")) else "\n"
    new = header + "\n" + gap + body
    return new, f"{action}; {source} ({date})"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="*", type=Path, help="default: package + tests")
    parser.add_argument("--write", action="store_true", help="apply the changes")
    parser.add_argument("--diff", action="store_true", help="show unified diffs")
    args = parser.parse_args(argv)

    root = repo_root()
    files = [p.resolve() for p in args.files] or default_files(root)
    template = TEMPLATE.read_text(encoding="utf-8").rstrip("\n")
    pattern = build_pattern()

    changed = ok = failed = 0
    for path in files:
        shown = path.relative_to(root) if path.is_relative_to(root) else path
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            print(f"ERROR   {shown}: cannot read file: {error}")
            failed += 1
            continue
        match = pattern.match(content)
        if match and match.group("filename") == path.name:
            # Right header; the date/year rules are the checker's job.
            ok += 1
            continue
        new, what = build_new_content(path, content, template)
        check = pattern.match(new)
        if not check or check.group("filename") != path.name:
            print(f"ERROR   {shown}: generated header does not match the template")
            failed += 1
            continue
        changed += 1
        print(f"{'WRITE' if args.write else 'WOULD '}   {shown}: {what}")
        if args.diff:
            diff = difflib.unified_diff(
                content.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{shown}",
                tofile=f"b/{shown}",
                n=1,
            )
            sys.stdout.writelines(diff)
            print()
        if args.write:
            path.write_text(new, encoding="utf-8", newline="\n")

    print(
        f"\n{len(files)} file(s): {ok} already fine, {changed} "
        f"{'rewritten' if args.write else 'would change'}, {failed} error(s)."
    )
    if changed and not args.write:
        print("Dry run: nothing written. Re-run with --write to apply (see --diff).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
