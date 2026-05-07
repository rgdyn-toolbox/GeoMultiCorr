#!/usr/bin/env python3
# coding=utf-8
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
# 
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
# 
# install_ASP.py
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
"""
Install the latest stable Ames Stereo Pipeline (ASP) release for Linux x86_64.

Usage:
    python3 install_ASP.py

After installation, the ASP bin/ directory is appended to PATH in ~/.bashrc.
Reload with:
    source ~/.bashrc
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys
import tarfile
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITHUB_API_URL = "https://api.github.com/repos/NeoGeographyToolkit/StereoPipeline/releases"
INSTALL_DIR = pathlib.Path.home() / "ASP_install"
BASHRC = pathlib.Path.home() / ".bashrc"

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

def resolve_asset_url() -> tuple[str, str]:
    """Return (download_url, filename) for the latest stable Linux x86_64 release."""
    print("Querying GitHub releases API for the latest stable ASP release...")
    req = urllib.request.Request(
        GITHUB_API_URL,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        releases = json.load(resp)

    for rel in releases:
        if rel.get("prerelease") or not SEMVER.match(rel.get("tag_name", "")):
            continue
        for asset in rel.get("assets", []):
            if asset["name"].endswith("x86_64-Linux.tar.bz2"):
                return asset["browser_download_url"], asset["name"]

    raise RuntimeError("No stable Linux x86_64 asset found on GitHub.")

def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)

def download(url: str, dest: pathlib.Path) -> None:
    """Download *url* to *dest*, skipping if already present."""
    if dest.exists():
        print(f"Archive already present, skipping download: {dest.name}")
        return
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    print()  # newline after progress bar

def extract(archive: pathlib.Path, dest_dir: pathlib.Path) -> pathlib.Path:
    """Extract *archive* into *dest_dir* and return the top-level directory."""
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive, "r:bz2") as tf:
        # Determine the top-level directory name inside the archive
        top = pathlib.Path(tf.getnames()[0]).parts[0]
        tf.extractall(dest_dir)
    extracted = dest_dir / top
    print(f"Extracted to: {extracted}")
    return extracted

def register_path(bin_dir: pathlib.Path) -> None:
    """Append an export PATH line to ~/.bashrc if not already present."""
    export_line = f'export PATH="$PATH:{bin_dir}"'
    marker = str(bin_dir)

    text = BASHRC.read_text() if BASHRC.exists() else ""
    if marker in text:
        print(f"PATH entry already present in {BASHRC}.")
        return

    with BASHRC.open("a") as f:
        f.write(f"\n# Added by install_ASP.py\n{export_line}\n")
    print(f"PATH entry added to {BASHRC}.")

def verify(bin_dir: pathlib.Path) -> None:
    stereo = bin_dir / "stereo"
    if stereo.exists():
        print("ASP installed successfully.")
    else:
        print(f"ERROR: 'stereo' binary not found in {bin_dir}.", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    # Resolve asset
    url, filename = resolve_asset_url()
    dirname = filename.removesuffix(".tar.bz2")
    print(f"Selected release asset: {filename}")

    # Prepare install directory
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    archive_path = INSTALL_DIR / filename
    download(url, archive_path)

    # Extract
    extracted_dir = INSTALL_DIR / dirname
    if not extracted_dir.exists():
        extract(archive_path, INSTALL_DIR)
    else:
        print(f"Already extracted: {extracted_dir.name}")

    # Register PATH
    bin_dir = extracted_dir / "bin"
    register_path(bin_dir)

    # Verify
    verify(bin_dir)

    print("\nDone. To activate ASP in your current shell, run:")
    print(f"  source {BASHRC}")

if __name__ == "__main__":
    main()
