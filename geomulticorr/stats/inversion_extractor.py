#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# inversion_extractor.py
# creation date: 2026-06-16.
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
"""Post-inversion raster statistics and spatial extraction.

Provides :class:`InversionExtractor` — an OOP interface for computing
raster statistics over TOT inversion outputs and extracting values at
point or polygon locations defined in a GeoPackage file.

Typical usage
-------------
>>> extractor = InversionExtractor(inv)
>>> stats_df = extractor.compute_stats()
>>> gdf = extractor.extract("/path/to/points.gpkg", buffer_m=5.0)
>>> gdf_wide = extractor.extract("/path/to/points.gpkg", output_format="wide")
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterstats

from geomulticorr.stats.stats import compute_raster_stats
from geomulticorr._logging import logger

if TYPE_CHECKING:
    from geomulticorr.inversion.tio_inversion import TIOInversion


_VALID_GEOM_TYPES = {"Point", "MultiPoint", "Polygon", "MultiPolygon"}
_STATS = ["mean", "median", "min", "max", "std", "count"]


class InversionExtractor:
    """Compute statistics and extract values from TIO inversion TOT rasters.

    Discovers all ``TOT_YYYYMMDD_{EW,NS,magn}.tif`` files produced by
    :meth:`~geomulticorr.inversion.tio_inversion.TIOInversion.post_process`
    and exposes two main methods:

    - :meth:`compute_stats` — per-raster descriptive statistics.
    - :meth:`extract` — pixel-value extraction at GPKG geometries, with
      optional buffering for zonal statistics.

    Args:
        inversion: A :class:`~geomulticorr.inversion.tio_inversion.TIOInversion`
            instance **after** ``post_process()`` has been run.
        components: Subset of ``["EW", "NS", "magn"]`` to consider.
            Defaults to all available components.

    Raises:
        FileNotFoundError: If no ``TOT_*.tif`` files are found (i.e.
            ``post_process()`` has not been run yet).
    """

    _COMP_DIR: dict[str, str] = {
        "EW": "inverse_EW",
        "NS": "inverse_NS",
        "magn": "inverse_magn",
    }

    def __init__(
        self,
        inversion: TIOInversion,
        components: list[str] | None = None,
    ) -> None:
        self.inversion = inversion
        self.tif_paths: dict[str, dict[str, Path]] = self._discover_tif_paths(components)
        if not self.tif_paths:
            raise FileNotFoundError(
                f"No TOT_*.tif files found under '{inversion.inversion_dir}'. "
                "Run TIOInversion.post_process() first."
            )
        logger.info(
            f"InversionExtractor ready — {len(self.available_components)} component(s), "
            f"{len(self.available_dates)} date(s)."
        )

    # ── discovery ──────────────────────────────────────────────────────────────

    def _discover_tif_paths(
        self, components: list[str] | None
    ) -> dict[str, dict[str, Path]]:
        comps = components or list(self._COMP_DIR.keys())
        result: dict[str, dict[str, Path]] = {}
        for comp in comps:
            subdir = self.inversion.inversion_dir / self._COMP_DIR[comp]
            if not subdir.exists():
                continue
            found = sorted(subdir.glob("TOT_*.tif"))
            if found:
                # Strip the "TOT_" prefix and the "_{comp}" component suffix
                # (e.g. "TOT_20210901_EW" -> "20210901") so EW/NS/magn share
                # identical date keys. A no-op on a pre-suffix file left over
                # from before TOT_*.tif carried a component suffix.
                result[comp] = {
                    p.stem.removeprefix("TOT_").removesuffix(f"_{comp}"): p
                    for p in found
                }
        return result

    # ── properties ─────────────────────────────────────────────────────────────

    @property
    def available_dates(self) -> list[str]:
        """Sorted list of YYYYMMDD date strings present in any component."""
        dates: set[str] = set()
        for date_map in self.tif_paths.values():
            dates.update(date_map.keys())
        return sorted(dates)

    @property
    def available_components(self) -> list[str]:
        """Components for which TOT rasters were found."""
        return list(self.tif_paths.keys())

    # ── raster statistics ──────────────────────────────────────────────────────

    def compute_stats(
        self,
        components: list[str] | None = None,
        percentiles: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute descriptive statistics for every TOT raster.

        Delegates per-raster computation to
        :func:`~geomulticorr.stats.stats.compute_raster_stats`.

        Args:
            components: Restrict to a subset of available components.
            percentiles: Percentile values to compute
                (default: ``[5, 25, 50, 75, 95]``).

        Returns:
            DataFrame with columns ``date``, ``component``,
            ``count_total``, ``count_valid``, ``valid_fraction``,
            ``mean``, ``median``, ``std``, ``nmad``, ``min``, ``max``,
            and one column per percentile (e.g. ``p5``).
        """
        comps = components or self.available_components
        records: list[dict] = []
        for comp in comps:
            if comp not in self.tif_paths:
                logger.warning(f"Component '{comp}' not available, skipping.")
                continue
            for date_str, path in sorted(self.tif_paths[comp].items()):
                stats = compute_raster_stats(path, percentiles=percentiles)
                if stats is None:
                    logger.warning(f"Could not read stats for {path.name}, skipping.")
                    continue
                records.append({
                    "date": _fmt_date(date_str),
                    "component": comp,
                    **stats,
                })
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ── spatial extraction ─────────────────────────────────────────────────────

    def extract(
        self,
        gpkg_path: str | Path,
        layer: str | None = None,
        buffer_m: float | None = None,
        components: list[str] | None = None,
        output_format: str = "long",
    ) -> gpd.GeoDataFrame:
        """Extract raster values at GPKG geometry locations.

        For each combination of component and date, samples the TOT raster at
        the given geometries.  When *buffer_m* is provided and the input
        contains point geometries, each point is buffered before extraction,
        yielding proper zonal statistics (mean/median/min/max/std/count).
        Polygon and MultiPolygon geometries are always treated as zones
        regardless of *buffer_m*.

        Point extraction without buffering returns a single sampled value
        per point; all stat columns (mean, median, min, max) are set to
        that value for attribute-table consistency.

        Args:
            gpkg_path: Path to a GeoPackage file (points or polygons).
            layer: Layer name within the GeoPackage. Defaults to the first layer.
            buffer_m: Buffer radius in metres (CRS units assumed to be metric).
                Applied only to Point/MultiPoint geometries.
            components: Restrict extraction to a subset of available components.
            output_format: ``"long"`` (default) — one row per
                (geometry × date × component); or ``"wide"`` — one row per
                geometry with compound column names
                ``{component}_{YYYY-MM-DD}_{stat}``.

        Returns:
            GeoDataFrame in the requested format.  The CRS matches the
            inversion rasters.

        Raises:
            ValueError: If the GeoPackage contains unsupported geometry types
                (e.g. LineString).
        """
        gpkg_path = Path(gpkg_path)
        gdf = gpd.read_file(gpkg_path, layer=layer)

        geom_types = set(gdf.geometry.geom_type.dropna().unique())
        unsupported = geom_types - _VALID_GEOM_TYPES
        if unsupported:
            raise ValueError(
                f"Unsupported geometry types in '{gpkg_path.name}': {unsupported}. "
                f"Expected one of {_VALID_GEOM_TYPES}."
            )

        # Reproject to raster CRS
        raster_crs = self._raster_crs()
        if gdf.crs is None:
            logger.warning("Input GeoPackage has no CRS — assuming it matches raster CRS.")
        elif gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        # Precompute point IDs
        point_ids = _extract_ids(gdf)

        # Optionally buffer points
        is_point = geom_types.issubset({"Point", "MultiPoint"})
        extract_gdf = gdf.copy()
        if buffer_m and is_point:
            extract_gdf["geometry"] = extract_gdf.geometry.buffer(buffer_m)
            logger.info(f"Points buffered by {buffer_m} m for zonal extraction.")

        use_zonal = (buffer_m is not None and is_point) or not is_point
        comps = components or self.available_components

        records: list[dict] = []
        for comp in comps:
            if comp not in self.tif_paths:
                logger.warning(f"Component '{comp}' not available, skipping.")
                continue
            for date_str, raster_path in sorted(self.tif_paths[comp].items()):
                date_fmt = _fmt_date(date_str)
                if use_zonal:
                    rows = _extract_zonal(
                        extract_gdf, raster_path, point_ids, date_fmt, comp
                    )
                else:
                    rows = _extract_points(
                        extract_gdf, raster_path, point_ids, date_fmt, comp
                    )
                records.extend(rows)

        result = gpd.GeoDataFrame(records, crs=raster_crs)

        if output_format == "wide":
            return _to_wide(result)
        return result

    # ── helpers ────────────────────────────────────────────────────────────────

    def _raster_crs(self) -> rasterio.crs.CRS:
        first_comp = next(iter(self.tif_paths))
        first_date = next(iter(self.tif_paths[first_comp]))
        with rasterio.open(self.tif_paths[first_comp][first_date]) as src:
            return src.crs


# ── module-level helpers (not exported) ────────────────────────────────────────

def _fmt_date(date_str: str) -> str:
    """Convert 'YYYYMMDD' string to 'YYYY-MM-DD'."""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def _extract_ids(gdf: gpd.GeoDataFrame) -> list:
    """Return a list of point identifiers, preferring named ID columns."""
    for col in ("point_id", "id", "name", "fid", "ID", "Name"):
        if col in gdf.columns:
            return gdf[col].tolist()
    return gdf.index.tolist()


def _extract_zonal(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    point_ids: list,
    date_fmt: str,
    comp: str,
) -> list[dict]:
    """Extract zonal statistics from a raster for each geometry."""
    zs = rasterstats.zonal_stats(
        gdf.geometry,
        str(raster_path),
        stats=_STATS,
        all_touched=False,
    )
    rows = []
    for i, stat in enumerate(zs):
        rows.append({
            "point_id":  point_ids[i],
            "date":      date_fmt,
            "component": comp,
            "mean":      stat.get("mean"),
            "median":    stat.get("median"),
            "min":       stat.get("min"),
            "max":       stat.get("max"),
            "std":       stat.get("std"),
            "count":     stat.get("count"),
            "geometry":  gdf.geometry.iloc[i],
        })
    return rows


def _extract_points(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    point_ids: list,
    date_fmt: str,
    comp: str,
) -> list[dict]:
    """Sample raster at point locations (nearest pixel, no buffer)."""
    values = rasterstats.point_query(
        gdf.geometry,
        str(raster_path),
        interpolate="nearest",
    )
    rows = []
    for i, val in enumerate(values):
        v = float(val) if val is not None else np.nan
        is_finite = np.isfinite(v)
        rows.append({
            "point_id":  point_ids[i],
            "date":      date_fmt,
            "component": comp,
            "mean":      v,
            "median":    v,
            "min":       v,
            "max":       v,
            "std":       0.0 if is_finite else np.nan,
            "count":     1 if is_finite else 0,
            "geometry":  gdf.geometry.iloc[i],
        })
    return rows


def _to_wide(long_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Pivot long-format extraction result to wide format.

    Output columns: ``point_id``, geometry, then
    ``{component}_{YYYY-MM-DD}_{stat}`` for each (component, date, stat)
    combination, sorted by component → date → stat.
    """
    geometry_map = (
        long_gdf.drop_duplicates("point_id")
        .set_index("point_id")["geometry"]
    )

    # Build a plain DataFrame from the long-format records
    df = pd.DataFrame(long_gdf.drop(columns="geometry"))

    stat_cols = ["mean", "median", "min", "max", "std", "count"]
    wide_frames: list[pd.DataFrame] = []
    for stat in stat_cols:
        pivoted = df.pivot_table(
            index="point_id",
            columns=["component", "date"],
            values=stat,
            aggfunc="first",
        )
        # Flatten MultiIndex columns: (comp, date) → "comp_date_stat"
        pivoted.columns = [f"{comp}_{date}_{stat}" for comp, date in pivoted.columns]
        wide_frames.append(pivoted)

    wide_df = pd.concat(wide_frames, axis=1)

    # Re-order columns: sort by component, date, then stat
    col_order = sorted(
        wide_df.columns,
        key=lambda c: (c.split("_", 1)[0], *c.rsplit("_", 1)),
    )
    wide_df = wide_df[col_order]

    wide_gdf = gpd.GeoDataFrame(
        wide_df.reset_index(),
        geometry=geometry_map.loc[wide_df.index].values,
        crs=long_gdf.crs,
    )
    return wide_gdf
