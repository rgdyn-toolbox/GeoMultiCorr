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
>>> gdf_wide.to_file("series_at_points.gpkg", layer="series")   # opens in QGIS
"""

from __future__ import annotations

import warnings
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

#: Columns the extraction itself writes. An attribute of the input layer with one
#: of these names is kept as ``src_<name>`` instead of overwriting it.
_RESERVED_COLUMNS = {"point_id", "date", "component", "geometry", "_row", *_STATS}

#: Sub-folder of an inversion directory holding each component's TOT rasters.
#: Shared with :mod:`geomulticorr.inversion._stack`, which imports from here
#: (``stats`` initialises before ``inversion``; the reverse would be a cycle).
COMP_DIR: dict[str, str] = {
    "EW": "inverse_EW",
    "NS": "inverse_NS",
    "magn": "inverse_magn",
}


def resolve_inversion_dir(source) -> Path:
    """The inversion directory behind *source*.

    Accepts a path (``str``/``Path``) or anything with an ``inversion_dir``
    attribute — a :class:`~geomulticorr.inversion.tio_inversion.TIOInversion`,
    duck-typed so this module never imports it. A fused product written by
    :mod:`geomulticorr.inversion.fusion` has no ``TIOInversion`` and is passed
    as its folder.
    """
    inv_dir = getattr(source, "inversion_dir", None)
    if inv_dir is None:
        inv_dir = source
    if not isinstance(inv_dir, (str, Path)):
        raise TypeError(
            "expected an inversion directory or an object with `.inversion_dir`, "
            f"got {type(source).__name__}"
        )
    return Path(inv_dir)


def discover_tot_rasters(
    inversion_dir: str | Path, components=None
) -> dict[str, dict[str, Path]]:
    """``{component: {YYYYMMDD: path}}`` for every ``TOT_<date>_<comp>.tif`` on disk.

    The date key strips the ``TOT_`` prefix and the ``_<comp>`` suffix, so EW,
    NS and magn share identical keys (a no-op on a pre-suffix file left over
    from before the rasters carried a component suffix). Components whose
    folder is absent or empty are simply missing from the result.
    """
    inversion_dir = Path(inversion_dir)
    comps = list(components) if components else list(COMP_DIR)
    result: dict[str, dict[str, Path]] = {}
    for comp in comps:
        subdir = inversion_dir / COMP_DIR[comp]
        if not subdir.exists():
            continue
        found = sorted(subdir.glob("TOT_*.tif"))
        if found:
            result[comp] = {
                p.stem.removeprefix("TOT_").removesuffix(f"_{comp}"): p for p in found
            }
    return result


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
            instance **after** ``post_process()`` (or ``launch(mode="python")``)
            has been run — or the inversion directory itself, which is how a
            fused product (:mod:`geomulticorr.inversion.fusion`) is read.
        components: Subset of ``["EW", "NS", "magn"]`` to consider.
            Defaults to all available components.

    Raises:
        FileNotFoundError: If no ``TOT_*.tif`` files are found (i.e.
            ``post_process()`` has not been run yet).
    """

    _COMP_DIR: dict[str, str] = COMP_DIR

    def __init__(
        self,
        inversion: TIOInversion | str | Path,
        components: list[str] | None = None,
    ) -> None:
        self.inversion = inversion
        self.inversion_dir: Path = resolve_inversion_dir(inversion)
        self.tif_paths: dict[str, dict[str, Path]] = self._discover_tif_paths(
            components
        )
        if not self.tif_paths:
            raise FileNotFoundError(
                f"No TOT_*.tif files found under '{self.inversion_dir}'. "
                "Run TIOInversion.post_process() (or launch(mode='python')) first."
            )
        logger.info(
            f"InversionExtractor ready — {len(self.available_components)} component(s), "
            f"{len(self.available_dates)} date(s)."
        )

    # ── discovery ──────────────────────────────────────────────────────────────

    def _discover_tif_paths(
        self, components: list[str] | None
    ) -> dict[str, dict[str, Path]]:
        # Resolved from `self.inversion` each time rather than the cached
        # `inversion_dir`: tests build instances via __new__ and set only
        # `inversion`.
        return discover_tot_rasters(resolve_inversion_dir(self.inversion), components)

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
                records.append(
                    {
                        "date": _fmt_date(date_str),
                        "component": comp,
                        **stats,
                    }
                )
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ── spatial extraction ─────────────────────────────────────────────────────

    def extract(
        self,
        source: str | Path | gpd.GeoDataFrame | None = None,
        layer: str | None = None,
        buffer_m: float | None = None,
        components: list[str] | None = None,
        output_format: str = "long",
        keep_attributes: bool = True,
        stats: tuple[str, ...] | list[str] = tuple(_STATS),
        *,
        gpkg_path: str | Path | None = None,
    ) -> gpd.GeoDataFrame:
        """Extract raster values at point or polygon locations.

        For each combination of component and date, samples the TOT raster at
        the given geometries.  When *buffer_m* is provided and the input
        contains point geometries, each point is buffered **only to measure**:
        the statistics are zonal over the buffer, but the output keeps the
        original point geometry.  Polygon and MultiPolygon geometries are
        always treated as zones regardless of *buffer_m* and keep their shape.

        Point extraction without buffering returns a single sampled value
        per point; every requested statistic is set to that value for
        attribute-table consistency.

        Args:
            source: A GeoPackage / vector file path, **or** a
                :class:`geopandas.GeoDataFrame` already in memory (points or
                polygons, any CRS: it is reprojected to the raster CRS).
            layer: Layer name within the file. Defaults to the first layer.
                Ignored for a GeoDataFrame.
            buffer_m: Buffer radius in metres (CRS units assumed to be metric).
                Applied only to Point/MultiPoint geometries.
            components: Restrict extraction to a subset of available components.
            output_format: ``"long"`` (default) — one row per
                (geometry × date × component); or ``"wide"`` — one row per
                geometry with compound column names
                ``{component}_{YYYY-MM-DD}_{stat}``: the attribute table of the
                input layer, extended with one column per date.
            keep_attributes: Carry the columns of the input layer into the result
                (default ``True``). An attribute whose name collides with a
                column written here (``point_id``, ``date``, ``component``, a
                statistic) is kept as ``src_<name>``.
            stats: Which statistics to compute, among ``"mean"``, ``"median"``,
                ``"min"``, ``"max"``, ``"std"``, ``"count"`` (default: all).
            gpkg_path: Deprecated alias of *source*.

        Returns:
            GeoDataFrame in the requested format, in the raster CRS, with the
            **input geometries** (a point stays a point). Write it with
            ``result.to_file(path, layer=...)``; wide format has one row per
            point and opens directly in QGIS.

        Raises:
            ValueError: If the layer contains unsupported geometry types
                (e.g. LineString), or *stats* holds an unknown name.
            TypeError: If *source* is missing or of an unsupported type.
        """
        if gpkg_path is not None:
            if source is not None:
                raise TypeError("pass `source`, not both `source` and `gpkg_path`")
            warnings.warn(
                "extract(gpkg_path=...) is deprecated; pass the layer as `source`.",
                DeprecationWarning,
                stacklevel=2,
            )
            source = gpkg_path
        if source is None:
            raise TypeError("extract() missing the required argument `source`")
        stats = _check_stats(stats)
        gdf, label = _load_source(source, layer)

        geom_types = set(gdf.geometry.geom_type.dropna().unique())
        unsupported = geom_types - _VALID_GEOM_TYPES
        if unsupported:
            raise ValueError(
                f"Unsupported geometry types in '{label}': {unsupported}. "
                f"Expected one of {_VALID_GEOM_TYPES}."
            )

        # Reproject to raster CRS
        raster_crs = self._raster_crs()
        if gdf.crs is None:
            logger.warning("Input layer has no CRS — assuming it matches raster CRS.")
        elif gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        gdf = gdf.reset_index(drop=True)

        # Precompute point IDs
        point_ids = _extract_ids(gdf)
        if len(set(point_ids)) != len(point_ids):
            logger.warning(
                "point_id is not unique in the input layer: the wide format keeps "
                "the first row of each id."
            )

        # The geometries the result carries are the input ones; buffers only
        # exist to measure.
        out_geoms = list(gdf.geometry)
        zone_geoms = out_geoms
        is_point = geom_types.issubset({"Point", "MultiPoint"})
        if buffer_m and is_point:
            zone_geoms = list(gdf.geometry.buffer(buffer_m))
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
                        zone_geoms, raster_path, point_ids, date_fmt, comp, stats
                    )
                else:
                    rows = _extract_points(
                        zone_geoms, raster_path, point_ids, date_fmt, comp, stats
                    )
                for row in rows:
                    row["geometry"] = out_geoms[row["_row"]]
                records.extend(rows)

        result = gpd.GeoDataFrame(records, crs=raster_crs)

        if keep_attributes and len(result):
            attrs = _attribute_frame(gdf)
            if len(attrs.columns):
                result = result.merge(
                    attrs, left_on="_row", right_index=True, how="left"
                )
        if "_row" in result.columns:
            result = result.drop(columns="_row")
        if len(result):  # geometry last, as in a vector layer
            result = result[
                [c for c in result.columns if c != "geometry"] + ["geometry"]
            ]

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


def _check_stats(stats) -> tuple[str, ...]:
    """Validate the requested statistics; keeps the given order, drops repeats."""
    if isinstance(stats, str):
        stats = (stats,)
    names = tuple(dict.fromkeys(stats))
    unknown = [s for s in names if s not in _STATS]
    if unknown or not names:
        raise ValueError(
            f"Unknown statistic(s) {unknown or list(stats)!r}; choose among {_STATS}."
        )
    return names


def _load_source(source, layer) -> tuple[gpd.GeoDataFrame, str]:
    """``(GeoDataFrame, label)`` from a path, a GeoDataFrame or a ``geoutils.Vector``."""
    if isinstance(source, gpd.GeoDataFrame):
        return source.copy(), "GeoDataFrame"
    if isinstance(source, (str, Path)):
        return gpd.read_file(Path(source), layer=layer), Path(source).name
    ds = getattr(source, "ds", None)  # geoutils.Vector
    if isinstance(ds, gpd.GeoDataFrame):
        return ds.copy(), "Vector"
    raise TypeError(
        "source must be a path or a GeoDataFrame, got " f"{type(source).__name__}"
    )


def _attribute_frame(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """The non-geometry columns of the input, positionally indexed, safe to merge.

    ``point_id`` is dropped (it is the identifier already written), and any
    column named like one of the extraction's own columns is renamed
    ``src_<name>``.
    """
    attrs = pd.DataFrame(gdf.drop(columns=gdf.geometry.name)).reset_index(drop=True)
    attrs = attrs.drop(columns=[c for c in ("point_id",) if c in attrs.columns])
    return attrs.rename(
        columns={c: f"src_{c}" for c in attrs.columns if c in _RESERVED_COLUMNS}
    )


def _extract_zonal(
    zone_geoms: list,
    raster_path: Path,
    point_ids: list,
    date_fmt: str,
    comp: str,
    stats: tuple[str, ...] = tuple(_STATS),
) -> list[dict]:
    """Extract zonal statistics from a raster for each zone geometry.

    Rows carry ``_row`` (the position in the input layer), not a geometry: the
    caller attaches the original one.
    """
    zs = rasterstats.zonal_stats(
        zone_geoms,
        str(raster_path),
        stats=list(stats),
        all_touched=False,
    )
    rows = []
    for i, stat in enumerate(zs):
        row = {
            "_row": i,
            "point_id": point_ids[i],
            "date": date_fmt,
            "component": comp,
        }
        row.update({name: stat.get(name) for name in stats})
        rows.append(row)
    return rows


def _extract_points(
    point_geoms: list,
    raster_path: Path,
    point_ids: list,
    date_fmt: str,
    comp: str,
    stats: tuple[str, ...] = tuple(_STATS),
) -> list[dict]:
    """Sample raster at point locations (nearest pixel, no buffer)."""
    values = rasterstats.point_query(
        point_geoms,
        str(raster_path),
        interpolate="nearest",
    )
    rows = []
    for i, val in enumerate(values):
        v = float(val) if val is not None else np.nan
        is_finite = np.isfinite(v)
        every = {
            "mean": v,
            "median": v,
            "min": v,
            "max": v,
            "std": 0.0 if is_finite else np.nan,
            "count": 1 if is_finite else 0,
        }
        row = {
            "_row": i,
            "point_id": point_ids[i],
            "date": date_fmt,
            "component": comp,
        }
        row.update({name: every[name] for name in stats})
        rows.append(row)
    return rows


def _to_wide(long_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Pivot long-format extraction result to wide format.

    Output columns: ``point_id``, the attributes carried from the input layer
    (if any), then ``{component}_{YYYY-MM-DD}_{stat}`` for each (component,
    date, stat) combination sorted by component → date → stat, then geometry.
    One row per ``point_id`` (the first one, if an id repeats).
    """
    stat_cols = [s for s in _STATS if s in long_gdf.columns]
    attr_cols = [
        c for c in long_gdf.columns if c not in _RESERVED_COLUMNS and c != "geometry"
    ]
    first = long_gdf.drop_duplicates("point_id").set_index("point_id")

    # Build a plain DataFrame from the long-format records
    df = pd.DataFrame(long_gdf.drop(columns="geometry"))

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
    if attr_cols:
        wide_df = pd.concat([first.loc[wide_df.index, attr_cols], wide_df], axis=1)

    wide_gdf = gpd.GeoDataFrame(
        wide_df.reset_index(),
        geometry=first.loc[wide_df.index, "geometry"].values,
        crs=long_gdf.crs,
    )
    return wide_gdf
