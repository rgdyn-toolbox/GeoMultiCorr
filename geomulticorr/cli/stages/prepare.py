"""``gmc prepare`` — create the project, its pzones and their reference layers."""

from __future__ import annotations

from geomulticorr._logging import logger
from geomulticorr.cli.stages._common import StageContext, clean, log_section


def _targets(spec, names: list[str]) -> list[str]:
    if spec.pz_name is None:
        return names
    if spec.pz_name not in names:
        raise ValueError(f"pz_name {spec.pz_name!r} is not a pzone of this project")
    return [spec.pz_name]


def run(ctx: StageContext, section, rec) -> None:
    session = ctx.open_session(create=True)
    existing = set(_existing(session))

    created = []
    for pz in section.pzones:
        if pz.name in existing:
            logger.info(f"pzone {pz.name!r} already exists — kept as is")
            continue
        log_section("prepare", pz.name)
        if pz.source is not None:
            session.insert_pzone_from_file(
                pz.source,
                pz.name,
                pz.shortname,
                layer=pz.layer,
                index=pz.index,
                dissolve=pz.dissolve,
            )
        elif pz.coordinates is not None:
            session.insert_pzone(pz.coordinates, pz.name, pz.shortname, crs=pz.crs)
        else:
            raise ValueError(f"pzone {pz.name!r}: give `source` or `coordinates`")
        created.append(pz.name)

    names = ctx.pz_names()
    for spec in section.reference_grid:
        for pz in _targets(spec, names):
            path = session.build_reference_grid(
                pz, resolution=spec.resolution, overwrite=spec.overwrite
            )
            rec.add_artifact(f"{pz}/reference_grid", path, ctx.project_root)
    for spec in section.reference_dem:
        for pz in _targets(spec, names):
            path = session.build_reference_dem(
                pz, spec.source, resampling=spec.resampling, overwrite=spec.overwrite
            )
            rec.add_artifact(f"{pz}/reference_dem", path, ctx.project_root)
    for spec in section.moving_areas:
        for pz in _targets(spec, names):
            path = session.load_moving_areas(
                pz,
                spec.source,
                layer=spec.layer,
                buffer=spec.buffer,
                clip=spec.clip,
                overwrite=spec.overwrite,
            )
            rec.add_artifact(f"{pz}/moving_areas", path, ctx.project_root)

    rec.pzones = names
    rec.summary.update(
        {
            "project": str(ctx.project_root),
            "n_pzones": len(names),
            "created": created,
            "pzones": _pzone_rows(session, names),
        }
    )


def _existing(session) -> list[str]:
    overview = session.get_pzones_overview()
    if overview is None or not len(overview):
        return []
    return [str(n) for n in overview["pz_name"]]


def _pzone_rows(session, names) -> list[dict]:
    overview = session.get_pzones_overview()
    if overview is None or not len(overview):
        return []
    rows = []
    for _, row in overview.iterrows():
        if str(row["pz_name"]) in names:
            rows.append(
                {
                    "pz_name": str(row["pz_name"]),
                    "pz_shortname": str(row.get("pz_shortname", "")),
                    "surface": clean(row.get("pz_surf")),
                }
            )
    return rows
