"""``gmc correct`` — stable-ground masks + correction pipeline, per pair."""

from __future__ import annotations

from geomulticorr.cli.registry import (
    AUTO,
    build_correction_pipeline,
    build_filter_pipeline,
    describe_steps,
)
from geomulticorr.cli.stages._common import (
    StageContext,
    log_section,
    median,
    pairs_table,
    pz_criterias,
)


def moving_areas_resolver(session, pz: str):
    """``{StableAreaMask: auto}`` → the pzone's stored moving-area outlines."""

    def resolve(name: str):
        areas = session.get_moving_areas(pz)
        if areas is None:
            raise ValueError(
                f"{name}: 'auto' but pzone {pz!r} has no moving areas; "
                "load them in the 'prepare' stage (moving_areas:)"
            )
        return areas

    return resolve


def resolve_options(session, pz: str, options: dict) -> dict:
    """``dem: auto`` → the pzone's reference DEM."""
    out = dict(options)
    if out.get("dem") == AUTO:
        out["dem"] = session.get_pzones(pz)[0].get_dem()
    return out


def run(ctx: StageContext, section, rec) -> None:
    session = ctx.open_session()
    image_type = ctx.image_type("correct")
    names = ctx.pz_names()
    corrections = build_correction_pipeline(section.corrections)

    per_pz = {}
    for pz in names:
        log_section("correct", pz)
        crit = pz_criterias(pz, section.criterias)
        masks = build_filter_pipeline(
            section.masks, resolve_auto=moving_areas_resolver(session, pz)
        )
        results = (
            session.apply_pairs_corrections(
                corrections,
                masks,
                criterias=crit,
                overwrite=section.overwrite,
                save_plot=section.save_plot,
                plot_level=section.plot_level,
                verbose=section.verbose,
                image_type=image_type,
                **resolve_options(session, pz, section.options),
            )
            or {}
        )
        failed = sorted(k for k, v in results.items() if not v)
        rows = pairs_table(session, crit, image_type)
        per_pz[pz] = {
            "n_pairs": len(results),
            "n_failed": len(failed),
            "failed": failed,
            **_nmad_summary(rows),
        }
        if failed:
            rec.status = "partial"
        rec.pairs.extend(rows)

    rec.pzones = names
    rec.summary["image_type"] = image_type
    rec.summary["masks"] = section.masks
    rec.summary["corrections"] = describe_steps(corrections)
    rec.summary["pzones"] = per_pz


def _nmad_summary(rows: list[dict]) -> dict:
    out = {}
    for comp in ("ew", "ns"):
        raw = median(r.get(f"{comp}_nmad_raw") for r in rows)
        corr = median(r.get(f"{comp}_nmad_corr") for r in rows)
        out[f"median_{comp}_nmad_raw"] = raw
        out[f"median_{comp}_nmad_corr"] = corr
        ratios = [
            r[f"{comp}_nmad_corr"] / r[f"{comp}_nmad_raw"]
            for r in rows
            if r.get(f"{comp}_nmad_raw") and r.get(f"{comp}_nmad_corr") is not None
        ]
        out[f"median_{comp}_nmad_ratio"] = median(ratios)
    return out
