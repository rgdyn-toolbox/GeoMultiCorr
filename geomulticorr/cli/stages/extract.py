"""``gmc extract`` — ASP disparity → metric EW/NS/CC rasters + raw stats."""

from __future__ import annotations

from geomulticorr.cli.config import forwarded
from geomulticorr.cli.stages._common import (
    StageContext,
    log_section,
    median,
    pairs_table,
    pz_criterias,
)


def run(ctx: StageContext, section, rec) -> None:
    session = ctx.open_session()
    image_type = ctx.image_type("extract")
    names = ctx.pz_names()
    kwargs = forwarded(section, exclude={"criterias"})

    per_pz = {}
    for pz in names:
        log_section("extract", pz)
        crit = pz_criterias(pz, section.criterias)
        results = (
            session.extract_pairs_raw_displacements(
                criterias=crit, image_type=image_type, **kwargs
            )
            or {}
        )
        failed = sorted(k for k, v in results.items() if v is None)
        rows = pairs_table(session, crit, image_type)
        per_pz[pz] = {
            "n_pairs": len(results),
            "n_failed": len(failed),
            "failed": failed,
            "median_ew_nmad_raw": median(r.get("ew_nmad_raw") for r in rows),
            "median_ns_nmad_raw": median(r.get("ns_nmad_raw") for r in rows),
            "median_cc_quality050": median(r.get("cc_quality050") for r in rows),
        }
        if failed:
            rec.status = "partial"
        rec.pairs.extend(rows)

    rec.pzones = names
    rec.summary["image_type"] = image_type
    rec.summary["pzones"] = per_pz
