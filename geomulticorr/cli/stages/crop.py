"""``gmc crop`` — sieve a raw image bank onto every pzone and register the thumbs."""

from __future__ import annotations

from collections import Counter

from geomulticorr.cli.config import forwarded
from geomulticorr.cli.stages._common import (
    StageContext,
    log_section,
    median,
    thumbs_table,
)


def run(ctx: StageContext, section, rec) -> None:
    session = ctx.open_session()
    image_type = ctx.image_type("crop")
    names = ctx.pz_names()

    if section.georasters_bank_path is not None:
        log_section("crop", "map + sieve")
        map_kwargs = {k: getattr(section, k) for k in section.MAP_KEYS}
        georasters = session.map_georasters_bank(
            section.georasters_bank_path,
            image_type=image_type,
            suffix=section.suffix,
            **map_kwargs,
        )
        sieve_kwargs = forwarded(
            section, exclude={"georasters_bank_path", *section.MAP_KEYS}
        )
        written = session.sieve_bulk(
            georasters_map=georasters,
            image_type=image_type,
            pzone_filter=ctx.cfg.pzones,
            **sieve_kwargs,
        )
        rec.summary["n_sources"] = int(len(georasters)) if georasters is not None else 0
        rec.summary["written"] = {str(k): len(v) for k, v in (written or {}).items()}
        session.update_thumbs()

    for spec in section.register_thumbs:
        log_section("crop", f"register → {spec.pz_name}")
        paths = session.register_existing_thumbs(**spec.model_dump())
        rec.summary.setdefault("registered", {})[spec.pz_name] = len(paths or [])

    if section.backfill_resolution:
        for pz in names:
            session.backfill_thumb_resolution(pz)

    thumbs = [row for pz in names for row in thumbs_table(session, pz, image_type)]
    rec.pzones = names
    rec.summary.update(_thumb_summary(thumbs))
    rec.summary["image_type"] = image_type
    rec.summary["thumbs"] = thumbs


def _thumb_summary(thumbs: list[dict]) -> dict:
    by_sensor = Counter(t["sensor"] for t in thumbs)
    dates = [t["date"] for t in thumbs]
    return {
        "n_thumbs": len(thumbs),
        "thumbs_per_sensor": dict(by_sensor),
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
        "median_res_m": median(t["res"] for t in thumbs),
    }
