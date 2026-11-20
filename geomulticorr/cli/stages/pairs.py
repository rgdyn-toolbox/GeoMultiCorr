"""``gmc pairs`` — commit the pair network, one ``update_pairs`` call per group."""

from __future__ import annotations

from geomulticorr._logging import logger
from geomulticorr.cli.stages._common import (
    StageContext,
    add_figure_paths,
    clean,
    log_section,
    pairs_table,
    pz_criterias,
    thumbs_table,
)


def run(ctx: StageContext, section, rec) -> None:
    session = ctx.open_session()
    image_type = ctx.image_type("pairs")
    names = ctx.pz_names()

    first_call = True
    per_pz = {}
    if not section.groups:
        logger.info("pairs: no group — the committed pairs are kept and reported")
    for pz in names:
        log_section("pairs", pz)
        for group in section.groups:
            kwargs = group.model_dump()
            if not first_call and not kwargs["append"]:
                # `append=False` overwrites the whole Pairs layer: honoured for
                # the first call only, or the stage would delete what it just
                # created for the previous group or pzone.
                logger.info("pairs: later groups append to the pairs just written")
                kwargs["append"] = True
            session.update_pairs(pz_name=pz, image_type=image_type, **kwargs)
            first_call = False

        for view in section.figures:
            written = session.save_pairs_figure(
                view,
                criterias=pz_criterias(pz),
                formats=tuple(section.formats),
                pz_name=pz,
            )
            add_figure_paths(rec, ctx, f"{pz}/pairs_{view}", written)

        per_pz[pz] = _network_summary(session, pz, image_type)
        rec.pairs.extend(pairs_table(session, pz_criterias(pz), image_type))

    rec.pzones = names
    rec.summary["image_type"] = image_type
    rec.summary["pzones"] = per_pz


def _network_summary(session, pz: str, image_type: str) -> dict:
    from geomulticorr.inversion.tio_inversion import connected_date_components
    from geomulticorr.utils._pairs_frame import pairs_frame_from_overview, pairs_stats

    overview = session.get_pairs_overview(pz_criterias(pz), image_type=image_type)
    if overview is None or not len(overview):
        return {"n_pairs": 0}
    stats = pairs_stats(pairs_frame_from_overview(overview))
    edges = [
        (str(r["pa_left_date"]), str(r["pa_right_date"]))
        for _, r in overview.iterrows()
    ]
    try:
        components = connected_date_components(edges)
        stats["n_components"] = len(components)
    except Exception as exc:  # connectivity is informative, never blocking
        logger.debug(f"pairs: connectivity not computed: {exc}")
    stats["thumbs"] = thumbs_table(session, pz, image_type)
    return {k: clean(v) for k, v in stats.items()}
