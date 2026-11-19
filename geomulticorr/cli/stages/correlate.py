"""``gmc correlate`` — correlation plan → ASP scripts → launch (or cluster sync)."""

from __future__ import annotations

from collections import Counter

from geomulticorr.cli.stages._common import (
    StageContext,
    accepted,
    add_figure_paths,
    check_kwargs,
    log_section,
    pairs_table,
    pz_criterias,
)


def run(ctx: StageContext, section, rec) -> None:
    session = ctx.open_session()
    image_type = ctx.image_type("correlate")
    names = ctx.pz_names()
    check_kwargs(
        session.prepare_pairs_correlation, section.options, "correlate.options"
    )

    per_pz = {}
    for pz in names:
        log_section("correlate", pz)
        crit = pz_criterias(pz, section.criterias)
        entry: dict = {}

        if section.sync:
            session.sync_pairs_after_cluster(
                criterias=crit, asp_bin_dir=section.asp_bin_dir, image_type=image_type
            )
            entry["synced"] = True
        else:
            plan = _plan(ctx, session, section, pz, crit, image_type, rec)
            common = {
                "criterias": crit,
                "cluster": section.cluster,
                "asp_bin_dir": section.asp_bin_dir,
                "overwrite": section.overwrite,
                "overwrite_scripts": section.overwrite_scripts,
                "nodes": section.nodes,
                "cores": section.cores,
                "walltime": section.walltime,
                "correval": section.correval,
                "image_type": image_type,
            }
            scripts = session.prepare_pairs_correlation(
                plan=plan, **common, **section.options
            )
            entry["n_scripts"] = len(scripts or [])
            if section.launch:
                launch = session.launch_pairs_correlation
                codes = launch(
                    dry_run=section.dry_run,
                    **accepted(launch, {**common, **section.options}),
                )
                codes = list(codes or [])
                entry["n_launched"] = len(codes)
                entry["n_failed"] = sum(1 for c in codes if c != 0)
                if entry["n_failed"]:
                    rec.status = "partial"

        rows = pairs_table(session, crit, image_type)
        entry["status_counts"] = dict(Counter(str(r.get("status")) for r in rows))
        per_pz[pz] = entry
        rec.pairs.extend(rows)

    rec.pzones = names
    rec.summary["image_type"] = image_type
    rec.summary["pzones"] = per_pz


def _plan(ctx, session, section, pz, crit, image_type, rec):
    """The plan to prepare with: a given file, a derived one, or none."""
    from geomulticorr.correlation._correlation_plan import plan_path

    if section.plan is not None:
        rec.add_artifact(f"{pz}/correlation_plan", section.plan, ctx.project_root)
        return str(section.plan)
    if section.explore is None:
        return None
    frame, _ = session.explore_correlation_params(
        criterias=crit,
        pz_name=pz,
        interactive=False,
        savefig=False,
        plan_name=section.plan_name,
        image_type=image_type,
        **section.explore,
    )
    plan = session._last_corr_params["plan"]
    if frame is not None and len(frame):
        for view in ("design_map", "kernel_search"):
            written = session.save_correlation_figure(
                view=view, frame=frame, formats=("png",), pz_name=pz
            )
            add_figure_paths(rec, ctx, f"{pz}/correlation_{view}", written)
    directory = session.pz_dir(pz, "image_correlation")
    rec.add_artifact(
        f"{pz}/correlation_plan",
        plan_path(directory, section.plan_name),
        ctx.project_root,
    )
    rec.summary.setdefault("plans", {})[pz] = _plan_groups(plan)
    return plan


def _plan_groups(plan: dict) -> list[dict]:
    """Group rows of a plan for the report: key, pair count, kernel, search."""
    groups = (plan or {}).get("groups") or {}
    rows = []
    items = groups.items() if isinstance(groups, dict) else enumerate(groups)
    for key, group in items:
        if not isinstance(group, dict):
            continue
        params = group.get("params", group)
        rows.append(
            {
                "group": str(key),
                "n_pairs": group.get("n_pairs"),
                "corr_kernel": params.get("corr_kernel"),
                "corr_search": params.get("corr_search"),
                "corr_seed_mode": params.get("corr_seed_mode"),
            }
        )
    return rows
