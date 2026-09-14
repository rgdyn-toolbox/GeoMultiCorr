#!/usr/bin/env python3
"""Collect M2M downloads already staged under a label, without re-requesting.

A `download-request` that timed out client-side still completed on the USGS
side: the files are staged and the quota is already spent. Re-running the
notebook would pay for them a second time. This picks them up instead.

    python resume_m2m.py                      # list what is queued
    python resume_m2m.py --label gmc_L8_L1    # fetch it
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from shapely.geometry import box

from geomulticorr.data_sources import M2MClient, M2MImageFinder

ARCHIVE_ROOT = Path("downloads_regional")
BBOX = (-69.772323, -33.211749, -69.328401, -32.865616)


def main(label: str | None, output_dir: Path) -> None:
    client = M2MClient(
        username=os.environ.get("USGS_M2M_USERNAME"),
        api_token=os.environ.get("USGS_M2M_TOKEN"),
    )
    client.login()

    limits = client.rate_limit_summary()
    for row in limits.get("remainingLimits", []):
        if row.get("limitType") == "user":
            print(f"quota: {row.get('recentDownloadCount')} recent downloads remaining")

    queued = client.download_search(label=label)
    if not queued:
        print(f"\nNothing queued{f' under {label!r}' if label else ''}.")
        client.logout()
        return

    by_label: dict[str, list] = {}
    for rec in queued:
        by_label.setdefault(rec.get("label") or "(none)", []).append(rec)

    print(f"\n{len(queued)} record(s) in the queue:")
    for lbl, recs in sorted(by_label.items()):
        size = sum(r.get("filesize") or 0 for r in recs) / 1e9
        kinds: dict[str, int] = {}
        for r in recs:
            name = r.get("displayId") or ""
            kinds[name.rsplit("_", 1)[-1]] = kinds.get(name.rsplit("_", 1)[-1], 0) + 1
        print(f"  {lbl:<14} {len(recs):>4} files, {size:5.1f} GB   {kinds}")

    if label is None:
        print("\nPass --label <name> to fetch one of these.")
        client.logout()
        return

    finder = M2MImageFinder(client, aoi=box(*BBOX))
    print(f"\nFetching {label!r} into {output_dir} ...")
    paths = finder.resume_label(
        label,
        output_dir=output_dir,
        keep_original_names=True,
        organize_by="sensor_pathrow",
        show_progress=True,
    )
    print(f"\n{len(paths)} file(s) written.")
    for p in paths[:5]:
        print(f"  {p.relative_to(output_dir)}")
    if len(paths) > 5:
        print(f"  ... {len(paths) - 5} more")
    client.logout()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="e.g. gmc_L8_L1")
    ap.add_argument("--output-dir", type=Path, default=ARCHIVE_ROOT)
    args = ap.parse_args()
    main(args.label, args.output_dir)
