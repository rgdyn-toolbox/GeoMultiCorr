#!/usr/bin/env python3
"""Pinpoint an M2M 403: token, roles, or dataset access.

Run from the GeoMultiCorr project root (so .env is picked up):

    python diagnose_m2m.py
"""
import os

from dotenv import load_dotenv
load_dotenv()

from geomulticorr.data_sources import M2MClient, M2MImageFinder
from shapely.geometry import box

client = M2MClient(
    username=os.environ.get("USGS_M2M_USERNAME"),
    api_token=os.environ.get("USGS_M2M_TOKEN"),
)

print("1. login ..............", end=" ")
try:
    client.login()
    print(f"OK  (token {client.session_token[:8]}…)")
except Exception as e:
    raise SystemExit(f"FAILED\n   {e}")

print("2. permissions ........", end=" ")
try:
    perms = client.permissions()
    print(perms)
    if "download" not in perms:
        print("\n   >>> 'download' is NOT granted. This is the cause of the 403.")
        print("   >>> Request it at https://ers.cr.usgs.gov/profile/access")
        print("   >>> Searching works without it; download-options does not.")
    else:
        print("   'download' granted — the 403 is not a missing role.")
except Exception as e:
    print(f"FAILED\n   {e}")
    perms = []

print("3. scene-search .......", end=" ")
aoi = box(-69.772323, -33.211749, -69.328401, -32.865616)
finder = M2MImageFinder(client, aoi=aoi, year_range=(2023, 2023), cloud_max=50)
try:
    gdf = finder.search_metadata(collections=["L8_L1"])
    print(f"OK  ({len(gdf)} scenes)")
except Exception as e:
    raise SystemExit(f"FAILED\n   {e}")

if gdf.empty:
    raise SystemExit("No scenes to test download-options with.")

# One scene only: isolates authorisation from any batch-size effect.
print("4. download-options x1 .", end=" ")
one = gdf.iloc[0]
try:
    opts = client.download_options("landsat_ot_c2_l1", [one["image_id"]])
    avail = [o.get("productName") for o in opts if o.get("available")]
    print(f"OK  ({len(opts)} products, {len(avail)} available)")
    print("\n   Available products for", one["display_id"], ":")
    for name in avail:
        print(f"     - {name}")
    b8 = [n for n in avail if n and "band 8" in n.lower()]
    print(f"\n   Band 8 offered separately: {bool(b8)}  {b8}")
except Exception as e:
    print("FAILED")
    print(f"   {e}")
    print("\n   Single-scene request also fails => authorisation, not batch size.")
    raise SystemExit(1)

print("\n5. download-options x100", end=" ")
try:
    batch = list(gdf["image_id"][:100])
    opts = client.download_options("landsat_ot_c2_l1", batch)
    print(f"OK  ({len(opts)} products for {len(batch)} scenes)")
    print("\nAll checks passed — retry the notebook download cell.")
except Exception as e:
    print("FAILED")
    print(f"   {e}")
    print("\n   Single scene works but 100 does not => reduce M2M_BATCH (try 20).")

client.logout()
