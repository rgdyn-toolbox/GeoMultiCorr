"""Every GMC stage on one real site, one project per sensor (same-sensor pairs only).

Each test starts from a stored stage output and compares with the next one by tolerance,
never byte equality. ASP is needed only by the test marked ``asp``. Tests of stages a
project does not hold yet (no moving-area outline, so no corrections) skip.
"""

from __future__ import annotations

import hashlib
import shutil

import geoutils as gu
import numpy as np
import pytest
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box

from geomulticorr.core.session import open_gmc_session
from geomulticorr.corrections import (
    CCFilter,
    MedianCentering,
    OutlierFilter,
    RampCorrection,
    StableAreaMask,
    make_corrections,
)
from geomulticorr.inversion.pytio.config import TIOConfig
from geomulticorr.inversion.tio_inversion import TIOInversion
from geomulticorr.similarity.selection import sensor_family

from .conftest import (
    PROJECTS,
    SAMPLE_POINTS,
    SITE,
    load_asset,
    nmad,
    read_band,
    read_manifest,
)

SENSORS = list(PROJECTS)
CC_THRESHOLD = 0.5
OUTLIER_BOUNDS_M = (-500, 500)

#: Sensors whose corrected maps show the rock glacier: median speed over the moving areas
#: must exceed this multiple of the stable-ground median, in every forward pair. Measured on
#: the v2 data: SPOT 2.4-3.2, Planet 1.3-2.2; Sentinel-2 and Landsat 0.7-1.1 (no signal).
MOTION_RATIO = {"spot": 2.0, "planetscope": 1.2}


def pair_dirs(root, pz):
    return sorted(p for p in (root / pz / "image_correlation").iterdir() if p.is_dir())


def require_stage(folder, suffix):
    if not (folder / f"{folder.name}{suffix}").exists():
        pytest.skip(f"this project holds no {suffix} yet")


def moving_mask(session, pz, reference_path):
    with rasterio.open(reference_path) as src:
        shape, transform, crs = src.shape, src.transform, src.crs
    outlines = session.get_moving_areas(pz).ds.to_crs(crs)
    return geometry_mask(outlines.geometry, shape, transform, invert=True)


@pytest.mark.parametrize("sensor", SENSORS)
@pytest.mark.parametrize("kind", ["project", "bank"])
def test_files_match_their_manifest(sensor, kind):
    name = PROJECTS[sensor] if kind == "project" else f"bank_{SITE}_{sensor}"
    root = load_asset(name)
    for relative, entry in read_manifest(root)["files"].items():
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert digest == entry["sha256"], relative


@pytest.mark.parametrize("sensor", SENSORS)
def test_project_opens_relocated_with_every_pair_complete(
    project_from, sensor, tmp_path
):
    session, cached, manifest = project_from(sensor)
    thumbs = session.get_thumbs()
    pairs = session.get_pairs()
    assert len(thumbs) >= 3 and len(pairs) > 0
    assert all(str(tmp_path) in str(t.th_path) for t in thumbs)
    assert {p.pa_status for p in pairs} == {"complete"}
    for pair in pairs:
        left, right = pair.pa_left.th_sensor, pair.pa_right.th_sensor
        assert sensor_family(left) == sensor_family(right), pair.pa_key


@pytest.mark.parametrize("sensor", SENSORS)
def test_bank_reproduces_the_project_thumbs(project_from, sensor, tmp_path):
    """map_georasters_bank + sieve_bulk on the bank give the stored thumbs back."""
    _, cached, manifest = project_from(sensor)
    if manifest.get("thumbs_are_placeholders"):
        pytest.skip("results-only project: its thumbs are placeholders")
    bank = load_asset(f"bank_{SITE}_{sensor}") / sensor
    (tmp_path / "rebuild").mkdir()
    fresh = open_gmc_session(
        tmp_path / "rebuild" / manifest["project"], epsg_code=manifest["epsg"]
    )

    pz = manifest["pz_name"]
    fresh.insert_pzone(box(*manifest["aoi_bounds"]), pz, "TST", crs=manifest["epsg"])
    fresh.map_georasters_bank(bank)
    fresh.sieve_bulk(
        target_resolution=manifest["resolution_m"],
        selected_band=manifest["selected_band"],
        pzone_filter=[pz],
        print_summary=False,
    )
    stored = sorted((cached / pz / "optical").glob("*.tif"))
    rebuilt = sorted(fresh.pz_dir(pz, "optical").glob("*.tif"))
    assert [p.name for p in rebuilt] == [p.name for p in stored]
    for new, old in zip(rebuilt, stored):
        with rasterio.open(new) as a, rasterio.open(old) as b:
            assert a.transform == b.transform and a.shape == b.shape and a.crs == b.crs
            assert np.allclose(a.read(1), b.read(1), equal_nan=True)


@pytest.mark.parametrize("sensor", SENSORS)
def test_extraction_reproduces_the_stored_maps(project_from, sensor):
    session, cached, manifest = project_from(sensor)
    pz = manifest["pz_name"]
    pair = session.get_pairs()[0]
    results = session.extract_pairs_raw_displacements(
        criterias=pair.pa_key, overwrite=True, save_plot=False, print_summary=False
    )
    assert results[pair.pa_key]
    stored = cached / pz / "image_correlation" / pair.pa_key
    for suffix in ("-F_EW.tif", "-F_NS.tif"):
        new = read_band(pair.pa_path / f"{pair.pa_key}{suffix}")
        old = read_band(stored / f"{pair.pa_key}{suffix}")
        assert new.shape == old.shape
        assert abs(np.nanmedian(new) - np.nanmedian(old)) < 1e-3
        assert abs(nmad(new) - nmad(old)) < 1e-3


@pytest.mark.parametrize("sensor", SENSORS)
def test_reverse_pair_measures_the_opposite_motion(sensor):
    root = load_asset(PROJECTS[sensor])
    manifest = read_manifest(root)
    folders = {f.name: f for f in pair_dirs(root, manifest["pz_name"])}
    checked = 0
    for key, folder in folders.items():
        pz, rest = key.split("_", 1)
        left, right = rest.split("__")
        reverse = f"{pz}_{right}__{left}"
        if reverse not in folders or key > reverse:
            continue
        forward = read_band(folder / f"{key}-F_EW.tif")
        backward = read_band(folders[reverse] / f"{reverse}-F_EW.tif")
        assert abs(np.nanmedian(forward + backward)) < 0.3 * manifest["resolution_m"]
        checked += 1
    assert checked > 0


def _stable_path(folder, key, comp):
    """Fit-mask file of a pair: ``-F_<comp>_stable.tif``, or the old ``-F_<comp>mask.tif``."""
    new = folder / f"{key}-F_{comp}_stable.tif"
    return new if new.exists() else folder / f"{key}-F_{comp}mask.tif"


@pytest.mark.parametrize("sensor", SENSORS)
def test_corrections_reproduce_and_center_stable_ground(sensor):
    root = load_asset(PROJECTS[sensor])
    manifest = read_manifest(root)
    folder = pair_dirs(root, manifest["pz_name"])[0]
    require_stage(folder, "-F_EW_corr.tif")
    key = folder.name
    x_raw = gu.Raster(str(folder / f"{key}-F_EW.tif"), nodata=0)
    y_raw = gu.Raster(str(folder / f"{key}-F_NS.tif"), nodata=0)
    cc = gu.Raster(str(folder / f"{key}-F_CC.tif"))
    # the pinned dataset (v2) predates the -F_*_stable.tif names
    x_stable = read_band(_stable_path(folder, key, "EW")) == 1
    y_stable = read_band(_stable_path(folder, key, "NS")) == 1
    x_kept, y_kept = (CCFilter(CC_THRESHOLD) + OutlierFilter(OUTLIER_BOUNDS_M)).apply(
        x_raw, y_raw, cc=cc
    )
    x_corr, y_corr = make_corrections(
        x_kept,
        y_kept,
        MedianCentering() + RampCorrection(),
        x_stable=x_stable,
        y_stable=y_stable,
    )
    stored = read_band(folder / f"{key}-F_EW_corr.tif")
    computed = np.ma.filled(x_corr.data.astype(float), np.nan)
    assert abs(np.nanmedian(computed) - np.nanmedian(stored)) < 1e-2
    for corrected, stable in ((x_corr, x_stable), (y_corr, y_stable)):
        values = np.ma.filled(corrected.data.astype(float), np.nan)[stable]
        assert abs(np.nanmedian(values)) < 0.2 * manifest["resolution_m"]


@pytest.mark.parametrize("sensor", sorted(MOTION_RATIO))
def test_moving_areas_move_faster_than_stable_ground(project_from, sensor):
    session, cached, manifest = project_from(sensor)
    pz = manifest["pz_name"]
    forward = [
        f
        for f in pair_dirs(cached, pz)
        if f.name.split("__")[0].split("_")[1] < f.name.split("__")[1].split("_")[0]
    ]
    assert forward
    for folder in forward:
        require_stage(folder, "-F_EW_corr.tif")
        ew = read_band(folder / f"{folder.name}-F_EW_corr.tif")
        ns = read_band(folder / f"{folder.name}-F_NS_corr.tif")
        speed = np.hypot(ew, ns)
        moving = moving_mask(session, pz, folder / f"{folder.name}-F_EW_corr.tif")
        ratio = np.nanmedian(speed[moving]) / np.nanmedian(speed[~moving])
        assert ratio > MOTION_RATIO[sensor], f"{folder.name}: ratio {ratio:.2f}"


@pytest.mark.parametrize("sensor", SENSORS)
def test_inverted_series_shows_the_rock_glacier_flowing_north_west(sensor, tmp_path):
    """Every sensor sees the fastest point move, coarse ones included (measured on v2: SPOT
    40 m over 2014-2021, Planet 26 m, Landsat 19 m, Sentinel-2 10.5 m; stable 0-3.5 m).
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    from geomulticorr.stats.inversion_extractor import InversionExtractor

    cached = load_asset(PROJECTS[sensor])
    manifest = read_manifest(cached)
    inversion = cached / manifest["pz_name"] / "inversion" / "reference"
    if not (inversion / "inverse_EW").is_dir():
        pytest.skip("this project holds no inversion yet")
    points = tmp_path / "points.gpkg"
    gpd.GeoDataFrame(
        {"point_id": list(SAMPLE_POINTS)},
        geometry=[Point(xy) for xy in SAMPLE_POINTS.values()],
        crs=manifest["epsg"],
    ).to_file(points, driver="GPKG")

    frame = InversionExtractor(inversion, components=["EW", "NS"]).extract(
        points, buffer_m=30.0
    )
    frame["date"] = pd.to_datetime(frame["date"])
    wide = frame.pivot_table(
        index=["point_id", "date"], columns="component", values="mean"
    )
    shift = {}
    for point_id in SAMPLE_POINTS:
        series = wide.loc[point_id].sort_index()
        shift[point_id] = series.iloc[-1] - series.iloc[0]
    fast, stable = shift["rock_glacier"], shift["stable"]
    total = float(np.hypot(fast["EW"], fast["NS"]))
    assert fast["EW"] < 0 and fast["NS"] > 0, f"not north-west: {dict(fast)}"
    assert total > 5.0
    assert total > 5 * float(np.hypot(stable["EW"], stable["NS"]))


@pytest.mark.parametrize("sensor", SENSORS)
def test_inversion_reproduces_the_stored_series(project_from, sensor):
    session, cached, manifest = project_from(sensor)
    pz = manifest["pz_name"]
    stored = sorted(
        (cached / pz / "inversion" / "reference" / "inverse_EW").glob("TOT_*_EW.tif")
    )
    if not stored:
        pytest.skip("this project holds no inversion yet")
    inv = TIOInversion(
        session,
        session.get_pairs(),
        inversion_name="rerun",
        filter_pipeline=CCFilter(CC_THRESHOLD) + OutlierFilter(OUTLIER_BOUNDS_M),
        tio_binaries_dir=None,
    )
    inv.prepare_inversion(
        weight_mode="quality_spatial", solver=TIOConfig(), print_summary=False
    )
    inv.launch(mode="python")
    rerun = session.pz_dir(pz, "inversion") / "rerun" / "inverse_EW" / stored[-1].name
    difference = read_band(rerun) - read_band(stored[-1])
    assert np.isfinite(difference).any()
    assert np.nanpercentile(np.abs(difference), 99) < 0.05


def test_apply_pairs_corrections_keeps_moving_ground_with_stable_area_mask(
    project_from,
):
    session, cached, manifest = project_from("sentinel2")
    pz = manifest["pz_name"]
    try:
        session.get_moving_areas(pz)
    except FileNotFoundError:
        pytest.skip("this project holds no moving-area outline yet")
    pair = session.get_pairs()[0]
    results = session.apply_pairs_corrections(
        MedianCentering() + RampCorrection(),
        CCFilter(CC_THRESHOLD) + StableAreaMask(session.get_moving_areas(pz)),
        criterias=pair.pa_key,
        overwrite=True,
        save_plot=False,
        print_summary=False,
    )
    assert results[pair.pa_key]
    corrected = read_band(pair.pa_ew_corr_path)
    moving = moving_mask(session, pz, pair.pa_ew_corr_path)
    assert np.isfinite(corrected)[moving].mean() > 0.2


@pytest.mark.parametrize("sensor", SENSORS)
def test_same_sensor_similarity_runs(project_from, sensor):
    session, cached, manifest = project_from(sensor)
    if manifest.get("thumbs_are_placeholders"):
        pytest.skip("results-only project: its thumbs are placeholders")
    pz = manifest["pz_name"]
    table = session.compute_thumbs_similarity(
        pz, save=False, plots=False, progress=False, moving_areas=None
    )
    n = len(session.get_thumbs())
    assert len(table) == n * (n - 1) / 2
    assert table["cross_sensor"].eq(False).all()
    assert table["trial_peak_median"].notna().any()


@pytest.mark.asp
@pytest.mark.skipif(
    shutil.which("parallel_stereo") is None, reason="ASP is not installed"
)
def test_real_correlation_reproduces_the_stored_field(project_from):
    session, cached, manifest = project_from("landsat")
    pair = session.get_pairs()[0]
    stored = cached / manifest["pz_name"] / "image_correlation" / pair.pa_key
    for path in pair.pa_path.iterdir():
        path.unlink()
    correlation = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in manifest["correlation"].items()
    }
    session.prepare_pairs_correlation(
        criterias=pair.pa_key, print_summary=False, **correlation
    )
    session.launch_pairs_correlation(criterias=pair.pa_key, print_summary=False)
    session.extract_pairs_raw_displacements(
        criterias=pair.pa_key, save_plot=False, print_summary=False
    )
    new = read_band(pair.pa_ew_path)
    old = read_band(stored / f"{pair.pa_key}-F_EW.tif")
    both = np.isfinite(new) & np.isfinite(old)
    assert both.mean() > 0.5
    assert (
        abs(np.median(new[both]) - np.median(old[both]))
        < 0.1 * manifest["resolution_m"]
    )
    assert np.corrcoef(new[both], old[both])[0, 1] > 0.9
