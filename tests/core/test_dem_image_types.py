"""Multi-temporal DEMs end to end: bank → sieve (+ terrain products) → thumbs → pairs.

One real new-layout project is built per module (mapping and sieving open
rasters, so it is done once), then each test reads a different aspect of it.
The image type is carried by the folder — ``elevation/<type>/`` for thumbs and
``image_correlation/<type>/`` for pairs — while optical keeps its old paths.
"""
from __future__ import annotations

import os
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from geomulticorr import open_gmc_session
from geomulticorr.core.pair import Pair
from geomulticorr.core.session import (
    CORR_INPUT_DEFAULTS,
    Session,
    _corr_input_params,
)
from geomulticorr.core.thumb import Thumb
from geomulticorr.correlation._correlation_plan import (
    input_params_differ_from_file,
    parse_params_file,
)
from geomulticorr.utils._image_types import (
    image_type_of_thumb_path,
    normalise_image_type,
    pair_subdir,
    pzone_root_of_thumb_path,
    type_column,
)

EPSG = 32632
X0, Y0, RES, N = 500000.0, 5100000.0, 3.0, 60
PZ = "PZA"  # no "_": THUMBNAME_PATTERN forbids it in the pzone segment
DEMS = {
    "pleiades_2019-08-14_dem.tif": 0,
    "pleiades_2020-08-20_dem.tif": 1,   # shifted by one pixel eastwards
    "uav_2021-08-03.tif": 2,
}


def _surface(shift_px: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    rows, cols = np.mgrid[0:N, 0:N + 4].astype(float)
    hill = 80.0 * np.exp(-(((rows - 30) / 18) ** 2 + ((cols - 32) / 18) ** 2))
    texture = rng.normal(scale=0.8, size=hill.shape)
    z = 2000.0 + hill + texture
    return z[:, 2 - min(shift_px, 2): 2 - min(shift_px, 2) + N]


def _write_dem(path, shift_px):
    z = _surface(shift_px).astype("float32")
    z[40, 40] = -9999.0  # a void, as real DEMs have
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=1,
                       dtype="float32", crs=f"EPSG:{EPSG}", nodata=-9999.0,
                       transform=from_origin(X0, Y0, RES, RES)) as dst:
        dst.write(z, 1)


def _prepare(session, **kwargs):
    """prepare_pairs_correlation with ASP's binary lookup and script writer faked."""
    def _fake_write(pair, **kw):
        path = pair.pa_path / f"{pair.pa_key}_CorrelationJob.sh"
        path.write_text("# fake\n")
        return path

    with patch("shutil.which", return_value="/fake/parallel_stereo"), patch(
        "geomulticorr.correlation.correlation.ASP.write_bash_script", side_effect=_fake_write,
    ):
        return session.prepare_pairs_correlation(print_summary=False, **kwargs)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("demproj")
    bank = root / "bank"
    bank.mkdir()
    for name, shift in DEMS.items():
        _write_dem(bank / name, shift)

    session = open_gmc_session(root / "proj", epsg_code=EPSG)
    session.insert_pzone([(X0 + 30, Y0 - 150), (X0 + 150, Y0 - 150),
                          (X0 + 150, Y0 - 30), (X0 + 30, Y0 - 30)], PZ, "A")
    bank_map = session.map_georasters_bank(bank, image_type="dem", epsg_code=EPSG,
                                           max_workers=1)
    written = session.sieve_bulk(image_type="dem", target_resolution=RES,
                                 terrain_products=("hillshade", "mdhillshade", "slope"),
                                 print_summary=False)
    session.update_thumbs()
    session.validate_all_thumbs()
    return session, root, bank, bank_map, written


# --------------------------------------------------------------------------- #
class TestImageTypeTable:
    def test_folder_decides_the_type(self, tmp_path):
        assert image_type_of_thumb_path(tmp_path / "p/optical/p_2020-01-01_s.tif") == "opt"
        assert image_type_of_thumb_path(tmp_path / "p/opticals/p_2020-01-01_s.tif") == "opt"
        assert image_type_of_thumb_path(tmp_path / "p/elevation/slope/x.tif") == "slope"
        # a type-named folder outside elevation/ is not a type
        assert image_type_of_thumb_path(tmp_path / "p/slope/x.tif") == "opt"

    def test_pzone_root(self, tmp_path):
        assert pzone_root_of_thumb_path(tmp_path / "p/optical/x.tif") == tmp_path / "p"
        assert pzone_root_of_thumb_path(tmp_path / "p/elevation/dem/x.tif") == tmp_path / "p"

    def test_pair_subdir_is_empty_for_optical(self):
        assert pair_subdir("opt") == ""
        assert pair_subdir("mdhillshade") == "mdhillshade"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="image_type"):
            normalise_image_type("aspect")
        assert normalise_image_type(None) == "opt"
        assert normalise_image_type(" Slope ") == "slope"

    def test_rows_predating_the_column_read_as_optical(self):
        old = pd.DataFrame({"th_path": ["a", "b"]})
        assert type_column(old, "th_type").tolist() == ["opt", "opt"]
        mixed = pd.DataFrame({"th_type": ["slope", None]})
        assert type_column(mixed, "th_type").tolist() == ["slope", "opt"]


# --------------------------------------------------------------------------- #
class TestBankMapping:
    def test_dem_sensor_inferred_and_dated(self, project):
        _, _, _, bank_map, _ = project
        by_name = bank_map.set_index("filename")
        assert by_name.loc["pleiades_2019-08-14_dem.tif", "sensor"] == "pleiades"
        assert by_name.loc["uav_2021-08-03.tif", "sensor"] == "uav"
        assert set(bank_map["sensor_family"]) == {"dem"}
        assert set(bank_map["image_type"]) == {"dem"}

    def test_dem_map_does_not_overwrite_the_optical_one(self, project):
        session, *_ = project
        assert session._protomap_layer("opt") == "Protomap"
        assert session._protomap_layer("dem") == "Protomap_dem"
        assert len(session.get_georasters_map("dem")) == len(DEMS)
        with pytest.raises(ValueError, match="raster-bank"):
            session._protomap_layer("slope")


# --------------------------------------------------------------------------- #
class TestSieveDem:
    def test_every_product_lands_in_its_folder(self, project):
        session, root, *_ = project
        for t in ("dem", "hillshade", "mdhillshade", "slope"):
            names = sorted(p.name for p in session.thumb_dir(PZ, t).glob("*.tif"))
            assert names == [f"{PZ}_2019-08-14_pleiades.tif", f"{PZ}_2020-08-20_pleiades.tif",
                             f"{PZ}_2021-08-03_uav.tif"], t
            assert session.pair_dir(PZ, t).is_dir()
        # nothing leaked into the optical folder or the correction DEM
        assert not list(session.thumb_dir(PZ, "opt").glob("*.tif"))
        assert not list(session.pz_dir(PZ, "reference_dem").glob("*.tif"))

    def test_encodings(self, project):
        session, *_ = project
        name = f"{PZ}_2019-08-14_pleiades.tif"
        with rasterio.open(session.thumb_dir(PZ, "dem") / name) as ds:
            dem = ds.read(1)
            assert ds.dtypes[0] == "float32" and ds.nodata == -9999.0
            # elevations survived: a uint16 cast would have truncated them
            valid = dem[dem != -9999.0]
            assert valid.min() > 1990 and not np.allclose(valid, np.round(valid))
        for t in ("hillshade", "mdhillshade"):
            with rasterio.open(session.thumb_dir(PZ, t) / name) as ds:
                assert ds.dtypes[0] == "uint8" and ds.nodata == 0
                assert ds.read(1).max() > 1
        with rasterio.open(session.thumb_dir(PZ, "slope") / name) as ds:
            s = ds.read(1)
            assert ds.dtypes[0] == "float32" and ds.nodata == -9999.0
            assert 0 <= s[s != -9999.0].min() and s[s != -9999.0].max() < 90

    def test_voids_and_border_become_nodata(self, project):
        session, *_ = project
        name = f"{PZ}_2019-08-14_pleiades.tif"
        with rasterio.open(session.thumb_dir(PZ, "slope") / name) as ds:
            s = ds.read(1)
        assert (s[0] == -9999.0).all() and (s[:, -1] == -9999.0).all()
        assert (s == -9999.0).sum() > 2 * (s.shape[0] + s.shape[1])  # the void too

    def test_grid_shared_by_every_product_and_date(self, project):
        session, *_ = project
        grids = set()
        for t in ("dem", "hillshade", "mdhillshade", "slope"):
            for p in session.thumb_dir(PZ, t).glob("*.tif"):
                with rasterio.open(p) as ds:
                    grids.add((ds.shape, tuple(ds.transform), ds.crs.to_epsg()))
        assert len(grids) == 1

    def test_skip_existing_is_per_product(self, project):
        session, *_ = project
        name = f"{PZ}_2019-08-14_pleiades.tif"
        md = session.thumb_dir(PZ, "mdhillshade") / name
        slope = session.thumb_dir(PZ, "slope") / name
        before = md.stat().st_mtime_ns
        slope_bytes = slope.read_bytes()
        slope.unlink()
        session.sieve_bulk(image_type="dem", target_resolution=RES,
                           terrain_products=("hillshade", "mdhillshade", "slope"),
                           print_summary=False)
        assert slope.read_bytes() == slope_bytes     # recomputed identically
        assert md.stat().st_mtime_ns == before       # untouched

    def test_bad_products_raise(self, project):
        session, *_ = project
        with pytest.raises(ValueError, match="terrain product"):
            session.sieve_bulk(image_type="dem", terrain_products=("aspect",))
        with pytest.raises(ValueError, match="Nothing to write"):
            session.sieve_bulk(image_type="dem", terrain_products=(), keep_dem=False)
        with pytest.raises(ValueError, match="image_type"):
            session.sieve_bulk(image_type="slope")

    def test_reference_dem_keeps_the_old_single_file(self, project):
        session, *_ = project
        session.sieve_bulk(image_type="reference_dem", target_resolution=RES,
                           out_dtype="float32", print_summary=False)
        assert (session.pz_dir(PZ, "reference_dem") / f"{PZ}_dem.tif").exists()


# --------------------------------------------------------------------------- #
class TestThumbsAndPairs:
    def test_update_thumbs_registers_every_type(self, project):
        session, *_ = project
        counts = session._thumbs["th_type"].value_counts().to_dict()
        assert counts == {"dem": 3, "hillshade": 3, "mdhillshade": 3, "slope": 3}

    def test_pairs_follow_the_type(self, project):
        session, *_ = project
        out = session.update_pairs(strategy="consecutive", image_type="mdhillshade")
        assert len(out) == 2 and set(out["pa_type"]) == {"mdhillshade"}
        for pa_path in out["pa_path"]:
            assert f"{os.sep}image_correlation{os.sep}mdhillshade{os.sep}" in pa_path

    def test_append_adds_a_second_type_without_collision(self, project):
        session, *_ = project
        session.update_pairs(strategy="consecutive", image_type="mdhillshade")
        out = session.update_pairs(strategy="consecutive", image_type="slope", append=True)
        assert out["pa_type"].value_counts().to_dict() == {"mdhillshade": 2, "slope": 2}
        # same pa_key in both types, but distinct directories
        keys = out["pa_path"].map(lambda p: p.rsplit(os.sep, 1)[-1])
        assert keys.duplicated().sum() == 2
        assert not out["pa_path"].duplicated().any()
        assert len(session.get_pairs(image_type="slope")) == 2

    def test_pair_round_trips_from_its_path(self, project):
        session, *_ = project
        session.update_pairs(strategy="consecutive", image_type="slope")
        pair = session.get_pairs(image_type="slope")[0]
        pair.pa_path.mkdir(parents=True, exist_ok=True)
        again = Pair(session, target_path=pair.pa_path)
        assert again.pa_type == "slope"
        assert again.pa_key == pair.pa_path.name
        assert again.pa_left.th_path == pair.pa_left.th_path
        assert "elevation" in again.pa_left.th_path

    def test_pairs_never_cross_types(self, project):
        session, *_ = project
        name_a = f"{PZ}_2019-08-14_pleiades.tif"
        name_b = f"{PZ}_2020-08-20_pleiades.tif"
        a = Thumb(session.thumb_dir(PZ, "slope") / name_a)
        b = Thumb(session.thumb_dir(PZ, "mdhillshade") / name_b)
        with pytest.raises(AssertionError, match="image types"):
            a + b

    def test_optical_pa_path_is_unchanged(self, project):
        session, *_ = project
        opt_dir = session.thumb_dir(PZ, "opt")
        src = session.thumb_dir(PZ, "hillshade")
        for name in (f"{PZ}_2019-08-14_pleiades.tif", f"{PZ}_2020-08-20_pleiades.tif"):
            shutil.copy(src / name, opt_dir / name)
        try:
            a = Thumb(opt_dir / f"{PZ}_2019-08-14_pleiades.tif")
            b = Thumb(opt_dir / f"{PZ}_2020-08-20_pleiades.tif")
            pair = a + b
            assert a.th_type == "opt" and pair.pa_type == "opt"
            assert pair.pa_path == (session.pz_dir(PZ, "image_correlation") / pair.pa_key).absolute()
        finally:
            for p in opt_dir.glob("*.tif"):
                p.unlink()

    def test_legacy_layout_refuses_elevation_types(self, project):
        session, *_ = project
        session._legacy_layout = True
        try:
            with pytest.raises(ValueError, match="new folder layout"):
                session.thumb_dir(PZ, "slope")
            with pytest.raises(ValueError, match="new folder layout"):
                session.pair_dir(PZ, "slope")
        finally:
            session._legacy_layout = False


# --------------------------------------------------------------------------- #
class TestCorrelationInputs:
    def test_type_defaults_and_explicit_override(self):
        assert _corr_input_params("opt", {"nodata_value": None}) == CORR_INPUT_DEFAULTS["opt"]
        assert _corr_input_params("mdhillshade", {})["nodata_value"] == 0
        assert _corr_input_params("slope", {})["individual_normalization"] is True
        assert _corr_input_params("slope", {"nodata_value": -1.0})["nodata_value"] == -1.0

    def test_params_file_carries_the_encoding(self, project):
        session, *_ = project
        session.update_pairs(strategy="consecutive", image_type="mdhillshade")
        scripts = _prepare(session, image_type="mdhillshade")
        assert len(scripts) == 2
        pair = session.get_pairs(image_type="mdhillshade")[0]
        params = parse_params_file(pair.pa_path / f"{pair.pa_key}_CorrParameters.txt")
        assert params["nodata-value"] == "0"
        assert "individually-normalize" in params

    def test_changing_the_encoding_regenerates(self, project, caplog):
        session, *_ = project
        session.update_pairs(strategy="consecutive", image_type="slope")
        _prepare(session, image_type="slope")
        pair = session.get_pairs(image_type="slope")[0]
        pfile = pair.pa_path / f"{pair.pa_key}_CorrParameters.txt"
        assert parse_params_file(pfile)["nodata-value"] == "-9999.0"
        assert not input_params_differ_from_file(pfile, _corr_input_params("slope", {}))
        _prepare(session, image_type="slope", nodata_value=-32768.0)
        assert parse_params_file(pfile)["nodata-value"] == "-32768.0"

    def test_optical_file_without_the_lines_reads_as_unchanged(self, tmp_path):
        pfile = tmp_path / "x_CorrParameters.txt"
        pfile.write_text("prefilter-mode 2\ncorr-kernel 21 21\n")
        assert not input_params_differ_from_file(pfile, CORR_INPUT_DEFAULTS["opt"])
        assert input_params_differ_from_file(pfile, CORR_INPUT_DEFAULTS["slope"])

    def test_keyed_settings_refuse_a_mixed_type_selection(self, project):
        session, *_ = project
        session.update_pairs(strategy="consecutive", image_type="mdhillshade")
        session.update_pairs(strategy="consecutive", image_type="slope", append=True)
        key = session.get_pairs(image_type="slope")[0].pa_key
        with pytest.raises(ValueError, match="more than one image type"):
            _prepare(session, params_by_pair={key: {"corr_kernel": (15, 15)}})
        # scoped to one type it is unambiguous
        assert _prepare(session, image_type="slope",
                        params_by_pair={key: {"corr_kernel": (15, 15)}})


@pytest.mark.parametrize("method", [
    "get_pairs", "get_pairs_overview", "get_failed_pairs", "update_pairs",
    "prepare_pairs_correlation", "launch_pairs_correlation", "sync_pairs_after_cluster",
    "reset_pairs_outputs", "extract_pairs_raw_displacements", "apply_pairs_corrections",
    "explore_correlation_params",
])
def test_every_pair_selecting_method_takes_image_type(method):
    """criterias cannot separate types: "hillshade" is a substring of "mdhillshade"."""
    import inspect
    assert "image_type" in inspect.signature(getattr(Session, method)).parameters
