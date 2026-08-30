"""Tests for the pzone reference grid, persisted DEM, and moving-area import.

These three artifacts replace the previous pattern of hunting for "some xDisp
raster" to match against, and of passing a DEM reprojected against one
arbitrary pair to a run that loops over all of them.
"""
import pathlib
import tempfile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import Polygon, box

import geoutils as gu

from geomulticorr import open_gmc_session
from geomulticorr.utils._grid import grid_key, grids_match


EPSG = 2154  # RGF93 / Lambert-93 — the geodatabase template CRS
# A pzone AOI in metres, deliberately NOT aligned to a 10 m lattice so the
# canonical snapping has something to do.
AOI = [(900_005.0, 6_450_003.0), (900_305.0, 6_450_003.0),
       (900_305.0, 6_450_203.0), (900_005.0, 6_450_203.0)]


@pytest.fixture
def session(tmp_path):
    """A new-layout project with one pzone."""
    s = open_gmc_session(tmp_path / "gridproj", epsg_code=EPSG)
    s.insert_pzone(AOI, "TestPz", "TP")
    return s


def _dem_source(path, epsg=EPSG, res=10.0, origin=(899_800.0, 6_450_500.0),
                height=80, width=80):
    """A synthetic DEM covering the AOI with a tilted elevation surface."""
    yy, xx = np.mgrid[0:height, 0:width]
    data = (1000.0 + 0.5 * xx + 0.3 * yy).astype("float32")
    gu.Raster.from_array(
        np.ma.MaskedArray(data),
        transform=Affine(res, 0.0, origin[0], 0.0, -res, origin[1]),
        crs=CRS.from_epsg(epsg),
        nodata=None,
    ).to_file(str(path))
    return path


class TestBuildReferenceGrid:
    def test_creates_grid_file(self, session, tmp_path):
        out = session.build_reference_grid("TestPz", resolution=10.0)
        assert out.exists()
        assert out == tmp_path / "gridproj" / "TestPz" / "reference_raster" / "TestPz_grid.tif"

    def test_grid_matches_canonical_grid(self, session):
        """The persisted grid is exactly what _compute_canonical_grid computes."""
        session.build_reference_grid("TestPz", resolution=10.0)
        grid = session.get_reference_grid("TestPz")

        pz_vec = session._pzone_vector("TestPz")
        bounds, (w, h) = session._compute_canonical_grid(pz_vec, session.epsg, 10.0)

        assert (grid.width, grid.height) == (w, h)
        assert grid.bounds.left == pytest.approx(bounds.left)
        assert grid.bounds.bottom == pytest.approx(bounds.bottom)
        assert grid.bounds.right == pytest.approx(bounds.right)
        assert grid.bounds.top == pytest.approx(bounds.top)

    def test_bounds_snapped_outward_to_lattice(self, session):
        """Bounds land on a 10 m lattice anchored at the CRS origin."""
        session.build_reference_grid("TestPz", resolution=10.0)
        b = session.get_reference_grid("TestPz").bounds
        for edge in (b.left, b.bottom, b.right, b.top):
            assert edge % 10.0 == pytest.approx(0.0)

    def test_resolution_is_respected(self, session):
        session.build_reference_grid("TestPz", resolution=5.0)
        grid = session.get_reference_grid("TestPz")
        assert grid.res == pytest.approx((5.0, 5.0))

    def test_crs_is_session_epsg(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        assert session.get_reference_grid("TestPz").crs.to_epsg() == EPSG

    def test_file_is_tiny(self, session):
        """All-ones 1-bit: size must not scale with pixel count."""
        session.build_reference_grid("TestPz", resolution=1.0)  # 300x200 px
        assert session.reference_grid_path("TestPz").stat().st_size < 50_000

    def test_encoding_is_one_bit_all_ones(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        with rasterio.open(session.reference_grid_path("TestPz")) as ds:
            assert ds.tags(1, ns="IMAGE_STRUCTURE").get("NBITS") == "1"
            assert set(np.unique(ds.read(1))) == {1}

    def test_roundtrip_preserves_grid_exactly(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        a = session.get_reference_grid("TestPz")
        b = session.get_reference_grid("TestPz")
        assert grid_key(a) == grid_key(b)

    def test_requires_exactly_one_of_resolution_or_from_pair(self, session):
        with pytest.raises(ValueError, match="exactly one"):
            session.build_reference_grid("TestPz")
        with pytest.raises(ValueError, match="exactly one"):
            session.build_reference_grid("TestPz", resolution=10.0, from_pair=object())

    def test_overwrite_false_keeps_existing(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        first = grid_key(session.get_reference_grid("TestPz"))
        session.build_reference_grid("TestPz", resolution=5.0)  # no overwrite
        assert grid_key(session.get_reference_grid("TestPz")) == first

    def test_overwrite_true_rebuilds(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        session.build_reference_grid("TestPz", resolution=5.0, overwrite=True)
        assert session.get_reference_grid("TestPz").res == pytest.approx((5.0, 5.0))

    def test_unknown_pzone_raises(self, session):
        with pytest.raises(KeyError, match="No pzone named"):
            session.build_reference_grid("NotAPzone", resolution=10.0)


class TestBuildReferenceGridFromPair:
    def test_adopts_pair_grid_exactly(self, session, tmp_path):
        """from_pair= reproduces the source raster's grid bit for bit."""
        ew = tmp_path / "fake_ew.tif"
        transform = Affine(3.0, 0.0, 900_001.0, 0.0, -3.0, 6_450_202.0)
        gu.Raster.from_array(
            np.ma.MaskedArray(np.zeros((37, 41), dtype="float32")),
            transform=transform,
            crs=CRS.from_epsg(EPSG),
            nodata=None,
        ).to_file(str(ew))

        class _FakePair:
            pa_key = "TestPz_2024-01-01-ps_2024-01-15-ps"
            pa_ew_path = ew

        session.build_reference_grid("TestPz", from_pair=_FakePair())
        grid = session.get_reference_grid("TestPz")

        assert grids_match(grid, gu.Raster(str(ew), load_data=False))
        assert (grid.width, grid.height) == (41, 37)

    def test_missing_ew_raster_raises(self, session, tmp_path):
        class _FakePair:
            pa_key = "TestPz_missing"
            pa_ew_path = tmp_path / "does_not_exist.tif"

        with pytest.raises(FileNotFoundError, match="no EW raster"):
            session.build_reference_grid("TestPz", from_pair=_FakePair())


class TestGetReferenceGrid:
    def test_missing_grid_raises_with_guidance(self, session):
        with pytest.raises(FileNotFoundError, match="build_reference_grid"):
            session.get_reference_grid("TestPz")

    def test_is_lazy(self, session):
        """load_data=False — metadata available, no pixels read."""
        session.build_reference_grid("TestPz", resolution=10.0)
        grid = session.get_reference_grid("TestPz")
        assert grid.is_loaded is False
        assert grid.width > 0 and grid.height > 0
        assert grid.bounds is not None


class TestBuildReferenceDem:
    def test_dem_lands_on_reference_grid(self, session, tmp_path):
        """The whole point: stored DEM shares the grid, so fit() never regrids."""
        session.build_reference_grid("TestPz", resolution=10.0)
        out = session.build_reference_dem("TestPz", _dem_source(tmp_path / "src.tif"))

        assert out.exists()
        assert grids_match(
            gu.Raster(str(out), load_data=False),
            session.get_reference_grid("TestPz"),
        )

    def test_written_where_pzone_get_dem_reads(self, session, tmp_path):
        """Pzone.get_dem() starts working with no change to it."""
        from geomulticorr.core.pzone import Pzone

        session.build_reference_grid("TestPz", resolution=10.0)
        session.build_reference_dem("TestPz", _dem_source(tmp_path / "src.tif"))

        pz = Pzone("TestPz", session)
        dem = pz.get_dem()
        assert dem is not False
        assert grids_match(dem, session.get_reference_grid("TestPz"))

    def test_elevation_values_are_preserved(self, session, tmp_path):
        session.build_reference_grid("TestPz", resolution=10.0)
        out = session.build_reference_dem("TestPz", _dem_source(tmp_path / "src.tif"))
        data = gu.Raster(str(out)).data
        assert np.isfinite(data).any()
        assert 900.0 < float(np.ma.median(data)) < 1200.0

    def test_accepts_raster_object(self, session, tmp_path):
        session.build_reference_grid("TestPz", resolution=10.0)
        src = gu.Raster(str(_dem_source(tmp_path / "src.tif")))
        out = session.build_reference_dem("TestPz", src)
        assert out.exists()

    def test_reprojects_dem_from_a_different_crs(self, session, tmp_path):
        """A WGS84 DEM must land on the Lambert-93 grid."""
        session.build_reference_grid("TestPz", resolution=10.0)
        grid = session.get_reference_grid("TestPz")

        # Derive the WGS84 footprint from the grid rather than hardcoding it,
        # then pad generously so the source comfortably covers the AOI.
        wxmin, wymin, wxmax, wymax = grid.get_bounds_projected(
            out_crs=CRS.from_epsg(4326)
        )
        pad = 0.01
        wxmin, wymin, wxmax, wymax = wxmin - pad, wymin - pad, wxmax + pad, wymax + pad

        n = 80
        res_x, res_y = (wxmax - wxmin) / n, (wymax - wymin) / n
        yy, xx = np.mgrid[0:n, 0:n]
        gu.Raster.from_array(
            np.ma.MaskedArray((1000.0 + xx + yy).astype("float32")),
            transform=Affine(res_x, 0.0, wxmin, 0.0, -res_y, wymax),
            crs=CRS.from_epsg(4326),
            nodata=None,
        ).to_file(str(tmp_path / "wgs.tif"))

        out = session.build_reference_dem("TestPz", tmp_path / "wgs.tif")
        assert grids_match(
            gu.Raster(str(out), load_data=False),
            session.get_reference_grid("TestPz"),
        )
        # The regridded DEM must carry real elevations, not a nodata field.
        assert np.isfinite(gu.Raster(str(out)).data).any()

    def test_disjoint_dem_raises(self, session, tmp_path):
        session.build_reference_grid("TestPz", resolution=10.0)
        far = _dem_source(tmp_path / "far.tif", origin=(100_000.0, 6_000_000.0))
        with pytest.raises(ValueError, match="does not overlap"):
            session.build_reference_dem("TestPz", far)

    def test_requires_reference_grid_first(self, session, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_reference_grid"):
            session.build_reference_dem("TestPz", _dem_source(tmp_path / "src.tif"))

    def test_missing_source_raises(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        with pytest.raises(FileNotFoundError, match="DEM source not found"):
            session.build_reference_dem("TestPz", "/nonexistent/dem.tif")

    def test_overwrite_false_keeps_existing(self, session, tmp_path):
        session.build_reference_grid("TestPz", resolution=10.0)
        out = session.build_reference_dem("TestPz", _dem_source(tmp_path / "src.tif"))
        mtime = out.stat().st_mtime_ns
        session.build_reference_dem("TestPz", _dem_source(tmp_path / "src.tif"))
        assert out.stat().st_mtime_ns == mtime


class TestLoadMovingAreas:
    def _polygons(self, epsg=EPSG):
        """Two polygons inside the AOI."""
        return gpd.GeoDataFrame(
            {"name": ["rg1", "rg2"]},
            geometry=[
                box(900_050.0, 6_450_050.0, 900_120.0, 6_450_120.0),
                box(900_180.0, 6_450_100.0, 900_250.0, 6_450_170.0),
            ],
            crs=f"EPSG:{epsg}",
        )

    def test_writes_gpkg_in_vector_dir(self, session, tmp_path):
        session.build_reference_grid("TestPz", resolution=10.0)
        out = session.load_moving_areas("TestPz", self._polygons())
        assert out.exists()
        assert out.parent == tmp_path / "gridproj" / "TestPz" / "vector"
        assert out.name == "TestPz_moving-areas.gpkg"

    def test_roundtrip_via_get_moving_areas(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        session.load_moving_areas("TestPz", self._polygons())
        vec = session.get_moving_areas("TestPz")
        assert isinstance(vec, gu.Vector)
        assert len(vec.ds) == 2
        assert vec.crs.to_epsg() == EPSG

    def test_reprojects_source_from_another_crs(self, session):
        """THE bug: wrong-CRS polygons used to silently produce an empty mask.

        StableAreaMask's path/GeoDataFrame branch rasterizes without any CRS
        handling, so EPSG:4326 polygons over a projected grid fall off the
        raster entirely -> nothing masked -> corrections fitted on moving
        terrain. Storing them reprojected is what prevents that.
        """
        session.build_reference_grid("TestPz", resolution=10.0)
        wgs = self._polygons().to_crs(epsg=4326)
        assert wgs.crs.to_epsg() == 4326

        session.load_moving_areas("TestPz", wgs)
        stored = session.get_moving_areas("TestPz")

        assert stored.crs.to_epsg() == EPSG
        # And the geometries actually land inside the grid.
        grid = session.get_reference_grid("TestPz")
        gxmin, gymin, gxmax, gymax = grid.bounds
        sxmin, symin, sxmax, symax = stored.ds.total_bounds
        assert sxmin >= gxmin - 1 and sxmax <= gxmax + 1
        assert symin >= gymin - 1 and symax <= gymax + 1

    def test_stored_polygons_produce_a_real_mask(self, session):
        """End-to-end: the stored vector masks a non-trivial pixel count."""
        from geomulticorr.corrections.masks import StableAreaMask

        session.build_reference_grid("TestPz", resolution=10.0)
        session.load_moving_areas("TestPz", self._polygons().to_crs(epsg=4326))

        grid = session.get_reference_grid("TestPz")
        disp = gu.Raster.from_array(
            np.ma.MaskedArray(np.zeros((grid.height, grid.width), dtype="float32")),
            transform=grid.transform,
            crs=grid.crs,
            nodata=None,
        )

        keep = StableAreaMask(session.get_moving_areas("TestPz")).generate_mask(disp)
        assert keep.dtype == bool
        assert 0 < (~keep).sum() < keep.size, "expected some but not all pixels masked"

    def test_buffer_grows_the_excluded_area(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        session.load_moving_areas("TestPz", self._polygons())
        plain = session.get_moving_areas("TestPz").ds.geometry.area.sum()

        session.load_moving_areas("TestPz", self._polygons(), buffer=25.0, overwrite=True)
        buffered = session.get_moving_areas("TestPz").ds.geometry.area.sum()

        assert buffered > plain

    def test_polygons_outside_pzone_raise(self, session):
        session.build_reference_grid("TestPz", resolution=10.0)
        far = gpd.GeoDataFrame(
            geometry=[box(100_000.0, 6_000_000.0, 100_100.0, 6_000_100.0)],
            crs=f"EPSG:{EPSG}",
        )
        with pytest.raises(ValueError, match="No moving-area polygon overlaps"):
            session.load_moving_areas("TestPz", far)

    def test_requires_reference_grid_first(self, session):
        with pytest.raises(FileNotFoundError, match="build_reference_grid"):
            session.load_moving_areas("TestPz", self._polygons())

    def test_missing_moving_areas_raises_with_guidance(self, session):
        with pytest.raises(FileNotFoundError, match="load_moving_areas"):
            session.get_moving_areas("TestPz")

    def test_crs_less_source_is_assumed_session_crs(self, session, caplog):
        session.build_reference_grid("TestPz", resolution=10.0)
        no_crs = self._polygons()
        no_crs.crs = None

        session.load_moving_areas("TestPz", no_crs)
        assert session.get_moving_areas("TestPz").crs.to_epsg() == EPSG
