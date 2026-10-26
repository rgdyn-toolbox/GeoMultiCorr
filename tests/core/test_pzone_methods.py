"""Unit tests for the Pzone methods, against a stub session.

The real Session needs a geodatabase; Pzone only calls a handful of its
methods (``pz_names``, ``pz_dir``, ``get_thumbs_overview``, ``get_thumbs``),
so a recorder stands in. Thumbs and pairs are fakes whose ``+`` mimics
``Thumb.__add__`` (including its ``AssertionError`` on an invalid combination).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import geoutils as gu

import geomulticorr.core.pzone as gmc_pzone
from geomulticorr.core.pzone import Pzone


class FakeSession:
    def __init__(self, root: Path, pz_names=("PZ",)):
        self.root = root
        self.pz_names = list(pz_names)
        self.calls: list = []
        self.overview = pd.DataFrame()
        self.thumbs: list = []
        for n in self.pz_names:
            (root / n).mkdir(parents=True, exist_ok=True)

    def pz_dir(self, pz_name, kind=None):
        base = self.root / pz_name
        return base if kind is None else base / kind

    def get_thumbs_overview(self, criterias):
        self.calls.append(("overview", criterias))
        return self.overview

    def get_thumbs(self, criterias):
        self.calls.append(("thumbs", criterias))
        return self.thumbs


class FakeThumb:
    def __init__(self, date: str, ok_with=None):
        self.th_date_datetime = dt.datetime.strptime(date, "%Y-%m-%d")
        self.name = date
        self._forbidden = set(ok_with or ())

    def __add__(self, other):
        if other.name in self._forbidden:
            raise AssertionError("invalid combination")
        dt_days = abs((other.th_date_datetime - self.th_date_datetime).days)
        return SimpleNamespace(
            pa_dt_days=dt_days, pair=(self.name, other.name),
            to_pdserie=lambda: pd.Series({"pa_dt_days": dt_days, "left": self.name, "right": other.name}),
        )


@pytest.fixture
def sess(tmp_path):
    return FakeSession(tmp_path)


@pytest.fixture
def pz(sess):
    return Pzone("PZ", sess)


class TestInit:
    def test_unknown_pzone_name(self, sess):
        with pytest.raises(AssertionError, match="not existing"):
            Pzone("Nope", sess)

    def test_missing_folder(self, sess):
        sess.pz_names.append("Ghost")
        with pytest.raises(AssertionError, match="no pzone folder"):
            Pzone("Ghost", sess)

    def test_attributes(self, pz, sess):
        assert pz.pz_name == "PZ"
        assert pz.pz_dem_path == sess.root / "PZ" / "reference_dem" / "PZ_dem.tif"

    def test_dir_delegates(self, pz, sess):
        assert pz.dir() == sess.root / "PZ"
        assert pz.dir("vector") == sess.root / "PZ" / "vector"


class TestThumbQueries:
    def test_overview_and_thumbs_prepend_criteria_and_pzone(self, pz, sess):
        pz.get_thumbs_overview("spot")
        pz.get_thumbs()
        assert sess.calls == [("overview", ["spot", "PZ"]), ("thumbs", ["", "PZ"])]

    def test_get_valid_thumbs_filters(self, pz, sess, monkeypatch):
        sess.overview = pd.DataFrame({
            "th_path": ["a.tif", "b.tif", "c.tif", "d.tif"],
            "th_valid": ["1", "0", "1", "1"],
            "th_type": ["opt", "opt", "slope", "opt"],
        })
        monkeypatch.setattr(gmc_pzone.gmc_thumb, "Thumb", lambda p: f"T:{p}")
        assert pz.get_valid_thumbs() == ["T:a.tif", "T:d.tif"]
        assert pz.get_valid_thumbs(image_type="slope") == ["T:c.tif"]
        assert pz.get_valid_thumbs(image_type=None) == ["T:a.tif", "T:c.tif", "T:d.tif"]

    def test_valid_thumbs_sensor_filter_goes_to_criteria(self, pz, sess, monkeypatch):
        sess.overview = pd.DataFrame({"th_path": [], "th_valid": [], "th_type": []})
        monkeypatch.setattr(gmc_pzone.gmc_thumb, "Thumb", lambda p: p)
        pz.get_valid_thumbs(sensor_filter="spot")
        assert sess.calls[-1] == ("overview", ["PZ", "spot"])


class TestPairBuilding:
    def test_get_pairs_uses_permutations_and_skips_invalid(self, pz, sess):
        a, b, c = FakeThumb("2020-01-01"), FakeThumb("2020-02-01"), FakeThumb("2020-03-01")
        a._forbidden = {"2020-03-01"}  # a+c is invalid
        sess.thumbs = [a, b, c]
        pairs = pz.get_pairs()
        assert len(pairs) == 5
        assert ("2020-01-01", "2020-03-01") not in [p.pair for p in pairs]

    def test_get_valid_pairs(self, pz, monkeypatch):
        a, b = FakeThumb("2020-01-01"), FakeThumb("2020-02-01")
        monkeypatch.setattr(pz, "get_valid_thumbs", lambda **k: [a, b])
        assert sorted(p.pair for p in pz.get_valid_pairs()) == [
            ("2020-01-01", "2020-02-01"), ("2020-02-01", "2020-01-01")]

    def test_overviews_are_frames_of_series(self, pz, monkeypatch):
        fake = [SimpleNamespace(to_pdserie=lambda: pd.Series({"k": 1}))]
        monkeypatch.setattr(pz, "get_pairs", lambda: fake)
        monkeypatch.setattr(pz, "get_valid_pairs", lambda: fake * 2)
        assert len(pz.get_pairs_overview()) == 1
        assert len(pz.get_valid_pairs_overview()) == 2

    def test_complete_pairs(self, pz, monkeypatch):
        mk = lambda s: SimpleNamespace(get_status=lambda: s)
        monkeypatch.setattr(pz, "get_pairs", lambda: [mk("complete"), mk("clipped"), mk("complete")])
        assert len(pz.get_complete_pairs()) == 2


class TestPairsWithStrategy:
    @pytest.fixture
    def thumbs(self, pz, monkeypatch):
        ts = [FakeThumb(d) for d in ("2020-01-01", "2020-01-11", "2020-03-01", "2021-01-01")]
        shuffled = [ts[2], ts[0], ts[3], ts[1]]  # sorted by date inside the method
        monkeypatch.setattr(pz, "get_valid_thumbs", lambda **k: shuffled)
        return ts

    def test_consecutive_is_date_ordered(self, pz, thumbs):
        pairs = pz.get_pairs_with_strategy("consecutive")
        assert [p.pair for p in pairs] == [
            ("2020-01-01", "2020-01-11"), ("2020-01-11", "2020-03-01"), ("2020-03-01", "2021-01-01")]

    def test_max_dt_filter(self, pz, thumbs):
        pairs = pz.get_pairs_with_strategy("consecutive", max_dt_days=60)
        assert all(p.pa_dt_days <= 60 for p in pairs) and len(pairs) == 2

    def test_min_dt_filter(self, pz, thumbs):
        pairs = pz.get_pairs_with_strategy("consecutive", min_dt_days=30)
        assert all(p.pa_dt_days >= 30 for p in pairs) and len(pairs) == 2

    def test_assertion_error_skips_pair(self, pz, thumbs):
        thumbs[0]._forbidden = {"2020-01-11"}
        assert len(pz.get_pairs_with_strategy("consecutive")) == 2

    def test_overview_frame(self, pz, thumbs):
        gdf = pz.get_valid_pairs_with_strategy_overview("consecutive", min_dt_days=30)
        assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) == 2

    def test_step_strategy_needs_max_step(self, pz, thumbs):
        with pytest.raises(ValueError, match="max_step"):
            pz.get_pairs_with_strategy("step")


class TestDem:
    def test_missing_dem_warns_and_returns_false(self, pz, caplog_gmc):
        assert pz.get_dem() is False
        assert "No dem for pzone PZ" in caplog_gmc.text

    def test_existing_dem_loaded(self, pz):
        pz.pz_dem_path.parent.mkdir(parents=True)
        with rasterio.open(
            pz.pz_dem_path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
            crs="EPSG:32632", transform=from_origin(0, 40, 10, 10),
        ) as dst:
            dst.write(np.ones((4, 4), dtype="float32"), 1)
        dem = pz.get_dem()
        assert isinstance(dem, gu.Raster) and dem.is_loaded


class TestPzFull:
    def test_logs_complete_and_abort(self, pz, monkeypatch):
        def mk(key, exc=None):
            def pa_full(*a):
                if exc:
                    raise exc
            return SimpleNamespace(pa_key=key, pa_full=pa_full)

        monkeypatch.setattr(pz, "get_valid_pairs", lambda: [
            mk("ok"), mk("val", ValueError), mk("assert", AssertionError)])
        assert pz.pz_full(32632) == {"COMPLETE": ["ok"], "ABORT": ["val", "assert"]}


class FakeMA:
    def __init__(self, value, shape=(2, 2)):
        self.array = np.full(shape, value, dtype="int64")
        self.shape = shape

    def getShape(self):
        return self.shape

    def cropFromRaster(self, other):
        return FakeMA(self.array.flat[0], other.shape)

    def copy(self):
        c = FakeMA(0, self.shape)
        c.array = self.array.copy()
        return c

    def __iadd__(self, other):
        self.array = self.array + other.array
        return self


class TestMovingAreas:
    def test_get_moving_areas_collects_per_pair(self, pz, monkeypatch):
        mk = lambda v: SimpleNamespace(get_moving_areas=lambda n, m: (v, n, m))
        monkeypatch.setattr(pz, "get_complete_pairs", lambda: [mk(1), mk(2)])
        assert pz.get_moving_areas(n_clusters=3, mode="x") == [(1, 3, "x"), (2, 3, "x")]

    def test_add_moving_areas_without_pairs_raises(self, pz, monkeypatch):
        monkeypatch.setattr(pz, "get_moving_areas", lambda n, m: [])
        with pytest.raises(ValueError, match="No complete pair"):
            pz.add_moving_areas()

    def test_add_moving_areas_sums_all_maps(self, pz, monkeypatch):
        monkeypatch.setattr(pz, "get_moving_areas", lambda n, m: [FakeMA(1), FakeMA(1), FakeMA(1)])
        assert int(pz.add_moving_areas().array[0, 0]) == 3

    def test_add_moving_areas_crops_mismatched_shapes(self, pz, monkeypatch):
        mas = [FakeMA(1, (3, 3))] + [FakeMA(1, (3, 3))] * 2 + [FakeMA(1, (2, 2)), FakeMA(1, (3, 3))]
        monkeypatch.setattr(pz, "get_moving_areas", lambda n, m: mas)
        out = pz.add_moving_areas()
        assert out.array.dtype.kind == "i"

    def test_cluster_addition_labels_two_levels(self, pz, monkeypatch):
        ma = FakeMA(0, (4, 4))
        ma.array[:, 2:] = 10
        monkeypatch.setattr(pz, "add_moving_areas", lambda: ma)
        out = pz.cluster_addition()
        assert out.array.shape == (4, 4)
        assert set(np.unique(out.array)) == {0, 1}
        assert len(set(out.array[:, 0])) == 1 and out.array[0, 0] != out.array[0, 3]


class TestVectorizeMultitemporal:
    def _raster(self, pz):
        vec = pz.dir("vector")
        vec.mkdir(parents=True)
        data = np.zeros((40, 40), dtype="uint8")
        data[5:25, 5:25] = 1
        with rasterio.open(
            vec / "PZ_moving-areas_round-0.tif", "w", driver="GTiff", height=40, width=40,
            count=1, dtype="uint8", crs="EPSG:32632", transform=from_origin(500000, 4649000, 10, 10),
        ) as dst:
            dst.write(data, 1)
        return vec / "PZ_moving-areas_round-0.gpkg"

    def test_keeps_only_moving_polygons(self, pz):
        gpkg = self._raster(pz)
        pz.vectorize_multitemporal_moving_areas(epsg=32632)
        gdf = gpd.read_file(gpkg, layer="PZ_moving-areas_round-0")
        assert (gdf["raster_val"] == 1).all() and len(gdf) == 1

    def test_min_surf_changes_layer_name(self, pz):
        gpkg = self._raster(pz)
        pz.vectorize_multitemporal_moving_areas(epsg=32632, min_surf=10)
        gdf = gpd.read_file(gpkg, layer="PZ_moving-areas_round-0_features-sup-10")
        assert len(gdf) == 1
