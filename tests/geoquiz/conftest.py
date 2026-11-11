"""Offline fixtures: a tiny synthetic store, so tests need no downloads."""
import geopandas as gpd
import numpy as np
import pytest
import shapely

pytest.importorskip("skia")
pytest.importorskip("osmium")

from geoquiz import licences, schema
from geoquiz.providers import default_registry
from geoquiz.regions import RegionResolver
from geoquiz.schema import LOD, Feature
from geoquiz.store import FeatureStore


def _norm(geoms, feature, source, **kw):
    lic = licences.get(source)
    return schema.normalise(gpd.GeoDataFrame(geometry=list(geoms), crs="EPSG:4326"), feature=feature, source=source,
                            licence=source, attribution=lic.attribution, data_version="test", **kw)


@pytest.fixture()
def store(tmp_path):
    """Squareland (ISO3 SQL): a 1° x 1° country at 8-9°E, 46-47°N, with a city at its centre."""
    s = FeatureStore(tmp_path / "store")
    country = shapely.box(8, 46, 9, 47)
    s.write(_norm([country, shapely.box(20, 50, 21, 51)], Feature.COUNTRY, "natural_earth", name=["Squareland", "Otherland"],
                  source_id=["SQL", "OTH"], rank=[0.5, 0.5],
                  extras={"x_iso_a2": np.array(["CH", None], dtype=object), "x_pop": [1e6, 2e6],
                          "x_continent": ["Europe", "Europe"]}), "natural_earth", Feature.COUNTRY)
    s.write(_norm([shapely.box(-10, 35, 30, 60)], Feature.LAND, "natural_earth"), "natural_earth", Feature.LAND,
            lods=(LOD.FULL,))
    # one sparse Natural Earth river, and a denser HydroRIVERS tree draining west along 46.5°N
    s.write(_norm([shapely.LineString([(7.5, 46.5), (9.5, 46.5)])], Feature.RIVER, "natural_earth", rank=[0.9]),
            "natural_earth", Feature.RIVER)
    xs = np.linspace(9.4, 7.6, 19)                       # trunk reaches, upstream -> downstream
    geoms, dist, rank = [], [], []
    for i in range(len(xs) - 1):
        geoms.append(shapely.LineString([(xs[i], 46.5), (xs[i + 1], 46.5)]))
        dist.append((len(xs) - 2 - i) * 7.7)
        rank.append(0.9)
        for side in (-1, 1):                              # a tributary joining at each reach's downstream end
            geoms.append(shapely.LineString([(xs[i + 1] + 0.03, 46.5 + side * 0.4), (xs[i + 1], 46.5)]))
            dist.append((len(xs) - 2 - i) * 7.7)
            rank.append(0.3)
    n = len(geoms)
    s.write(_norm(geoms, Feature.RIVER, "hydrosheds", rank=rank, source_id=np.arange(n),
                  extras={"x_discharge_cms": np.where(np.array(rank) > 0.5, 100.0, 2.0), "x_strahler": np.ones(n, np.int16),
                          "x_length_km": np.full(n, 7.7), "x_dist_dn_km": np.array(dist), "x_dist_up_km": np.zeros(n),
                          "x_main_riv": np.ones(n, np.int64), "x_next_down": np.zeros(n, np.int64)}),
            "hydrosheds", Feature.RIVER, "eu")
    s.write(_norm([shapely.Point(8.5, 46.5).buffer(0.05)], Feature.LAKE, "hydrosheds", name=["Lake Test"], rank=[0.6],
                  extras={"x_area_km2": [40.0], "x_type": np.array([1], np.int8)}), "hydrosheds", Feature.LAKE, "eu")
    # a city: GHSL urban centre plus an OSM street grid, denser towards (8.5, 46.5)
    s.write(_norm([shapely.box(8.45, 46.47, 8.55, 46.53)], Feature.URBAN_CENTRE, "ghsl", name=["Gridville"], source_id=[7],
                  rank=[0.3], extras={"x_names_all": np.array(["Gridville; Sübürb"], dtype=object),
                                      "x_country": np.array(["Squareland"], dtype=object), "x_pop": [120000.0],
                                      "x_area_km2": [35.0]}), "ghsl", Feature.URBAN_CENTRE, lods=(LOD.FULL,))
    streets, cls = [], []
    for k in np.linspace(-0.04, 0.04, 33):
        for a, b in (((8.46, 46.5 + k * 0.6), (8.54, 46.5 + k * 0.6)), ((8.5 + k, 46.475), (8.5 + k, 46.525))):
            line = shapely.LineString([a, b]).segmentize(0.0025)       # shared vertices at the crossings
            streets.append(line)
            cls.append("residential")
    ranks = [0.3] * len(streets)
    for y in (46.2, 46.8):                                # two cross-country primary roads
        streets.append(shapely.LineString([(7.6, y), (9.4, y)]).segmentize(0.05))
        cls.append("primary")
        ranks.append(0.8)
    n = len(streets)
    s.write(_norm([shapely.LineString([(7.6, 45.5), (7.6, 47.5)])], Feature.COASTLINE, "natural_earth"),
            "natural_earth", Feature.COASTLINE)
    s.write(_norm(streets, Feature.STREET, "osm", rank=np.array(ranks), source_id=[f"w{i}" for i in range(n)],
                  extras={"x_class": np.array(cls, dtype=object), "x_bridge": np.zeros(n, bool),
                          "x_tunnel": np.zeros(n, bool)}), "osm", Feature.STREET, "testland", lods=(LOD.FULL,))
    return s


@pytest.fixture()
def registry(store):
    return default_registry(store)


@pytest.fixture()
def resolver(registry):
    return RegionResolver(registry)
