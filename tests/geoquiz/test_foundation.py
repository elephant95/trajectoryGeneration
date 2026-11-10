"""Phase 1: schema, store, licences, providers, region resolver."""
import geopandas as gpd
import numpy as np
import pytest
import shapely

from geoquiz import licences, schema
from geoquiz.ingest import RawCache
from geoquiz.providers.geoboundaries import lfs_url, licence_from_metadata
from geoquiz.providers.osm import OsmProvider
from geoquiz.regions import fit_city_window, fold, radius_for_budget
from geoquiz.schema import LOD, Feature, Scale
from geoquiz.store import FeatureStore


def test_normalise_enforces_schema():
    g = gpd.GeoDataFrame(geometry=[shapely.Point(2600000, 1200000)], crs="EPSG:2056")
    out = schema.normalise(g, feature=Feature.RIVER, source="s", licence="l", attribution="a", data_version="v",
                           rank=[3.0], extras={"x_note": ["hi"]})
    assert list(out.columns)[: len(schema.COLUMNS) - 1] == [c for c in schema.COLUMNS if c != "geometry"]
    assert out.crs.to_epsg() == 4326 and 7 < out.geometry.x[0] < 8          # reprojected to WGS84
    assert out["rank"][0] == 1.0                                             # clipped into 0..1
    with pytest.raises(ValueError):
        schema.normalise(g, feature=Feature.RIVER, source="s", licence="l", attribution="a", data_version="v",
                         extras={"note": ["must be prefixed"]})
    with pytest.raises(ValueError):
        schema.validate(out.rename(columns={"rank": "importance"}))


def test_store_bbox_read_and_lod(store):
    inside = store.read("hydrosheds", Feature.RIVER, bbox=(8.0, 46.45, 8.2, 46.55))
    everything = store.read("hydrosheds", Feature.RIVER)
    assert 0 < len(inside) < len(everything)
    assert inside.geometry.intersects(shapely.box(8.0, 46.45, 8.2, 46.55)).all()
    assert store.partitions("hydrosheds", Feature.RIVER) == ["eu"]
    # a missing LOD falls back to the next finer one
    assert len(store.read("ghsl", Feature.URBAN_CENTRE, lod=LOD.COARSE)) == 1
    assert store.read("osm", Feature.RAIL).empty
    assert set(store.inventory()["source"]) == {"natural_earth", "hydrosheds", "ghsl", "osm"}


def test_simplified_lod_has_fewer_vertices(tmp_path):
    s = FeatureStore(tmp_path)
    x = np.linspace(0, 1, 2000)
    wiggle = shapely.LineString(np.column_stack([x, 0.0001 * np.sin(x * 500)]))
    g = schema.normalise(gpd.GeoDataFrame(geometry=[wiggle], crs=4326), feature=Feature.RIVER, source="s", licence="l",
                         attribution="a", data_version="v")
    s.write(g, "s", Feature.RIVER)
    n = [shapely.get_num_coordinates(s.read("s", Feature.RIVER, lod=l).geometry[0]) for l in LOD]
    assert n[0] == 2000 and n[0] > n[1] >= n[2] >= 2


def test_credit_line_dedupes_and_puts_osm_first():
    line = licences.credit_line(["HydroSHEDS © WWF", "© OpenStreetMap contributors", "HydroSHEDS © WWF"])
    assert line == "© OpenStreetMap contributors · HydroSHEDS © WWF"


def test_geoboundaries_licence_comes_from_each_boundary():
    meta = {"boundaryID": "CHE-ADM0-1", "boundaryLicense": "Federal Office of Topography swisstopo License",
            "boundarySource": "Federal Office of Topography swisstopo", "licenseSource": "www.swisstopo.admin.ch/ogd"}
    lic = licence_from_metadata(meta)
    assert lic.attribution == "Federal Office of Topography swisstopo" and lic.commercial_use and not lic.share_alike
    assert not licence_from_metadata({**meta, "boundaryLicense": "CC BY-NC 4.0"}).commercial_use
    assert licence_from_metadata({**meta, "boundaryLicense": "Open Data Commons Open Database License 1.0"}).share_alike
    assert lfs_url("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/x.geojson") == \
        "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/releaseData/x.geojson"


def test_registry_prefers_dense_source_and_falls_back(registry, resolver):
    country = resolver.country("SQL")
    choice = {c.provider: c for c in registry.explain(Feature.RIVER, country)}
    assert choice["hydrosheds"].chosen and not choice["natural_earth"].chosen
    assert choice["natural_earth"].coverage < 0.5 <= choice["hydrosheds"].coverage     # one river is too sparse
    assert (registry.fetch(Feature.RIVER, country)["source"] == "hydrosheds").all()
    # where the preferred source has nothing, the next one is used even below the coverage bar
    other = resolver.country("Otherland")
    assert registry.fetch(Feature.RIVER, other).empty
    registry.providers.pop("hydrosheds")
    assert (registry.fetch(Feature.RIVER, country)["source"] == "natural_earth").all()
    # every row carries a licence key the registry can resolve
    assert licences.get(registry.fetch(Feature.RIVER, country)["licence"].iloc[0]).attribution == "Natural Earth"


def test_resolver_country_and_city(resolver):
    c = resolver.country("squareland")
    assert (c.id, c.kind, c.iso_a2) == ("SQL", Scale.COUNTRY, "CH")
    assert c.area_km2() == pytest.approx(111.2 * 77.0, rel=0.03)           # equal-area projection
    w, s, e, n = c.bbox(15_000)
    assert w < 8 and e > 9 and s < 46 and n > 47
    with pytest.raises(KeyError):
        resolver.country("Atlantis")
    city = resolver.city("suburb")                                          # accent-insensitive, any name in the centre
    assert city.name == "Gridville" and city.id == "ghsl:7" and city.country == "Squareland"
    assert fold("Zürich") == "zurich"


def test_city_window_follows_density_budget(registry, resolver):
    city = resolver.city("Gridville")
    streets = registry.fetch(Feature.STREET, city)
    small = fit_city_window(city, streets, budget_km=40, aspect=1.0)
    large = fit_city_window(city, streets, budget_km=160, aspect=1.0)
    assert small.window_radius_m < large.window_radius_m
    assert abs(small.centre[0] - 8.5) < 0.01 and abs(small.centre[1] - 46.5) < 0.01   # recentred on the densest part
    x0, y0, x1, y1 = small.extent_xy()
    km = streets.geometry.to_crs(small.crs).clip_by_rect(x0, y0, x1, y1).length.sum() / 1000
    assert km == pytest.approx(40, rel=0.15)
    mid, length = np.array([[0.0, 0.0], [100.0, 0.0]]), np.array([1000.0, 1000.0])
    assert radius_for_budget(mid, length, (0, 0), 5.0, r_min=10) == 100.0   # budget above what exists: take it all


def test_raw_cache_records_existing_files(tmp_path):
    raw = RawCache(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "file.bin").write_bytes(b"abc")
    path = raw.fetch("src", "https://example.invalid/file.bin")            # present: no network access
    rec = raw.record("src/file.bin")
    assert path.read_bytes() == b"abc" and rec["bytes"] == 3 and not rec["downloaded_by_ingest"]
    assert rec["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(ValueError):
        raw.fetch("src", "https://example.invalid/file.bin", sha256="0" * 64)


def test_osm_adapter_parses_an_extract(tmp_path):
    osm = tmp_path / "tiny.osm"
    osm.write_text("""<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
 <node id="1" version="1" lat="46.50" lon="8.50"/><node id="2" version="1" lat="46.50" lon="8.51"/>
 <node id="3" version="1" lat="46.51" lon="8.51"/><node id="4" version="1" lat="46.51" lon="8.50"/>
 <way id="10" version="1"><nd ref="1"/><nd ref="2"/><tag k="highway" v="primary"/><tag k="name" v="Main St"/><tag k="bridge" v="yes"/></way>
 <way id="11" version="1"><nd ref="2"/><nd ref="3"/><tag k="highway" v="bus_stop"/></way>
 <way id="12" version="1"><nd ref="3"/><nd ref="4"/><tag k="railway" v="rail"/><tag k="service" v="yard"/></way>
 <way id="13" version="1"><nd ref="1"/><nd ref="4"/><tag k="waterway" v="river"/></way>
 <way id="14" version="1"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="natural" v="water"/></way>
</osm>""")
    store = FeatureStore(tmp_path / "store")
    assert OsmProvider(store).ingest(RawCache(tmp_path / "raw"), pbf=osm) == "tiny"
    streets = store.read("osm", Feature.STREET)
    assert len(streets) == 1 and streets["name"][0] == "Main St" and streets["rank"][0] == 0.8
    assert streets["x_bridge"][0] and streets["source_id"][0] == "w10" and streets["licence"][0] == "osm"
    assert store.read("osm", Feature.RAIL)["rank"][0] == 0.15              # yard track ranks low
    assert len(store.read("osm", Feature.WATERWAY)) == 1
    assert store.read("osm", Feature.WATER).geom_type.isin(["Polygon", "MultiPolygon"]).all()
    assert licences.get("osm").share_alike
