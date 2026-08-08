"""Real Swiss rail network loader.

Primary path: download swissTLM3D Railway (opendata.swiss) or the OSM
Switzerland-Railways export (HDX), per `data.rail_source`, and cache as
GeoJSON in the projected CRS.

Fallback: when those sources are unreachable (e.g. no network egress to
opendata.swiss/HDX), build an *approximate offline topology sketch* —
straight segments between the actual configured city coordinates, following
the real SBB trunk-line pattern (Gotthard, Lötschberg/Simplon, the
Mittelland spine, etc). This is clearly not survey-accurate geometry; it
exists purely so `network_distance` and the reveal overlay have something
truthful-in-topology to compare against when the authoritative data can't be
fetched. It is labelled as such in logs and in the returned object.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import LineString, MultiLineString

from swiss_rail_rl.config import Config
from swiss_rail_rl.data.cities import Point

# Real SBB trunk-line topology, expressed as city-name pairs. Used only as the
# offline fallback sketch — filtered to whatever names are actually present
# in the loaded point set, so it degrades gracefully with a custom config.
_TRUNK_LINE_EDGES: list[tuple[str, str]] = [
    ("Genève", "Lausanne"),
    ("Lausanne", "Fribourg"),
    ("Fribourg", "Bern"),
    ("Lausanne", "Neuchâtel"),
    ("Neuchâtel", "Biel/Bienne"),
    ("Biel/Bienne", "Bern"),
    ("Bern", "Olten"),
    ("Olten", "Basel"),
    ("Bern", "Thun"),
    ("Thun", "Brig"),
    ("Basel", "Olten"),
    ("Olten", "Luzern"),
    ("Olten", "Aarau"),
    ("Aarau", "Zürich"),
    ("Zürich", "Winterthur"),
    ("Winterthur", "St. Gallen"),
    ("Zürich", "Zug"),
    ("Zug", "Luzern"),
    ("Zürich", "Chur"),
    ("Zug", "Bellinzona"),
    ("Bellinzona", "Lugano"),
    ("Brig", "Sion"),
    ("Sion", "Lausanne"),
    ("St. Gallen", "Chur"),
]


@dataclass
class RailNetwork:
    lines: MultiLineString
    crs: str
    source: str  # e.g. "swisstlm3d", "osm_hdx", or "offline_sketch"

    def is_authoritative(self) -> bool:
        return self.source != "offline_sketch"


def _offline_sketch(points: list[Point], crs: str) -> RailNetwork:
    warnings.warn(
        "Real rail geometry download unavailable — using an approximate "
        "offline trunk-line sketch (straight segments between real city "
        "coordinates, following the actual SBB route topology) instead of "
        "authoritative swisstlm3d/OSM geometry. Fetch real data once network "
        "access is available for an accurate comparison.",
        stacklevel=2,
    )
    by_name = {p.name: p for p in points}
    segments = []
    for a, b in _TRUNK_LINE_EDGES:
        if a in by_name and b in by_name:
            pa, pb = by_name[a], by_name[b]
            segments.append(LineString([(pa.x, pa.y), (pb.x, pb.y)]))
    if not segments:
        segments = [LineString([(p.x, p.y) for p in points])]
    return RailNetwork(lines=MultiLineString(segments), crs=crs, source="offline_sketch")


def _try_download_real(config: Config, points: list[Point]) -> RailNetwork | None:
    """Attempts the configured real source. Returns None on any failure so the
    caller can fall back to the offline sketch; never raises.
    """
    import requests

    try:
        if config.data.rail_source == "osm_hdx":
            # HDX Switzerland Railways export (ODbL). Resource URL changes over
            # time; resolve it via the HDX API rather than hardcoding a link.
            api = "https://data.humdata.org/api/3/action/package_show?id=hotosm_che_railways"
            resp = requests.get(api, timeout=15)
            resp.raise_for_status()
            resources = resp.json()["result"]["resources"]
            geojson_url = next(r["url"] for r in resources if r["format"].lower() == "geojson")
            data = requests.get(geojson_url, timeout=30).json()
        else:
            # swissTLM3D Railway via the swisstopo STAC/ogd API on opendata.swiss.
            api = (
                "https://data.geo.admin.ch/api/stac/v1/collections/"
                "ch.swisstopo.swisstlm3d/items"
            )
            resp = requests.get(api, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        geoms = []
        for feat in data.get("features", []):
            geom = feat.get("geometry", {})
            if geom.get("type") == "LineString":
                geoms.append(LineString(geom["coordinates"]))
            elif geom.get("type") == "MultiLineString":
                geoms.extend(LineString(c) for c in geom["coordinates"])
        if not geoms:
            return None
        return RailNetwork(lines=MultiLineString(geoms), crs="EPSG:4326", source=config.data.rail_source)
    except Exception:
        return None


def _reproject(rail: RailNetwork, dst_crs: str) -> RailNetwork:
    if rail.crs == dst_crs:
        return rail
    from pyproj import Transformer

    transformer = Transformer.from_crs(rail.crs, dst_crs, always_xy=True)
    lines = [
        LineString([transformer.transform(x, y) for x, y in line.coords]) for line in rail.lines.geoms
    ]
    return RailNetwork(lines=MultiLineString(lines), crs=dst_crs, source=rail.source)


def _load_local_override(path: str) -> RailNetwork:
    """Loads a hand-downloaded GeoJSON FeatureCollection (WGS84 lon/lat) of
    LineString/MultiLineString railway geometries — e.g. swissTLM3D or the
    OSM Switzerland-Railways export fetched manually and dropped on disk.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    geoms = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "LineString":
            geoms.append(LineString(geom["coordinates"]))
        elif geom.get("type") == "MultiLineString":
            geoms.extend(LineString(c) for c in geom["coordinates"])
    if not geoms:
        raise ValueError(f"{path} has no LineString/MultiLineString features")
    return RailNetwork(lines=MultiLineString(geoms), crs="EPSG:4326", source=f"local:{Path(path).name}")


def load_rail_network(config: Config, points: list[Point]) -> RailNetwork:
    cache_dir = config.cache_dir()
    cache_path = cache_dir / f"rail_{config.data.rail_source}.geojson"
    dst_crs = config.region.crs_projected

    if config.data.rail_local_path:
        return _reproject(_load_local_override(config.data.rail_local_path), dst_crs)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        lines = [LineString(coords) for coords in payload["lines"]]
        return RailNetwork(lines=MultiLineString(lines), crs=payload["crs"], source=payload["source"])

    rail = _try_download_real(config, points)
    if rail is not None:
        rail = _reproject(rail, dst_crs)
    else:
        rail = _offline_sketch(points, dst_crs)

    payload = {
        "crs": rail.crs,
        "source": rail.source,
        "lines": [list(line.coords) for line in rail.lines.geoms],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return rail
