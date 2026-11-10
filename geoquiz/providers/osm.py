"""OpenStreetMap from pre-downloaded PBF extracts (Geofabrik), parsed with pyosmium.

One extract becomes thematic layers in the store (streets, rail, waterways,
water, optionally buildings), partitioned by extract name. No Overpass or
Nominatim calls: those services' usage policies rule out batch use.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
import shapely

from .. import licences, schema
from ..ingest import RawCache
from ..schema import LOD, Feature, Scale
from .base import StoreProvider

GEOFABRIK = "https://download.geofabrik.de/{extract}-latest.osm.pbf"

#: highway=* values drawn as streets, with importance rank.
STREET_RANK = {
    "motorway": 1.0, "motorway_link": 0.85, "trunk": 0.9, "trunk_link": 0.75,
    "primary": 0.8, "primary_link": 0.7, "secondary": 0.65, "secondary_link": 0.55,
    "tertiary": 0.5, "tertiary_link": 0.45, "unclassified": 0.35, "residential": 0.3,
    "living_street": 0.28, "pedestrian": 0.25, "service": 0.12,
    "footway": 0.05, "path": 0.05, "cycleway": 0.05, "steps": 0.05, "track": 0.05,
}
RAIL_RANK = {"rail": 0.9, "subway": 0.7, "narrow_gauge": 0.6, "light_rail": 0.6, "tram": 0.4, "funicular": 0.3}
RAIL_MINOR_SERVICE = {"yard", "siding", "spur", "crossover"}
WATERWAY_RANK = {"river": 0.9, "canal": 0.6, "stream": 0.3}

LINE_KEYS = ("highway", "railway", "waterway")
THEMES = ("streets", "rail", "waterways", "water", "buildings")


class _Rows:
    def __init__(self):
        self.wkb, self.rank, self.name, self.sid, self.cls, self.bridge, self.tunnel = [], [], [], [], [], [], []

    def add(self, wkb, rank, name, sid, cls, tags=None):
        self.wkb.append(wkb); self.rank.append(rank); self.name.append(name); self.sid.append(sid); self.cls.append(cls)
        self.bridge.append(bool(tags) and tags.get("bridge", "no") != "no")
        self.tunnel.append(bool(tags) and tags.get("tunnel", "no") != "no")

    def frame(self, **common) -> gpd.GeoDataFrame:
        g = gpd.GeoDataFrame(geometry=shapely.from_wkb(self.wkb) if self.wkb else [], crs=schema.CRS)
        return schema.normalise(g, **common, rank=self.rank or None, name=self.name or None, source_id=self.sid or None,
                                extras={"x_class": np.array(self.cls, dtype=object),
                                        "x_bridge": np.array(self.bridge, dtype=bool),
                                        "x_tunnel": np.array(self.tunnel, dtype=bool)})


class OsmProvider(StoreProvider):
    name = "osm"
    licence = licences.OSM_ODBL
    features = {
        Feature.STREET: {Scale.CITY},
        Feature.RAIL: {Scale.CITY, Scale.COUNTRY},
        Feature.WATERWAY: {Scale.CITY},
        Feature.WATER: {Scale.CITY},
        Feature.BUILDING: {Scale.CITY},
    }

    def ingest(self, raw: RawCache, extract: str | None = None, pbf: Path | None = None,
               themes: tuple[str, ...] = ("streets", "rail", "waterways", "water"), refresh: bool = False) -> str:
        """Normalise one extract. ``extract`` is a Geofabrik id such as
        ``europe/switzerland``; alternatively pass a local ``pbf`` path.
        Returns the partition name."""
        if (extract is None) == (pbf is None):
            raise ValueError("pass exactly one of extract= or pbf=")
        unknown = set(themes) - set(THEMES)
        if unknown:
            raise ValueError(f"unknown themes {sorted(unknown)}; expected a subset of {THEMES}")
        if pbf is None:
            pbf = raw.fetch(self.name, GEOFABRIK.format(extract=extract), refresh=refresh)
        pbf = Path(pbf)
        partition = pbf.name.split(".")[0].removesuffix("-latest")
        header = osmium.io.Reader(str(pbf)).header()
        version = header.get("osmosis_replication_timestamp") or header.get("timestamp") or "unknown"
        common = dict(source=self.name, licence=self.name, attribution=self.licence.attribution, data_version=version)
        fab = osmium.geom.WKBFactory()

        if {"streets", "rail", "waterways"} & set(themes):
            streets, rail, water_lines = _Rows(), _Rows(), _Rows()
            ways = (osmium.FileProcessor(str(pbf)).with_locations()
                    .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                    .with_filter(osmium.filter.KeyFilter(*LINE_KEYS)))
            for w in ways:
                t = w.tags
                hw, rw, ww = t.get("highway"), t.get("railway"), t.get("waterway")
                target = None
                if hw in STREET_RANK and t.get("area") != "yes":
                    target, rank, cls = streets, STREET_RANK[hw], hw
                elif rw in RAIL_RANK:
                    minor = t.get("service") in RAIL_MINOR_SERVICE
                    target, rank, cls = rail, 0.15 if minor else RAIL_RANK[rw], rw
                elif ww in WATERWAY_RANK:
                    target, rank, cls = water_lines, WATERWAY_RANK[ww], ww
                if target is None:
                    continue
                try:
                    wkb = fab.create_linestring(w)
                except Exception:        # way with missing node locations at the extract edge
                    continue
                target.add(wkb, rank, t.get("name"), f"w{w.id}", cls, t)
            for theme, feature, rows in (("streets", Feature.STREET, streets), ("rail", Feature.RAIL, rail),
                                         ("waterways", Feature.WATERWAY, water_lines)):
                if theme in themes:
                    self.store.write(rows.frame(feature=feature, **common), self.name, feature, partition,
                                     lods=(LOD.FULL, LOD.MEDIUM))

        area_keys = (["natural", "water", "waterway"] if "water" in themes else []) + (["building"] if "buildings" in themes else [])
        if area_keys:
            water, buildings = _Rows(), _Rows()
            key_filter = osmium.filter.KeyFilter(*area_keys)
            for a in osmium.FileProcessor(str(pbf)).with_areas(key_filter).with_filter(key_filter):
                if not a.is_area():
                    continue
                t = a.tags
                if "water" in themes and (t.get("natural") == "water" or t.get("waterway") == "riverbank"):
                    target, cls = water, t.get("water") or "water"
                elif "buildings" in themes and "building" in t:
                    target, cls = buildings, t.get("building")
                else:
                    continue
                try:
                    wkb = fab.create_multipolygon(a)
                except Exception:        # invalid or incomplete multipolygon
                    continue
                target.add(wkb, 0.5, t.get("name"), f"{'w' if a.from_way() else 'r'}{a.orig_id()}", cls)
            if "water" in themes:
                self.store.write(water.frame(feature=Feature.WATER, **common), self.name, Feature.WATER, partition,
                                 lods=(LOD.FULL, LOD.MEDIUM))
            if "buildings" in themes:
                self.store.write(buildings.frame(feature=Feature.BUILDING, **common), self.name, Feature.BUILDING,
                                 partition, lods=(LOD.FULL,))
        self.save_licence(self.name, self.licence)
        return partition
