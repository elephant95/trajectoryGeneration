"""Overture Maps: roads, rail and water read by bounding box from the public
cloud GeoParquet release. Its transportation and base themes are derived from
OpenStreetMap and carry ODbL, so the obligations match the OSM adapter's.

Used as the fallback for city features when no OSM extract covers the region.
The read happens once at ingest; rendering still touches only the local store.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely

from .. import schema
from ..ingest import RawCache
from ..licences import Licence
from ..schema import LOD, Feature, Scale
from .base import StoreProvider
from .osm import RAIL_RANK, STREET_RANK, WATERWAY_RANK

BUCKET = "overturemaps-us-west-2"
REGION = "us-west-2"
#: Partitions holding a whole country's main roads and rail are named with this prefix.
COUNTRY_PREFIX = "country-"
THEMES = ("roads", "rail", "water", "buildings")
#: Road classes worth drawing at country scale.
MAJOR_ROADS = ("motorway", "trunk", "primary", "secondary")
RAIL_CLASS = {"standard_gauge": "rail", "narrow_gauge": "narrow_gauge", "subway": "subway",
              "light_rail": "light_rail", "tram": "tram", "funicular": "funicular", "unknown": "rail"}

OVERTURE_ODBL = Licence(
    id="ODbL-1.0",
    name="Open Database License 1.0 (Overture transportation and base themes)",
    url="https://docs.overturemaps.org/attribution",
    attribution="© OpenStreetMap contributors, Overture Maps Foundation",
    attribution_full=("Map data © OpenStreetMap contributors, Overture Maps Foundation, "
                      "available under the Open Database License."),
    share_alike=True,
)


def _has_flag(flags, flag: str) -> bool:
    if flags is None:
        return False
    return any(v is not None and flag in list(v) for v in (f.get("values") for f in flags))


class OvertureProvider(StoreProvider):
    name = "overture"
    licence = OVERTURE_ODBL
    features = {
        Feature.STREET: {Scale.CITY, Scale.COUNTRY},
        Feature.RAIL: {Scale.CITY, Scale.COUNTRY},
        Feature.WATERWAY: {Scale.CITY},
        Feature.WATER: {Scale.CITY},
        Feature.BUILDING: {Scale.CITY},
    }

    def partitions_for(self, feature: Feature, region):
        # a country pull holds main roads only: it must never pass for a city's street network
        if feature is Feature.STREET and region.kind is Scale.CITY:
            return [p for p in self.store.partitions(self.name, feature) if not p.startswith(COUNTRY_PREFIX)]
        return None

    @staticmethod
    def _fs():
        import pyarrow.fs as fs
        return fs.S3FileSystem(anonymous=True, region=REGION)

    def latest_release(self) -> str:
        import pyarrow.fs as fs
        infos = self._fs().get_file_info(fs.FileSelector(f"{BUCKET}/release/", recursive=False))
        return sorted(i.base_name for i in infos)[-1]

    def _read(self, release: str, theme: str, type_: str, bbox, columns: list[str], extra_filter=None):
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        d = ds.dataset(f"{BUCKET}/release/{release}/theme={theme}/type={type_}/", filesystem=self._fs(), format="parquet")
        w, s, e, n = bbox
        flt = ((pc.field("bbox", "xmin") < e) & (pc.field("bbox", "xmax") > w)
               & (pc.field("bbox", "ymin") < n) & (pc.field("bbox", "ymax") > s))
        if extra_filter is not None:
            flt = flt & extra_filter
        return d.to_table(filter=flt, columns=columns).to_pandas()

    def ingest(self, raw: RawCache, partition: str, bbox: tuple[float, float, float, float],
               themes: tuple[str, ...] = ("roads", "rail", "water"), road_classes: tuple[str, ...] | None = None,
               release: str | None = None, refresh: bool = False) -> str:
        """Pull one bounding box (west, south, east, north) into the store as ``partition``.

        ``themes`` is a subset of roads, rail, water, buildings. ``road_classes``
        limits roads to those classes (for a whole country, pass the major ones:
        the filter runs in the cloud read, so minor streets are never downloaded).
        """
        unknown = set(themes) - set(THEMES)
        if unknown:
            raise ValueError(f"unknown themes {sorted(unknown)}; expected a subset of {THEMES}")
        release = release or self.latest_release()
        common = dict(source=self.name, licence=self.name, attribution=self.licence.attribution, data_version=release)
        if "roads" in themes or "rail" in themes:
            self._ingest_segments(release, partition, bbox, themes, road_classes, common)
        if "water" in themes:
            self._ingest_water(release, partition, bbox, common)
        if "buildings" in themes:
            b = self._read(release, "buildings", "building", bbox, ["id", "class", "names", "geometry"])
            geom = shapely.from_wkb(b["geometry"].to_numpy())
            n = len(b)
            out = schema.normalise(
                gpd.GeoDataFrame(geometry=geom, crs=schema.CRS), feature=Feature.BUILDING, **common, rank=np.full(n, 0.5),
                name=b["names"].map(lambda v: v.get("primary") if isinstance(v, dict) else None).to_numpy(),
                source_id=b["id"].to_numpy(),
                extras={"x_class": b["class"].fillna("building").to_numpy(object),
                        "x_bridge": np.zeros(n, bool), "x_tunnel": np.zeros(n, bool)})
            self.store.write(out, self.name, Feature.BUILDING, partition, lods=(LOD.FULL,))
        self.save_licence(self.name, self.licence)
        return partition

    def _ingest_segments(self, release, partition, bbox, themes, road_classes, common) -> None:
        import pyarrow.compute as pc
        keep = None
        if road_classes is not None or "roads" not in themes or "rail" not in themes:
            parts = []
            if "roads" in themes:
                road = pc.field("subtype") == "road"
                parts.append(road & pc.field("class").isin(list(road_classes)) if road_classes is not None else road)
            if "rail" in themes:
                parts.append(pc.field("subtype") == "rail")
            keep = parts[0] if len(parts) == 1 else parts[0] | parts[1]
        seg = self._read(release, "transportation", "segment", bbox,
                         ["id", "subtype", "class", "names", "road_flags", "rail_flags", "geometry"], keep)
        geom = shapely.from_wkb(seg["geometry"].to_numpy())
        names = seg["names"].map(lambda n: n.get("primary") if isinstance(n, dict) else None)
        for subtype, theme, feature in (("road", "roads", Feature.STREET), ("rail", "rail", Feature.RAIL)):
            if theme not in themes:
                continue
            m = (seg["subtype"] == subtype).to_numpy()
            s = seg[m]
            if subtype == "road":
                cls = s["class"].fillna("unclassified").replace({"unknown": "unclassified"})
                rank = cls.map(STREET_RANK).fillna(0.05)
                flags = s["road_flags"]
            else:
                cls = s["class"].fillna("unknown").map(RAIL_CLASS).fillna("rail")
                rank = cls.map(RAIL_RANK).fillna(0.3)
                flags = s["rail_flags"]
            out = schema.normalise(
                gpd.GeoDataFrame(geometry=geom[m], crs=schema.CRS), feature=feature, **common,
                rank=rank.to_numpy(), name=names[m].to_numpy(), source_id=s["id"].to_numpy(),
                extras={"x_class": cls.to_numpy(object),
                        "x_bridge": flags.map(lambda f: _has_flag(f, "is_bridge")).to_numpy(bool),
                        "x_tunnel": flags.map(lambda f: _has_flag(f, "is_tunnel")).to_numpy(bool)},
            )
            self.store.write(out, self.name, feature, partition, lods=(LOD.FULL, LOD.MEDIUM))

    def _ingest_water(self, release, partition, bbox, common) -> None:
        wat = self._read(release, "base", "water", bbox, ["id", "subtype", "class", "names", "geometry"])
        geom = shapely.from_wkb(wat["geometry"].to_numpy())
        kinds = shapely.get_type_id(geom)
        names = wat["names"].map(lambda n: n.get("primary") if isinstance(n, dict) else None).to_numpy()
        cls = wat["class"].fillna(wat["subtype"]).fillna("water").to_numpy(object)
        false = np.zeros(len(wat), dtype=bool)
        for mask, feature in ((np.isin(kinds, [1, 5]), Feature.WATERWAY), (np.isin(kinds, [3, 6]), Feature.WATER)):
            rank = (np.array([WATERWAY_RANK.get(c, 0.2) for c in cls[mask]])
                    if feature is Feature.WATERWAY else np.full(mask.sum(), 0.5))
            out = schema.normalise(
                gpd.GeoDataFrame(geometry=geom[mask], crs=schema.CRS), feature=feature, **common,
                rank=rank, name=names[mask], source_id=wat["id"].to_numpy()[mask],
                extras={"x_class": cls[mask], "x_bridge": false[mask], "x_tunnel": false[mask]},
            )
            self.store.write(out, self.name, feature, partition, lods=(LOD.FULL, LOD.MEDIUM))
