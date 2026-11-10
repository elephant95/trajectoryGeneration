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
        Feature.STREET: {Scale.CITY},
        Feature.RAIL: {Scale.CITY},
        Feature.WATERWAY: {Scale.CITY},
        Feature.WATER: {Scale.CITY},
    }

    @staticmethod
    def _fs():
        import pyarrow.fs as fs
        return fs.S3FileSystem(anonymous=True, region=REGION)

    def latest_release(self) -> str:
        import pyarrow.fs as fs
        infos = self._fs().get_file_info(fs.FileSelector(f"{BUCKET}/release/", recursive=False))
        return sorted(i.base_name for i in infos)[-1]

    def _read(self, release: str, theme: str, type_: str, bbox, columns: list[str]):
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        d = ds.dataset(f"{BUCKET}/release/{release}/theme={theme}/type={type_}/", filesystem=self._fs(), format="parquet")
        w, s, e, n = bbox
        flt = ((pc.field("bbox", "xmin") < e) & (pc.field("bbox", "xmax") > w)
               & (pc.field("bbox", "ymin") < n) & (pc.field("bbox", "ymax") > s))
        return d.to_table(filter=flt, columns=columns).to_pandas()

    def ingest(self, raw: RawCache, partition: str, bbox: tuple[float, float, float, float],
               release: str | None = None, refresh: bool = False) -> str:
        """Pull one bounding box (west, south, east, north) into the store as ``partition``."""
        release = release or self.latest_release()
        common = dict(source=self.name, licence=self.name, attribution=self.licence.attribution, data_version=release)

        seg = self._read(release, "transportation", "segment", bbox,
                         ["id", "subtype", "class", "names", "road_flags", "rail_flags", "geometry"])
        geom = shapely.from_wkb(seg["geometry"].to_numpy())
        names = seg["names"].map(lambda n: n.get("primary") if isinstance(n, dict) else None)
        for subtype, feature in (("road", Feature.STREET), ("rail", Feature.RAIL)):
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
        self.save_licence(self.name, self.licence)
        return partition
