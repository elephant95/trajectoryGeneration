"""Natural Earth: silhouettes, coastlines, major rivers and lakes, locator globe."""
from __future__ import annotations

import numpy as np
import pyogrio

from .. import licences, schema
from ..ingest import RawCache
from ..schema import LOD, Feature, Scale
from .base import StoreProvider

CDN = "https://naciscdn.org/naturalearth"

#: (feature, partition) -> path under the CDN
LAYERS = {
    (Feature.RIVER, "global"): "10m/physical/ne_10m_rivers_lake_centerlines",
    (Feature.RIVER, "europe"): "10m/physical/ne_10m_rivers_europe",
    (Feature.RIVER, "north_america"): "10m/physical/ne_10m_rivers_north_america",
    (Feature.RIVER, "australia"): "10m/physical/ne_10m_rivers_australia",
    (Feature.LAKE, "global"): "10m/physical/ne_10m_lakes",
    (Feature.LAKE, "europe"): "10m/physical/ne_10m_lakes_europe",
    (Feature.LAKE, "north_america"): "10m/physical/ne_10m_lakes_north_america",
    (Feature.LAKE, "australia"): "10m/physical/ne_10m_lakes_australia",
    (Feature.COASTLINE, "global"): "10m/physical/ne_10m_coastline",
    (Feature.COUNTRY, "global"): "10m/cultural/ne_10m_admin_0_countries",
    (Feature.LAND, "global"): "110m/physical/ne_110m_land",
}


class NaturalEarthProvider(StoreProvider):
    name = "natural_earth"
    licence = licences.NATURAL_EARTH
    features = {
        Feature.RIVER: {Scale.COUNTRY},
        Feature.LAKE: {Scale.COUNTRY},
        Feature.COASTLINE: {Scale.COUNTRY},
        Feature.COUNTRY: {Scale.COUNTRY},
        Feature.LAND: {Scale.COUNTRY},
    }

    def ingest(self, raw: RawCache, refresh: bool = False) -> None:
        for (feature, partition), rel in LAYERS.items():
            archive = raw.fetch(self.name, f"{CDN}/{rel}.zip", refresh=refresh)
            d = raw.extract(archive)
            stem = rel.rsplit("/", 1)[-1]
            version_file = d / f"{stem}.VERSION.txt"
            version = version_file.read_text().strip() if version_file.exists() else "unknown"
            gdf = pyogrio.read_dataframe(d / f"{stem}.shp")
            gdf.columns = [c.lower() for c in gdf.columns]
            common = dict(feature=feature, source=self.name, licence=self.name,
                          attribution=self.licence.attribution, data_version=version)
            if feature is Feature.COUNTRY:
                pop = gdf["pop_est"].fillna(0).clip(lower=1)
                out = schema.normalise(
                    gdf, **common, name=gdf["name"], source_id=gdf["adm0_a3"],
                    rank=(np.log10(pop) - 3) / 6.2,
                    extras={"x_iso_a2": gdf["iso_a2_eh"].where(gdf["iso_a2_eh"] != "-99", None).to_numpy(),
                            "x_pop": gdf["pop_est"].to_numpy(float),
                            "x_continent": gdf["continent"].to_numpy()},
                )
            else:
                rank = 1 - gdf["scalerank"].astype(float) / 12 if "scalerank" in gdf else 0.5
                name = gdf["name"] if "name" in gdf else None
                sid = gdf["ne_id"] if "ne_id" in gdf else np.arange(len(gdf))
                out = schema.normalise(gdf, **common, name=name, source_id=sid, rank=rank)
            lods = (LOD.FULL,) if feature is Feature.LAND else (LOD.FULL, LOD.MEDIUM, LOD.COARSE)
            self.store.write(out, self.name, feature, partition, lods=lods)
        self.save_licence(self.name, self.licence)

    def countries(self, lod: LOD = LOD.MEDIUM):
        if not self.store.has(self.name, Feature.COUNTRY):
            raise ValueError("Natural Earth is not in the data store yet. Run: geoquiz ingest --sources natural_earth,flags "
                             "(or use ingest_missing=True / --ingest-missing)")
        return self.store.read(self.name, Feature.COUNTRY, lod=lod)

    def land(self):
        return self.store.read(self.name, Feature.LAND)
