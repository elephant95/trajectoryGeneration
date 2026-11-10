"""Provider interface and registry.

Every data source sits behind :class:`FeatureProvider` and returns the common
schema, so the renderer sees geometry, a feature type and a rank, never the
source. The registry picks the best source per feature and scale from a
priority table and falls back when a source's coverage of the region is poor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import geopandas as gpd
import numpy as np

from .. import licences, schema
from ..licences import Licence
from ..regions import Region
from ..schema import LOD, Feature, Scale
from ..store import FeatureStore


@runtime_checkable
class FeatureProvider(Protocol):
    name: str
    licence: Licence

    def supports(self, feature: Feature, scale: Scale) -> bool: ...
    def coverage(self, feature: Feature, region: Region) -> float: ...   # 0..1 quality estimate
    def fetch(self, feature: Feature, region: Region, lod: LOD) -> gpd.GeoDataFrame: ...


#: Line density (km per km²) at which a layer counts as fully covering a region.
#: Starting values (estimates) to tune once the planner has real candidates.
REFERENCE_DENSITY = {
    Feature.RIVER: 0.10,
    Feature.STREET: 5.0,
    Feature.RAIL: 0.05,
    Feature.WATERWAY: 0.05,
}


class StoreProvider:
    """Base class for providers backed by the local feature store."""

    name: ClassVar[str]
    licence: ClassVar[Licence]
    #: feature -> scales this source is good for
    features: ClassVar[dict[Feature, set[Scale]]] = {}
    #: default extraction buffer around the region, so features run off the frame naturally
    buffer_m: ClassVar[dict[Scale, float]] = {Scale.COUNTRY: 15_000.0, Scale.CITY: 500.0}

    def __init__(self, store: FeatureStore):
        self.store = store

    def supports(self, feature: Feature, scale: Scale) -> bool:
        return scale in self.features.get(Feature(feature), set())

    def fetch(self, feature: Feature, region: Region, lod: LOD = LOD.FULL, buffer_m: float | None = None) -> gpd.GeoDataFrame:
        buf = self.buffer_m[region.kind] if buffer_m is None else buffer_m
        return self.store.read(self.name, feature, lod=lod, bbox=region.bbox(buf))

    def coverage(self, feature: Feature, region: Region) -> float:
        """0 when the store holds nothing for the region; for line layers, density against a reference."""
        feature = Feature(feature)
        gdf = self.fetch(feature, region, LOD.MEDIUM, buffer_m=0.0)
        if gdf.empty:
            return 0.0
        ref = REFERENCE_DENSITY.get(feature)
        if ref is None:
            return 1.0
        x0, y0, x1, y1 = region.extent_xy()
        clip = region.projected() if not region.window_radius_m else None
        proj = gdf.geometry.to_crs(region.crs)
        proj = proj.clip_by_rect(x0, y0, x1, y1) if clip is None else proj.intersection(clip)
        density = (proj.length.sum() / 1000.0) / max(region.area_km2(), 1e-9)
        return float(np.clip(density / ref, 0.0, 1.0))

    # -- licence persistence ------------------------------------------------
    def save_licence(self, key: str, licence: Licence) -> None:
        """Record a licence in the store so later processes can resolve the key."""
        path = self.store.root / "_licences.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        data[key] = licence.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        licences.register(licence, key)


def load_store_licences(store: FeatureStore) -> None:
    path = store.root / "_licences.json"
    if path.exists():
        for key, d in json.loads(path.read_text()).items():
            licences.register(Licence(**d), key)


#: Source preference per (feature, scale), best first.
PRIORITY: dict[tuple[Feature, Scale], list[str]] = {
    (Feature.RIVER, Scale.COUNTRY): ["hydrosheds", "natural_earth"],
    (Feature.LAKE, Scale.COUNTRY): ["hydrosheds", "natural_earth"],
    (Feature.COASTLINE, Scale.COUNTRY): ["natural_earth"],
    (Feature.LAND, Scale.COUNTRY): ["natural_earth"],
    (Feature.COUNTRY, Scale.COUNTRY): ["natural_earth", "geoboundaries"],
    (Feature.ADMIN, Scale.COUNTRY): ["geoboundaries"],
    (Feature.URBAN_CENTRE, Scale.COUNTRY): ["ghsl"],
    (Feature.URBAN_CENTRE, Scale.CITY): ["ghsl"],
    (Feature.STREET, Scale.CITY): ["osm", "overture"],
    (Feature.RAIL, Scale.CITY): ["osm", "overture"],
    (Feature.RAIL, Scale.COUNTRY): ["osm"],
    (Feature.WATERWAY, Scale.CITY): ["osm", "overture"],
    (Feature.WATER, Scale.CITY): ["osm", "overture"],
    (Feature.BUILDING, Scale.CITY): ["osm"],
    (Feature.LAKE, Scale.CITY): ["hydrosheds"],
}


@dataclass(frozen=True)
class Choice:
    provider: str
    coverage: float
    chosen: bool


class ProviderRegistry:
    def __init__(self, providers: list[FeatureProvider], priority: dict | None = None, min_coverage: float = 0.5):
        self.providers = {p.name: p for p in providers}
        self.priority = priority or PRIORITY
        self.min_coverage = min_coverage

    def get(self, name: str) -> FeatureProvider:
        return self.providers[name]

    def candidates(self, feature: Feature, scale: Scale) -> list[FeatureProvider]:
        names = self.priority.get((Feature(feature), Scale(scale)), [])
        return [self.providers[n] for n in names if n in self.providers and self.providers[n].supports(feature, scale)]

    def explain(self, feature: Feature, region: Region) -> list[Choice]:
        """Coverage of every candidate, and which one :meth:`fetch` would use.

        The first source in priority order that reaches ``min_coverage`` wins;
        if none does, the one with the best coverage is used.
        """
        cands = self.candidates(feature, region.kind)
        cov = [p.coverage(feature, region) for p in cands]
        pick = next((i for i, c in enumerate(cov) if c >= self.min_coverage), None)
        if pick is None and cands and max(cov) > 0:
            pick = int(np.argmax(cov))
        return [Choice(p.name, c, i == pick) for i, (p, c) in enumerate(zip(cands, cov))]

    def fetch(self, feature: Feature, region: Region, lod: LOD = LOD.FULL, provider: str | None = None) -> gpd.GeoDataFrame:
        if provider is None:
            chosen = [c.provider for c in self.explain(feature, region) if c.chosen]
            if not chosen:
                return schema.empty()
            provider = chosen[0]
        return self.providers[provider].fetch(feature, region, lod)
