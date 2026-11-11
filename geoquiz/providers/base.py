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
import shapely

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
    (Feature.RIVER, Scale.COUNTRY): 0.10,
    (Feature.STREET, Scale.CITY): 5.0,
    (Feature.STREET, Scale.COUNTRY): 0.05,     # major roads only
    (Feature.RAIL, Scale.CITY): 0.05,
    (Feature.RAIL, Scale.COUNTRY): 0.02,
    (Feature.WATERWAY, Scale.CITY): 0.05,
}


#: Features sampled when estimating coverage of a large region.
COVERAGE_SAMPLE = 40_000


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

    def partitions_for(self, feature: Feature, region: Region) -> list[str] | None:
        """Partitions that may answer a request; None means all of them."""
        return None

    def fetch(self, feature: Feature, region: Region, lod: LOD = LOD.FULL, buffer_m: float | None = None,
              bbox: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
        """Features in the region (plus a buffer), or in an explicit WGS84 ``bbox``."""
        if bbox is None:
            bbox = region.bbox(self.buffer_m[region.kind] if buffer_m is None else buffer_m)
        out = self.store.read(self.name, feature, self.partitions_for(Feature(feature), region), lod=lod, bbox=bbox)
        if len(out) and out["source_id"].notna().all():
            # the same feature can sit in two partitions (a city pull inside a country pull)
            out = out.drop_duplicates("source_id").reset_index(drop=True)
        return out

    def coverage(self, feature: Feature, region: Region) -> float:
        """0 when the store holds nothing for the region; for line layers, density against a reference."""
        feature = Feature(feature)
        ref = REFERENCE_DENSITY.get((feature, region.kind))
        gdf = self.fetch(feature, region, LOD.COARSE if region.kind is Scale.COUNTRY else LOD.MEDIUM, buffer_m=0.0)
        if gdf.empty:
            return 0.0
        if ref is None:
            return 1.0
        scale_up = 1.0
        if len(gdf) > COVERAGE_SAMPLE:                     # an estimate: a sample is enough
            scale_up = len(gdf) / COVERAGE_SAMPLE
            gdf = gdf.sample(COVERAGE_SAMPLE, random_state=0)
        x0, y0, x1, y1 = region.extent_xy()
        proj = gdf.geometry.to_crs(region.crs)
        if region.window_radius_m:
            proj = proj.clip_by_rect(x0, y0, x1, y1)
        else:
            outline = shapely.simplify(region.projected(), 1000.0)
            shapely.prepare(outline)
            proj = proj[shapely.intersects(outline, proj.to_numpy())]     # whole features touching the target
        density = (proj.length.sum() * scale_up / 1000.0) / max(region.area_km2(), 1e-9)
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
    (Feature.RAIL, Scale.COUNTRY): ["osm", "overture"],
    (Feature.STREET, Scale.COUNTRY): ["osm", "overture"],
    (Feature.WATERWAY, Scale.CITY): ["osm", "overture"],
    (Feature.WATER, Scale.CITY): ["osm", "overture"],
    (Feature.BUILDING, Scale.CITY): ["osm", "overture"],
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
        self._coverage: dict = {}

    def get(self, name: str) -> FeatureProvider:
        return self.providers[name]

    def forget_coverage(self) -> None:
        """Call after ingesting: cached coverage estimates no longer hold."""
        self._coverage.clear()

    def candidates(self, feature: Feature, scale: Scale) -> list[FeatureProvider]:
        names = self.priority.get((Feature(feature), Scale(scale)), [])
        return [self.providers[n] for n in names if n in self.providers and self.providers[n].supports(feature, scale)]

    def explain(self, feature: Feature, region: Region) -> list[Choice]:
        """Coverage of every candidate, and which one :meth:`fetch` would use.

        The first source in priority order that reaches ``min_coverage`` wins;
        if none does, the one with the best coverage is used.
        """
        cands = self.candidates(feature, region.kind)
        cov = []
        for p in cands:                                   # coverage reads the store: compute once per target
            k = (p.name, Feature(feature), region.id, region.window_radius_m, region.centre)
            if k not in self._coverage:
                self._coverage[k] = p.coverage(feature, region)
            cov.append(self._coverage[k])
        pick = next((i for i, c in enumerate(cov) if c >= self.min_coverage), None)
        if pick is None and cands and max(cov) > 0:
            pick = int(np.argmax(cov))
        return [Choice(p.name, c, i == pick) for i, (p, c) in enumerate(zip(cands, cov))]

    def fetch(self, feature: Feature, region: Region, lod: LOD = LOD.FULL, provider: str | None = None,
              bbox: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
        """Fetch from the chosen provider. ``bbox`` overrides the region's own extent (the provider is
        still chosen by its coverage of the region)."""
        if provider is None:
            chosen = [c.provider for c in self.explain(feature, region) if c.chosen]
            if not chosen:
                # nothing inside the region itself (a landlocked country's coast): the wider frame may still have some
                for cand in self.candidates(feature, region.kind) if bbox is not None else []:
                    out = cand.fetch(feature, region, lod, bbox=bbox)
                    if len(out):
                        return out
                return schema.empty()
            provider = chosen[0]
        return self.providers[provider].fetch(feature, region, lod, bbox=bbox)
