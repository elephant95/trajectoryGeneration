"""Provider adapters: one per data source, all returning the common schema."""
from __future__ import annotations

from ..store import FeatureStore
from .base import FeatureProvider, ProviderRegistry, StoreProvider, load_store_licences
from .geoboundaries import GeoBoundariesProvider
from .ghsl import GhslProvider
from .hydrosheds import HydroShedsProvider
from .natural_earth import NaturalEarthProvider
from .osm import OsmProvider
from .overture import OvertureProvider

ALL = (NaturalEarthProvider, HydroShedsProvider, OsmProvider, OvertureProvider, GeoBoundariesProvider, GhslProvider)


def default_registry(store: FeatureStore) -> ProviderRegistry:
    load_store_licences(store)
    return ProviderRegistry([cls(store) for cls in ALL])


__all__ = ["FeatureProvider", "ProviderRegistry", "StoreProvider", "default_registry",
           "NaturalEarthProvider", "HydroShedsProvider", "OsmProvider", "OvertureProvider", "GeoBoundariesProvider", "GhslProvider"]
