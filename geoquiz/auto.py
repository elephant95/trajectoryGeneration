"""Fetch whatever a puzzle needs that the store does not hold yet.

Ingest stays a separate, explicit step by default. With ``ingest_missing``
the renderer may call this first: it downloads once per target (never per
frame) and then renders from the local store as usual.
"""
from __future__ import annotations

from .assets import ingest_flags
from .ingest import RawCache
from .providers.base import ProviderRegistry
from .providers.overture import COUNTRY_PREFIX, MAJOR_ROADS
from .puzzles import DEFAULT_FEATURES, LAYERS, country_frame_bbox
from .regions import RegionResolver, fold
from .schema import Feature, Scale

#: Natural Earth continent -> HydroRIVERS regional tiles that cover it.
HYDRO_REGIONS = {
    "Europe": ("eu", "si"), "Asia": ("as", "si", "eu"), "Africa": ("af",), "North America": ("na", "ar", "gr"),
    "South America": ("sa",), "Oceania": ("au",),
}


def partition_name(text: str) -> str:
    return fold(text).replace(" ", "-").replace("/", "-")


def ensure_basics(registry: ProviderRegistry, raw: RawCache, cities: bool = False, log=print) -> list[str]:
    """The data every puzzle needs: countries and flags, and for city work the GHSL city list."""
    done = []
    store = registry.get("natural_earth").store
    if not store.has("natural_earth", Feature.COUNTRY):
        log("first use: downloading Natural Earth (countries, coastline, about 15 MB) …")
        registry.get("natural_earth").ingest(raw)
        done.append("natural_earth")
    if not (raw.root / "flag_icons" / "4x3").exists():
        try:
            ingest_flags(raw)
            done.append("flags")
        except Exception as e:                       # flags are decoration: never block a puzzle on them
            log(f"could not download flags ({e}); answer cards will have no flag")
    if cities and not store.has("ghsl", Feature.URBAN_CENTRE):
        log("first use: downloading the GHSL city list (about 310 MB, once) …")
        registry.get("ghsl").ingest(raw)
        done.append("ghsl")
    if done:
        registry.forget_coverage()
    return done


def ensure_data(registry: ProviderRegistry, resolver: RegionResolver, raw: RawCache, kind: str, key: str,
                features: tuple[str, ...] | None = None, country: str | None = None, log=print) -> list[str]:
    """Ingest what is missing for this puzzle. Returns a description of each download made."""
    scale = Scale(kind)
    features = tuple(features) if features else DEFAULT_FEATURES[scale]
    specs = LAYERS[scale]
    done = []
    done += ensure_basics(registry, raw, cities=scale is Scale.CITY or "cities" in features, log=log)
    store = registry.get("natural_earth").store
    region = resolver.country(key) if scale is Scale.COUNTRY else resolver.city(key, country)

    def lacking(name: str) -> bool:
        return name in features and not any(c.coverage > 0 for c in registry.explain(specs[name].feature, region))

    if scale is Scale.CITY:
        themes = [t for t, names in (("roads", ("streets",)), ("rail", ("rail",)), ("water", ("water", "waterways")),
                                     ("buildings", ("buildings",))) if any(lacking(n) for n in names)]
        if lacking("streets") or (themes and "roads" not in themes and
                                  not any(c.coverage > 0 for c in registry.explain(Feature.STREET, region))):
            themes = sorted(set(themes) | {"roads"})          # streets size the window even when not drawn
        if themes:
            log(f"ingesting Overture {', '.join(themes)} for {region.name} …")
            part = partition_name(region.name) + ("" if set(themes) >= {"roads", "rail", "water"} else "-" + "-".join(themes))
            registry.get("overture").ingest(raw, part, region.bbox(3000.0), themes=tuple(themes))
            done.append(f"overture:{part}")
            registry.forget_coverage()
        return done

    if lacking("rivers") or lacking("lakes"):
        table = registry.get("natural_earth").countries()
        continent = table.loc[table["source_id"] == region.id, "x_continent"].iloc[0]
        have = set(store.partitions("hydrosheds", Feature.RIVER))
        todo = tuple(r for r in HYDRO_REGIONS.get(continent, ()) if r not in have)
        if todo:
            log(f"ingesting HydroSHEDS regions {', '.join(todo)} …")
            registry.get("hydrosheds").ingest(raw, regions=todo)
            done.append(f"hydrosheds:{','.join(todo)}")
    themes = [t for t, n in (("roads", "roads"), ("rail", "rail")) if lacking(n)]
    if themes:
        log(f"ingesting Overture {', '.join(themes)} for {region.name} (whole-country read; large countries take a while) …")
        part = f"{COUNTRY_PREFIX}{region.id.lower()}-{'-'.join(themes)}"
        registry.get("overture").ingest(raw, part, country_frame_bbox(region), themes=tuple(themes), road_classes=MAJOR_ROADS)
        done.append(f"overture:{part}")
    if "borders" in features and not store.has("geoboundaries", Feature.ADMIN, f"{region.id}-ADM1"):
        log(f"ingesting geoBoundaries ADM1 for {region.id} …")
        registry.get("geoboundaries").ingest(raw, region.id, ("ADM1",))
        done.append(f"geoboundaries:{region.id}")
    if done:
        registry.forget_coverage()
    return done
