"""Command line: ingest data, inspect the store, resolve regions, render Reels.

    geoquiz ingest --sources natural_earth,hydrosheds,ghsl,flags
    geoquiz ingest --sources geoboundaries --country CHE
    geoquiz ingest --sources osm --osm-extract europe/switzerland
    geoquiz inventory
    geoquiz resolve city Zürich
    geoquiz render country CHE
    geoquiz render city Zürich
    geoquiz spike                    # phase 0: Switzerland's rivers and Zürich's streets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .paths import DataPaths

SOURCES = ("natural_earth", "hydrosheds", "ghsl", "geoboundaries", "osm", "overture", "flags")


def _context(args):
    from .ingest import RawCache
    from .providers import default_registry
    from .regions import RegionResolver
    from .store import FeatureStore
    paths = DataPaths(Path(args.data).resolve()) if args.data else DataPaths.default()
    store = FeatureStore(paths.store)
    registry = default_registry(store)
    return paths, store, RawCache(paths.raw), registry, RegionResolver(registry)


def cmd_ingest(args) -> int:
    paths, store, raw, registry, resolver = _context(args)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - set(SOURCES)
    if unknown:
        sys.exit(f"unknown sources {sorted(unknown)}; choose from {', '.join(SOURCES)}")
    for name in sources:
        t = time.perf_counter()
        if name == "flags":
            from .assets import ingest_flags
            ingest_flags(raw, refresh=args.refresh)
        elif name == "hydrosheds":
            registry.get(name).ingest(raw, regions=tuple(args.hydro_regions.split(",")), refresh=args.refresh)
        elif name == "geoboundaries":
            if not args.country:
                sys.exit("geoboundaries needs --country ISO3")
            lic = registry.get(name).ingest(raw, args.country.upper(), tuple(args.adm.split(",")), refresh=args.refresh)
            for adm, l in lic.items():
                print(f"  {args.country.upper()} {adm}: licence '{l.name}', commercial use {'yes' if l.commercial_use else 'NO'}")
        elif name == "osm":
            if not (args.osm_extract or args.osm_pbf):
                sys.exit("osm needs --osm-extract (e.g. europe/switzerland) or --osm-pbf FILE")
            registry.get(name).ingest(raw, extract=args.osm_extract, pbf=args.osm_pbf,
                                      themes=tuple(args.osm_themes.split(",")), refresh=args.refresh)
        elif name == "overture":
            if not args.city:
                sys.exit("overture needs --city NAME (the bounding box comes from its GHSL urban centre)")
            from .regions import fold
            region = resolver.city(args.city)
            registry.get(name).ingest(raw, fold(region.name).replace(" ", "-"), region.bbox(3000.0))
        else:
            registry.get(name).ingest(raw, refresh=args.refresh)
        print(f"{name}: done in {time.perf_counter() - t:.1f} s")
    return 0


def cmd_inventory(args) -> int:
    from . import licences
    _, store, *_ = _context(args)
    inv = store.inventory()
    if inv.empty:
        print("store is empty: run `geoquiz ingest` first")
        return 0
    inv["MB"] = (inv.pop("bytes") / 1e6).round(1)
    print(inv.to_string(index=False))
    print("\nlicences:")
    for key, l in sorted(licences.known().items()):
        flags = ", ".join(f for f, on in (("share-alike", l.share_alike), ("NON-COMMERCIAL", not l.commercial_use)) if on)
        print(f"  {key}: {l.name}" + (f" [{flags}]" if flags else ""))
    return 0


def cmd_resolve(args) -> int:
    *_, registry, resolver = _context(args)
    from .schema import Feature
    region = resolver.country(args.key) if args.kind == "country" else resolver.city(args.key)
    out = {"kind": region.kind.value, "id": region.id, "name": region.name, "country": region.country,
           "centre_lonlat": region.centre, "crs": region.crs, "population": region.population,
           "area_km2": round(region.area_km2(), 1), "bbox": region.bbox()}
    feats = (Feature.RIVER, Feature.LAKE) if args.kind == "country" else (Feature.STREET, Feature.RAIL, Feature.WATER)
    out["providers"] = {f.value: [{"provider": c.provider, "coverage": round(c.coverage, 2), "chosen": c.chosen}
                                  for c in registry.explain(f, region)] for f in feats}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _render(args, kind: str, key: str, index: int | None = None) -> dict:
    from . import puzzles
    from .reel import ReelSpec, render_reel
    from .render.style import STYLES
    paths, _, _, registry, resolver = _context(args)
    style = STYLES[args.style]
    t = time.perf_counter()
    if kind == "country":
        puzzle = puzzles.country_rivers(registry, resolver, key, style=style, outside=args.outside)
    else:
        puzzle = puzzles.city_streets(registry, resolver, key, style=style, budget_km=args.budget_km)
    build_s = time.perf_counter() - t
    out = Path(args.out) if args.out else paths.out
    meta = render_reel(puzzle, out, raw_root=paths.raw, land=registry.get("natural_earth").land(),
                       spec=ReelSpec(duration=args.duration), index=index)
    tm = meta["timing"]
    print(f"{meta['folder']}: scene {build_s:.1f} s, setup {tm['setup_s']} s, "
          f"{meta['video']['frames']} frames in {tm['render_and_encode_s']} s "
          f"({tm['frame_ms_median']} ms/frame median), {meta['video']['bytes'] / 1e6:.1f} MB")
    print(f"  credit: {meta['attribution']['on_screen']}")
    return meta


def cmd_render(args) -> int:
    _render(args, args.kind, args.key)
    return 0


def cmd_spike(args) -> int:
    _render(args, "country", "CHE", index=1)
    _render(args, "city", "Zürich", index=2)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geoquiz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data root (default: $GEOQUIZ_DATA or ./geoquiz_data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="download sources and normalise them into the store")
    p.add_argument("--sources", default="natural_earth,hydrosheds,ghsl,flags", help=f"comma list of: {', '.join(SOURCES)}")
    p.add_argument("--hydro-regions", default="eu", help="HydroRIVERS regions: af,ar,as,au,eu,gr,na,sa,si")
    p.add_argument("--country", help="ISO3 code for geoboundaries")
    p.add_argument("--adm", default="ADM0,ADM1", help="geoBoundaries levels")
    p.add_argument("--osm-extract", help="Geofabrik extract id, e.g. europe/switzerland")
    p.add_argument("--osm-pbf", help="local .osm.pbf instead of a download")
    p.add_argument("--osm-themes", default="streets,rail,waterways,water", help="also available: buildings")
    p.add_argument("--city", help="city name for overture (bounding box of its GHSL urban centre)")
    p.add_argument("--refresh", action="store_true", help="re-download even if already cached")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("inventory", help="list what the store holds, and the licences in play")
    p.set_defaults(fn=cmd_inventory)

    p = sub.add_parser("resolve", help="resolve a country or city and show which providers cover it")
    p.add_argument("kind", choices=["country", "city"])
    p.add_argument("key")
    p.set_defaults(fn=cmd_resolve)

    def render_opts(p):
        p.add_argument("--out", help="output directory (default: <data>/out)")
        p.add_argument("--style", default="neon", choices=["neon"])
        p.add_argument("--duration", type=float, default=15.0, help="seconds")
        p.add_argument("--outside", default="dim", choices=["dim", "cut"],
                       help="country mode: dim features beyond the border (harder) or cut them (easier)")
        p.add_argument("--budget-km", type=float, default=350.0, help="city mode: km of streets the window holds")

    p = sub.add_parser("render", help="render one Reel")
    p.add_argument("kind", choices=["country", "city"])
    p.add_argument("key")
    render_opts(p)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("spike", help="phase 0: Switzerland's rivers and Zürich's streets")
    render_opts(p)
    p.set_defaults(fn=cmd_spike)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
