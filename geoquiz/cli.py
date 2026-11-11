"""Command line: ingest data, inspect the store, render Reels one at a time or in batches.

    geoquiz ingest                                   # Natural Earth, HydroSHEDS (Europe), GHSL, flags
    geoquiz ingest --sources hydrosheds --hydro-regions na,sa
    geoquiz ingest --sources overture --city Paris   # streets, rail and water for one city
    geoquiz ingest --sources overture --country AUT  # main roads and rail for one country
    geoquiz list features | styles | countries | cities --in France
    geoquiz render country Brazil
    geoquiz render country Austria --features roads,rail --style blueprint --ingest-missing
    geoquiz render city Paris --features streets,water --style ink --ingest-missing
    geoquiz batch batch.yaml                         # many Reels from one config file
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


def _csv(text: str | None) -> tuple[str, ...] | None:
    return tuple(t.strip() for t in text.split(",") if t.strip()) if text else None


# -- ingest -----------------------------------------------------------------------------

def cmd_ingest(args) -> int:
    paths, store, raw, registry, resolver = _context(args)
    sources = _csv(args.sources)
    unknown = set(sources) - set(SOURCES)
    if unknown:
        sys.exit(f"unknown sources {sorted(unknown)}; choose from {', '.join(SOURCES)}")
    for name in sources:
        t = time.perf_counter()
        if name == "flags":
            from .assets import ingest_flags
            ingest_flags(raw, refresh=args.refresh)
        elif name == "hydrosheds":
            regions = _csv(args.hydro_regions)
            if regions == ("all",):
                from .providers.hydrosheds import REGIONS as regions
            registry.get(name).ingest(raw, regions=tuple(regions), refresh=args.refresh)
        elif name == "geoboundaries":
            if not args.country:
                sys.exit("geoboundaries needs --country ISO3")
            lic = registry.get(name).ingest(raw, args.country.upper(), _csv(args.adm), refresh=args.refresh)
            for adm, l in lic.items():
                print(f"  {args.country.upper()} {adm}: licence '{l.name}', commercial use {'yes' if l.commercial_use else 'NO'}")
        elif name == "osm":
            if not (args.osm_extract or args.osm_pbf):
                sys.exit("osm needs --osm-extract (e.g. europe/switzerland) or --osm-pbf FILE")
            registry.get(name).ingest(raw, extract=args.osm_extract, pbf=args.osm_pbf,
                                      themes=_csv(args.osm_themes), refresh=args.refresh)
        elif name == "overture":
            from .auto import partition_name
            from .providers.overture import MAJOR_ROADS
            if args.city:
                region = resolver.city(args.city, args.in_country)
                themes = _csv(args.overture_themes) or ("roads", "rail", "water")
                registry.get(name).ingest(raw, partition_name(region.name), region.bbox(3000.0), themes=themes)
            elif args.country:
                region = resolver.country(args.country)
                themes = _csv(args.overture_themes) or ("roads", "rail")
                from .puzzles import country_frame_bbox
                from .providers.overture import COUNTRY_PREFIX
                registry.get(name).ingest(raw, f"{COUNTRY_PREFIX}{region.id.lower()}-{'-'.join(themes)}", country_frame_bbox(region),
                                          themes=themes, road_classes=MAJOR_ROADS)
            else:
                sys.exit("overture needs --city NAME (all streets, rail, water) or --country ISO3 (main roads, rail)")
        else:
            registry.get(name).ingest(raw, refresh=args.refresh)
        print(f"{name}: done in {time.perf_counter() - t:.1f} s")
    return 0


# -- inspection -----------------------------------------------------------------------------

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


def cmd_list(args) -> int:
    from .puzzles import DEFAULT_FEATURES, LAYERS, OUTSIDE_MODES, REVEAL_MODES
    from .render.style import STYLES
    from .schema import Scale
    if args.what == "features":
        for scale in Scale:
            print(f"{scale.value} (default: {','.join(DEFAULT_FEATURES[scale])})")
            for s in LAYERS[scale].values():
                print(f"  {s.name:<10} {s.label:<18} {('get it with: ' + s.ingest_hint) if s.ingest_hint else ''}")
        print(f"reveal modes: {', '.join(REVEAL_MODES)}\noutside modes (country): {', '.join(OUTSIDE_MODES)}")
    elif args.what == "styles":
        for s in STYLES.values():
            print(f"{s.name:<10} {'light ink on dark' if s.mode == 'add' else 'dark ink on paper'}")
    else:
        *_, registry, resolver = _context(args)
        from .regions import fold
        if args.what == "countries":
            t = registry.get("natural_earth").countries()
            if args.in_:
                t = t[t["x_continent"].map(fold) == fold(args.in_)]
            for _, r in t.sort_values("name").iterrows():
                print(f"{r['source_id']}  {r['name']:<36} {r['x_continent']:<14} pop {r['x_pop']:>13,.0f}")
        else:
            t = registry.get("ghsl").urban_centres()
            if args.in_:
                t = t[t["x_country"].map(fold) == fold(args.in_)]
            t = t[t["x_pop"] >= args.min_pop].sort_values("x_pop", ascending=False).head(args.limit)
            for _, r in t.iterrows():
                print(f"{r['name']:<32} {r['x_country']:<28} pop {r['x_pop']:>12,.0f}")
    return 0


def cmd_resolve(args) -> int:
    *_, registry, resolver = _context(args)
    from .puzzles import LAYERS
    region = resolver.country(args.key) if args.kind == "country" else resolver.city(args.key)
    out = {"kind": region.kind.value, "id": region.id, "name": region.name, "country": region.country,
           "centre_lonlat": region.centre, "crs": region.crs, "population": region.population,
           "area_km2": round(region.area_km2(), 1), "bbox": region.bbox()}
    out["features"] = {name: [{"provider": c.provider, "coverage": round(c.coverage, 2), "chosen": c.chosen}
                              for c in registry.explain(s.feature, region)] for name, s in LAYERS[region.kind].items()}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


# -- rendering -------------------------------------------------------------------------------

#: Every per-puzzle setting, with its default. The same names work as CLI flags and as batch-file keys.
RENDER_DEFAULTS = {
    "features": None, "style": "neon", "outside": "keep", "padding": 0.12, "buffer_km": 15.0, "budget_km": 350.0,
    "radius_km": None, "density": 1.0, "reveal": "auto", "country": None, "question": None, "hint": None,
    "answer": None, "fact": None, "duration": 15.0, "phases": None, "size": "1080x1920", "fps": 30, "zoom": 0.05,
    "crf": 18, "audio": None, "colors": None, "ingest_missing": False,
}


def render_one(ctx, kind: str, key: str, settings: dict, out: Path, index: int | None = None) -> dict:
    """Render one puzzle from a settings dict (see RENDER_DEFAULTS)."""
    from . import puzzles
    from .reel import ReelSpec, render_reel
    from .render.style import STYLES, with_overrides
    from .scene import Frame
    paths, _, raw, registry, resolver = ctx
    unknown = set(settings) - set(RENDER_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}; allowed: {', '.join(RENDER_DEFAULTS)}")
    s = {**RENDER_DEFAULTS, **settings}
    if s["style"] not in STYLES:
        raise ValueError(f"unknown style {s['style']!r}; choose from {', '.join(STYLES)}")
    style = with_overrides(STYLES[s["style"]], s["colors"]) if s["colors"] else STYLES[s["style"]]
    features = _csv(s["features"]) if isinstance(s["features"], str) else (tuple(s["features"]) if s["features"] else None)
    w, h = (int(v) for v in str(s["size"]).lower().split("x"))
    frame = Frame(width=w, height=h, fps=int(s["fps"]))
    question = s["question"]
    if isinstance(question, str):
        question = tuple(q.strip() for q in question.split("|"))
    options = puzzles.PuzzleOptions(
        outside=s["outside"], padding=float(s["padding"]), buffer_km=float(s["buffer_km"]),
        budget_km=float(s["budget_km"]), radius_km=float(s["radius_km"]) if s["radius_km"] else None,
        density=float(s["density"]), reveal=s["reveal"], country=s["country"],
        question=tuple(question) if question else None, hint=s["hint"], answer=s["answer"], fact=s["fact"])
    if s["ingest_missing"]:
        from .auto import ensure_data
        ensure_data(registry, resolver, raw, kind, key, features, s["country"])
    t = time.perf_counter()
    puzzle = puzzles.build_puzzle(registry, resolver, kind, key, features, frame, style, options)
    build_s = time.perf_counter() - t
    phases = tuple(float(p) for p in (s["phases"].split(",") if isinstance(s["phases"], str) else s["phases"])) if s["phases"] else None
    spec = ReelSpec(duration=float(s["duration"]), phases=phases, crf=int(s["crf"]), zoom=float(s["zoom"]), audio=s["audio"])
    land = registry.get("natural_earth").land()
    meta = render_reel(puzzle, out, raw_root=paths.raw, land=land, spec=spec, index=index)
    tm = meta["timing"]
    print(f"{meta['folder']}: scene {build_s:.1f} s, {meta['video']['frames']} frames in {tm['render_and_encode_s']} s "
          f"({tm['frame_ms_median']} ms/frame), {meta['video']['bytes'] / 1e6:.1f} MB | {meta['attribution']['on_screen']}")
    return meta


def _settings_from_args(args) -> dict:
    return {k: getattr(args, k) for k in RENDER_DEFAULTS if getattr(args, k, None) is not None}


def cmd_render(args) -> int:
    from .puzzles import MissingData
    ctx = _context(args)
    try:
        render_one(ctx, args.kind, args.key, _settings_from_args(args), Path(args.out) if args.out else ctx[0].out)
    except (MissingData, KeyError, ValueError) as e:
        sys.exit(f"error: {e.args[0] if e.args else e}")
    return 0


def cmd_spike(args) -> int:
    ctx = _context(args)
    out = Path(args.out) if args.out else ctx[0].out
    s = _settings_from_args(args)
    render_one(ctx, "country", "CHE", s, out, index=1)
    render_one(ctx, "city", "Zürich", {**s, "answer": s.get("answer", "Zürich")}, out, index=2)
    return 0


def cmd_batch(args) -> int:
    """Render every puzzle in a YAML config. See examples/batch.yaml."""
    import numpy as np
    import yaml
    from .puzzles import MissingData
    from .regions import fold
    ctx = _context(args)
    paths, _, _, registry, _ = ctx
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    defaults = cfg.get("defaults", {})
    out = Path(args.out or cfg.get("out") or paths.out)
    jobs = [dict(p) for p in cfg.get("puzzles", [])]
    for rnd in cfg.get("random", []) if isinstance(cfg.get("random"), list) else [cfg["random"]] if cfg.get("random") else []:
        rnd = dict(rnd)
        rng = np.random.default_rng(rnd.pop("seed", 0))
        kind, count = rnd.pop("kind"), int(rnd.pop("count"))
        min_pop, where = float(rnd.pop("min_population", 0)), rnd.pop("in", None)
        if kind == "country":
            t = registry.get("natural_earth").countries()
            t = t[t["x_pop"] >= max(min_pop, 1)]
            if where:
                t = t[t["x_continent"].map(fold).isin([fold(w) for w in ([where] if isinstance(where, str) else where)])]
            names = t["source_id"].tolist()
        else:
            t = registry.get("ghsl").urban_centres()
            t = t[t["x_pop"] >= min_pop]
            if where:
                t = t[t["x_country"].map(fold).isin([fold(w) for w in ([where] if isinstance(where, str) else where)])]
            names = t["name"].tolist()
            rnd.setdefault("ingest_missing", True)
        if not names:
            sys.exit(f"random: nothing matches kind={kind}, in={where}, min_population={min_pop}")
        for pick in rng.choice(len(names), size=min(count, len(names)), replace=False):
            job = {"kind": kind, "target": names[int(pick)], **rnd}
            if kind == "city":
                job.setdefault("country", t["x_country"].iloc[int(pick)])
            jobs.append(job)
    if not jobs:
        sys.exit("the config lists no puzzles")
    failed = []
    for i, job in enumerate(jobs, start=int(cfg.get("start_index", 1))):
        job = {**defaults, **job}
        kind, target = job.pop("kind"), str(job.pop("target"))
        try:
            render_one(ctx, kind, target, job, out, index=i)
        except (MissingData, KeyError, ValueError) as e:
            failed.append((kind, target, e.args[0] if e.args else str(e)))
            print(f"SKIPPED {kind} {target}: {failed[-1][2]}")
    print(f"{len(jobs) - len(failed)} of {len(jobs)} rendered into {out}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geoquiz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data root (default: $GEOQUIZ_DATA or ./geoquiz_data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="download sources and normalise them into the store")
    p.add_argument("--sources", default="natural_earth,hydrosheds,ghsl,flags", help=f"comma list of: {', '.join(SOURCES)}")
    p.add_argument("--hydro-regions", default="eu", help="HydroRIVERS regions: af,ar,as,au,eu,gr,na,sa,si or 'all'")
    p.add_argument("--country", help="country (ISO3 or name) for geoboundaries, or for overture main roads and rail")
    p.add_argument("--adm", default="ADM0,ADM1", help="geoBoundaries levels")
    p.add_argument("--osm-extract", help="Geofabrik extract id, e.g. europe/switzerland")
    p.add_argument("--osm-pbf", help="local .osm.pbf instead of a download")
    p.add_argument("--osm-themes", default="streets,rail,waterways,water", help="also available: buildings")
    p.add_argument("--city", help="city name for overture (bounding box of its GHSL urban centre)")
    p.add_argument("--in-country", help="country name, to disambiguate --city")
    p.add_argument("--overture-themes", help="subset of roads,rail,water,buildings")
    p.add_argument("--refresh", action="store_true", help="re-download even if already cached")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("inventory", help="list what the store holds, and the licences in play")
    p.set_defaults(fn=cmd_inventory)

    p = sub.add_parser("list", help="list features, styles, countries or cities")
    p.add_argument("what", choices=["features", "styles", "countries", "cities"])
    p.add_argument("--in", dest="in_", help="countries: a continent; cities: a country name")
    p.add_argument("--min-pop", type=float, default=0, help="cities: minimum population")
    p.add_argument("--limit", type=int, default=50, help="cities: how many to show")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("resolve", help="resolve a country or city and show which providers cover each feature")
    p.add_argument("kind", choices=["country", "city"])
    p.add_argument("key")
    p.set_defaults(fn=cmd_resolve)

    def render_opts(p):
        g = p.add_argument_group("what to draw")
        g.add_argument("--features", help="comma list, primary first (see `geoquiz list features`)")
        g.add_argument("--style", help="neon (default), ink, blueprint, ember, mono")
        g.add_argument("--reveal", help="auto (default), network, rank, radial, sweep")
        g.add_argument("--density", type=float, help="country: multiply the stroke budget (0.5 sparser, 2 denser)")
        g.add_argument("--ingest-missing", action="store_true", default=None, help="download data the store lacks for this target")
        g = p.add_argument_group("framing")
        g.add_argument("--outside", help="country: keep (default; square frame with neighbours, shape hidden until the "
                                         "answer), dim, or cut (easiest)")
        g.add_argument("--padding", type=float, help="country: surrounding area shown, as a fraction of its size (0.12)")
        g.add_argument("--budget-km", type=float, help="city: km of streets the window holds (350)")
        g.add_argument("--radius-km", type=float, help="city: fixed half-width of the window instead of a budget")
        g.add_argument("--country", help="city: country name, to disambiguate")
        g = p.add_argument_group("text")
        g.add_argument("--question", help="one or two lines separated by '|'")
        g.add_argument("--hint")
        g.add_argument("--answer", help="display name of the answer")
        g.add_argument("--fact")
        g = p.add_argument_group("video")
        g.add_argument("--duration", type=float, help="seconds (15); phases scale with it")
        g.add_argument("--phases", help="seconds of reveal,countdown,answer,outro, e.g. 10,3,4,1 (overrides --duration)")
        g.add_argument("--size", help="WIDTHxHEIGHT (1080x1920); the layout is designed for 9:16")
        g.add_argument("--fps", type=int)
        g.add_argument("--zoom", type=float, help="slow push-in over the clip (0.05); 0 turns it off")
        g.add_argument("--crf", type=int, help="x264 quality (18)")
        g.add_argument("--audio", help="music file to embed (looped or cut to length); silent otherwise")
        p.add_argument("--out", help="output directory (default: <data>/out)")

    p = sub.add_parser("render", help="render one Reel")
    p.add_argument("kind", choices=["country", "city"])
    p.add_argument("key", help="country name or ISO3 code, or city name")
    render_opts(p)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("spike", help="phase 0: Switzerland's rivers and Zürich's streets")
    render_opts(p)
    p.set_defaults(fn=cmd_spike)

    p = sub.add_parser("batch", help="render many Reels from a YAML config")
    p.add_argument("config")
    p.add_argument("--out", help="output directory (default: the config's `out`, else <data>/out)")
    p.set_defaults(fn=cmd_batch)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
