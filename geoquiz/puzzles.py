"""Puzzle builder: a target, a list of features and some options -> a Scene and its text.

One builder covers both modes. A country can be drawn from its rivers, lakes,
coast, roads, rail, internal borders or cities; a city from its streets, rail,
water, waterways or buildings; in any combination. The first feature in the
list is the *primary* one: it drives the reveal, the accent colour and the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import geopandas as gpd
import numpy as np
import shapely
import skia
from scipy.spatial import cKDTree

from . import licences
from .providers.base import ProviderRegistry
from .regions import Region, RegionResolver, fit_city_window, shp_transform
from .render.overlay import Texts, map_box
from .render.style import NEON, Style
from .scene import (FillLayer, Frame, LineLayer, Scene, View, network_distance, normalise_times, outline_times,
                    path_lengths, project_lines, project_polygons, stroke_widths, upstream_distance)
from .schema import LOD, Feature, Scale


@dataclass(frozen=True)
class LayerSpec:
    name: str                      # what the user asks for: "rivers", "roads", ...
    feature: Feature
    geom: str                      # "line" | "fill" | "outline" (polygon boundaries) | "point"
    label: str                     # wording for hints and facts
    z: int                         # draw order, back to front
    min_rank: float = 0.0
    widths: tuple[float, float] = (1.0, 1.0)
    gamma: float = 1.5
    opacity: float = 1.0
    budget_px: float | None = None   # total stroke length kept (most important first); None keeps everything
    network: bool = False            # a connected network: reveal by distance along it
    ingest_hint: str = ""


COUNTRY_LAYERS = {s.name: s for s in (
    LayerSpec("coast", Feature.COASTLINE, "line", "coastline", 1, widths=(1.6, 1.6), opacity=0.9,
              ingest_hint="geoquiz ingest --sources natural_earth"),
    LayerSpec("rivers", Feature.RIVER, "line", "rivers", 2, widths=(0.7, 6.5), gamma=2.2, budget_px=85_000,
              ingest_hint="geoquiz ingest --sources hydrosheds --hydro-regions <region>"),
    LayerSpec("lakes", Feature.LAKE, "fill", "lakes", 3, opacity=0.92,
              ingest_hint="geoquiz ingest --sources hydrosheds --hydro-regions <region>"),
    LayerSpec("roads", Feature.STREET, "line", "main roads", 4, min_rank=0.6, widths=(0.8, 3.6), gamma=2.0,
              budget_px=45_000, network=True, ingest_hint="geoquiz ingest --sources overture --country <ISO3>"),
    LayerSpec("rail", Feature.RAIL, "line", "railways", 5, min_rank=0.55, widths=(1.1, 1.9), opacity=0.9,
              budget_px=30_000, network=True, ingest_hint="geoquiz ingest --sources overture --country <ISO3>"),
    LayerSpec("borders", Feature.ADMIN, "outline", "internal borders", 6, widths=(1.3, 1.3), opacity=0.85,
              ingest_hint="geoquiz ingest --sources geoboundaries --country <ISO3>"),
    LayerSpec("cities", Feature.URBAN_CENTRE, "point", "cities", 7, opacity=0.95,
              ingest_hint="geoquiz ingest --sources ghsl"),
)}

CITY_LAYERS = {s.name: s for s in (
    LayerSpec("water", Feature.WATER, "fill", "water", 0, opacity=0.30),
    LayerSpec("waterways", Feature.WATERWAY, "line", "rivers and canals", 1, min_rank=0.3, widths=(1.0, 3.0), opacity=0.9),
    LayerSpec("buildings", Feature.BUILDING, "fill", "buildings", 2, opacity=0.55,
              ingest_hint="geoquiz ingest --sources overture --city <name> --overture-themes roads,rail,water,buildings"),
    LayerSpec("streets", Feature.STREET, "line", "streets", 3, min_rank=0.2, widths=(1.0, 5.0), gamma=1.6, network=True,
              ingest_hint="geoquiz ingest --sources overture --city <name>"),
    LayerSpec("rail", Feature.RAIL, "line", "railways", 4, min_rank=0.55, widths=(0.9, 1.5), opacity=0.6, network=True),
)}

LAYERS = {Scale.COUNTRY: COUNTRY_LAYERS, Scale.CITY: CITY_LAYERS}
DEFAULT_FEATURES = {Scale.COUNTRY: ("rivers", "lakes"), Scale.CITY: ("streets", "rail", "water", "waterways")}
REVEAL_MODES = ("auto", "network", "rank", "radial", "sweep")
OUTSIDE_MODES = ("keep", "dim", "cut")


@dataclass(frozen=True)
class PuzzleOptions:
    """Everything that shapes one puzzle besides the target, the features and the style."""
    outside: str = "keep"          # country: "keep" shows neighbours at full strength in a square frame (shape
    #                                hidden until the answer), "dim" darkens them, "cut" removes them (easiest)
    padding: float = 0.12          # country: how much surrounding area to show, as a fraction of the country's size
    buffer_km: float = 15.0        # country, outside=dim|cut: how far beyond the border features are kept
    budget_km: float = 350.0       # city: km of streets the window holds
    radius_km: float | None = None  # city: fixed half-width of the window instead of a street budget
    density: float = 1.0           # multiplies each layer's stroke budget (country mode): <1 sparser, >1 denser
    reveal: str = "auto"           # how the primary feature appears: auto | network | rank | radial | sweep
    country: str | None = None     # disambiguates a city name ("Cambridge", country="United Kingdom")
    question: tuple[str, ...] | None = None
    hint: str | None = None
    answer: str | None = None      # display name override (GHSL spells Zürich "Zurich")
    fact: str | None = None

    def __post_init__(self):
        if self.outside not in OUTSIDE_MODES:
            raise ValueError(f"outside must be one of {OUTSIDE_MODES}")
        if self.reveal not in REVEAL_MODES:
            raise ValueError(f"reveal must be one of {REVEAL_MODES}")


@dataclass
class Puzzle:
    slug: str                    # e.g. "CHE_rivers-lakes_neon"
    feature: str                 # the features drawn, primary first, joined by "+"
    scene: Scene
    texts: Texts
    style: Style
    accent: tuple[float, float, float]
    gain: np.ndarray | None      # (H, W) brightness mask: soft edges, dimming outside the target
    extracts: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)   # the clipped layers that were drawn
    sizes: dict[str, np.ndarray] = field(default_factory=dict)   # per layer, per feature: stroke width or dot
    #                                                              radius in frame pixels (absent for fills)
    params: dict = field(default_factory=dict)
    answer: dict = field(default_factory=dict)


class MissingData(ValueError):
    """A requested feature has no data in the store for this target."""


# -- small helpers ----------------------------------------------------------------

def _source_record(layer: str, gdf: gpd.GeoDataFrame, choices=None, on_screen: bool = True) -> dict:
    """Licence and provenance of one drawn layer, from the rows themselves."""
    recs = []
    for key in gdf["licence"].unique():                 # one layer can mix licences (geoBoundaries per country)
        part = gdf[gdf["licence"] == key]
        recs.append({
            "layer": layer, "source": part["source"].iloc[0],
            "data_version": sorted(part["data_version"].unique().tolist()),
            "licence_key": key, "licence": licences.get(key).to_dict(),
            "attribution": part["attribution"].iloc[0], "features": int(len(part)),
            "provider_choice": [c.__dict__ for c in choices] if choices else None, "on_screen": on_screen,
        })
    return recs


def _edge_gain(frame: Frame, box, fade_x: float, fade_y: float) -> np.ndarray:
    """1 inside the map box, falling smoothly to 0 at its edges: features run off the frame naturally."""
    l, t, r, b = box
    x = np.arange(frame.width, dtype=np.float32) + 0.5
    y = np.arange(frame.height, dtype=np.float32) + 0.5
    gx = np.clip(np.minimum(x - l, r - x) / max(fade_x, 1e-6), 0, 1)
    gy = np.clip(np.minimum(y - t, b - y) / max(fade_y, 1e-6), 0, 1)
    smooth = lambda v: v * v * (3 - 2 * v)
    return np.outer(smooth(gy), smooth(gx)).astype(np.float32)


def _inside_mask(frame: Frame, xy: np.ndarray, start: np.ndarray, blur_px: float = 2.0) -> np.ndarray:
    """Soft 0..1 mask of the target's outline."""
    surface = skia.Surface(frame.width, frame.height)
    c = surface.getCanvas()
    c.clear(skia.ColorBLACK)
    path = skia.Path()
    pts = xy.tolist()
    for s, e in zip(start[:-1], start[1:]):
        path.moveTo(*pts[s])
        for k in range(s + 1, e):
            path.lineTo(*pts[k])
        path.close()
    path.setFillType(skia.PathFillType.kEvenOdd)
    c.drawPath(path, skia.Paint(AntiAlias=True, Color=skia.ColorWHITE))
    m = surface.toarray()[..., 0].astype(np.float32) / 255.0
    return cv2.GaussianBlur(m, (0, 0), blur_px) if blur_px else m


def _starts(idx: np.ndarray) -> np.ndarray:
    return (np.concatenate([[0], np.flatnonzero(np.diff(idx)) + 1, [len(idx)]]) if len(idx) else np.zeros(1, int)).astype(np.int64)


def _lod_for(view: View) -> LOD:
    """Coarsest stored level of detail whose simplification stays under about a pixel."""
    m_per_px = 1.0 / view.scale
    return LOD.COARSE if m_per_px > 600 else LOD.MEDIUM if m_per_px > 60 else LOD.FULL


GRID = (12, 10)        # cells across and down the map band for spatially even thinning


def _thin_by_budget(rank: np.ndarray, row: np.ndarray, length_px: np.ndarray, mid_px: np.ndarray, box,
                    budget_px: float) -> np.ndarray:
    """Mask over paths: in every cell of a grid over the map, keep the most important features until
    their stroke length fills the cell's share of the budget.

    Thinning by importance alone would spend the whole budget where rivers are
    biggest (the Amazon) and leave dry regions empty; an even share per cell
    keeps the density, and so the difficulty, uniform across the frame.
    """
    l, t, r, b = box
    cx = np.clip(((mid_px[:, 0] - l) / (r - l) * GRID[0]).astype(int), 0, GRID[0] - 1)
    cy = np.clip(((mid_px[:, 1] - t) / (b - t) * GRID[1]).astype(int), 0, GRID[1] - 1)
    cell = cy * GRID[0] + cx
    order = np.lexsort((-rank[row], cell))                    # by cell, most important first
    c, ln = cell[order], length_px[order]
    cum = np.cumsum(ln)
    first = np.concatenate([[0], np.flatnonzero(np.diff(c)) + 1])
    before = cum - ln - np.repeat((cum - ln)[first], np.diff(np.concatenate([first, [len(c)]])))
    keep = np.zeros(len(row), bool)
    keep[order] = before < budget_px / (GRID[0] * GRID[1])
    # a feature is drawn whole or not at all
    n_rows = int(row.max()) + 1 if len(row) else 0
    return np.bincount(row, weights=keep, minlength=n_rows)[row] > 0


def _thin_by_level(rank: np.ndarray, row: np.ndarray, length_px: np.ndarray, budget_px: float) -> np.ndarray:
    """Mask over paths for a network: keep whole importance classes, most important first, while they
    fit the budget (the top class always stays). Cutting inside a class would leave a road network in fragments."""
    levels = np.unique(rank)[::-1]
    per_level = np.array([length_px[rank[row] == lv].sum() for lv in levels])
    n = max(int(np.searchsorted(np.cumsum(per_level), budget_px, "right")), 1)
    return rank[row] >= levels[n - 1]


def _importance(gdf) -> np.ndarray:
    """Rank used for thinning. Rivers too small to have a discharge rank are ordered by how much
    river lies upstream, so dry regions keep their main channels rather than an arbitrary subset."""
    rank = gdf["rank"].to_numpy(float)
    if "x_dist_up_km" in gdf:
        rank = rank + 1e-3 * np.log1p(gdf["x_dist_up_km"].to_numpy(float))
    return rank


def _preselect(gdf, region: Region, view: View, box, budget_px: float, factor: float = 3.0):
    """Cheap first cut for huge layers, before any projection of geometry: the same per-cell rule as
    :func:`_thin_by_budget`, on bounding-box centres and approximate lengths, with head-room."""
    b = shapely.bounds(gdf.geometry.to_numpy())
    x, y = region.to_xy().transform((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)
    mid = view.px(np.column_stack([x, y]))
    approx_px = (gdf["x_length_km"].to_numpy() * 1000.0 if "x_length_km" in gdf
                 else np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * 111_000.0) * view.scale
    keep = _thin_by_budget(_importance(gdf), np.arange(len(gdf)), approx_px, mid, box, budget_px * factor)
    return gdf[keep].reset_index(drop=True)


def country_frame_bbox(region: Region, frame: Frame = Frame(), padding: float = 0.12, margin_px: float = 140.0):
    """WGS84 bounding box of everything a country puzzle with outside="keep" can show.
    Ingest this much around a country, or its neighbours will be blank."""
    x0, y0, x1, y1 = region.projected().bounds
    pad = padding * max(x1 - x0, y1 - y0)
    view = View.fit((x0 - pad, y0 - pad, x1 + pad, y1 + pad), map_box(frame))
    rect = shapely.box(*view.metric_rect(frame, margin_px)).segmentize(max((x1 - x0) / 16, 1.0))
    return shp_transform(region.to_lonlat().transform, rect).bounds


def _circles(centres: np.ndarray, radii: np.ndarray, n: int = 20):
    """Ring vertices for filled dots (the last vertex repeats the first)."""
    a = np.linspace(0, 2 * np.pi, n + 1)
    unit = np.column_stack([np.cos(a), np.sin(a)])
    xy = (centres[:, None, :] + radii[:, None, None] * unit[None]).reshape(-1, 2).astype(np.float32)
    start = np.arange(len(centres) + 1, dtype=np.int64) * (n + 1)
    return xy, start


# -- reveal times -------------------------------------------------------------------

def _primary_times(mode: str, spec: LayerSpec, gdf, row, xy_m, start, origin, box_m) -> np.ndarray:
    """Reveal time per vertex of the primary line layer."""
    counts = np.diff(start)
    if mode == "auto":
        mode = "upstream" if "x_dist_dn_km" in gdf else "network" if spec.network else "radial"
    if mode == "network" and "x_dist_dn_km" in gdf:
        mode = "upstream"                                  # a river network's natural order
    if mode == "upstream":
        d = upstream_distance(xy_m, start, gdf["x_dist_dn_km"].to_numpy()[row], gdf["x_main_riv"].to_numpy()[row])
        return normalise_times(d)
    x0, y0, x1, y1 = box_m
    inside = (xy_m[:, 0] >= x0) & (xy_m[:, 0] <= x1) & (xy_m[:, 1] >= y0) & (xy_m[:, 1] <= y1)
    inside = inside if inside.any() else np.ones(len(xy_m), bool)
    if mode == "network":
        d = network_distance(xy_m, start, origin)
        return normalise_times(d, float(np.quantile(d[inside], 0.985)))
    if mode == "radial":
        d = np.hypot(xy_m[:, 0] - origin[0], xy_m[:, 1] - origin[1])
        return normalise_times(d, float(np.quantile(d[inside], 0.985)))
    if mode == "sweep":                                    # west to east
        return np.clip((xy_m[:, 0] - x0) / max(x1 - x0, 1e-9), 0, 1).astype(np.float32)
    # rank: the most important features are drawn first, each traced along its length
    along, total = path_lengths(xy_m, start)
    order = np.argsort(-gdf["rank"].to_numpy()[row], kind="stable")
    before = np.empty(len(total))
    before[order] = np.cumsum(total[order]) - total[order]
    t = (np.repeat(before, counts) + along) / max(total.sum(), 1e-9)
    return t.astype(np.float32)


def _point_times(mode: str, pts_m: np.ndarray, rank: np.ndarray, origin, box_m) -> np.ndarray:
    """Reveal time per object for a primary layer of fills or dots."""
    if mode == "rank":
        t = np.empty(len(rank))
        t[np.argsort(-rank, kind="stable")] = np.linspace(0, 1, len(rank))
        return t.astype(np.float32)
    if mode == "sweep":
        return np.clip((pts_m[:, 0] - box_m[0]) / max(box_m[2] - box_m[0], 1e-9), 0, 1).astype(np.float32)
    d = np.hypot(pts_m[:, 0] - origin[0], pts_m[:, 1] - origin[1])
    return normalise_times(d, float(np.quantile(d, 0.985)) if len(d) else 1.0)


class _Follow:
    """Secondary layers appear when the primary feature's reveal reaches them."""

    def __init__(self, xy_px: np.ndarray, time: np.ndarray):
        self.tree = cKDTree(xy_px) if len(xy_px) else None
        self.time = time

    def at(self, xy_px: np.ndarray) -> np.ndarray:
        if self.tree is None or not len(xy_px):
            return np.zeros(len(xy_px), np.float32)
        return self.time[self.tree.query(xy_px)[1]].astype(np.float32)

    def first(self, xy_px: np.ndarray, start: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
        """Earliest time over each group of rings (a lake appears when the reveal first touches it)."""
        t = self.at(xy_px)
        out = np.full(n_groups, np.inf, np.float32)
        np.minimum.at(out, np.repeat(group, np.diff(start)), t)
        return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


# -- the builder ------------------------------------------------------------------------

def build_puzzle(registry: ProviderRegistry, resolver: RegionResolver, kind: str, key: str,
                 features: tuple[str, ...] | None = None, frame: Frame = Frame(), style: Style = NEON,
                 options: PuzzleOptions = PuzzleOptions()) -> Puzzle:
    """Build a puzzle for a country or city from any supported mix of features."""
    scale = Scale(kind)
    specs_all = LAYERS[scale]
    explicit = features is not None
    features = tuple(features) if features else DEFAULT_FEATURES[scale]
    unknown = [f for f in features if f not in specs_all]
    if unknown:
        raise ValueError(f"unknown {scale.value} features {unknown}; choose from {', '.join(specs_all)}")
    if len(set(features)) != len(features):
        raise ValueError("a feature is listed twice")
    box = map_box(frame)

    # ---- region, view, and the area features are drawn in --------------------------------
    if scale is Scale.COUNTRY:
        region = resolver.country(key)
        outline = region.projected()
        x0, y0, x1, y1 = outline.bounds
        if options.outside == "keep":
            pad = options.padding * max(x1 - x0, y1 - y0)
            view = View.fit((x0 - pad, y0 - pad, x1 + pad, y1 + pad), box)
        else:
            view = View.fit(outline.bounds, map_box(frame, margin=0.04))
        fx0, fy0, fx1, fy1 = view.metric_rect(frame)
        band_top = view.y0 - box[1] / view.scale
        band_bottom = view.y0 - box[3] / view.scale
        area_m = (fx0, band_bottom, fx1, band_top)                        # the map band, in metres
        origin = _largest_city_xy(registry, region, outline) or region.centre_xy()
        keep_geom = (shapely.box(*area_m).buffer(60 / view.scale) if options.outside == "keep"
                     else shapely.simplify(outline, 0.5 / view.scale).buffer(options.buffer_km * 1000.0))
        shapely.prepare(keep_geom)
    else:
        base = resolver.city(key, options.country)
        aspect = (box[3] - box[1]) / (box[2] - box[0])
        if options.radius_km:
            region = base.with_window(options.radius_km * 1000.0, aspect=aspect)
            streets_all = registry.fetch(Feature.STREET, region, LOD.FULL)
            if len(streets_all):                                          # centre on the densest streets nearby
                region = fit_city_window(base, streets_all[streets_all["rank"] >= 0.2], 1e9, aspect=aspect)
                region = region.with_window(options.radius_km * 1000.0, centre=region.centre, aspect=aspect)
        else:
            streets_all = registry.fetch(Feature.STREET, base, LOD.FULL)
            if streets_all.empty:
                raise MissingData(f"no street data covers {base.name}, which sizes a city's window. "
                                  f"Run: {CITY_LAYERS['streets'].ingest_hint.replace('<name>', repr(base.name))} "
                                  "(or pass a fixed radius_km)")
            region = fit_city_window(base, streets_all[streets_all["rank"] >= 0.2], options.budget_km, aspect=aspect)
        outline = None
        view = View.fit(region.extent_xy(), box)
        area_m = region.extent_xy()
        origin = region.centre_xy()
        keep_geom = None
    fetch_rect = shapely.box(*view.metric_rect(frame, 140)).segmentize(max((area_m[2] - area_m[0]) / 16, 1.0))
    fetch_bbox = shp_transform(region.to_lonlat().transform, fetch_rect).bounds
    lod = _lod_for(view)

    # ---- fetch and project every requested layer -------------------------------------------
    built, sources, extracts, sizes, missing = [], [], {}, {}, []
    for name in features:
        spec = specs_all[name]
        choices = registry.explain(spec.feature, region)
        gdf = registry.fetch(spec.feature, region, lod, bbox=fetch_bbox)
        gdf = gdf[gdf["rank"] >= spec.min_rank]
        if spec.feature is Feature.WATERWAY and "x_tunnel" in gdf:
            gdf = gdf[~gdf["x_tunnel"].fillna(False).astype(bool)]         # culverted streams are not visible
        if spec.feature is Feature.ADMIN and "x_iso3" in gdf and not (gdf["x_iso3"] == region.id).any():
            gdf = gdf.iloc[:0]                                             # only neighbours' regions are in the store
        gdf = gdf.reset_index(drop=True)
        budget = spec.budget_px * options.density * frame.width / 1080 if spec.budget_px else None
        if budget and scale is Scale.COUNTRY and not spec.network and len(gdf) > 20_000:
            gdf = _preselect(gdf, region, view, box, budget)
        if keep_geom is not None and len(gdf):
            gdf = gdf[shapely.intersects(keep_geom, gdf.geometry.to_crs(region.crs).to_numpy())].reset_index(drop=True)
        if gdf.empty:
            missing.append(name)
            continue
        item = _project(spec, gdf, region, view, frame, options, box)
        if item is None:
            missing.append(name)
            continue
        built.append(item)
        sources += _source_record(name, item["gdf"], choices)
        extracts[name] = item["gdf"]
        if "size" in item:
            sizes[name] = item["size"]
    if missing and (explicit or len(missing) == len(features)):
        def why(m: str) -> str:
            stored = any(p.store.partitions(p.name, specs_all[m].feature)
                         for p in registry.candidates(specs_all[m].feature, scale) if hasattr(p, "store"))
            if stored:
                return f"{m}: none found there (the data is in the store; this place may simply have none, or needs its own download: {specs_all[m].ingest_hint or 'see geoquiz list features'})"
            return f"{m}: not downloaded yet — {specs_all[m].ingest_hint or 'not in the store'}"
        raise MissingData(f"no data for {', '.join(missing)} in {region.name}. " + "; ".join(why(m) for m in missing))

    # ---- reveal times: the primary feature leads, the rest follow it ---------------------------
    primary = built[0]
    spec = primary["spec"]
    if spec.geom in ("line", "outline"):
        mode = options.reveal if spec.geom == "line" else ("radial" if options.reveal in ("auto", "network") else options.reveal)
        primary["time"] = _primary_times(mode, spec, primary["gdf"], primary["row"], primary["xy_m"], primary["start"],
                                         origin, area_m)
        follow = _Follow(view.px(primary["xy_m"]), primary["time"])
    else:
        mode = "radial" if options.reveal in ("auto", "network") else options.reveal
        primary["time"] = _point_times(mode, primary["centres_m"], primary["rank"], origin, area_m)
        follow = _Follow(view.px(primary["centres_m"]), primary["time"])
    layers = []
    for item in built:
        s = item["spec"]
        color = style.color(s.feature)
        if s.geom in ("line", "outline"):
            xy = view.px(item["xy_m"])
            time = item.get("time")
            if time is None:
                time = follow.at(xy)
            layers.append((s.z, LineLayer(s.name, s.feature, xy, item["start"], item["width"], time, color, s.opacity)))
        else:
            xy = view.px(item["xy_m"]) if s.geom == "fill" else item["xy_px"]
            time = item.get("time")
            if time is None:
                time = follow.first(xy, item["start"], item["ring_poly"], item["n"])
            layers.append((s.z, FillLayer(s.name, s.feature, xy, item["start"], item["ring_poly"], time, color, s.opacity)))
    layers = [l for _, l in sorted(layers, key=lambda zl: zl[0])]

    # ---- outline, gain mask and the answer ------------------------------------------------------
    stats, answer_layers, inside_xy, inside_start = {}, [], None, None
    gain = _edge_gain(frame, box, 70 * frame.width / 1080, 130 * frame.width / 1080)
    if scale is Scale.COUNTRY:
        rings = shapely.get_rings(shapely.get_parts(shapely.simplify(outline, 0.35 / view.scale)))
        oxy_m, oidx = shapely.get_coordinates(rings, return_index=True)
        inside_start = _starts(oidx)
        inside_xy = view.px(oxy_m)
        inside = _inside_mask(frame, inside_xy, inside_start)
        if options.outside != "keep":
            dim = style.outside_dim if options.outside == "dim" else 0.0
            gain = (dim + (1 - dim) * inside) * _edge_gain(frame, box, 40, 60)
        answer_layers.append(LineLayer("border", Feature.COUNTRY, inside_xy, inside_start,
                                       np.full(len(inside_start) - 1, 3.0 * frame.width / 1080, np.float32),
                                       outline_times(inside_xy, inside_start), style.color(Feature.COUNTRY), 0.95))
        stats["frame_fill"] = round(float(inside.mean()), 4)
    else:
        x0, y0, x1, y1 = area_m
        stats["window_km"] = [round((x1 - x0) / 1000, 2), round((y1 - y0) / 1000, 2)]
        stats["frame_fill"] = 1.0
        ghsl = licences.get("ghsl")                      # located with GHSL, though nothing of GHSL is drawn
        sources.append({"layer": "(city extent)", "source": "ghsl", "data_version": ["R2024A V1.1"],
                        "licence_key": "ghsl", "licence": ghsl.to_dict(), "attribution": ghsl.attribution,
                        "features": 1, "provider_choice": None, "on_screen": False})

    # ---- fact, text, metadata -----------------------------------------------------------------
    measure, fact = _fact(primary, scale, region, outline, area_m)
    stats.update(measure)
    labels = [specs_all[i["spec"].name].label for i in built][:3]      # a hint, not an inventory
    hint = options.hint or "Guess it from its " + (labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1])
    name = options.answer or region.name
    scene = Scene(frame, view, region, layers=layers, answer_layers=answer_layers, inside=inside_xy,
                  inside_start=inside_start, sources=sources, stats=stats)
    texts = Texts(question=options.question or ((f"Which {scale.value}", "is this?")), hint=hint, answer=name,
                  fact=options.fact or fact, credit=licences.credit_line(scene.attributions()))
    iso_a2 = region.iso_a2
    if scale is Scale.CITY and region.country:
        try:
            iso_a2 = resolver.country(region.country).iso_a2
        except KeyError:
            pass
    drawn = [i["spec"].name for i in built]
    target_id = region.id if scale is Scale.COUNTRY else region.name.replace(" ", "-").replace("/", "-")
    return Puzzle(
        slug=f"{target_id}_{'-'.join(drawn)}_{style.name}", feature="+".join(drawn), scene=scene, texts=texts,
        style=style, accent=style.accent(spec.feature), gain=gain, extracts=extracts, sizes=sizes,
        params={"features": drawn, "outside": options.outside, "padding": options.padding,
                "buffer_km": options.buffer_km, "budget_km": options.budget_km, "radius_km": options.radius_km,
                "density": options.density, "reveal": options.reveal, "lod": int(lod),
                "window_radius_m": round(region.window_radius_m, 1) if region.window_radius_m else None,
                "centre": [round(c, 6) for c in region.centre]},
        answer={"answer": name, "source_name": region.name, "kind": scale.value, "id": region.id,
                "country": region.country, "iso_a2": iso_a2, "fact": texts.fact,
                "fact_source": (f"Computed from the {spec.label} layer drawn "
                                f"({sources[0]['source']} {sources[0]['data_version'][0]}).") if not options.fact else "supplied"},
    )


def _largest_city_xy(registry, region: Region, outline):
    """Projected centre of the country's most populous GHSL urban centre (where networks grow from)."""
    try:
        cities = registry.get("ghsl").fetch(Feature.URBAN_CENTRE, region, buffer_m=0.0)
    except KeyError:
        return None
    if cities.empty:
        return None
    pts = shapely.centroid(cities.geometry.to_crs(region.crs).to_numpy())
    inside = shapely.contains(outline, pts)
    if not inside.any():
        return None
    best = pts[inside][int(np.argmax(cities["x_pop"].to_numpy()[inside]))]
    return (best.x, best.y)


def _project(spec: LayerSpec, gdf, region: Region, view: View, frame: Frame, options: PuzzleOptions, box) -> dict | None:
    """Pixel-ready arrays for one layer (reveal times are filled in afterwards)."""
    u = frame.width / 1080
    if spec.geom == "point":
        pts = shapely.get_coordinates(shapely.centroid(gdf.geometry.to_crs(region.crs).to_numpy()))
        x0, y0, x1, y1 = view.metric_rect(frame, 20)
        keep = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        keep &= np.isin(np.arange(len(gdf)), np.argsort(-gdf["rank"].to_numpy(), kind="stable")[: int(400 * options.density)])
        if not keep.any():
            return None
        gdf, pts = gdf[keep].reset_index(drop=True), pts[keep]
        rank = gdf["rank"].to_numpy()
        radius = (2.2 + 10.0 * rank ** 1.3) * u
        xy, start = _circles(view.px(pts), radius)
        return {"spec": spec, "gdf": gdf, "size": radius, "xy_px": xy, "start": start, "ring_poly": np.arange(len(gdf)),
                "n": len(gdf), "centres_m": pts, "rank": rank}
    if spec.geom == "fill":
        xy_m, start, ring_poly, row = project_polygons(gdf, region, view, frame)
        if not len(row):
            return None
        polys = gdf.geometry.to_crs(region.crs).to_numpy()
        big = shapely.area(polys) * view.scale ** 2 >= 4.0                # drop specks under ~2 x 2 px
        if spec.feature is Feature.BUILDING:
            big[:] = True
        keep_poly = big[row]
        if not keep_poly.any():
            return None
        if not keep_poly.all():                                           # rebuild ring arrays for the kept polygons
            ring_keep = keep_poly[ring_poly]
            vert_keep = np.repeat(ring_keep, np.diff(start))
            xy_m = xy_m[vert_keep]
            start = np.concatenate([[0], np.cumsum(np.diff(start)[ring_keep])]).astype(np.int64)
            remap = np.cumsum(keep_poly) - 1
            ring_poly, row = remap[ring_poly[ring_keep]], row[keep_poly]
        gdf_kept = gdf.iloc[np.unique(row)].reset_index(drop=True)
        return {"spec": spec, "gdf": gdf_kept, "xy_m": xy_m, "start": start, "ring_poly": ring_poly, "n": len(row),
                "centres_m": shapely.get_coordinates(shapely.centroid(polys[row])), "rank": gdf["rank"].to_numpy()[row],
                "row": row, "polys": polys[row]}
    if spec.geom == "outline":
        lines = gdf.copy()
        lines["geometry"] = lines.geometry.boundary
        xy_m, start, row = project_lines(lines, region, view, frame)
    else:
        # a network keeps its shared vertices at city scale; at country scale segments are split at junctions
        simplify = None if (spec.network and region.kind is Scale.CITY) else 0.35
        xy_m, start, row = project_lines(gdf, region, view, frame, simplify_px=simplify, margin_px=120)
    if not len(row):
        return None
    rank = gdf["rank"].to_numpy()
    if spec.budget_px and region.kind is Scale.COUNTRY:
        _, total = path_lengths(xy_m, start)
        budget = spec.budget_px * options.density * u
        if spec.network:
            keep = _thin_by_level(rank, row, total * view.scale, budget)
        else:
            mid = view.px(xy_m[(start[:-1] + start[1:] - 1) // 2])
            keep = _thin_by_budget(_importance(gdf), row, total * view.scale, mid, box, budget)
        if not keep.all():
            vert_keep = np.repeat(keep, np.diff(start))
            xy_m = xy_m[vert_keep]
            start = np.concatenate([[0], np.cumsum(np.diff(start)[keep])]).astype(np.int64)
            row = row[keep]
    used = np.unique(row)
    remap = np.full(len(gdf), -1)
    remap[used] = np.arange(len(used))
    gdf = gdf.iloc[used].reset_index(drop=True)
    row = remap[row]
    rank = gdf["rank"].to_numpy()
    if spec.budget_px and len(rank) > 1 and rank.max() > rank.min():      # spread widths over what is actually drawn
        rank = (rank - rank.min()) / (rank.max() - rank.min())
    size = stroke_widths(rank, spec.widths[0] * u, spec.widths[1] * u, spec.gamma)
    return {"spec": spec, "gdf": gdf, "xy_m": xy_m, "start": start, "row": row, "width": size[row], "size": size}


def _fact(primary: dict, scale: Scale, region: Region, outline, area_m) -> tuple[dict, str]:
    """A number from the puzzle's own data, and the sentence that states it."""
    spec, gdf = primary["spec"], primary["gdf"]
    x0, y0, x1, y1 = area_m
    w_km, h_km = (x1 - x0) / 1000, (y1 - y0) / 1000
    where = "inside its borders" if scale is Scale.COUNTRY else f"in this {w_km:.1f} × {h_km:.1f} km view of the centre"
    geom = gdf.geometry.to_crs(region.crs).to_numpy()
    if spec.geom == "line":
        clip = shapely.simplify(outline, 200.0) if scale is Scale.COUNTRY else shapely.box(*area_m)
        shapely.prepare(clip)
        whole = shapely.contains(clip, geom)                # only features crossing the edge need cutting
        crossing = ~whole & shapely.intersects(clip, geom)
        km = float((shapely.length(geom[whole]).sum()
                    + shapely.length(shapely.intersection(geom[crossing], clip)).sum()) / 1000)
        return {f"{spec.name}_km": round(km, 1), "features_drawn": int(len(gdf))}, \
            f"This map shows {km:,.0f} km of {spec.label} {where}" if scale is Scale.COUNTRY else f"{km:,.0f} km of {spec.label} {where}"
    target = outline if scale is Scale.COUNTRY else shapely.box(*area_m)
    n = int(shapely.intersects(geom, target).sum())
    if spec.name == "cities":
        text = f"{n:,} cities of 50,000 people or more {where}"
    elif spec.name == "borders":
        text = f"{n:,} regions {where}"
    else:
        text = f"{n:,} {spec.label} {where}" if spec.name != "water" else f"{n:,} bodies of water {where}"
    return {f"{spec.name}_count": n, "features_drawn": int(len(gdf))}, text


# -- the two phase-0 puzzles, kept as shortcuts ---------------------------------------------------

def country_rivers(registry, resolver, key: str, frame: Frame = Frame(), style: Style = NEON, **options) -> Puzzle:
    return build_puzzle(registry, resolver, "country", key, ("rivers", "lakes"), frame, style, PuzzleOptions(**options))


def city_streets(registry, resolver, key: str, frame: Frame = Frame(), style: Style = NEON, **options) -> Puzzle:
    return build_puzzle(registry, resolver, "city", key, ("streets", "rail", "water", "waterways"), frame, style,
                        PuzzleOptions(**options))
