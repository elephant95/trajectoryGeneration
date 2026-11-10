"""Puzzle builders: region + providers -> a Scene and the text that goes with it.

Two puzzles exist so far, one per mode: a country from its rivers and lakes,
and a city from its streets. Each returns a :class:`Puzzle` holding everything
the Reel assembler needs, including the clipped layers (kept so the exact
data behind a video can be shared on request, as ODbL asks).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import geopandas as gpd
import numpy as np
import shapely
import skia

from . import licences
from .providers.base import ProviderRegistry
from .regions import Region, RegionResolver, fit_city_window
from .render.overlay import Texts, map_box
from .render.style import NEON, Style
from .scene import (FillLayer, Frame, LineLayer, Scene, View, nearest_time, network_distance, normalise_times,
                    outline_times, project_lines, project_polygons, stroke_widths, upstream_distance)
from .schema import LOD, Feature


@dataclass
class Puzzle:
    slug: str                    # e.g. "CHE_rivers_neon"
    feature: str
    scene: Scene
    texts: Texts
    style: Style
    accent: tuple[float, float, float]
    gain: np.ndarray | None      # (H, W) brightness mask: soft edges, dimming outside the target
    extracts: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)   # the clipped layers that were drawn
    params: dict = field(default_factory=dict)
    answer: dict = field(default_factory=dict)


def _source_record(layer: str, gdf: gpd.GeoDataFrame, choices=None) -> dict:
    """Licence and provenance of one drawn layer, from the rows themselves."""
    key = gdf["licence"].iloc[0]
    lic = licences.get(key)
    return {
        "layer": layer,
        "source": gdf["source"].iloc[0],
        "data_version": sorted(gdf["data_version"].unique().tolist()),
        "licence_key": key,
        "licence": lic.to_dict(),
        "attribution": gdf["attribution"].iloc[0],
        "features": int(len(gdf)),
        "provider_choice": [c.__dict__ for c in choices] if choices else None,
    }


def _fetch(registry: ProviderRegistry, feature: Feature, region: Region, lod: LOD):
    choices = registry.explain(feature, region)
    return registry.fetch(feature, region, lod), choices


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


# -- country: rivers and lakes ----------------------------------------------------

def country_rivers(registry: ProviderRegistry, resolver: RegionResolver, key: str, frame: Frame = Frame(),
                   style: Style = NEON, outside: str = "dim", buffer_km: float = 15.0,
                   min_discharge_cms: float = 0.4) -> Puzzle:
    """Guess the country from its river network.

    ``outside`` is the difficulty knob: ``"dim"`` keeps rivers beyond the
    border at reduced brightness so the silhouette stays hidden (harder),
    ``"cut"`` removes them and so reveals the outline (easier).
    """
    if outside not in ("dim", "cut"):
        raise ValueError("outside must be 'dim' or 'cut'")
    region = resolver.country(key)
    outline = region.projected()
    view = View.fit(outline.bounds, map_box(frame, margin=0.04))
    hull = outline.buffer(buffer_km * 1000.0)

    rivers, river_choice = _fetch(registry, Feature.RIVER, region, LOD.MEDIUM)
    lakes, lake_choice = _fetch(registry, Feature.LAKE, region, LOD.MEDIUM)
    if rivers.empty:
        raise ValueError(f"no river data covers {region.name}")
    if "x_discharge_cms" in rivers:
        rivers = rivers[rivers["x_discharge_cms"] >= min_discharge_cms]
    near = lambda g: g[shapely.intersects(g.geometry.to_crs(region.crs).to_numpy(), hull)].reset_index(drop=True)
    rivers, lakes = near(rivers), near(lakes)

    xy_m, start, row = project_lines(rivers, region, view, frame)
    xy = view.px(xy_m)
    if "x_dist_dn_km" in rivers:      # grow upstream from where each basin leaves the scene
        d = upstream_distance(xy_m, start, rivers["x_dist_dn_km"].to_numpy()[row], rivers["x_main_riv"].to_numpy()[row])
        time = normalise_times(d)
    else:                             # no network attributes: major rivers first (a hint ladder)
        time = np.repeat(1.0 - rivers["rank"].to_numpy()[row], np.diff(start)).astype(np.float32)
        time = normalise_times(time - time.min())
    river_layer = LineLayer("rivers", Feature.RIVER, xy, start,
                            stroke_widths(rivers["rank"].to_numpy()[row], 0.7, 6.5, gamma=2.2), time,
                            style.color(Feature.RIVER))
    layers = [river_layer]
    sources = [_source_record("rivers", rivers, river_choice)]
    extracts = {"rivers": rivers}
    if len(lakes):
        lxy_m, lstart, lring_poly, lrow = project_polygons(lakes, region, view, frame)
        if len(lrow):
            cent = view.px(shapely.get_coordinates(shapely.centroid(lakes.geometry.to_crs(region.crs).to_numpy()[lrow])))
            layers.append(FillLayer("lakes", Feature.LAKE, view.px(lxy_m), lstart, lring_poly,
                                    nearest_time(cent, xy, time), style.color(Feature.LAKE), opacity=0.92))
            sources.append(_source_record("lakes", lakes, lake_choice))
            extracts["lakes"] = lakes

    # the outline: mask for dimming, and the border drawn during the answer
    rings = shapely.get_rings(shapely.get_parts(shapely.simplify(outline, 0.35 / view.scale)))
    oxy_m, oidx = shapely.get_coordinates(rings, return_index=True)
    ostart = np.concatenate([[0], np.flatnonzero(np.diff(oidx)) + 1, [len(oidx)]])
    oxy = view.px(oxy_m)
    inside = _inside_mask(frame, oxy, ostart)
    dim = style.outside_dim if outside == "dim" else 0.0
    gain = (dim + (1 - dim) * inside) * _edge_gain(frame, map_box(frame), 40, 60)
    border = LineLayer("border", Feature.COUNTRY, oxy, ostart, np.full(len(ostart) - 1, 3.0, np.float32),
                       outline_times(oxy, ostart), style.color(Feature.COUNTRY), opacity=0.95)

    river_km = float(shapely.length(shapely.intersection(rivers.geometry.to_crs(region.crs).to_numpy(), outline)).sum() / 1000)
    lakes_inside = int(shapely.intersects(lakes.geometry.to_crs(region.crs).to_numpy(), outline).sum()) if len(lakes) else 0
    scene = Scene(frame, view, region, layers=layers, answer_layers=[border], inside=oxy, inside_start=ostart,
                  sources=sources,
                  stats={"river_km_inside": round(river_km, 1), "lakes_inside": lakes_inside,
                         "river_reaches": int(len(rivers)),
                         "frame_fill": round(float(inside.mean()), 4)})
    fact = f"This map shows {river_km:,.0f} km of rivers inside its borders"
    texts = Texts(question=("Which country", "is this?"), hint="Guess it from its rivers and lakes",
                  answer=region.name, fact=fact, credit=licences.credit_line(scene.attributions()))
    return Puzzle(
        slug=f"{region.id}_rivers_{style.name}", feature="rivers", scene=scene, texts=texts, style=style,
        accent=style.color(Feature.RIVER), gain=gain, extracts=extracts,
        params={"outside": outside, "buffer_km": buffer_km, "min_discharge_cms": min_discharge_cms},
        answer={"answer": region.name, "iso3": region.id, "iso_a2": region.iso_a2, "fact": fact,
                "fact_source": f"Computed from the river layer drawn ({sources[0]['source']} {sources[0]['data_version'][0]}), "
                               "clipped to the country outline."},
    )


# -- city: streets ----------------------------------------------------------------

def city_streets(registry: ProviderRegistry, resolver: RegionResolver, key: str, frame: Frame = Frame(),
                 style: Style = NEON, budget_km: float = 350.0, country: str | None = None,
                 min_street_rank: float = 0.2, min_rail_rank: float = 0.55) -> Puzzle:
    """Guess the city from its street network.

    The window is sized by a density budget (``budget_km`` of streets) around
    the densest part of the GHSL urban centre, and streets grow outward from
    the centre by network distance.
    """
    box = map_box(frame)
    aspect = (box[3] - box[1]) / (box[2] - box[0])
    centre0 = resolver.city(key, country)
    all_streets = registry.fetch(Feature.STREET, centre0, LOD.FULL)
    if all_streets.empty:
        raise ValueError(f"no street data covers {centre0.name}")
    region = fit_city_window(centre0, all_streets[all_streets["rank"] >= min_street_rank], budget_km, aspect=aspect)
    view = View.fit(region.extent_xy(), box)
    cx, cy = region.centre_xy()

    streets, street_choice = _fetch(registry, Feature.STREET, region, LOD.FULL)
    streets = streets[streets["rank"] >= min_street_rank].reset_index(drop=True)
    xy_m, start, row = project_lines(streets, region, view, frame, simplify_px=None, margin_px=120)
    d = network_distance(xy_m, start, (cx, cy))
    x0, y0, x1, y1 = region.extent_xy()
    in_window = (xy_m[:, 0] >= x0) & (xy_m[:, 0] <= x1) & (xy_m[:, 1] >= y0) & (xy_m[:, 1] <= y1)
    d_max = float(np.quantile(d[in_window], 0.985))
    street_layer = LineLayer("streets", Feature.STREET, view.px(xy_m), start,
                             stroke_widths(streets["rank"].to_numpy()[row], 1.0, 5.0, gamma=1.6),
                             normalise_times(d, d_max), style.color(Feature.STREET))
    crow = lambda pts: normalise_times(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) * 1.25, d_max)

    layers, sources, extracts = [], [], {"streets": streets}
    water, water_choice = _fetch(registry, Feature.WATER, region, LOD.FULL)
    if len(water):
        wxy_m, wstart, wring_poly, wrow = project_polygons(water, region, view, frame)
        if len(wrow):
            polys = water.geometry.to_crs(region.crs).to_numpy()[wrow]
            t = normalise_times(shapely.distance(polys, shapely.Point(cx, cy)) * 1.25, d_max)
            layers.append(FillLayer("water", Feature.WATER, view.px(wxy_m), wstart, wring_poly, t,
                                    style.color(Feature.WATER), opacity=0.30))
            sources.append(_source_record("water", water, water_choice))
            extracts["water"] = water
    for name, feature, min_rank, w in (("waterways", Feature.WATERWAY, 0.3, (1.0, 3.0)),
                                       ("rail", Feature.RAIL, min_rail_rank, (0.9, 1.5))):
        g, choice = _fetch(registry, feature, region, LOD.FULL)
        g = g[g["rank"] >= min_rank]
        if feature is Feature.WATERWAY and "x_tunnel" in g:
            g = g[~g["x_tunnel"]]                       # culverted streams are not visible on the ground
        g = g.reset_index(drop=True)
        if g.empty:
            continue
        gxy_m, gstart, grow = project_lines(g, region, view, frame)
        if len(grow):
            layers.append(LineLayer(name, feature, view.px(gxy_m), gstart,
                                    stroke_widths(g["rank"].to_numpy()[grow], *w), crow(gxy_m), style.color(feature),
                                    opacity=0.6 if feature is Feature.RAIL else 0.9))
            sources.append(_source_record(name, g, choice))
            extracts[name] = g
    layers.append(street_layer)
    layers.sort(key=lambda l: ("water", "waterways", "streets", "rail").index(l.name))   # back to front
    sources.insert(0, _source_record("streets", streets, street_choice))
    # the city is located with GHSL but nothing of GHSL is drawn: credit it in the caption only
    ghsl = licences.get("ghsl")
    sources.append({"layer": "(city extent)", "source": "ghsl", "data_version": ["R2024A V1.1"], "licence_key": "ghsl",
                    "licence": ghsl.to_dict(), "attribution": ghsl.attribution, "features": 1,
                    "provider_choice": None, "on_screen": False})

    seg = np.hypot(*np.diff(xy_m, axis=0).T)
    ok = np.ones(len(seg), bool)
    ok[start[1:-1] - 1] = False
    street_km = float(seg[ok & in_window[:-1] & in_window[1:]].sum() / 1000)
    w_km, h_km = (x1 - x0) / 1000, (y1 - y0) / 1000
    scene = Scene(frame, view, region, layers=layers, sources=sources,
                  stats={"street_km_in_window": round(street_km, 1), "window_km": [round(w_km, 2), round(h_km, 2)],
                         "reveal_distance_m": round(d_max), "frame_fill": 1.0})
    fact = f"{street_km:,.0f} km of streets in this {w_km:.1f} × {h_km:.1f} km view of the centre"
    texts = Texts(question=("Which city", "is this?"), hint="Guess it from its streets",
                  answer=region.name, fact=fact, credit=licences.credit_line(scene.attributions()))
    iso_a2 = None
    if region.country:                                   # flag from the country the urban centre lies in
        try:
            iso_a2 = resolver.country(region.country).iso_a2
        except KeyError:
            pass
    return Puzzle(
        slug=f"{region.name.replace(' ', '-')}_streets_{style.name}", feature="streets", scene=scene, texts=texts,
        style=style, accent=style.color(Feature.STREET), gain=_edge_gain(frame, box, 70, 130), extracts=extracts,
        params={"budget_km": budget_km, "min_street_rank": min_street_rank, "min_rail_rank": min_rail_rank,
                "window_radius_m": round(region.window_radius_m, 1), "centre": [round(c, 6) for c in region.centre]},
        answer={"answer": region.name, "country": region.country, "ghsl_id": region.id, "iso_a2": iso_a2, "fact": fact,
                "fact_source": f"Computed from the street layer drawn ({sources[0]['source']} {sources[0]['data_version'][0]})."},
    )
