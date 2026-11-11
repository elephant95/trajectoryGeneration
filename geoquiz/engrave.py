"""Laser-engraver export: one picture per feature type, plus one with everything.

Every file is cut exactly to the target's contour (the country outline, or a
city's window) and drawn on the same canvas at the same scale, so the layers
register when they are engraved one after another, for example to fill each
with a different epoxy before engraving the next.

Per layer: an SVG in millimetres and a black-and-white PNG. By default lines
are exported as *areas* as wide as their stroke (closed shapes a laser can
fill-engrave and you can pour into); ``lines="centerline"`` writes single
paths for vector scoring instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import shapely
import skia

from .puzzles import LAYERS, Puzzle
from .schema import Scale

#: Colours used in the combined SVG; laser software such as LightBurn maps each colour to its own layer.
LAYER_COLORS = ["#0000ff", "#ff0000", "#00a000", "#ff8000", "#a000ff", "#00a0a0", "#804000"]
CONTOUR_COLOR = "#000000"


def _path_d(geom, fmt: str = "{:.3f}") -> str:
    """SVG path data for polygons or lines already in millimetres."""
    out = []
    for part in shapely.get_parts(geom):
        t = shapely.get_type_id(part)
        if t == 3:
            for ring in shapely.get_rings(part):
                c = shapely.get_coordinates(ring)[:-1]
                out.append("M" + " L".join(f"{fmt.format(x)},{fmt.format(y)}" for x, y in c) + " Z")
        elif t in (1, 2):
            c = shapely.get_coordinates(part)
            if len(c) >= 2:
                out.append("M" + " L".join(f"{fmt.format(x)},{fmt.format(y)}" for x, y in c))
    return " ".join(out)


def _svg(width_mm: float, height_mm: float, groups: list[dict], title: str) -> str:
    """``groups``: dicts with id, mode ("fill" | "stroke"), color, and either geom or (geoms, widths)."""
    body = []
    for g in groups:
        if g["mode"] == "fill":
            d = _path_d(g["geom"])
            if d:
                body.append(f'<g id="{g["id"]}" fill="{g["color"]}" fill-rule="evenodd" stroke="none"><path d="{d}"/></g>')
        else:
            paths = "".join(f'<path d="{_path_d(geom)}" stroke-width="{w:.3f}"/>'
                            for geom, w in zip(g["geoms"], g["widths"]) if not shapely.is_empty(geom))
            body.append(f'<g id="{g["id"]}" fill="none" stroke="{g["color"]}" stroke-linecap="round" '
                        f'stroke-linejoin="round">{paths}</g>')
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:.3f}mm" height="{height_mm:.3f}mm" '
            f'viewBox="0 0 {width_mm:.3f} {height_mm:.3f}">\n<title>{title}</title>\n' + "\n".join(body) + "\n</svg>\n")


def _raster(width_mm: float, height_mm: float, groups: list[dict], dpi: int) -> np.ndarray:
    """Black on white bitmap (uint8, 0 or 255) of the groups."""
    k = dpi / 25.4
    w, h = int(round(width_mm * k)), int(round(height_mm * k))
    surface = skia.Surface(w, h)
    c = surface.getCanvas()
    c.clear(skia.ColorWHITE)
    c.scale(k, k)
    for g in groups:
        paint = skia.Paint(AntiAlias=True, Color=skia.ColorBLACK)
        items = [(g["geom"], None)] if g["mode"] == "fill" else list(zip(g["geoms"], g["widths"]))
        for geom, width in items:
            path = skia.Path()
            for part in shapely.get_parts(geom):
                rings = shapely.get_rings(part) if shapely.get_type_id(part) == 3 else [part]
                for ring in rings:
                    pts = shapely.get_coordinates(ring).tolist()
                    if len(pts) < 2:
                        continue
                    path.moveTo(*pts[0])
                    for pt in pts[1:]:
                        path.lineTo(*pt)
                    if shapely.get_type_id(part) == 3:
                        path.close()
            if width is None:
                path.setFillType(skia.PathFillType.kEvenOdd)
                paint.setStyle(skia.Paint.kFill_Style)
            else:
                paint.setStyle(skia.Paint.kStroke_Style)
                paint.setStrokeWidth(float(width))
                paint.setStrokeCap(skia.Paint.kRound_Cap)
                paint.setStrokeJoin(skia.Paint.kRound_Join)
            c.drawPath(path, paint)
    grey = surface.toarray()[..., 0]
    return np.where(grey < 128, 0, 255).astype(np.uint8)


def export_engraving(puzzle: Puzzle, out_dir: Path, width_mm: float = 200.0, margin_mm: float = 5.0,
                     min_line_mm: float = 0.3, line_scale: float = 1.0, lines: str = "outline",
                     contour_mm: float = 0.4, contour_in_layers: bool = False, dpi: int = 508,
                     mirror: bool = False) -> dict:
    """Write engraving files for ``puzzle`` into ``out_dir``; returns the manifest.

    ``width_mm`` is the width of the whole canvas including ``margin_mm`` on
    each side. Line widths follow the video's (scaled to the canvas), times
    ``line_scale``, and never go below ``min_line_mm``: set that to the finest
    groove your laser and filler can handle. ``mirror`` flips left-right, for
    engraving the back of clear material.
    """
    if lines not in ("outline", "centerline"):
        raise ValueError("lines must be 'outline' or 'centerline'")
    region, view = puzzle.scene.region, puzzle.scene.view
    is_country = region.kind is Scale.COUNTRY
    clip = region.projected() if is_country else shapely.box(*region.extent_xy())
    x0, y0, x1, y1 = clip.bounds
    s = (width_mm - 2 * margin_mm) / (x1 - x0)                 # millimetres per metre
    height_mm = (y1 - y0) * s + 2 * margin_mm
    mm_per_px = s / view.scale

    def to_mm(geom):
        def f(c):
            x = (c[:, 0] - x0) * s + margin_mm
            return np.column_stack([width_mm - x if mirror else x, (y1 - c[:, 1]) * s + margin_mm])
        return shapely.transform(geom, f)

    tol_m = 0.02 / s                                            # 0.02 mm: below anything a laser resolves
    shapely.prepare(clip)
    contour = to_mm(shapely.simplify(clip, tol_m))
    contour_group = {"id": "contour", "mode": "stroke", "color": CONTOUR_COLOR,
                     "geoms": [shapely.boundary(contour)], "widths": [contour_mm]}

    specs = LAYERS[region.kind]
    groups, info = {}, {}
    for name, gdf in puzzle.extracts.items():
        spec = specs[name]
        geom = gdf.geometry.to_crs(region.crs).to_numpy()
        if spec.geom == "outline":
            geom = shapely.boundary(geom)
        if spec.geom == "point":
            geom = shapely.buffer(shapely.centroid(geom), puzzle.sizes[name] * mm_per_px * line_scale / s, quad_segs=12)
        inside = shapely.contains(clip, geom)
        crossing = ~inside & shapely.intersects(clip, geom)
        geom = np.concatenate([geom[inside], shapely.intersection(geom[crossing], clip)])
        order = np.concatenate([np.flatnonzero(inside), np.flatnonzero(crossing)])
        if not len(geom):
            continue
        if spec.geom in ("line", "outline"):
            width = np.maximum(puzzle.sizes[name][order] * mm_per_px * line_scale, min_line_mm)
            geom = shapely.simplify(geom, tol_m)
            if lines == "outline":
                area = shapely.union_all(shapely.buffer(geom, width / 2 / s, quad_segs=4))
                area = shapely.intersection(area, clip)         # the buffer may poke past the contour
                parts = shapely.get_parts(area)                 # drop specks too small to engrave or fill
                area = shapely.union_all(parts[shapely.area(parts) * s * s >= min_line_mm ** 2])
                groups[name] = {"id": name, "mode": "fill", "geom": to_mm(shapely.simplify(area, tol_m))}
            else:
                wq = np.round(width / 0.05) * 0.05              # merge near-equal widths into one path each
                levels = np.unique(wq)
                groups[name] = {"id": name, "mode": "stroke", "widths": levels.tolist(),
                                "geoms": [to_mm(shapely.multilinestrings(
                                    [g for g in shapely.get_parts(geom[wq == lv]) if shapely.get_type_id(g) == 1]))
                                    for lv in levels]}
            info[name] = {"kind": "lines", "features": int(len(geom)),
                          "line_mm": [round(float(width.min()), 2), round(float(width.max()), 2)]}
        else:
            area = shapely.union_all(shapely.make_valid(geom))
            groups[name] = {"id": name, "mode": "fill", "geom": to_mm(shapely.simplify(area, tol_m))}
            info[name] = {"kind": "areas", "features": int(len(geom))}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    def write(stem: str, layer_groups: list[dict], colored: bool):
        gs = [{**g, "color": g.get("color") or (LAYER_COLORS[i % len(LAYER_COLORS)] if colored else "#000000")}
              for i, g in enumerate(layer_groups)]
        (out_dir / f"{stem}.svg").write_text(_svg(width_mm, height_mm, gs, f"{region.name} — {stem}"), encoding="utf-8")
        cv2.imwrite(str(out_dir / f"{stem}.png"), _raster(width_mm, height_mm, gs, dpi), [cv2.IMWRITE_PNG_BILEVEL, 1])
        files[stem] = [f"{stem}.svg", f"{stem}.png"]

    write("contour", [contour_group], colored=False)
    names = list(groups)
    for name in names:
        write(name, [groups[name]] + ([contour_group] if contour_in_layers else []), colored=False)
    write("all", [groups[n] for n in names] + [contour_group], colored=True)

    manifest = {
        "target": {"kind": region.kind.value, "id": region.id, "name": region.name},
        "canvas_mm": [round(width_mm, 3), round(height_mm, 3)], "margin_mm": margin_mm,
        "scale": f"1:{round(1000 / s):,}", "dpi": dpi, "lines": lines, "mirror": mirror,
        "min_line_mm": min_line_mm, "contour_mm": contour_mm, "contour_in_layers": contour_in_layers,
        "layers": info, "files": files,
        "all_svg_colors": {**{n: LAYER_COLORS[i % len(LAYER_COLORS)] for i, n in enumerate(names)}, "contour": CONTOUR_COLOR},
        "registration": "Every file shares one canvas and origin: import each at the same position and they line up.",
        "attribution": [src["licence"]["attribution_full"] for src in puzzle.scene.sources if src.get("on_screen", True)],
    }
    (out_dir / "engrave.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["folder"] = str(out_dir)
    return manifest
