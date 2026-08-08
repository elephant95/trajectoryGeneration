"""Projects the config's WGS84 point set into the metric CRS used for all
distance/cost math. No download needed — coordinates come straight from
config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer

from swiss_rail_rl.config import Config


@dataclass(frozen=True)
class Point:
    name: str
    lon: float
    lat: float
    x: float
    y: float
    is_anchor: bool


def project_points(config: Config) -> list[Point]:
    """Returns anchors first, then intermediates, all projected to config.region.crs_projected."""
    transformer = Transformer.from_crs("EPSG:4326", config.region.crs_projected, always_xy=True)

    points: list[Point] = []
    for spec in config.points.anchors:
        x, y = transformer.transform(spec.lon, spec.lat)
        points.append(Point(spec.name, spec.lon, spec.lat, x, y, is_anchor=True))
    for spec in config.points.intermediate:
        x, y = transformer.transform(spec.lon, spec.lat)
        points.append(Point(spec.name, spec.lon, spec.lat, x, y, is_anchor=False))
    return points


def bounding_box(points: list[Point], pad_m: float = 20_000.0) -> tuple[float, float, float, float]:
    """Returns (xmin, ymin, xmax, ymax) in the projected CRS, padded."""
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return (min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)
