"""Optional lake/river polygons for the water-crossing surcharge. Gated behind
`data.water_enabled` — when disabled (default), `WaterMask.length_in_water`
always returns 0 and no data is fetched. When enabled but real data can't be
fetched, falls back to the handful of largest Swiss lakes as coarse circles
around their well-known centroids, which is enough to make the surcharge
meaningfully affect a handful of lakeside routes without pretending to be
survey-accurate hydrography.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from shapely.geometry import LineString, MultiPolygon, Point as ShapelyPoint
from shapely.ops import unary_union

from swiss_rail_rl.config import Config
from swiss_rail_rl.data.cities import Point

# name, WGS84 lon/lat centroid, approximate radius in metres — coarse stand-ins
# for the largest Swiss lakes, used only when real hydrography can't be fetched.
_MAJOR_LAKES = [
    ("Lac Léman", 6.6000, 46.4500, 15000.0),
    ("Bodensee", 9.3800, 47.6000, 10000.0),
    ("Neuenburgersee", 6.8600, 46.9000, 6000.0),
    ("Vierwaldstättersee", 8.4000, 47.0000, 5000.0),
    ("Zürichsee", 8.6600, 47.2500, 5000.0),
    ("Lago Maggiore", 8.8500, 46.0000, 4000.0),
    ("Lago di Lugano", 8.9600, 45.9700, 3000.0),
    ("Thunersee", 7.7200, 46.6800, 4000.0),
    ("Bielersee", 7.1500, 47.0800, 3500.0),
]


@dataclass
class WaterMask:
    enabled: bool
    polygons: MultiPolygon | None
    source: str

    def length_in_water(self, a_xy: tuple[float, float], b_xy: tuple[float, float]) -> float:
        if not self.enabled or self.polygons is None:
            return 0.0
        seg = LineString([a_xy, b_xy])
        inter = seg.intersection(self.polygons)
        return float(inter.length)


def _fallback_lakes(dst_crs: str) -> MultiPolygon:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    polys = []
    for _name, lon, lat, radius_m in _MAJOR_LAKES:
        x, y = transformer.transform(lon, lat)
        polys.append(ShapelyPoint(x, y).buffer(radius_m))
    return MultiPolygon(polys)


def load_water_mask(config: Config, points: list[Point]) -> WaterMask:
    if not config.data.water_enabled:
        return WaterMask(enabled=False, polygons=None, source="disabled")

    cache_dir = config.cache_dir()
    cache_path = cache_dir / "water_mask.wkt"
    dst_crs = config.region.crs_projected

    if cache_path.exists():
        from shapely import wkt

        geom = wkt.loads(cache_path.read_text(encoding="utf-8"))
        return WaterMask(enabled=True, polygons=MultiPolygon(unary_union(geom).geoms if geom.geom_type == "MultiPolygon" else [geom]), source="cached")

    # Real hydrography (swissTLM3D lakes/rivers) would be fetched here, mirroring
    # data/rail.py's approach. opendata.swiss access is not assumed available,
    # so we go straight to the coarse fallback and label it clearly.
    warnings.warn(
        "Water surcharge enabled but real hydrography fetch is not implemented "
        "for this environment — using coarse circular stand-ins for the "
        "largest Swiss lakes instead of survey-accurate polygons.",
        stacklevel=2,
    )
    polygons = _fallback_lakes(dst_crs)
    cache_path.write_text(polygons.wkt, encoding="utf-8")
    return WaterMask(enabled=True, polygons=polygons, source="fallback_lakes")
