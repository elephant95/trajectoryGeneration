"""Region resolver: a country or city name -> geometry, centre, window, projection.

Countries come from Natural Earth admin-0 (the silhouette). Cities come from
GHSL urban centres, which use one definition worldwide; the rendered frame is
a fixed window around the city sized by a density budget, never an
administrative outline.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from shapely.geometry.base import BaseGeometry

from .schema import CRS, Feature, LOD, Scale

#: National grids used instead of a local equal-area projection where one is standard.
NATIONAL_GRIDS = {
    "CHE": "EPSG:2056",   # LV95
    "LIE": "EPSG:2056",
    "GBR": "EPSG:27700",
    "FRA": "EPSG:2154",
    "NLD": "EPSG:28992",
    "AUT": "EPSG:31287",
    "DEU": "EPSG:25832",
}


def shp_transform(fn, geom: BaseGeometry) -> BaseGeometry:
    """Apply a pyproj transform function to every coordinate of a geometry."""
    return shapely.transform(geom, lambda c: np.column_stack(fn(c[:, 0], c[:, 1])))


def local_laea(lon: float, lat: float) -> str:
    """Lambert azimuthal equal-area centred on the target (metres, north up at the centre)."""
    return f"+proj=laea +lat_0={lat:.6f} +lon_0={lon:.6f} +datum=WGS84 +units=m +no_defs"


def fold(text: str) -> str:
    """Case- and accent-insensitive key for name matching ('Zürich' -> 'zurich')."""
    t = unicodedata.normalize("NFKD", text.replace("﻿", ""))
    return "".join(c for c in t if not unicodedata.combining(c)).casefold().strip()


@dataclass(frozen=True)
class Region:
    kind: Scale
    id: str                      # ISO 3166-1 alpha-3, or "ghsl:<urban centre id>"
    name: str
    geometry: BaseGeometry       # WGS84: country outline, or urban-centre polygon
    centre: tuple[float, float]  # lon, lat
    crs: str                     # projected CRS used for all metric work and rendering
    country: str | None = None   # ISO3 (country) or country name (city)
    iso_a2: str | None = None    # for the flag
    population: float | None = None
    window_radius_m: float | None = None   # city: half-width of the rendered window
    window_aspect: float = 16 / 9          # city: window height / width

    # -- projection helpers -------------------------------------------------
    def to_xy(self) -> Transformer:
        return Transformer.from_crs(CRS, self.crs, always_xy=True)

    def to_lonlat(self) -> Transformer:
        return Transformer.from_crs(self.crs, CRS, always_xy=True)

    def projected(self) -> BaseGeometry:
        return shp_transform(self.to_xy().transform, self.geometry)

    def centre_xy(self) -> tuple[float, float]:
        return self.to_xy().transform(*self.centre)

    def extent_xy(self) -> tuple[float, float, float, float]:
        """Projected extent of what gets rendered: the window if set, else the geometry."""
        if self.window_radius_m:
            cx, cy = self.centre_xy()
            r = self.window_radius_m
            return (cx - r, cy - r * self.window_aspect, cx + r, cy + r * self.window_aspect)
        return self.projected().bounds

    def bbox(self, buffer_m: float = 0.0) -> tuple[float, float, float, float]:
        """WGS84 bounding box of the rendered extent plus a buffer, for store reads."""
        x0, y0, x1, y1 = self.extent_xy()
        ring = shapely.box(x0 - buffer_m, y0 - buffer_m, x1 + buffer_m, y1 + buffer_m).segmentize(
            max((x1 - x0) / 16, 1.0))
        return shp_transform(self.to_lonlat().transform, ring).bounds

    def area_km2(self) -> float:
        x0, y0, x1, y1 = self.extent_xy()
        return ((x1 - x0) * (y1 - y0) if self.window_radius_m else self.projected().area) / 1e6

    def with_window(self, radius_m: float, centre: tuple[float, float] | None = None,
                    aspect: float = 16 / 9) -> "Region":
        """A copy with a render window; re-centres the projection when the centre moves."""
        if centre is None:
            return replace(self, window_radius_m=radius_m, window_aspect=aspect)
        return replace(self, window_radius_m=radius_m, window_aspect=aspect, centre=centre, crs=local_laea(*centre))


class RegionResolver:
    """Looks regions up through the provider registry (never a live API)."""

    def __init__(self, registry):
        self.registry = registry

    def country(self, key: str, mainland_only: bool = True) -> Region:
        """Resolve a country by ISO3 code or name."""
        table = self.registry.get("natural_earth").countries(LOD.FULL)
        k = fold(key)
        hit = table[(table["source_id"].map(fold) == k) | (table["name"].map(fold) == k)]
        if hit.empty:
            raise KeyError(f"no country matches {key!r}")
        row = hit.iloc[0]
        geom = row.geometry
        lon, lat = geom.representative_point().coords[0]
        if mainland_only:
            geom = mainland(geom, local_laea(lon, lat))
        c = geom.centroid
        iso3 = row["source_id"]
        return Region(
            kind=Scale.COUNTRY, id=iso3, name=row["name"], geometry=geom, centre=(c.x, c.y),
            crs=NATIONAL_GRIDS.get(iso3, local_laea(c.x, c.y)), country=iso3,
            iso_a2=row.get("x_iso_a2"), population=float(row["x_pop"]) if row.get("x_pop") is not None else None,
        )

    def city(self, key: str, country: str | None = None) -> Region:
        """Resolve a city to its GHSL urban centre (largest by population on ties).

        The returned region has no window yet: call :func:`fit_city_window`
        with the layer that drives the puzzle.
        """
        table = self.registry.get("ghsl").urban_centres()
        k = fold(key)
        names = table["x_names_all"].fillna(table["name"]).map(lambda s: [fold(p) for p in str(s).split(";")])
        hit = table[names.map(lambda parts: k in parts)]
        if country is not None:
            hit = hit[hit["x_country"].map(fold) == fold(country)]
        if hit.empty:
            raise KeyError(f"no GHSL urban centre matches {key!r}")
        row = hit.sort_values("x_pop", ascending=False).iloc[0]
        p = row.geometry.representative_point()
        return Region(
            kind=Scale.CITY, id=f"ghsl:{row['source_id']}", name=row["name"], geometry=row.geometry,
            centre=(p.x, p.y), crs=local_laea(p.x, p.y), country=row["x_country"], population=float(row["x_pop"]),
        )


def mainland(geom: BaseGeometry, crs: str, max_gap_km: float = 600.0) -> BaseGeometry:
    """Drop far-away parts (overseas territories) so framing follows the main landmass."""
    if geom.geom_type != "MultiPolygon":
        return geom
    to_xy = Transformer.from_crs(CRS, crs, always_xy=True).transform
    parts = list(geom.geoms)
    proj = [shp_transform(to_xy, p) for p in parts]
    main = max(proj, key=lambda p: p.area)
    keep = [p for p, q in zip(parts, proj) if q.distance(main) <= max_gap_km * 1000]
    return shapely.MultiPolygon(keep) if len(keep) > 1 else keep[0]


# -- city windows ---------------------------------------------------------------

def segments(lines: gpd.GeoSeries) -> tuple[np.ndarray, np.ndarray]:
    """Midpoints (n, 2) and lengths (n,) of every segment of a projected line layer."""
    coords, idx = shapely.get_coordinates(lines.to_numpy(), return_index=True)
    same = idx[1:] == idx[:-1]
    a, b = coords[:-1][same], coords[1:][same]
    return (a + b) / 2, np.hypot(*(b - a).T)


def densest_point(mid: np.ndarray, length: np.ndarray, cell_m: float = 200.0, sigma_m: float = 600.0) -> tuple[float, float]:
    """Where the layer is densest: the peak of a smoothed length-per-cell grid."""
    x0, y0 = mid.min(axis=0)
    ix = ((mid[:, 0] - x0) / cell_m).astype(int)
    iy = ((mid[:, 1] - y0) / cell_m).astype(int)
    grid = np.zeros((iy.max() + 1, ix.max() + 1))
    np.add.at(grid, (iy, ix), length)
    grid = gaussian_filter(grid, sigma_m / cell_m, mode="constant")
    j, i = np.unravel_index(grid.argmax(), grid.shape)
    return (x0 + (i + 0.5) * cell_m, y0 + (j + 0.5) * cell_m)


def radius_for_budget(mid: np.ndarray, length: np.ndarray, centre: tuple[float, float], budget_km: float,
                      aspect: float = 16 / 9, r_min: float = 500.0, r_max: float = 15_000.0) -> float:
    """Half-width r of the window (2r wide, 2r*aspect tall) holding ``budget_km`` of lines."""
    dx = np.abs(mid[:, 0] - centre[0])
    dy = np.abs(mid[:, 1] - centre[1]) / aspect
    need = np.maximum(dx, dy)            # smallest r whose window contains the segment
    order = np.argsort(need)
    cum = np.cumsum(length[order]) / 1000.0
    k = int(np.searchsorted(cum, budget_km))
    r = need[order][min(k, len(order) - 1)] if len(order) else r_min
    return float(np.clip(r, r_min, r_max))


def fit_city_window(region: Region, lines: gpd.GeoDataFrame, budget_km: float = 300.0,
                    aspect: float = 16 / 9, recentre: bool = True) -> Region:
    """Size (and optionally centre) a city's window from a line layer and a density budget.

    ``lines`` is the WGS84 layer driving the puzzle (for example streets). The
    centre moves to the densest part of the layer inside the urban centre,
    which lands on the historic core far more reliably than a polygon centroid.
    """
    proj = lines.geometry.to_crs(region.crs)
    mid, length = segments(proj)
    if not len(mid):
        raise ValueError("cannot size a window from an empty layer")
    centre_xy = region.centre_xy()
    if recentre:
        inside = shapely.contains_xy(region.projected(), mid[:, 0], mid[:, 1])
        if inside.any():
            centre_xy = densest_point(mid[inside], length[inside])
    r = radius_for_budget(mid, length, centre_xy, budget_km, aspect=aspect)
    lon, lat = region.to_lonlat().transform(*centre_xy)
    return region.with_window(r, centre=(lon, lat), aspect=aspect)
