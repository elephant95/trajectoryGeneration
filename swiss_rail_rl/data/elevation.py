"""Elevation raster loader.

Primary path: download real open elevation data (Terrarium/SRTM-derived
tiles, served publicly over S3 — the practical stand-in for swissALTI3D/SRTM
when swisstopo's own API isn't reachable from the current network) and warp
it into the project's metric CRS. If no network access is available at all,
fall back to a procedurally generated terrain that mimics Switzerland's
coarse geography (flat Mittelland band, Alpine wall to the south, Jura hills
to the north-west) so the rest of the pipeline still runs end-to-end.

Everything is cached as a single GeoTIFF under `data.cache_dir` so repeat
runs never re-fetch.
"""
from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swiss_rail_rl.config import Config
from swiss_rail_rl.data.cities import Point, bounding_box

_TERRARIUM_ZOOM = 9  # ~200-250m/px at Swiss latitudes; matches default sample_step_m
_TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_GRID_RES_M = 200.0  # output raster resolution in the projected CRS


@dataclass
class ElevationGrid:
    array: np.ndarray  # (rows, cols), float32 metres, row 0 = north
    xmin: float
    ymax: float
    res: float  # metres/pixel
    crs: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    def extent(self) -> tuple[float, float, float, float]:
        rows, cols = self.array.shape
        return (self.xmin, self.xmin + cols * self.res, self.ymax - rows * self.res, self.ymax)

    def _to_pixel(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = (xs - self.xmin) / self.res
        row = (self.ymax - ys) / self.res
        return row, col

    def sample(self, xs: np.ndarray | float, ys: np.ndarray | float) -> np.ndarray:
        """Bilinear elevation lookup at projected-CRS coordinates. Out-of-bounds clamps to edge."""
        from scipy.ndimage import map_coordinates

        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        row, col = self._to_pixel(xs, ys)
        rows, cols = self.array.shape
        row = np.clip(row, 0, rows - 1)
        col = np.clip(col, 0, cols - 1)
        vals = map_coordinates(self.array, [row, col], order=1, mode="nearest")
        return vals

    def hillshade(self, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
        """Standard analytic hillshade, normalised to [0, 1] for use as a background image."""
        z = self.array.astype(np.float64)
        gy, gx = np.gradient(z, self.res)
        slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        az = np.deg2rad(360.0 - azimuth_deg + 90.0)
        alt = np.deg2rad(altitude_deg)
        shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        shaded = np.clip(shaded, 0, 1)
        return shaded


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2**zoom

    def _lat(yy: int) -> float:
        r = math.pi * (1 - 2 * yy / n)
        return math.degrees(math.atan(math.sinh(r)))

    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = _lat(y)
    lat_min = _lat(y + 1)
    return lon_min, lat_min, lon_max, lat_max


def _fetch_terrarium_mosaic(
    bbox_wgs84: tuple[float, float, float, float], zoom: int, cache_dir: Path
) -> tuple[np.ndarray, float, float, float]:
    """Downloads and stitches Terrarium PNG tiles covering bbox_wgs84 = (lon_min, lat_min, lon_max, lat_max).
    Returns (elevation_array, lon_min, lat_max, deg_per_px) in plain WGS84 pixel space.
    Raises on any network failure — caller decides the fallback.
    """
    import requests
    from PIL import Image

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    x0, y0 = _lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y1 = _lonlat_to_tile(lon_max, lat_min, zoom)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    tile_cache = cache_dir / "terrarium_tiles" / str(zoom)
    tile_cache.mkdir(parents=True, exist_ok=True)

    n_tiles_x = x1 - x0 + 1
    n_tiles_y = y1 - y0 + 1
    mosaic = np.zeros((n_tiles_y * 256, n_tiles_x * 256), dtype=np.float32)

    session = requests.Session()
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile_path = tile_cache / f"{tx}_{ty}.png"
            if tile_path.exists():
                data = tile_path.read_bytes()
            else:
                url = _TERRARIUM_URL.format(z=zoom, x=tx, y=ty)
                resp = session.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.content
                tile_path.write_bytes(data)
            img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
            elev = img[:, :, 0] * 256.0 + img[:, :, 1] + img[:, :, 2] / 256.0 - 32768.0
            row0 = (ty - y0) * 256
            col0 = (tx - x0) * 256
            mosaic[row0 : row0 + 256, col0 : col0 + 256] = elev

    top_left_lon_min, _, _, top_left_lat_max = _tile_bounds(x0, y0, zoom)
    _, _, bottom_right_lon_max, _ = _tile_bounds(x1, y0, zoom)
    deg_per_px = (bottom_right_lon_max - top_left_lon_min) / (n_tiles_x * 256)
    return mosaic, top_left_lon_min, top_left_lat_max, deg_per_px


def _warp_to_projected(
    mosaic: np.ndarray,
    lon_min: float,
    lat_max: float,
    deg_per_px: float,
    src_crs: str,
    dst_crs: str,
    dst_bbox: tuple[float, float, float, float],
    dst_res: float,
) -> ElevationGrid:
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    src_transform = Affine.translation(lon_min, lat_max) * Affine.scale(deg_per_px, -deg_per_px)

    xmin, ymin, xmax, ymax = dst_bbox
    cols = int(math.ceil((xmax - xmin) / dst_res))
    rows = int(math.ceil((ymax - ymin) / dst_res))
    dst_transform = Affine.translation(xmin, ymax) * Affine.scale(dst_res, -dst_res)
    dst = np.zeros((rows, cols), dtype=np.float32)

    reproject(
        source=mosaic,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return ElevationGrid(array=dst, xmin=xmin, ymax=ymax, res=dst_res, crs=dst_crs)


def _synthetic_terrain(
    dst_bbox: tuple[float, float, float, float],
    points: list[Point],
    dst_res: float,
    dst_crs: str,
    seed: int,
) -> ElevationGrid:
    """Procedural terrain resembling Switzerland's coarse geography: a low, flat
    Mittelland band running roughly Geneva-Bern-Zurich-St.Gallen, an Alpine wall
    rising sharply to the south (Valais/Ticino/Graubünden), and a lower Jura
    ridge along the north-west border. Used only when real elevation data is
    unreachable, so the rest of the pipeline can still be exercised end-to-end.
    """
    warnings.warn(
        "Elevation download unavailable — using synthetic Swiss-like terrain "
        "as a fallback. Real elevation data will be used automatically once "
        "network access to the configured source is available.",
        stacklevel=2,
    )
    xmin, ymin, xmax, ymax = dst_bbox
    cols = int(math.ceil((xmax - xmin) / dst_res))
    rows = int(math.ceil((ymax - ymin) / dst_res))
    xs = xmin + (np.arange(cols) + 0.5) * dst_res
    ys = ymax - (np.arange(rows) + 0.5) * dst_res
    gx, gy = np.meshgrid(xs, ys)

    # Normalise the north-south axis to [0, 1] (0 = south edge, 1 = north edge).
    t = (gy - ymin) / max(ymax - ymin, 1.0)

    # Alps: a broad ridge centred a bit south of the middle, rising steeply.
    alps = 2600.0 * np.exp(-((t - 0.30) ** 2) / (2 * 0.10**2))
    # Mittelland plateau: gentle base elevation ~450m across the middle band.
    plateau = 450.0 + 150.0 * np.exp(-((t - 0.62) ** 2) / (2 * 0.20**2))
    # Jura: a lower ridge near the north-west edge.
    s = (gx - xmin) / max(xmax - xmin, 1.0)
    jura = 500.0 * np.exp(-((s - 0.15) ** 2) / (2 * 0.08**2)) * np.exp(-((t - 0.75) ** 2) / (2 * 0.15**2))

    base = 300.0 + plateau * 0.3 + alps + jura

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(rows, cols)).astype(np.float32)
    from scipy.ndimage import gaussian_filter

    noise = gaussian_filter(noise, sigma=max(1.0, 4000.0 / dst_res)) * 180.0

    array = np.clip(base + noise, 190.0, 4500.0).astype(np.float32)
    return ElevationGrid(array=array, xmin=xmin, ymax=ymax, res=dst_res, crs=dst_crs)


def load_elevation(config: Config, points: list[Point]) -> ElevationGrid:
    from pyproj import Transformer

    cache_dir = config.cache_dir()
    cache_path = cache_dir / f"elevation_{config.data.elevation_source}.npz"

    dst_bbox = bounding_box(points)
    dst_crs = config.region.crs_projected

    if cache_path.exists():
        with np.load(cache_path) as npz:
            return ElevationGrid(
                array=npz["array"],
                xmin=float(npz["xmin"]),
                ymax=float(npz["ymax"]),
                res=float(npz["res"]),
                crs=str(npz["crs"]),
            )

    try:
        transformer = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
        xmin, ymin, xmax, ymax = dst_bbox
        corners_lon = []
        corners_lat = []
        for cx, cy in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]:
            lon, lat = transformer.transform(cx, cy)
            corners_lon.append(lon)
            corners_lat.append(lat)
        bbox_wgs84 = (min(corners_lon), min(corners_lat), max(corners_lon), max(corners_lat))

        mosaic, lon_min, lat_max, deg_per_px = _fetch_terrarium_mosaic(
            bbox_wgs84, _TERRARIUM_ZOOM, cache_dir
        )
        grid = _warp_to_projected(
            mosaic, lon_min, lat_max, deg_per_px, "EPSG:4326", dst_crs, dst_bbox, _GRID_RES_M
        )
    except Exception as exc:  # network unavailable, tile server unreachable, etc.
        warnings.warn(f"Real elevation fetch failed ({exc!r}); falling back to synthetic terrain.")
        grid = _synthetic_terrain(dst_bbox, points, _GRID_RES_M, dst_crs, seed=config.rl.seed)

    np.savez_compressed(
        cache_path, array=grid.array, xmin=grid.xmin, ymax=grid.ymax, res=grid.res, crs=grid.crs
    )
    return grid
