"""Scene builder: common-schema layers -> pixel-space layers with reveal times.

Everything upstream of this module works in WGS84 or metres; everything
downstream works in pixels. A scene layer carries geometry in frame pixels, a
stroke width from the feature's rank, and a reveal time in 0..1 per vertex
(lines) or per polygon (fills). The renderer turns those into a reveal-time
map, so animation cost depends on frame size, not on feature count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .regions import Region
from .schema import Feature

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Frame:
    width: int = 1080
    height: int = 1920
    fps: int = 30

    @property
    def aspect(self) -> float:
        return self.height / self.width


@dataclass(frozen=True)
class View:
    """Metres -> pixels: uniform scale, north up."""
    scale: float      # pixels per metre
    x0: float         # metric x of pixel column 0
    y0: float         # metric y of pixel row 0

    @classmethod
    def fit(cls, extent: tuple[float, float, float, float], box: tuple[float, float, float, float]) -> "View":
        """Fit a metric extent (x0, y0, x1, y1) centred inside a pixel box (left, top, right, bottom)."""
        ex0, ey0, ex1, ey1 = extent
        l, t, r, b = box
        s = min((r - l) / (ex1 - ex0), (b - t) / (ey1 - ey0))
        cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
        return cls(scale=s, x0=cx - ((l + r) / 2) / s, y0=cy + ((t + b) / 2) / s)

    def px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.column_stack([(xy[:, 0] - self.x0) * self.scale, (self.y0 - xy[:, 1]) * self.scale]).astype(np.float32)

    def metric_rect(self, frame: Frame, margin_px: float = 0.0) -> tuple[float, float, float, float]:
        """The metric rectangle covered by the frame, grown by a pixel margin."""
        m = margin_px / self.scale
        return (self.x0 - m, self.y0 - frame.height / self.scale - m, self.x0 + frame.width / self.scale + m, self.y0 + m)


@dataclass
class LineLayer:
    name: str
    feature: Feature
    xy: np.ndarray        # (N, 2) float32 pixel coordinates of every vertex
    start: np.ndarray     # (M + 1,) vertex offsets of each path
    width: np.ndarray     # (M,) stroke width in pixels
    time: np.ndarray      # (N,) reveal time per vertex, 0..1
    color: RGB
    opacity: float = 1.0

    @property
    def n_paths(self) -> int:
        return len(self.start) - 1


@dataclass
class FillLayer:
    name: str
    feature: Feature
    xy: np.ndarray        # (N, 2) float32 ring vertices
    ring_start: np.ndarray  # (R + 1,) vertex offsets of each ring
    ring_poly: np.ndarray   # (R,) polygon index of each ring
    time: np.ndarray      # (P,) reveal time per polygon, 0..1
    color: RGB
    opacity: float = 0.6


@dataclass
class Scene:
    frame: Frame
    view: View
    region: Region
    layers: list = field(default_factory=list)          # puzzle layers, back to front
    answer_layers: list = field(default_factory=list)   # drawn during the answer reveal
    inside: np.ndarray | None = None                    # ring vertices of the target outline, for dimming outside
    inside_start: np.ndarray | None = None
    sources: list[dict] = field(default_factory=list)   # per layer: source, version, licence, attribution
    stats: dict = field(default_factory=dict)

    def attributions(self) -> list[str]:
        return [s["attribution"] for s in self.sources if s.get("on_screen", True)]


# -- geometry -> pixel arrays ---------------------------------------------------

def project_lines(gdf: gpd.GeoDataFrame, region: Region, view: View, frame: Frame,
                  simplify_px: float | None = 0.35, margin_px: float = 40.0):
    """Project, simplify to pixel tolerance, clip to the frame, explode to single paths.

    Returns (xy_metres, start offsets, row index of each path into ``gdf``).
    Pass ``simplify_px=None`` for network layers whose shared vertices must survive.
    """
    geom = gdf.geometry.to_crs(region.crs).to_numpy()
    if simplify_px:
        geom = shapely.simplify(geom, simplify_px / view.scale, preserve_topology=False)
    geom = shapely.clip_by_rect(geom, *view.metric_rect(frame, margin_px))
    parts, row = shapely.get_parts(geom, return_index=True)
    keep = (shapely.get_type_id(parts) == 1) & ~shapely.is_empty(parts)
    parts, row = parts[keep], row[keep]
    xy, idx = shapely.get_coordinates(parts, return_index=True)
    start = np.concatenate([[0], np.flatnonzero(np.diff(idx)) + 1, [len(idx)]]) if len(idx) else np.zeros(1, int)
    return xy, start.astype(np.int64), row


def project_polygons(gdf: gpd.GeoDataFrame, region: Region, view: View, frame: Frame,
                     simplify_px: float | None = 0.35, margin_px: float = 40.0):
    """Like :func:`project_lines` for polygons. Returns (xy_metres, ring_start, ring_poly, row index per polygon)."""
    geom = gdf.geometry.to_crs(region.crs).to_numpy()
    if simplify_px:
        geom = shapely.simplify(geom, simplify_px / view.scale, preserve_topology=True)
    geom = shapely.clip_by_rect(shapely.make_valid(geom), *view.metric_rect(frame, margin_px))
    polys, row = shapely.get_parts(geom, return_index=True)
    keep = (shapely.get_type_id(polys) == 3) & ~shapely.is_empty(polys)
    polys, row = polys[keep], row[keep]
    rings, ring_poly = shapely.get_rings(polys, return_index=True)
    xy, idx = shapely.get_coordinates(rings, return_index=True)
    start = np.concatenate([[0], np.flatnonzero(np.diff(idx)) + 1, [len(idx)]]) if len(idx) else np.zeros(1, int)
    return xy, start.astype(np.int64), ring_poly, row


def stroke_widths(rank: np.ndarray, w_min: float, w_max: float, gamma: float = 1.5) -> np.ndarray:
    """Stroke width from importance rank: thin for minor features, wide for major ones."""
    return (w_min + (w_max - w_min) * np.clip(rank, 0, 1) ** gamma).astype(np.float32)


def path_lengths(xy: np.ndarray, start: np.ndarray):
    """Per-vertex distance from the start of its path, and each path's total length."""
    seg = np.zeros(len(xy))
    if len(xy) > 1:
        seg[1:] = np.hypot(*np.diff(xy, axis=0).T)
    seg[start[:-1]] = 0.0
    cum = np.cumsum(seg)
    along = cum - np.repeat(cum[start[:-1]], np.diff(start))
    total = cum[start[1:] - 1] - cum[start[:-1]]
    return along, total


# -- reveal times ---------------------------------------------------------------

def normalise_times(d: np.ndarray, d_max: float | None = None, quantile: float = 0.99) -> np.ndarray:
    """Distances -> reveal times in 0..1, robust to a few far outliers."""
    finite = d[np.isfinite(d)]
    if d_max is None:
        d_max = float(np.quantile(finite, quantile)) if len(finite) else 1.0
    return np.clip(d / max(d_max, 1e-9), 0.0, 1.0).astype(np.float32)


def network_distance(xy: np.ndarray, start: np.ndarray, origin: tuple[float, float], snap_m: float = 0.1,
                     detour: float = 1.4) -> np.ndarray:
    """Shortest-path distance along a line network from the vertex nearest ``origin``.

    Vertices closer than ``snap_m`` are one node. Vertices the network does not
    reach (disconnected pieces) get straight-line distance times ``detour``, so
    they still appear at a plausible moment.
    """
    key = np.round(xy / snap_m).astype(np.int64)
    _, node = np.unique(key, axis=0, return_inverse=True)
    node = node.ravel()
    a, b = np.arange(len(xy) - 1), np.arange(1, len(xy))
    ok = np.ones(len(xy) - 1, bool)
    ok[start[1:-1] - 1] = False            # no edge across path boundaries
    a, b = a[ok], b[ok]
    w = np.hypot(*(xy[b] - xy[a]).T)
    n = int(node.max()) + 1
    graph = coo_matrix((w, (node[a], node[b])), shape=(n, n)).tocsr()
    src = node[int(np.argmin(np.hypot(xy[:, 0] - origin[0], xy[:, 1] - origin[1])))]
    d = dijkstra(graph, directed=False, indices=src)[node]
    crow = np.hypot(xy[:, 0] - origin[0], xy[:, 1] - origin[1]) * detour
    return np.where(np.isfinite(d), d, crow)


def upstream_distance(xy: np.ndarray, start: np.ndarray, dist_dn_km: np.ndarray, basin: np.ndarray) -> np.ndarray:
    """Per-vertex distance (metres) upstream from where each basin leaves the scene.

    ``dist_dn_km`` is each path's distance from its downstream end to the sea
    (HydroRIVERS DIST_DN_KM); paths are digitised in flow direction. Rivers
    then grow from the frame edge or the sea up into their headwaters, trunks first.
    """
    along, total = path_lengths(xy, start)
    counts = np.diff(start)
    d = np.repeat(dist_dn_km * 1000.0 + total, counts) - along
    b = np.repeat(basin, counts)
    order = np.argsort(b, kind="stable")
    bs = b[order]
    first = np.concatenate([[0], np.flatnonzero(np.diff(bs)) + 1])
    exit_d = np.minimum.reduceat(d[order], first)
    return d - np.repeat(exit_d, np.diff(np.concatenate([first, [len(bs)]])))[np.argsort(order, kind="stable")]


def nearest_time(points: np.ndarray, ref_xy: np.ndarray, ref_time: np.ndarray) -> np.ndarray:
    """Reveal time of the nearest reference vertex (ties a lake to the river that reaches it)."""
    if not len(ref_xy):
        return np.zeros(len(points), np.float32)
    _, i = cKDTree(ref_xy).query(points)
    return ref_time[i].astype(np.float32)


def outline_times(xy: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Draw-on times for an outline: every ring is traced once, start to end, over 0..1."""
    along, total = path_lengths(xy, start)
    return (along / np.repeat(np.maximum(total, 1e-9), np.diff(start))).astype(np.float32)
