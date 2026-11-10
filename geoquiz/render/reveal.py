"""Reveal engine: animate by a reveal-time map instead of redrawing vectors.

The scene is drawn once with Skia (the crisp final image) and once as a
time map in which every inked pixel stores the moment it appears. A frame is
then a threshold of that map. Per-frame work touches only the pixels that are
mid-transition, and the glow is blurred at reduced resolution, so cost
depends on the frame size and not on how many streets a city has.
"""
from __future__ import annotations

import cv2
import numpy as np
import skia

from ..scene import FillLayer, Frame, LineLayer

SHIFT = 3                     # sub-pixel bits for the time-map rasteriser
MAX_STEP_PX = 3.0             # longest piece of a line that shares one reveal time
TIME_BUCKETS = 512            # time resolution of the map (finer than one frame for any sane reveal)


def _path_for(xy: np.ndarray, starts, ends, close: bool = False) -> skia.Path:
    path = skia.Path()
    pts = xy.tolist()
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        path.moveTo(*pts[s])
        for k in range(s + 1, e):
            path.lineTo(*pts[k])
        if close:
            path.close()
    return path


def draw_layers(frame: Frame, layers: list) -> np.ndarray:
    """Draw layers back to front; returns premultiplied RGBA uint8 (H, W, 4)."""
    surface = skia.Surface(frame.width, frame.height)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorTRANSPARENT)
    for layer in layers:
        r, g, b = layer.color
        paint = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(r, g, b, layer.opacity))
        if isinstance(layer, FillLayer):
            path = _path_for(layer.xy, layer.ring_start[:-1], layer.ring_start[1:], close=True)
            path.setFillType(skia.PathFillType.kEvenOdd)
            paint.setStyle(skia.Paint.kFill_Style)
            canvas.drawPath(path, paint)
            continue
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeCap(skia.Paint.kRound_Cap)
        paint.setStrokeJoin(skia.Paint.kRound_Join)
        bucket = np.round(layer.width * 4).astype(int)          # quarter-pixel width classes, thin first
        for wq in np.unique(bucket):
            sel = np.flatnonzero(bucket == wq)
            paint.setStrokeWidth(wq / 4)
            canvas.drawPath(_path_for(layer.xy, layer.start[:-1][sel], layer.start[1:][sel]), paint)
    return surface.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)


def time_map(frame: Frame, layers: list) -> np.ndarray:
    """Per-pixel reveal time in 0..1 (inf where nothing is drawn), float32 (H, W)."""
    T = np.full((frame.height, frame.width), np.inf, np.float32)
    for layer in layers:
        if isinstance(layer, FillLayer):
            xy = np.round(layer.xy * (1 << SHIFT)).astype(np.int32)
            order = np.argsort(layer.ring_poly, kind="stable")
            bounds = np.searchsorted(layer.ring_poly[order], np.arange(len(layer.time) + 1))
            for p in np.argsort(-layer.time):                    # later first, so earlier wins overlaps
                rings = [xy[layer.ring_start[r]:layer.ring_start[r + 1]] for r in order[bounds[p]:bounds[p + 1]]]
                if rings:
                    cv2.fillPoly(T, rings, float(layer.time[p]), lineType=cv2.LINE_8, shift=SHIFT)
            continue
        if layer.n_paths == 0:
            continue
        n = len(layer.xy)
        a = np.arange(n - 1)
        ok = np.ones(n - 1, bool)
        ok[layer.start[1:-1] - 1] = False
        a = a[ok]
        # split long segments so the reveal travels along them instead of popping them in whole
        p0, p1 = layer.xy[a].astype(np.float64), layer.xy[a + 1].astype(np.float64)
        pieces = np.maximum(np.ceil(np.hypot(*(p1 - p0).T) / MAX_STEP_PX).astype(np.int64), 1)
        a = np.repeat(a, pieces)
        k = np.arange(len(a)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
        n = np.repeat(pieces, pieces)
        f0, f1 = (k / n)[:, None], ((k + 1) / n)[:, None]
        p0, p1 = np.repeat(p0, pieces, axis=0), np.repeat(p1, pieces, axis=0)
        seg = np.round(np.stack([p0 + (p1 - p0) * f0, p0 + (p1 - p0) * f1], axis=1) * (1 << SHIFT)).astype(np.int32)
        t0, t1 = layer.time[a], layer.time[a + 1]
        t = t0 + (t1 - t0) * (f0[:, 0] + f1[:, 0]) / 2
        tb = np.minimum((t * TIME_BUCKETS).astype(np.int32), TIME_BUCKETS - 1)
        path_of = np.repeat(np.arange(layer.n_paths), np.diff(layer.start))[a]
        thick = np.ceil(layer.width[path_of]).astype(np.int32) + 2           # cover the anti-aliased fringe
        key = tb.astype(np.int64) * 64 + np.minimum(thick, 63)
        order = np.argsort(-key, kind="stable")                              # later first, so earlier wins overlaps
        key_sorted = key[order]
        cuts = np.concatenate([[0], np.flatnonzero(np.diff(key_sorted)) + 1, [len(order)]])
        for s, e in zip(cuts[:-1], cuts[1:]):
            k = key_sorted[s]
            cv2.polylines(T, list(seg[order[s:e]]), False, float((k // 64 + 0.5) / TIME_BUCKETS),
                          thickness=int(k % 64), lineType=cv2.LINE_8, shift=SHIFT)
    return T


class RevealStack:
    """A crisp image plus its time map; ``frame(p)`` returns the ink visible at progress ``p``.

    Progress must not decrease between calls (call :meth:`reset` to start over):
    pixels that finished appearing are written once into a running image and
    never touched again.
    """

    def __init__(self, rgba: np.ndarray, T: np.ndarray, fade: float = 0.02, head: float = 0.7):
        h, w = T.shape
        self.shape = (h, w)
        self.fade = max(fade, 1e-6)
        self.head = head
        emission = rgba[..., :3].reshape(-1, 3)
        idx = np.flatnonzero(rgba[..., 3].reshape(-1) > 0)
        t = T.reshape(-1)[idx]
        t = np.where(np.isfinite(t), t, 1.0)           # inked but missed by the time map: appear last
        order = np.argsort(t, kind="stable")
        self.idx, self.t = idx[order], t[order]
        self.col = emission[self.idx].astype(np.float32)
        self.heat = (rgba[..., 3].reshape(-1)[self.idx].astype(np.float32) * head)[:, None]
        self.ink_fraction = len(idx) / (h * w)
        self.reset()

    def reset(self) -> None:
        self.ink = np.zeros((self.shape[0] * self.shape[1], 3), np.uint8)
        self.done = 0
        self.last_p = -np.inf

    @property
    def end(self) -> float:
        """Progress at which every pixel has fully appeared."""
        return 1.0 + self.fade

    def frame(self, p: float) -> np.ndarray:
        if p < self.last_p:
            raise ValueError("progress went backwards; call reset() first")
        self.last_p = p
        k1 = int(np.searchsorted(self.t, p - self.fade, "right"))
        if k1 > self.done:
            self.ink[self.idx[self.done:k1]] = self.col[self.done:k1]
            self.done = k1
        k2 = int(np.searchsorted(self.t, p, "right"))
        out = self.ink.copy()
        if k2 > k1:
            m = ((p - self.t[k1:k2]) / self.fade)[:, None]
            hot = self.heat[k1:k2] * (4 * m * (1 - m))          # white-hot while appearing
            out[self.idx[k1:k2]] = np.clip(self.col[k1:k2] * m + hot, 0, 255)
        return out.reshape(*self.shape, 3)

    def moving_fraction(self, p: float) -> float:
        """Share of frame pixels mid-transition at progress ``p`` (a cost and QC metric)."""
        k1, k2 = np.searchsorted(self.t, [p - self.fade, p], "right")
        return float(k2 - k1) / (self.shape[0] * self.shape[1])


def build_stack(frame: Frame, layers: list, fade: float, head: float, gain: np.ndarray | None = None) -> RevealStack:
    """Rasterise layers into a RevealStack. ``gain`` (H, W) in 0..1 dims the image (soft edges, outside-dimming)."""
    rgba = draw_layers(frame, layers)
    if gain is not None:
        rgba = (rgba.astype(np.float32) * gain[..., None]).astype(np.uint8)
    return RevealStack(rgba, time_map(frame, layers), fade=fade, head=head)


def compose(ink: np.ndarray, background: np.ndarray, glow_near: float, glow_far: float) -> np.ndarray:
    """Neon composite: background + crisp ink + two blurred copies, blurred at reduced resolution."""
    h, w = ink.shape[:2]
    q = cv2.resize(ink, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    near = cv2.GaussianBlur(q, (0, 0), 2.0)
    far = cv2.GaussianBlur(cv2.resize(q, (w // 16, h // 16), interpolation=cv2.INTER_AREA), (0, 0), 2.5)
    far = cv2.resize(far, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
    halo = cv2.addWeighted(near, glow_near, far, glow_far, 0)
    halo = cv2.resize(halo, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.add(cv2.add(background, halo), ink)
