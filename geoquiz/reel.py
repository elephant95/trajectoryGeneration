"""Reel assembler: a Puzzle -> one folder holding the video and its metadata.

Output per puzzle: ``video.mp4`` (1080 x 1920, H.264, AAC), ``cover.jpg``,
``caption.txt``, ``answer.json``, ``metadata.json`` and ``data/`` with the
clipped layers behind the video.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import skia

from . import __version__, licences
from .assets import flag_svg
from .puzzles import Puzzle
from .render.encode import VideoWriter
from .render.overlay import Overlay, Timeline, render_globe
from .render.reveal import build_stack, compose


@dataclass(frozen=True)
class ReelSpec:
    duration: float = 15.0      # total length; the phases scale with it unless ``phases`` is set
    phases: tuple[float, float, float, float] | None = None   # seconds of reveal, countdown, answer, outro
    crf: int = 18
    preset: str = "medium"
    zoom: float = 0.05          # slow push-in over the clip, as a fraction of the frame
    audio: str | None = None    # path to a music file to embed; silent track otherwise

    def timeline(self) -> Timeline:
        return Timeline.from_phases(*self.phases) if self.phases else Timeline.for_duration(self.duration)


def _background(frame, style) -> np.ndarray:
    """Dark background with a soft lift behind the map band."""
    y, x = np.mgrid[0:frame.height, 0:frame.width].astype(np.float32)
    d = np.hypot((x - frame.width / 2) / (frame.width * 0.75), (y - frame.height / 2) / (frame.height * 0.45))
    k = np.exp(-3.0 * d * d)[..., None]        # smooth falloff, no visible edge
    bg = np.array(style.background, np.float32) * (1 - k) + np.array(style.background_centre, np.float32) * k
    return (bg * 255).astype(np.uint8)


def _zoom(img: np.ndarray, z: float) -> np.ndarray:
    if z <= 1.0:
        return img
    h, w = img.shape[:2]
    M = np.array([[z, 0, (1 - z) * w / 2], [0, z, (1 - z) * h / 2]], np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)


def render_reel(puzzle: Puzzle, out_dir: Path, raw_root: Path | None = None, land=None,
                spec: ReelSpec = ReelSpec(), index: int | None = None) -> dict:
    """Render ``puzzle`` into ``out_dir/<slug>/``; returns the metadata that was written."""
    t_start = time.perf_counter()
    scene, style, frame = puzzle.scene, puzzle.style, puzzle.scene.frame
    tl = spec.timeline()
    invert = style.mode == "subtract"
    folder = Path(out_dir) / (f"{index:04d}_{puzzle.slug}" if index is not None else puzzle.slug)
    folder.mkdir(parents=True, exist_ok=True)

    stack = build_stack(frame, scene.layers, style.fade, style.head, puzzle.gain, invert=invert)
    answer_stack = build_stack(frame, scene.answer_layers, 0.03, style.head, invert=invert) if scene.answer_layers else None
    bg = _background(frame, style)
    globe = render_globe(land, *scene.region.centre, 118, style, puzzle.accent) if land is not None and len(land) else None
    flag = flag_svg(raw_root, puzzle.answer.get("iso_a2")) if raw_root else None
    overlay = Overlay(frame, style, puzzle.texts, tl, flag_svg=flag, globe=globe, accent=puzzle.accent)
    t_setup = time.perf_counter() - t_start

    rgba = np.zeros((frame.height, frame.width, 4), np.uint8)
    rgba[..., 3] = 255
    surface = skia.Surface(rgba)                  # shares memory with ``rgba``
    canvas = surface.getCanvas()
    n_frames = int(round(tl.duration * frame.fps))
    cover_at = int(tl.reveal_end * frame.fps) + 1
    frame_ms, moving = [], []
    t_frames = time.perf_counter()
    with VideoWriter(folder / "video.mp4", frame, crf=spec.crf, preset=spec.preset,
                     audio=Path(spec.audio) if spec.audio else None) as video:
        for i in range(n_frames):
            t0 = time.perf_counter()
            t = i / frame.fps
            p = min(t / tl.reveal_end, 1.0) * stack.end
            ink = stack.frame(p)
            moving.append(stack.moving_fraction(p))
            if answer_stack is not None and t >= tl.countdown_end:
                pa = min((t - tl.countdown_end) / 1.4, 1.0) * answer_stack.end
                ink = cv2.add(ink, answer_stack.frame(pa))
            img = compose(ink, bg, style.glow_near, style.glow_far, style.mode)
            if t > tl.answer_end:                 # outro: the map fades so the clip loops into its first frame
                o = min((t - tl.answer_end) / ((tl.duration - tl.answer_end) * 0.6), 1.0)
                img = cv2.addWeighted(img, 1 - o, bg, o, 0)
            img = _zoom(img, 1.0 + spec.zoom * t / tl.duration)
            rgba[..., :3] = img
            overlay.draw(canvas, t)
            video.write(rgba)
            if i == cover_at:
                cv2.imwrite(str(folder / "cover.jpg"), cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            frame_ms.append((time.perf_counter() - t0) * 1000)
        t_frames = time.perf_counter() - t_frames
    t_total = time.perf_counter() - t_start

    # -- sidecar files ----------------------------------------------------------
    data_dir = folder / "data"
    data_dir.mkdir(exist_ok=True)
    for name, gdf in puzzle.extracts.items():
        gdf.to_parquet(data_dir / f"{name}.parquet", index=False)

    full_credits = []
    for s in scene.sources:
        text = s["licence"]["attribution_full"]
        if text not in full_credits:
            full_credits.append(text)
    question = " ".join(puzzle.texts.question)
    (folder / "caption.txt").write_text(
        f"{question} {puzzle.texts.hint}.\nAnswer at the end — tell us how fast you got it.\n\n"
        f"#geography #geoquiz #maps #mapquiz #guessthe{scene.region.kind.value} #cartography\n\n"
        "Credits: " + " ".join(full_credits) + "\n", encoding="utf-8")
    (folder / "answer.json").write_text(json.dumps(puzzle.answer, indent=2, ensure_ascii=False), encoding="utf-8")

    region = scene.region
    params = {"spec": asdict(spec), "frame": asdict(frame), "style": style.name, "feature": puzzle.feature,
              **puzzle.params, "data_versions": {s["layer"]: s["data_version"] for s in scene.sources}}
    meta = {
        "slug": puzzle.slug,
        "generator": f"geoquiz {__version__}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": {"kind": region.kind.value, "id": region.id, "name": region.name, "country": region.country,
                   "centre_lonlat": [round(c, 6) for c in region.centre], "crs": region.crs,
                   "population": region.population},
        "feature": puzzle.feature,
        "style": style.name,
        "params": params,
        "params_hash": hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest(),
        "sources": scene.sources,
        "attribution": {"on_screen": puzzle.texts.credit, "caption": full_credits},
        "share_alike_layers": [s["layer"] for s in scene.sources if s["licence"]["share_alike"]],
        "qc": {**scene.stats, "ink_fraction": round(stack.ink_fraction, 4),
               "moving_fraction_peak": round(float(max(moving)), 4)},
        "video": {"file": "video.mp4", "width": frame.width, "height": frame.height, "fps": frame.fps,
                  "frames": n_frames, "duration_s": tl.duration, "bytes": (folder / "video.mp4").stat().st_size},
        "timing": {"setup_s": round(t_setup, 2), "render_and_encode_s": round(t_frames, 2), "total_s": round(t_total, 2),
                   "frame_ms_median": round(float(np.median(frame_ms)), 1)},
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    meta["folder"] = str(folder)
    return meta
