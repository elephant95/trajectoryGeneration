"""Turns the training frame log + a final reveal into an MP4 (§10).

Stage 1 animates the network evolving over training: hillshade background,
labelled city dots, edges/junctions cross-fading between logged checkpoints,
and a corner panel with the live cost curve. Stage 2 holds on the final
learned network, fades in the real Swiss rail geometry in a contrasting
colour, and shows the fitted-weight sentence plus a takeaway line before an
end card.

`render.backend` selects the implementation: "matplotlib" (default, used
here for both stages) or "manim" for a more polished stage-2 reveal — see
`render_manim_reveal` below, which requires the optional `manim` extra.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swiss_rail_rl.baselines import Network
from swiss_rail_rl.config import Config
from swiss_rail_rl.data.cities import Point
from swiss_rail_rl.data.elevation import ElevationGrid
from swiss_rail_rl.data.rail import RailNetwork

logger = logging.getLogger(__name__)

_LEARNED_COLOR = "#ff6b35"
_REAL_COLOR = "#2ec4b6"
_JUNCTION_COLOR = "#ffd166"
_CITY_COLOR = "#f2f2f2"
_BG_DARK = "#0b1622"


def _load_frames(frames_dir: Path) -> list[dict]:
    frames = []
    for path in sorted(frames_dir.glob("frame_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            frames.append(json.load(f))
    return frames


def _load_cost_csv(csv_path: Path) -> tuple[list[float], list[float]]:
    if not csv_path.exists():
        return [], []
    steps, costs = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            costs.append(float(row["best_cost"]))
    return steps, costs


def _setup_axes(fig, resolution: tuple[int, int]):
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.80])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(_BG_DARK)
    return ax


def _draw_hillshade(ax, elevation: ElevationGrid, show: bool):
    if not show:
        return
    shade = elevation.hillshade()
    ax.imshow(shade, cmap="gray", extent=elevation.extent(), alpha=0.35, aspect="equal", zorder=0)


def _draw_cities(ax, points: list[Point]):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    ax.scatter(xs, ys, s=40 if len(points) < 30 else 22, color=_CITY_COLOR, edgecolors="black", linewidths=0.6, zorder=5)
    for p in points:
        ax.annotate(
            p.name, (p.x, p.y), textcoords="offset points", xytext=(4, 4),
            fontsize=7, color="white", zorder=6,
        )


def _draw_network(ax, edges: list[list[list[float]]], junctions: list[list[float]], alpha: float, color: str):
    for a, b in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=2.0, alpha=alpha, zorder=4, solid_capstyle="round")
    if junctions:
        jx = [j[0] for j in junctions]
        jy = [j[1] for j in junctions]
        ax.scatter(jx, jy, s=30, color=_JUNCTION_COLOR, alpha=alpha, zorder=4.5, marker="D")


def _cost_panel(fig, steps: list[float], costs: list[float], current_step_idx: int, label: str):
    ax = fig.add_axes([0.60, 0.86, 0.36, 0.11])
    ax.set_facecolor(_BG_DARK)
    for spine in ax.spines.values():
        spine.set_color("#556")
    ax.tick_params(colors="#aab", labelsize=6)
    if steps:
        ax.plot(steps, costs, color=_LEARNED_COLOR, linewidth=1.4)
        idx = min(current_step_idx, len(steps) - 1)
        ax.scatter([steps[idx]], [costs[idx]], color="white", s=14, zorder=5)
    ax.set_title(label, color="white", fontsize=7, loc="left")


def render_video(
    config: Config,
    captions: dict,
    points: list[Point],
    elevation: ElevationGrid,
    real: RailNetwork,
    landscape: bool = False,
) -> Path:
    out_dir = config.out_path()
    frames = _load_frames(out_dir / "frames")
    if not frames:
        raise RuntimeError("No training frames found under out/frames — run `train` before `render`.")
    steps, costs = _load_cost_csv(out_dir / "training_log.csv")

    resolution = (config.render.resolution[1], config.render.resolution[0]) if landscape else tuple(config.render.resolution)
    dpi = 120
    figsize = (resolution[0] / dpi, resolution[1] / dpi)

    fps = config.render.fps
    hold_frames_per_checkpoint = max(1, fps // 2)

    import imageio

    output_path = Path(config.render.output_path)
    if landscape:
        output_path = output_path.with_name(output_path.stem + "_landscape" + output_path.suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_caption = captions.get("stage1", {})
    stage2_caption = captions.get("stage2", {})

    fitted_weights_path = out_dir / "fitted_weights.yaml"
    fitted_sentence = None
    if fitted_weights_path.exists():
        import yaml

        with open(fitted_weights_path, "r", encoding="utf-8") as f:
            fitted_sentence = yaml.safe_load(f).get("summary")
    if not fitted_sentence:
        fitted_sentence = stage2_caption.get("fitted_weight_template", "").format(
            slope_weight=config.cost.slope_weight
        )

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8, macro_block_size=1)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=_BG_DARK)
    xmin, xmax, ymin, ymax = elevation.extent()

    logger.info("Rendering stage 1 (training progression) — %d checkpoints", len(frames))
    for i, frame in enumerate(frames):
        for sub in range(hold_frames_per_checkpoint):
            ax = _setup_axes(fig, resolution)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            _draw_hillshade(ax, elevation, config.render.show_elevation_hillshade)
            _draw_cities(ax, points)

            fade = min(1.0, (sub + 1) / hold_frames_per_checkpoint) if i == 0 else 1.0
            _draw_network(ax, frame["edges"], frame["junctions"], fade, _LEARNED_COLOR)

            fig.text(0.03, 0.90, stage1_caption.get("title", ""), color="white", fontsize=9, wrap=True)
            fig.text(
                0.03, 0.02,
                f"step {frame['step']:,} | cost {frame['total_cost']:,.0f} | "
                f"{'connected' if frame['connected'] else 'DISCONNECTED'}",
                color="#ccd", fontsize=7,
            )
            _cost_panel(fig, steps, costs, i, stage1_caption.get("cost_panel_label", "cost"))

            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(img)
            fig.clear()

    logger.info("Rendering stage 2 (reveal)")
    final_frame = frames[-1]
    reveal_frames = int(config.render.reveal_hold_seconds * fps)
    fade_frames = max(1, fps)  # 1 second fade-in for the real network
    for i in range(reveal_frames + fade_frames):
        ax = _setup_axes(fig, resolution)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        _draw_hillshade(ax, elevation, config.render.show_elevation_hillshade)
        _draw_cities(ax, points)
        _draw_network(ax, final_frame["edges"], final_frame["junctions"], 1.0, _LEARNED_COLOR)

        real_alpha = min(1.0, i / fade_frames)
        for line in real.lines.geoms:
            xs, ys = zip(*line.coords)
            ax.plot(xs, ys, color=_REAL_COLOR, linewidth=1.6, alpha=real_alpha * 0.9, zorder=3.5)

        fig.text(0.03, 0.90, stage2_caption.get("hold_label", ""), color=_LEARNED_COLOR, fontsize=9)
        if real_alpha > 0.05:
            fig.text(0.03, 0.87, stage2_caption.get("reveal_label", ""), color=_REAL_COLOR, fontsize=9, alpha=real_alpha)
        fig.text(0.03, 0.12, fitted_sentence, color="white", fontsize=7, wrap=True, alpha=real_alpha)
        fig.text(0.03, 0.06, stage2_caption.get("takeaway", ""), color="#ccd", fontsize=7, wrap=True, alpha=real_alpha)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(img)
        fig.clear()

    end_card_frames = fps * 2
    for _ in range(end_card_frames):
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(_BG_DARK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, stage2_caption.get("end_card", ""), color="white", fontsize=12, ha="center", va="center", wrap=True, transform=ax.transAxes)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(img)
        fig.clear()

    writer.close()
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def render_manim_reveal(config: Config, points: list[Point], real: RailNetwork, final_network: dict) -> Path:
    """Optional polished stage-2 reveal using manim (`render.backend: "manim"`).
    Requires the `manim` optional dependency (`pip install .[manim]`).
    """
    try:
        import manim  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "render.backend is 'manim' but the manim package isn't installed. "
            "Install it with `pip install .[manim]` or switch render.backend to 'matplotlib'."
        ) from exc

    raise NotImplementedError(
        "The manim reveal scene is a documented extension point (matplotlib's stage-2 reveal "
        "in render_video() already produces the same beats — hold, fade-in overlay, captions, "
        "end card — and is the default/tested path). Wire a manim.Scene here for extra polish."
    )
