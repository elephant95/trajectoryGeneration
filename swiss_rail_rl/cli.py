"""CLI subcommands: fetch, fit, train, render, all (§3 of the build spec).
Each is independently runnable and re-runs cheaply from cache.
"""
from __future__ import annotations

import argparse
import logging
import sys

import yaml

from swiss_rail_rl.baselines import mst_baseline, steiner_heuristic
from swiss_rail_rl.config import Config, load_captions, load_config
from swiss_rail_rl.cost import CostOracle
from swiss_rail_rl.data.cities import Point, bounding_box, project_points
from swiss_rail_rl.data.elevation import ElevationGrid, load_elevation
from swiss_rail_rl.data.rail import RailNetwork, load_rail_network
from swiss_rail_rl.data.water import WaterMask, load_water_mask

logger = logging.getLogger("swiss_rail_rl")


def _load_context(config: Config) -> tuple[list[Point], ElevationGrid, RailNetwork, WaterMask]:
    points = project_points(config)
    elevation = load_elevation(config, points)
    real = load_rail_network(config, points)
    water = load_water_mask(config, points)
    return points, elevation, real, water


def _build_cost_oracle(config: Config, elevation: ElevationGrid, water: WaterMask) -> CostOracle:
    cost_cfg = config.cost
    fitted_path = config.out_path() / "fitted_weights.yaml"
    if fitted_path.exists():
        with open(fitted_path, "r", encoding="utf-8") as f:
            fitted = yaml.safe_load(f).get("fitted", {})
        if fitted:
            cost_cfg = cost_cfg.model_copy(update=fitted)
            logger.info("Using fitted cost weights from %s: %s", fitted_path, fitted)
    return CostOracle(cost_cfg, elevation, water)


def cmd_fetch(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    points, elevation, real, water = _load_context(config)
    logger.info(
        "Fetched/cached %d points, elevation grid %s, rail network source=%s (%d segments), water=%s",
        len(points), elevation.shape, real.source, len(real.lines.geoms), water.source,
    )


def cmd_fit(args: argparse.Namespace) -> None:
    from swiss_rail_rl.fit import fit_weights

    config = load_config(args.config)
    if not config.fit.enabled:
        logger.info("fit.enabled is false in config — skipping.")
        return
    points, elevation, real, water = _load_context(config)
    result = fit_weights(config, points, elevation, water, real)
    print(result["summary"])


def cmd_train(args: argparse.Namespace) -> None:
    from swiss_rail_rl.train import train as run_train

    config = load_config(args.config)
    points, elevation, real, water = _load_context(config)
    oracle = _build_cost_oracle(config, elevation, water)
    city_coords = [(p.x, p.y) for p in points]
    run_train(config, oracle, city_coords, resume=not args.no_resume)


def cmd_render(args: argparse.Namespace) -> None:
    from swiss_rail_rl.render import render_video

    config = load_config(args.config)
    captions = load_captions(config.captions_path)
    points, elevation, real, _water = _load_context(config)
    render_video(config, captions, points, elevation, real, landscape=args.landscape)


def cmd_baseline(args: argparse.Namespace) -> None:
    """Not one of the five spec subcommands, but a handy sanity check (§12
    step 4): compute and print the classical baseline networks without
    touching RL or rendering at all.
    """
    config = load_config(args.config)
    points, elevation, real, water = _load_context(config)
    oracle = _build_cost_oracle(config, elevation, water)
    city_coords = [(p.x, p.y) for p in points]
    bbox = bounding_box(points)

    mst = mst_baseline(oracle, city_coords)
    print(f"MST baseline cost: {mst.cost:,.1f}")

    steiner = steiner_heuristic(oracle, city_coords, config.steiner, bbox)
    print(f"Steiner heuristic cost: {steiner.cost:,.1f} ({len(steiner.junctions)} junctions used)")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_fetch(args)
    cmd_fit(args)
    cmd_train(args)
    cmd_render(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swiss-rail-rl", description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="Download/cache cities, elevation, real rail network, water").set_defaults(func=cmd_fetch)
    sub.add_parser("fit", help="Fit cost weights against the real network").set_defaults(func=cmd_fit)

    p_train = sub.add_parser("train", help="Train the RL agent, logging frames")
    p_train.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint and start fresh")
    p_train.set_defaults(func=cmd_train)

    p_render = sub.add_parser("render", help="Render the training progression + reveal to MP4")
    p_render.add_argument("--landscape", action="store_true", help="Render a 16:9 variant instead of vertical")
    p_render.set_defaults(func=cmd_render)

    sub.add_parser("baseline", help="Print MST/Steiner baseline costs only (fast sanity check)").set_defaults(func=cmd_baseline)

    p_all = sub.add_parser("all", help="Run fetch -> fit -> train -> render")
    p_all.add_argument("--no-resume", action="store_true")
    p_all.add_argument("--landscape", action="store_true")
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args.func(args)


if __name__ == "__main__":
    main()
