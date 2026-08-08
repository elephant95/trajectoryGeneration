"""Fits cost weights so the cost-optimal network resembles the real Swiss
network (§8 of the build spec). The inner loop uses the cheap Steiner
heuristic baseline — never full RL, or a 60-eval search would be
impossibly slow.
"""
from __future__ import annotations

import itertools
import logging

import networkx as nx
import numpy as np
import yaml
from shapely.geometry import LineString

from swiss_rail_rl.baselines import Network, steiner_heuristic
from swiss_rail_rl.config import Config, CostConfig
from swiss_rail_rl.cost import CostOracle
from swiss_rail_rl.data.cities import Point, bounding_box
from swiss_rail_rl.data.elevation import ElevationGrid
from swiss_rail_rl.data.rail import RailNetwork
from swiss_rail_rl.data.water import WaterMask

logger = logging.getLogger(__name__)

XY = tuple[float, float]


def _snap_city_components(real: RailNetwork, points: list[Point], snap_tolerance_m: float) -> dict[str, int | None]:
    """Builds a graph from the real rail line segments and maps each city to
    its connected component, if it lies within snap_tolerance_m of the network.
    """
    g = nx.Graph()

    def _key(xy: XY) -> XY:
        return (round(xy[0] / 25.0) * 25.0, round(xy[1] / 25.0) * 25.0)

    for line in real.lines.geoms:
        coords = [_key(c) for c in line.coords]
        for a, b in zip(coords[:-1], coords[1:]):
            g.add_edge(a, b)

    comp_of: dict[XY, int] = {}
    for comp_id, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            comp_of[n] = comp_id

    nodes = np.array(list(comp_of.keys())) if comp_of else np.zeros((0, 2))
    result: dict[str, int | None] = {}
    for p in points:
        if len(nodes) == 0:
            result[p.name] = None
            continue
        d = np.hypot(nodes[:, 0] - p.x, nodes[:, 1] - p.y)
        idx = int(np.argmin(d))
        if d[idx] <= snap_tolerance_m:
            result[p.name] = comp_of[tuple(nodes[idx])]
        else:
            result[p.name] = None
    return result


def network_distance(
    learned: Network,
    real: RailNetwork,
    points: list[Point],
    sample_step_m: float,
    topology_weight: float,
    snap_tolerance_m: float,
) -> float:
    """Lower = the learned network's geometry and connectivity pattern more
    closely resemble the real Swiss rail network.
    """
    real_geom = real.lines
    total_weighted_dist = 0.0
    total_len = 0.0
    for a, b in learned.edges:
        seg = LineString([a, b])
        if seg.length == 0:
            continue
        n_samples = max(2, int(seg.length / sample_step_m) + 1)
        dists = [seg.interpolate(t, normalized=True).distance(real_geom) for t in np.linspace(0, 1, n_samples)]
        avg_dist = float(np.mean(dists))
        total_weighted_dist += avg_dist * seg.length
        total_len += seg.length
    geometry_term = total_weighted_dist / total_len if total_len > 0 else 0.0

    real_components = _snap_city_components(real, points, snap_tolerance_m)
    learned_graph = nx.Graph()
    learned_graph.add_nodes_from(learned.nodes)
    learned_graph.add_edges_from(learned.edges)
    by_name = {p.name: p for p in points}

    mismatches = 0
    total_pairs = 0
    names = [p.name for p in points]
    for name_a, name_b in itertools.combinations(names, 2):
        pa, pb = by_name[name_a], by_name[name_b]
        learned_conn = nx.has_path(learned_graph, (pa.x, pa.y), (pb.x, pb.y))
        ca, cb = real_components.get(name_a), real_components.get(name_b)
        real_conn = ca is not None and ca == cb
        if learned_conn != real_conn:
            mismatches += 1
        total_pairs += 1
    topology_term = mismatches / total_pairs if total_pairs > 0 else 0.0

    return geometry_term + topology_weight * topology_term


def _candidate_network(
    param_vector: np.ndarray,
    free_params: list[str],
    base_cost_cfg: CostConfig,
    elevation: ElevationGrid,
    water: WaterMask,
    city_coords: list[XY],
    steiner_cfg,
    bbox,
) -> tuple[Network, CostOracle]:
    overrides = {name: float(val) for name, val in zip(free_params, param_vector)}
    cost_cfg = base_cost_cfg.model_copy(update=overrides)
    oracle = CostOracle(cost_cfg, elevation, water)
    network = steiner_heuristic(oracle, city_coords, steiner_cfg, bbox)
    return network, oracle


def fit_weights(
    config: Config,
    points: list[Point],
    elevation: ElevationGrid,
    water: WaterMask,
    real: RailNetwork,
) -> dict:
    city_coords = [(p.x, p.y) for p in points]
    bbox = bounding_box(points)
    free_params = config.fit.free_params
    x0 = np.array([getattr(config.cost, p) for p in free_params], dtype=float)
    lo_mult, hi_mult = config.fit.param_search_range
    bounds = [(v * lo_mult, v * hi_mult) for v in x0]

    def objective(vec: np.ndarray) -> float:
        vec = np.clip(vec, [b[0] for b in bounds], [b[1] for b in bounds])
        network, _oracle = _candidate_network(
            vec, free_params, config.cost, elevation, water, city_coords, config.steiner, bbox
        )
        return network_distance(
            network,
            real,
            points,
            config.fit.geometry_sample_step_m,
            config.fit.topology_weight,
            config.fit.snap_tolerance_m,
        )

    evals_used = 0
    best_x, best_score = x0.copy(), objective(x0)
    evals_used += 1

    if config.fit.method == "cma":
        import cma

        sigma0 = float(np.mean([(b[1] - b[0]) for b in bounds])) * 0.15
        es = cma.CMAEvolutionStrategy(
            x0.tolist(),
            max(sigma0, 1e-6),
            {"bounds": [[b[0] for b in bounds], [b[1] for b in bounds]], "maxfevals": config.fit.max_evals, "verbose": -9},
        )
        while not es.stop() and evals_used < config.fit.max_evals:
            solutions = es.ask()
            scores = [objective(np.array(s)) for s in solutions]
            evals_used += len(solutions)
            es.tell(solutions, scores)
            gen_best_idx = int(np.argmin(scores))
            if scores[gen_best_idx] < best_score:
                best_score = scores[gen_best_idx]
                best_x = np.array(solutions[gen_best_idx])
    elif config.fit.method == "grid":
        n_per_dim = max(2, int(round(config.fit.max_evals ** (1.0 / max(len(free_params), 1)))))
        grids = [np.linspace(b[0], b[1], n_per_dim) for b in bounds]
        for combo in itertools.islice(itertools.product(*grids), config.fit.max_evals):
            vec = np.array(combo)
            score = objective(vec)
            evals_used += 1
            if score < best_score:
                best_score, best_x = score, vec
    else:  # random
        rng = np.random.default_rng(config.rl.seed)
        for _ in range(config.fit.max_evals - 1):
            vec = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            score = objective(vec)
            evals_used += 1
            if score < best_score:
                best_score, best_x = score, vec

    fitted = dict(zip(free_params, [float(v) for v in best_x]))
    summary = _summary_sentence(fitted)
    result = {
        "fitted": fitted,
        "score": float(best_score),
        "method": config.fit.method,
        "evals_used": evals_used,
        "real_source": real.source,
        "summary": summary,
    }

    out_path = config.out_path() / "fitted_weights.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    logger.info(summary)
    return result


def _summary_sentence(fitted: dict[str, float]) -> str:
    if "slope_weight" not in fitted:
        return f"Fitted weights: {fitted}"
    sw = fitted["slope_weight"]
    return (
        f"Fitted slope_weight = {sw:.1f} — the network behaves as if each unit "
        f"of squared gradient is ~{sw:.0f}x as costly as a metre of flat track, "
        "i.e. Swiss planners strongly avoided steep terrain."
    )
