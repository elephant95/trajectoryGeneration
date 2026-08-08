"""Classical reference solvers: minimum spanning tree over cities, and a
greedy Steiner-point insertion heuristic on top of it. These exist so we
know the target cost *before* spending RL compute (§6 of the spec), and
because `fit.py` needs a cheap inner-loop solver — full RL is far too slow
to run inside a weight-fitting search.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from swiss_rail_rl.config import SteinerConfig
from swiss_rail_rl.cost import CostOracle

XY = tuple[float, float]


@dataclass
class Network:
    nodes: list[XY]              # cities followed by used junctions
    edges: list[tuple[XY, XY]]
    junctions: list[XY] = field(default_factory=list)
    cost: float = 0.0

    def connected(self, city_coords: list[XY]) -> bool:
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)
        if not city_coords:
            return True
        components = list(nx.connected_components(g))
        for comp in components:
            if any(c in comp for c in city_coords):
                if not all(c in comp for c in city_coords):
                    return False
        return True


def mst_baseline(cost_oracle: CostOracle, city_coords: list[XY]) -> Network:
    """Minimum spanning tree over cities only, using EdgeCost as edge weight."""
    g = nx.Graph()
    g.add_nodes_from(city_coords)
    for i in range(len(city_coords)):
        for j in range(i + 1, len(city_coords)):
            w = cost_oracle.edge_cost(city_coords[i], city_coords[j])
            if math.isfinite(w):
                g.add_edge(city_coords[i], city_coords[j], weight=w)

    mst = nx.minimum_spanning_tree(g, weight="weight")
    edges = list(mst.edges())
    cost = cost_oracle.network_cost(edges, junctions=[])
    return Network(nodes=list(city_coords), edges=edges, junctions=[], cost=cost)


def _grid_candidates(bbox: tuple[float, float, float, float], resolution_m: float) -> list[XY]:
    xmin, ymin, xmax, ymax = bbox
    xs = _frange(xmin, xmax, resolution_m)
    ys = _frange(ymin, ymax, resolution_m)
    return [(x, y) for x in xs for y in ys]


def _frange(start: float, stop: float, step: float) -> list[float]:
    n = max(1, int((stop - start) / step))
    return [start + i * step for i in range(n + 1)]


def _adaptive_candidates(edges: list[tuple[XY, XY]]) -> list[XY]:
    cands: set[XY] = set()
    adjacency: dict[XY, list[XY]] = defaultdict(list)
    for a, b in edges:
        cands.add(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
        adjacency[a].append(b)
        adjacency[b].append(a)
    for node, neighbors in adjacency.items():
        if len(neighbors) >= 2:
            xs = [n[0] for n in neighbors] + [node[0]]
            ys = [n[1] for n in neighbors] + [node[1]]
            cands.add((sum(xs) / len(xs), sum(ys) / len(ys)))
    return list(cands)


def _generate_candidates(
    mode: str,
    bbox: tuple[float, float, float, float],
    resolution_km: float,
    current_nodes: list[XY],
    current_edges: list[tuple[XY, XY]],
    min_separation_m: float = 500.0,
) -> list[XY]:
    candidates: list[XY] = []
    if mode in ("grid", "candidate_set"):
        candidates.extend(_grid_candidates(bbox, resolution_km * 1000.0))
    if mode in ("adaptive", "candidate_set"):
        candidates.extend(_adaptive_candidates(current_edges))

    filtered = []
    for c in candidates:
        if all(math.dist(c, n) > min_separation_m for n in current_nodes):
            filtered.append(c)
    return filtered


def farthest_point_sample(points: list[XY], k: int) -> list[XY]:
    """Greedily picks up to k points from `points` that are spread out, so a
    capped candidate pool still covers the region instead of clustering.
    """
    if len(points) <= k:
        return list(points)
    remaining = list(points)
    picked = [remaining.pop(0)]
    while len(picked) < k and remaining:
        best_idx, best_dist = 0, -1.0
        for idx, p in enumerate(remaining):
            d = min(math.dist(p, q) for q in picked)
            if d > best_dist:
                best_dist, best_idx = d, idx
        picked.append(remaining.pop(best_idx))
    return picked


def steiner_heuristic(
    cost_oracle: CostOracle,
    city_coords: list[XY],
    steiner_cfg: SteinerConfig,
    bbox: tuple[float, float, float, float],
) -> Network:
    """Greedily inserts the junction that most reduces total network cost
    (MST cost over cities+junctions, plus junction_cost per junction used),
    up to `steiner.max_junctions` times. Stops early once no candidate helps.
    """
    current_nodes = list(city_coords)
    current_edges = mst_baseline(cost_oracle, city_coords).edges
    current_total = sum(cost_oracle.edge_cost(a, b) for a, b in current_edges)
    junctions: list[XY] = []

    for _ in range(steiner_cfg.max_junctions):
        candidates = _generate_candidates(
            steiner_cfg.junction_grid, bbox, steiner_cfg.candidate_resolution_km, current_nodes, current_edges
        )
        if not candidates:
            break

        best_gain = 0.0
        best_candidate = None
        best_edges = None
        best_raw_cost = None

        for cand in candidates:
            trial_nodes = current_nodes + [cand]
            g = nx.Graph()
            g.add_nodes_from(trial_nodes)
            for i in range(len(trial_nodes)):
                for j in range(i + 1, len(trial_nodes)):
                    w = cost_oracle.edge_cost(trial_nodes[i], trial_nodes[j])
                    if math.isfinite(w):
                        g.add_edge(trial_nodes[i], trial_nodes[j], weight=w)
            if g.number_of_nodes() > 1 and not nx.is_connected(g):
                continue
            mst = nx.minimum_spanning_tree(g, weight="weight")
            raw_cost = sum(cost_oracle.edge_cost(a, b) for a, b in mst.edges())
            new_total = raw_cost + cost_oracle.cfg.junction_cost
            gain = current_total - new_total
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
                best_edges = list(mst.edges())
                best_raw_cost = raw_cost

        if best_candidate is None:
            break

        current_nodes.append(best_candidate)
        current_edges = best_edges
        current_total = best_raw_cost + cost_oracle.cfg.junction_cost * (len(junctions) + 1)
        junctions.append(best_candidate)

    cost = cost_oracle.network_cost(current_edges, junctions)
    return Network(nodes=current_nodes, edges=current_edges, junctions=junctions, cost=cost)
