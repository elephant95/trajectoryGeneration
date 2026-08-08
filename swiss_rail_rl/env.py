"""Gymnasium environment: sequential rail-network construction.

Implements Design A from the spec (discrete + junction placement): each step
the agent adds an edge between two existing nodes, places a new junction at
one of a fixed set of candidate positions, or stops. Design B (continuous
junction positions) is left as a documented extension point — `action_design`
is a config knob precisely so a second implementation can be swapped in
later without touching train.py/cli.py.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from swiss_rail_rl.baselines import _generate_candidates, farthest_point_sample, mst_baseline
from swiss_rail_rl.config import Config
from swiss_rail_rl.cost import CostOracle
from swiss_rail_rl.data.cities import bounding_box

XY = tuple[float, float]


class SwissRailEnv(gym.Env):
    """State: fixed city coordinates, current junctions/edges, connectivity.
    Action: discrete index into [place-junction candidates] ∪ [add-edge pairs] ∪ [STOP].
    Reward: dense per-step frugality shaping (-marginal edge cost) plus a
    terminal -TotalCost(G) if all cities end up connected, or a large
    connectivity penalty if not (§6 of the build spec).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, cost_oracle: CostOracle, city_coords: list[XY]):
        super().__init__()
        if config.rl.action_design != "A":
            raise NotImplementedError(
                "rl.action_design 'B' (continuous junction positions) is not implemented; "
                "use 'A'. This is a deliberate extension point, not a bug."
            )
        self.config = config
        self.cost_oracle = cost_oracle
        self.cities: list[XY] = list(city_coords)
        self.n_cities = len(self.cities)
        self.max_junctions = config.steiner.max_junctions
        self.max_steps = config.rl.max_steps

        bbox = bounding_box([_FakePoint(x, y) for x, y in self.cities])
        self.bbox = bbox
        mst = mst_baseline(cost_oracle, self.cities)
        raw_candidates = _generate_candidates(
            config.steiner.junction_grid,
            bbox,
            config.steiner.candidate_resolution_km,
            self.cities,
            mst.edges,
        )
        if not raw_candidates:
            raw_candidates = [((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)]
        self.candidate_positions: list[XY] = farthest_point_sample(
            raw_candidates, config.steiner.candidate_pool_cap
        )
        self.n_candidates = len(self.candidate_positions)

        # Reference cost used to scale the disconnected-network penalty (see
        # config.rl.disconnected_penalty_weight) — the MST is an upper bound
        # on the achievable connected cost, so scaling off it guarantees
        # connecting is always preferred over stopping early disconnected.
        self._mst_cost_scaled = mst.cost * config.rl.reward_scale

        self.n_node_slots = self.n_cities + self.max_junctions
        self._edge_pairs: list[tuple[int, int]] = [
            (i, j) for i in range(self.n_node_slots) for j in range(i + 1, self.n_node_slots)
        ]
        self.n_edge_actions = len(self._edge_pairs)
        self._edge_index = {pair: idx for idx, pair in enumerate(self._edge_pairs)}

        self.STOP = self.n_candidates + self.n_edge_actions
        self.n_actions = self.STOP + 1
        self.action_space = spaces.Discrete(self.n_actions)

        obs_dim = self.n_cities * 2 + self.max_junctions * 3 + self.n_edge_actions + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(config.rl.seed)
        self.junction_positions: list[XY | None] = [None] * self.max_junctions
        self.junction_active: list[bool] = [False] * self.max_junctions
        self.adjacency: set[tuple[int, int]] = set()
        self.step_count = 0

    # -- helpers ---------------------------------------------------------
    def _node_xy(self, idx: int) -> XY:
        if idx < self.n_cities:
            return self.cities[idx]
        slot = idx - self.n_cities
        pos = self.junction_positions[slot]
        assert pos is not None
        return pos

    def _node_valid(self, idx: int) -> bool:
        if idx < self.n_cities:
            return True
        return self.junction_active[idx - self.n_cities]

    def _normalize(self, xy: XY) -> tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bbox
        nx_ = 2.0 * (xy[0] - xmin) / max(xmax - xmin, 1.0) - 1.0
        ny_ = 2.0 * (xy[1] - ymin) / max(ymax - ymin, 1.0) - 1.0
        return nx_, ny_

    def _used_junction_positions(self) -> list[XY]:
        used = []
        for slot in range(self.max_junctions):
            if not self.junction_active[slot]:
                continue
            node_idx = self.n_cities + slot
            if any(node_idx in pair for pair in self.adjacency):
                used.append(self.junction_positions[slot])
        return used

    def _get_obs(self) -> np.ndarray:
        parts = []
        for xy in self.cities:
            parts.extend(self._normalize(xy))
        for slot in range(self.max_junctions):
            if self.junction_active[slot]:
                nx_, ny_ = self._normalize(self.junction_positions[slot])
                parts.extend([nx_, ny_, 1.0])
            else:
                parts.extend([0.0, 0.0, 0.0])
        adj_flags = [1.0 if pair in self.adjacency else 0.0 for pair in self._edge_pairs]
        parts.extend(adj_flags)
        parts.append(self.step_count / max(self.max_steps, 1))
        return np.asarray(parts, dtype=np.float32)

    def _current_edges_xy(self) -> list[tuple[XY, XY]]:
        return [(self._node_xy(i), self._node_xy(j)) for i, j in self.adjacency]

    def _current_network_cost(self) -> float:
        return self.cost_oracle.network_cost(self._current_edges_xy(), self._used_junction_positions())

    def _connectivity(self) -> tuple[bool, int]:
        g = nx.Graph()
        g.add_nodes_from(range(self.n_node_slots))
        g.add_edges_from(self.adjacency)
        city_nodes = list(range(self.n_cities))
        comp_of = {}
        for comp_id, comp in enumerate(nx.connected_components(g)):
            for n in comp:
                comp_of[n] = comp_id
        city_components = {comp_of[c] for c in city_nodes}
        return len(city_components) <= 1, len(city_components)

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.junction_positions = [None] * self.max_junctions
        self.junction_active = [False] * self.max_junctions
        self.adjacency = set()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        rcfg = self.config.rl
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        if action == self.STOP:
            terminated = True
        elif action < self.n_candidates:
            cand_xy = self.candidate_positions[action]
            empty_slot = next((s for s in range(self.max_junctions) if not self.junction_active[s]), None)
            already_placed = any(
                self.junction_active[s] and self.junction_positions[s] == cand_xy
                for s in range(self.max_junctions)
            )
            if empty_slot is None or already_placed:
                reward -= rcfg.invalid_action_penalty
            else:
                self.junction_positions[empty_slot] = cand_xy
                self.junction_active[empty_slot] = True
        else:
            edge_idx = action - self.n_candidates
            i, j = self._edge_pairs[edge_idx]
            if not (self._node_valid(i) and self._node_valid(j)) or (i, j) in self.adjacency:
                reward -= rcfg.invalid_action_penalty
            else:
                a_xy, b_xy = self._node_xy(i), self._node_xy(j)
                marginal_cost = self.cost_oracle.edge_cost(a_xy, b_xy)
                self.adjacency.add((i, j))
                reward -= marginal_cost * rcfg.reward_scale

        self.step_count += 1
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        if terminated or truncated:
            connected, n_components = self._connectivity()
            total_cost = self._current_network_cost()
            info["connected"] = connected
            info["n_components"] = n_components
            info["total_cost"] = total_cost
            if connected:
                reward += -total_cost * rcfg.reward_scale
            else:
                reward += -rcfg.disconnected_penalty_weight * self._mst_cost_scaled * n_components

        return self._get_obs(), reward, terminated, truncated, info

    def current_network(self) -> dict:
        """Nodes + edges + cost + connectivity, for the trainer to snapshot
        into the frame log that the renderer animates.
        """
        connected, n_components = self._connectivity()
        used_junctions = self._used_junction_positions()
        return {
            "cities": [list(xy) for xy in self.cities],
            "junctions": [list(xy) for xy in used_junctions],
            "edges": [[list(a), list(b)] for a, b in self._current_edges_xy()],
            "total_cost": self._current_network_cost(),
            "connected": connected,
            "n_components": n_components,
        }


class _FakePoint:
    """Minimal stand-in so bounding_box() (which expects .x/.y) can be reused
    on plain (x, y) tuples without importing the full Point/city-projection
    machinery into the env module.
    """

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
