"""The cost function. Every other module (env, baselines, fit, render) computes
network cost only through CostOracle — never re-implements the formula.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import numpy as np

from swiss_rail_rl.config import CostConfig
from swiss_rail_rl.data.elevation import ElevationGrid
from swiss_rail_rl.data.water import WaterMask

XY = tuple[float, float]


class CostOracle:
    """edge_cost(a_xy, b_xy) -> float and network_cost(edges, junctions) -> float
    are the only public entry points every other module should use.
    """

    def __init__(self, cost_config: CostConfig, elevation: ElevationGrid, water: WaterMask):
        self.cfg = cost_config
        self.elevation = elevation
        self.water = water
        self._edge_cache: dict[tuple[XY, XY], float] = {}

    def _slope_penalty(self, a_xy: XY, b_xy: XY, length: float) -> float:
        if length <= 0:
            return 0.0
        n_steps = max(1, int(math.ceil(length / self.cfg.sample_step_m)))
        t = np.linspace(0.0, 1.0, n_steps + 1)
        xs = a_xy[0] + t * (b_xy[0] - a_xy[0])
        ys = a_xy[1] + t * (b_xy[1] - a_xy[1])
        elevations = self.elevation.sample(xs, ys)

        d_elev = np.abs(np.diff(elevations))
        d_horiz = np.full(n_steps, length / n_steps)
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = np.where(d_horiz > 0, d_elev / d_horiz, 0.0)
        softramp = (grad / self.cfg.slope_soft_cap) ** self.cfg.slope_exponent
        return float(np.sum(d_horiz * softramp))

    def edge_cost(self, a_xy: XY, b_xy: XY) -> float:
        key = (a_xy, b_xy) if a_xy <= b_xy else (b_xy, a_xy)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached

        length = math.dist(a_xy, b_xy)
        cost = self.cfg.base_length_weight * length
        cost += self.cfg.slope_weight * self._slope_penalty(a_xy, b_xy, length)

        water_len = self.water.length_in_water(a_xy, b_xy)
        if water_len > 0:
            if self.cfg.water_hard_block:
                self._edge_cache[key] = math.inf
                return math.inf
            cost += self.cfg.water_surcharge * water_len

        self._edge_cache[key] = cost
        return cost

    def network_cost(
        self, edges: Iterable[tuple[XY, XY]], junctions: Iterable[XY] = ()
    ) -> float:
        total = sum(self.edge_cost(a, b) for a, b in edges)
        n_junctions = len(set(junctions))
        total += self.cfg.junction_cost * n_junctions
        return total

    def precompute_city_matrix(self, city_coords: list[XY]) -> dict[tuple[int, int], float]:
        """Precomputes the full city<->city cost matrix once. Junction edges are
        deliberately NOT cached here since junction positions move during search.
        """
        matrix: dict[tuple[int, int], float] = {}
        for i in range(len(city_coords)):
            for j in range(i + 1, len(city_coords)):
                c = self.edge_cost(city_coords[i], city_coords[j])
                matrix[(i, j)] = c
                matrix[(j, i)] = c
        return matrix
