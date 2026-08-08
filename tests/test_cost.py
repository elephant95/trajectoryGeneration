import math

from swiss_rail_rl.cost import CostOracle


def test_flat_edge_cost_is_length_only(flat_oracle):
    a, b = (0.0, 0.0), (10_000.0, 0.0)
    cost = flat_oracle.edge_cost(a, b)
    assert math.isclose(cost, 10_000.0, rel_tol=1e-6)


def test_steep_edge_costs_more_than_flat(default_cost_config, ramp_elevation, no_water):
    oracle = CostOracle(default_cost_config, ramp_elevation, no_water)
    flat_edge = oracle.edge_cost((0.0, 0.0), (0.0, 5_000.0))  # constant x -> flat
    steep_edge = oracle.edge_cost((0.0, 0.0), (30_000.0, 0.0))  # crosses the steep ramp
    assert steep_edge > flat_edge


def test_water_hard_block_returns_inf(default_cost_config, flat_elevation):
    from shapely.geometry import MultiPolygon, Point

    from swiss_rail_rl.data.water import WaterMask

    cfg = default_cost_config.model_copy(update={"water_hard_block": True})
    lake = MultiPolygon([Point(5_000, 5_000).buffer(2_000)])
    water = WaterMask(enabled=True, polygons=lake, source="test")
    oracle = CostOracle(cfg, flat_elevation, water)
    cost = oracle.edge_cost((0.0, 5_000.0), (10_000.0, 5_000.0))
    assert math.isinf(cost)


def test_network_cost_includes_junction_cost(flat_oracle):
    edges = [((0.0, 0.0), (5_000.0, 0.0)), ((5_000.0, 0.0), (10_000.0, 0.0))]
    junctions = [(5_000.0, 0.0)]
    cost = flat_oracle.network_cost(edges, junctions)
    expected = 5_000.0 + 5_000.0 + flat_oracle.cfg.junction_cost
    assert math.isclose(cost, expected, rel_tol=1e-6)
