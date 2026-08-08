import networkx as nx

from swiss_rail_rl.baselines import mst_baseline, steiner_heuristic
from swiss_rail_rl.data.cities import bounding_box, project_points


def test_mst_connects_all_cities(small_config, flat_oracle):
    points = project_points(small_config)
    city_coords = [(p.x, p.y) for p in points]
    net = mst_baseline(flat_oracle, city_coords)

    g = nx.Graph()
    g.add_edges_from(net.edges)
    assert nx.is_connected(g)
    assert set(g.nodes) == set(city_coords)


def test_steiner_heuristic_never_worse_than_mst(small_config, flat_oracle):
    points = project_points(small_config)
    city_coords = [(p.x, p.y) for p in points]
    bbox = bounding_box(points)

    mst = mst_baseline(flat_oracle, city_coords)
    steiner = steiner_heuristic(flat_oracle, city_coords, small_config.steiner, bbox)

    assert steiner.cost <= mst.cost + 1e-6
