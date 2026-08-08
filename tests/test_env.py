import numpy as np

from swiss_rail_rl.data.cities import project_points
from swiss_rail_rl.env import SwissRailEnv


def test_env_step_and_reset(small_config, flat_oracle):
    points = project_points(small_config)
    city_coords = [(p.x, p.y) for p in points]
    env = SwissRailEnv(small_config, flat_oracle, city_coords)

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    rng = np.random.default_rng(0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < small_config.rl.max_steps + 1:
        action = int(rng.integers(0, env.n_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        steps += 1

    assert terminated or truncated
    network = env.current_network()
    assert len(network["cities"]) == len(city_coords)


def test_stop_action_terminates_immediately(small_config, flat_oracle):
    points = project_points(small_config)
    city_coords = [(p.x, p.y) for p in points]
    env = SwissRailEnv(small_config, flat_oracle, city_coords)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.STOP)
    assert terminated
    assert info["connected"] is False  # no edges added yet


def test_action_masks_matches_step_validity(small_config, flat_oracle):
    points = project_points(small_config)
    city_coords = [(p.x, p.y) for p in points]
    env = SwissRailEnv(small_config, flat_oracle, city_coords)
    env.reset(seed=0)

    mask = env.action_masks()
    assert mask.shape == (env.n_actions,)
    assert mask[env.STOP]  # STOP always legal
    assert mask.sum() > 1  # some edges should be addable from a fresh state

    # An edge just added must become illegal (already present) in the next mask.
    edge_action = next(a for a in range(env.n_candidates, env.STOP) if mask[a])
    env.step(edge_action)
    assert not env.action_masks()[edge_action]
