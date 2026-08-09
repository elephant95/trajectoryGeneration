"""PPO training with per-checkpoint frame logging (§7 of the build spec).

Uses sb3-contrib's MaskablePPO rather than plain PPO. The discrete action
space (§6 Design A) is mostly *invalid* at any given state — most edges
already exist or reference an unplaced junction slot — so unmasked PPO
spends most of its exploration budget on invalid-action penalties instead
of learning useful edges. Env.action_masks() (env.py) is the single source
of truth for legality; MaskablePPO restricts sampling to legal actions only,
which is a large sample-efficiency win for this kind of construction task
and is a drop-in swap (`rl.algorithm` stays "PPO" conceptually — this is a
masked variant of it, not a different algorithm).

Every `rl.checkpoint_every` steps we run one greedy (deterministic) rollout
with the current policy, snapshot `env.current_network()`, and append it to
`out/frames/frame_XXXXX.json`. That frame stream is exactly what render.py
animates. A scalar `training_log.csv` (step, best_cost, connected,
n_junctions) drives the on-screen cost curve. Model checkpoints are saved
after every logging step so training is resumable.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from swiss_rail_rl.baselines import mst_baseline, steiner_heuristic
from swiss_rail_rl.config import Config
from swiss_rail_rl.cost import CostOracle
from swiss_rail_rl.data.cities import bounding_box
from swiss_rail_rl.env import SwissRailEnv

logger = logging.getLogger(__name__)

XY = tuple[float, float]


def _make_env(config: Config, cost_oracle: CostOracle, city_coords: list[XY]) -> SwissRailEnv:
    return SwissRailEnv(config, cost_oracle, city_coords)


def greedy_rollout(model, env: SwissRailEnv, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    n_steps = 0
    while not done and n_steps < env.max_steps + 1:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        done = terminated or truncated
        n_steps += 1
    return env.current_network()


def log_baselines(config: Config, cost_oracle: CostOracle, city_coords: list[XY], out_dir: Path) -> dict:
    """Classical reference solvers, logged before spending RL compute (§6):
    the RL agent should approach or beat these costs.
    """
    bbox = bounding_box_from_xy(city_coords)
    mst = mst_baseline(cost_oracle, city_coords)
    steiner = steiner_heuristic(cost_oracle, city_coords, config.steiner, bbox)
    result = {
        "mst_cost": mst.cost,
        "steiner_cost": steiner.cost,
        "steiner_n_junctions": len(steiner.junctions),
    }
    logger.info(
        "Baselines — MST cost: %.1f | Steiner heuristic cost: %.1f (%d junctions). "
        "RL should approach or beat the Steiner cost; a big gap signals a reward/action-space bug.",
        result["mst_cost"],
        result["steiner_cost"],
        result["steiner_n_junctions"],
    )
    with open(out_dir / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def bounding_box_from_xy(city_coords: list[XY]):
    class _P:
        def __init__(self, x, y):
            self.x, self.y = x, y

    return bounding_box([_P(x, y) for x, y in city_coords])


def train(config: Config, cost_oracle: CostOracle, city_coords: list[XY], resume: bool = True) -> Path:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    out_dir = config.out_path()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_log.csv"
    model_path = out_dir / "maskable_ppo_model.zip"
    vecnorm_path = out_dir / "vecnormalize.pkl"

    baselines = log_baselines(config, cost_oracle, city_coords, out_dir)

    def _thunk():
        return Monitor(_make_env(config, cost_oracle, city_coords))

    raw_vec_env = DummyVecEnv([_thunk for _ in range(config.rl.n_envs)])
    # Raw episode returns here span orders of magnitude across configs/episodes
    # (millions to billions before reward_scale, and reward_scale alone can't
    # fix that within an episode). Unnormalized, the critic never gets a
    # learnable target — explained_variance stays pinned near 0 no matter how
    # long training runs. VecNormalize's running reward normalization fixes
    # that; obs are left alone since env.py already hand-normalizes them.
    if config.rl.normalize_reward and resume and vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), raw_vec_env)
        logger.info("Resumed VecNormalize stats from %s", vecnorm_path)
    elif config.rl.normalize_reward:
        vec_env = VecNormalize(raw_vec_env, norm_obs=False, norm_reward=True, gamma=config.rl.gamma)
    else:
        vec_env = raw_vec_env
    eval_env = _make_env(config, cost_oracle, city_coords)

    if resume and model_path.exists():
        model = MaskablePPO.load(model_path, env=vec_env)
        logger.info("Resumed MaskablePPO model from %s", model_path)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=config.rl.learning_rate,
            gamma=config.rl.gamma,
            ent_coef=config.rl.ent_coef,
            seed=config.rl.seed,
            verbose=1,
        )

    start_step = int(model.num_timesteps)
    n_remaining = max(0, config.rl.total_timesteps - start_step)
    n_checkpoints = max(1, n_remaining // config.rl.checkpoint_every)

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "best_cost", "connected", "n_junctions", "mst_baseline_cost", "steiner_baseline_cost"])

        existing_frames = sorted(frames_dir.glob("frame_*.json"))
        frame_idx = len(existing_frames)

        for _ in range(n_checkpoints):
            model.learn(total_timesteps=config.rl.checkpoint_every, reset_num_timesteps=False, progress_bar=False)

            network = greedy_rollout(model, eval_env, seed=config.rl.seed)
            frame_path = frames_dir / f"frame_{frame_idx:05d}.json"
            with open(frame_path, "w", encoding="utf-8") as ff:
                json.dump({"step": int(model.num_timesteps), **network}, ff)

            writer.writerow(
                [
                    int(model.num_timesteps),
                    network["total_cost"],
                    network["connected"],
                    len(network["junctions"]),
                    baselines["mst_cost"],
                    baselines["steiner_cost"],
                ]
            )
            f.flush()
            model.save(model_path)
            if isinstance(vec_env, VecNormalize):
                vec_env.save(str(vecnorm_path))
            logger.info(
                "step=%d cost=%.1f connected=%s n_junctions=%d",
                int(model.num_timesteps),
                network["total_cost"],
                network["connected"],
                len(network["junctions"]),
            )
            frame_idx += 1

    return model_path
