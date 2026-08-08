"""Loads and validates config.yaml. Every tunable knob in the project lives here —
downstream modules must read values through the typed Config object, never
hardcode a number that could plausibly belong in config.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class RegionConfig(BaseModel):
    country: str = "CH"
    crs_projected: str = "EPSG:2056"


class PointSpec(BaseModel):
    name: str
    lon: float
    lat: float


class PointsConfig(BaseModel):
    anchors: list[PointSpec]
    intermediate: list[PointSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_names(self) -> "PointsConfig":
        names = [p.name for p in self.anchors] + [p.name for p in self.intermediate]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate point names in config: {sorted(dupes)}")
        if len(self.anchors) < 2:
            raise ValueError("need at least 2 anchor cities to connect")
        return self


class CostConfig(BaseModel):
    base_length_weight: float = 1.0
    slope_weight: float = 50.0
    slope_exponent: float = 2.0
    slope_soft_cap: float = 0.06
    water_surcharge: float = 5.0
    water_hard_block: bool = False
    junction_cost: float = 20000.0
    sample_step_m: float = 250.0


class SteinerConfig(BaseModel):
    max_junctions: int = 8
    junction_grid: Literal["grid", "adaptive", "candidate_set"] = "adaptive"
    candidate_resolution_km: float = 15.0
    candidate_pool_cap: int = 40  # caps the RL discrete action space to a tractable size


class RLConfig(BaseModel):
    algorithm: Literal["PPO"] = "PPO"
    total_timesteps: int = 2_000_000
    seed: int = 0
    n_envs: int = 8
    checkpoint_every: int = 25_000
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    action_design: Literal["A", "B"] = "A"
    max_steps: int = 60
    reward_scale: float = 1.0e-5          # scales -cost into a learning-friendly range
    invalid_action_penalty: float = 1.0   # scaled-reward penalty for a masked/no-op action
    disconnected_penalty_weight: float = 50.0  # scaled-reward penalty per disconnected component at STOP


class FitConfig(BaseModel):
    enabled: bool = True
    method: Literal["cma", "grid", "random"] = "cma"
    free_params: list[str] = Field(default_factory=lambda: ["slope_weight"])
    objective: Literal["network_distance"] = "network_distance"
    max_evals: int = 60
    topology_weight: float = 2000.0     # weight of the connectivity-mismatch term vs geometry (metres)
    geometry_sample_step_m: float = 500.0
    snap_tolerance_m: float = 5000.0    # how close a city must be to a real rail vertex to count as "on" it
    param_search_range: tuple[float, float] = (0.05, 20.0)  # multiplicative bounds around each param's config value


class DataConfig(BaseModel):
    elevation_source: Literal["swissALTI3D", "SRTM"] = "swissALTI3D"
    rail_source: Literal["swisstlm3d", "osm_hdx"] = "swisstlm3d"
    cache_dir: str = "./data_cache"
    water_enabled: bool = False


class RenderConfig(BaseModel):
    fps: int = 30
    resolution: tuple[int, int] = (1080, 1920)
    backend: Literal["matplotlib", "manim"] = "matplotlib"
    show_elevation_hillshade: bool = True
    reveal_hold_seconds: float = 4.0
    output_path: str = "./out/swiss_rail_rl.mp4"


class Config(BaseModel):
    region: RegionConfig = RegionConfig()
    points: PointsConfig
    cost: CostConfig = CostConfig()
    steiner: SteinerConfig = SteinerConfig()
    rl: RLConfig = RLConfig()
    fit: FitConfig = FitConfig()
    data: DataConfig = DataConfig()
    render: RenderConfig = RenderConfig()

    captions_path: str = "./captions.yaml"
    out_dir: str = "./out"

    def cache_dir(self) -> Path:
        p = Path(self.data.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def out_path(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"config file not found: {path}. Every parameter must live in "
            "config.yaml — see the shipped example."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)


def load_captions(path: str | Path = "captions.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
