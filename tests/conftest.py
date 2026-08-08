import numpy as np
import pytest

from swiss_rail_rl.config import Config, CostConfig, PointSpec, PointsConfig, SteinerConfig
from swiss_rail_rl.cost import CostOracle
from swiss_rail_rl.data.elevation import ElevationGrid
from swiss_rail_rl.data.water import WaterMask


@pytest.fixture
def flat_elevation() -> ElevationGrid:
    array = np.zeros((50, 50), dtype=np.float32)
    return ElevationGrid(array=array, xmin=0.0, ymax=50_000.0, res=1000.0, crs="EPSG:2056")


@pytest.fixture
def ramp_elevation() -> ElevationGrid:
    """Elevation rising steeply along x — cheap edges near x=0, expensive near x=max."""
    array = np.zeros((50, 50), dtype=np.float32)
    for col in range(50):
        array[:, col] = col * 200.0  # 200m per km => 20% grade, well past the 6% soft cap
    return ElevationGrid(array=array, xmin=0.0, ymax=50_000.0, res=1000.0, crs="EPSG:2056")


@pytest.fixture
def no_water() -> WaterMask:
    return WaterMask(enabled=False, polygons=None, source="disabled")


@pytest.fixture
def default_cost_config() -> CostConfig:
    return CostConfig()


@pytest.fixture
def flat_oracle(default_cost_config, flat_elevation, no_water) -> CostOracle:
    return CostOracle(default_cost_config, flat_elevation, no_water)


@pytest.fixture
def small_config() -> Config:
    return Config(
        points=PointsConfig(
            anchors=[
                PointSpec(name="A", lon=6.0, lat=46.0),
                PointSpec(name="B", lon=7.0, lat=46.5),
            ],
            intermediate=[
                PointSpec(name="C", lon=6.5, lat=46.2),
            ],
        ),
        steiner=SteinerConfig(max_junctions=2, candidate_resolution_km=10.0),
    )
