from typing import Any
import torch
from wake_control_gym.core import Observation, ObservationType, Simulator

MIN_DIRECTION = 0
MAX_DIRECTION = 360


class TurbineWindDirection(Observation):
    obs_type: ObservationType = 'local'
    dim: int = 1
    measurement_point_indices: list[Any]

    def __init__(self, simulator: Simulator) -> None:
        self.low = torch.tensor([MIN_DIRECTION], device=simulator.device, dtype=simulator.dtype)
        self.high = torch.tensor([MAX_DIRECTION], device=simulator.device, dtype=simulator.dtype)
        points = list(zip(*simulator.layout))
        self.measurement_point_indices = simulator.register_measurement_points(points)

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.wind_direction()[self.measurement_point_indices].view(-1, self.dim)
