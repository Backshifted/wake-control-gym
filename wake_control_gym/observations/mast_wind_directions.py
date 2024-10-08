from typing import Any
import torch
from wake_control_gym.core import MeasurementPoint, Observation, ObservationType, Simulator


MIN_DIRECTION = 0
MAX_DIRECTION = 360


class MastWindDirections(Observation):
    obs_type: ObservationType = 'global'
    measurement_point_indices: list[Any]

    def __init__(self, simulator: Simulator, measurement_points: list[MeasurementPoint]) -> None:
        self.dim = len(measurement_points)
        self.low = torch.full(
            (self.dim,), MIN_DIRECTION, device=simulator.device, dtype=simulator.dtype
        )
        self.high = torch.full(
            (self.dim,), MAX_DIRECTION, device=simulator.device, dtype=simulator.dtype
        )
        self.measurement_point_indices = simulator.register_measurement_points(measurement_points)

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.wind_direction()[self.measurement_point_indices]
