from typing import Any
import torch
from wake_control_gym.core import ObservationType, Simulator


MIN_DIRECTION = 0
MAX_DIRECTION = 3600


class FarmWindDirections:
    obs_type: ObservationType = 'global'
    dim: int
    low: torch.Tensor
    high: torch.Tensor

    metadata: dict[str, Any] = {}

    measurement_point_indices: list[int]

    def __init__(self, simulator: Simulator) -> None:
        self.dim = simulator.num_turbines
        self.low = torch.full(
            (simulator.num_turbines,),
            MIN_DIRECTION,
            device=simulator.device,
            dtype=simulator.dtype,
        )
        self.high = torch.full(
            (simulator.num_turbines,),
            MAX_DIRECTION,
            device=simulator.device,
            dtype=simulator.dtype,
        )
        points = list(zip(*simulator.layout))
        self.measurement_point_indices = simulator.add_measurement_points(points)

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.wind_direction()[self.measurement_point_indices]
