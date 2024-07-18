import torch

from wake_control_gym.core import ObservationType, Simulator


MIN_DIRECTION = 0
MAX_DIRECTION = 3600


class FarmWindDirections:
    obs_type: ObservationType = 'global'
    dim: int
    low: torch.Tensor
    high: torch.Tensor

    def __init__(self, simulator: Simulator) -> None:
        self.dim = simulator.num_turbines
        self.low = torch.full(
            (simulator.num_turbines, 1),
            MIN_DIRECTION,
            device=simulator.device,
            dtype=simulator.dtype,
        )
        self.high = torch.full(
            (simulator.num_turbines, 1),
            MAX_DIRECTION,
            device=simulator.device,
            dtype=simulator.dtype,
        )

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.wind_direction(simulator.layout)
