import torch

from wake_control_gym.core import ObservationType, Simulator

MIN_WINDSPEED = 0
MAX_WINDSPEED = 20


class FarmWindSpeeds:
    obs_type: ObservationType = 'global'
    dim: int
    low: torch.Tensor
    high: torch.Tensor

    def __init__(self, simulator: Simulator) -> None:
        self.dim = simulator.num_turbines
        self.low = torch.full(
            (simulator.num_turbines, 1),
            MIN_WINDSPEED,
            device=simulator.device,
            dtype=simulator.dtype,
        )
        self.high = torch.full(
            (simulator.num_turbines, 1),
            MAX_WINDSPEED,
            device=simulator.device,
            dtype=simulator.dtype,
        )

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.wind_speed(simulator.layout)
