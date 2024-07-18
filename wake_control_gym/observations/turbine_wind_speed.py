import torch

from wake_control_gym.core import ObservationType, Simulator

MIN_WINDSPEED = 0
MAX_WINDSPEED = 20


class TurbineWindSpeed:
    obs_type: ObservationType = 'local'
    dim: int = 1
    low: torch.Tensor
    high: torch.Tensor

    def __init__(self, simulator: Simulator) -> None:
        self.low = torch.tensor([MIN_WINDSPEED], device=simulator.device, dtype=simulator.dtype)
        self.high = torch.tensor([MAX_WINDSPEED], device=simulator.device, dtype=simulator.dtype)

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.wind_speed(simulator.layout).view(-1, self.dim)
