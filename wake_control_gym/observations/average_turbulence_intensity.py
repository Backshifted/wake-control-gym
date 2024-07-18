import torch

from wake_control_gym.core import ObservationType, Simulator

MIN_TURBULENCE = 0
MAX_TURBULENCE = 2


class AverageTurbulenceIntensity:
    obs_type: ObservationType = 'global'
    dim: int = 1
    low: torch.Tensor
    high: torch.Tensor

    def __init__(self, simulator: Simulator) -> None:
        self.low = torch.tensor([MIN_TURBULENCE], device=simulator.device, dtype=simulator.dtype)
        self.high = torch.tensor([MAX_TURBULENCE], device=simulator.device, dtype=simulator.dtype)

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.farm_turbulence_intensity()
