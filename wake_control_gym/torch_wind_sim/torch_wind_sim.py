import torch

from wake_control_gym.core import Simulator, TurbineLayout


class TorchWindSim(Simulator):
    def __init__(
        self,
        layout: TurbineLayout,
        device: str | torch.device | int,
        dtype: torch.dtype,
    ) -> None:
        pass
