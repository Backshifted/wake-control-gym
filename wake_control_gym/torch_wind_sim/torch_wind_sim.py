import torch

from wake_control_gym.core import TurbineLayout


class TorchWindSim:
    def __init__(
        self,
        layout: TurbineLayout,
        device: str | torch.device | int,
        dtype: torch.dtype,
    ) -> None:
        pass
