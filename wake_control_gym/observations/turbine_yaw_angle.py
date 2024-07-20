from typing import Any
import torch
from wake_control_gym.core import ObservationType, Simulator


class TurbineYawAngle:
    obs_type: ObservationType = 'local'
    dim: int = 1
    low: torch.Tensor
    high: torch.Tensor

    metadata: dict[str, Any] = {}

    def __init__(self, simulator: Simulator) -> None:
        self.low = torch.tensor(
            [simulator.valid_yaw_range[0]],
            device=simulator.device,
            dtype=simulator.dtype,
        )
        self.high = torch.tensor(
            [simulator.valid_yaw_range[1]],
            device=simulator.device,
            dtype=simulator.dtype,
        )

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.yaw_angles().view(-1, self.dim)
