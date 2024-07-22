import torch
from wake_control_gym.core import Observation, ObservationType, Simulator


class TurbineYawAngle(Observation):
    obs_type: ObservationType = 'local'
    dim: int = 1

    def __init__(self, simulator: Simulator) -> None:
        min_yaw, max_yaw = simulator.valid_yaw_range
        self.low = torch.tensor([min_yaw], device=simulator.device, dtype=simulator.dtype)
        self.high = torch.tensor([max_yaw], device=simulator.device, dtype=simulator.dtype)

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.yaw_angles().view(-1, self.dim)
