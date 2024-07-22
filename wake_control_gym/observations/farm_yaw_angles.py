import torch
from wake_control_gym.core import Observation, ObservationType, Simulator


class FarmYawAngles(Observation):
    obs_type: ObservationType = 'global'

    def __init__(self, simulator: Simulator) -> None:
        min_yaw, max_yaw = simulator.valid_yaw_range
        self.dim = simulator.num_turbines
        self.low = torch.full((self.dim,), min_yaw, device=simulator.device, dtype=simulator.dtype)
        self.high = torch.full((self.dim,), max_yaw, device=simulator.device, dtype=simulator.dtype)

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.yaw_angles()
