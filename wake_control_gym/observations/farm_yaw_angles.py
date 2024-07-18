import torch

from wake_control_gym.core import ObservationType, Simulator


class FarmYawAngles:
    obs_type: ObservationType = 'global'
    dim: int
    low: torch.Tensor
    high: torch.Tensor

    def __init__(self, simulator: Simulator) -> None:
        self.dim = simulator.num_turbines
        self.low = torch.full(
            (simulator.num_turbines,),
            simulator.valid_yaw_range[0],
            device=simulator.device,
            dtype=simulator.dtype,
        )
        self.high = torch.full(
            (simulator.num_turbines,),
            simulator.valid_yaw_range[1],
            device=simulator.device,
            dtype=simulator.dtype,
        )

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.yaw_angles()
