import torch

from wake_control_gym.core import Simulator


class TurbinePowerReward:
    reward_range: tuple[torch.Tensor, torch.Tensor]

    def __init__(self, simulator: Simulator) -> None:
        self.reward_range = (
            torch.full(
                (simulator.num_turbines,),
                simulator.turbine_power_range[0],
                device=simulator.device,
                dtype=simulator.dtype,
            ),
            torch.full(
                (simulator.num_turbines,),
                simulator.turbine_power_range[1],
                device=simulator.device,
                dtype=simulator.dtype,
            ),
        )

    def __call__(self, simulator: Simulator) -> torch.Tensor:
        return simulator.turbine_power()
