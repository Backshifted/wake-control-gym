import torch

from wake_control_gym.core import Simulator


class FarmPowerReward:
    reward_range: tuple[torch.Tensor, torch.Tensor]

    def __init__(self, simulator: Simulator) -> None:
        self.reward_range = (
            simulator.turbine_power_range[0] * simulator.num_turbines,
            simulator.turbine_power_range[1] * simulator.num_turbines,
        )

    def __call__(self, simulator: Simulator) -> torch.Tensor:
        return simulator.turbine_power().sum()
