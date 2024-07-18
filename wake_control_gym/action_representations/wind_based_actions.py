import gymnasium as gym
import numpy as np
import torch

from wake_control_gym.core import Simulator


class WindBasedActions:
    space: gym.spaces.Box
    max_yaw_angle: float

    def __init__(self, simulator: Simulator) -> None:
        self.max_yaw_angle = simulator.valid_yaw_range[1]
        self.space = gym.spaces.Box(
            low=np.full((simulator.num_turbines,), -1),
            high=np.full((simulator.num_turbines,), 1),
        )

    def __call__(self, action: torch.Tensor, simulator: Simulator) -> torch.Tensor:
        return action * self.max_yaw_angle
