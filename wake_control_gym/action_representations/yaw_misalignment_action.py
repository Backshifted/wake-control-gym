import gymnasium as gym
import numpy as np
import torch

from wake_control_gym.core import Action, ActionRepresentation, Simulator


class YawMisalignmentAction(ActionRepresentation):
    """Convert [-1, 1]^n to [min yaw, max yaw]^n."""

    space: gym.spaces.Box
    max_yaw_angle: float

    def __init__(self, simulator: Simulator) -> None:
        self.max_yaw_angle = np.abs(simulator.valid_yaw_range).max()
        self.space = gym.spaces.Box(
            low=np.full((simulator.num_turbines,), -1),
            high=np.full((simulator.num_turbines,), 1),
        )

    def __call__(self, action: torch.Tensor, simulator: Simulator) -> Action:
        return Action(yaw_angles=action * self.max_yaw_angle)
