import gymnasium as gym
import torch

from wake_control_gym.core import Simulator

DEFAULT_STEP_SIZES = [-1, 0, 1]


class DiscreteStepActions:
    space: gym.spaces.MultiDiscrete
    valid_yaw_range: tuple[float, float]
    step_sizes: torch.Tensor

    def __init__(self, simulator: Simulator, step_sizes: list[float] | None = None) -> None:
        if step_sizes is None:
            step_sizes = DEFAULT_STEP_SIZES

        self.space = gym.spaces.MultiDiscrete([len(step_sizes)] * simulator.num_turbines)
        self.step_sizes = torch.tensor(step_sizes, device=simulator.device, dtype=simulator.dtype)

    def __call__(self, action: torch.Tensor, simulator: Simulator) -> torch.Tensor:
        yaw_angles = simulator.yaw_angles()
        delta_yaws = torch.take(self.step_sizes, action)
        return yaw_angles + delta_yaws
