from typing import Any

import numpy as np
import torch

from wake_control_gym.core import TurbineLayout


class FlorisAdapter:
    # Misc
    num_turbines: int
    layout: TurbineLayout

    # Metadata
    can_render_human: bool
    valid_yaw_range: tuple[float, float]
    turbine_power_range: tuple[float, float]

    # PyTorch Misc
    device: str | torch.device | int
    dtype: torch.dtype

    def __init__(
        self,
        layout: TurbineLayout,
        device: str | torch.device | int,
        dtype: torch.dtype,
        *,
        floris_config: str | dict[str, Any],
    ) -> None:
        self.layout = layout
        self.device = device
        self.dtype = dtype

    def reset(self, seed: int, **kwargs) -> None:
        """Reset the simulator with a new seed."""
        ...

    def step(self, yaw_angles: torch.Tensor):
        """
        Perform a simulation step with the given yaw angles.

        Parameters
        ----------
        yaw_angles : torch.Tensor
            A tensor of shape (num_turbines,) or with additional batch
            dimensions (..., num_turbines), representing the yaw angles
            for each turbine in the layout.
        """
        ...

    def farm_turbulence_intensity(self) -> torch.Tensor:
        """Evaluate the farm average turbulence intensity.

        Returns
        -------
        torch.Tensor
            A tensor of shape (1,)
        """
        ...

    def turbine_power(self) -> torch.Tensor:
        """Evaluate the instantaneous power at each turbine.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_turbines,)
        """
        ...

    def yaw_angles(self) -> torch.Tensor:
        """Get the turbine yaw angles.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_turbines,)
        """
        ...

    def wind_speed(self, locations: tuple[list[float], list[float]]) -> torch.Tensor:
        """Get the wind speed measurement at the given locations.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_points,)
        """
        ...

    def wind_direction(self, locations: tuple[list[float], list[float]]) -> torch.Tensor:
        """Get the wind direction measurement in degrees at the given locations.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_points,)
        """
        ...

    def close(self) -> None:
        """Dispose of the simulation environment,
        perform any deconstruction here."""
        ...

    def render(self) -> np.ndarray:
        """Render the environment to an rgb frame.

        Returns
        -------
        np.ndarray
            A numpy array of shape (height, width, 3) containing the frame data.
        """
        ...

    def render_human(self) -> None:
        """Displays the environment, using some external method."""
        ...
