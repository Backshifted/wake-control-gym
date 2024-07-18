from typing import Any, Literal, NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import torch


ObservationType = Literal['global', 'local']


class TurbineLayout(NamedTuple):
    x: list[float]
    y: list[float]


class Simulator(Protocol):
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


class RewardFunc(Protocol):
    reward_range: torch.Tensor

    def __call__(self, simulator: Simulator) -> torch.Tensor:
        """Compute the reward for the current state of the environment.

        Parameters
        ----------
        simulator : Simulator
            The simulator for the environment.
        """
        ...


class Observation(Protocol):
    obs_type: ObservationType
    dim: int
    low: torch.Tensor
    high: torch.Tensor

    # Set externally by the gym environment.
    metadata: dict[str, Any] = {'index': 0}

    def __init__(self, simulator: Simulator) -> None:
        """Setup the `dim`, `low`, and `high` variables using the simulator.
        The simulator reference should not be stored.

        Parameters
        ----------
        simulator : Simulator
            The simulator for the environment.
        """
        ...

    def __call__(self, simulator: Simulator) -> torch.Tensor:
        """Generate an observation from the simulator.

        Parameters
        ----------
        simulator : Simulator
            The simulator for the environment.

        Returns
        -------
        torch.Tensor
            A tensor of shape (obs_dim,) or (num_turbines, obs_dim)
        """
        ...


class ActionRepresentation(Protocol):
    space: gym.Space

    def __init__(self, simulator: Simulator) -> None:
        """Setup the `space` variable using the simulator.
        The simulator reference should not be stored.

        Parameters
        ----------
        simulator : Simulator
            The simulator for the environment.
        """
        ...

    def __call__(self, action: torch.Tensor, simulator: Simulator) -> torch.Tensor:
        """Convert an action to the a set of yaw-misalignment angles.

        Parameters
        ----------
        action : torch.Tensor
            An action from the action representation space.
        simulator : Simulator
            The simulator for the environment.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_turbines,) representing the
            yaw-misalignment angles.
        """
        ...


class SimulatorInitFunc(Protocol):
    def __call__(
        self,
        layout: TurbineLayout,
        device: str | torch.device | int,
        dtype: torch.dtype,
    ) -> Simulator:
        """Prepare the simulation environment for execution.

        Parameters
        ----------
        layout : TurbineLayout
            The layout of the turbines to be used in the simulation.
        """
        ...


class NewObservationFunc(Protocol):
    def __call__(self, simulator: Simulator) -> Observation: ...
class NewRewardFuncFunc(Protocol):
    def __call__(self, simulator: Simulator) -> RewardFunc: ...
class NewActionRepresentationFunc(Protocol):
    def __call__(self, simulator: Simulator) -> ActionRepresentation: ...
