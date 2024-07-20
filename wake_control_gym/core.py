from typing import Any, Literal, NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import torch


ObservationType = Literal['global', 'local']
MeasurementPoint = tuple[float, float] | tuple[float, float, float]


class TurbineLayout(NamedTuple):
    x: list[float]
    y: list[float]


class Action(NamedTuple):
    yaw_angles: torch.Tensor | None = None
    pitch_angles: torch.Tensor | None = None


class Simulator(Protocol):
    # Misc
    num_turbines: int
    layout: TurbineLayout

    # Metadata
    valid_yaw_range: tuple[float, float]
    turbine_power_range: tuple[float, float]

    # PyTorch Misc
    device: str | torch.device | int
    dtype: torch.dtype

    def reset(self, seed: int, options: dict[str, Any] | None = None) -> None:
        """Reset the simulator with a new seed."""
        ...

    def step(self, action: Action) -> None:
        """
        Perform a simulation step with the given yaw angles and/or pitch angles
        in the action.
        """
        ...

    def add_measurement_points(self, measurement_points: list[MeasurementPoint]) -> list[int]:
        """Add measurement points for the flow field. If only x and y are
        given, the hub height is used for the z-value.

        Parameters
        ----------
        measurement_points : list[MeasurementPoint]
            The list of measurement points to add.

        Returns
        -------
        list[int]
            A list of indices, where each index represents the position
            of the original measurment point to the simulator's
            set of measurement points.
        """
        ...

    def farm_turbulence_intensity(self) -> torch.Tensor:
        """Evaluate the farm turbulence intensity.

        Returns
        -------
        torch.Tensor
            A tensor of shape (1,)
        """
        ...

    def turbine_power(self) -> torch.Tensor:
        """Evaluate the power generated at each turbine in MWh.

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

    def wind_speed(self) -> torch.Tensor:
        """Get the wind speed at the simulator's measurement points.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_points,)
        """
        ...

    def wind_direction(self) -> torch.Tensor:
        """Get the wind direction at the simulator's measurement points.

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
    # Observation types: Local, global, turbine, met mast ?
    # Local / global determine how
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

    def __call__(self, action: torch.Tensor, simulator: Simulator) -> Action:
        """Convert an action to the a set of yaw-misalignment angles.

        Parameters
        ----------
        action : torch.Tensor
            An action from the action representation space.
        simulator : Simulator
            The simulator for the environment.

        Returns
        -------
        Action
            Tensors of shape (num_turbines,) representing the
            yaw-misalignment angles and/or pitch angles.
        """
        ...


class SimulatorInitFunc(Protocol):
    def __call__(
        self,
        layout: TurbineLayout,
        seed: int,
        device: str | torch.device | int,
        dtype: torch.dtype,
    ) -> Simulator:
        """Prepare the simulation environment for execution.

        Parameters
        ----------
        layout : TurbineLayout
            The layout of the turbines to be used in the simulation.
        seed : int
            The seed for the pseudo-random number generator of the simulator.
        """
        ...


class NewObservationFunc(Protocol):
    def __call__(self, simulator: Simulator) -> Observation: ...
class NewRewardFuncFunc(Protocol):
    def __call__(self, simulator: Simulator) -> RewardFunc: ...
class NewActionRepresentationFunc(Protocol):
    def __call__(self, simulator: Simulator) -> ActionRepresentation: ...
