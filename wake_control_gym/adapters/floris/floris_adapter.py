from typing import Any

import floris
import floris.layout_visualization
import numpy as np
import torch

from wake_control_gym.core import Action, MeasurementPoint, TurbineLayout
from wake_control_gym.adapters.floris.visualisation import FlorisCutPlane


def compute_turbine_power_range(
    fmodel: floris.FlorisModel,
    time_delta_seconds: int,
) -> tuple[float, float]:
    # Assume homogeneous turbines
    turbine_type = fmodel.core.farm.turbine_type_map.flatten()[0]
    power_table = np.array(fmodel.core.farm.turbine_power_thrust_tables[turbine_type]['power'])
    # Convert from kW to MWh
    power_table = power_table * 1e-3 * time_delta_seconds / 3600
    # Min is 0 because floris uses 0 as fill_value when interpolating the power.
    return 0, power_table.max()


class FlorisAdapter:
    # Floris
    fmodel: floris.FlorisModel
    floris_config: str | dict[str, Any]
    _turbine_power: torch.Tensor  # (num_turbines,)
    _u: torch.Tensor  # (len(measurement_points),)
    _v: torch.Tensor  # (len(measurement_points),)
    _farm_ti: torch.Tensor  # (1,)
    _input_wind_direction: float

    # Options
    valid_yaw_range: tuple[float, float]
    time_delta_seconds: int
    max_angular_velocity: float  # Degrees per second
    visualisation_resolution: tuple[int, int]

    # Internals
    num_turbines: int
    layout: TurbineLayout
    max_delta_yaw: float
    measurement_point_lists: tuple[list[float], list[float], list[float]]
    # In MWh
    turbine_power_range: tuple[float, float]
    turbine_power_factor: float
    visualisation: FlorisCutPlane | None = None

    # Constants / metadata
    min_windspeed: float = 0
    max_windspeed: float = 20

    # PyTorch
    device: str | torch.device | int
    dtype: torch.dtype

    def __init__(
        self,
        layout: TurbineLayout,
        seed: int,
        device: str | torch.device | int,
        dtype: torch.dtype,
        *,
        floris_config: str | dict[str, Any],
        valid_yaw_range: tuple[float, float] = (-30, 30),
        time_delta_seconds: int = 1,
        max_angular_velocity: float = 1,
        visualisation_resolution: tuple[int, int] = (200, 100),
    ) -> None:
        self.floris_config = floris_config
        self.layout = layout
        self.device = device
        self.dtype = dtype
        self.valid_yaw_range = valid_yaw_range
        self.time_delta_seconds = time_delta_seconds
        self.max_angular_velocity = max_angular_velocity
        self.max_delta_yaw = max_angular_velocity * time_delta_seconds
        self.visualisation_resolution = visualisation_resolution

        self.num_turbines = len(layout.x)
        self.measurement_point_lists = ([], [], [])

        # TODO: Wind process.
        self.fmodel = floris.FlorisModel(floris_config)
        self.fmodel.set(layout_x=layout.x, layout_y=layout.y)
        self.fmodel.run()
        self.turbine_power_range = compute_turbine_power_range(self.fmodel, time_delta_seconds)
        # Convert from W to MWh
        self.turbine_power_factor = 1e-6 * time_delta_seconds / 3600

    def reset(self, seed: int, options: dict[str, Any] | None = None) -> None:
        if options is None:
            options = {}

        self.fmodel = floris.FlorisModel(self.floris_config)
        self.fmodel.set(layout_x=self.layout.x, layout_y=self.layout.y, **options)
        self._run_measurements()

    def step(
        self,
        action: Action,
    ) -> None:
        """
        Perform a simulation step with the given yaw angles. Clipping the
        yaw angles to the according to the maximum angular velocity
        and yaw boundaries.

        Parameters
        ----------
        yaw_angles : torch.Tensor | None, optional
            A tensor of shape (num_turbines,) or with additional batch
            dimensions (..., num_turbines), representing the yaw angles
            for each turbine in the layout, by default None
        pitch_angles : torch.Tensor | None, optional
            A tensor of shape (num_turbines,) or with additional batch
            dimensions (..., num_turbines), representing the pitch angles
            for each turbine in the layout, by default None
        """
        if action.yaw_angles is not None:
            yaw_angles = action.yaw_angles.detach().cpu().numpy()
            min_yaws = self.fmodel.core.farm.yaw_angles - self.max_delta_yaw
            max_yaws = self.fmodel.core.farm.yaw_angles + self.max_delta_yaw
            yaw_angles = np.clip(yaw_angles, min_yaws, max_yaws)
            yaw_angles = np.clip(yaw_angles, *self.valid_yaw_range)
            # TODO / Profiling: fmodel.set is verrrrry expensive.
            self.fmodel.set(yaw_angles=yaw_angles)
        if action.pitch_angles is not None:
            raise NotImplementedError('Pitch angle control-action not implemented')
            # pitch_angles = action.pitch_angles.detach().cpu().numpy()

        self._run_measurements()

    def _run_measurements(self) -> None:
        # Must run model to evaluate turbine power
        self.fmodel.run()
        self._turbine_power = torch.tensor(
            self.fmodel.get_turbine_powers() * self.turbine_power_factor,
            device=self.device,
            dtype=self.dtype,
        )

        if len(self.measurement_point_lists[0]) == 0:
            return

        # Sample all the wind measurements
        # TODO / Profiling: This function is expensive
        self.fmodel.sample_flow_at_points(*self.measurement_point_lists)
        # [..., 0, 0] to remove turbine grid points (see sample_flow_at_points)
        u = self.fmodel.core.flow_field.u_sorted[0, :, 0, 0]
        v = self.fmodel.core.flow_field.v_sorted[0, :, 0, 0]
        ti = self.fmodel.turbulence_intensities.mean()
        self._u = torch.tensor(u, device=self.device, dtype=self.dtype)
        self._v = torch.tensor(v, device=self.device, dtype=self.dtype)
        self._farm_ti = torch.tensor([ti], device=self.device, dtype=self.dtype)
        # Should only be one wind direction
        self._input_wind_direction = self.fmodel.core.flow_field.wind_directions[0]

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
        hub_height: float = self.fmodel.core.farm.hub_heights.flatten()[0]
        points: list[MeasurementPoint] = list(zip(*self.measurement_point_lists))
        point_dict: dict[MeasurementPoint, int] = dict(zip(points, range(len(points))))
        point_indices: list[int] = []

        for point in measurement_points:
            if len(point) == 2:
                point = (point[0], point[1], hub_height)
            # Deduplicate
            if point not in point_dict:
                point_dict[point] = len(point_dict)
            point_indices.append(point_dict[point])

        points = list(point_dict.keys())
        self.measurement_point_lists = tuple([list(xyz) for xyz in zip(*points)])
        return point_indices

    def farm_turbulence_intensity(self) -> torch.Tensor:
        """Evaluate the farm turbulence intensity.

        Returns
        -------
        torch.Tensor
            A tensor of shape (1,)
        """
        return self._farm_ti

    def turbine_power(self) -> torch.Tensor:
        """Evaluate the instantaneous power at each turbine.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_turbines,)
        """
        return self._turbine_power

    def yaw_angles(self) -> torch.Tensor:
        """Get the turbine yaw angles.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_turbines,)
        """
        return torch.tensor(
            self.fmodel.core.farm.yaw_angles,
            device=self.device,
            dtype=self.dtype,
        )

    def wind_speed(self) -> torch.Tensor:
        """Get the wind speed at the simulator's measurement points.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_points,)
        """
        return torch.sqrt(self._u**2 + self._v**2)

    def wind_direction(self) -> torch.Tensor:
        """Get the wind direction at the simulator's measurement points.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_points,)
        """
        return (torch.rad2deg(torch.arctan2(self._v, self._u)) + self._input_wind_direction) % 360

    def close(self) -> None:
        """Dispose of the simulation environment,
        perform any deconstruction here."""
        pass

    def render(self) -> np.ndarray:
        """Render the environment to an rgb frame.

        Returns
        -------
        np.ndarray
            A numpy array of shape (height, width, 3) containing the frame data.
        """
        if self.visualisation is None:
            self.visualisation = FlorisCutPlane(self.visualisation_resolution)

        return self.visualisation.as_rgb(self.fmodel)
