from typing import Any, Sequence

import cv2
import gymnasium as gym
import numpy as np
import torch

from wake_control_gym.core import (
    ActionRepresentation,
    NewActionRepresentationFunc,
    NewObservationFunc,
    NewRewardFuncFunc,
    Observation,
    RewardFunc,
    Simulator,
    SimulatorInitFunc,
    TurbineLayout,
)
from wake_control_gym.observations import (
    FarmWindDirections,
    FarmWindSpeeds,
    FarmYawAngles,
    TurbulenceIntensity,
)
from wake_control_gym.action_representations import YawMisalignmentAction
from wake_control_gym.reward_functions import FarmPowerReward
from wake_control_gym.torch_wind_sim import TorchWindSim


def _create_observation_space(
    observations: list[NewObservationFunc],
    simulator: Simulator,
    is_multi_agent: bool = False,
    seed: int | None = None,
) -> tuple[gym.spaces.Box, torch.Tensor, torch.Tensor, list[Observation], list[Observation]]:
    """Reads lows and highs from the observations and constructs an Box space."""
    observations = [obs(simulator) for obs in observations]
    global_observations = [obs for obs in observations if obs.obs_type == 'global']
    local_observations = [obs for obs in observations if obs.obs_type == 'local']

    assert (
        global_observations or local_observations
    ), 'Observations required, single-state mdp not supported.'

    assert (
        is_multi_agent or not local_observations
    ), 'Cannot have local observations in single-agent environment.'

    index = 0

    for obs in global_observations:
        obs.metadata['index'] = index
        index += obs.dim
    for obs in local_observations:
        obs.metadata['index'] = index
        index += obs.dim

    low = torch.tensor([], device=simulator.device, dtype=simulator.dtype)
    high = torch.tensor([], device=simulator.device, dtype=simulator.dtype)

    if global_observations:
        global_lows = [obs.low for obs in global_observations]
        low = torch.cat([low, *global_lows])
        global_high = [obs.high for obs in global_observations]
        high = torch.cat([high, *global_high])

    if is_multi_agent:
        low = low.repeat(simulator.num_turbines, 1)
        high = high.repeat(simulator.num_turbines, 1)

    if local_observations:
        local_low = [obs.low.repeat(simulator.num_turbines, 1) for obs in local_observations]
        local_high = [obs.high.repeat(simulator.num_turbines, 1) for obs in local_observations]
        low = torch.cat([low, *local_low], dim=1)
        high = torch.cat([high, *local_high], dim=1)

    observation_space = gym.spaces.Box(low.cpu().numpy(), high.cpu().numpy(), seed=seed)
    return observation_space, low, high, global_observations, local_observations


def _create_observation(
    simulator: Simulator,
    global_observations: list[Observation],
    local_observations: list[Observation],
    is_multi_agent: bool,
) -> torch.Tensor:
    """Iterates over the observation functions and concatenates the result."""
    observation = torch.tensor([], device=simulator.device, dtype=simulator.dtype)

    if not global_observations and not local_observations:
        observation = torch.zeros((1,), device=simulator.device, dtype=simulator.dtype)

    if global_observations:
        global_obs = [obs(simulator) for obs in global_observations]
        observation = torch.cat([observation, *global_obs])

    if is_multi_agent:
        observation = observation.view(1, -1).repeat(simulator.num_turbines, 1)

    if local_observations:
        local_obs = [obs(simulator) for obs in local_observations]
        observation = torch.cat([observation, *local_obs], dim=1)

    return observation


# def _create_observation(
#     simulator: Simulator,
#     global_observations: list[Observation],
#     local_observations: list[Observation],
#     is_multi_agent: bool,
# ) -> torch.Tensor:
#     """Iterates over the observation functions and concatenates the result."""
#     obs = torch.cat([obs(simulator) for obs in global_observations])

#     if is_multi_agent:
#         obs = obs.repeat(simulator.num_turbines, 1)
#     if len(local_observations) == 0:
#         return obs

#     local_obs = torch.cat([obs(simulator) for obs in local_observations], dim=1)
#     return torch.cat([obs, local_obs], dim=1)


class WakeControlEnv(gym.Env[gym.spaces.Box, gym.spaces.Box]):
    """Must call reset before stepping."""

    metadata = {'render_modes': ['rgb_array', 'human']}

    # Internals
    simulator: Simulator
    local_observations: list[Observation]
    global_observations: list[Observation]
    observation_low: torch.Tensor
    observation_high: torch.Tensor
    cv2_window_name: str = 'Wake Control Env Viewer'
    _num_turbines: int
    _layout: TurbineLayout
    _seed: int
    _torch_random: torch.Generator

    # Options
    action_representation: ActionRepresentation
    reward_func: RewardFunc
    is_multi_agent: bool
    enable_info: bool

    def __init__(
        self,
        layout: TurbineLayout | tuple[list[float], list[float]],
        observations: Sequence[NewObservationFunc] = (
            TurbulenceIntensity,
            FarmWindSpeeds,
            FarmWindDirections,
            FarmYawAngles,
        ),
        init_simulator: SimulatorInitFunc = TorchWindSim,
        new_action_representation: NewActionRepresentationFunc = YawMisalignmentAction,
        new_reward_func: NewRewardFuncFunc = FarmPowerReward,
        is_multi_agent: bool = False,
        enable_info: bool = True,
        seed: int | None = None,
        device: str | torch.device | int = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # Options
        self.is_multi_agent = is_multi_agent
        self.enable_info = enable_info

        if seed is None:
            self._seed = torch.random.seed()
        else:
            self._seed = seed
        self._torch_random = torch.random.manual_seed(self._seed)

        if not isinstance(layout, TurbineLayout):
            self._layout = TurbineLayout(*layout)
        self._num_turbines = len(self.layout.x)
        self.simulator = init_simulator(self.layout, seed=seed, device=device, dtype=dtype)
        (
            self.observation_space,
            self.observation_low,
            self.observation_high,
            self.global_observations,
            self.local_observations,
        ) = _create_observation_space(observations, self.simulator, is_multi_agent, self._seed)
        self.action_representation = new_action_representation(self.simulator)
        self.action_space = self.action_representation.space
        self.reward_func = new_reward_func(self.simulator)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, Any]]:
        info = {}

        # Convert the action passed
        simulator_action = self.action_representation(action, self.simulator)
        self.simulator.step(simulator_action)
        observation = _create_observation(
            self.simulator,
            self.global_observations,
            self.local_observations,
            self.is_multi_agent,
        )
        reward = self.reward_func(self.simulator)
        info = self._create_simulator_info()

        if self.render_mode == 'human':
            frame = self.simulator.render()
            cv2.imshow(self.cv2_window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)

            if key == 27:
                self.render_mode = 'rgb_array'
                self._close_render_window()

        return observation, reward, False, False, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is None:
            self._seed = torch.random.seed()
        else:
            self._seed = seed

        self._torch_random = torch.random.manual_seed(self._seed)
        self._close_render_window()
        self.simulator.reset(seed=self._seed, options=options)
        info = self._create_simulator_info()
        info['seed'] = self._seed
        observation = _create_observation(
            self.simulator,
            self.global_observations,
            self.local_observations,
            self.is_multi_agent,
        )

        return observation, info

    def close(self):
        self._close_render_window()

    def render(self) -> np.ndarray | None:
        """Render the environment to an rgb frame.

        Returns
        -------
        np.ndarray | None
            A numpy array of shape (height, width, 3) containing the frame data.
        """
        if self.render_mode == 'human':
            return None

        return self.simulator.render()

    def _create_simulator_info(self) -> dict[str, Any]:
        if self.enable_info:
            return {
                # 'turbine_power': self.simulator.turbine_power(),
            }

        return {}

    def _close_render_window(self):
        if cv2.getWindowProperty(self.cv2_window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow(self.cv2_window_name)

    # def _create_observation(self) -> torch.Tensor:
    #     return torch.from_numpy(self.observation_space.sample()).to(
    #         device=self.device, dtype=self.dtype
    #     )

    @property
    def device(self) -> str | torch.device | int:
        return self.simulator.device

    @property
    def dtype(self) -> torch.dtype:
        return self.simulator.dtype

    @property
    def layout(self) -> TurbineLayout:
        return self._layout

    @property
    def num_turbines(self) -> int:
        return self._num_turbines

    @property
    def seed(self) -> int:
        return self._seed
