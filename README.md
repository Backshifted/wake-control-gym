# Wake Control Gym

Wake Control Gym is a `pytorch` based `gymnasium` environment for optimising wind farm wake control strategies using (multi-agent) reinforcement learning. This environment allows users to interact with various components such as observations, actions, and reward functions in a customisable manner. Additionally, environment supports using any simulator as a backend, provided an adapter exists or the user writes creates one.

## Installation

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/backshifted/wake-control-gym.git
    cd wake-control-gym
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the package:

    ```bash
    pip install -e .
    ```

## Usage

The environment can be configured can be configured to use any observations, action representations, reward functions, and simulators. By default the environment is configured as single-agent reinforcement learning formulation discussed by Neustroev et al. [^1]: Single-agent; observations for farm tubulence intensity, wind speed and direction at each turbine, yaw angles for each turbine; wind-based action representation and farm power based reward.

Currently there is no default implementation for the simulator, WIP.

### Example Usage

```python
from functools import partial
import torch
from wake_control_gym import WakeControlEnv
from wake_control_gym.adapters.floris import FlorisAdapter

# Three turbine single-line layout defined in rotor diameters.
D = 126
layout = (
    [0, 6 * D, 12 * D],
    [0,     0,      0],
)
init_floris = partial(FlorisAdapter, floris_config='./emgauss.yaml')
env = WakeControlEnv(layout=layout, init_simulator=init_floris)
env.render_mode = 'human'

for _ in range(1000):
    action = env.action_space.sample()
    action = torch.tensor(env.action_space.sample())
    obs, reward, _, _, info = env.step(action)
```

## Components

### Simulator

The environment can be customised to fit any simulator through adapters, see `wake_control_gym.core.Simulator` for the base class and `wake_control_gym.adapters.floris.FlorisAdapter` for an implementation.

### Observations

Observations can be used to customise the environment, and are used according to the following life-cycle:

1. The environment instantiates the observations
1. Observations perform setup
   - Setting `high`, `low`, `dim`, and `obs_type`.
   - If the observation requires measurements from the simulator, the measurement points should be registered to the simulator.
1. Using the observations, the environment constructs an observation space by joining all observations.
   - Observations are in ordered according to `'global'` first, `'local'` second, and the order in which the individual observations were passed.
   - To slice the individual observation from the joint observation, the environment sets `Observation.index` which can be used in conjunction with the `Observation.dim` to create a slice from the observation tensor.
1. When the environment steps or resets the observation functions are called and shaped to match the observation space.

#### Global Observations

Global observatios can be used in both single-agent and multi-agent settings, they should return a tensor in shape: `(joint_observation_dim,)`.

#### Multi-Agent Observations

If the environment is multi-agent, the global observations are repeated for each agent to match the shape of the final observation space: `(num_turbines, joint_observation_dim)`.

#### Local Observations

Local observations are per-definition multi-agent and must return an observation for each agent. When called, the observation should return a tensor of shape `(num_agents, observation_dim)`.

#### Reading Simulator Data

The simulator base class supports batching observations, which requires access to the necessary measurement points ahead of time.

```python
MeasurementPoint = tuple[float, float] | tuple[float, float, float] # (x, y [,z])
```

Therefore, observations must register any required measurement points through simulator.register_measurement_points(...) in their constructor. The simulator should return an index with which the caller can access the measurement data when getting wind observations from the simulator.

```python
from wake_control_gym import WakeControlEnv
from wake_control_gym.core import MeasurementPoint, Observation, ObservationType, Simulator

MIN_WINDSPEED = 0
MAX_WINDSPEED = 20

class MastWindSpeeds(Observation):
    obs_type: ObservationType = 'global' # Observed by all agents in multi-agent settings
    measurement_point_indices: list[Any]

    def __init__(self, simulator: Simulator, measurement_points: list[MeasurementPoint]) -> None:
        self.dim = len(measurement_points)
        self.low = torch.full((self.dim,), MIN_WINDSPEED, device=simulator.device, dtype=simulator.dtype)
        self.high = torch.full((self.dim,), MAX_WINDSPEED, device=simulator.device, dtype=simulator.dtype)
        self.measurement_point_indices = simulator.register_measurement_points(measurement_points)

    def __call__(self, simulator: Simulator, *args, **kwargs) -> torch.Tensor:
        return simulator.wind_speed()[self.measurement_point_indices]

points = [(500, 0), (1500, 0)] # meters
env = WakeControlEnv(..., observations=[..., lambda sim: MastWindSpeeds(sim, points)])
```

### Action Representations

Action representations define action space of the environment and how these actions are applied within the simulator. Wake Control Gym provides customisable action representations to interact with the simulator.

#### Yaw Misalignment Action

The `YawMisalignmentAction` representation allows continuous adjustments to the yaw angles within a specified range. The action space is defined as a continuous range from -1 to 1, which is then scaled to the valid yaw angle range of the simulator.

```python
from wake_control_gym import WakeControlEnv
from wake_control_gym.core import Action, ActionRepresentation, Simulator

class YawMisalignmentAction(ActionRepresentation):
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

env = WakeControlEnv(..., action_representation=action_rep)
```

### Reward Functions

Reward functions define how the performance of the control strategy is measured. Wake Control Gym provides customizable reward functions to evaluate different aspects of the wind farm performance.

```python
from wake_control_gym import WakeControlEnv
from wake_control_gym.core import RewardFunction, Simulator

class FarmPowerReward(RewardFunction):
    reward_range: tuple[torch.Tensor, torch.Tensor]

    def __init__(self, simulator: Simulator) -> None:
        self.reward_range = (
            simulator.turbine_power_range[0] * simulator.num_turbines,
            simulator.turbine_power_range[1] * simulator.num_turbines,
        )

    def __call__(self, simulator: Simulator) -> torch.Tensor:
        return simulator.turbine_power().sum()
```

## References

[^1]: Neustroev, G., Andringa, S. P. E., Verzijlbergh, R. A., & de Weerdt, M. M. (2022). Deep Reinforcement Learning for Active Wake Control. In International Conference on Autonomous Agents and Multi-Agent Systems. Online: IFAAMAS, May 2022. [GitHub](https://github.com/AlgTUDelft/wind-farm-env).
