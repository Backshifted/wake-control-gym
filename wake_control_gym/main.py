from functools import partial

import gymnasium as gym
import numpy as np
import torch

from wake_control_gym import WakeControlEnv
from wake_control_gym.observations import (
    TurbulenceIntensity,
    TurbineWindDirection,
    TurbineWindSpeed,
    TurbineYawAngle,
    MastWindDirections,
)
from wake_control_gym.adapters.floris import FlorisAdapter


def main():
    D = 126  # Rotor diameter
    layout = (
        [0, 6 * D, 12 * D],
        [0, 0, 0],
    )
    observations = [
        TurbulenceIntensity,
        TurbineWindDirection,
        partial(MastWindDirections, measurement_points=[(300, 0)]),
        TurbineWindSpeed,
        TurbineYawAngle,
    ]
    init_floris = partial(FlorisAdapter, floris_config='./floris-jensen.yaml')

    env = WakeControlEnv(
        layout=layout,
        observations=observations,
        init_simulator=init_floris,
        is_multi_agent=True,
    )

    # env.render_mode = 'human'
    yaws = torch.arange(30).view(30, 1).repeat(1, 3)
    obs, info = env.reset()

    for yaw_config in yaws:
        obs, reward, _, _, info = env.step(yaw_config)
        print(obs, reward)

    env.close()


if __name__ == '__main__':
    main()
