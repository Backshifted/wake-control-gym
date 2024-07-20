from typing import Any

import gymnasium as gym
import torch

from wake_control_gym.wake_control_env import WakeControlEnv


class NoiseObservation(gym.ObservationWrapper):
    env: WakeControlEnv
    noise_generator: torch.Generator

    # Options
    noise_scale: float

    def __init__(self, env: gym.Env, *, noise_scale: float = 0.1):
        # TODO / FUTURE: Add adaptive noise scale based on observation type
        super().__init__(env)
        self.noise_scale = noise_scale
        self.noise_generator = torch.Generator(self.env.device)
        self.noise_generator.manual_seed(self.env.seed)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._torch_random = torch.Generator()
        self.noise_generator.manual_seed(self.env.seed)
        return self.observation(obs), info

    def observation(self, observation: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            observation.size(),
            generator=self.noise_generator,
            dtype=observation.dtype,
            device=observation.device,
        )
        return observation + noise * self.noise_scale
