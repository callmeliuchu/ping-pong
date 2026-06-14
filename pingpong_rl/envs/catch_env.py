from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class CatchEnvConfig:
    width: int = 800
    height: int = 400
    ball_radius: int = 8
    ball_speed_x_min: float = 4.0
    ball_speed_x_max: float = 6.0
    ball_speed_y_max: float = 3.0
    paddle_width: int = 12
    paddle_height: int = 96
    paddle_speed: float = 6.0
    agent_x: float = 40.0
    max_ball_speed: float = 10.0
    max_steps: int = 240
    shaping_scale: float = 0.02
    move_penalty: float = 0.001
    hit_reward: float = 1.0
    miss_penalty: float = -1.0


class CatchEnv(gym.Env):
    """Stage 1 environment: move one paddle to catch balls from right to left."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, config: CatchEnvConfig | None = None):
        super().__init__()
        self.config = config or CatchEnvConfig()
        self.render_mode = render_mode
        self.renderer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.agent_y = 0.0
        self.agent_vy = 0.0
        self.steps = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        self.ball_x = float(options.get("ball_x", self.config.width - 80))
        self.ball_y = float(self.np_random.uniform(
            self.config.ball_radius,
            self.config.height - self.config.ball_radius,
        ))
        self.ball_vx = -float(self.np_random.uniform(
            self.config.ball_speed_x_min,
            self.config.ball_speed_x_max,
        ))
        self.ball_vy = float(self.np_random.uniform(
            -self.config.ball_speed_y_max,
            self.config.ball_speed_y_max,
        ))
        self.agent_y = float(self.config.height / 2)
        self.agent_vy = 0.0
        self.steps = 0

        return self._get_obs(), self._get_info(agent_hit=False, agent_miss=False)

    def step(self, action: int):
        self.steps += 1
        previous_distance = abs(self.ball_y - self.agent_y)

        self._apply_agent_action(int(action))
        self._update_ball()

        agent_hit = self._collides_with_agent_paddle()
        agent_miss = self.ball_x + self.config.ball_radius < 0

        reward = self._compute_reward(
            previous_distance=previous_distance,
            action=int(action),
            agent_hit=agent_hit,
            agent_miss=agent_miss,
        )

        terminated = bool(agent_hit or agent_miss)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        obs = self._get_obs()
        info = self._get_info(agent_hit=agent_hit, agent_miss=agent_miss)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return None

        if self.renderer is None:
            from pingpong_rl.render.pygame_renderer import PygameRenderer

            self.renderer = PygameRenderer(self.config.width, self.config.height)

        return self.renderer.render_catch(self, mode=self.render_mode)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _apply_agent_action(self, action: int) -> None:
        if action == 1:
            self.agent_vy = -self.config.paddle_speed
        elif action == 2:
            self.agent_vy = self.config.paddle_speed
        else:
            self.agent_vy = 0.0

        self.agent_y += self.agent_vy
        half_paddle = self.config.paddle_height / 2
        self.agent_y = float(np.clip(
            self.agent_y,
            half_paddle,
            self.config.height - half_paddle,
        ))

    def _update_ball(self) -> None:
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y <= self.config.ball_radius:
            self.ball_y = float(self.config.ball_radius)
            self.ball_vy *= -1
        elif self.ball_y >= self.config.height - self.config.ball_radius:
            self.ball_y = float(self.config.height - self.config.ball_radius)
            self.ball_vy *= -1

    def _collides_with_agent_paddle(self) -> bool:
        paddle_left = self.config.agent_x - self.config.paddle_width / 2
        paddle_right = self.config.agent_x + self.config.paddle_width / 2
        paddle_top = self.agent_y - self.config.paddle_height / 2
        paddle_bottom = self.agent_y + self.config.paddle_height / 2

        ball_left = self.ball_x - self.config.ball_radius
        ball_right = self.ball_x + self.config.ball_radius
        ball_top = self.ball_y - self.config.ball_radius
        ball_bottom = self.ball_y + self.config.ball_radius

        return bool(
            ball_right >= paddle_left
            and ball_left <= paddle_right
            and ball_bottom >= paddle_top
            and ball_top <= paddle_bottom
            and self.ball_vx < 0
        )

    def _compute_reward(
        self,
        previous_distance: float,
        action: int,
        agent_hit: bool,
        agent_miss: bool,
    ) -> float:
        reward = 0.0

        current_distance = abs(self.ball_y - self.agent_y)
        reward += self.config.shaping_scale * (previous_distance - current_distance) / self.config.height

        if action != 0:
            reward -= self.config.move_penalty

        if agent_hit:
            reward += self.config.hit_reward
        elif agent_miss:
            reward += self.config.miss_penalty

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        obs = np.array(
            [
                self.ball_x / self.config.width * 2 - 1,
                self.ball_y / self.config.height * 2 - 1,
                self.ball_vx / self.config.max_ball_speed,
                self.ball_vy / self.config.max_ball_speed,
                self.agent_y / self.config.height * 2 - 1,
                self.agent_vy / self.config.paddle_speed,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _get_info(self, agent_hit: bool, agent_miss: bool) -> dict:
        return {
            "agent_hit": agent_hit,
            "agent_miss": agent_miss,
            "steps": self.steps,
            "distance_to_ball": abs(self.ball_y - self.agent_y),
        }
