from __future__ import annotations

from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pingpong_rl.envs.opponents import (
    ModelOpponent,
    OpponentConfig,
    predict_y_at_x,
    random_action,
    scripted_action,
)


@dataclass(frozen=True)
class PongEnvConfig:
    width: int = 800
    height: int = 400
    ball_radius: int = 8
    ball_speed_x_min: float = 4.0
    ball_speed_x_max: float = 6.0
    ball_speed_y_max: float = 3.0
    paddle_width: int = 12
    paddle_height: int = 80
    paddle_speed: float = 6.0
    opponent_speed: float = 4.0
    agent_x: float = 40.0
    max_ball_speed: float = 10.0
    max_steps: int = 1000
    hit_reward: float = 0.2
    score_reward: float = 1.0
    miss_penalty: float = -1.0
    shaping_scale: float = 0.01
    move_penalty: float = 0.001
    attack_reward: float = 0.03
    serve_to_agent_probability: float = 0.5
    opponent_return_miss_probability: float = 0.0
    opponent: OpponentConfig = OpponentConfig()


DIFFICULTY_CONFIGS: dict[str, PongEnvConfig] = {
    "easy": PongEnvConfig(ball_speed_x_min=3.0, ball_speed_x_max=4.0, opponent_speed=2.0, paddle_height=100),
    "medium": PongEnvConfig(ball_speed_x_min=4.0, ball_speed_x_max=6.0, opponent_speed=4.0, paddle_height=80),
    "hard": PongEnvConfig(ball_speed_x_min=6.0, ball_speed_x_max=8.0, opponent_speed=5.0, paddle_height=60),
    "expert": PongEnvConfig(ball_speed_x_min=8.0, ball_speed_x_max=10.0, opponent_speed=6.0, paddle_height=50),
}


def make_pong_config(stage: int, opponent_model_path: str | None = None) -> PongEnvConfig:
    if stage == 2:
        base = DIFFICULTY_CONFIGS["easy"]
        return replace(
            base,
            paddle_height=120,
            hit_reward=0.25,
            attack_reward=0.06,
            serve_to_agent_probability=1.0,
            opponent_return_miss_probability=0.95,
            opponent=OpponentConfig(kind="scripted", speed=base.opponent_speed, random_action_probability=0.35),
        )
    if stage == 3:
        base = DIFFICULTY_CONFIGS["medium"]
        return replace(base, opponent=OpponentConfig(kind="scripted", speed=base.opponent_speed))
    if stage == 4:
        base = DIFFICULTY_CONFIGS["hard"]
        return replace(base, opponent=OpponentConfig(kind="predictive", speed=base.opponent_speed))
    if stage == 5:
        base = DIFFICULTY_CONFIGS["medium"]
        return replace(base, opponent=OpponentConfig(kind="model", speed=base.opponent_speed, model_path=opponent_model_path))
    raise ValueError(f"Unsupported Pong stage: {stage}")


class PongEnv(gym.Env):
    """Stages 2-5: Pong-style rally, scripted/predictive/model opponents."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, config: PongEnvConfig | None = None):
        super().__init__()
        self.config = config or PongEnvConfig()
        self.render_mode = render_mode
        self.renderer = None
        self.model_opponent = None
        self.opponent_x = self.config.width - self.config.agent_x

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.agent_y = 0.0
        self.agent_vy = 0.0
        self.opponent_y = 0.0
        self.opponent_vy = 0.0
        self.steps = 0
        self.rally_length = 0
        self.agent_hits = 0
        self.opponent_hits = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.ball_x = float(self.config.width / 2)
        self.ball_y = float(self.np_random.uniform(self.config.ball_radius, self.config.height - self.config.ball_radius))
        direction = -1 if self.np_random.random() < self.config.serve_to_agent_probability else 1
        self.ball_vx = direction * float(self.np_random.uniform(self.config.ball_speed_x_min, self.config.ball_speed_x_max))
        self.ball_vy = float(self.np_random.uniform(-self.config.ball_speed_y_max, self.config.ball_speed_y_max))
        self.agent_y = float(self.config.height / 2)
        self.opponent_y = float(self.config.height / 2)
        self.agent_vy = 0.0
        self.opponent_vy = 0.0
        self.steps = 0
        self.rally_length = 0
        self.agent_hits = 0
        self.opponent_hits = 0
        return self._get_obs(), self._get_info(False, False, False)

    def step(self, action: int):
        self.steps += 1
        previous_distance = abs(self.ball_y - self.agent_y)

        self._apply_agent_action(int(action))
        opponent_action = self._opponent_action()
        self._apply_opponent_action(opponent_action)
        agent_hit, opponent_hit = self._update_physics()

        agent_score = self.ball_x - self.config.ball_radius > self.config.width
        agent_miss = self.ball_x + self.config.ball_radius < 0
        reward = self._compute_reward(previous_distance, int(action), agent_hit, agent_score, agent_miss)
        terminated = bool(agent_score or agent_miss)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        info = self._get_info(agent_hit, agent_score, agent_miss)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return None
        if self.renderer is None:
            from pingpong_rl.render.pygame_renderer import PygameRenderer

            self.renderer = PygameRenderer(self.config.width, self.config.height)
        return self.renderer.render_pong(self, mode=self.render_mode)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _apply_agent_action(self, action: int) -> None:
        self.agent_vy = self._action_to_velocity(action, self.config.paddle_speed)
        self.agent_y = self._clamp_paddle(self.agent_y + self.agent_vy)

    def _apply_opponent_action(self, action: int) -> None:
        self.opponent_vy = self._action_to_velocity(action, self.config.opponent.speed)
        self.opponent_y = self._clamp_paddle(self.opponent_y + self.opponent_vy)

    def _opponent_action(self) -> int:
        if self.config.opponent.random_action_probability > 0 and self.np_random.random() < self.config.opponent.random_action_probability:
            return random_action(self.np_random)

        kind = self.config.opponent.kind
        if kind == "random":
            return random_action(self.np_random)
        if kind == "scripted":
            return scripted_action(self.ball_y, self.opponent_y)
        if kind == "predictive":
            target_y = predict_y_at_x(
                self.ball_x,
                self.ball_y,
                self.ball_vx,
                self.ball_vy,
                self.opponent_x,
                self.config.height,
                self.config.ball_radius,
            )
            return scripted_action(target_y, self.opponent_y)
        if kind == "model":
            if not self.config.opponent.model_path:
                return scripted_action(self.ball_y, self.opponent_y)
            if self.model_opponent is None:
                self.model_opponent = ModelOpponent(self.config.opponent.model_path)
            return self.model_opponent.predict(self._get_mirrored_obs_for_opponent())
        raise ValueError(f"Unsupported opponent kind: {kind}")

    def _update_physics(self) -> tuple[bool, bool]:
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y <= self.config.ball_radius:
            self.ball_y = float(self.config.ball_radius)
            self.ball_vy *= -1
        elif self.ball_y >= self.config.height - self.config.ball_radius:
            self.ball_y = float(self.config.height - self.config.ball_radius)
            self.ball_vy *= -1

        agent_hit = self._collides_with_paddle(self.config.agent_x, self.agent_y) and self.ball_vx < 0
        opponent_hit = self._collides_with_paddle(self.opponent_x, self.opponent_y) and self.ball_vx > 0
        if opponent_hit and self.config.opponent_return_miss_probability > 0:
            opponent_hit = bool(self.np_random.random() >= self.config.opponent_return_miss_probability)

        if agent_hit:
            self.ball_x = self.config.agent_x + self.config.paddle_width / 2 + self.config.ball_radius
            self._bounce_from_paddle(self.agent_y, self.agent_vy, direction=1)
            self.rally_length += 1
            self.agent_hits += 1
        elif opponent_hit:
            self.ball_x = self.opponent_x - self.config.paddle_width / 2 - self.config.ball_radius
            self._bounce_from_paddle(self.opponent_y, self.opponent_vy, direction=-1)
            self.rally_length += 1
            self.opponent_hits += 1

        self.ball_vx = float(np.clip(self.ball_vx, -self.config.max_ball_speed, self.config.max_ball_speed))
        self.ball_vy = float(np.clip(self.ball_vy, -self.config.max_ball_speed, self.config.max_ball_speed))
        return bool(agent_hit), bool(opponent_hit)

    def _bounce_from_paddle(self, paddle_y: float, paddle_vy: float, direction: int) -> None:
        speed_x = min(abs(self.ball_vx) * 1.03 + 0.05, self.config.max_ball_speed)
        self.ball_vx = direction * speed_x
        offset = (self.ball_y - paddle_y) / (self.config.paddle_height / 2)
        self.ball_vy += offset * 2.0 + paddle_vy * 0.15

    def _collides_with_paddle(self, paddle_x: float, paddle_y: float) -> bool:
        paddle_left = paddle_x - self.config.paddle_width / 2
        paddle_right = paddle_x + self.config.paddle_width / 2
        paddle_top = paddle_y - self.config.paddle_height / 2
        paddle_bottom = paddle_y + self.config.paddle_height / 2
        ball_left = self.ball_x - self.config.ball_radius
        ball_right = self.ball_x + self.config.ball_radius
        ball_top = self.ball_y - self.config.ball_radius
        ball_bottom = self.ball_y + self.config.ball_radius
        return bool(ball_right >= paddle_left and ball_left <= paddle_right and ball_bottom >= paddle_top and ball_top <= paddle_bottom)

    def _compute_reward(self, previous_distance: float, action: int, agent_hit: bool, agent_score: bool, agent_miss: bool) -> float:
        reward = 0.0
        if self.ball_vx < 0:
            current_distance = abs(self.ball_y - self.agent_y)
            reward += self.config.shaping_scale * (previous_distance - current_distance) / self.config.height
        if action != 0:
            reward -= self.config.move_penalty
        if agent_hit:
            reward += self.config.hit_reward
            center_offset = abs(self.ball_y - self.agent_y) / (self.config.paddle_height / 2)
            reward += self.config.attack_reward * min(center_offset, 1.0)
        if agent_score:
            reward += self.config.score_reward
        elif agent_miss:
            reward += self.config.miss_penalty
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        return self._normalize_obs(
            self.ball_x,
            self.ball_y,
            self.ball_vx,
            self.ball_vy,
            self.agent_y,
            self.agent_vy,
            self.opponent_y,
            self.opponent_vy,
        )

    def _get_mirrored_obs_for_opponent(self) -> np.ndarray:
        return self._normalize_obs(
            self.config.width - self.ball_x,
            self.ball_y,
            -self.ball_vx,
            self.ball_vy,
            self.opponent_y,
            self.opponent_vy,
            self.agent_y,
            self.agent_vy,
        )

    def _normalize_obs(self, ball_x, ball_y, ball_vx, ball_vy, agent_y, agent_vy, opponent_y, opponent_vy) -> np.ndarray:
        obs = np.array(
            [
                ball_x / self.config.width * 2 - 1,
                ball_y / self.config.height * 2 - 1,
                ball_vx / self.config.max_ball_speed,
                ball_vy / self.config.max_ball_speed,
                agent_y / self.config.height * 2 - 1,
                agent_vy / self.config.paddle_speed,
                opponent_y / self.config.height * 2 - 1,
                opponent_vy / max(self.config.opponent.speed, 1e-6),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _get_info(self, agent_hit: bool, agent_score: bool, agent_miss: bool) -> dict:
        return {
            "agent_hit": agent_hit,
            "agent_score": agent_score,
            "agent_miss": agent_miss,
            "agent_hits": self.agent_hits,
            "opponent_hits": self.opponent_hits,
            "rally_length": self.rally_length,
            "steps": self.steps,
        }

    def _clamp_paddle(self, y: float) -> float:
        half = self.config.paddle_height / 2
        return float(np.clip(y, half, self.config.height - half))

    @staticmethod
    def _action_to_velocity(action: int, speed: float) -> float:
        if action == 1:
            return -speed
        if action == 2:
            return speed
        return 0.0
