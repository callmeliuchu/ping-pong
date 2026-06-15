from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pingpong_rl.envs.variety_technique_env import VarietyTechniqueConfig, VarietyTechniqueEnv


@dataclass(frozen=True)
class SelfPlayVarietyConfig(VarietyTechniqueConfig):
    opponent_model_path: str | None = None
    opponent_deterministic: bool = True
    opponent_reaction_noise: float = 0.02


class SelfPlayVarietyEnv(VarietyTechniqueEnv):
    """Stage 12: Stage 11 variety physics with a historical PPO model as opponent."""

    def __init__(self, render_mode: str | None = None, config: SelfPlayVarietyConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or SelfPlayVarietyConfig())
        self.self_play_config = config or SelfPlayVarietyConfig()
        self.history_opponent: PPO | None = None
        self.history_opponent_loaded = False
        self.model_opponent_steps = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.model_opponent_steps = 0
        obs, info = super().reset(seed=seed, options=options)
        self._opponent_min_x_seen = self.opponent_x
        self._opponent_max_x_seen = self.opponent_x
        return obs, info

    def _load_history_opponent(self) -> None:
        if self.history_opponent_loaded:
            return
        self.history_opponent_loaded = True
        model_path = self.self_play_config.opponent_model_path
        if not model_path:
            return
        path = Path(model_path)
        if path.exists() or path.with_suffix(".zip").exists():
            self.history_opponent = PPO.load(path)

    def _apply_opponent_policy(self) -> None:
        self._load_history_opponent()
        if self.history_opponent is None:
            return super()._apply_opponent_policy()

        mirrored_obs = self._get_mirrored_obs_for_history_opponent()
        action, _ = self.history_opponent.predict(
            mirrored_obs,
            deterministic=self.self_play_config.opponent_deterministic,
        )
        self._apply_model_opponent_action(np.asarray(action, dtype=np.float32))
        self.model_opponent_steps += 1

    def _get_mirrored_obs_for_history_opponent(self) -> np.ndarray:
        target_y = self.predict_ball_y_at_x(self.opponent_x) if self.ball_vx > 0 else self.config.table_y - 102
        desired_angle = -0.25 if self.ball_vx > 0 else -0.12
        outside_margin = (self.config.table_right - self.opponent_x) / max(self.config.width - self.config.table_right, 1.0)
        obs = np.array(
            [
                (self.config.width - self.ball_x) / self.config.width * 2 - 1,
                self.ball_y / self.config.height * 2 - 1,
                -self.ball_vx / self.config.max_ball_speed,
                self.ball_vy / self.config.max_ball_speed,
                self.ball_spin / self.config.max_spin,
                (self.config.width - self.opponent_x) / self.config.width * 2 - 1,
                self.opponent_y / self.config.height * 2 - 1,
                -self.opponent_vx / self.config.paddle_x_speed,
                self.opponent_vy / self.config.paddle_y_speed,
                -self.opponent_angle / self.config.max_paddle_angle,
                (self.config.width - self.agent_x) / self.config.width * 2 - 1,
                self.agent_y / self.config.height * 2 - 1,
                -self.agent_angle / self.config.max_paddle_angle,
                target_y / self.config.height * 2 - 1,
                (target_y - self.opponent_y) / self.config.height,
                desired_angle / self.config.max_paddle_angle,
                self.attack_intent,
                (self.opponent_max_x_seen() - self.opponent_min_x_seen()) / max(self.config.opponent_x_max - self.config.opponent_x_min, 1.0) * 2 - 1,
                self.brush_intent,
                self.stroke_power * 2 - 1,
                np.clip(outside_margin, -1.0, 1.0),
                np.clip(self.max_topspin / self.config.max_spin, -1.0, 1.0),
                np.clip((self.loop_attempts + self.drive_attempts + self.smash_attempts) / 8.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def opponent_min_x_seen(self) -> float:
        return getattr(self, "_opponent_min_x_seen", self.opponent_x)

    def opponent_max_x_seen(self) -> float:
        return getattr(self, "_opponent_max_x_seen", self.opponent_x)

    def _apply_model_opponent_action(self, action: np.ndarray) -> None:
        raw_attack = float(np.clip(action[3], -1.0, 1.0))
        if self.ball_vx > 0:
            target_x = self.opponent_x
            if self.required_landing_side == "opponent" and self.shot_landed:
                target_x = self.ball_x + 42.0 - max(raw_attack, 0.0) * 34.0
            elif self.ball_x > self.config.net_x:
                target_x = self.ball_x + 52.0 - max(raw_attack, 0.0) * 26.0
            target_x = float(np.clip(target_x, self.config.opponent_x_min, self.config.opponent_x_max))
            target_y = self.predict_ball_y_at_x(target_x)
            desired_angle = np.clip(0.24 + abs((target_y - self.opponent_y) / 180.0), 0.0, self.config.max_paddle_angle)
        else:
            target_y = self.config.table_y - 102
            target_x = self.config.table_right + 32.0 - max(raw_attack, 0.0) * 18.0
            desired_angle = 0.18

        residual = self.config.assist_residual_scale
        base_vx = float(np.clip(target_x - self.opponent_x, -self.config.opponent_x_speed, self.config.opponent_x_speed))
        base_vy = float(np.clip(target_y - self.opponent_y, -self.config.opponent_speed, self.config.opponent_speed))
        base_angle_delta = float(np.clip(desired_angle - self.opponent_angle, -self.config.paddle_angle_speed, self.config.paddle_angle_speed))
        noise = self.self_play_config.opponent_reaction_noise
        self.opponent_vx = float(base_vx - np.clip(action[0], -1.0, 1.0) * self.config.opponent_x_speed * residual)
        self.opponent_vy = float(base_vy + np.clip(action[1], -1.0, 1.0) * self.config.opponent_speed * residual)
        if noise > 0.0:
            self.opponent_vx += float(self.np_random.normal(0.0, noise * self.config.opponent_x_speed))
            self.opponent_vy += float(self.np_random.normal(0.0, noise * self.config.opponent_speed))
        self.opponent_x = float(np.clip(self.opponent_x + self.opponent_vx, self.config.opponent_x_min, self.config.opponent_x_max))
        self.opponent_y = self._clamp_paddle_y(self.opponent_y + self.opponent_vy)
        self.opponent_angle = float(
            np.clip(
                self.opponent_angle + base_angle_delta - np.clip(action[2], -1.0, 1.0) * self.config.paddle_angle_speed * residual,
                -self.config.max_paddle_angle,
                self.config.max_paddle_angle,
            )
        )
        self._opponent_min_x_seen = min(self.opponent_min_x_seen(), self.opponent_x)
        self._opponent_max_x_seen = max(self.opponent_max_x_seen(), self.opponent_x)

    def _get_info(self, agent_hit: bool, agent_score: bool, agent_miss: bool, rally_success: bool, legal_landing: bool) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update(
            {
                "stage": 12,
                "self_play_opponent": self.self_play_config.opponent_model_path,
                "model_opponent_steps": self.model_opponent_steps,
                "opponent_model_loaded": self.history_opponent is not None,
            }
        )
        return info
