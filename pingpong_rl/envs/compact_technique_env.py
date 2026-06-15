from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.advanced_strokes_env import AdvancedStrokesConfig, AdvancedStrokesEnv


@dataclass(frozen=True)
class CompactTechniqueConfig(AdvancedStrokesConfig):
    paddle_height: int = 88
    paddle_width: int = 18
    paddle_y_speed: float = 14.0
    paddle_angle_speed: float = 0.18
    assist_residual_scale: float = 0.50
    target_rally_length: int = 30
    max_steps: int = 3000
    min_loop_angle: float = 0.16
    min_drive_angle: float = 0.10
    block_penalty: float = 0.18
    angled_contact_reward: float = 0.22
    compact_landing_reward: float = 0.58


class CompactTechniqueEnv(AdvancedStrokesEnv):
    """Stage 10: compact paddle with angled drive and loop technique constraints."""

    def __init__(self, render_mode: str | None = None, config: CompactTechniqueConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or CompactTechniqueConfig())
        self.angled_hits = 0
        self.compact_technique_hits = 0
        self.compact_technique_landings = 0
        self.block_hits = 0
        self.block_landings = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.angled_hits = 0
        self.compact_technique_hits = 0
        self.compact_technique_landings = 0
        self.block_hits = 0
        self.block_landings = 0
        return obs, info

    def _apply_agent_action(self, action: np.ndarray) -> None:
        raw_attack = float(np.clip(action[3], -1.0, 1.0))
        self.attack_intent = raw_attack
        self.brush_intent = float(np.clip(0.40 + 0.42 * np.clip(action[4], -1.0, 1.0) + 0.20 * max(raw_attack, 0.0), -1.0, 1.0))
        self.stroke_power = float(np.clip(0.52 + 0.34 * max(raw_attack, 0.0) + 0.28 * np.clip(action[5], -1.0, 1.0), 0.0, 1.0))

        if self.ball_vx < 0:
            intercept_x = self.agent_x
            if self.required_landing_side == "agent" and self.shot_landed:
                intercept_x = self.ball_x - 38.0 + raw_attack * 54.0 - max(self.brush_intent, 0.0) * 14.0
            elif self.ball_x < self.config.net_x:
                intercept_x = self.ball_x - 50.0 + raw_attack * 36.0
            target_x = float(np.clip(intercept_x, self.config.agent_x_min, self.config.agent_x_max))
            target_y = self.predict_ball_y_at_x(target_x)
            swing_lift = 0.42 * self.brush_intent + 0.22 * self.stroke_power
            desired_angle = -np.clip(abs((target_y - self.agent_y) / 105.0) + swing_lift, 0.0, 1.0) * self.config.max_paddle_angle
        else:
            target_y = self.config.table_y - 102
            target_x = self.config.table_left - 32.0 + max(raw_attack, 0.0) * 38.0 + self.stroke_power * 16.0
            desired_angle = -np.clip(0.22 * self.brush_intent + 0.20 * max(raw_attack, 0.0), 0.0, 0.58)

        base_vx = float(np.clip(target_x - self.agent_x, -self.config.paddle_x_speed, self.config.paddle_x_speed))
        base_vy = float(np.clip(target_y - self.agent_y, -self.config.paddle_y_speed, self.config.paddle_y_speed))
        base_angle_delta = float(np.clip(desired_angle - self.agent_angle, -self.config.paddle_angle_speed, self.config.paddle_angle_speed))
        residual = self.config.assist_residual_scale
        self.agent_vx = float(base_vx + np.clip(action[0], -1.0, 1.0) * self.config.paddle_x_speed * residual)
        self.agent_vy = float(base_vy + np.clip(action[1], -1.0, 1.0) * self.config.paddle_y_speed * residual)
        self.agent_x = float(np.clip(self.agent_x + self.agent_vx, self.config.agent_x_min, self.config.agent_x_max))
        self.agent_y = self._clamp_paddle_y(self.agent_y + self.agent_vy)
        self.agent_angle = float(
            np.clip(
                self.agent_angle + base_angle_delta + np.clip(action[2], -1.0, 1.0) * self.config.paddle_angle_speed * residual,
                -self.config.max_paddle_angle,
                self.config.max_paddle_angle,
            )
        )
        self.agent_min_x_seen = min(self.agent_min_x_seen, self.agent_x)
        self.agent_max_x_seen = max(self.agent_max_x_seen, self.agent_x)
        self.outside_steps += int(self.agent_x < self.config.table_left)

    def _classify_stroke(self, speed: float, brush: float, power: float) -> str:
        angled_loop = self._closed_racket_angle(self.config.min_loop_angle)
        angled_drive = self._closed_racket_angle(self.config.min_drive_angle)
        if angled_loop and brush >= 0.28 and self.ball_spin >= self.config.loop_spin_threshold:
            return "loop"
        if angled_drive and power >= 0.78 and speed >= self.config.smash_speed_threshold:
            return "smash"
        if angled_loop and brush <= -0.34 and self.ball_spin <= -1.2:
            return "chop"
        if angled_drive and power >= 0.52 and speed >= self.config.drive_speed_threshold:
            return "drive"
        return "block"

    def _closed_racket_angle(self, threshold: float) -> bool:
        return bool(self.agent_angle <= -threshold)

    def _bounce_from_paddle(self, player: str) -> None:
        before_hits = self.agent_hits if player == "agent" else None
        super()._bounce_from_paddle(player)
        if player != "agent" or before_hits is None:
            return
        angled = self._closed_racket_angle(self.config.min_drive_angle)
        self.angled_hits += int(angled)
        self.compact_technique_hits += int(self.last_stroke_type in {"loop", "drive", "smash"})
        self.block_hits += int(self.last_stroke_type == "block")

    def _reward(
        self,
        previous_distance: float,
        action: np.ndarray,
        agent_hit: bool,
        legal_landing: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
    ) -> float:
        reward = super()._reward(previous_distance, action, agent_hit, legal_landing, agent_score, agent_miss, rally_success)
        if agent_hit and self._closed_racket_angle(self.config.min_drive_angle):
            reward += self.config.angled_contact_reward
        if agent_hit and self.last_stroke_type == "block":
            reward -= self.config.block_penalty
        if legal_landing and self.last_hitter == "agent" and self.last_stroke_type in {"loop", "drive", "smash"}:
            reward += self.config.compact_landing_reward
        return float(reward)

    def _get_info(self, agent_hit: bool, agent_score: bool, agent_miss: bool, rally_success: bool, legal_landing: bool) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        if legal_landing and self.last_hitter == "agent":
            self.compact_technique_landings += int(self.last_stroke_type in {"loop", "drive", "smash"})
            self.block_landings += int(self.last_stroke_type == "block")
        agent_hits = max(self.agent_hits, 1)
        info.update(
            {
                "stage": 10,
                "paddle_height": self.config.paddle_height,
                "angled_hits": self.angled_hits,
                "angled_hit_rate": self.angled_hits / agent_hits,
                "compact_technique_hits": self.compact_technique_hits,
                "compact_technique_landings": self.compact_technique_landings,
                "compact_technique_rate": self.compact_technique_hits / agent_hits,
                "block_hits": self.block_hits,
                "block_landings": self.block_landings,
                "block_rate": self.block_hits / agent_hits,
            }
        )
        return info
