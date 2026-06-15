from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from pingpong_rl.envs.competitive_env import CompetitivePingPongConfig, CompetitiveRealisticEnv


@dataclass(frozen=True)
class AdvancedStrokesConfig(CompetitivePingPongConfig):
    agent_x_min: float = 55.0
    agent_x_max: float = 330.0
    opponent_x_min: float = 570.0
    opponent_x_max: float = 845.0
    paddle_x_speed: float = 8.0
    paddle_y_speed: float = 12.0
    paddle_angle_speed: float = 0.15
    max_paddle_angle: float = 0.92
    opponent_speed: float = 17.0
    opponent_x_speed: float = 8.0
    max_ball_speed: float = 17.0
    max_spin: float = 14.0
    spin_lift: float = 0.019
    spin_bounce_coupling: float = 0.095
    target_rally_length: int = 34
    max_steps: int = 3000
    assist_residual_scale: float = 0.58
    technique_reward: float = 0.28
    outside_position_reward: float = 0.002
    loop_spin_threshold: float = 1.1
    drive_speed_threshold: float = 7.5
    smash_speed_threshold: float = 9.0


class AdvancedStrokesEnv(CompetitiveRealisticEnv):
    """Stage 9: wider paddle movement with drive, loop, chop, and smash intents."""

    def __init__(self, render_mode: str | None = None, config: AdvancedStrokesConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or AdvancedStrokesConfig())
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32)
        self.brush_intent = 0.0
        self.stroke_power = 0.5
        self.last_stroke_type = "none"
        self.loop_attempts = 0
        self.loop_landings = 0
        self.drive_attempts = 0
        self.drive_landings = 0
        self.smash_attempts = 0
        self.smash_landings = 0
        self.chop_attempts = 0
        self.chop_landings = 0
        self.outside_steps = 0
        self.outside_hits = 0
        self.max_topspin = 0.0
        self.max_backspin = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self.brush_intent = 0.0
        self.stroke_power = 0.5
        self.last_stroke_type = "none"
        self.loop_attempts = 0
        self.loop_landings = 0
        self.drive_attempts = 0
        self.drive_landings = 0
        self.smash_attempts = 0
        self.smash_landings = 0
        self.chop_attempts = 0
        self.chop_landings = 0
        self.outside_steps = 0
        self.outside_hits = 0
        self.max_topspin = 0.0
        self.max_backspin = 0.0
        self.agent_x = self.config.table_left - 4.0
        self.agent_min_x_seen = self.agent_x
        self.agent_max_x_seen = self.agent_x
        return self._get_obs(), self._get_info(False, False, False, False, False)

    def _apply_agent_action(self, action: np.ndarray) -> None:
        raw_attack = float(np.clip(action[3], -1.0, 1.0))
        self.attack_intent = raw_attack
        self.brush_intent = float(np.clip(0.36 + 0.46 * np.clip(action[4], -1.0, 1.0) + 0.16 * max(raw_attack, 0.0), -1.0, 1.0))
        self.stroke_power = float(np.clip(0.55 + 0.32 * max(raw_attack, 0.0) + 0.28 * np.clip(action[5], -1.0, 1.0), 0.0, 1.0))

        if self.ball_vx < 0:
            intercept_x = self.agent_x
            if self.required_landing_side == "agent" and self.shot_landed:
                intercept_x = self.ball_x - 32.0 + raw_attack * 56.0 - max(self.brush_intent, 0.0) * 18.0
            elif self.ball_x < self.config.net_x:
                intercept_x = self.ball_x - 46.0 + raw_attack * 40.0
            target_x = float(np.clip(intercept_x, self.config.agent_x_min, self.config.agent_x_max))
            target_y = self.predict_ball_y_at_x(target_x)
            swing_lift = 0.34 * self.brush_intent + 0.20 * self.stroke_power
            desired_angle = np.clip((target_y - self.agent_y) / 118.0 + swing_lift, -1.0, 1.0) * self.config.max_paddle_angle
        else:
            target_y = self.config.table_y - 102
            target_x = self.config.table_left - 26.0 + max(raw_attack, 0.0) * 38.0 + self.stroke_power * 18.0
            desired_angle = np.clip(0.16 * self.brush_intent + 0.18 * raw_attack, -0.55, 0.55)

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

    def _move_ball(self) -> None:
        self.ball_vy += self.config.gravity + self.config.spin_lift * self.ball_spin
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        speed = float(np.hypot(self.ball_vx, self.ball_vy))
        if speed > self.config.max_ball_speed:
            scale = self.config.max_ball_speed / speed
            self.ball_vx *= scale
            self.ball_vy *= scale

    def _bounce_from_paddle(self, player: str) -> None:
        if player != "agent":
            return super()._bounce_from_paddle(player)

        direction = 1
        paddle_x = self.agent_x
        paddle_y = self.agent_y
        paddle_vx = self.agent_vx
        paddle_vy = self.agent_vy
        angle = self.agent_angle
        attack = max(self.attack_intent, 0.0)
        brush = self.brush_intent
        power = self.stroke_power

        landing_min = self.config.net_x + 62
        landing_max = min(self.opponent_x - 30, self.config.table_right - 24)
        if power > 0.78 and attack > 0.15:
            base_landing_x = self.config.table_right - 56
        elif brush > 0.25:
            base_landing_x = self.config.deep_landing_x + 44.0 + attack * 35.0
        elif brush < -0.32:
            base_landing_x = self.config.net_x + 110.0 + attack * 38.0
        else:
            base_landing_x = self.config.deep_landing_x + attack * 58.0

        offset = float(np.clip((self.ball_y - paddle_y) / (self.config.paddle_height / 2), -1.0, 1.0))
        self.last_contact_quality = float(1.0 - min(abs(offset), 1.0))
        self.ball_x = paddle_x + direction * (self.config.paddle_width / 2 + self.config.ball_radius + 1)
        speed = min(
            max(abs(self.ball_vx) * (0.90 + 0.18 * power + 0.06 * attack) + 0.20 + max(paddle_vx, 0.0) * 0.11, self.config.ball_speed_x_min),
            self.config.max_ball_speed * 0.72,
        )
        self.ball_vx = direction * speed
        landing_x = base_landing_x + direction * angle * 118.0 + offset * 32.0 + paddle_vx * 2.8
        landing_x = float(np.clip(landing_x, landing_min, landing_max))
        self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius)

        if brush > 0.25:
            self.ball_vy -= 0.55 + 0.38 * brush + 0.22 * power
        elif brush < -0.32:
            self.ball_vy -= 0.35 + 0.18 * power
        else:
            self.ball_vy -= 0.72 + 0.55 * power + max(0.0, abs(angle) - 0.15) * 0.42

        spin_delta = angle * (4.7 + 1.6 * power) + paddle_vy * 0.10 - offset * 1.8 + brush * (8.0 + 2.4 * power)
        self.ball_spin = float(np.clip(self.ball_spin * 0.36 + spin_delta, -self.config.max_spin, self.config.max_spin))
        self.max_topspin = max(self.max_topspin, self.ball_spin)
        self.max_backspin = max(self.max_backspin, -self.ball_spin)
        self.outside_hits += int(self.agent_x < self.config.table_left)
        self.last_stroke_type = self._classify_stroke(speed, brush, power)
        self.loop_attempts += int(self.last_stroke_type == "loop")
        self.drive_attempts += int(self.last_stroke_type == "drive")
        self.smash_attempts += int(self.last_stroke_type == "smash")
        self.chop_attempts += int(self.last_stroke_type == "chop")

    def _classify_stroke(self, speed: float, brush: float, power: float) -> str:
        if brush >= 0.28 and self.ball_spin >= self.config.loop_spin_threshold:
            return "loop"
        if power >= 0.78 and speed >= self.config.smash_speed_threshold:
            return "smash"
        if brush <= -0.34 and self.ball_spin <= -1.2:
            return "chop"
        if power >= 0.52 and speed >= self.config.drive_speed_threshold:
            return "drive"
        return "block"

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
        reward = super()._reward(previous_distance, action[:4], agent_hit, legal_landing, agent_score, agent_miss, rally_success)
        if self.agent_x < self.config.table_left:
            reward += self.config.outside_position_reward
        if agent_hit:
            if self.last_stroke_type in {"loop", "drive", "smash", "chop"}:
                reward += self.config.technique_reward
            reward += 0.080 * min(max(self.ball_spin, 0.0), self.config.max_spin)
            reward -= 0.025 * min(max(-self.ball_spin, 0.0), self.config.max_spin)
            reward += 0.018 * min(abs(self.ball_vx), self.config.max_ball_speed)
        if legal_landing and self.last_hitter == "agent":
            if self.last_stroke_type == "loop":
                reward += 1.10
            elif self.last_stroke_type == "drive":
                reward += 0.38
            elif self.last_stroke_type == "smash":
                reward += 0.48
            elif self.last_stroke_type == "chop":
                reward += 0.04
        return float(reward)

    def _get_info(self, agent_hit: bool, agent_score: bool, agent_miss: bool, rally_success: bool, legal_landing: bool) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        if legal_landing and self.last_hitter == "agent":
            self.loop_landings += int(self.last_stroke_type == "loop")
            self.drive_landings += int(self.last_stroke_type == "drive")
            self.smash_landings += int(self.last_stroke_type == "smash")
            self.chop_landings += int(self.last_stroke_type == "chop")
        info.update(
            {
                "stage": 9,
                "brush_intent": self.brush_intent,
                "stroke_power": self.stroke_power,
                "last_stroke_type": self.last_stroke_type,
                "loop_attempts": self.loop_attempts,
                "loop_landings": self.loop_landings,
                "drive_attempts": self.drive_attempts,
                "drive_landings": self.drive_landings,
                "smash_attempts": self.smash_attempts,
                "smash_landings": self.smash_landings,
                "chop_attempts": self.chop_attempts,
                "chop_landings": self.chop_landings,
                "outside_steps": self.outside_steps,
                "outside_hits": self.outside_hits,
                "outside_play": self.outside_steps > 0 or self.outside_hits > 0,
                "max_topspin": self.max_topspin,
                "max_backspin": self.max_backspin,
            }
        )
        return info

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        outside_margin = (self.config.table_left - self.agent_x) / max(self.config.table_left - self.config.agent_x_min, 1.0)
        extras = np.array(
            [
                self.brush_intent,
                self.stroke_power * 2 - 1,
                np.clip(outside_margin, -1.0, 1.0),
                np.clip(self.max_topspin / self.config.max_spin, -1.0, 1.0),
                np.clip((self.loop_attempts + self.drive_attempts + self.smash_attempts) / 8.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate([base, extras]), -1.0, 1.0).astype(np.float32)
