from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.compact_technique_env import CompactTechniqueConfig, CompactTechniqueEnv


@dataclass(frozen=True)
class VarietyTechniqueConfig(CompactTechniqueConfig):
    max_spin: float = 18.0
    spin_lift: float = 0.030
    max_ball_speed: float = 18.5
    target_rally_length: int = 34
    style_reward: float = 0.70
    arc_peak_reward: float = 0.45
    deep_drive_reward: float = 0.45
    trail_length: int = 90


class VarietyTechniqueEnv(CompactTechniqueEnv):
    """Stage 11: compact technique with visibly varied loops, drives, smashes, and chops."""

    STYLES = ("high_loop", "drive", "loop", "smash")

    def __init__(self, render_mode: str | None = None, config: VarietyTechniqueConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or VarietyTechniqueConfig())
        self.style_index = 0
        self.current_style = self.STYLES[0]
        self.last_target_style = self.current_style
        self.ball_trail: list[tuple[float, float]] = []
        self.agent_shot_start_y = 0.0
        self.agent_shot_min_y = 0.0
        self.agent_shot_max_x = 0.0
        self.agent_shot_peak_arc = 0.0
        self.high_arc_landings = 0
        self.deep_drive_landings = 0
        self.smash_score_attempts = 0
        self.chop_landings = 0
        self.style_match_landings = 0
        self.unique_styles_landed: set[str] = set()
        self.high_arc_episodes = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.style_index = int(self.np_random.integers(0, len(self.STYLES)))
        self.current_style = self.STYLES[self.style_index]
        self.last_target_style = self.current_style
        self.ball_trail = [(self.ball_x, self.ball_y)]
        self.agent_shot_start_y = self.ball_y
        self.agent_shot_min_y = self.ball_y
        self.agent_shot_max_x = self.ball_x
        self.agent_shot_peak_arc = 0.0
        self.high_arc_landings = 0
        self.deep_drive_landings = 0
        self.smash_score_attempts = 0
        self.chop_landings = 0
        self.style_match_landings = 0
        self.unique_styles_landed = set()
        self.high_arc_episodes = 0
        return obs, info

    def _advance_style(self) -> None:
        self.style_index = (self.style_index + 1) % len(self.STYLES)
        self.current_style = self.STYLES[self.style_index]

    def _style_params(self) -> tuple[float, float, float]:
        if self.current_style == "high_loop":
            return 0.94, 0.48, -0.52
        if self.current_style == "loop":
            return 0.78, 0.58, -0.42
        if self.current_style == "drive":
            return 0.34, 0.90, -0.18
        if self.current_style == "smash":
            return 0.18, 1.00, -0.14
        if self.current_style == "chop":
            return -0.58, 0.38, -0.26
        return 0.40, 0.62, -0.20

    def _apply_agent_action(self, action: np.ndarray) -> None:
        style_brush, style_power, style_angle = self._style_params()
        raw_attack = float(np.clip(action[3], -1.0, 1.0))
        self.attack_intent = raw_attack
        self.brush_intent = float(np.clip(style_brush + 0.08 * np.clip(action[4], -1.0, 1.0), -1.0, 1.0))
        self.stroke_power = float(np.clip(style_power + 0.08 * np.clip(action[5], -1.0, 1.0), 0.0, 1.0))

        if self.ball_vx < 0:
            intercept_x = self.agent_x
            if self.required_landing_side == "agent" and self.shot_landed:
                forward = 58.0 if self.current_style in {"drive", "smash"} else 28.0
                intercept_x = self.ball_x - 42.0 + max(raw_attack, 0.0) * forward
            elif self.ball_x < self.config.net_x:
                intercept_x = self.ball_x - 52.0 + max(raw_attack, 0.0) * 26.0
            target_x = float(np.clip(intercept_x, self.config.agent_x_min, self.config.agent_x_max))
            target_y = self.predict_ball_y_at_x(target_x)
            desired_angle = style_angle - np.clip(abs((target_y - self.agent_y) / 180.0), 0.0, 0.24)
        else:
            target_y = self.config.table_y - 102
            target_x = self.config.table_left - 32.0 + self.stroke_power * 18.0
            desired_angle = style_angle * 0.75

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
        super()._move_ball()
        self.ball_trail.append((self.ball_x, self.ball_y))
        if len(self.ball_trail) > self.config.trail_length:
            self.ball_trail = self.ball_trail[-self.config.trail_length :]
        if self.last_hitter == "agent" and not self.shot_landed:
            self.agent_shot_min_y = min(self.agent_shot_min_y, self.ball_y)
            self.agent_shot_max_x = max(self.agent_shot_max_x, self.ball_x)
            self.agent_shot_peak_arc = max(self.agent_shot_peak_arc, self.agent_shot_start_y - self.agent_shot_min_y)

    def _bounce_from_paddle(self, player: str) -> None:
        if player != "agent":
            return super()._bounce_from_paddle(player)

        style = self.current_style
        self.last_target_style = style
        super()._bounce_from_paddle(player)
        self._shape_variety_ball(style)
        self.last_stroke_type = self._stroke_type_for_style(style)
        self.loop_attempts += int(self.last_stroke_type == "loop")
        self.drive_attempts += int(self.last_stroke_type == "drive")
        self.smash_attempts += int(self.last_stroke_type == "smash")
        self.chop_attempts += int(self.last_stroke_type == "chop")
        self.agent_shot_start_y = self.ball_y
        self.agent_shot_min_y = self.ball_y
        self.agent_shot_max_x = self.ball_x
        self.agent_shot_peak_arc = 0.0
        self._advance_style()

    def _shape_variety_ball(self, style: str) -> None:
        if style == "high_loop":
            self.ball_vx = min(max(self.ball_vx + 1.1, 8.1), 9.0)
            landing_x = self.config.table_right - 70.0
            self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius) - 2.45
            self.ball_spin = float(np.clip(max(self.ball_spin, 6.6), -self.config.max_spin, self.config.max_spin))
        elif style == "loop":
            self.ball_vx = min(max(self.ball_vx + 1.0, 8.2), 9.2)
            landing_x = self.config.table_right - 88.0
            self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius) - 1.10
            self.ball_spin = float(np.clip(max(self.ball_spin, 5.5), -self.config.max_spin, self.config.max_spin))
        elif style == "drive":
            self.ball_vx = min(max(self.ball_vx + 1.4, 9.4), 10.6)
            landing_x = self.config.table_right - 118.0
            self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius) - 0.45
            self.ball_spin = float(np.clip(max(self.ball_spin, 2.8), -self.config.max_spin, self.config.max_spin))
        elif style == "smash":
            self.ball_vx = min(max(self.ball_vx + 1.8, 9.6), 11.2)
            landing_x = self.config.table_right - 118.0
            self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius) - 0.10
            self.ball_spin = float(np.clip(max(self.ball_spin, 1.8), -self.config.max_spin, self.config.max_spin))
        elif style == "chop":
            self.ball_vx = min(max(self.ball_vx - 0.8, 5.0), 5.9)
            landing_x = self.config.net_x + 88.0
            self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius) + 0.55
            self.ball_spin = float(np.clip(-3.6, -self.config.max_spin, self.config.max_spin))

    def _stroke_type_for_style(self, style: str) -> str:
        if style in {"high_loop", "loop"}:
            return "loop"
        if style == "drive":
            return "drive"
        if style == "smash":
            return "smash"
        if style == "chop":
            return "chop"
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
        reward = super()._reward(previous_distance, action, agent_hit, legal_landing, agent_score, agent_miss, rally_success)
        if legal_landing and self.last_hitter == "agent":
            if self.last_target_style == "high_loop" and self.agent_shot_peak_arc >= 42.0:
                reward += self.config.arc_peak_reward
            if self.last_target_style in {"drive", "smash"} and self.ball_x >= self.config.table_right - 95:
                reward += self.config.deep_drive_reward
            if self._stroke_type_for_style(self.last_target_style) == self.last_stroke_type:
                reward += self.config.style_reward
        return float(reward)

    def _get_info(self, agent_hit: bool, agent_score: bool, agent_miss: bool, rally_success: bool, legal_landing: bool) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        if legal_landing and self.last_hitter == "agent":
            if self.last_target_style == "high_loop" and self.agent_shot_peak_arc >= 42.0:
                self.high_arc_landings += 1
                self.high_arc_episodes = 1
            if self.last_target_style in {"drive", "smash"} and self.ball_x >= self.config.table_right - 180:
                self.deep_drive_landings += 1
            if self.last_target_style == "smash":
                self.smash_score_attempts += 1
            if self.last_target_style == "chop":
                self.chop_landings += 1
            if self._stroke_type_for_style(self.last_target_style) == self.last_stroke_type:
                self.style_match_landings += 1
            self.unique_styles_landed.add(self.last_target_style)

        info.update(
            {
                "stage": 11,
                "current_style": self.current_style,
                "last_target_style": self.last_target_style,
                "high_arc_landings": self.high_arc_landings,
                "deep_drive_landings": self.deep_drive_landings,
                "smash_score_attempts": self.smash_score_attempts,
                "style_match_landings": self.style_match_landings,
                "unique_styles_landed": len(self.unique_styles_landed),
                "high_arc_episode": self.high_arc_episodes > 0,
                "agent_shot_peak_arc": self.agent_shot_peak_arc,
            }
        )
        return info
