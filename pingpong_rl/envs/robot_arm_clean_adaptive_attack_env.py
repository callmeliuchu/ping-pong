from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.robot_arm_adaptive_attack_env import (
    RobotArmAdaptiveAttackConfig,
    RobotArmAdaptiveAttackEnv,
)


@dataclass(frozen=True)
class RobotArmCleanAdaptiveAttackConfig(RobotArmAdaptiveAttackConfig):
    """Stage 19: keep adaptive attack while reducing wrong-side gift scoring."""

    wrong_side_score_penalty: float = 18.0
    combo_attack_reward: float = 0.86
    adaptive_score_reward: float = 3.8
    clean_combo_score_reward: float = 4.6
    red_safe_return_margin: float = 160.0
    red_safe_return_width: float = 70.0
    red_safe_return_speed: float = 6.2
    red_safe_return_spin_scale: float = 0.18
    clean_attack_rally_target: int = 2


class RobotArmCleanAdaptiveAttackEnv(RobotArmAdaptiveAttackEnv):
    """Stage 19: adaptive placement attack with safer red returns and cleaner score shaping."""

    def __init__(self, render_mode: str | None = None, config: RobotArmCleanAdaptiveAttackConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmCleanAdaptiveAttackConfig())
        self.config: RobotArmCleanAdaptiveAttackConfig
        self.robot_arm_clean_adaptive_attack_enabled = True
        self.clean_combo_scores = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.clean_combo_scores = 0
        info = self._get_info(False, False, False, False, False)
        return obs, info

    def _bounce_from_paddle(self, player: str) -> None:
        super()._bounce_from_paddle(player)
        if player != "opponent" or not self.opponent_model_loaded:
            return

        center = self.config.table_left + self.config.red_safe_return_margin
        target = center + 0.05 * (self.agent_x - self.config.agent_x_min)
        landing_x = float(
            np.clip(
                target,
                self.config.table_left + self.config.red_safe_return_width,
                self.config.net_x - self.config.red_safe_return_margin,
            )
        )
        self.ball_vx = -min(abs(self.ball_vx), self.config.red_safe_return_speed)
        self.ball_spin = float(
            np.clip(
                self.ball_spin * self.config.red_safe_return_spin_scale,
                -self.config.max_spin * self.config.red_safe_return_spin_scale,
                self.config.max_spin * self.config.red_safe_return_spin_scale,
            )
        )
        self.ball_vy = self._safe_red_return_velocity(landing_x)

    def _safe_red_return_velocity(self, target_x: float) -> float:
        best_vy = self._aimed_vertical_velocity_with_spin(
            target_x,
            self.config.table_y - self.config.ball_radius,
            self.ball_spin,
        )
        best_error = float("inf")
        for candidate_vy in np.linspace(-7.8, 1.2, 55):
            bounce_x = self._first_table_bounce_x(float(candidate_vy))
            if bounce_x is None or bounce_x >= self.config.net_x:
                continue
            error = abs(bounce_x - target_x)
            if error < best_error:
                best_error = error
                best_vy = float(candidate_vy)
        return float(best_vy)

    def _first_table_bounce_x(self, initial_vy: float) -> float | None:
        x = float(self.ball_x)
        y = float(self.ball_y)
        vx = float(self.ball_vx)
        vy = float(initial_vy)
        spin = float(self.ball_spin)
        net_top = self.config.table_y - self.config.net_height
        for _ in range(260):
            prev_x = x
            prev_y = y
            vy += self.config.gravity + self.config.spin_lift * spin
            x += vx
            y += vy
            crosses_net = (prev_x - self.config.net_x) * (x - self.config.net_x) <= 0
            if crosses_net and abs(x - prev_x) > 1e-6:
                alpha = (self.config.net_x - prev_x) / (x - prev_x)
                net_y = prev_y + alpha * (y - prev_y)
                if net_y + self.config.ball_radius >= net_top:
                    return None
            over_table = self.config.table_left <= x <= self.config.table_right
            above_table = y + self.config.ball_radius >= self.config.table_y and vy > 0
            if over_table and above_table:
                return float(x)
            if x < self.config.table_left - 40 or y > self.config.height + 80:
                return None
        return None

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
        if agent_score and self.point_reason == "wrong_side_landing":
            reward -= self.config.wrong_side_score_penalty
        elif agent_score and self.combo_attack_landings > 0 and self.rally_length >= self.config.clean_attack_rally_target:
            reward += self.config.clean_combo_score_reward
        return float(reward)

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        if (
            agent_score
            and self.point_reason != "wrong_side_landing"
            and self.combo_attack_landings > 0
            and self.rally_length >= self.config.clean_attack_rally_target
        ):
            self.clean_combo_scores += 1
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update(
            {
                "stage": 19,
                "robot_arm_clean_adaptive_attack_enabled": True,
                "clean_combo_scores": self.clean_combo_scores,
            }
        )
        return info
