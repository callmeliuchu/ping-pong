from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.robot_arm_league_env import RobotArmLeagueConfig, RobotArmLeagueEnv


@dataclass(frozen=True)
class RobotArmTacticalLeagueConfig(RobotArmLeagueConfig):
    """Stage 16: reduce gift points and reward cleaner tactical wins."""

    short_rally_target: int = 10
    short_win_penalty: float = 18.0
    opponent_return_reward: float = 0.65
    sustained_win_bonus: float = 3.2
    clean_score_reward: float = 2.8
    wrong_side_score_penalty: float = 4.0
    opponent_model_residual_scale: float = 0.55


class RobotArmTacticalLeagueEnv(RobotArmLeagueEnv):
    """Stage 16: robot-arm league with spin-aware red returns and clean-point shaping."""

    def __init__(self, render_mode: str | None = None, config: RobotArmTacticalLeagueConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmTacticalLeagueConfig())
        self.config: RobotArmTacticalLeagueConfig
        self.clean_agent_scores = 0
        self.wrong_side_agent_scores = 0
        self.robot_arm_tactical_league_enabled = True

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.clean_agent_scores = 0
        self.wrong_side_agent_scores = 0
        info = self._get_info(False, False, False, False, False)
        return obs, info

    def _aimed_vertical_velocity_with_spin(self, target_x: float, target_y: float, spin: float) -> float:
        if abs(self.ball_vx) < 1e-6:
            return 0.0
        t = abs((target_x - self.ball_x) / self.ball_vx)
        if t <= 1e-6:
            return 0.0
        acceleration = self.config.gravity + self.config.spin_lift * spin
        vy = (target_y - self.ball_y - 0.5 * acceleration * t * t) / t
        return float(np.clip(vy, -self.config.max_ball_speed, self.config.max_ball_speed))

    def _bounce_from_paddle(self, player: str) -> None:
        super()._bounce_from_paddle(player)
        if player != "opponent" or not self.opponent_model_loaded:
            return

        landing_x = float(
            np.clip(
                self.config.table_left + 130.0 + 0.08 * (self.agent_x - self.config.agent_x_min),
                self.config.table_left + 70.0,
                self.config.net_x - 130.0,
            )
        )
        self.ball_vy = self._aimed_vertical_velocity_with_spin(
            landing_x,
            self.config.table_y - self.config.ball_radius,
            self.ball_spin,
        )
        if self.red_last_stroke_type in {"loop", "drive"}:
            self.ball_vy -= 0.12

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
        if agent_score:
            if self.point_reason == "wrong_side_landing":
                reward -= self.config.wrong_side_score_penalty
            elif self.rally_length >= self.config.short_rally_target:
                reward += self.config.clean_score_reward
        return float(reward)

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        if agent_score:
            if self.point_reason == "wrong_side_landing":
                self.wrong_side_agent_scores += 1
            else:
                self.clean_agent_scores += 1
        info.update(
            {
                "stage": 16,
                "robot_arm_tactical_league_enabled": True,
                "clean_agent_scores": self.clean_agent_scores,
                "wrong_side_agent_scores": self.wrong_side_agent_scores,
            }
        )
        return info
