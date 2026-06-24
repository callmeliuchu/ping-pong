from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.robot_arm_bilateral_league_env import (
    RedRobotArmBilateralLeagueEnv,
    RobotArmBilateralLeagueConfig,
    RobotArmBilateralLeagueEnv,
)


def _apply_red_scoring_pressure(env: RobotArmBilateralLeagueEnv) -> None:
    red_is_model = getattr(env, "opponent_model_loaded", False) or getattr(env, "controlled_side", "blue") == "red"
    if not red_is_model or env.rally_length < 4:
        return

    ramp = float(np.clip((env.rally_length - 3) / 5.0, 0.0, 1.0))
    pressure = float(
        np.clip(
            0.34 * min(abs(env.opponent_vx) / max(env.config.paddle_x_speed, 1.0), 1.0)
            + 0.33 * min(max(-env.opponent_vy, 0.0) / max(env.config.paddle_y_speed, 1.0), 1.0)
            + 0.33 * min(max(env.ball_spin, 0.0) / max(env.config.max_spin, 1.0), 1.0),
            0.0,
            1.0,
        )
    )
    left_target = env.config.table_left + 78.0
    right_target = env.config.net_x - 82.0
    target_x = left_target if env.agent_x > env.config.table_left + 178.0 else right_target
    target_x = float(
        np.clip(
            (1.0 - env.config.red_aggressive_target_scale * ramp) * (env.config.table_left + 185.0)
            + env.config.red_aggressive_target_scale * ramp * target_x,
            env.config.table_left + 58.0,
            env.config.net_x - 58.0,
        )
    )
    env.ball_vy = env._aimed_vertical_velocity(target_x, env.config.table_y - env.config.ball_radius)
    env.ball_vy -= ramp * (0.10 + 0.24 * pressure)
    env.ball_vx = -min(abs(env.ball_vx) + ramp * (0.14 + 0.34 * pressure), env.config.max_ball_speed)
    env.ball_spin = float(np.clip(env.ball_spin + ramp * 0.55 * pressure, -env.config.max_spin, env.config.max_spin))


@dataclass(frozen=True)
class RobotArmRedScoringBilateralConfig(RobotArmBilateralLeagueConfig):
    """Stage 23: preserve Stage 22 while making red learn active scoring."""

    red_score_reward: float = 16.0
    red_miss_penalty: float = -8.0
    red_legal_landing_reward: float = 2.2
    red_hit_reward: float = 2.1
    red_rally_reward: float = 0.22
    red_forced_finish_reward: float = 10.0
    red_stalemate_penalty: float = 6.5
    red_deep_landing_reward: float = 1.25
    red_wide_landing_reward: float = 0.90
    red_pressure_finish_reward: float = 4.0
    red_short_score_penalty: float = 6.0
    red_attack_speed_bonus: float = 0.40
    red_aggressive_target_scale: float = 0.62
    red_pressure_finish_min_rally: int = 8
    red_pressure_finish_threshold: float = 0.45
    red_pressure_unreachable_threshold: float = 0.32


class RobotArmRedScoringBilateralEnv(RobotArmBilateralLeagueEnv):
    """Stage 23 blue-side environment with a stronger red scorer pool."""

    def __init__(self, render_mode: str | None = None, config: RobotArmRedScoringBilateralConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmRedScoringBilateralConfig())
        self.config: RobotArmRedScoringBilateralConfig
        self.robot_arm_red_scoring_bilateral_enabled = True

    def _bounce_from_paddle(self, player: str) -> None:
        super()._bounce_from_paddle(player)
        if player != "opponent":
            return
        _apply_red_scoring_pressure(self)

    def _apply_rules_after_bounce(self, table_side: str | None, net_hit: bool) -> bool:
        legal_landing = super()._apply_rules_after_bounce(table_side, net_hit)
        self._apply_red_pressure_finish(legal_landing)
        return legal_landing

    def _apply_red_pressure_finish(self, legal_landing: bool) -> None:
        if (
            not legal_landing
            or self.point_winner is not None
            or self.last_hitter != "opponent"
            or self.rally_length < self.config.red_pressure_finish_min_rally
        ):
            return

        left_depth = float(np.clip((self.config.net_x - self.ball_x) / (self.config.net_x - self.config.table_left), 0.0, 1.0))
        wide_pressure = float(np.clip(abs(self.ball_x - self.agent_x) / 155.0, 0.0, 1.0))
        speed_pressure = float(np.clip(abs(self.ball_vx) / max(self.config.max_ball_speed, 1.0), 0.0, 1.0))
        spin_pressure = float(np.clip(max(self.ball_spin, 0.0) / max(self.config.max_spin, 1.0), 0.0, 1.0))
        target_y = self.predict_ball_y_at_x(self.agent_x) if self.ball_vx < 0 else self.ball_y
        reach_pressure = float(np.clip(abs(target_y - self.agent_y) / max(self.config.paddle_height, 1.0), 0.0, 1.4))
        pressure = 0.36 * left_depth + 0.34 * wide_pressure + 0.20 * speed_pressure + 0.10 * spin_pressure

        if pressure + 0.35 * reach_pressure >= self.config.red_pressure_finish_threshold or (
            pressure >= self.config.red_pressure_unreachable_threshold and reach_pressure >= 0.20
        ):
            self._award_point("opponent", "receiver_missed")

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update({"stage": 23, "robot_arm_red_scoring_bilateral_enabled": True})
        return info


class RedRobotArmRedScoringBilateralEnv(RedRobotArmBilateralLeagueEnv):
    """Stage 23 role-swap environment where training controls the red scorer."""

    def __init__(self, render_mode: str | None = None, config: RobotArmRedScoringBilateralConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmRedScoringBilateralConfig())
        self.config: RobotArmRedScoringBilateralConfig
        self.robot_arm_red_scoring_bilateral_enabled = True

    def _bounce_from_paddle(self, player: str) -> None:
        super()._bounce_from_paddle(player)
        if player == "opponent":
            _apply_red_scoring_pressure(self)

    def _apply_rules_after_bounce(self, table_side: str | None, net_hit: bool) -> bool:
        legal_landing = super()._apply_rules_after_bounce(table_side, net_hit)
        RobotArmRedScoringBilateralEnv._apply_red_pressure_finish(self, legal_landing)
        return legal_landing

    def _red_reward(
        self,
        previous_distance: float,
        action: np.ndarray,
        red_hit: bool,
        legal_landing: bool,
        red_score: bool,
        red_miss: bool,
        rally_success: bool,
    ) -> float:
        reward = 0.0
        if self.ball_vx > 0:
            reward += self.config.shaping_scale * (previous_distance - self._red_distance_to_target())
        reward -= self.config.move_penalty * float(np.sum(np.square(action)))
        reward -= self.config.joint_center_penalty * float(np.mean(np.square(self.opponent_joint_angles)))

        if red_hit:
            reward += self.config.red_hit_reward
            reward += self.config.red_rally_reward * min(self.rally_length, 12)
            if self.red_last_stroke_type in {"loop", "drive", "chop"}:
                reward += self.config.technique_reward

        if legal_landing and self.last_hitter == "opponent":
            reward += self.config.red_legal_landing_reward
            left_depth = np.clip((self.config.net_x - self.ball_x) / (self.config.net_x - self.config.table_left), 0.0, 1.0)
            wide_pressure = np.clip(abs(self.ball_x - self.agent_x) / 170.0, 0.0, 1.0)
            reward += self.config.red_deep_landing_reward * float(left_depth)
            reward += self.config.red_wide_landing_reward * float(wide_pressure)
            if self.red_last_stroke_type == "loop":
                reward += self.config.topspin_landing_reward
            if abs(self.ball_vx) >= 7.3:
                reward += self.config.drive_landing_reward + self.config.red_attack_speed_bonus
        elif legal_landing and self.last_hitter == "agent":
            reward += 0.12

        if red_score:
            reward += self.config.red_score_reward
            if self._red_forced_finish(red_score):
                reward += self.config.red_forced_finish_reward
            if self.point_reason in {"receiver_missed", "second_bounce"}:
                reward += self.config.red_pressure_finish_reward
            shortfall = max(self.config.clean_attack_rally_target - self.rally_length, 0)
            if shortfall:
                reward -= self.config.red_short_score_penalty * shortfall / max(self.config.clean_attack_rally_target, 1)

        if red_miss:
            reward += self.config.red_miss_penalty
        if rally_success:
            reward -= self.config.red_stalemate_penalty
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
        info.update({"stage": 23, "robot_arm_red_scoring_bilateral_enabled": True})
        return info
