from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.robot_arm_tactical_league_env import (
    RobotArmTacticalLeagueConfig,
    RobotArmTacticalLeagueEnv,
)


@dataclass(frozen=True)
class RobotArmAttackLeagueConfig(RobotArmTacticalLeagueConfig):
    """Stage 17: keep clean rallies, then reward pressure that can win points."""

    short_rally_target: int = 8
    short_win_penalty: float = 10.0
    opponent_return_reward: float = 0.52
    sustained_win_bonus: float = 3.6
    clean_score_reward: float = 4.2
    wrong_side_score_penalty: float = 7.5
    opponent_model_residual_scale: float = 0.48
    attack_landing_reward: float = 0.82
    high_pressure_reward: float = 0.95
    attack_score_reward: float = 7.5
    forced_error_reward: float = 4.0
    pressure_tracking_offset: float = 150.0
    pressure_slowdown: float = 0.62
    pressure_threshold: float = 0.55
    pressure_gain: float = 1.72
    clean_attack_rally_target: int = 1


class RobotArmAttackLeagueEnv(RobotArmTacticalLeagueEnv):
    """Stage 17: tactical robot-arm league with explicit attacking pressure."""

    def __init__(self, render_mode: str | None = None, config: RobotArmAttackLeagueConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmAttackLeagueConfig())
        self.config: RobotArmAttackLeagueConfig
        self.robot_arm_attack_league_enabled = True
        self.attack_landings = 0
        self.high_pressure_landings = 0
        self.clean_attack_scores = 0
        self.forced_error_scores = 0
        self.last_attack_pressure = 0.0
        self.max_attack_pressure = 0.0

    def _pressure_tracking_offset(self) -> float:
        return self.config.pressure_tracking_offset

    def _pressure_slowdown(self) -> float:
        return self.config.pressure_slowdown

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.attack_landings = 0
        self.high_pressure_landings = 0
        self.clean_attack_scores = 0
        self.forced_error_scores = 0
        self.last_attack_pressure = 0.0
        self.max_attack_pressure = 0.0
        info = self._get_info(False, False, False, False, False)
        return obs, info

    def _agent_attack_pressure(self) -> float:
        speed = np.clip((abs(self.ball_vx) - 6.7) / 4.0, 0.0, 1.0)
        topspin = np.clip(max(self.ball_spin, 0.0) / 6.0, 0.0, 1.0)
        deep = np.clip((self.ball_x - (self.config.net_x + 118.0)) / 190.0, 0.0, 1.0)
        wide = np.clip(abs(self.ball_x - self.opponent_x) / 150.0, 0.0, 1.0)
        loop_arc = np.clip(self.agent_shot_peak_arc / 42.0, 0.0, 1.0)
        pressure = 0.30 * speed + 0.28 * topspin + 0.23 * deep + 0.13 * wide + 0.06 * loop_arc
        return float(np.clip(pressure * self.config.pressure_gain, 0.0, 1.0))

    def _apply_model_opponent_action(self, action: np.ndarray) -> None:
        speeds = np.array(
            [self.config.arm_shoulder_speed, self.config.arm_elbow_speed, self.config.arm_wrist_speed],
            dtype=np.float32,
        )
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        mirrored = np.array([clipped[0], -clipped[1], -clipped[2]], dtype=np.float32)
        pressure = self.last_attack_pressure if self.ball_vx > 0 and self.last_hitter == "agent" else 0.0
        if self.ball_vx > 0:
            target_x = float(np.clip(self.ball_x + 34.0, self.config.opponent_x_min, self.config.opponent_x_max))
            target_y = self.predict_ball_y_at_x(target_x)
            vertical_misread = 1.0 if self.ball_y < self.opponent_y else -0.85
            target_y += pressure * self._pressure_tracking_offset() * vertical_misread
            desired_angle = 0.40 if self.ball_y < self.opponent_y else -0.08
            desired_angle += 0.16 * pressure
        else:
            target_x = self.config.table_right - 72.0
            target_y = self.config.table_y - 98.0
            desired_angle = 0.22
            self.last_attack_pressure = max(0.0, self.last_attack_pressure * 0.90)
        target_y = float(np.clip(target_y, self.config.table_y - 205.0, self.config.table_y - self.config.paddle_height / 2))
        target_angles = self._ik_for_paddle("opponent", target_x, target_y, desired_angle)
        pressure_slowdown = 1.0 - self._pressure_slowdown() * pressure
        base_limit = speeds * max(0.52, pressure_slowdown)
        base_delta = np.clip(target_angles - self.opponent_joint_angles, -base_limit, base_limit)
        residual_delta = mirrored * speeds * self.config.opponent_model_residual_scale
        self.opponent_joint_velocities = np.clip(base_delta + residual_delta, -speeds * 1.18, speeds * 1.18)
        self.opponent_joint_angles = self._clip_joint_angles(self.opponent_joint_angles + self.opponent_joint_velocities)
        self._sync_opponent_from_arm()
        self.opponent_tracking_error = float(abs(target_y - self.opponent_y))

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
            pressure = self._agent_attack_pressure()
            self.last_attack_pressure = pressure
            self.max_attack_pressure = max(self.max_attack_pressure, pressure)
            if self.last_stroke_type in {"loop", "drive"} or self.agent_last_topspin_like or self.agent_last_drive_like:
                self.attack_landings += 1
                reward += self.config.attack_landing_reward * (0.45 + pressure)
            if pressure >= self.config.pressure_threshold:
                self.high_pressure_landings += 1
                reward += self.config.high_pressure_reward * pressure
        if agent_score:
            clean_attack = (
                self.point_reason != "wrong_side_landing"
                and self.rally_length >= self.config.clean_attack_rally_target
                and self.max_attack_pressure >= self.config.pressure_threshold
            )
            forced_error = clean_attack and self.point_reason in {"receiver_missed", "second_bounce"}
            if clean_attack:
                reward += self.config.attack_score_reward * self.max_attack_pressure
            if forced_error:
                reward += self.config.forced_error_reward
        return float(reward)

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        if agent_score:
            clean_attack = (
                self.point_reason != "wrong_side_landing"
                and self.rally_length >= self.config.clean_attack_rally_target
                and self.max_attack_pressure >= self.config.pressure_threshold
            )
            self.clean_attack_scores += int(clean_attack)
            self.forced_error_scores += int(clean_attack and self.point_reason in {"receiver_missed", "second_bounce"})
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update(
            {
                "stage": 17,
                "robot_arm_attack_league_enabled": True,
                "attack_landings": self.attack_landings,
                "high_pressure_landings": self.high_pressure_landings,
                "clean_attack_scores": self.clean_attack_scores,
                "forced_error_scores": self.forced_error_scores,
                "last_attack_pressure": self.last_attack_pressure,
                "max_attack_pressure": self.max_attack_pressure,
            }
        )
        return info
