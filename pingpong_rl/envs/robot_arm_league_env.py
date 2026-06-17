from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pingpong_rl.envs.robot_arm_env import RobotArmPingPongConfig, RobotArmPingPongEnv


@dataclass(frozen=True)
class RobotArmLeagueConfig(RobotArmPingPongConfig):
    """Stage 15: robot-arm league opponents loaded from historical PPO models."""

    opponent_model_paths: tuple[str, ...] = field(default_factory=tuple)
    opponent_deterministic: bool = True
    short_rally_target: int = 8
    short_win_penalty: float = 12.0
    opponent_return_reward: float = 0.45
    sustained_win_bonus: float = 2.4
    deep_landing_reward: float = 0.22
    side_switch_reward: float = 0.12
    opponent_model_residual_scale: float = 0.70


class RobotArmLeagueEnv(RobotArmPingPongEnv):
    """Stage 15: blue robot arm trains against a pool of mirrored robot-arm PPO opponents."""

    def __init__(self, render_mode: str | None = None, config: RobotArmLeagueConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmLeagueConfig())
        self.config: RobotArmLeagueConfig
        self.history_opponent = None
        self.history_opponent_path = ""
        self.opponent_model_loaded = False
        self.model_opponent_steps = 0
        self.red_loop_attempts = 0
        self.red_drive_attempts = 0
        self.red_topspin_attempts = 0
        self.red_last_stroke_type = "none"
        self.robot_arm_league_enabled = True

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.history_opponent = None
        self.history_opponent_path = ""
        self.opponent_model_loaded = False
        self.model_opponent_steps = 0
        self.red_loop_attempts = 0
        self.red_drive_attempts = 0
        self.red_topspin_attempts = 0
        self.red_last_stroke_type = "none"
        self._load_history_opponent()
        info = self._get_info(False, False, False, False, False)
        return obs, info

    def _load_history_opponent(self) -> None:
        if self.history_opponent is not None or not self.config.opponent_model_paths:
            return

        candidates = [Path(path) for path in self.config.opponent_model_paths]
        existing = [path for path in candidates if path.exists() or path.with_suffix(".zip").exists()]
        if not existing:
            return
        index = int(self.np_random.integers(0, len(existing)))
        selected = existing[index]
        from stable_baselines3 import PPO

        self.history_opponent = PPO.load(selected)
        self.history_opponent_path = str(selected)
        self.opponent_model_loaded = True

    def _apply_opponent_policy(self) -> None:
        self._load_history_opponent()
        if self.history_opponent is None:
            return super()._apply_opponent_policy()

        obs = self._get_mirrored_obs_for_opponent()
        action, _ = self.history_opponent.predict(obs, deterministic=self.config.opponent_deterministic)
        self._apply_model_opponent_action(np.asarray(action, dtype=np.float32))
        self.model_opponent_steps += 1

    def _apply_model_opponent_action(self, action: np.ndarray) -> None:
        speeds = np.array(
            [self.config.arm_shoulder_speed, self.config.arm_elbow_speed, self.config.arm_wrist_speed],
            dtype=np.float32,
        )
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        mirrored = np.array([clipped[0], -clipped[1], -clipped[2]], dtype=np.float32)
        if self.ball_vx > 0:
            target_x = float(np.clip(self.ball_x + 34.0, self.config.opponent_x_min, self.config.opponent_x_max))
            target_y = self.predict_ball_y_at_x(target_x)
            desired_angle = 0.34 if self.ball_y < self.opponent_y else -0.06
        else:
            target_x = self.config.table_right - 72.0
            target_y = self.config.table_y - 98.0
            desired_angle = 0.22
        target_angles = self._ik_for_paddle("opponent", target_x, target_y, desired_angle)
        base_delta = np.clip(target_angles - self.opponent_joint_angles, -speeds, speeds)
        residual_delta = mirrored * speeds * self.config.opponent_model_residual_scale
        self.opponent_joint_velocities = np.clip(base_delta + residual_delta, -speeds * 1.35, speeds * 1.35)
        self.opponent_joint_angles = self._clip_joint_angles(self.opponent_joint_angles + self.opponent_joint_velocities)
        self._sync_opponent_from_arm()
        self.opponent_tracking_error = float(abs(target_y - self.opponent_y))

    def _get_mirrored_obs_for_opponent(self) -> np.ndarray:
        target_y = self.predict_ball_y_at_x(self.opponent_x) if self.ball_vx > 0 else self.config.table_y - 98.0
        target_x = self.opponent_x
        if self.ball_vx > 0:
            target_x = float(np.clip(self.ball_x + 36.0, self.config.opponent_x_min, self.config.opponent_x_max))

        angle_scales = np.array([self.config.shoulder_max, self.config.elbow_max, self.config.wrist_max], dtype=np.float32)
        speed_scales = np.array(
            [self.config.arm_shoulder_speed, self.config.arm_elbow_speed, self.config.arm_wrist_speed],
            dtype=np.float32,
        )
        mirrored_red_angles = np.array(
            [self.opponent_joint_angles[0], -self.opponent_joint_angles[1], -self.opponent_joint_angles[2]],
            dtype=np.float32,
        )
        mirrored_red_velocities = np.array(
            [self.opponent_joint_velocities[0], -self.opponent_joint_velocities[1], -self.opponent_joint_velocities[2]],
            dtype=np.float32,
        )
        obs = np.array(
            [
                (self.config.width - self.ball_x) / self.config.width * 2 - 1,
                self.ball_y / self.config.height * 2 - 1,
                -self.ball_vx / self.config.max_ball_speed,
                self.ball_vy / self.config.max_ball_speed,
                self.ball_spin / self.config.max_spin,
                *(mirrored_red_angles / angle_scales),
                *(mirrored_red_velocities / speed_scales),
                (self.config.width - self.opponent_x) / self.config.width * 2 - 1,
                self.opponent_y / self.config.height * 2 - 1,
                -self.opponent_vx / max(self.config.paddle_x_speed, 1.0),
                self.opponent_vy / max(self.config.paddle_y_speed, 1.0),
                -self.opponent_angle / self.config.max_paddle_angle,
                (self.config.width - self.agent_x) / self.config.width * 2 - 1,
                self.agent_y / self.config.height * 2 - 1,
                -self.agent_angle / self.config.max_paddle_angle,
                (self.config.width - target_x) / self.config.width * 2 - 1,
                target_y / self.config.height * 2 - 1,
                (self.opponent_x - target_x) / self.config.width,
                (target_y - self.opponent_y) / self.config.height,
                np.clip(self.rally_length / max(self.config.target_rally_length, 1), -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _bounce_from_paddle(self, player: str) -> None:
        super()._bounce_from_paddle(player)
        if player != "opponent" or not self.opponent_model_loaded:
            return

        wrist_brush = float(-self.opponent_joint_velocities[2] / max(self.config.arm_wrist_speed, 1e-6))
        upward_brush = float(max(-self.opponent_vy / max(self.config.paddle_y_speed, 1.0), 0.0))
        downward_brush = float(max(self.opponent_vy / max(self.config.paddle_y_speed, 1.0), 0.0))
        closed_face = float(max(self.opponent_angle / max(self.config.max_paddle_angle, 1e-6), 0.0))
        open_face = float(max(-self.opponent_angle / max(self.config.max_paddle_angle, 1e-6), 0.0))
        forward_swing = float(max(-self.opponent_vx / max(self.config.paddle_x_speed, 1.0), 0.0))
        center_contact = max(self.last_contact_quality, 0.0)

        topspin_intent = max(0.0, 0.45 * max(wrist_brush, 0.0) + 0.33 * upward_brush + 0.22 * closed_face)
        chop_intent = max(0.0, 0.48 * max(-wrist_brush, 0.0) + 0.34 * downward_brush + 0.18 * open_face)
        drive_intent = max(0.0, 0.62 * forward_swing + 0.38 * center_contact)
        if topspin_intent >= max(chop_intent, 0.18):
            spin_delta = 2.0 + 4.3 * topspin_intent
            self.ball_vx = -min(abs(self.ball_vx) + 0.24 + 0.38 * forward_swing, self.config.max_ball_speed)
            self.ball_vy -= 0.14 + 0.36 * upward_brush
        elif drive_intent >= 0.58 and abs(self.ball_spin) < 4.8:
            spin_delta = 0.7 + 1.2 * closed_face - 0.5 * open_face
            self.ball_vx = -min(abs(self.ball_vx) + 0.62 + 0.48 * forward_swing, self.config.max_ball_speed)
            self.ball_vy += 0.16 * downward_brush
        else:
            spin_delta = -1.1 - 3.0 * chop_intent
            self.ball_vx = -max(abs(self.ball_vx) - 0.15 * chop_intent, self.config.ball_speed_x_min)
            self.ball_vy -= 0.07 * open_face

        self.ball_spin = float(np.clip(0.40 * self.ball_spin + spin_delta, -self.config.max_spin, self.config.max_spin))
        red_drive_like = abs(self.ball_vx) >= 7.3
        red_topspin_like = self.ball_spin >= 2.0
        if red_topspin_like and self.ball_spin >= 3.0:
            self.red_last_stroke_type = "loop"
        elif red_drive_like:
            self.red_last_stroke_type = "drive"
        elif self.ball_spin <= -2.0:
            self.red_last_stroke_type = "chop"
        else:
            self.red_last_stroke_type = "block"
        self.red_loop_attempts += int(self.red_last_stroke_type == "loop")
        self.red_drive_attempts += int(red_drive_like)
        self.red_topspin_attempts += int(red_topspin_like)
        landing_x = float(
            np.clip(
                self.config.table_left + 185.0 + 0.12 * (self.agent_x - self.config.agent_x_min),
                self.config.table_left + 85.0,
                self.config.net_x - 78.0,
            )
        )
        self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius)
        if self.red_last_stroke_type == "loop":
            self.ball_vy -= 0.36
        elif self.red_last_stroke_type == "drive":
            self.ball_vy -= 0.18

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
        if legal_landing and self.last_hitter == "opponent":
            reward += self.config.opponent_return_reward
        if legal_landing and self.last_hitter == "agent":
            deep = max(0.0, (self.ball_x - (self.config.net_x + 130.0)) / 190.0)
            reward += self.config.deep_landing_reward * min(deep, 1.0)
            if abs(self.ball_x - self.opponent_x) > 80.0:
                reward += self.config.side_switch_reward
        if agent_score:
            shortfall = max(self.config.short_rally_target - self.rally_length, 0)
            if shortfall > 0:
                reward -= self.config.short_win_penalty * shortfall / max(self.config.short_rally_target, 1)
                if self.point_reason == "wrong_side_landing":
                    reward -= 1.2
            else:
                reward += self.config.sustained_win_bonus
        if rally_success:
            reward += self.config.sustained_win_bonus
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
        info.update(
            {
                "stage": 15,
                "robot_arm_league_enabled": True,
                "opponent_model_loaded": self.opponent_model_loaded,
                "opponent_model_path": self.history_opponent_path,
                "model_opponent_steps": self.model_opponent_steps,
                "red_last_stroke_type": self.red_last_stroke_type,
                "red_loop_attempts": self.red_loop_attempts,
                "red_drive_attempts": self.red_drive_attempts,
                "red_topspin_attempts": self.red_topspin_attempts,
            }
        )
        return info
